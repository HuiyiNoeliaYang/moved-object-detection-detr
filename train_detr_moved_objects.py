import argparse
import json
import os
import time
from typing import Dict, List, Sequence
import torch
from torch.utils.data import DataLoader, Subset
from transformers import DetrForObjectDetection

from configs.detr_config import CLASS_ID_TO_NAME, CLASS_NAME_TO_ID, NUM_CLASSES
from datasets.moved_object_dataset import build_datasets_from_metadata

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DETR on VIRAT motion pairs (Option 2).")
    default_metadata = os.path.join(SCRIPT_DIR, "cv_data_hw2", "metadata", "pairs.csv")
    parser.add_argument("--metadata-csv", type=str, default=default_metadata, help="Path to metadata CSV.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to store checkpoints and logs.")
    parser.add_argument("--run-name", type=str, default="", help="Optional run name. Defaults to regime+timestamp.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training and validation.")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument("--train-only-moved", action="store_true", help="Filter dataset to samples with moved objects.")
    parser.add_argument(
        "--target-long-side",
        type=int,
        default=800,
        help="Resize frames so longest side matches this value before diff computation.",
    )
    parser.add_argument(
        "--regime",
        type=str,
        default="full",
        choices=["full", "backbone", "head", "transformer"],
        help="Fine-tuning strategy.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/test split ratio from metadata.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset split and torch.")
    parser.add_argument(
        "--train-limit",
        type=int,
        default=0,
        help="If >0, use only this many samples from the training set (for overfit tests).",
    )
    parser.add_argument(
        "--val-limit",
        type=int,
        default=0,
        help="If >0, use only this many samples from the validation set.",
    )
    parser.add_argument(
        "--print-example",
        action="store_true",
        help="After each epoch, print one prediction vs ground truth from the val set.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def collate_fn(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, List[Dict[str, torch.Tensor]]]:
    pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)
    labels = [item["labels"] for item in batch]
    return {"pixel_values": pixel_values, "labels": labels}


def move_targets_to_device(labels: List[Dict[str, torch.Tensor]], device: torch.device) -> List[Dict[str, torch.Tensor]]:
    moved = []
    for target in labels:
        moved.append({k: v.to(device) for k, v in target.items()})
    return moved


@torch.no_grad()
def print_one_example(model, dataloader, device, score_thresh: float = 0.1) -> None:
    model.eval()
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"]
        outputs = model(pixel_values=pixel_values)
        probas = outputs.logits.softmax(-1)[..., :-1].cpu()
        keep = probas.max(-1).values > score_thresh
        pred_boxes = outputs.pred_boxes.cpu()

        k = 0
        pred_scores = probas[k][keep[k]]
        pred_classes = pred_scores.argmax(-1).cpu()
        pred_conf = pred_scores.max(-1).values.cpu()
        pred_boxes_norm = pred_boxes[k][keep[k]]
        target = labels[k]
        gt_boxes = target["boxes"]
        gt_classes = target["class_labels"]
        height, width = target["size"].tolist()

        pred_boxes_abs = torch.zeros_like(pred_boxes_norm)
        pred_boxes_abs[:, 0] = (pred_boxes_norm[:, 0] - pred_boxes_norm[:, 2] / 2.0) * width
        pred_boxes_abs[:, 1] = (pred_boxes_norm[:, 1] - pred_boxes_norm[:, 3] / 2.0) * height
        pred_boxes_abs[:, 2] = (pred_boxes_norm[:, 0] + pred_boxes_norm[:, 2] / 2.0) * width
        pred_boxes_abs[:, 3] = (pred_boxes_norm[:, 1] + pred_boxes_norm[:, 3] / 2.0) * height

        print("[example] preds (class, conf, [xmin, ymin, xmax, ymax]):")
        if len(pred_boxes_abs) == 0:
            print("  (none)")
        else:
            for box, cls, conf in zip(pred_boxes_abs, pred_classes, pred_conf):
                cls_name = CLASS_ID_TO_NAME.get(int(cls), str(int(cls)))
                box_list = [round(float(x), 1) for x in box.tolist()]
                print(f"  {cls_name}, {conf:.3f}, {box_list}")

        print("  gts (class, [xmin, ymin, xmax, ymax]):")
        if len(gt_boxes) == 0:
            print("  (none)")
        else:
            for box, cls in zip(gt_boxes, gt_classes):
                cls_name = CLASS_ID_TO_NAME.get(int(cls), str(int(cls)))
                box_list = [round(float(x), 1) for x in box.tolist()]
                print(f"  {cls_name}, {box_list}")
        break


def determine_trainable_params(model: DetrForObjectDetection, regime: str) -> List[str]:
    for param in model.parameters():
        param.requires_grad = True

    if regime == "full":
        return [name for name, p in model.named_parameters() if p.requires_grad]

    for param in model.parameters():
        param.requires_grad = False

    def mark_trainable(prefixes: Sequence[str]) -> None:
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in prefixes):
                param.requires_grad = True

    if regime == "backbone":
        mark_trainable(["model.backbone", "model.input_proj"])
    elif regime == "head":
        mark_trainable(["class_labels_classifier", "bbox_predictor"])
    elif regime == "transformer":
        mark_trainable(
            [
                "model.encoder",
                "model.decoder",
                "model.query_position_embeddings",
                "model.row_embed",
                "model.col_embed",
            ]
        )
    else:
        raise ValueError(f"Unknown regime: {regime}")

    return [name for name, p in model.named_parameters() if p.requires_grad]


def train_one_epoch(model, dataloader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        pixel_values = batch["pixel_values"].to(device)
        labels = move_targets_to_device(batch["labels"], device)
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(dataloader), 1)


def evaluate(model, dataloader, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = move_targets_to_device(batch["labels"], device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            total_loss += outputs.loss.item()
    return total_loss / max(len(dataloader), 1)


def save_checkpoint(model, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    train_ds, val_ds = build_datasets_from_metadata(
        csv_path=args.metadata_csv,
        train_ratio=args.train_ratio,
        seed=args.seed,
        target_long_side=args.target_long_side,
        only_with_moved_objects=args.train_only_moved,
    )

    if args.train_limit and args.train_limit > 0:
        train_ds = Subset(train_ds, range(min(args.train_limit, len(train_ds))))
    if args.val_limit and args.val_limit > 0:
        val_ds = Subset(val_ds, range(min(args.val_limit, len(val_ds))))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=NUM_CLASSES,
        id2label=CLASS_ID_TO_NAME,
        label2id=CLASS_NAME_TO_ID,
        ignore_mismatched_sizes=True,
    )
    trainable_names = determine_trainable_params(model, args.regime)
    model.to(device)

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.regime}_{timestamp}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[{run_name}] Epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if args.print_example:
            print_one_example(model, val_loader, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, os.path.join(run_dir, "best_model.pt"))

        save_checkpoint(model, os.path.join(run_dir, "latest_model.pt"))

        with open(os.path.join(run_dir, "training_log.json"), "w") as f:
            json.dump(
                {
                    "run_name": run_name,
                    "regime": args.regime,
                    "trainable_parameters": trainable_names,
                    "hyperparameters": {
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "learning_rate": args.learning_rate,
                        "weight_decay": args.weight_decay,
                        "target_long_side": args.target_long_side,
                        "train_only_moved": args.train_only_moved,
                    },
                    "dataset_sizes": {"train": len(train_ds), "val": len(val_ds)},
                    "history": history,
                    "best_val_loss": best_val_loss,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()

