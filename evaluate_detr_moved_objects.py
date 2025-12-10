import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.ops import box_iou

from configs.detr_config import CLASS_ID_TO_NAME, CLASS_NAME_TO_ID, NUM_CLASSES
from datasets.moved_object_dataset import MovedObjectDataset, build_datasets_from_metadata
from train_detr_moved_objects import collate_fn
from transformers import DetrForObjectDetection


def _cxcywh_to_xyxy_abs(boxes_norm: torch.Tensor, width: float, height: float) -> torch.Tensor:
    """
    Convert normalized cx,cy,w,h (all in [0,1]) to absolute xyxy in pixels.
    """
    if boxes_norm.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    cx = boxes_norm[:, 0] * width
    cy = boxes_norm[:, 1] * height
    w = boxes_norm[:, 2] * width
    h = boxes_norm[:, 3] * height
    x_min = cx - w / 2.0
    y_min = cy - h / 2.0
    x_max = cx + w / 2.0
    y_max = cy + h / 2.0
    return torch.stack([x_min, y_min, x_max, y_max], dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DETR fine-tuned models on VIRAT motion pairs.")
    parser.add_argument("--metadata-csv", type=str, default=os.path.join("cv_data_hw2", "metadata", "pairs.csv"))
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model state_dict (.pt).")
    parser.add_argument("--regime", type=str, default="full", help="Name for logging/visualization directories.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-plots", type=int, default=10, help="Number of qualitative figures to generate.")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--visuals-dir", type=str, default="visuals")
    parser.add_argument(
        "--print-examples",
        type=int,
        default=0,
        help="If >0, print predictions and ground truth for this many samples.",
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> DetrForObjectDetection:
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=NUM_CLASSES,
        id2label=CLASS_ID_TO_NAME,
        label2id=CLASS_NAME_TO_ID,
        ignore_mismatched_sizes=True,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, dataloader, device, iou_threshold: float, print_examples: int = 0) -> Dict[str, float]:
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    examples_printed = 0
    global_sample_idx = 0
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        outputs = model(pixel_values=pixel_values)
        probas = outputs.logits.softmax(-1)[..., :-1].cpu()
        keep = probas.max(-1).values > 0.5
        pred_boxes = outputs.pred_boxes.cpu()
        labels = batch["labels"]

        batch_idx = 0
        for k in range(pixel_values.size(0)):
            pred_scores = probas[k][keep[k]]
            pred_classes = pred_scores.argmax(-1).cpu()
            pred_conf = pred_scores.max(-1).values.cpu()
            pred_boxes_norm = pred_boxes[k][keep[k]]
            target = labels[k]
            gt_boxes_norm = target["boxes"]
            gt_classes = target["class_labels"]
            height, width = target["size"].tolist()

            pred_boxes_abs = torch.zeros_like(pred_boxes_norm)
            pred_boxes_abs[:, 0] = (pred_boxes_norm[:, 0] - pred_boxes_norm[:, 2] / 2.0) * width
            pred_boxes_abs[:, 1] = (pred_boxes_norm[:, 1] - pred_boxes_norm[:, 3] / 2.0) * height
            pred_boxes_abs[:, 2] = (pred_boxes_norm[:, 0] + pred_boxes_norm[:, 2] / 2.0) * width
            pred_boxes_abs[:, 3] = (pred_boxes_norm[:, 1] + pred_boxes_norm[:, 3] / 2.0) * height

            gt_boxes_abs = _cxcywh_to_xyxy_abs(gt_boxes_norm, width, height)

            if examples_printed < print_examples:
                print(f"[example {examples_printed + 1}] sample_idx={global_sample_idx}")
                print("  preds (class, conf, [xmin, ymin, xmax, ymax]):")
                if len(pred_boxes_abs) == 0:
                    print("    (none)")
                else:
                    for box, cls, conf in zip(pred_boxes_abs, pred_classes, pred_conf):
                        cls_name = CLASS_ID_TO_NAME.get(int(cls), str(int(cls)))
                        box_list = [round(float(x), 1) for x in box.tolist()]
                        print(f"    {cls_name}, {conf:.3f}, {box_list}")
                print("  gts (class, [xmin, ymin, xmax, ymax]):")
                if len(gt_boxes_abs) == 0:
                    print("    (none)")
                else:
                    for box, cls in zip(gt_boxes_abs, gt_classes):
                        cls_name = CLASS_ID_TO_NAME.get(int(cls), str(int(cls)))
                        box_list = [round(float(x), 1) for x in box.tolist()]
                        print(f"    {cls_name}, {box_list}")
                examples_printed += 1

            if len(gt_boxes_abs) == 0 and len(pred_boxes_abs) == 0:
                continue
            if len(gt_boxes_abs) == 0:
                for cls in pred_classes.tolist():
                    stats[cls]["fp"] += 1
                continue
            if len(pred_boxes_abs) == 0:
                for cls in gt_classes.tolist():
                    stats[cls]["fn"] += 1
                continue

            ious = box_iou(pred_boxes_abs, gt_boxes_abs)
            gt_matched = set()
            pred_matched = set()
            for pred_idx in range(len(pred_boxes_abs)):
                max_iou, gt_idx = torch.max(ious[pred_idx], dim=0)
                cls_pred = int(pred_classes[pred_idx])
                cls_gt = int(gt_classes[gt_idx])
                if max_iou >= iou_threshold and cls_pred == cls_gt and gt_idx.item() not in gt_matched:
                    stats[cls_pred]["tp"] += 1
                    gt_matched.add(gt_idx.item())
                    pred_matched.add(pred_idx)
                else:
                    stats[cls_pred]["fp"] += 1
            for gt_idx in range(len(gt_boxes_abs)):
                if gt_idx not in gt_matched:
                    cls_gt = int(gt_classes[gt_idx])
                    stats[cls_gt]["fn"] += 1
        batch_idx += 1
        global_sample_idx += pixel_values.size(0)

    metrics = {}
    for cls_id, counts in stats.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[CLASS_ID_TO_NAME.get(cls_id, str(cls_id))] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return metrics


def plot_qualitative(model, dataset: MovedObjectDataset, device, save_dir: str, max_plots: int) -> None:
    os.makedirs(save_dir, exist_ok=True)
    indices = list(range(min(max_plots, len(dataset))))
    for idx in indices:
        sample = dataset[idx]
        record = dataset.records[idx]
        diff_tensor = sample["pixel_values"].unsqueeze(0).to(device)
        outputs = model(pixel_values=diff_tensor)
        probas = outputs.logits.softmax(-1)[..., :-1][0]
        keep = probas.max(-1).values > 0.5
        boxes_norm = outputs.pred_boxes[0][keep].cpu()
        classes = probas.argmax(-1)[keep].cpu()

        img1 = Image.open(record.img1_path).convert("RGB")
        img2 = Image.open(record.img2_path).convert("RGB")
        diff_img = torch.clamp(sample["pixel_values"], 0, 1).permute(1, 2, 0).numpy()
        height, width = sample["labels"]["size"].tolist()
        boxes = _cxcywh_to_xyxy_abs(boxes_norm, width, height)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(img1)
        axes[0].set_title("Frame 1")
        axes[1].imshow(img2)
        axes[1].set_title("Frame 2")
        axes[2].imshow(diff_img)
        axes[2].set_title("Diff (input)")

        for ax in axes:
            ax.axis("off")

        # Draw predictions and GT on diff plot
        ax_diff = axes[2]
        for box, cls in zip(boxes, classes):
            xmin, ymin, xmax, ymax = box.tolist()
            rect = plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                edgecolor="lime",
                linewidth=2,
            )
            ax_diff.add_patch(rect)
            ax_diff.text(
                xmin,
                ymin,
                CLASS_ID_TO_NAME.get(int(cls), str(int(cls))),
                color="white",
                bbox={"facecolor": "lime", "alpha": 0.5},
            )

        target = sample["labels"]
        gt_boxes_abs = _cxcywh_to_xyxy_abs(target["boxes"], width, height)
        for box, cls in zip(gt_boxes_abs, target["class_labels"]):
            xmin, ymin, xmax, ymax = box.tolist()
            rect = plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                edgecolor="red",
                linewidth=2,
            )
            ax_diff.add_patch(rect)
            ax_diff.text(
                xmin,
                ymin,
                CLASS_ID_TO_NAME.get(int(cls), str(int(cls))),
                color="white",
                bbox={"facecolor": "red", "alpha": 0.5},
            )

        save_path = os.path.join(save_dir, f"sample_{idx:04d}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    _, test_dataset = build_datasets_from_metadata(
        csv_path=args.metadata_csv,
        train_ratio=0.8,
        seed=42,
        target_long_side=800,
        only_with_moved_objects=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = load_model(args.checkpoint, device)
    metrics = evaluate(model, test_loader, device, args.iou_threshold, args.print_examples)

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, f"{args.regime}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    visuals_path = os.path.join(args.visuals_dir, args.regime)
    plot_qualitative(model, test_dataset, device, visuals_path, args.max_plots)
    print(f"Saved qualitative figures to {visuals_path}")


if __name__ == "__main__":
    main()

