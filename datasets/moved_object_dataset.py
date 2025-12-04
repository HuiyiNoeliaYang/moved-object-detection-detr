import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

try:
    # Allows optional reuse of the metadata helpers created earlier.
    from cv_data_hw2.metadata_utils import load_and_split as load_and_split_metadata
except ModuleNotFoundError:  # pragma: no cover - fallback if package import path isn't set up
    load_and_split_metadata = None  # type: ignore


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DatasetRecord:
    pair_id: str
    img1_path: str
    img2_path: str
    matched_ann_path: str
    has_moved_objects: bool

    @classmethod
    def from_row(cls, row: Dict[str, str]) -> "DatasetRecord":
        return cls(
            pair_id=row["pair_id"],
            img1_path=row["img1_path"],
            img2_path=row["img2_path"],
            matched_ann_path=row.get("matched_ann_path", ""),
            has_moved_objects=row.get("has_moved_objects", "0") in {"1", "True", "true"},
        )


def _resize_with_long_side(img: Image.Image, target_long_side: int) -> Image.Image:
    if target_long_side <= 0:
        return img
    w, h = img.size
    long_side = max(w, h)
    if long_side == 0 or long_side == target_long_side:
        return img
    scale = target_long_side / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.BILINEAR)


def _tensorize(img: Image.Image) -> torch.Tensor:
    tensor = TF.to_tensor(img)
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


class MovedObjectDataset(Dataset):
    """Dataset that yields DETR-ready samples built from VIRAT frame pairs and matched annotations."""

    def __init__(
        self,
        records: Sequence[Dict[str, str]],
        target_long_side: int = 800,
        only_with_moved_objects: bool = False,
    ) -> None:
        dataset_records = [DatasetRecord.from_row(r) for r in records]
        if only_with_moved_objects:
            dataset_records = [r for r in dataset_records if r.has_moved_objects and os.path.exists(r.matched_ann_path)]
        self.records = dataset_records
        self.target_long_side = target_long_side

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        img1 = Image.open(record.img1_path).convert("RGB")
        img2 = Image.open(record.img2_path).convert("RGB")

        orig_w, orig_h = img1.size
        if img2.size != (orig_w, orig_h):
            # Ensure scaling factors remain valid even if source frames somehow differ slightly.
            img2 = img2.resize((orig_w, orig_h), Image.BILINEAR)

        img1 = _resize_with_long_side(img1, self.target_long_side)
        img2 = _resize_with_long_side(img2, self.target_long_side)
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.BILINEAR)

        img1_tensor = _tensorize(img1)
        img2_tensor = _tensorize(img2)
        diff_tensor = (img2_tensor - img1_tensor).abs()

        new_w, new_h = img1.size
        scale_x = new_w / float(orig_w)
        scale_y = new_h / float(orig_h)
        target = self._build_target(record, scale_x, scale_y, new_w, new_h)

        return {
            "pixel_values": diff_tensor,
            "labels": target,
        }

    def _build_target(
        self,
        record: DatasetRecord,
        scale_x: float,
        scale_y: float,
        width: int,
        height: int,
    ) -> Dict[str, torch.Tensor]:
        boxes: List[List[float]] = []
        labels: List[int] = []

        ann_path = record.matched_ann_path
        if ann_path and os.path.exists(ann_path):
            with open(ann_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 6:
                        continue
                    _, x, y, w, h, cls = parts[:6]
                    x = float(x) * scale_x
                    y = float(y) * scale_y
                    w = float(w) * scale_x
                    h = float(h) * scale_y
                    x_min = max(0.0, min(width, x))
                    y_min = max(0.0, min(height, y))
                    x_max = max(0.0, min(width, x + w))
                    y_max = max(0.0, min(height, y + h))
                    if x_max <= x_min or y_max <= y_min:
                        continue
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(int(cls))

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        class_labels = labels_tensor
        target = {
            "boxes": boxes_tensor,
            "class_labels": class_labels,
            "labels": class_labels.clone(),  # optional alias for downstream use
            "image_id": torch.tensor([hash(record.pair_id) & 0xFFFFFFFF], dtype=torch.int64),
            "area": (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
            if boxes
            else torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.zeros((boxes_tensor.shape[0],), dtype=torch.int64),
            "orig_size": torch.tensor([height, width], dtype=torch.int64),
            "size": torch.tensor([height, width], dtype=torch.int64),
        }
        return target


def build_datasets_from_metadata(
    csv_path: Optional[str] = None,
    train_ratio: float = 0.8,
    seed: Optional[int] = 42,
    target_long_side: int = 800,
    only_with_moved_objects: bool = False,
) -> Tuple[MovedObjectDataset, MovedObjectDataset]:
    if load_and_split_metadata is None:
        raise ImportError("cv_data_hw2.metadata_utils could not be imported; ensure the package path is available.")
    train_records, test_records = load_and_split_metadata(csv_path=csv_path, train_ratio=train_ratio, seed=seed)
    train_dataset = MovedObjectDataset(
        train_records,
        target_long_side=target_long_side,
        only_with_moved_objects=only_with_moved_objects,
    )
    test_dataset = MovedObjectDataset(
        test_records,
        target_long_side=target_long_side,
        only_with_moved_objects=only_with_moved_objects,
    )
    return train_dataset, test_dataset

