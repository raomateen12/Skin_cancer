"""
Quick sanity check for the HAM10000 DataLoaders.

Loads train/val/test CSVs from data/processed and verifies that
images can be loaded, transforms applied, and batches collated.
Does NOT train anything.
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Adjust import path when running as a module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import (
    HAM10000Dataset,
    get_train_transforms,
    get_eval_transforms,
    class_to_idx,
    idx_to_class,
    CLASS_NAMES,
)

PROCESSED_DIR = Path("data/processed")
BATCH_SIZE = 4
IMAGE_SIZE = 224


def collate_fn(batch):
    """Custom collate to handle metadata dicts in the batch."""
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    metadata = [item[2] for item in batch]
    return images, labels, metadata


def make_loader(csv_path, transform, shuffle=False):
    dataset = HAM10000Dataset(
        csv_file=str(csv_path),
        image_size=IMAGE_SIZE,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,  # 0 is safest on Windows for quick checks
        collate_fn=collate_fn,
    )
    return dataset, loader


def check_split(name, csv_path, transform, shuffle=False):
    if not csv_path.exists():
        print(f"  [{name}] CSV not found: {csv_path}")
        print("   Run: python -m src.prepare_data")
        return

    dataset, loader = make_loader(csv_path, transform, shuffle)
    print(f"\n[{name}]  {len(dataset)} samples")

    images, labels, metadata = next(iter(loader))
    print(f"  image tensor shape : {images.shape}")
    print(f"  labels             : {labels.tolist()}")
    print(f"  image_ids          : {[m['image_id'] for m in metadata]}")
    print(f"  dx codes           : {[m['dx'] for m in metadata]}")


def main():
    print("=== DataLoader Sanity Check ===")
    print(f"\nClass mapping ({len(class_to_idx)} classes):")
    for cls, idx in class_to_idx.items():
        print(f"  [{idx}] {cls:6s} — {CLASS_NAMES[cls]}")

    train_csv = PROCESSED_DIR / "train.csv"
    val_csv   = PROCESSED_DIR / "val.csv"
    test_csv  = PROCESSED_DIR / "test.csv"

    check_split("TRAIN", train_csv, get_train_transforms(IMAGE_SIZE), shuffle=True)
    check_split("VAL",   val_csv,   get_eval_transforms(IMAGE_SIZE))
    check_split("TEST",  test_csv,  get_eval_transforms(IMAGE_SIZE))

    print("\nSanity check complete. Dataloaders are working.")


if __name__ == "__main__":
    main()
