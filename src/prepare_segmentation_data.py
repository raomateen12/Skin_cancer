"""
src/prepare_segmentation_data.py
Dataset preparation script for ISIC 2018 Task 1 (Lesion Boundary Segmentation).

Matches ISIC binary mask files (e.g. ISIC_0024306_segmentation.png or ISIC_0024306.png)
to their source dermoscopy images (e.g. ISIC_0024306.jpg) by ISIC_XXXXXXX identifier.
Splits matched pairs into 80% train and 20% validation sets with fixed random seed.

Outputs:
  - data/processed/segmentation_train.csv
  - data/processed/segmentation_val.csv

Columns:
  image_id, image_path, mask_path
"""

from __future__ import annotations
import argparse
import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split


ISIC_PATTERN = re.compile(r"(ISIC_\d{7})", re.IGNORECASE)


def extract_image_id(filename: str) -> Optional[str]:
    """Extract canonical uppercase ISIC_XXXXXXX from filename."""
    match = ISIC_PATTERN.search(filename)
    if match:
        return match.group(1).upper()
    # Fallback to stem if formatted like ISIC_0024306
    stem = Path(filename).stem
    if stem.upper().startswith("ISIC_"):
        return stem.split("_")[0] + "_" + stem.split("_")[1]
    return None


def index_directory_images(directory: Path) -> dict[str, Path]:
    """Scan directory recursively and map image_id -> Path."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_map: dict[str, Path] = {}

    if not directory.exists():
        return image_map

    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in valid_exts:
            img_id = extract_image_id(path.name)
            if img_id and img_id not in image_map:
                image_map[img_id] = path

    return image_map


def prepare_segmentation_splits(
    images_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    aux_images_dir: Optional[Path] = None,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Match masks to images, create 80/20 train/val split, and save CSV files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Indexing masks from: {masks_dir}")
    mask_map = index_directory_images(masks_dir)
    print(f"  Found {len(mask_map):,} mask files.")

    print(f"Indexing source images from: {images_dir}")
    image_map = index_directory_images(images_dir)
    if aux_images_dir and aux_images_dir.exists():
        print(f"Indexing auxiliary source images from: {aux_images_dir}")
        aux_map = index_directory_images(aux_images_dir)
        for k, v in aux_map.items():
            if k not in image_map:
                image_map[k] = v
    print(f"  Found {len(image_map):,} source images in total.")

    # Match image_id across masks and images
    matched_rows: list[dict[str, str]] = []
    unmatched_masks: list[str] = []

    for img_id, mask_path in mask_map.items():
        if img_id in image_map:
            matched_rows.append({
                "image_id": img_id,
                "image_path": str(image_map[img_id]),
                "mask_path": str(mask_path),
            })
        else:
            unmatched_masks.append(img_id)

    if not matched_rows:
        raise ValueError(
            f"No matching image-mask pairs found between:\n"
            f"  Images: {images_dir}\n"
            f"  Masks:  {masks_dir}\n"
            f"Please verify directory paths and ISIC_XXXXXXX naming."
        )

    df_matched = pd.DataFrame(matched_rows).sort_values("image_id").reset_index(drop=True)
    print(f"\nSuccessfully matched: {len(df_matched):,} image-mask pairs.")
    if unmatched_masks:
        print(f"Warning: {len(unmatched_masks):,} masks had no corresponding source image.")

    # 80 / 20 Train / Validation Split
    train_df, val_df = train_test_split(
        df_matched,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
    )

    train_df = train_df.sort_values("image_id").reset_index(drop=True)
    val_df = val_df.sort_values("image_id").reset_index(drop=True)

    train_csv = output_dir / "segmentation_train.csv"
    val_csv = output_dir / "segmentation_val.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    print(f"\nSaved segmentation splits:")
    print(f"  Train: {len(train_df):,} rows -> {train_csv}")
    print(f"  Val:   {len(val_df):,} rows -> {val_csv}")

    return train_df, val_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare train/val splits for ISIC 2018 Task 1 Lesion Segmentation."
    )
    parser.add_argument(
        "--images_dir", "--images-dir",
        type=Path,
        default=Path("data/raw/ISIC2018_Task1_Training_Input"),
        help="Directory containing ISIC 2018 Task 1 RGB images (or HAM10000 raw image dir)",
    )
    parser.add_argument(
        "--masks_dir", "--masks-dir",
        type=Path,
        default=Path("data/raw/ISIC2018_Task1_Training_GroundTruth"),
        help="Directory containing ISIC 2018 Task 1 binary mask PNGs",
    )
    parser.add_argument(
        "--aux_images_dir", "--aux-images-dir",
        type=Path,
        default=Path("data/raw"),
        help="Optional fallback directory to search for source images (e.g. HAM10000 part 1/2 folders)",
    )
    parser.add_argument(
        "--output_dir", "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory to save segmentation_train.csv and segmentation_val.csv",
    )
    parser.add_argument(
        "--train_ratio", "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data allocated to training set (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic train/val splitting (default: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    prepare_segmentation_splits(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        output_dir=args.output_dir,
        aux_images_dir=args.aux_images_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
