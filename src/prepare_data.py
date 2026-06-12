"""
Prepare the HAM10000 dataset:
  - Scan data/raw recursively for all .jpg images
  - Match images to metadata CSV
  - Add readable disease labels and integer label_id
  - Create stratified 80/10/10 train/val/test split
  - Save CSVs and class mapping to data/processed/
"""

import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
CONFIG_FILE = Path("configs/config.yaml")

# Readable disease names for each dx code
LABEL_NAMES = {
    "akiec": "Actinic keratoses",
    "bcc":   "Basal cell carcinoma",
    "bkl":   "Benign keratosis-like lesions",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic nevi",
    "vasc":  "Vascular lesions",
}


def load_seed():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("seed", 42)
    return 42


def find_images(raw_dir):
    """Recursively scan raw_dir and build a dict of image_id -> full path."""
    print(f"Scanning {raw_dir} for .jpg images...")
    image_map = {}
    for img_path in raw_dir.rglob("*.jpg"):
        image_id = img_path.stem  # filename without extension
        image_map[image_id] = str(img_path)
    print(f"  Found {len(image_map)} images.")
    return image_map


def find_metadata(raw_dir):
    """Look for HAM10000_metadata.csv inside raw_dir."""
    candidates = list(raw_dir.rglob("HAM10000_metadata.csv"))
    if not candidates:
        print("[ERROR] HAM10000_metadata.csv not found inside data/raw.")
        print("  Run: python -m src.download_kaggle_dataset")
        sys.exit(1)
    if len(candidates) > 1:
        print(f"  Multiple metadata CSVs found, using: {candidates[0]}")
    return candidates[0]


def main():
    seed = load_seed()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load metadata
    meta_path = find_metadata(RAW_DIR)
    df = pd.read_csv(meta_path)
    print(f"\nLoaded metadata: {len(df)} rows")

    # Build image map and attach paths
    image_map = find_images(RAW_DIR)
    df["image_path"] = df["image_id"].map(image_map)

    missing = df["image_path"].isna().sum()
    if missing > 0:
        print(f"  Warning: {missing} rows dropped (image file not found).")
    df = df.dropna(subset=["image_path"]).reset_index(drop=True)
    print(f"  Remaining rows after dropping missing images: {len(df)}")

    # Add readable label and integer label_id
    df["label_name"] = df["dx"].map(LABEL_NAMES)
    classes = sorted(LABEL_NAMES.keys())
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    df["label_id"] = df["dx"].map(class_to_idx)

    # Print class distribution
    print("\nClass distribution:")
    dist = df["dx"].value_counts()
    for dx, count in dist.items():
        print(f"  {dx:6s} ({LABEL_NAMES.get(dx, dx)}): {count}")

    # Stratified 80/10/10 split
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["dx"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["dx"], random_state=seed
    )

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print(f"\nSplits saved to {PROCESSED_DIR}:")
    print(f"  train: {len(train_df)} samples")
    print(f"  val:   {len(val_df)} samples")
    print(f"  test:  {len(test_df)} samples")

    # Save class mapping
    mapping = {
        "class_to_idx": class_to_idx,
        "idx_to_class": {str(v): k for k, v in class_to_idx.items()},
        "label_names": LABEL_NAMES,
    }
    mapping_path = PROCESSED_DIR / "class_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"  class_mapping.json saved.")


if __name__ == "__main__":
    main()
