import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Maps short Kaggle dx codes to readable disease names
CLASS_NAMES = {
    "akiec": "Actinic keratoses",
    "bcc":   "Basal cell carcinoma",
    "bkl":   "Benign keratosis-like lesions",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic nevi",
    "vasc":  "Vascular lesions",
}

# Sorted so the index is consistent across runs
CLASSES = sorted(CLASS_NAMES.keys())
class_to_idx = {cls: i for i, cls in enumerate(CLASSES)}
idx_to_class = {i: cls for cls, i in class_to_idx.items()}


class HAM10000Dataset(Dataset):
    """
    PyTorch Dataset for the HAM10000 skin lesion dataset.

    Expects a CSV with at least: image_id, dx, image_path
    and optionally: age, sex, localization
    """

    def __init__(self, csv_file, image_size=224, transform=None, use_metadata=False):
        self.df = pd.read_csv(csv_file)
        self.image_size = image_size
        self.transform = transform
        self.use_metadata = use_metadata

        # Make sure required columns exist
        required = ["image_id", "dx", "image_path"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        image = Image.open(row["image_path"]).convert("RGB")
        image = np.array(image)

        # Apply Albumentations transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        label = class_to_idx[row["dx"]]

        metadata = {
            "image_id":     row["image_id"],
            "dx":           row["dx"],
            "age":          float(row["age"]) if pd.notna(row.get("age")) else None,
            "sex":          str(row["sex"]) if pd.notna(row.get("sex")) else "unknown",
            "localization": str(row["localization"]) if pd.notna(row.get("localization")) else "unknown",
            "image_path":   row["image_path"],
        }

        if self.use_metadata:
            return image, label, metadata
        return image, label, metadata


def collate_image_only(batch):
    """Collate function for standard image-only training (images, labels)."""
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return images, labels


def collate_with_metadata(batch):
    """
    Collate function for multimodal fusion training.
    Returns:
      images: (B, 3, H, W) FloatTensor
      labels: (B,) LongTensor
      metadata: dict with lists of 'age', 'sex', 'localization', 'image_id'
    """
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    
    metadata_list = [item[2] for item in batch]
    metadata_batch = {
        "age": [m["age"] for m in metadata_list],
        "sex": [m["sex"] for m in metadata_list],
        "localization": [m["localization"] for m in metadata_list],
        "image_id": [m["image_id"] for m in metadata_list],
    }
    return images, labels, metadata_batch


def get_train_transforms(image_size=224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            p=0.3
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


def get_eval_transforms(image_size=224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])

