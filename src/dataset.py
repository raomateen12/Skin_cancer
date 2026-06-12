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

    def __init__(self, csv_file, image_size=224, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_size = image_size
        self.transform = transform

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
            "age":          row.get("age", None),
            "sex":          row.get("sex", None),
            "localization": row.get("localization", None),
            "image_path":   row["image_path"],
        }

        return image, label, metadata


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
