"""
src/segmentation_dataset.py
PyTorch Dataset and DataLoader utilities for binary skin lesion segmentation.

Features:
  - Synchronised geometric augmentations (flips, rotations) across image and mask pairs
  - Bilinear interpolation for RGB images, Nearest Neighbor for binary masks
  - ImageNet normalization for images, float tensor [1, H, W] in {0.0, 1.0} for masks
"""

from __future__ import annotations
import random
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SegmentationDataset(Dataset):
    """
    Dataset loading paired dermoscopy images and lesion boundary binary masks.

    Parameters
    ----------
    data_source : str | Path | pd.DataFrame
        CSV path or DataFrame with columns: image_id, image_path, mask_path
    img_size : tuple[int, int], default=(224, 224)
        Target dimensions (H, W) for network input
    is_train : bool, default=True
        If True, applies random synchronized geometric augmentations
    normalize : bool, default=True
        If True, applies ImageNet mean/std normalization to images
    """

    def __init__(
        self,
        data_source: Union[str, Path, pd.DataFrame],
        img_size: Tuple[int, int] = (224, 224),
        is_train: bool = True,
        normalize: bool = True,
    ):
        super().__init__()
        if isinstance(data_source, (str, Path)):
            self.df = pd.read_csv(data_source)
        elif isinstance(data_source, pd.DataFrame):
            self.df = data_source.copy()
        else:
            raise TypeError(f"Unsupported data_source type: {type(data_source)}")

        # Ensure required columns exist
        if "image_id" not in self.df.columns or "image_path" not in self.df.columns:
            raise ValueError(f"Missing required columns (image_id, image_path) in dataset: {set(self.df.columns)}")

        self.has_masks = "mask_path" in self.df.columns and self.df["mask_path"].notna().any()

        self.img_size = img_size
        self.is_train = is_train
        self.normalize = normalize

        self.norm_transform = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) if normalize else None

    def __len__(self) -> int:
        return len(self.df)

    def _apply_transforms(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Resize (Bilinear for Image, Nearest for Mask to avoid interpolation blur)
        image = TF.resize(image, self.img_size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.img_size, interpolation=TF.InterpolationMode.NEAREST)

        # 2. Geometric augmentations (Synchronized for Training)
        if self.is_train:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # Random 90/180/270 degree rotation
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

            # Subtle random rotation (-15 to 15 deg)
            if random.random() > 0.5:
                angle = random.uniform(-15.0, 15.0)
                image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        # 3. Convert Image to Tensor [3, H, W] (0.0 to 1.0)
        img_tensor = TF.to_tensor(image)
        if self.norm_transform is not None:
            img_tensor = self.norm_transform(img_tensor)

        # 4. Convert Mask to Tensor [1, H, W] ({0.0, 1.0})
        mask_np = np.array(mask, dtype=np.float32)
        # Threshold at 128 (ISIC masks are 0 or 255)
        mask_binary = (mask_np >= 128.0).astype(np.float32)
        if mask_binary.ndim == 2:
            mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)  # [1, H, W]
        else:
            mask_tensor = torch.from_numpy(mask_binary[:, :, 0]).unsqueeze(0)

        return img_tensor, mask_tensor

    def __getitem__(self, idx: int) -> dict[str, Union[torch.Tensor, str, bool]]:
        row = self.df.iloc[idx]
        img_path = str(row["image_path"])
        image_id = str(row["image_id"])

        image = Image.open(img_path).convert("RGB")
        has_gt = False

        if "mask_path" in row and pd.notna(row["mask_path"]) and Path(str(row["mask_path"])).exists():
            mask = Image.open(str(row["mask_path"])).convert("L")
            has_gt = True
        else:
            # Create placeholder mask of same size as image
            mask = Image.new("L", image.size, color=0)

        img_tensor, mask_tensor = self._apply_transforms(image, mask)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "image_id": image_id,
            "image_path": img_path,
            "mask_path": str(row.get("mask_path", "")),
            "has_ground_truth": has_gt,
        }


def get_segmentation_loaders(
    train_csv: Union[str, Path, pd.DataFrame],
    val_csv: Union[str, Path, pd.DataFrame],
    batch_size: int = 16,
    img_size: Tuple[int, int] = (224, 224),
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Factory helper to build high-throughput Train and Validation DataLoader instances."""
    train_ds = SegmentationDataset(train_csv, img_size=img_size, is_train=True)
    val_ds = SegmentationDataset(val_csv, img_size=img_size, is_train=False)

    use_cuda = torch.cuda.is_available()
    pin = pin_memory and use_cuda

    tr_workers = min(num_workers, len(train_ds)) if num_workers > 0 else 0
    val_workers = min(num_workers, len(val_ds)) if num_workers > 0 else 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=tr_workers,
        pin_memory=pin,
        persistent_workers=persistent_workers and (tr_workers > 0),
        prefetch_factor=2 if tr_workers > 0 else None,
        drop_last=len(train_ds) > batch_size,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=pin,
        persistent_workers=persistent_workers and (val_workers > 0),
        prefetch_factor=2 if val_workers > 0 else None,
    )

    return train_loader, val_loader

