"""
Multimodal Metadata Fusion Model for Skin Lesion Classification.
Combines 1280-dim EfficientNet-B0 visual representations with tabular patient metadata (age, sex, localization).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

DEFAULT_STATS_PATH = Path("data/processed/metadata_stats.json")

# Fallback default statistics derived from HAM10000 train split
DEFAULT_METADATA_STATS = {
    "mean_age": 51.9152,
    "age_min": 0.0,
    "age_max": 85.0,
    "sex_categories": ["male", "female"],
    "localization_categories": [
        "back",
        "lower extremity",
        "trunk",
        "upper extremity",
        "abdomen",
        "face",
        "chest",
        "unknown",
    ],
    "metadata_dim": 11,
}


class MetadataEncoder:
    """
    Encodes tabular patient metadata (age, sex, localization) into fixed-dimension FloatTensors.
    Input dimensions (11 total):
      - age (1 dim): normalized to [0, 1] with mean imputation for missing values.
      - sex (2 dims): one-hot encoding [male=[1,0], female=[0,1], unknown=[0,0]].
      - localization (8 dims): one-hot encoding across top-8 locations; others map to 'unknown'.
    """

    def __init__(self, stats_path: Optional[Union[str, Path]] = None):
        self.stats = self._load_stats(stats_path)
        self.mean_age = float(self.stats.get("mean_age", 51.9152))
        self.age_min = float(self.stats.get("age_min", 0.0))
        self.age_max = float(self.stats.get("age_max", 85.0))
        self.sex_categories = self.stats.get("sex_categories", ["male", "female"])
        self.localization_categories = self.stats.get(
            "localization_categories",
            [
                "back",
                "lower extremity",
                "trunk",
                "upper extremity",
                "abdomen",
                "face",
                "chest",
                "unknown",
            ],
        )
        self.loc_to_idx = {loc: i for i, loc in enumerate(self.localization_categories)}
        self.metadata_dim = 1 + len(self.sex_categories) + len(self.localization_categories)

    def _load_stats(self, stats_path: Optional[Union[str, Path]]) -> dict:
        path = Path(stats_path) if stats_path else DEFAULT_STATS_PATH
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return DEFAULT_METADATA_STATS

    def encode_single(
        self,
        age: Optional[float] = None,
        sex: Optional[str] = None,
        localization: Optional[str] = None,
    ) -> np.ndarray:
        """Encodes a single patient's metadata into an 11-dimensional float vector."""
        vec = np.zeros(self.metadata_dim, dtype=np.float32)

        # 1. Age: impute NaN with train mean, normalize to [0, 1]
        if age is None or (isinstance(age, float) and (np.isnan(age) or np.isinf(age))):
            val_age = self.mean_age
        else:
            try:
                val_age = float(age)
            except (ValueError, TypeError):
                val_age = self.mean_age
        norm_age = (val_age - self.age_min) / max(1.0, (self.age_max - self.age_min))
        vec[0] = float(np.clip(norm_age, 0.0, 1.0))

        # 2. Sex: male=[1,0], female=[0,1], unknown/other=[0,0]
        sex_str = str(sex).strip().lower() if sex is not None else "unknown"
        if sex_str == "male":
            vec[1] = 1.0
        elif sex_str == "female":
            vec[2] = 1.0

        # 3. Localization: one-hot for top 8; anything else maps to 'unknown'
        loc_str = str(localization).strip().lower() if localization is not None else "unknown"
        idx = self.loc_to_idx.get(loc_str, self.loc_to_idx.get("unknown", 7))
        vec[3 + idx] = 1.0

        return vec

    def encode_batch(
        self,
        metadata: Union[Dict[str, Any], List[Dict[str, Any]], torch.Tensor, np.ndarray],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Converts batch metadata into a torch.FloatTensor of shape (B, 11).
        Supports metadata as:
          - Pre-encoded Tensor/ndarray of shape (B, 11)
          - Dict with keys 'age', 'sex', 'localization' where values are lists/tensors
          - List of metadata dicts [{'age': ..., 'sex': ..., 'localization': ...}, ...]
        """
        if isinstance(metadata, torch.Tensor):
            return metadata.to(device) if device else metadata
        if isinstance(metadata, np.ndarray):
            t = torch.from_numpy(metadata.astype(np.float32))
            return t.to(device) if device else t

        if isinstance(metadata, dict):
            # Dict of lists / tensors format
            ages = metadata.get("age", [])
            sexes = metadata.get("sex", [])
            locs = metadata.get("localization", [])

            # Handle scalar or tensor forms
            if isinstance(ages, (int, float)) or (isinstance(ages, torch.Tensor) and ages.ndim == 0):
                ages = [ages]
                sexes = [sexes]
                locs = [locs]
            elif isinstance(ages, torch.Tensor):
                ages = ages.tolist()

            batch_size = max(len(ages), len(sexes), len(locs))
            if batch_size == 0:
                batch_size = 1
                ages = [None]
                sexes = [None]
                locs = [None]

            encoded_rows = []
            for i in range(batch_size):
                a = ages[i] if i < len(ages) else None
                s = sexes[i] if i < len(sexes) else None
                l = locs[i] if i < len(locs) else None
                encoded_rows.append(self.encode_single(a, s, l))

            arr = np.stack(encoded_rows, axis=0)
            t = torch.from_numpy(arr)
            return t.to(device) if device else t

        if isinstance(metadata, list):
            # List of dicts format
            encoded_rows = []
            for item in metadata:
                if isinstance(item, dict):
                    encoded_rows.append(
                        self.encode_single(
                            item.get("age"),
                            item.get("sex"),
                            item.get("localization"),
                        )
                    )
                else:
                    encoded_rows.append(self.encode_single())
            arr = np.stack(encoded_rows, axis=0) if encoded_rows else np.zeros((0, self.metadata_dim), dtype=np.float32)
            t = torch.from_numpy(arr)
            return t.to(device) if device else t

        # Fallback single empty item
        vec = self.encode_single()
        t = torch.from_numpy(vec).unsqueeze(0)
        return t.to(device) if device else t


class MetadataFusionModel(nn.Module):
    """
    Multimodal Skin Lesion Classifier.
    Fuses 1280-dim EfficientNet-B0 image features with 11-dim encoded patient metadata.
    """

    def __init__(
        self,
        num_classes: int = 7,
        freeze_backbone: bool = False,
        pretrained_weights: bool = True,
        stats_path: Optional[Union[str, Path]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        self.encoder = MetadataEncoder(stats_path=stats_path)
        self.metadata_dim = self.encoder.metadata_dim  # 11

        # 1. Visual Backbone: EfficientNet-B0 (1280-dim features)
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained_weights else None
        base_model = efficientnet_b0(weights=weights)

        # Extract features without classifier head
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.image_feature_dim = 1280

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
            for param in self.avgpool.parameters():
                param.requires_grad = False

        # 2. Metadata Branch: 11 -> 64 -> 64
        self.metadata_branch = nn.Sequential(
            nn.Linear(self.metadata_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.metadata_feature_dim = 64

        # 3. Multimodal Fusion Head: (1280 + 64 = 1344) -> 256 -> num_classes (7)
        fusion_in_dim = self.image_feature_dim + self.metadata_feature_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def extract_image_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 1280-dim visual representation vector from input image batch."""
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        return feat

    def forward(
        self,
        image_tensor: torch.Tensor,
        metadata: Union[Dict[str, Any], List[Dict[str, Any]], torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        """
        Forward pass for multimodal classification.
        Parameters:
          image_tensor: (B, 3, H, W) normalized RGB image batch
          metadata: dictionary of metadata fields, list of dicts, or pre-encoded (B, 11) tensor
        Returns:
          logits: (B, num_classes) classification logits
        """
        # 1. Process image features
        img_feats = self.extract_image_features(image_tensor)  # (B, 1280)

        # 2. Process metadata features
        meta_tensor = self.encoder.encode_batch(metadata, device=image_tensor.device)  # (B, 11)
        meta_feats = self.metadata_branch(meta_tensor)  # (B, 64)

        # 3. Concatenate and pass through fusion head
        fused = torch.cat([img_feats, meta_feats], dim=1)  # (B, 1344)
        logits = self.fusion_head(fused)  # (B, 7)
        return logits


def get_metadata_fusion_model(
    num_classes: int = 7,
    freeze_backbone: bool = False,
    pretrained_weights: bool = True,
    stats_path: Optional[Union[str, Path]] = None,
) -> MetadataFusionModel:
    """Helper factory for instantiating MetadataFusionModel."""
    return MetadataFusionModel(
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        pretrained_weights=pretrained_weights,
        stats_path=stats_path,
    )
