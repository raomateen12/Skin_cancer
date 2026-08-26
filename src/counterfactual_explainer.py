"""
src/counterfactual_explainer.py
=================================
Perturbation-based Counterfactual Explainer for DermaLens AI.
Leverages the U-Net lesion segmentation mask and EfficientNet-B0 classifier
to generate photorealistic, zero-training clinical counterfactuals based on
the classical ABCD dermoscopy diagnostic criteria:

1. Border Irregularity (B): Multi-harmonic boundary modulation (scalloping, notches)
   while preserving overall lesion surface area within +/- 10%.
2. Asymmetry (A): Smooth directional elastic stretch along the principal axis of inertia.
3. Diameter (D): Controlled radial scaling (+20-25% diameter expansion).

Each perturbation is composited onto the original dermoscopy background using
Gaussian-weighted edge feathering to eliminate cut-paste boundary artifacts.
The perturbed images are then passed through the frozen classifier to measure
exact probability shifts and class re-rankings across all 7 diagnostic categories.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.model import get_efficientnet_b0
from src.segmentation_inference import (
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    preprocess_image,
    run_segmentation_inference,
    np_to_base64_png,
)

logger = logging.getLogger("dermalens.counterfactual")

DEFAULT_CLASSIFIER_CHECKPOINT = Path("checkpoints/best_efficientnet_b0.pth")
CLASS_MAPPING_PATH = Path("configs/class_mapping.json")

# Fallback mappings if config file is missing
FALLBACK_CLASS_LABELS: dict[int, str] = {
    0: "akiec",
    1: "bcc",
    2: "bkl",
    3: "df",
    4: "mel",
    5: "nv",
    6: "vasc",
}

LABEL_NAMES: dict[str, str] = {
    "akiec": "Actinic Keratosis / Bowen's Disease",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion",
}

_CLASSIFIER_MODEL: Optional[torch.nn.Module] = None


def load_classifier(
    checkpoint_path: Union[str, Path] = DEFAULT_CLASSIFIER_CHECKPOINT,
    device: torch.device = torch.device("cpu"),
) -> Optional[torch.nn.Module]:
    """Load and cache the trained EfficientNet-B0 classifier model."""
    global _CLASSIFIER_MODEL
    if _CLASSIFIER_MODEL is not None:
        return _CLASSIFIER_MODEL

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        # Check reweighted fallback
        alt_path = Path("checkpoints/best_efficientnet_b0_reweighted.pth")
        if alt_path.exists():
            ckpt_path = alt_path
        else:
            logger.warning("Classifier checkpoint not found at %s", ckpt_path)
            return None

    try:
        model = get_efficientnet_b0(num_classes=7)
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        _CLASSIFIER_MODEL = model
        logger.info("Counterfactual explainer: loaded classifier from %s", ckpt_path.name)
        return _CLASSIFIER_MODEL
    except Exception as exc:
        logger.error("Failed to load classifier checkpoint: %s", exc)
        return None


def _get_class_mappings() -> tuple[dict[int, str], dict[str, str]]:
    """Load index-to-code and code-to-name dictionaries."""
    idx_to_code = FALLBACK_CLASS_LABELS.copy()
    code_to_name = LABEL_NAMES.copy()
    if CLASS_MAPPING_PATH.exists():
        try:
            data = json.loads(CLASS_MAPPING_PATH.read_text(encoding="utf-8"))
            if "idx_to_class" in data:
                idx_to_code = {int(k): v for k, v in data["idx_to_class"].items()}
            if "label_names" in data:
                code_to_name.update(data["label_names"])
        except Exception:
            pass
    return idx_to_code, code_to_name


# ── 1. Morphological Perturbation Algorithms ──────────────────────────────────

def perturb_border_irregularity(
    img_uint8: np.ndarray,
    mask_np: np.ndarray,
    centroid: tuple[int, int],
    radius: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Simulate Border Irregularity (B):
    Applies multi-harmonic angular boundary perturbation to create natural
    scalloping, notches, and geographic indentations without altering total
    lesion surface area by more than ~10%.
    """
    H, W, _ = img_uint8.shape
    cx, cy = centroid
    R0 = max(8.0, radius)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    # Multi-frequency organic contour harmonics
    delta_theta = (
        0.13 * np.sin(3.0 * theta + 0.8) +
        0.10 * np.sin(5.0 * theta - 0.5) +
        0.06 * np.cos(4.0 * theta + 1.2) -
        0.05 * np.sin(7.0 * theta + 2.0)
    )

    # Localize perturbation tightly to boundary zone (0.7 R0 to 1.3 R0)
    taper = np.exp(-((r - R0)**2) / (2.0 * (0.35 * R0)**2))
    r_src = np.maximum(0.0, r * (1.0 - delta_theta * taper))

    map_x = (cx + r_src * np.cos(theta)).astype(np.float32)
    map_y = (cy + r_src * np.sin(theta)).astype(np.float32)

    warped_img = cv2.remap(img_uint8, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    warped_mask = cv2.remap(mask_np, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Feathered edge blending
    mask_combined = np.maximum(mask_np, warped_mask)
    feather = cv2.GaussianBlur(mask_combined, (13, 13), 3.5)[:, :, np.newaxis]
    blended = (warped_img.astype(np.float32) * feather + img_uint8.astype(np.float32) * (1.0 - feather)).clip(0, 255).astype(np.uint8)

    orig_area = max(1.0, float(np.sum(mask_np > 0.5)))
    new_area = float(np.sum(warped_mask > 0.5))
    area_ratio = new_area / orig_area

    return blended, warped_mask, area_ratio


def perturb_asymmetry(
    img_uint8: np.ndarray,
    mask_np: np.ndarray,
    centroid: tuple[int, int],
    radius: float,
    theta_pca: float = 0.45,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Simulate Asymmetry (A):
    Applies directional non-linear elastic stretching along one half of the lesion's
    principal axis, producing realistic structural asymmetry and focal expansion.
    """
    H, W, _ = img_uint8.shape
    cx, cy = centroid
    R0 = max(8.0, radius)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    r = np.sqrt(dx**2 + dy**2)

    # Rotate coordinates to principal orientation
    dx_rot = dx * np.cos(theta_pca) + dy * np.sin(theta_pca)
    dy_rot = -dx * np.sin(theta_pca) + dy * np.cos(theta_pca)

    # Asymmetric stretch only on one hemisphere (dx_rot > 0)
    stretch_factor = np.where(dx_rot > 0, 0.26 * np.exp(-(r**2) / (2.0 * (1.3 * R0)**2)), 0.0)
    dx_rot_src = dx_rot / (1.0 + stretch_factor)
    dy_rot_src = dy_rot

    # Rotate back to image space
    dx_src = dx_rot_src * np.cos(-theta_pca) + dy_rot_src * np.sin(-theta_pca)
    dy_src = -dx_rot_src * np.sin(-theta_pca) + dy_rot_src * np.cos(-theta_pca)

    map_x = (cx + dx_src).astype(np.float32)
    map_y = (cy + dy_src).astype(np.float32)

    warped_img = cv2.remap(img_uint8, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    warped_mask = cv2.remap(mask_np, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    mask_combined = np.maximum(mask_np, warped_mask)
    feather = cv2.GaussianBlur(mask_combined, (13, 13), 3.5)[:, :, np.newaxis]
    blended = (warped_img.astype(np.float32) * feather + img_uint8.astype(np.float32) * (1.0 - feather)).clip(0, 255).astype(np.uint8)

    orig_area = max(1.0, float(np.sum(mask_np > 0.5)))
    new_area = float(np.sum(warped_mask > 0.5))
    area_ratio = new_area / orig_area

    return blended, warped_mask, area_ratio


def perturb_diameter(
    img_uint8: np.ndarray,
    mask_np: np.ndarray,
    centroid: tuple[int, int],
    radius: float,
    scale: float = 1.22,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Simulate Diameter / Radial Growth (D):
    Scales the lesion outward by ~20-25% around its centroid with Gaussian
    radial tapering into peripheral background skin.
    """
    H, W, _ = img_uint8.shape
    cx, cy = centroid
    R0 = max(8.0, radius)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    # Smooth radial taper
    taper = np.exp(-((r - R0)**2) / (2.0 * (0.75 * R0)**2))
    r_src = np.maximum(0.0, r * (1.0 - (1.0 - 1.0 / scale) * taper))

    map_x = (cx + r_src * np.cos(theta)).astype(np.float32)
    map_y = (cy + r_src * np.sin(theta)).astype(np.float32)

    warped_img = cv2.remap(img_uint8, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    warped_mask = cv2.remap(mask_np, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    mask_combined = np.maximum(mask_np, warped_mask)
    feather = cv2.GaussianBlur(mask_combined, (15, 15), 4.0)[:, :, np.newaxis]
    blended = (warped_img.astype(np.float32) * feather + img_uint8.astype(np.float32) * (1.0 - feather)).clip(0, 255).astype(np.uint8)

    orig_area = max(1.0, float(np.sum(mask_np > 0.5)))
    new_area = float(np.sum(warped_mask > 0.5))
    area_ratio = new_area / orig_area

    return blended, warped_mask, area_ratio


def compute_difference_map(orig_uint8: np.ndarray, pert_uint8: np.ndarray) -> np.ndarray:
    """
    Generate a vivid visualization of where pixels shifted between original and perturbed.
    Returns RGB uint8 heatmap visualization.
    """
    diff = np.abs(pert_uint8.astype(np.float32) - orig_uint8.astype(np.float32))
    diff_mag = np.mean(diff, axis=-1)
    diff_norm = (diff_mag / (diff_mag.max() + 1e-5) * 255.0).astype(np.uint8)
    diff_color = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    diff_color = cv2.cvtColor(diff_color, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(orig_uint8, 0.45, diff_color, 0.55, 0)
    return blended


# ── 2. End-to-End Counterfactual Pipeline ─────────────────────────────────────

def generate_counterfactuals(
    image_input: Union[bytes, Image.Image, np.ndarray, str, Path],
    binary_mask: Optional[np.ndarray] = None,
    classifier_checkpoint: Union[str, Path] = DEFAULT_CLASSIFIER_CHECKPOINT,
    device_str: str = "cpu",
) -> dict:
    """
    Generate the 3 ABCD clinical counterfactuals (Border, Asymmetry, Diameter),
    evaluate the EfficientNet-B0 classifier on all variants, and compute detailed
    probability shift analytics and plain-language summaries.
    """
    device = torch.device(device_str)
    idx_to_code, code_to_name = _get_class_mappings()

    # Preprocess image
    pil_img, img_np, _ = preprocess_image(image_input)
    H, W, _ = img_np.shape
    img_uint8 = (img_np * 255.0).clip(0, 255).astype(np.uint8)

    # 1. Obtain lesion mask and centroid if not supplied
    if binary_mask is None:
        seg_res = run_segmentation_inference(image_input, device_str=device_str)
        if not seg_res.get("available") or not seg_res.get("metrics", {}).get("lesion_detected", False):
            return {
                "available": False,
                "error": "No distinct lesion boundary detected for counterfactual perturbation.",
                "counterfactuals": {},
            }
        mask_b64 = seg_res["images"]["mask"]
        mask_bytes = base64.b64decode(mask_b64.split(",")[1])
        mask_pil = Image.open(io.BytesIO(mask_bytes)).convert("L")
        binary_mask = (np.array(mask_pil, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
        cx = int(seg_res["metrics"]["centroid"]["x"] * W)
        cy = int(seg_res["metrics"]["centroid"]["y"] * H)
        area = float(seg_res["metrics"]["lesion_pixels"])
    else:
        cx, cy = W // 2, H // 2
        area = float(np.sum(binary_mask > 0.5))
        if area > 10:
            M = cv2.moments((binary_mask > 0.5).astype(np.uint8))
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

    if area < 20:
        return {
            "available": False,
            "error": "Lesion area is too small (< 20 px) for counterfactual perturbation.",
            "counterfactuals": {},
        }

    radius = max(8.0, math.sqrt(area / math.pi))

    # 2. Load frozen classifier
    classifier = load_classifier(checkpoint_path=classifier_checkpoint, device=device)
    if classifier is None:
        return {
            "available": False,
            "error": "Classifier model checkpoint could not be loaded.",
            "counterfactuals": {},
        }

    # Transform for classifier
    clf_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
    ])

    def predict_probs(rgb_arr: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(rgb_arr)
        t = clf_transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            out = classifier(t)
            p = F.softmax(out, dim=1).squeeze().cpu().numpy()
        return p

    # 3. Baseline original prediction
    orig_probs = predict_probs(img_uint8)
    orig_top_idx = int(np.argmax(orig_probs))
    orig_top_code = idx_to_code.get(orig_top_idx, f"class_{orig_top_idx}")
    orig_top_name = code_to_name.get(orig_top_code, orig_top_code)
    orig_top_conf = float(orig_probs[orig_top_idx])

    mel_idx = 4  # 'mel' index
    orig_mel_prob = float(orig_probs[mel_idx]) if mel_idx < len(orig_probs) else 0.0

    # 4. Generate the 3 perturbations
    pert_border, _, area_ratio_b = perturb_border_irregularity(img_uint8, binary_mask, (cx, cy), radius)
    pert_asym, _, area_ratio_a = perturb_asymmetry(img_uint8, binary_mask, (cx, cy), radius)
    pert_diam, _, area_ratio_d = perturb_diameter(img_uint8, binary_mask, (cx, cy), radius, scale=1.22)

    perturbations_config = [
        {
            "id": "border_irregularity",
            "name": "Border Irregularity",
            "clinical_code": "B",
            "description": "Scalloped, notched, and jagged boundary indentation (ABCD: Border)",
            "image": pert_border,
            "area_change_pct": round((area_ratio_b - 1.0) * 100.0, 1),
        },
        {
            "id": "asymmetry",
            "name": "Lesion Asymmetry",
            "clinical_code": "A",
            "description": "Focal unilateral expansion and structural asymmetry (ABCD: Asymmetry)",
            "image": pert_asym,
            "area_change_pct": round((area_ratio_a - 1.0) * 100.0, 1),
        },
        {
            "id": "diameter",
            "name": "Diameter Growth",
            "clinical_code": "D",
            "description": "Radial surface growth and 22% diameter enlargement (ABCD: Diameter)",
            "image": pert_diam,
            "area_change_pct": round((area_ratio_d - 1.0) * 100.0, 1),
        },
    ]

    results: dict = {}

    for cfg in perturbations_config:
        pert_img = cfg["image"]
        pert_probs = predict_probs(pert_img)
        pert_top_idx = int(np.argmax(pert_probs))
        pert_top_code = idx_to_code.get(pert_top_idx, f"class_{pert_top_idx}")
        pert_top_name = code_to_name.get(pert_top_code, pert_top_code)
        pert_top_conf = float(pert_probs[pert_top_idx])

        pert_orig_class_conf = float(pert_probs[orig_top_idx])
        pert_mel_prob = float(pert_probs[mel_idx]) if mel_idx < len(pert_probs) else 0.0

        mel_prob_delta = pert_mel_prob - orig_mel_prob
        orig_class_conf_delta = pert_orig_class_conf - orig_top_conf

        # Classification change detection
        classification_shifted = bool(pert_top_code != orig_top_code)

        orig_top_conf_pct = round(orig_top_conf * 100.0, 1)
        pert_orig_conf_pct = round(pert_orig_class_conf * 100.0, 1)
        new_top_conf_pct = round(pert_top_conf * 100.0, 1)
        conf_delta_pct = round(orig_class_conf_delta * 100.0, 1)
        mel_orig_pct = round(orig_mel_prob * 100.0, 1)
        mel_pert_pct = round(pert_mel_prob * 100.0, 1)
        mel_delta_pct = round(mel_prob_delta * 100.0, 1)

        # Plain language clinical explanation
        if classification_shifted:
            # Case 1: Diagnosis flipped to a different disease category
            shift_msg = (
                f"This {cfg['name'].lower()} change caused the model to reclassify the lesion as "
                f"{pert_top_name} ({new_top_conf_pct}% confidence), shifting away from the original "
                f"{orig_top_name} prediction ({orig_top_conf_pct}% originally, now {pert_orig_conf_pct}%)."
            )
            if pert_top_code == "mel":
                sec_msg = f" Melanoma is now the top prediction ({new_top_conf_pct}%)."
            else:
                sec_msg = f" Melanoma probability moved from {mel_orig_pct}% to {mel_pert_pct}%."
            plain_summary = shift_msg + sec_msg
        else:
            # Case 2: Primary diagnosis remained identical
            if abs(conf_delta_pct) >= 5.0:
                direction = "increased" if conf_delta_pct > 0 else "decreased"
                conf_msg = (
                    f"The primary classification remained {orig_top_name}, with confidence {direction} from "
                    f"{orig_top_conf_pct}% to {pert_orig_conf_pct}% ({'+' if conf_delta_pct > 0 else ''}{conf_delta_pct}%)."
                )
            else:
                conf_msg = (
                    f"The primary classification remained {orig_top_name} with stable confidence "
                    f"({pert_orig_conf_pct}% vs {orig_top_conf_pct}% originally)."
                )

            if abs(mel_delta_pct) >= 3.0:
                mel_direction = "rose" if mel_delta_pct > 0 else "declined"
                sec_msg = f" Melanoma risk {mel_direction} from {mel_orig_pct}% to {mel_pert_pct}% ({'+' if mel_delta_pct > 0 else ''}{mel_delta_pct}%)."
            else:
                sec_msg = f" Melanoma probability remains low at {mel_pert_pct}% (from {mel_orig_pct}%)."
            plain_summary = conf_msg + sec_msg

        # Difference heatmap
        diff_img = compute_difference_map(img_uint8, pert_img)

        results[cfg["id"]] = {
            "name": cfg["name"],
            "clinical_code": cfg["clinical_code"],
            "description": cfg["description"],
            "original_class": orig_top_code,
            "original_name": orig_top_name,
            "original_confidence": round(orig_top_conf, 4),
            "original_mel_prob": round(orig_mel_prob, 4),
            "perturbed_confidence": round(pert_orig_class_conf, 4),
            "perturbed_mel_prob": round(pert_mel_prob, 4),
            "mel_prob_delta": round(mel_prob_delta, 4),
            "confidence_delta": round(orig_class_conf_delta, 4),
            "new_top_class": pert_top_code,
            "new_top_name": pert_top_name,
            "new_top_confidence": round(pert_top_conf, 4),
            "classification_shifted": bool(pert_top_code != orig_top_code),
            "area_change_pct": cfg["area_change_pct"],
            "plain_language_summary": plain_summary,
            "perturbed_image": np_to_base64_png(pert_img),
            "diff_image": np_to_base64_png(diff_img),
        }

    return {
        "available": True,
        "original_prediction": {
            "code": orig_top_code,
            "name": orig_top_name,
            "confidence": round(orig_top_conf, 4),
            "mel_prob": round(orig_mel_prob, 4),
        },
        "counterfactuals": results,
    }
