"""
src/segmentation_inference.py
==============================
Inference and explainability helper for the trained U-Net Skin Lesion Boundary
Segmentation model.

Provides:
- Model loading and caching (singleton pattern)
- Forward inference on raw bytes / PIL Images / NumPy arrays
- Binary mask generation and probability heatmap computation
- Lesion boundary contour overlay rendering (high-contrast clinical aesthetic)
- Lesion morphological metrics extraction:
    * Lesion Area Percentage (% of FOV)
    * Perimeter (pixels)
    * Compactness / Isoperimetric Quotient (4 * pi * Area / Perimeter^2)
    * Border Irregularity Index (asymmetry and convexity deficit)
    * Centroid coordinates & bounding box
- Base64 data URI export for FastAPI and Next.js frontend consumption
"""

from __future__ import annotations

import base64
import io
import logging
import math
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image

from src.unet_model import get_unet

logger = logging.getLogger("dermalens.segmentation")

# ── Defaults & Constants ──────────────────────────────────────────────────────
DEFAULT_CHECKPOINT_PATH = Path("checkpoints/best_unet.pth")
IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_UNET_MODEL: Optional[torch.nn.Module] = None
_UNET_METADATA: dict = {}


def get_segmentation_model(
    checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT_PATH,
    device_str: str = "cpu",
    force_reload: bool = False,
) -> Optional[torch.nn.Module]:
    """
    Load and cache the trained U-Net model from checkpoint.
    Returns None if the checkpoint file does not exist.
    """
    global _UNET_MODEL, _UNET_METADATA
    if _UNET_MODEL is not None and not force_reload:
        return _UNET_MODEL

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        logger.warning("U-Net checkpoint not found at: %s", ckpt_path)
        return None

    try:
        device = torch.device(device_str)
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

        base_channels = ckpt.get("base_channels", 32)
        model = get_unet(in_channels=3, out_channels=1, base_channels=base_channels)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        _UNET_MODEL = model
        _UNET_METADATA = {
            "epoch": ckpt.get("epoch"),
            "val_dice": ckpt.get("val_dice"),
            "val_iou": ckpt.get("val_iou"),
            "val_loss": ckpt.get("val_loss"),
            "base_channels": base_channels,
            "img_size": ckpt.get("img_size", IMG_SIZE),
            "checkpoint_path": str(ckpt_path),
        }
        logger.info(
            "Loaded U-Net successfully (base_channels=%d, val_dice=%.4f, val_iou=%.4f).",
            base_channels,
            ckpt.get("val_dice", 0.0) or 0.0,
            ckpt.get("val_iou", 0.0) or 0.0,
        )
        return _UNET_MODEL
    except Exception as exc:
        logger.error("Failed to load U-Net checkpoint (%s: %s)", type(exc).__name__, exc, exc_info=True)
        return None


def get_segmentation_metadata() -> dict:
    """Return checkpoint metadata (epoch, val_dice, val_iou, etc.)."""
    return dict(_UNET_METADATA)


def preprocess_image(image_input: Union[bytes, Image.Image, np.ndarray, str, Path]) -> tuple[Image.Image, np.ndarray, torch.Tensor]:
    """
    Load, resize to 224x224, and normalize image.
    Returns:
        (pil_image_224, np_rgb_float32_normalized_0_1, input_tensor_1_3_H_W)
    """
    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        if image_input.dtype == np.uint8:
            pil_img = Image.fromarray(image_input).convert("RGB")
        else:
            pil_img = Image.fromarray((image_input * 255).astype(np.uint8)).convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    pil_img_224 = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    img_np = np.array(pil_img_224, dtype=np.float32) / 255.0

    tensor = (img_np - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.tensor(tensor, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    return pil_img_224, img_np, tensor


def compute_morphological_metrics(binary_mask: np.ndarray) -> dict:
    """
    Compute quantitative clinical morphological descriptors from the 2D binary mask.
    Mask shape: (H, W) with values {0, 1} or bool.
    """
    mask_uint8 = (binary_mask > 0.5).astype(np.uint8) * 255
    total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
    lesion_pixels = int(np.sum(mask_uint8 > 0))
    area_pct = round(float((lesion_pixels / total_pixels) * 100.0), 2)

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours or lesion_pixels < 10:
        return {
            "lesion_detected": False,
            "area_pct": 0.0,
            "lesion_pixels": lesion_pixels,
            "perimeter_px": 0.0,
            "compactness": 0.0,
            "border_irregularity_score": 0.0,
            "aspect_ratio": 1.0,
            "solidity": 0.0,
            "centroid": {"x": 0.5, "y": 0.5},
            "bounding_box": {"x": 0, "y": 0, "width": 0, "height": 0},
            "num_lesion_components": 0,
        }

    # Largest contour is the primary lesion
    main_contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(main_contour))
    perimeter = float(cv2.arcLength(main_contour, closed=True))

    if area < 5.0 or perimeter <= 0.0:
        return {
            "lesion_detected": False,
            "area_pct": area_pct,
            "lesion_pixels": lesion_pixels,
            "perimeter_px": round(perimeter, 1),
            "compactness": 0.0,
            "border_irregularity_score": 0.0,
            "aspect_ratio": 1.0,
            "solidity": 0.0,
            "centroid": {"x": 0.5, "y": 0.5},
            "bounding_box": {"x": 0, "y": 0, "width": 0, "height": 0},
            "num_lesion_components": len(contours),
        }

    # Compactness (Isoperimetric Quotient): 4 * pi * Area / Perimeter^2 (1.0 = perfect circle)
    compactness = 0.0
    if perimeter > 0:
        compactness = min(1.0, float((4.0 * math.pi * area) / (perimeter ** 2)))

    # Convex hull for border irregularity & solidity
    hull = cv2.convexHull(main_contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = min(1.0, float(area / hull_area)) if hull_area > 0 else 1.0

    # Border irregularity index (0 = very regular/smooth, 1 = highly irregular/spiculated)
    # Combines 1 - compactness and 1 - solidity
    irregularity = float(0.6 * (1.0 - compactness) + 0.4 * (1.0 - solidity))
    irregularity = max(0.0, min(1.0, irregularity))

    # Bounding box & aspect ratio
    x, y, w, h = cv2.boundingRect(main_contour)
    aspect_ratio = float(max(w, h) / max(min(w, h), 1))

    # Centroid
    moments = cv2.moments(main_contour)
    if moments["m00"] != 0:
        cx = float(moments["m10"] / moments["m00"]) / binary_mask.shape[1]
        cy = float(moments["m01"] / moments["m00"]) / binary_mask.shape[0]
    else:
        cx, cy = 0.5, 0.5

    return {
        "lesion_detected": True,
        "area_pct": area_pct,
        "lesion_pixels": lesion_pixels,
        "perimeter_px": round(float(perimeter), 1),
        "compactness": round(float(compactness), 3),
        "solidity": round(float(solidity), 3),
        "border_irregularity_score": round(float(irregularity), 3),
        "aspect_ratio": round(float(aspect_ratio), 2),
        "centroid": {"x": round(cx, 3), "y": round(cy, 3)},
        "bounding_box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
        "num_lesion_components": len(contours),
    }


def draw_segmentation_overlay(
    img_uint8: np.ndarray,
    binary_mask: np.ndarray,
    prob_map: np.ndarray,
    contour_color: tuple[int, int, int] = (0, 230, 255),  # Cyan/Lime
    fill_alpha: float = 0.28,
) -> np.ndarray:
    """
    Render a high-contrast boundary contour overlay with translucent lesion tinting.
    img_uint8: (H, W, 3) RGB uint8.
    binary_mask: (H, W) float/bool.
    prob_map: (H, W) float [0, 1].
    """
    overlay = img_uint8.copy()
    mask_bool = binary_mask > 0.5

    # Translucent lesion tint (subtle cool highlight)
    tint_layer = img_uint8.copy()
    tint_color = np.array([20, 180, 240], dtype=np.uint8)  # Azure/cyan highlight
    tint_layer[mask_bool] = (
        tint_layer[mask_bool] * (1.0 - fill_alpha) + tint_color * fill_alpha
    ).astype(np.uint8)

    # Find contours and draw high-visibility smoothed boundary lines
    mask_uint8 = (mask_bool.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

    # Draw outer glow and crisp primary border
    cv2.drawContours(tint_layer, contours, -1, (0, 60, 120), thickness=4, lineType=cv2.LINE_AA)
    cv2.drawContours(tint_layer, contours, -1, contour_color, thickness=2, lineType=cv2.LINE_AA)

    return tint_layer


def draw_heatmap_overlay(img_uint8: np.ndarray, prob_map: np.ndarray, colormap: int = cv2.COLORMAP_MAGMA, alpha: float = 0.55) -> np.ndarray:
    """
    Create a colormapped probability heatmap overlay on top of the original dermoscopy image.
    """
    prob_clipped = np.clip(prob_map, 0.0, 1.0)
    heatmap_uint8 = (prob_clipped * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(img_uint8, 1.0 - alpha, heatmap_colored, alpha, 0)
    return blended


def np_to_base64_png(arr_rgb_uint8: np.ndarray) -> str:
    """Encode uint8 RGB NumPy image as data:image/png;base64,... URI."""
    pil = Image.fromarray(arr_rgb_uint8)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def run_segmentation_inference(
    image_input: Union[bytes, Image.Image, np.ndarray, str, Path],
    threshold: float = 0.5,
    checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT_PATH,
    device_str: str = "cpu",
) -> dict:
    """
    Complete end-to-end lesion segmentation pipeline.

    Returns dict with:
      - available: bool
      - error: Optional[str]
      - metrics: dict (area_pct, perimeter, compactness, border_irregularity_score, etc.)
      - images: {
            "original": base64_png,
            "mask": base64_png,
            "overlay": base64_png,
            "heatmap": base64_png
        }
      - raw_prob_summary: {min, max, mean}
    """
    model = get_segmentation_model(checkpoint_path=checkpoint_path, device_str=device_str)
    if model is None:
        return {
            "available": False,
            "error": f"Segmentation model checkpoint not available at {checkpoint_path}",
            "metrics": {},
            "images": {},
        }

    try:
        pil_img, img_np, tensor = preprocess_image(image_input)
        img_uint8 = (img_np * 255.0).clip(0, 255).astype(np.uint8)

        device = next(model.parameters()).device
        tensor = tensor.to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        binary_mask = (probs > threshold).astype(np.float32)

        # Morphological descriptors
        morph_metrics = compute_morphological_metrics(binary_mask)

        # Render visualizations
        overlay_rgb = draw_segmentation_overlay(img_uint8, binary_mask, probs)
        heatmap_rgb = draw_heatmap_overlay(img_uint8, probs)
        mask_rgb = np.stack([(binary_mask * 255).astype(np.uint8)] * 3, axis=-1)

        # Base64 encodings
        orig_b64 = np_to_base64_png(img_uint8)
        mask_b64 = np_to_base64_png(mask_rgb)
        overlay_b64 = np_to_base64_png(overlay_rgb)
        heatmap_b64 = np_to_base64_png(heatmap_rgb)

        return {
            "available": True,
            "threshold": threshold,
            "metrics": morph_metrics,
            "images": {
                "original": orig_b64,
                "mask": mask_b64,
                "overlay": overlay_b64,
                "heatmap": heatmap_b64,
            },
            "raw_prob_summary": {
                "min": round(float(probs.min()), 4),
                "max": round(float(probs.max()), 4),
                "mean": round(float(probs.mean()), 4),
            },
            "checkpoint_info": get_segmentation_metadata(),
        }

    except Exception as exc:
        logger.error("Segmentation inference failed (%s: %s)", type(exc).__name__, exc, exc_info=True)
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "metrics": {},
            "images": {},
        }
