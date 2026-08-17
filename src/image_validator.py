"""
DermaLens AI — Image Validation Gate
=====================================
Validates whether an uploaded image is a suitable close-up skin lesion image
before ML inference is performed.

Rejects: portraits/selfies, screenshots, documents, logos, objects, landscapes.
Allows:  dermoscopic close-ups, macro skin photos, clinical lesion photos.
Warns:   borderline cases (low skin ratio, uncertain composition).

Conservative design: fail-open.  If this module crashes, the predict
endpoint proceeds rather than blocking a valid upload.

Usage (from api/main.py):
    from src.image_validator import validate_skin_lesion_image
    result = validate_skin_lesion_image(image_bytes)
    if not result["is_valid"]:
        return rejection_response(result)
"""

from __future__ import annotations

import io
import logging
from typing import Any

logger = logging.getLogger("dermalens.validator")


# ── Public API ───────────────────────────────────────────────────────────────

def validate_skin_lesion_image(image_bytes: bytes) -> dict[str, Any]:
    """
    Validate whether uploaded image is a plausible close-up skin lesion photo.

    Returns one of:
      {"is_valid": True,  "confidence": float, "warnings": [], "reason": None}
      {"is_valid": True,  "confidence": float, "warnings": [...], "reason": None}  ← uncertain
      {"is_valid": False, "confidence": float, "reason": str, "guidance": str, "warnings": []}
    """
    try:
        import numpy as np
        from PIL import Image as PILImage

        pil_img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = pil_img.size

        # ── Check 1: Extreme aspect ratio ────────────────────────────────────
        # Dermoscopic images are roughly square (≤ 2:1).
        # Screenshots, panoramas and phone UIs are often 16:9+ or very tall.
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > 3.5:
            return _reject(
                "extreme_aspect_ratio",
                "This image does not appear to be a close-up skin lesion photo "
                "(extreme aspect ratio detected).",
                "Please upload a close-up photo of the skin area or lesion with "
                "roughly equal width and height.",
                0.95,
            )

        # ── Check 2: Minimum resolution ──────────────────────────────────────
        if w < 50 or h < 50:
            return _reject(
                "too_small",
                "The uploaded image is too small to analyze meaningfully.",
                "Please upload a higher-resolution, close-up photo of the skin "
                "lesion (at least 50 × 50 pixels).",
                0.99,
            )

        # Resize to 224×224 for all subsequent pixel-level checks
        thumb = pil_img.resize((224, 224))
        img_np = np.array(thumb, dtype=np.float32)
        img_uint8 = img_np.clip(0, 255).astype(np.uint8)
        total_pixels = 224 * 224

        r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]

        # ── Check 3: Near-white pixel ratio (screenshot / document) ──────────
        near_white = ((r > 230) & (g > 230) & (b > 230)).sum()
        white_ratio = float(near_white) / total_pixels
        if white_ratio > 0.65:
            return _reject(
                "screenshot_white",
                "This image appears to be a screenshot, document, or webpage "
                "rather than a skin photo.",
                "Please upload a direct photo of the skin area or lesion "
                "in good lighting.",
                0.93,
            )

        # ── Check 4: Near-black pixel ratio (dark UI / video frame) ──────────
        near_black = ((r < 25) & (g < 25) & (b < 25)).sum()
        black_ratio = float(near_black) / total_pixels
        if black_ratio > 0.60:
            return _reject(
                "screenshot_dark",
                "This image appears to be a dark-background screenshot or video "
                "frame, not a skin photo.",
                "Please upload a clear, well-lit close-up photo of the skin area "
                "or lesion.",
                0.91,
            )

        # ── Check 5: Global color variance (flat / solid-color images) ───────
        global_std = float(img_np.std())
        if global_std < 8.0:
            return _reject(
                "flat_image",
                "The image appears nearly uniform in color and does not resemble "
                "a skin lesion photo.",
                "Please upload a clear, close-up photo of the skin area or lesion.",
                0.85,
            )

        # ── OpenCV-dependent checks (fail-safe if cv2 missing) ───────────────
        skin_ratio: float = 0.5          # default — assume valid if cv2 absent
        edge_density: float = 0.0
        warnings: list[str] = []

        try:
            import cv2

            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

            # ── Check 6: Face / portrait detection ───────────────────────────
            # Close-up dermoscopic images should NOT contain a recognisable
            # human face.  If OpenCV's frontal-face cascade fires, this is
            # almost certainly a selfie or portrait — reject it.
            cascade_path = (
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            cascade = cv2.CascadeClassifier(cascade_path)
            if not cascade.empty():
                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.05,
                    minNeighbors=4,
                    minSize=(35, 35),
                )
                if len(faces) > 0:
                    face_area = sum(int(fw) * int(fh) for (_, _, fw, fh) in faces)
                    face_ratio = face_area / total_pixels
                    logger.info(
                        "Face detected: count=%d  face_ratio=%.3f",
                        len(faces),
                        face_ratio,
                    )
                    # Even a small face detection (>3% of image) in a 224×224
                    # thumbnail indicates a portrait/selfie context.
                    if face_ratio > 0.03:
                        return _reject(
                            "portrait_face_detected",
                            "This image appears to be a portrait or selfie, not a "
                            "close-up skin lesion photo.",
                            "Please upload a clear, close-up photo of the specific "
                            "skin area or lesion only — not a portrait or full-face "
                            "photo.",
                            0.92,
                        )
            else:
                logger.warning(
                    "Haar cascade not loaded — skipping face detection"
                )

            # ── Check 7: Edge density (screenshot / document / text) ──────────
            # Screenshots and documents have many sharp edges (text, icons, borders).
            # Skin close-ups have smooth gradients with relatively few sharp edges.
            edges = cv2.Canny(gray, 50, 150)
            edge_density = float(edges.mean()) / 255.0
            logger.info("Edge density: %.4f", edge_density)
            if edge_density > 0.18:
                return _reject(
                    "high_edge_density",
                    "This image appears to contain heavy text, UI elements, or a "
                    "screenshot rather than a skin lesion photo.",
                    "Please upload a direct photo of the skin area or lesion "
                    "in good lighting.",
                    0.88,
                )

            # ── Check 8: Skin-tone pixel ratio (YCrCb space) ─────────────────
            # Standard dermatology skin-tone range across all ethnicities.
            ycrcb = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2YCrCb)
            Y, Cr, Cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]
            skin_mask = (
                (Y > 60) & (Y < 255) &
                (Cr > 120) & (Cr < 185) &
                (Cb > 60) & (Cb < 135)
            )
            skin_ratio = float(skin_mask.sum()) / total_pixels
            logger.info(
                "Validation stats — skin_ratio=%.3f  white=%.3f  "
                "black=%.3f  edge=%.4f  std=%.2f",
                skin_ratio, white_ratio, black_ratio, edge_density, global_std,
            )

            if skin_ratio < 0.04:
                return _reject(
                    "no_skin_tone",
                    "This image does not appear to contain skin tones. "
                    "It may be an object, logo, or unrelated photo.",
                    "Please upload a clear, close-up photo of a skin area or "
                    "lesion in good lighting.",
                    0.87,
                )

            # Uncertain — low skin ratio → allow but warn
            if skin_ratio < 0.12:
                logger.info(
                    "Validation WARN: low skin_ratio=%.3f — allowing with warning",
                    skin_ratio,
                )
                warnings.append(
                    "Image quality or skin-lesion relevance is uncertain. "
                    "For best results, please use a clear, close-up photo of "
                    "the skin area or lesion."
                )

        except ImportError:
            logger.warning(
                "OpenCV not available — skipping face-detection, edge and "
                "skin-tone checks"
            )

        logger.info(
            "Validation PASS: skin_ratio=%.3f  face_check=ok  "
            "edge=%.4f  std=%.2f",
            skin_ratio, edge_density, global_std,
        )

        if warnings:
            return {
                "is_valid": True,
                "confidence": 0.65,
                "warnings": warnings,
                "reason": None,
                "guidance": None,
            }

        return {
            "is_valid": True,
            "confidence": 0.95,
            "warnings": [],
            "reason": None,
            "guidance": None,
        }

    except Exception as exc:
        # Fail-open: validation crash should never block a valid upload.
        logger.warning(
            "Validation exception (%s: %s) — allowing prediction (fail-open)",
            type(exc).__name__,
            exc,
        )
        return {
            "is_valid": True,
            "confidence": 0.5,
            "warnings": [
                "Image validation encountered an unexpected error; "
                "proceeding with caution."
            ],
            "reason": None,
            "guidance": None,
        }


# ── Internal helpers ─────────────────────────────────────────────────────────

def _reject(
    code: str,
    reason: str,
    guidance: str,
    confidence: float,
) -> dict[str, Any]:
    logger.info("Validation REJECT [%s]: %s", code, reason)
    return {
        "is_valid": False,
        "confidence": confidence,
        "reason": reason,
        "guidance": guidance,
        "warnings": [],
    }
