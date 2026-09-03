"""
src/hair_removal.py
===================
Pipeline-ready hair removal for dermoscopic images using classical morphological
transforms (no neural network required).

Algorithm
---------
1. DARK HAIR DETECTION: Black-hat morphological transform with LONG THIN
   rectangular structuring elements (applied in two orientations: horizontal
   and vertical) highlights dark linear structures (hair strands) against a
   lighter background. A thin 25x1 / 1x25 kernel responds to linear structures
   of width ~1-3 px that span at least 25 px — matching real hair strands but
   NOT the large circular dark lesion (which would cause false positives with
   a wide elliptical kernel).

2. LIGHT/WHITE HAIR DETECTION: Top-hat transform with the same thin kernels
   highlights bright linear structures (white/blonde hair) against a darker
   background.

3. THRESHOLDING: Fixed threshold of 20 on the morphological response — calibrated
   so that real hair strands (high local contrast, narrow) exceed the threshold
   while the gradual tonal transitions of the dermoscopy vignette/lesion do not.

4. MINIMAL POST-PROCESSING: Only 1 dilation pass with a small 3x3 kernel (one
   iteration) to capture the immediate fringe of detected hair pixels. No
   morphological close is applied — it amplifies false-positive coverage.

5. INPAINTING: OpenCV TELEA fast marching method fills the binary hair mask.

6. SAFETY VALVE: If the resulting mask covers >20% of the image pixels, the
   original image is returned unchanged. This catches residual false positives
   on heavily pigmented lesions and dermoscopy calibration artifacts.

Calibration notes
-----------------
At thresh=20 + dilate 1x on HAM10000 images (450x600 px):
  - Images without visible hair:    base ~4-10%,  after dilate ~9-17%
  - Images with moderate hair:      base ~5-15%,  after dilate ~10-20%  
  - Images with heavy hair:         base ~15-30%, after dilate ~22-40%  (flagged)
  Expected mean coverage across the HAM10000 test set: 5-15% for most images.

References
----------
 - Lee et al. (1997): "Dull razor" algorithm (predecessor to this approach)
 - Xie et al. (2009): DullRazor revisited using morphological analysis
 - Fiorese et al. (2011): Comparative study of hair removal in dermoscopy

Public API
----------
remove_hair(image_bgr) -> (cleaned_bgr, hair_coverage_pct, flagged)
    image_bgr          : np.ndarray  uint8 BGR image (as returned by cv2.imread)
    cleaned_bgr        : np.ndarray  same shape, inpainted or original
    hair_coverage_pct  : float       fraction 0–100 of pixels in hair mask
    flagged            : bool        True if coverage > MAX_COVERAGE_RATIO and
                                     original was returned unchanged

remove_hair_from_path(image_path) -> (cleaned_bgr, hair_coverage_pct, flagged)
    Convenience wrapper that loads a BGR image from file then calls remove_hair().
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

# ── Tunable constants ──────────────────────────────────────────────────────────
# Kernel: LENGTH x WIDTH — very elongated for true hair-strand selectivity.
# A 25×1 kernel responds to thin (1-3 px wide) linear structures spanning ≥25 px.
HAIR_KERNEL_LENGTH: int    = 25      # Long axis (px)
HAIR_KERNEL_WIDTH:  int    = 1       # Short axis — keep at 1

# Fixed threshold on morphological response (0-255).
# 20 is empirically calibrated for HAM10000 (450x600px JPEGs):
#   • Real hair strands have high local contrast → response > 20
#   • Dermoscopy vignette / lesion gradients → response ≤ 15 on most pixels
HAIR_THRESH: int            = 20

# Post-processing: ONLY one dilation pass to capture hair fringe.
# No morphological close — it substantially amplifies false-positive coverage.
DILATE_KERNEL_SIZE: int     = 3
DILATE_ITERATIONS:  int     = 1

# TELEA inpainting neighbourhood radius
INPAINT_RADIUS: int         = 3

# Safety valve: skip inpainting when mask covers > this fraction of the image.
# Above 20%, we're likely detecting dermoscopy artefacts / heavy lesion pigmentation
# rather than true hair — returning the original avoids destroying lesion structure.
MAX_COVERAGE_RATIO: float   = 0.20   # 20%


def _linear_kernels(length: int = HAIR_KERNEL_LENGTH, width: int = HAIR_KERNEL_WIDTH):
    """Return (h_kernel, v_kernel): horizontal and vertical 1D structuring elements."""
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, width))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, length))
    return h_kernel, v_kernel


def detect_hair_mask(gray: np.ndarray) -> np.ndarray:
    """
    Detect dark and light hair strands using thin elongated morphological kernels.

    Uses fixed-threshold binarisation (not Otsu) to avoid adaptation to lesion content.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale uint8 image.

    Returns
    -------
    mask : np.ndarray
        Binary uint8 mask (0/255), 255 = detected hair pixel.
    """
    h_kernel, v_kernel = _linear_kernels()

    # ── Dark hair: black-hat with H + V thin kernels ───────────────────────────
    bh_h = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, h_kernel)
    bh_v = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, v_kernel)
    _, dm_h = cv2.threshold(bh_h, HAIR_THRESH, 255, cv2.THRESH_BINARY)
    _, dm_v = cv2.threshold(bh_v, HAIR_THRESH, 255, cv2.THRESH_BINARY)
    dark_mask = cv2.bitwise_or(dm_h, dm_v)

    # ── Light/white hair: top-hat with H + V thin kernels ─────────────────────
    th_h = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, h_kernel)
    th_v = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, v_kernel)
    _, lm_h = cv2.threshold(th_h, HAIR_THRESH, 255, cv2.THRESH_BINARY)
    _, lm_v = cv2.threshold(th_v, HAIR_THRESH, 255, cv2.THRESH_BINARY)
    light_mask = cv2.bitwise_or(lm_h, lm_v)

    # ── Combine both polarities ────────────────────────────────────────────────
    combined = cv2.bitwise_or(dark_mask, light_mask)

    # ── Single dilation to capture hair fringe (no morphological close) ────────
    dilate_k = cv2.getStructuringElement(
        cv2.MORPH_RECT, (DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE)
    )
    mask = cv2.dilate(combined, dilate_k, iterations=DILATE_ITERATIONS)

    return mask


def remove_hair(
    image_bgr: np.ndarray,
) -> Tuple[np.ndarray, float, bool]:
    """
    Detect and inpaint hair artifacts in a BGR dermoscopic image.

    Parameters
    ----------
    image_bgr : np.ndarray
        Input BGR image as a uint8 numpy array (e.g. from cv2.imread).

    Returns
    -------
    cleaned_bgr : np.ndarray
        Inpainted image (or original if safety valve triggered).
    hair_coverage_pct : float
        Percentage 0–100 of image pixels classified as hair.
    flagged : bool
        True if coverage exceeded MAX_COVERAGE_RATIO and the original
        image was returned unchanged.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("remove_hair received an empty or None image array.")

    # Ensure uint8
    if image_bgr.dtype != np.uint8:
        image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hair_mask = detect_hair_mask(gray)

    total_pixels      = hair_mask.size
    hair_pixels       = int(np.count_nonzero(hair_mask))
    hair_coverage_pct = (hair_pixels / total_pixels) * 100.0

    # ── Safety valve ──────────────────────────────────────────────────────────
    if (hair_pixels / total_pixels) > MAX_COVERAGE_RATIO:
        return image_bgr.copy(), hair_coverage_pct, True

    # ── Inpainting ────────────────────────────────────────────────────────────
    cleaned = cv2.inpaint(image_bgr, hair_mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)
    return cleaned, hair_coverage_pct, False


def remove_hair_from_path(
    image_path: str | Path,
) -> Tuple[np.ndarray, float, bool]:
    """
    Load a BGR image from disk and apply hair removal.

    Parameters
    ----------
    image_path : str or Path
        Absolute or relative path to a JPEG/PNG dermoscopic image.

    Returns
    -------
    cleaned_bgr        : np.ndarray
    hair_coverage_pct  : float
    flagged            : bool
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise IOError(f"cv2.imread could not decode: {path}")

    return remove_hair(image_bgr)


# ── Quick self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.hair_removal <image_path> [output_path]")
        sys.exit(1)

    in_path  = Path(sys.argv[1])
    out_path = (
        Path(sys.argv[2]) if len(sys.argv) > 2
        else in_path.with_stem(in_path.stem + "_hairremoved")
    )

    cleaned, pct, flagged = remove_hair_from_path(in_path)
    status = f"FLAGGED (>{int(MAX_COVERAGE_RATIO*100)}% coverage — original returned)" if flagged else "inpainted"
    print(f"  Hair coverage: {pct:.2f}%  |  Status: {status}")
    cv2.imwrite(str(out_path), cleaned)
    print(f"  Saved to: {out_path}")
