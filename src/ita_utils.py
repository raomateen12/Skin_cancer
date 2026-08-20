"""
src/ita_utils.py
Reusable ITA (Individual Typology Angle) skin-tone estimation utilities.

Extracted from src/calibration_fairness.py so they can be consumed by:
  - src/calibration_fairness.py   (audit)
  - src/train.py                  (reweighted training)
  - Any future script needing per-image ITA labels

Public API
----------
compute_ita_for_image(image_path, patch=20)
    → (ita_value: float | None, mean_b_star: float | None)

ita_to_group(ita_value)
    → "light" | "intermediate" | "dark" | "unknown"

is_formula_unstable(mean_b_star, threshold=5.0)
    → bool  (|b*| < threshold  →  ITA denominator near-zero)

label_split(df, out_csv, split_name, patch=20, b_thresh=5.0)
    → pd.DataFrame   (df with added ita_value, ita_group, ita_formula_unstable)
    Saves result to out_csv and prints group counts.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ── ITA thresholds (Hall et al. 2022 / Chardon et al. 1991) ──────────────────
ITA_BINS = [("dark", None, 10), ("intermediate", 10, 41), ("light", 41, None)]
B_STAR_UNSTABLE_THRESHOLD = 5.0   # |b*| below this → formula degenerate
MIN_GROUP_WARN = 30


# ── Low-level Lab conversion (pure Python, no cv2 dependency) ─────────────────

def _linearise(c: float) -> float:
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _rgb_to_cielab(r: float, g: float, b: float) -> tuple[float, float, float]:
    """sRGB (0-255) → CIELab D65. Returns (L*, a*, b*)."""
    r_l, g_l, b_l = _linearise(r), _linearise(g), _linearise(b)

    X = r_l * 0.4124564 + g_l * 0.3575761 + b_l * 0.1804375
    Y = r_l * 0.2126729 + g_l * 0.7151522 + b_l * 0.0721750
    Z = r_l * 0.0193339 + g_l * 0.1191920 + b_l * 0.9503041

    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883

    def f(t: float) -> float:
        d = 6 / 29
        return t ** (1/3) if t > d**3 else t / (3 * d**2) + 4/29

    fx, fy, fz = f(X/Xn), f(Y/Yn), f(Z/Zn)
    return 116*fy - 16, 500*(fx - fy), 200*(fy - fz)


# ── Public API ────────────────────────────────────────────────────────────────

def compute_ita_for_image(
    image_input: str | Path | bytes | Image.Image,
    patch: int = 20,
) -> tuple[float | None, float | None]:
    """
    Estimate ITA = atan((L*-50)/b*) × (180/π) from four corner patches.

    Parameters
    ----------
    image_input : str | Path | bytes | PIL.Image.Image
        Image path, raw bytes, or PIL Image object.
    patch : int
        Corner patch dimension in pixels.

    Returns
    -------
    ita_value   : float | None   (None if image unreadable or |b*| < 1e-6)
    mean_b_star : float | None   (mean b* across 4 corners; None on failure)
    """
    try:
        import io
        if isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        elif isinstance(image_input, (bytes, bytearray)):
            img = Image.open(io.BytesIO(image_input)).convert("RGB")
        else:
            img = Image.open(image_input).convert("RGB")

        w, h = img.size
        px = max(1, min(patch, w // 4, h // 4))

        corners = [
            img.crop((0,    0,    px,   px)),   # top-left
            img.crop((w-px, 0,    w,    px)),   # top-right
            img.crop((0,    h-px, px,   h)),    # bottom-left
            img.crop((w-px, h-px, w,    h)),    # bottom-right
        ]

        all_pixels: list[list[float]] = []
        for c in corners:
            arr = np.array(c, dtype=np.float32).reshape(-1, 3)
            all_pixels.extend(arr.tolist())

        arr_all = np.array(all_pixels, dtype=np.float32)
        med_r = float(np.median(arr_all[:, 0]))
        med_g = float(np.median(arr_all[:, 1]))
        med_b = float(np.median(arr_all[:, 2]))

        L, _, b_lab = _rgb_to_cielab(med_r, med_g, med_b)

        if abs(b_lab) < 1e-6:
            return None, float(b_lab)

        ita = math.atan((L - 50) / b_lab) * (180 / math.pi)
        return float(ita), float(b_lab)

    except Exception:
        return None, None


def ita_to_group(ita_value: float | None) -> str:
    """Bin ITA value into skin-tone group label ('light', 'intermediate', 'dark', or 'unknown')."""
    if ita_value is None:
        return "unknown"
    if ita_value < 10:
        return "dark"
    if ita_value <= 41:
        return "intermediate"
    return "light"


compute_ita_group = ita_to_group


def is_formula_unstable(mean_b_star: float | None, threshold: float = B_STAR_UNSTABLE_THRESHOLD) -> bool:
    """True when |b*| is below threshold — ITA denominator degeneracy zone."""
    if mean_b_star is None:
        return True   # unknown images are always unreliable
    return abs(mean_b_star) < threshold


def label_split(
    df: pd.DataFrame,
    out_csv: Path,
    split_name: str = "split",
    patch: int = 20,
    b_thresh: float = B_STAR_UNSTABLE_THRESHOLD,
) -> pd.DataFrame:
    """
    Add ITA labels to a dataframe and save to CSV.

    Adds columns:
        ita_value             float | NaN
        ita_group             "light" | "intermediate" | "dark" | "unknown"
        ita_formula_unstable  bool
    """
    ita_vals, ita_grps, ita_unstable = [], [], []

    print(f"\n  Computing ITA for {split_name} split ({len(df)} images) …")
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  ITA [{split_name}]"):
        ita, b_star = compute_ita_for_image(str(row["image_path"]), patch=patch)
        grp     = ita_to_group(ita)
        unstable = is_formula_unstable(b_star, threshold=b_thresh)
        ita_vals.append(ita)
        ita_grps.append(grp)
        ita_unstable.append(unstable)

    df = df.copy()
    df["ita_value"]            = ita_vals
    df["ita_group"]            = ita_grps
    df["ita_formula_unstable"] = ita_unstable

    out_csv = Path(out_csv)
    df.to_csv(out_csv, index=False)
    print(f"  Saved → {out_csv}")

    counts = df["ita_group"].value_counts()
    n_unstable = int(df["ita_formula_unstable"].sum())
    print(f"\n  ITA group counts [{split_name}]:")
    for grp in ["light", "intermediate", "dark", "unknown"]:
        n = int(counts.get(grp, 0))
        warn = " ⚠ BELOW 30" if 0 < n < MIN_GROUP_WARN else ""
        print(f"    {grp:15s}: {n:>5d}{warn}")
    print(f"    formula-unstable (|b*|<{b_thresh}): {n_unstable} "
          f"({100*n_unstable/max(len(df),1):.1f}%)")

    return df
