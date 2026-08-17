"""
Calibration Fairness Audit — DermaLens AI
==========================================
Measures whether model calibration differs across estimated skin-tone groups
using the Individual Typology Angle (ITA) method.

This is a NO-TRAINING, inference-only script using the existing checkpoint.

Methodology:
  - Skin-tone proxy: ITA computed from corner-pixel sampling on dermoscopic images
  - Calibration metrics: ECE, MCE (15-bin), multiclass Brier score
  - Matched-sample-size comparison: 1000 bootstrap iterations (Ricci Lara et al. arXiv:2305.05101)

Usage:
    python -m src.calibration_fairness
    python -m src.calibration_fairness --model_name efficientnet_b0 --split test

Outputs (results/calibration_fairness/):
    calibration_by_ita_group.csv
    calibration_matched_sampling.csv
    reliability_diagram_<group>.png  (one per ITA group)
    calibration_fairness_report.txt
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset import get_eval_transforms, class_to_idx, idx_to_class
from src.model import get_efficientnet_b0, get_resnet50

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_TEMPLATE = "checkpoints/best_{model_name}.pth"
CLASS_MAPPING_PATH  = "data/processed/class_mapping.json"
OUT_DIR             = Path("results/calibration_fairness")
ITA_BINS            = [("dark", None, 10), ("intermediate", 10, 41), ("light", 41, None)]
N_CAL_BINS          = 15
N_BOOTSTRAP         = 1000
MIN_GROUP_WARN      = 30

ITA_LIMITATION = (
    "Skin-tone estimated via ITA on dermoscopic close-up images using corner-pixel "
    "sampling; this is a proxy estimate, not clinically verified Fitzpatrick typing, "
    "and may be less reliable than on clinical/macro photographs where more surrounding "
    "skin is visible."
)


# ── Step 1: ITA skin-tone estimation ─────────────────────────────────────────

def _rgb_to_cielab_single(r: float, g: float, b: float) -> tuple[float, float, float]:
    """
    Convert a single sRGB triplet (0–255) to CIELab D65.
    Implements the standard two-step: sRGB → XYZ → Lab.
    """
    # Normalise to [0, 1] and linearise
    def linearise(c: float) -> float:
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r_lin, g_lin, b_lin = linearise(r), linearise(g), linearise(b)

    # sRGB → XYZ (D65)
    X = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    Y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    Z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041

    # D65 illuminant reference white
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883

    def f(t: float) -> float:
        delta = 6 / 29
        return t ** (1 / 3) if t > delta ** 3 else t / (3 * delta ** 2) + 4 / 29

    fx, fy, fz = f(X / Xn), f(Y / Yn), f(Z / Zn)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_val = 200 * (fy - fz)
    return L, a, b_val


def estimate_ita(image_path: str, patch: int = 20) -> float | None:
    """
    Estimate ITA from the four corner patches of the image.

    ITA = atan((L* − 50) / b*) × (180/π)

    Returns None if the image cannot be loaded or b* ≈ 0.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        # Clamp patch size to image dimensions
        px = min(patch, w // 4, h // 4)
        if px < 1:
            px = 1

        corners = [
            img.crop((0, 0, px, px)),                  # top-left
            img.crop((w - px, 0, w, px)),              # top-right
            img.crop((0, h - px, px, h)),              # bottom-left
            img.crop((w - px, h - px, w, h)),          # bottom-right
        ]

        all_pixels: list[tuple[int, int, int]] = []
        for patch_img in corners:
            arr = np.array(patch_img, dtype=np.float32).reshape(-1, 3)
            all_pixels.extend(arr.tolist())

        arr_all = np.array(all_pixels, dtype=np.float32)
        med_r, med_g, med_b = float(np.median(arr_all[:, 0])), \
                               float(np.median(arr_all[:, 1])), \
                               float(np.median(arr_all[:, 2]))

        L, _, b_lab = _rgb_to_cielab_single(med_r, med_g, med_b)

        if abs(b_lab) < 1e-6:
            return None

        ita = math.atan((L - 50) / b_lab) * (180 / math.pi)
        return float(ita)

    except Exception:
        return None


def ita_to_group(ita: float | None) -> str:
    if ita is None:
        return "unknown"
    if ita < 10:
        return "dark"
    if ita <= 41:
        return "intermediate"
    return "light"


def compute_ita_for_test_split(
    df: pd.DataFrame,
    out_csv: Path,
    patch: int = 20,
) -> pd.DataFrame:
    """Compute ITA for every image in the dataframe and save extended CSV."""
    ita_values, ita_groups = [], []

    print("\n── Step 1: Estimating ITA skin-tone proxy ──────────────────────────────")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="ITA estimation"):
        ita = estimate_ita(str(row["image_path"]), patch=patch)
        ita_values.append(ita)
        ita_groups.append(ita_to_group(ita))

    df = df.copy()
    df["ita_value"] = ita_values
    df["ita_group"] = ita_groups

    df.to_csv(out_csv, index=False)
    print(f"  Extended CSV → {out_csv}")

    counts = df["ita_group"].value_counts()
    print("\n  ITA group counts:")
    for grp in ["light", "intermediate", "dark", "unknown"]:
        n = int(counts.get(grp, 0))
        flag = " ⚠ BELOW 30 (low statistical power)" if 0 < n < MIN_GROUP_WARN else ""
        print(f"    {grp:15s}: {n:>4d}{flag}")
    if counts.get("unknown", 0) > 0:
        print(f"  ⚠  {counts.get('unknown', 0)} images could not be estimated (ITA=None).")

    return df


# ── Step 2: Inference ─────────────────────────────────────────────────────────

class _SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(str(row["image_path"])).convert("RGB")
        img_np = np.array(img)
        if self.transform:
            img_np = self.transform(image=img_np)["image"]
        label = int(row["label_id"])
        return img_np, label


def run_inference(
    df: pd.DataFrame,
    model_name: str,
    checkpoint_path: str,
    num_classes: int,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        probs   — (N, num_classes) softmax probability matrix
        preds   — (N,) predicted class indices
        labels  — (N,) true class indices
    """
    print("\n── Step 2: Running inference ───────────────────────────────────────────")

    device = torch.device("cpu")

    if model_name == "efficientnet_b0":
        model = get_efficientnet_b0(num_classes)
    elif model_name == "resnet50":
        model = get_resnet50(num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"  Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    transform = get_eval_transforms(224)
    dataset = _SimpleDataset(df, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Inference"):
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return (
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_labels, axis=0),
    )


# ── Calibration metrics ───────────────────────────────────────────────────────

def ece_mce(
    confs: np.ndarray,
    corrects: np.ndarray,
    n_bins: int = N_CAL_BINS,
) -> tuple[float, float]:
    """
    ECE and MCE from confidence vector and binary correct vector.
    confs: predicted confidence (max prob), shape (N,)
    corrects: 1 if prediction correct else 0, shape (N,)
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, mce = 0.0, 0.0
    n = len(confs)

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs > lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        b_acc = corrects[mask].mean()
        b_conf = confs[mask].mean()
        b_n = mask.sum()
        gap = abs(b_acc - b_conf)
        ece += (b_n / n) * gap
        mce = max(mce, gap)

    return float(ece), float(mce)


def brier_score_multiclass(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """
    Multiclass Brier score: mean squared error between softmax vector and one-hot true label.
    Lower is better; 0 = perfect.
    """
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def calibration_metrics_for_group(
    probs: np.ndarray,
    preds: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    n_bins: int = N_CAL_BINS,
) -> dict:
    if len(labels) == 0:
        return {"ece": float("nan"), "mce": float("nan"), "brier": float("nan"),
                "accuracy": float("nan"), "n": 0}
    confs = probs.max(axis=1)
    corrects = (preds == labels).astype(float)
    ece, mce = ece_mce(confs, corrects, n_bins)
    brier = brier_score_multiclass(probs, labels, num_classes)
    accuracy = corrects.mean()
    return {"ece": ece, "mce": mce, "brier": brier, "accuracy": accuracy, "n": len(labels)}


# ── Step 3: Matched-sample-size bootstrap ────────────────────────────────────

def matched_bootstrap(
    group_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    num_classes: int,
    n_iter: int = N_BOOTSTRAP,
    rng_seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Matched-sample-size bootstrap (Ricci Lara et al. arXiv:2305.05101).

    FIX: Every group is resampled WITH REPLACEMENT so every group,
    including the one at N_min, gets a proper bootstrap confidence interval.
    Larger groups are also subsampled to N_min (with replacement) so
    comparisons remain size-matched.

    Returns:
        (summary_df, iter_records)
        summary_df   — DataFrame with mean ± std of each metric per group
        iter_records — dict mapping group name to list of per-iteration metric dicts
                       (needed for downstream significance testing)
    """
    print(f"\n── Step 3: Matched-sample bootstrap ({n_iter} iterations, with replacement) ─")
    rng = np.random.default_rng(rng_seed)

    groups = [g for g, (_, _, lbl) in group_data.items() if len(lbl) > 0]
    sizes = {g: len(group_data[g][2]) for g in groups}
    n_min = min(sizes.values())
    print(f"  Group sizes: { {g: sizes[g] for g in groups} }")
    print(f"  N_min (resample target, with replacement): {n_min}")

    records: dict[str, list[dict]] = {g: [] for g in groups}

    for _ in tqdm(range(n_iter), desc="  Bootstrap"):
        for g in groups:
            probs_g, preds_g, labels_g = group_data[g]
            # WITH REPLACEMENT for ALL groups — this gives every group,
            # including the smallest one (N_min), a proper bootstrap CI.
            idx = rng.choice(len(labels_g), size=n_min, replace=True)
            m = calibration_metrics_for_group(
                probs_g[idx], preds_g[idx], labels_g[idx], num_classes
            )
            records[g].append(m)

    rows = []
    for g in groups:
        df_iter = pd.DataFrame(records[g])
        row = {"group": g, "n_min": n_min}
        for col in ["ece", "mce", "brier", "accuracy"]:
            row[f"{col}_mean"] = df_iter[col].mean()
            row[f"{col}_std"] = df_iter[col].std()
        rows.append(row)
    return pd.DataFrame(rows), records


# ── Statistical significance testing ─────────────────────────────────────────

def compute_significance_tests(
    iter_records: dict[str, list[dict]],
    n_permutations: int = 10_000,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    For each group pair x metric (ECE, Brier) compute:
      - observed mean difference across bootstrap iterations
      - 95% CI of the difference distribution (2.5th–97.5th percentile)
      - two-sided permutation p-value (pooled reshuffling of the two
        1000-element bootstrap vectors)
      - significance flag: True when the 95% CI excludes zero

    Returns a DataFrame with columns:
        group_a, group_b, metric, observed_diff, ci_lower, ci_upper,
        p_value, significant
    """
    rng = np.random.default_rng(rng_seed + 1)
    groups = list(iter_records.keys())
    metrics = ["ece", "brier"]
    rows = []

    pairs = [
        (groups[i], groups[j])
        for i in range(len(groups))
        for j in range(i + 1, len(groups))
    ]

    for g_a, g_b in pairs:
        a_data = {m: np.array([r[m] for r in iter_records[g_a]], dtype=float) for m in metrics}
        b_data = {m: np.array([r[m] for r in iter_records[g_b]], dtype=float) for m in metrics}

        for metric in metrics:
            a = a_data[metric]
            b = b_data[metric]

            # Distribution of per-iteration differences
            diff_dist = a - b
            observed_diff = float(diff_dist.mean())
            ci_lower     = float(np.percentile(diff_dist, 2.5))
            ci_upper     = float(np.percentile(diff_dist, 97.5))

            # Permutation test: pool both 1000-element vectors, reshuffle
            pooled = np.concatenate([a, b])
            n_a    = len(a)
            abs_obs = abs(observed_diff)
            count_extreme = 0
            for _ in range(n_permutations):
                rng.shuffle(pooled)
                perm_diff = pooled[:n_a].mean() - pooled[n_a:].mean()
                if abs(perm_diff) >= abs_obs:
                    count_extreme += 1
            # +1 continuity correction (Phipson & Smyth 2010)
            p_value = (count_extreme + 1) / (n_permutations + 1)

            # Significant when 95% CI excludes zero
            significant = bool((ci_lower > 0) or (ci_upper < 0))

            rows.append({
                "group_a":      g_a,
                "group_b":      g_b,
                "metric":       metric,
                "observed_diff": round(observed_diff, 6),
                "ci_lower":     round(ci_lower, 6),
                "ci_upper":     round(ci_upper, 6),
                "p_value":      round(p_value, 6),
                "significant":  significant,
            })

    df_out = pd.DataFrame(rows)
    print(df_out.to_string(index=False))
    return df_out


# ── Confound check — class distribution ──────────────────────────────────────

def class_distribution_confound_check(
    df: pd.DataFrame,
    save_path: Path,
) -> dict:
    """
    Crosstab of dx (lesion class) x ita_group with chi-square test.
    Saves CSV with counts and within-group percentages, plus chi2 footer.
    Returns dict: {"chi2": float, "p_value": float, "dof": int}
    """
    from scipy.stats import chi2_contingency

    df_valid = df[df["ita_group"].isin(["light", "intermediate", "dark"])].copy()

    # Raw count crosstab (rows = dx class, columns = ita_group)
    ct_count = pd.crosstab(df_valid["dx"], df_valid["ita_group"])

    # Within-group percentages (column-normalised)
    ct_pct = ct_count.div(ct_count.sum(axis=0), axis=1) * 100

    # Build combined "N (P%)" display
    combined_rows = {}
    for dx_cls in ct_count.index:
        row = {}
        for grp in ct_count.columns:
            n = int(ct_count.loc[dx_cls, grp])
            p = ct_pct.loc[dx_cls, grp]
            row[grp] = f"{n} ({p:.1f}%)"
        row["row_total"] = str(int(ct_count.loc[dx_cls].sum()))
        combined_rows[dx_cls] = row

    combined_df = pd.DataFrame(combined_rows).T.reset_index()
    combined_df = combined_df.rename(columns={"index": "dx_class"})

    # Chi-square test on raw counts
    chi2_stat, p_val, dof, _ = chi2_contingency(ct_count.values)

    # Append chi-square footer rows
    footer_rows = [
        {"dx_class": "--- CHI-SQUARE TEST OF INDEPENDENCE ---",
         **{c: "" for c in ct_count.columns}, "row_total": ""},
        {"dx_class": "chi2_statistic",
         **{c: "" for c in ct_count.columns}, "row_total": f"{chi2_stat:.4f}"},
        {"dx_class": "p_value",
         **{c: "" for c in ct_count.columns}, "row_total": f"{p_val:.6f}"},
        {"dx_class": "degrees_of_freedom",
         **{c: "" for c in ct_count.columns}, "row_total": str(dof)},
    ]
    footer_df = pd.DataFrame(footer_rows)
    out_df = pd.concat([combined_df, footer_df], ignore_index=True)
    out_df.to_csv(save_path, index=False)
    print(f"  Class distribution CSV -> {save_path}")

    # Also print the plain crosstab for quick review
    print("\n  Count crosstab (dx vs ita_group):")
    print(ct_count.to_string())
    print(f"\n  Chi-square={chi2_stat:.4f}  p={p_val:.6f}  dof={dof}")

    return {"chi2": float(chi2_stat), "p_value": float(p_val), "dof": int(dof)}


# ── Class-stratified calibration analysis ───────────────────────────────────

def _class_stratified_report_lines(
    qualifying: list,
    skipped: list,
    report_parts: list,
    min_per_group: int,
) -> list[str]:
    """Build the plain-English report lines for the class-stratified section."""
    lines = [
        "",
        "── CLASS-STRATIFIED ANALYSIS ──────────────────────────────────────────────",
        f"  Threshold: >= {min_per_group} samples per ITA group within a class",
        "  (Checks whether Brier-score disparities persist WITHIN individual classes)",
        "",
        "  CLASSES SKIPPED (insufficient samples in at least one ITA group):",
    ]
    for dx_cls, counts, reason in skipped:
        lines.append(
            f"    {dx_cls:8s}: light={counts.get('light',0):3d},"
            f" intermediate={counts.get('intermediate',0):3d},"
            f" dark={counts.get('dark',0):3d}"
            f"  [below threshold: {reason}]"
        )

    if not qualifying:
        lines.append("  No class qualifies — cannot perform within-class analysis.")
        return lines

    lines += ["", "  QUALIFYING CLASSES:"]
    for dx_cls, gm, _, _ in report_parts:
        parts = []
        for g in ["light", "intermediate", "dark"]:
            if g in gm:
                parts.append(f"{g} n={gm[g]['n']}")
        lines.append(f"    {dx_cls:8s}: {', '.join(parts)}")

    lines += ["", "  WITHIN-CLASS BRIER SCORE COMPARISON:"]

    all_sig_results = []
    for dx_cls, group_metrics, sig_results, n_min_cls in report_parts:
        lines += [
            "",
            f"  [{dx_cls.upper()}]  (bootstrap N_min={n_min_cls},"
            f" with replacement, 1 000 iterations)",
        ]
        for g in ["light", "intermediate", "dark"]:
            if g in group_metrics:
                m = group_metrics[g]
                lines.append(
                    f"    {g:13s}: n={m['n']:3d}"
                    f"  acc={m['acc']:.4f}  Brier={m['brier']:.4f}"
                )
        lines.append(
            f"    {'Pair':30s} {'Diff':>8s}  {'95% CI':>22s}  {'Sig?':>5s}"
        )
        lines.append("    " + "-" * 70)
        for g_a, g_b, diff, ci_lo, ci_hi, sig in sig_results:
            pair_str = f"{g_a} vs {g_b}"
            ci_str   = f"[{ci_lo:.4f}, {ci_hi:.4f}]"
            sig_str  = "YES *" if sig else "no"
            lines.append(
                f"    {pair_str:30s} {diff:>8.4f}  {ci_str:>24s}  {sig_str:>5s}"
            )
            all_sig_results.append((dx_cls, g_a, g_b, diff, ci_lo, ci_hi, sig))

        # Plain-English interpretation per class
        dark_results = {
            (g_a, g_b): sig
            for (g_a, g_b, d, lo, hi, sig) in sig_results
            if "dark" in (g_a, g_b)
        }
        lines.append("")
        for (g_a, g_b), sig in dark_results.items():
            other = g_b if g_a == "dark" else g_a
            if sig:
                lines.append(
                    f"    => dark vs {other}: Brier gap IS significant within"
                    f" {dx_cls} (gap PERSISTS after class stratification)"
                )
            else:
                lines.append(
                    f"    => dark vs {other}: Brier gap is NOT significant"
                    f" within {dx_cls} (gap DISAPPEARS within this class)"
                )

    # Overall verdict
    dark_sig_pairs = [
        (cls, g_a, g_b)
        for (cls, g_a, g_b, d, lo, hi, sig) in all_sig_results
        if sig and ("dark" in (g_a, g_b))
    ]

    lines += ["", "  OVERALL VERDICT:"]
    if dark_sig_pairs:
        lines += [
            "  The Brier-score disparity involving the dark ITA group PERSISTS",
            "  within at least one qualifying lesion class even after controlling",
            "  for class composition. This means the disparity is only PARTIALLY",
            "  explained by class-distribution imbalance; a genuine model-level",
            "  performance difference across estimated skin-tone groups cannot",
            "  be ruled out. Further investigation is warranted.",
        ]
        for cls, g_a, g_b in dark_sig_pairs:
            lines.append(f"  ⚠  Significant within-class gap: {cls} ({g_a} vs {g_b})")
    else:
        lines += [
            "  The Brier-score disparity does NOT persist within any individual",
            "  qualifying lesion class. Within-class ITA group differences are",
            "  NOT statistically significant. This strongly suggests the disparity",
            "  observed overall is FULLY explained by class-distribution imbalance:",
            "  darker-skin images have proportionally more hard classes (mel, bkl)",
            "  and fewer easy classes (nv) than lighter-skin images in HAM10000.",
            "  There is no strong evidence of the model being genuinely worse for",
            "  darker estimated skin tones within individual lesion classes.",
        ]

    return lines


def class_stratified_analysis(
    df: pd.DataFrame,
    probs: np.ndarray,
    preds: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    n_bootstrap: int = N_BOOTSTRAP,
    out_dir: Path = OUT_DIR,
    min_per_group: int = 15,
    rng_seed: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Class-conditional Brier-score fairness analysis.

    For each dx class with >= min_per_group samples in every ITA group,
    compute per-group Brier score and run a matched-size with-replacement
    bootstrap (1000 iterations) to check whether pairwise Brier differences
    are statistically significant WITHIN that class.

    Uses the already-computed probs/preds/labels arrays — no re-inference.

    Returns (results_df, report_lines).
    """
    rng = np.random.default_rng(rng_seed + 2)
    groups = ["light", "intermediate", "dark"]
    dx_classes = sorted(df["dx"].unique())

    print(
        f"\n── Class-stratified analysis (min {min_per_group}/group) ─────────────────"
    )

    # Step 1: qualify classes
    qualifying, skipped = [], []
    for dx_cls in dx_classes:
        dx_mask = df["dx"].values == dx_cls
        counts = {
            g: int((dx_mask & (df["ita_group"].values == g)).sum())
            for g in groups
        }
        below = [g for g in groups if counts[g] < min_per_group]
        if not below:
            qualifying.append(dx_cls)
            print(f"  QUALIFY  {dx_cls:8s}: {counts}")
        else:
            reason = ", ".join(f"{g}={counts[g]}" for g in below)
            skipped.append((dx_cls, counts, reason))
            print(f"  SKIP     {dx_cls:8s}: {counts}  [below threshold: {reason}]")

    empty_cols = [
        "dx_class", "group_a", "group_b", "n_a", "n_b",
        "brier_diff", "ci_lower", "ci_upper", "significant",
    ]
    if not qualifying:
        print("  No qualifying classes.")
        empty_df = pd.DataFrame(columns=empty_cols)
        report_lines = _class_stratified_report_lines([], skipped, [], min_per_group)
        return empty_df, report_lines

    # Steps 2 & 3: per-class per-group metrics + bootstrap
    all_rows = []
    report_parts = []

    for dx_cls in qualifying:
        dx_mask = df["dx"].values == dx_cls
        group_data_cls, group_metrics_cls = {}, {}

        for g in groups:
            mask = dx_mask & (df["ita_group"].values == g)
            if mask.sum() == 0:
                continue
            p_g, pr_g, l_g = probs[mask], preds[mask], labels[mask]
            group_data_cls[g] = (p_g, pr_g, l_g)
            group_metrics_cls[g] = {
                "brier": brier_score_multiclass(p_g, l_g, num_classes),
                "acc":   float((pr_g == l_g).mean()),
                "n":     int(mask.sum()),
            }

        sizes_cls = {g: len(v[2]) for g, v in group_data_cls.items()}
        n_min_cls = min(sizes_cls.values())

        # Bootstrap
        iter_brier_cls = {g: [] for g in group_data_cls}
        for _ in range(n_bootstrap):
            for g, (p_g, pr_g, l_g) in group_data_cls.items():
                idx = rng.choice(len(l_g), size=n_min_cls, replace=True)
                iter_brier_cls[g].append(
                    brier_score_multiclass(p_g[idx], l_g[idx], num_classes)
                )

        # Pairwise comparison
        g_list = list(group_data_cls.keys())
        pairs_cls = [
            (g_list[i], g_list[j])
            for i in range(len(g_list))
            for j in range(i + 1, len(g_list))
        ]
        sig_results_cls = []
        for g_a, g_b in pairs_cls:
            a = np.array(iter_brier_cls[g_a])
            b = np.array(iter_brier_cls[g_b])
            diff_dist    = a - b
            observed_diff = float(diff_dist.mean())
            ci_lower     = float(np.percentile(diff_dist, 2.5))
            ci_upper     = float(np.percentile(diff_dist, 97.5))
            significant  = bool((ci_lower > 0) or (ci_upper < 0))

            all_rows.append({
                "dx_class":   dx_cls,
                "group_a":    g_a,
                "group_b":    g_b,
                "n_a":        group_metrics_cls[g_a]["n"],
                "n_b":        group_metrics_cls[g_b]["n"],
                "brier_diff": round(observed_diff, 6),
                "ci_lower":   round(ci_lower, 6),
                "ci_upper":   round(ci_upper, 6),
                "significant": significant,
            })
            sig_results_cls.append((g_a, g_b, observed_diff, ci_lower, ci_upper, significant))

        report_parts.append((dx_cls, group_metrics_cls, sig_results_cls, n_min_cls))

    results_df = pd.DataFrame(all_rows)
    save_path  = out_dir / "class_stratified_analysis.csv"
    results_df.to_csv(save_path, index=False)
    print(f"  Class-stratified CSV -> {save_path}")
    print(results_df.to_string(index=False))

    report_lines = _class_stratified_report_lines(
        qualifying, skipped, report_parts, min_per_group
    )
    return results_df, report_lines


# ── Step 4: Reliability diagram ───────────────────────────────────────────────

def plot_reliability_diagram(
    probs: np.ndarray,
    preds: np.ndarray,
    labels: np.ndarray,
    group_name: str,
    save_path: Path,
    n_bins: int = N_CAL_BINS,
) -> None:
    confs = probs.max(axis=1)
    corrects = (preds == labels).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_mid = (bins[:-1] + bins[1:]) / 2
    bin_accs, bin_confs, bin_ns = [], [], []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs > lo) & (confs <= hi)
        if mask.sum() == 0:
            bin_accs.append(float("nan"))
            bin_confs.append((lo + hi) / 2)
            bin_ns.append(0)
        else:
            bin_accs.append(corrects[mask].mean())
            bin_confs.append(confs[mask].mean())
            bin_ns.append(int(mask.sum()))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")

    valid = [i for i, a in enumerate(bin_accs) if not math.isnan(a)]
    if valid:
        x_vals = [bin_confs[i] for i in valid]
        y_vals = [bin_accs[i] for i in valid]
        ax.bar(
            [bin_confs[i] for i in valid],
            [bin_accs[i] for i in valid],
            width=1 / n_bins,
            alpha=0.6,
            align="center",
            color="steelblue",
            label="Fraction correct",
        )
        ax.plot(x_vals, y_vals, "o-", color="steelblue", ms=5)

    ece, _ = ece_mce(confs, corrects, n_bins)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Fraction correct")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Reliability Diagram — ITA group: {group_name}\n(n={len(labels)}, ECE={ece:.4f})")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(
    path: Path,
    model_name: str,
    split: str,
    overall: dict,
    raw_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    group_counts: dict[str, int],
    sig_df: "pd.DataFrame | None" = None,
    chi2_result: "dict | None" = None,
    class_strat_lines: "list | None" = None,
) -> None:
    lines = [
        "=" * 72,
        "  DermaLens AI — Calibration Fairness Report",
        f"  Model   : {model_name}",
        f"  Split   : {split}",
        "=" * 72,
        "",
        "── OVERALL METRICS ─────────────────────────────────────────────────────",
        f"  Accuracy : {overall['accuracy']:.4f}",
        f"  ECE      : {overall['ece']:.4f}",
        f"  MCE      : {overall['mce']:.4f}",
        f"  Brier    : {overall['brier']:.4f}",
        f"  N total  : {overall['n']}",
        "",
        "── ITA GROUP SAMPLE COUNTS ─────────────────────────────────────────────",
    ]
    for g, n in group_counts.items():
        flag = " ⚠ BELOW 30 — low statistical power" if 0 < n < MIN_GROUP_WARN else ""
        lines.append(f"  {g:15s}: {n:>4d}{flag}")

    lines += [
        "",
        "── RAW (UNMATCHED) CALIBRATION PER ITA GROUP ───────────────────────────",
        f"  {'Group':15s} {'N':>5s} {'Accuracy':>10s} {'ECE':>8s} {'MCE':>8s} {'Brier':>8s}",
        "  " + "-" * 60,
    ]
    for _, row in raw_df.iterrows():
        lines.append(
            f"  {str(row['group']):15s} {int(row['n']):>5d}"
            f" {row['accuracy']:>10.4f} {row['ece']:>8.4f}"
            f" {row['mce']:>8.4f} {row['brier']:>8.4f}"
        )

    lines += [
        "",
        "── MATCHED-SIZE CALIBRATION (mean ± std across 1 000 bootstrap runs) ───",
        f"  {'Group':15s} {'N_min':>5s} {'ECE':>20s} {'MCE':>20s} {'Brier':>20s}",
        "  " + "-" * 82,
    ]
    for _, row in matched_df.iterrows():
        lines.append(
            f"  {str(row['group']):15s} {int(row['n_min']):>5d}"
            f" {row['ece_mean']:.4f} ± {row['ece_std']:.4f}   "
            f"{row['mce_mean']:.4f} ± {row['mce_std']:.4f}   "
            f"{row['brier_mean']:.4f} ± {row['brier_std']:.4f}"
        )

    # Flag significant ECE mean gaps (legacy flag, kept for compatibility)
    lines += ["", "── CALIBRATION DISPARITY FLAGS (matched ECE mean gap) ───────────────────"]
    ece_means = {
        str(row["group"]): row["ece_mean"]
        for _, row in matched_df.iterrows()
        if not math.isnan(row["ece_mean"])
    }
    pairs_flag = [(g1, g2) for i, g1 in enumerate(ece_means) for g2 in list(ece_means)[i+1:]]
    any_flag = False
    for g1, g2 in pairs_flag:
        gap = abs(ece_means[g1] - ece_means[g2])
        if gap > 0.05:
            lines.append(
                f"  ⚠  ECE gap between '{g1}' and '{g2}': {gap:.4f} > 0.05 threshold"
            )
            any_flag = True
    if not any_flag:
        lines.append("  No matched-ECE mean gap between any two groups exceeds 0.05.")

    # ── Statistical Significance section ─────────────────────────────────────
    if sig_df is not None and not sig_df.empty:
        lines += [
            "",
            "── STATISTICAL SIGNIFICANCE (ECE and Brier, matched bootstrap) ─────────",
            f"  {'Pair':35s} {'Metric':6s} {'Diff':>8s}  {'95% CI':>22s}  {'p-value':>8s}  {'Sig?':>5s}",
            "  " + "-" * 90,
        ]
        for _, r in sig_df.iterrows():
            pair_str = f"{r['group_a']} vs {r['group_b']}"
            ci_str   = f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"
            sig_str  = "YES *" if r["significant"] else "no"
            lines.append(
                f"  {pair_str:35s} {str(r['metric']):6s} {r['observed_diff']:>8.4f}"
                f"  {ci_str:>24s}  {r['p_value']:>8.4f}  {sig_str:>5s}"
            )
        sig_pairs = sig_df[sig_df["significant"] == True]
        lines.append("")
        if not sig_pairs.empty:
            lines.append("  * = 95% bootstrap CI of the difference excludes zero")
            for _, r in sig_pairs.iterrows():
                lines.append(
                    f"  ⚠  {r['group_a']} vs {r['group_b']}"
                    f" ({str(r['metric']).upper()}):"
                    f" diff={r['observed_diff']:.4f},"
                    f" CI=[{r['ci_lower']:.4f},{r['ci_upper']:.4f}],"
                    f" p={r['p_value']:.4f}"
                )
        else:
            lines.append(
                "  No group-pair difference is statistically significant"
                " (all 95% CIs include zero)."
            )

    # ── Confound Check section ────────────────────────────────────────────────
    if chi2_result is not None:
        chi2  = chi2_result["chi2"]
        p_chi = chi2_result["p_value"]
        dof   = chi2_result["dof"]
        lines += [
            "",
            "── CONFOUND CHECK — CLASS DISTRIBUTION ACROSS ITA GROUPS ──────────────",
            f"  Chi-square statistic : {chi2:.4f}",
            f"  p-value              : {p_chi:.6f}",
            f"  Degrees of freedom   : {dof}",
            "",
        ]
        if p_chi < 0.05:
            lines += [
                "  INTERPRETATION: Lesion-class distribution differs SIGNIFICANTLY across",
                "  ITA skin-tone groups (p < 0.05). This is a strong confound: different",
                "  lesion types are not equally represented across estimated skin-tone",
                "  groups in HAM10000. The observed accuracy and Brier-score gaps between",
                "  ITA groups are therefore at least partly explained by this class-",
                "  distribution imbalance rather than the model being differentially",
                "  calibrated across skin tones. Calibration-gap conclusions should be",
                "  interpreted with caution until class-stratified analyses are done.",
            ]
        else:
            lines += [
                "  INTERPRETATION: Lesion-class distribution does NOT differ significantly",
                "  across ITA skin-tone groups (p >= 0.05). Class-distribution confound",
                "  is unlikely to fully explain observed calibration disparities.",
            ]

    # ── Class-Stratified Analysis section ───────────────────────────────────
    if class_strat_lines:
        lines.extend(class_strat_lines)

    lines += [
        "",
        "── LIMITATION ──────────────────────────────────────────────────────────",
        f"  {ITA_LIMITATION}",
        "",
        "── METHODOLOGY REFERENCE ───────────────────────────────────────────────",
        "  Matched-sample-size bootstrap follows: Ricci Lara et al.,",
        "  'Addressing Fairness in Artificial Intelligence for Medical Imaging',",
        "  arXiv:2305.05101 (2023).",
        "",
        "  ITA binning (light/intermediate/dark) follows Hall et al. (2022) and",
        "  Chardon et al. (1991): ITA > 41 = light, 10–41 = intermediate, <10 = dark.",
        "=" * 72,
    ]

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n  Report → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibration fairness audit using ITA-estimated skin-tone groups."
    )
    parser.add_argument("--model_name", type=str, default="efficientnet_b0",
                        choices=["efficientnet_b0", "resnet50"])
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=20,
                        help="Corner patch size in pixels for ITA estimation")
    parser.add_argument("--n_bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--skip_ita", action="store_true",
                        help="Reuse existing test_with_ita.csv if already computed")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load class mapping
    num_classes = len(class_to_idx)
    checkpoint_path = CHECKPOINT_TEMPLATE.format(model_name=args.model_name)
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    csv_path = Path(f"data/processed/{args.split}.csv")
    ita_csv  = Path("data/processed/test_with_ita.csv")

    df_raw = pd.read_csv(csv_path)

    # ── Step 1: ITA estimation ────────────────────────────────────────────────
    if args.skip_ita and ita_csv.exists():
        print(f"\n  Reusing existing ITA CSV: {ita_csv}")
        df = pd.read_csv(ita_csv)
        counts = df["ita_group"].value_counts()
        print("  ITA group counts:")
        for grp in ["light", "intermediate", "dark", "unknown"]:
            n = int(counts.get(grp, 0))
            flag = " ⚠ BELOW 30" if 0 < n < MIN_GROUP_WARN else ""
            print(f"    {grp:15s}: {n:>4d}{flag}")
    else:
        df = compute_ita_for_test_split(df_raw, ita_csv, patch=args.patch_size)

    # ── Step 2: Inference ─────────────────────────────────────────────────────
    probs, preds, labels = run_inference(
        df, args.model_name, checkpoint_path, num_classes, args.batch_size
    )

    # ── Overall metrics ───────────────────────────────────────────────────────
    overall = calibration_metrics_for_group(probs, preds, labels, num_classes)
    print(f"\n  Overall — accuracy={overall['accuracy']:.4f}  "
          f"ECE={overall['ece']:.4f}  MCE={overall['mce']:.4f}  "
          f"Brier={overall['brier']:.4f}")

    # ── Per-group raw metrics ─────────────────────────────────────────────────
    print("\n── Per-group (unmatched) calibration ──────────────────────────────────")
    raw_rows = []
    group_data: dict[str, tuple] = {}
    group_counts: dict[str, int] = {}

    ita_groups_present = [g for g in ["light", "intermediate", "dark"]
                          if (df["ita_group"] == g).sum() > 0]

    for grp in ita_groups_present:
        mask = df["ita_group"].values == grp
        probs_g = probs[mask]
        preds_g = preds[mask]
        labels_g = labels[mask]

        m = calibration_metrics_for_group(probs_g, preds_g, labels_g, num_classes)
        raw_rows.append({"group": grp, **m})
        group_data[grp] = (probs_g, preds_g, labels_g)
        group_counts[grp] = int(mask.sum())

        print(f"  {grp:15s} n={m['n']:>4d}  acc={m['accuracy']:.4f}  "
              f"ECE={m['ece']:.4f}  MCE={m['mce']:.4f}  Brier={m['brier']:.4f}")

    # unknown group (count-only, not used in calibration)
    unk_n = int((df["ita_group"] == "unknown").sum())
    if unk_n > 0:
        group_counts["unknown"] = unk_n
        print(f"  {'unknown':15s} n={unk_n:>4d}  (skipped — ITA estimation failed)")

    raw_df = pd.DataFrame(raw_rows)
    raw_path = OUT_DIR / "calibration_by_ita_group.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"\n  Raw calibration CSV → {raw_path}")

    # ── Step 3: Matched-size bootstrap (with replacement, all groups) ────────
    matched_df, iter_records = matched_bootstrap(
        group_data, num_classes, n_iter=args.n_bootstrap
    )
    matched_path = OUT_DIR / "calibration_matched_sampling.csv"
    matched_df.to_csv(matched_path, index=False)
    print(f"  Matched calibration CSV -> {matched_path}")

    # ── Step 3b: Statistical significance tests (10 000 permutations) ────────
    print("\n── Computing significance tests ──────────────────────────────────────────")
    sig_df = compute_significance_tests(iter_records, n_permutations=10_000)
    sig_path = OUT_DIR / "significance_tests.csv"
    sig_df.to_csv(sig_path, index=False)
    print(f"  Significance tests CSV -> {sig_path}")

    # ── Step 3c: Confound check — class distribution by ITA group ────────────
    print("\n── Confound check: dx class distribution by ITA group ──────────────────")
    chi2_result = class_distribution_confound_check(
        df, OUT_DIR / "class_distribution_by_ita_group.csv"
    )

    # ── Step 3d: Class-stratified analysis ─────────────────────────────────
    _, class_strat_lines = class_stratified_analysis(
        df, probs, preds, labels, num_classes,
        n_bootstrap=args.n_bootstrap, out_dir=OUT_DIR,
    )

    # ── Step 4: Reliability diagrams ──────────────────────────────────────────
    print("\n── Generating reliability diagrams ─────────────────────────────────────")
    for grp, (probs_g, preds_g, labels_g) in group_data.items():
        fig_path = OUT_DIR / f"reliability_diagram_{grp}.png"
        plot_reliability_diagram(probs_g, preds_g, labels_g, grp, fig_path)
        print(f"  {grp}: {fig_path}")

    # ── Report ────────────────────────────────────────────────────────────────
    report_path = OUT_DIR / "calibration_fairness_report.txt"
    write_report(
        report_path, args.model_name, args.split,
        overall, raw_df, matched_df, group_counts,
        sig_df=sig_df, chi2_result=chi2_result,
        class_strat_lines=class_strat_lines,
    )

    # Print report to stdout too
    print("\n" + "=" * 72)
    print("  FINAL REPORT")
    print("=" * 72)
    with open(report_path, encoding="utf-8") as fh:
        print(fh.read())


if __name__ == "__main__":
    main()
