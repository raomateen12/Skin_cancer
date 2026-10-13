"""
src/gap_significance_test.py
============================
Paired Bootstrap Significance Test for Brier Score Gap Reduction
following Hair Removal Preprocessing on HAM10000.

Methodology
-----------
1. Loads per-image multiclass Brier scores for both conditions:
   - Baseline (no hair removal)
   - Hair-removed preprocessing
   If per-image scores are not already saved, computes them using the
   EfficientNet-B0 model checkpoint and saves them to:
     - results/calibration_fairness/per_image_brier.csv
     - results/calibration_fairness_hairremoved/per_image_brier.csv

2. For comparison (e.g. dark vs light):
   - gap_baseline = mean(brier_dark_baseline) - mean(brier_light_baseline)
   - gap_hairremoved = mean(brier_dark_hairremoved) - mean(brier_light_hairremoved)
   - observed_reduction = gap_baseline - gap_hairremoved

3. Paired Bootstrap Test (1000 iterations):
   - Resamples paired (baseline, hairremoved) scores with replacement,
     stratified by ITA group so group sample sizes remain identical.
   - Recomputes gap_baseline and gap_hairremoved on each resampled set.
   - Computes 95% CI (2.5th and 97.5th percentiles of bootstrap reductions).
   - Computes one-sided p-value: fraction of bootstrap iterations where
     gap_reduction <= 0 (null hypothesis: hair removal does not reduce gap).

4. Outputs:
   - results/calibration_fairness_hairremoved/gap_significance_test.csv
   - Plain-English verdict printed to stdout.

Usage
-----
    python -m src.gap_significance_test
    python -m src.gap_significance_test --n_bootstrap 1000 --seed 42
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.dataset import get_eval_transforms
from src.hair_removal import remove_hair
from src.model import get_efficientnet_b0

# ── Paths and Config ──────────────────────────────────────────────────────────
BASELINE_DIR = Path("results/calibration_fairness")
HAIRREMOVED_DIR = Path("results/calibration_fairness_hairremoved")
TEST_CSV_PATH = Path("data/processed/test_with_ita.csv")
CHECKPOINT_PATH = Path("checkpoints/best_efficientnet_b0.pth")
NUM_CLASSES = 7
DEFAULT_BOOTSTRAP = 1000
DEFAULT_SEED = 42


# ── Datasets for inference fallback ───────────────────────────────────────────
class _BaselineDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = str(row["image_path"])
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        img_t = self.transform(image=img_np)["image"]
        return img_t, int(row["label_id"])


class _HairRemovedDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = str(row["image_path"])
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            img_bgr = np.zeros((224, 224, 3), dtype=np.uint8)
        cleaned_bgr, _, _ = remove_hair(img_bgr)
        cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)
        img_t = self.transform(image=cleaned_rgb)["image"]
        return img_t, int(row["label_id"])


def compute_per_image_brier(
    df: pd.DataFrame,
    condition: str = "baseline",
    checkpoint_path: Path = CHECKPOINT_PATH,
    device: torch.device | None = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Runs model inference on the test split and computes per-image multiclass Brier score.
    condition: 'baseline' or 'hairremoved'
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_efficientnet_b0(NUM_CLASSES)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.to(device)
    model.eval()

    transform = get_eval_transforms(224)
    if condition == "baseline":
        dataset = _BaselineDataset(df, transform)
        desc = "  Inference (Baseline)"
    else:
        dataset = _HairRemovedDataset(df, transform)
        desc = "  Inference (Hair-Removed)"

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc=desc, leave=False):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(lbls.numpy())

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    brier_scores = np.sum((probs - one_hot) ** 2, axis=1)

    out_df = df[["image_id", "dx", "label_id", "ita_group"]].copy()
    col_name = f"brier_{condition}"
    out_df[col_name] = brier_scores
    return out_df


def load_or_compute_brier_scores(
    test_csv_path: Path = TEST_CSV_PATH,
) -> pd.DataFrame:
    """
    Loads baseline and hair-removed per-image Brier scores from CSV if available,
    or computes and saves them. Returns merged DataFrame with both conditions.
    """
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    HAIRREMOVED_DIR.mkdir(parents=True, exist_ok=True)

    base_brier_csv = BASELINE_DIR / "per_image_brier.csv"
    hr_brier_csv = HAIRREMOVED_DIR / "per_image_brier.csv"

    # 1. Baseline
    if base_brier_csv.exists():
        print(f"[*] Loading baseline per-image Brier scores from {base_brier_csv}")
        df_base = pd.read_csv(base_brier_csv)
    else:
        print(f"[*] Baseline per-image Brier scores not found. Computing now...")
        df_test = pd.read_csv(test_csv_path)
        from src.calibration_fairness_hairremoved import validate_and_remap_image_paths
        df_test = validate_and_remap_image_paths(df_test)
        df_base = compute_per_image_brier(df_test, condition="baseline")
        df_base.to_csv(base_brier_csv, index=False)
        print(f"[+] Saved baseline per-image Brier scores to {base_brier_csv}")

    # 2. Hair-removed
    if hr_brier_csv.exists():
        print(f"[*] Loading hair-removed per-image Brier scores from {hr_brier_csv}")
        df_hr = pd.read_csv(hr_brier_csv)
    else:
        print(f"[*] Hair-removed per-image Brier scores not found. Computing now...")
        df_test = pd.read_csv(test_csv_path)
        from src.calibration_fairness_hairremoved import validate_and_remap_image_paths
        df_test = validate_and_remap_image_paths(df_test)
        df_hr = compute_per_image_brier(df_test, condition="hairremoved")
        df_hr.to_csv(hr_brier_csv, index=False)
        print(f"[+] Saved hair-removed per-image Brier scores to {hr_brier_csv}")

    # Merge paired results
    merged = pd.merge(
        df_base[["image_id", "dx", "label_id", "ita_group", "brier_baseline"]],
        df_hr[["image_id", "brier_hairremoved"]],
        on="image_id",
        how="inner",
    )

    # Filter out unknown ITA group
    valid = merged[merged["ita_group"].isin(["light", "intermediate", "dark"])].copy()
    valid = valid.reset_index(drop=True)
    print(f"[*] Paired data loaded: {len(valid)} images across light, intermediate, dark groups.")
    return valid


# ── Paired Bootstrap Test ─────────────────────────────────────────────────────
def run_paired_bootstrap_test(
    df: pd.DataFrame,
    group_a: str = "dark",
    group_b: str = "light",
    n_bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> Tuple[float, float, float, float, bool]:
    """
    Performs a paired bootstrap test of the difference in Brier gaps:
      gap = mean(brier_A) - mean(brier_B)
      reduction = gap_baseline - gap_hairremoved

    Returns:
      (observed_reduction, ci_lower, ci_upper, p_value, significant)
    """
    rng = np.random.default_rng(seed)

    df_a = df[df["ita_group"] == group_a].reset_index(drop=True)
    df_b = df[df["ita_group"] == group_b].reset_index(drop=True)

    n_a = len(df_a)
    n_b = len(df_b)

    # Observed point estimates
    gap_baseline = df_a["brier_baseline"].mean() - df_b["brier_baseline"].mean()
    gap_hairremoved = df_a["brier_hairremoved"].mean() - df_b["brier_hairremoved"].mean()
    observed_reduction = float(gap_baseline - gap_hairremoved)

    a_base = df_a["brier_baseline"].values
    a_hr = df_a["brier_hairremoved"].values
    b_base = df_b["brier_baseline"].values
    b_hr = df_b["brier_hairremoved"].values

    bootstrap_reductions = []
    for _ in range(n_bootstrap):
        # Resample matched pairs with replacement within each group
        idx_a = rng.choice(n_a, size=n_a, replace=True)
        idx_b = rng.choice(n_b, size=n_b, replace=True)

        g_base = a_base[idx_a].mean() - b_base[idx_b].mean()
        g_hr = a_hr[idx_a].mean() - b_hr[idx_b].mean()
        bootstrap_reductions.append(g_base - g_hr)

    boot_arr = np.array(bootstrap_reductions)
    ci_lower = float(np.percentile(boot_arr, 2.5))
    ci_upper = float(np.percentile(boot_arr, 97.5))

    # One-sided p-value: fraction of bootstrap iterations where gap_reduction <= 0
    # (testing null hypothesis: hair removal does not reduce the gap)
    p_value = float(np.mean(boot_arr <= 0.0))

    # Statistically significant if 95% CI excludes zero (in the positive direction) AND p < 0.05
    significant = bool((ci_lower > 0.0) and (p_value < 0.05))

    return observed_reduction, ci_lower, ci_upper, p_value, significant


def print_plain_english_verdict(
    comparison: str,
    observed_reduction: float,
    ci_lower: float,
    ci_upper: float,
    p_value: float,
    significant: bool,
) -> None:
    """Prints the exact plain-English verdict formatted according to guidelines."""
    print("\n" + "=" * 80)
    print(f"  VERDICT FOR {comparison.upper()}")
    print("=" * 80)
    if significant:
        verdict = (
            f"The hair removal preprocessing produced a statistically significant reduction "
            f"in the {comparison.replace('_', '-')} calibration gap "
            f"(reduction={observed_reduction:.4f}, 95% CI=[{ci_lower:.4f}, {ci_upper:.4f}], p={p_value:.4f}). "
            f"Hair artifacts are a confirmed partial mediating factor."
        )
    else:
        verdict = (
            f"The observed gap reduction ({observed_reduction:.4f}) is not statistically significant "
            f"(95% CI=[{ci_lower:.4f}, {ci_upper:.4f}], p={p_value:.4f}). "
            f"Hair removal shows a trend toward improvement but the evidence is "
            f"insufficient to confirm it as a genuine effect at this sample size."
        )
    print(verdict)
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Paired bootstrap significance test for Brier score gap reduction"
    )
    parser.add_argument("--n_bootstrap", type=int, default=DEFAULT_BOOTSTRAP,
                        help="Number of bootstrap iterations (default 1000)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed (default 42)")
    args = parser.parse_args()

    # 1. Load or compute per-image Brier scores
    df_paired = load_or_compute_brier_scores(TEST_CSV_PATH)

    # 2 & 3. Run test for dark_vs_light (primary) and dark_vs_intermediate (secondary)
    comparisons = [
        ("dark_vs_light", "dark", "light"),
        ("dark_vs_intermediate", "dark", "intermediate"),
    ]

    results = []
    for comp_name, grp_a, grp_b in comparisons:
        print(f"\n[*] Running paired bootstrap test ({args.n_bootstrap} iterations) for {comp_name}...")
        obs_red, ci_lo, ci_hi, p_val, sig = run_paired_bootstrap_test(
            df_paired,
            group_a=grp_a,
            group_b=grp_b,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
        results.append({
            "comparison": comp_name,
            "observed_reduction": round(obs_red, 6),
            "ci_lower": round(ci_lo, 6),
            "ci_upper": round(ci_hi, 6),
            "p_value": round(p_val, 6),
            "significant": sig,
        })

    results_df = pd.DataFrame(results)

    # 5. Save results to CSV
    out_csv = HAIRREMOVED_DIR / "gap_significance_test.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\n[+] Saved results to {out_csv}:")
    print(results_df.to_string(index=False))

    # 6. Print plain-English verdicts
    for r in results:
        print_plain_english_verdict(
            comparison=r["comparison"],
            observed_reduction=r["observed_reduction"],
            ci_lower=r["ci_lower"],
            ci_upper=r["ci_upper"],
            p_value=r["p_value"],
            significant=r["significant"],
        )


if __name__ == "__main__":
    main()
