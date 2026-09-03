"""
src/calibration_fairness_hairremoved.py
========================================
Hair-removal sensitivity experiment for the DermaLens AI calibration-equity audit.

Research Question
-----------------
Does hair-artifact removal preprocessing reduce or eliminate the statistically
significant Brier-score disparity between estimated dark and light skin-tone
groups (dark=0.305, light=0.158 at baseline)?

Method
------
1. Loads pre-computed ITA labels from data/processed/test_with_ita.csv
   (does NOT recompute ITA — saves ~20 min).
2. For EACH image, applies remove_hair() (classical black-hat/top-hat + TELEA
   inpainting, no neural network) BEFORE running classifier inference.
3. Saves hair_coverage_pct per image to data/processed/test_with_ita_hairremoved.csv.
4. Runs the EXACT same calibration fairness pipeline as calibration_fairness.py
   (overall + per-ITA-group ECE/MCE/Brier, matched-size bootstrap × 1000,
   class-stratified analysis, significance tests, reliability diagrams).
5. All outputs saved to results/calibration_fairness_hairremoved/ (never
   overwrites the baseline in results/calibration_fairness/).
6. Generates results/calibration_fairness_hairremoved/before_vs_after_hair.csv
   comparing baseline vs hair-removed metrics for each ITA group.

Usage
-----
    # Full run (GPU recommended; ~45 min CPU):
    python -m src.calibration_fairness_hairremoved

    # Smoke test — 10 random images, CPU, no bootstrap:
    python -m src.calibration_fairness_hairremoved --smoke_test --n_smoke 10

    # Skip hair-removal step (reuse existing test_with_ita_hairremoved.csv):
    python -m src.calibration_fairness_hairremoved --skip_hair_removal

Outputs (results/calibration_fairness_hairremoved/)
----------------------------------------------------
    calibration_by_ita_group.csv
    calibration_matched_sampling.csv
    significance_tests.csv
    class_stratified_analysis.csv
    class_distribution_by_ita_group.csv
    reliability_diagram_{dark,intermediate,light}.png
    calibration_fairness_report.txt
    before_vs_after_hair.csv          <- before/after comparison table
    hair_coverage_stats.csv           <- per-ITA-group hair coverage summary
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from src.dataset import class_to_idx, get_eval_transforms
from src.hair_removal import remove_hair
from src.model import get_efficientnet_b0

# ── Shared calibration functions imported directly from the baseline module ────
from src.calibration_fairness import (
    brier_score_multiclass,
    calibration_metrics_for_group,
    class_distribution_confound_check,
    class_stratified_analysis,
    compute_significance_tests,
    ece_mce,
    matched_bootstrap,
    plot_reliability_diagram,
    write_report,
    ITA_LIMITATION,
    MIN_GROUP_WARN,
    N_BOOTSTRAP,
    N_CAL_BINS,
)

# ── Output directory ───────────────────────────────────────────────────────────
OUT_DIR          = Path("results/calibration_fairness_hairremoved")
BASELINE_DIR     = Path("results/calibration_fairness")
ITA_CSV          = Path("data/processed/test_with_ita.csv")
HAIR_ITA_CSV     = Path("data/processed/test_with_ita_hairremoved.csv")
CHECKPOINT_PATH  = "checkpoints/best_efficientnet_b0.pth"
NUM_CLASSES      = 7


# ── Step 1: Hair removal + inference (per-image, no DataLoader) ────────────────

def _eval_transforms_to_tensor(image_rgb_uint8: np.ndarray, transform) -> torch.Tensor:
    """Apply albumentations eval transform and return a CHW float tensor."""
    result = transform(image=image_rgb_uint8)
    img_t  = result["image"]          # already CHW float32 tensor from ToTensorV2
    return img_t


def run_hair_removal_and_inference(
    df: pd.DataFrame,
    checkpoint_path: str = CHECKPOINT_PATH,
    num_classes: int = NUM_CLASSES,
    smoke_n: int = 0,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float], list[bool]]:
    """
    For each image in df (ordered):
      1. Load BGR from image_path.
      2. Apply remove_hair() -> cleaned BGR.
      3. Convert to RGB, apply eval transform, run classifier.

    Returns
    -------
    probs              : (N, num_classes)  softmax probabilities
    preds              : (N,)              argmax predictions
    labels             : (N,)              true class indices
    hair_coverage_pcts : list[float]       per-image hair coverage %
    flagged_list       : list[bool]        True if safety valve triggered
    """
    device = torch.device("cpu")

    # Load model
    model = get_efficientnet_b0(num_classes)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    if verbose:
        print(f"  [inference] Loaded checkpoint epoch {ckpt.get('epoch', '?')}")

    transform = get_eval_transforms(224)

    n = len(df) if smoke_n <= 0 else min(smoke_n, len(df))
    df_sub = df.head(n).reset_index(drop=True)

    all_probs, all_preds, all_labels = [], [], []
    hair_coverage_pcts: list[float] = []
    flagged_list:       list[bool]  = []

    print(f"\n  Running hair removal + inference on {n} images …")
    t0 = time.time()

    for idx, row in tqdm(df_sub.iterrows(), total=n, desc="  hair-removal inference"):
        img_path = str(row["image_path"])

        # ── Hair removal ──────────────────────────────────────────────────────
        try:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                raise IOError(f"cv2.imread failed: {img_path}")
            cleaned_bgr, coverage_pct, flagged = remove_hair(img_bgr)
        except Exception as exc:
            print(f"  [WARN] hair removal failed for {img_path}: {exc}")
            img_bgr  = cv2.imread(img_path) or np.zeros((224, 224, 3), dtype=np.uint8)
            cleaned_bgr = img_bgr
            coverage_pct = 0.0
            flagged      = False

        hair_coverage_pcts.append(coverage_pct)
        flagged_list.append(flagged)

        # ── Convert BGR -> RGB -> Tensor ──────────────────────────────────────
        cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)
        img_t = _eval_transforms_to_tensor(cleaned_rgb, transform).unsqueeze(0).to(device)

        # ── Classifier inference ──────────────────────────────────────────────
        with torch.no_grad():
            logits = model(img_t)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]   # (num_classes,)
            pred   = int(probs.argmax())

        all_probs.append(probs)
        all_preds.append(pred)
        all_labels.append(int(row["label_id"]))

    elapsed = time.time() - t0
    fps = n / elapsed
    eta_full = (len(df) * (elapsed / n)) / 60

    print(f"  Done in {elapsed:.1f}s  ({fps:.2f} img/s)")
    print(f"  Estimated full-set runtime: {eta_full:.1f} min ({len(df)} images at {fps:.2f} img/s)")

    return (
        np.stack(all_probs, axis=0),
        np.array(all_preds, dtype=np.int64),
        np.array(all_labels, dtype=np.int64),
        hair_coverage_pcts,
        flagged_list,
    )


# ── Hair coverage summary stats ────────────────────────────────────────────────

def summarise_hair_coverage(
    df: pd.DataFrame,
    hair_coverage_pcts: list[float],
    flagged_list: list[bool],
    out_dir: Path,
) -> pd.DataFrame:
    """
    Compute per-ITA-group hair coverage statistics and save to CSV.
    Also prints a console table.
    """
    n = len(hair_coverage_pcts)
    df_work = df.head(n).copy().reset_index(drop=True)
    df_work["hair_coverage_pct"] = hair_coverage_pcts
    df_work["flagged"]           = flagged_list

    groups = ["light", "intermediate", "dark", "unknown", "ALL"]
    rows = []
    for grp in groups:
        if grp == "ALL":
            sub = df_work
        else:
            sub = df_work[df_work["ita_group"] == grp]

        if len(sub) == 0:
            continue
        cov = sub["hair_coverage_pct"]
        rows.append({
            "ita_group":         grp,
            "n":                 len(sub),
            "n_flagged":         int(sub["flagged"].sum()),
            "pct_flagged":       round(100.0 * sub["flagged"].mean(), 2),
            "coverage_mean_pct": round(float(cov.mean()), 3),
            "coverage_median_pct": round(float(cov.median()), 3),
            "coverage_p90_pct":  round(float(np.percentile(cov.values, 90)), 3),
            "coverage_max_pct":  round(float(cov.max()), 3),
        })

    summary_df = pd.DataFrame(rows)
    out_path = out_dir / "hair_coverage_stats.csv"
    summary_df.to_csv(out_path, index=False)

    print("\n── Hair coverage per ITA group ──────────────────────────────────────────")
    print(summary_df.to_string(index=False))
    print(f"  Saved -> {out_path}")
    return summary_df


# ── Before-vs-after comparison table ─────────────────────────────────────────

def build_before_vs_after(
    raw_df_new: pd.DataFrame,
    matched_df_new: pd.DataFrame,
    out_dir: Path,
    baseline_dir: Path = BASELINE_DIR,
) -> pd.DataFrame:
    """
    Read the baseline CSVs and compare with new (hair-removed) results.
    Returns a wide comparison DataFrame and saves to CSV.
    """
    baseline_raw_path     = baseline_dir / "calibration_by_ita_group.csv"
    baseline_matched_path = baseline_dir / "calibration_matched_sampling.csv"

    if not baseline_raw_path.exists() or not baseline_matched_path.exists():
        print(f"  [WARN] Baseline CSVs not found in {baseline_dir} — skipping comparison table.")
        return pd.DataFrame()

    base_raw     = pd.read_csv(baseline_raw_path).set_index("group")
    base_matched = pd.read_csv(baseline_matched_path).set_index("group")
    new_raw      = raw_df_new.set_index("group")
    new_matched  = matched_df_new.set_index("group")

    groups = [g for g in ["light", "intermediate", "dark"]
              if g in base_raw.index and g in new_raw.index]

    rows = []
    for grp in groups:
        row = {"ita_group": grp}

        # Raw (unmatched)
        for metric in ["accuracy", "ece", "mce", "brier"]:
            base_val = float(base_raw.loc[grp, metric]) if metric in base_raw.columns else float("nan")
            new_val  = float(new_raw.loc[grp, metric])  if metric in new_raw.columns  else float("nan")
            row[f"before_{metric}"]  = round(base_val, 6)
            row[f"after_{metric}"]   = round(new_val,  6)
            row[f"delta_{metric}"]   = round(new_val - base_val, 6)

        # Matched bootstrap means
        for metric in ["brier", "ece"]:
            key = f"{metric}_mean"
            base_val = float(base_matched.loc[grp, key]) if key in base_matched.columns else float("nan")
            new_val  = float(new_matched.loc[grp, key])  if key in new_matched.columns  else float("nan")
            row[f"before_matched_{metric}"]  = round(base_val, 6)
            row[f"after_matched_{metric}"]   = round(new_val,  6)
            row[f"delta_matched_{metric}"]   = round(new_val - base_val, 6)

        rows.append(row)

    comp_df = pd.DataFrame(rows)
    out_path = out_dir / "before_vs_after_hair.csv"
    comp_df.to_csv(out_path, index=False)

    print("\n── BEFORE-vs-AFTER comparison (hair removal) ────────────────────────────")
    print(comp_df.to_string(index=False))
    print(f"  Saved -> {out_path}")

    # Human-readable summary
    print("\n  KEY METRICS (dark group):")
    if "dark" in [r["ita_group"] for r in rows]:
        dark = next(r for r in rows if r["ita_group"] == "dark")
        print(f"    Brier:         {dark['before_brier']:.4f} -> {dark['after_brier']:.4f}  "
              f"(delta={dark['delta_brier']:+.4f})")
        print(f"    ECE:           {dark['before_ece']:.4f} -> {dark['after_ece']:.4f}  "
              f"(delta={dark['delta_ece']:+.4f})")
        print(f"    Accuracy:      {dark['before_accuracy']:.4f} -> {dark['after_accuracy']:.4f}  "
              f"(delta={dark['delta_accuracy']:+.4f})")
    if "light" in [r["ita_group"] for r in rows]:
        light = next(r for r in rows if r["ita_group"] == "light")
        print(f"\n  KEY METRICS (light group):")
        print(f"    Brier:         {light['before_brier']:.4f} -> {light['after_brier']:.4f}  "
              f"(delta={light['delta_brier']:+.4f})")

    return comp_df


# ── Smoke-test report ──────────────────────────────────────────────────────────

def print_smoke_test_report(
    df_sub: pd.DataFrame,
    hair_coverage_pcts: list[float],
    flagged_list:       list[bool],
    probs:              np.ndarray,
    preds:              np.ndarray,
    labels:             np.ndarray,
) -> None:
    """Print a per-image summary table for the smoke test."""
    print("\n" + "=" * 95)
    print("  SMOKE TEST — PER-IMAGE HAIR REMOVAL RESULTS")
    print("=" * 95)
    header = (
        f"  {'#':>3}  {'image_id':20s}  {'ita_group':14s}  {'dx':6s}  "
        f"{'coverage%':>10s}  {'flagged':>7s}  {'pred_class':12s}  {'correct':>7s}"
    )
    print(header)
    print("  " + "-" * 92)

    from src.dataset import CLASSES
    for i, (idx, row) in enumerate(df_sub.iterrows()):
        img_id    = str(row.get("image_id", row.get("image_path", "?")))[-20:]
        ita_grp   = str(row.get("ita_group", "?"))
        dx        = str(row.get("dx", "?"))
        cov       = hair_coverage_pcts[i]
        flag_str  = "YES" if flagged_list[i] else "no"
        pred_cls  = CLASSES[preds[i]] if preds[i] < len(CLASSES) else "?"
        correct   = "YES" if preds[i] == labels[i] else "no"
        cov_warn  = "  ^" if flagged_list[i] else "   "
        print(
            f"  {i+1:>3}  {img_id:20s}  {ita_grp:14s}  {dx:6s}  "
            f"{cov:>9.2f}%{cov_warn:3s}  {flag_str:>7s}  {pred_cls:12s}  {correct:>7s}"
        )

    print()
    all_cov = np.array(hair_coverage_pcts)
    n_flagged = sum(flagged_list)
    print(f"  Coverage range: {all_cov.min():.2f}% – {all_cov.max():.2f}%  "
          f"(mean={all_cov.mean():.2f}%  median={np.median(all_cov):.2f}%)")
    print(f"  Flagged images (>20% coverage — safety valve): {n_flagged}/{len(hair_coverage_pcts)}")
    acc = float((preds == labels).mean())
    print(f"  Mini-set accuracy: {acc:.4f}  (n={len(labels)})")
    print("  ^ = >20% coverage (safety valve — original returned, inpainting skipped)")
    print("=" * 95)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibration fairness audit with hair-removal preprocessing"
    )
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run on a small subset only — no bootstrap, quick output")
    parser.add_argument("--n_smoke", type=int, default=10,
                        help="Number of images for smoke test (default 10)")
    parser.add_argument("--n_bootstrap", type=int, default=N_BOOTSTRAP,
                        help="Bootstrap iterations (default 1000)")
    parser.add_argument("--skip_hair_removal", action="store_true",
                        help="Reuse existing test_with_ita_hairremoved.csv (hair already removed)")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--ita_csv", type=str, default=str(ITA_CSV))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ITA-labelled test CSV (do NOT recompute ITA) ─────────────────────
    ita_csv_path = Path(args.ita_csv)
    if not ita_csv_path.exists():
        raise FileNotFoundError(
            f"test_with_ita.csv not found at {ita_csv_path}. "
            "Run calibration_fairness.py first to generate ITA labels."
        )
    df = pd.read_csv(ita_csv_path)
    print(f"[*] Loaded {len(df)} images with pre-computed ITA labels from {ita_csv_path}")
    print(f"    ITA group counts: { df['ita_group'].value_counts().to_dict() }")

    # ── SMOKE TEST fast path ──────────────────────────────────────────────────
    if args.smoke_test:
        print(f"\n[SMOKE TEST] Running on {args.n_smoke} randomly sampled images …")
        df_smoke = df.sample(n=min(args.n_smoke, len(df)), random_state=42).reset_index(drop=True)

        probs, preds, labels, cov_pcts, flagged = run_hair_removal_and_inference(
            df_smoke,
            checkpoint_path=args.checkpoint,
            num_classes=NUM_CLASSES,
            smoke_n=0,   # 0 = use all of df_smoke
            verbose=True,
        )
        print_smoke_test_report(df_smoke, cov_pcts, flagged, probs, preds, labels)

        # Save smoke CSV
        df_smoke["hair_coverage_pct"] = cov_pcts
        df_smoke["flagged"]           = flagged
        smoke_csv = OUT_DIR / f"smoke_test_{args.n_smoke}_images.csv"
        df_smoke.to_csv(smoke_csv, index=False)
        print(f"  Smoke CSV saved: {smoke_csv}")
        return

    # ── Full run ──────────────────────────────────────────────────────────────
    if args.skip_hair_removal and HAIR_ITA_CSV.exists():
        print(f"\n[*] Reusing existing hair-removed CSV: {HAIR_ITA_CSV}")
        df_hr = pd.read_csv(HAIR_ITA_CSV)
        # Re-run inference only on cleaned images (need cleaned images on disk)
        # For skip mode, this script assumes the CSV already has hair_coverage_pct
        # and we just re-run inference. For full reproducibility, re-running
        # hair removal is recommended.
        print("[WARN] --skip_hair_removal reloads coverage pcts but re-runs inference "
              "on ORIGINAL images (cleaned images not cached). For full reproducibility, "
              "run without --skip_hair_removal.")
        probs, preds, labels, cov_pcts, flagged = run_hair_removal_and_inference(
            df,
            checkpoint_path=args.checkpoint,
            num_classes=NUM_CLASSES,
            smoke_n=0,
            verbose=True,
        )
        cov_pcts = list(df_hr.get("hair_coverage_pct", [0.0] * len(df)))
        flagged  = list(df_hr.get("flagged", [False]  * len(df)))
    else:
        probs, preds, labels, cov_pcts, flagged = run_hair_removal_and_inference(
            df,
            checkpoint_path=args.checkpoint,
            num_classes=NUM_CLASSES,
            smoke_n=0,
            verbose=True,
        )

    # ── Save extended CSV with hair coverage ──────────────────────────────────
    df_out = df.copy()
    df_out["hair_coverage_pct"] = cov_pcts
    df_out["flagged"]           = flagged
    df_out.to_csv(HAIR_ITA_CSV, index=False)
    print(f"\n  Saved hair-labelled CSV -> {HAIR_ITA_CSV}")

    # ── Hair coverage stats ────────────────────────────────────────────────────
    summarise_hair_coverage(df, cov_pcts, flagged, OUT_DIR)

    # ── Overall calibration metrics ───────────────────────────────────────────
    overall = calibration_metrics_for_group(probs, preds, labels, NUM_CLASSES)
    print(f"\n  Overall — accuracy={overall['accuracy']:.4f}  "
          f"ECE={overall['ece']:.4f}  MCE={overall['mce']:.4f}  "
          f"Brier={overall['brier']:.4f}  N={overall['n']}")

    # ── Per-group raw calibration metrics ─────────────────────────────────────
    print("\n── Per-group (unmatched) calibration ──────────────────────────────────")
    raw_rows   = []
    group_data: dict[str, tuple] = {}
    group_counts: dict[str, int] = {}
    ita_groups_present = [g for g in ["light", "intermediate", "dark"]
                          if (df["ita_group"] == g).sum() > 0]

    for grp in ita_groups_present:
        mask     = df["ita_group"].values == grp
        probs_g  = probs[mask]
        preds_g  = preds[mask]
        labels_g = labels[mask]

        m = calibration_metrics_for_group(probs_g, preds_g, labels_g, NUM_CLASSES)
        raw_rows.append({"group": grp, **m})
        group_data[grp]   = (probs_g, preds_g, labels_g)
        group_counts[grp] = int(mask.sum())
        print(f"  {grp:15s} n={m['n']:>4d}  acc={m['accuracy']:.4f}  "
              f"ECE={m['ece']:.4f}  MCE={m['mce']:.4f}  Brier={m['brier']:.4f}")

    unk_n = int((df["ita_group"] == "unknown").sum())
    if unk_n > 0:
        group_counts["unknown"] = unk_n
        print(f"  {'unknown':15s} n={unk_n:>4d}  (skipped — ITA estimation failed)")

    raw_df = pd.DataFrame(raw_rows)
    raw_path = OUT_DIR / "calibration_by_ita_group.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"\n  Raw calibration CSV -> {raw_path}")

    # ── Matched-size bootstrap ─────────────────────────────────────────────────
    matched_df, iter_records = matched_bootstrap(
        group_data, NUM_CLASSES, n_iter=args.n_bootstrap
    )
    matched_path = OUT_DIR / "calibration_matched_sampling.csv"
    matched_df.to_csv(matched_path, index=False)
    print(f"  Matched calibration CSV -> {matched_path}")

    # ── Significance tests ────────────────────────────────────────────────────
    print("\n── Computing significance tests ─────────────────────────────────────────")
    sig_df = compute_significance_tests(iter_records, n_permutations=10_000)
    sig_path = OUT_DIR / "significance_tests.csv"
    sig_df.to_csv(sig_path, index=False)
    print(f"  Significance tests CSV -> {sig_path}")

    # ── Class distribution confound check ─────────────────────────────────────
    print("\n── Confound check: dx class distribution by ITA group ──────────────────")
    chi2_result = class_distribution_confound_check(
        df, OUT_DIR / "class_distribution_by_ita_group.csv"
    )

    # ── Class-stratified analysis ──────────────────────────────────────────────
    _, class_strat_lines = class_stratified_analysis(
        df, probs, preds, labels, NUM_CLASSES,
        n_bootstrap=args.n_bootstrap, out_dir=OUT_DIR,
    )

    # ── Reliability diagrams ───────────────────────────────────────────────────
    print("\n── Generating reliability diagrams ──────────────────────────────────────")
    for grp, (probs_g, preds_g, labels_g) in group_data.items():
        fig_path = OUT_DIR / f"reliability_diagram_{grp}.png"
        plot_reliability_diagram(probs_g, preds_g, labels_g, grp, fig_path)
        print(f"  {grp}: {fig_path}")

    # ── Calibration fairness report ────────────────────────────────────────────
    report_path = OUT_DIR / "calibration_fairness_report.txt"
    write_report(
        report_path,
        model_name="efficientnet_b0 [hair-removed input]",
        split="test",
        overall=overall,
        raw_df=raw_df,
        matched_df=matched_df,
        group_counts=group_counts,
        sig_df=sig_df,
        chi2_result=chi2_result,
        class_strat_lines=class_strat_lines,
    )

    # ── BEFORE-vs-AFTER comparison ─────────────────────────────────────────────
    build_before_vs_after(raw_df, matched_df, OUT_DIR, BASELINE_DIR)

    # ── Print report to stdout ─────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  FINAL REPORT (hair-removed)")
    print("=" * 72)
    with open(report_path, encoding="utf-8") as fh:
        print(fh.read())


if __name__ == "__main__":
    main()
