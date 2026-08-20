"""
src/calibration_fairness_reweighted.py
=======================================
Runs the calibration fairness pipeline on the REWEIGHTED checkpoint
(checkpoints/best_efficientnet_b0_reweighted.pth) against the SAME test
set used for the baseline, then builds a direct before-vs-after comparison.

IMPORTANT: Does NOT touch results/calibration_fairness/ (baseline) at all.
All outputs go to results/calibration_fairness_reweighted/

Run from project root:
    python -m src.calibration_fairness_reweighted
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

# ── Re-use all metric/inference/bootstrap functions from the existing module ──
from src.calibration_fairness import (
    run_inference,
    calibration_metrics_for_group,
    brier_score_multiclass,
    matched_bootstrap,
    compute_significance_tests,
    CHECKPOINT_TEMPLATE,
    N_BOOTSTRAP,
)
from src.dataset import class_to_idx

# ── Paths ─────────────────────────────────────────────────────────────────────
REWEIGHTED_CKPT = "checkpoints/best_efficientnet_b0_reweighted.pth"
TEST_ITA_CSV    = "data/processed/test_with_ita.csv"
BASELINE_DIR    = Path("results/calibration_fairness")
OUT_DIR         = Path("results/calibration_fairness_reweighted")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME  = "efficientnet_b0"
NUM_CLASSES = len(class_to_idx)
N_BOOT      = N_BOOTSTRAP
MIN_PER_GRP = 15   # class-stratified threshold
SEED        = 42

print("=" * 70)
print("  Calibration Fairness — Reweighted vs Baseline Comparison")
print("=" * 70)

# ─── Load test set with existing ITA labels ───────────────────────────────────
print(f"\nLoading {TEST_ITA_CSV} …")
df = pd.read_csv(TEST_ITA_CSV)
print(f"  {len(df)} images  ita_groups={dict(df['ita_group'].value_counts())}")

# ─── Run inference ────────────────────────────────────────────────────────────
probs, preds, labels = run_inference(
    df, MODEL_NAME, REWEIGHTED_CKPT, NUM_CLASSES, batch_size=32
)

# ─── Step 1: Per-ITA-group calibration metrics ────────────────────────────────
print("\n── Per-group calibration metrics ───────────────────────────────────────")
groups_of_interest = ["light", "intermediate", "dark"]
group_rows = []
group_data = {}   # for bootstrap

for grp in groups_of_interest:
    mask = (df["ita_group"] == grp).values
    if mask.sum() == 0:
        continue
    m = calibration_metrics_for_group(probs[mask], preds[mask], labels[mask], NUM_CLASSES)
    m["group"] = grp
    group_rows.append(m)
    group_data[grp] = (probs[mask], preds[mask], labels[mask])
    print(f"  {grp:13s}: n={m['n']:>4d}  acc={m['accuracy']:.4f}"
          f"  ECE={m['ece']:.4f}  MCE={m['mce']:.4f}  Brier={m['brier']:.4f}")

cal_df = pd.DataFrame(group_rows)[["group", "ece", "mce", "brier", "accuracy", "n"]]
cal_csv = OUT_DIR / "calibration_by_ita_group.csv"
cal_df.to_csv(cal_csv, index=False)
print(f"  → {cal_csv}")

# ─── Step 2: Matched-size bootstrap + significance tests ─────────────────────
bootstrap_summary, iter_records = matched_bootstrap(group_data, NUM_CLASSES, N_BOOT, SEED)

sig_df = compute_significance_tests(iter_records)
sig_csv = OUT_DIR / "significance_tests.csv"
sig_df.to_csv(sig_csv, index=False)
print(f"  → {sig_csv}")

# ─── Step 3: Class-stratified analysis (nv focus) ────────────────────────────
print("\n── Class-stratified analysis ────────────────────────────────────────────")
# Count per (dx_class, ita_group)
dx_grp_counts = (
    df[df["ita_group"].isin(groups_of_interest)]
    .groupby(["dx", "ita_group"])
    .size()
    .unstack(fill_value=0)
)

strat_rows = []
for dx_cls in dx_grp_counts.index:
    counts = {g: int(dx_grp_counts.loc[dx_cls, g]) for g in groups_of_interest
              if g in dx_grp_counts.columns}
    below = [g for g in groups_of_interest if counts.get(g, 0) < MIN_PER_GRP]
    if below:
        print(f"  SKIP {dx_cls}: insufficient in {below} "
              f"(counts={counts})")
        continue

    print(f"  QUALIFY {dx_cls}: {counts}")
    cls_mask = (df["dx"] == dx_cls)

    # Per-group metrics within this class
    grp_m = {}
    cls_group_data = {}
    for grp in groups_of_interest:
        m_grp = cls_mask & (df["ita_group"] == grp)
        idx = m_grp.values.nonzero()[0]
        b = brier_score_multiclass(probs[idx], labels[idx], NUM_CLASSES)
        acc = float((preds[idx] == labels[idx]).mean())
        grp_m[grp] = {"n": len(idx), "brier": b, "acc": acc}
        cls_group_data[grp] = (probs[idx], preds[idx], labels[idx])
        print(f"    {grp:13s}: n={len(idx):3d}  acc={acc:.4f}  Brier={b:.4f}")

    # Bootstrap within this class
    rng = np.random.default_rng(SEED + 7)
    n_min = min(v["n"] for v in grp_m.values())
    brier_boot: dict[str, list] = {g: [] for g in groups_of_interest}
    for _ in range(N_BOOT):
        for g in groups_of_interest:
            pp, pd_, ll = cls_group_data[g]
            idx_b = rng.choice(len(ll), size=n_min, replace=True)
            brier_boot[g].append(brier_score_multiclass(pp[idx_b], ll[idx_b], NUM_CLASSES))

    pairs = [("light", "dark"), ("light", "intermediate"), ("intermediate", "dark")]
    for g_a, g_b in pairs:
        diff_dist = np.array(brier_boot[g_a]) - np.array(brier_boot[g_b])
        diff  = round(float(diff_dist.mean()), 6)
        ci_lo = round(float(np.percentile(diff_dist, 2.5)), 6)
        ci_hi = round(float(np.percentile(diff_dist, 97.5)), 6)
        sig   = bool((ci_lo > 0) or (ci_hi < 0))
        strat_rows.append({
            "dx_class": dx_cls, "group_a": g_a, "group_b": g_b,
            "n_a": grp_m[g_a]["n"], "n_b": grp_m[g_b]["n"],
            "brier_diff": diff, "ci_lower": ci_lo, "ci_upper": ci_hi,
            "significant": sig,
        })
        sig_str = "YES *" if sig else "no"
        print(f"    {g_a} vs {g_b}: diff={diff:+.4f} CI=[{ci_lo:.4f},{ci_hi:.4f}] {sig_str}")

strat_df = pd.DataFrame(strat_rows)
strat_csv = OUT_DIR / "class_stratified_analysis.csv"
strat_df.to_csv(strat_csv, index=False)
print(f"  → {strat_csv}")

# ─── Step 4: Before-vs-After comparison table ─────────────────────────────────
print("\n── Building before-vs-after comparison table ────────────────────────────")

# Load baseline CSVs (DO NOT MODIFY)
base_cal  = pd.read_csv(BASELINE_DIR / "calibration_by_ita_group.csv")
base_strat = pd.read_csv(BASELINE_DIR / "class_stratified_analysis.csv")

def get_group_metric(cal_df_: pd.DataFrame, group: str, metric: str) -> float:
    row = cal_df_[cal_df_["group"] == group]
    if row.empty:
        return float("nan")
    return float(row[metric].iloc[0])

def get_strat_row(strat_df_: pd.DataFrame, dx_cls: str, g_a: str, g_b: str) -> dict:
    r = strat_df_[(strat_df_["dx_class"] == dx_cls) &
                  (strat_df_["group_a"] == g_a) &
                  (strat_df_["group_b"] == g_b)]
    if r.empty:
        return {}
    return r.iloc[0].to_dict()

# Overall accuracy (all test images)
overall_acc_base  = float((pd.read_csv(TEST_ITA_CSV)["label_id"].values ==
                            np.load  # placeholder — computed below
                            if False else -999))  # computed below
# Compute from loaded inference results
overall_acc_rw    = float((preds == labels).mean())
# For baseline overall acc, recompute from per-group weighted average
base_overall_acc = sum(
    get_group_metric(base_cal, g, "accuracy") * get_group_metric(base_cal, g, "n")
    for g in ["light", "intermediate", "dark"]
) / sum(get_group_metric(base_cal, g, "n") for g in ["light", "intermediate", "dark"])
# Actually, load from full test CSV counts
mask_all = df["ita_group"].isin(["light", "intermediate", "dark"]).values
overall_acc_rw    = float((preds[mask_all] == labels[mask_all]).mean())

# For baseline, compute properly
base_cal_n = {g: int(get_group_metric(base_cal, g, "n")) for g in ["light","intermediate","dark"]}
base_cal_acc = {g: get_group_metric(base_cal, g, "accuracy") for g in ["light","intermediate","dark"]}
base_overall_acc = sum(base_cal_acc[g] * base_cal_n[g] for g in base_cal_n) / sum(base_cal_n.values())

# nv class stratified rows: light vs dark
nv_base = get_strat_row(base_strat, "nv", "light", "dark")
nv_rw   = next(
    (r for r in strat_rows if r["dx_class"] == "nv" and r["group_a"] == "light" and r["group_b"] == "dark"),
    {}
)

# Build comparison rows
def fmt(v):
    return round(float(v), 4) if not (isinstance(v, float) and math.isnan(v)) else float("nan")

comp_rows = [
    {
        "metric":            "overall_accuracy (light+inter+dark)",
        "baseline_dark":     fmt(get_group_metric(base_cal, "dark",  "accuracy")),
        "reweighted_dark":   fmt(get_group_metric(cal_df, "dark",    "accuracy")),
        "baseline_light":    fmt(get_group_metric(base_cal, "light", "accuracy")),
        "reweighted_light":  fmt(get_group_metric(cal_df, "light",   "accuracy")),
        "baseline_gap (dark-light)":   fmt(get_group_metric(base_cal, "dark", "accuracy") -
                                           get_group_metric(base_cal, "light", "accuracy")),
        "reweighted_gap (dark-light)": fmt(get_group_metric(cal_df, "dark", "accuracy") -
                                           get_group_metric(cal_df, "light", "accuracy")),
        "gap_change":        fmt((get_group_metric(cal_df, "dark", "accuracy") -
                                  get_group_metric(cal_df, "light", "accuracy")) -
                                 (get_group_metric(base_cal, "dark", "accuracy") -
                                  get_group_metric(base_cal, "light", "accuracy"))),
    },
    {
        "metric":            "overall_brier",
        "baseline_dark":     fmt(get_group_metric(base_cal, "dark",  "brier")),
        "reweighted_dark":   fmt(get_group_metric(cal_df, "dark",    "brier")),
        "baseline_light":    fmt(get_group_metric(base_cal, "light", "brier")),
        "reweighted_light":  fmt(get_group_metric(cal_df, "light",   "brier")),
        "baseline_gap (dark-light)":   fmt(get_group_metric(base_cal, "dark", "brier") -
                                           get_group_metric(base_cal, "light", "brier")),
        "reweighted_gap (dark-light)": fmt(get_group_metric(cal_df, "dark", "brier") -
                                           get_group_metric(cal_df, "light", "brier")),
        "gap_change":        fmt((get_group_metric(cal_df, "dark", "brier") -
                                  get_group_metric(cal_df, "light", "brier")) -
                                 (get_group_metric(base_cal, "dark", "brier") -
                                  get_group_metric(base_cal, "light", "brier"))),
    },
    {
        "metric":            "full_model_accuracy (all groups incl. unknown)",
        "baseline_dark":     "n/a",
        "reweighted_dark":   "n/a",
        "baseline_light":    fmt(base_overall_acc),
        "reweighted_light":  fmt(overall_acc_rw),
        "baseline_gap (dark-light)":   "n/a",
        "reweighted_gap (dark-light)": "n/a",
        "gap_change":        fmt(overall_acc_rw - base_overall_acc),
    },
]

# nv-class accuracy
if nv_base and nv_rw:
    # Per group from within nv — read accuracy from full strat
    nv_acc_base_light = float(base_strat[base_strat["dx_class"] == "nv"]["n_a"].iloc[0]) if "n_a" in base_strat.columns else float("nan")
    # Get from group metrics for nv within this run
    nv_grp_rw = next((r for r in strat_rows if r["dx_class"] == "nv"), {})
    # nv accuracy per group (baseline — not in strat CSV, compute approximation)
    nv_acc_note = "see class_stratified_analysis.csv for n counts"

# nv brier — the primary metric
nv_brier_base_light = float("nan")
nv_brier_base_dark  = float("nan")
nv_brier_rw_light   = float("nan")
nv_brier_rw_dark    = float("nan")
nv_diff_base = float("nan")
nv_diff_rw   = float("nan")
nv_sig_base  = None
nv_sig_rw    = None
nv_ci_base   = ("n/a", "n/a")
nv_ci_rw     = ("n/a", "n/a")

if nv_base:
    nv_diff_base = float(nv_base.get("brier_diff", float("nan")))
    nv_ci_base   = (float(nv_base.get("ci_lower", float("nan"))),
                    float(nv_base.get("ci_upper", float("nan"))))
    nv_sig_base  = bool(nv_base.get("significant", False))
if nv_rw:
    nv_diff_rw = float(nv_rw.get("brier_diff", float("nan")))
    nv_ci_rw   = (float(nv_rw.get("ci_lower", float("nan"))),
                  float(nv_rw.get("ci_upper", float("nan"))))
    nv_sig_rw  = bool(nv_rw.get("significant", False))

# Get actual per-group brier for nv from the matching strat_rows entries
for row_ in strat_rows:
    if row_["dx_class"] == "nv":
        # Recompute raw briers from inference (already printed above)
        pass  # grp_m is local — values were printed. Re-derive from cal summary.

# Derive nv brier from strat analysis brier_diff and known pairs
# We have light-vs-dark diff for baseline: -0.230742 (light-dark = light_brier - dark_brier)
# And for reweighted: from strat_rows

# For a cleaner table, use the Brier_diff column (light brier - dark brier)
# Positive diff = dark group has higher Brier (worse calibrated)
nv_row_base = {
    "metric":            "nv_class_brier light-vs-dark gap (positive = dark worse)",
    "baseline_dark":     "n/a",
    "reweighted_dark":   "n/a",
    "baseline_light":    "n/a",
    "reweighted_light":  "n/a",
    "baseline_gap (dark-light)":
        f"{-nv_diff_base:.4f} (CI=[{-nv_ci_base[1]:.4f},{-nv_ci_base[0]:.4f}], sig={nv_sig_base})"
        if not math.isnan(nv_diff_base) else "n/a",
    "reweighted_gap (dark-light)":
        f"{-nv_diff_rw:.4f} (CI=[{-nv_ci_rw[1]:.4f},{-nv_ci_rw[0]:.4f}], sig={nv_sig_rw})"
        if not math.isnan(nv_diff_rw) else "n/a",
    "gap_change":
        fmt((-nv_diff_rw) - (-nv_diff_base)) if not (math.isnan(nv_diff_base) or math.isnan(nv_diff_rw)) else "n/a",
}
comp_rows.append(nv_row_base)

comp_df = pd.DataFrame(comp_rows)
comp_csv = OUT_DIR / "before_vs_after_comparison.csv"
comp_df.to_csv(comp_csv, index=False)
print(f"\n  Before-vs-after CSV → {comp_csv}")
print(comp_df.to_string(index=False))

# ─── Step 5: Plain-English comparison report ──────────────────────────────────
print("\n── Writing comparison_report.txt ────────────────────────────────────────")

# Key numbers
base_acc_dark  = get_group_metric(base_cal, "dark",  "accuracy")
base_acc_light = get_group_metric(base_cal, "light", "accuracy")
rw_acc_dark    = get_group_metric(cal_df,   "dark",  "accuracy")
rw_acc_light   = get_group_metric(cal_df,   "light", "accuracy")

base_brier_dark  = get_group_metric(base_cal, "dark",  "brier")
base_brier_light = get_group_metric(base_cal, "light", "brier")
rw_brier_dark    = get_group_metric(cal_df,   "dark",  "brier")
rw_brier_light   = get_group_metric(cal_df,   "light", "brier")

base_gap_acc   = base_acc_dark   - base_acc_light
rw_gap_acc     = rw_acc_dark     - rw_acc_light
base_gap_brier = base_brier_dark - base_brier_light
rw_gap_brier   = rw_brier_dark   - rw_brier_light

# nv-class brier gap (dark - light, so positive means dark is worse)
nv_gap_base = -nv_diff_base  if not math.isnan(nv_diff_base) else float("nan")
nv_gap_rw   = -nv_diff_rw    if not math.isnan(nv_diff_rw)   else float("nan")
nv_gap_change = nv_gap_rw - nv_gap_base if not (math.isnan(nv_gap_base) or math.isnan(nv_gap_rw)) else float("nan")

lines = [
    "=" * 72,
    "  DermaLens AI — Calibration Fairness: Reweighted vs Baseline",
    "=" * 72,
    "",
    "Checkpoints compared:",
    "  Baseline   : checkpoints/best_efficientnet_b0.pth",
    "  Reweighted : checkpoints/best_efficientnet_b0_reweighted.pth",
    "  Test set   : data/processed/test_with_ita.csv  (same for both)",
    "",
    "=" * 72,
    "  (a) Did overall accuracy drop, and by how much?",
    "=" * 72,
    "",
    f"  Full-model accuracy (light+intermediate+dark groups combined):",
    f"    Baseline   : {base_overall_acc:.4f}",
    f"    Reweighted : {overall_acc_rw:.4f}",
    f"    Change     : {overall_acc_rw - base_overall_acc:+.4f}",
    "",
    f"  Per-group accuracy:",
    f"    {'Group':13s}  {'Baseline':>10s}  {'Reweighted':>10s}  {'Change':>8s}",
    "    " + "-" * 50,
    f"    {'light':13s}  {base_acc_light:>10.4f}  {rw_acc_light:>10.4f}  {rw_acc_light - base_acc_light:>+8.4f}",
    f"    {'intermediate':13s}  {get_group_metric(base_cal,'intermediate','accuracy'):>10.4f}  {get_group_metric(cal_df,'intermediate','accuracy'):>10.4f}  {get_group_metric(cal_df,'intermediate','accuracy') - get_group_metric(base_cal,'intermediate','accuracy'):>+8.4f}",
    f"    {'dark':13s}  {base_acc_dark:>10.4f}  {rw_acc_dark:>10.4f}  {rw_acc_dark - base_acc_dark:>+8.4f}",
    "",
]

# Verdict (a)
acc_change = overall_acc_rw - base_overall_acc
if abs(acc_change) < 0.005:
    lines.append("  VERDICT (a): Overall accuracy is ESSENTIALLY UNCHANGED (|change| < 0.5%).")
elif acc_change < 0:
    lines.append(f"  VERDICT (a): Overall accuracy DROPPED by {abs(acc_change)*100:.2f} percentage points.")
else:
    lines.append(f"  VERDICT (a): Overall accuracy IMPROVED by {acc_change*100:.2f} percentage points.")

lines += [
    "",
    "=" * 72,
    "  (b) Did the nv-class dark-vs-light Brier gap shrink, stay, or grow?",
    "=" * 72,
    "",
    "  Overall Brier score (dark - light gap, positive = dark is worse):",
    f"    Baseline   dark Brier  : {base_brier_dark:.4f}  |  light Brier: {base_brier_light:.4f}  |  gap: {base_gap_brier:+.4f}",
    f"    Reweighted dark Brier  : {rw_brier_dark:.4f}  |  light Brier: {rw_brier_light:.4f}  |  gap: {rw_gap_brier:+.4f}",
    f"    Gap change             : {rw_gap_brier - base_gap_brier:+.4f}",
    "",
]

if not math.isnan(nv_gap_base) and not math.isnan(nv_gap_rw):
    lines += [
        "  nv-CLASS Brier gap (dark - light), the PRIMARY fairness metric:",
        f"    Baseline   : {nv_gap_base:+.4f}  (CI=[{-nv_ci_base[1]:.4f},{-nv_ci_base[0]:.4f}], sig={nv_sig_base})",
        f"    Reweighted : {nv_gap_rw:+.4f}  (CI=[{-nv_ci_rw[1]:.4f},{-nv_ci_rw[0]:.4f}], sig={nv_sig_rw})",
        f"    Gap change : {nv_gap_change:+.4f}",
        "",
    ]
    if nv_gap_change < -0.01:
        lines.append(f"  VERDICT (b): nv-class dark-vs-light Brier gap SHRANK by {abs(nv_gap_change):.4f}.")
    elif nv_gap_change > 0.01:
        lines.append(f"  VERDICT (b): nv-class dark-vs-light Brier gap GREW by {nv_gap_change:.4f}.")
    else:
        lines.append("  VERDICT (b): nv-class dark-vs-light Brier gap is LARGELY UNCHANGED (|change| < 0.01).")
else:
    lines.append("  VERDICT (b): nv class not available in reweighted stratified analysis.")

lines += [
    "",
    "=" * 72,
    "  (c) Is the change statistically meaningful?",
    "=" * 72,
    "",
    "  The baseline and reweighted analyses each produce 95% bootstrap CIs.",
    "  We compare the gap confidence intervals to assess overlap/separation:",
    "",
]

if not math.isnan(nv_gap_base) and not math.isnan(nv_gap_rw):
    base_ci_lo_gap = -nv_ci_base[1]
    base_ci_hi_gap = -nv_ci_base[0]
    rw_ci_lo_gap   = -nv_ci_rw[1]
    rw_ci_hi_gap   = -nv_ci_rw[0]

    lines += [
        f"  Baseline   nv gap 95% CI: [{base_ci_lo_gap:.4f}, {base_ci_hi_gap:.4f}]  significant={nv_sig_base}",
        f"  Reweighted nv gap 95% CI: [{rw_ci_lo_gap:.4f}, {rw_ci_hi_gap:.4f}]  significant={nv_sig_rw}",
        "",
    ]

    ci_overlap = min(base_ci_hi_gap, rw_ci_hi_gap) - max(base_ci_lo_gap, rw_ci_lo_gap)
    if nv_sig_base and nv_sig_rw:
        lines.append("  Both gaps are individually statistically significant.")
        if ci_overlap < 0:
            lines.append(
                "  The two CIs do NOT overlap — the gap change IS statistically meaningful."
                " Reweighting produced a significant shift in the nv calibration disparity."
            )
        else:
            lines.append(
                f"  The two CIs OVERLAP (overlap = {ci_overlap:.4f})."
                " The gap change is NOT conclusively significant — the difference could"
                " be sampling variability alone."
            )
    elif nv_sig_base and not nv_sig_rw:
        lines += [
            "  Baseline gap was significant; reweighted gap is NOT significant.",
            "  This suggests reweighting DID reduce the disparity to below statistical",
            "  significance threshold, though CI overlap should be checked.",
        ]
    elif not nv_sig_base and not nv_sig_rw:
        lines.append("  Neither gap is individually significant — results are inconclusive.")
    else:
        lines.append("  Reweighted gap is significant but baseline was not — unexpected; review data.")

lines += [
    "",
    "  NOTE: This is a within-checkpoint comparison, not a formal two-sample",
    "  test of gap-change significance. For a definitive conclusion, a direct",
    "  permutation test on (gap_rw - gap_baseline) would be needed, requiring",
    "  storing per-sample predictions from both runs simultaneously.",
    "",
    "=" * 72,
    "  Output files (all in results/calibration_fairness_reweighted/):",
    "=" * 72,
    "  calibration_by_ita_group.csv    — per-group ECE/MCE/Brier/accuracy",
    "  significance_tests.csv          — matched bootstrap significance",
    "  class_stratified_analysis.csv   — nv-class within-group comparison",
    "  before_vs_after_comparison.csv  — side-by-side baseline vs reweighted",
    "  comparison_report.txt           — this file",
    "=" * 72,
]

report_path = OUT_DIR / "comparison_report.txt"
with open(report_path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")

print("\n".join(lines))
print(f"\n  Report written → {report_path}")
print("Done.")
