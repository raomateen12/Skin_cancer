"""
DermaLens AI — ITA Formula-Degeneracy & Source-Stratified Brier Diagnostic
src/ita_formula_source_check.py

Diagnostics:
  1. b*-degeneracy: Pearson corr( |b*| , |ITA| ) — near-zero b* causes
     extreme ITA independent of true darkness.
  2. Source-stratified Brier (nv): Rosendahl-only and Vienna-only dark vs
     light matched bootstrap — tests whether gap is institution-driven.
  3. Appends "Formula and Source Stratification" section to the report.

Run from project root:
    python src/ita_formula_source_check.py
"""

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.calibration_fairness import (
    run_inference,
    brier_score_multiclass,
    class_to_idx,
    CHECKPOINT_TEMPLATE,
    N_BOOTSTRAP,
)

# ─── Paths / constants ────────────────────────────────────────────────────────
ITA_CSV     = Path("data/processed/test_with_ita.csv")
REPORT_PATH = Path("results/calibration_fairness/calibration_fairness_report.txt")
OUT_DIR     = Path("results/calibration_fairness")
PATCH_SZ    = 20
B_STAR_THRESH = 5.0   # |b*| < 5 → ITA formula unstable
N_BOOT        = N_BOOTSTRAP   # 1000
SEED          = 42
MIN_GROUP_N   = 15    # minimum per group for bootstrap

OUT_DIR.mkdir(parents=True, exist_ok=True)


def source_from_path(p: str) -> str:
    p = str(p)
    if "part_1" in p:
        return "rosendahl"
    if "part_2" in p:
        return "vienna"
    return "unknown"


def corner_boxes(h: int, w: int, p: int) -> list[tuple]:
    return [
        (0,    p,    0,    p),
        (0,    p,    w-p,  w),
        (h-p,  h,    0,    p),
        (h-p,  h,    w-p,  w),
    ]


def bgr_patch_to_Lab_means(patch_bgr: np.ndarray) -> tuple[float, float, float]:
    """Returns (mean_L, mean_a_star, mean_b_star) in standard CIE range."""
    lab = cv2.cvtColor(
        patch_bgr.reshape(1, -1, 3).astype(np.uint8),
        cv2.COLOR_BGR2LAB,
    )[0]
    # OpenCV Lab: L in [0,255]->scale to [0,100], a/b in [0,255]->subtract 128
    L  = float(lab[:, 0].mean()) / 255.0 * 100.0
    a  = float(lab[:, 1].mean()) - 128.0
    b  = float(lab[:, 2].mean()) - 128.0
    return L, a, b


# ─────────────────────────────────────────────────────────────────────────────
# Load ITA CSV + source
# ─────────────────────────────────────────────────────────────────────────────
print("Loading ITA CSV …")
df = pd.read_csv(ITA_CSV)
df["source"] = df["image_path"].apply(source_from_path)
print(f"  {len(df)} rows  ita_groups={dict(df['ita_group'].value_counts())}")

# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC 1 — b* per-corner extraction for all images
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("  DIAGNOSTIC 1 — b* Formula-Degeneracy Check")
print("="*72)
print(f"\nExtracting corner Lab stats for all {len(df)} images …")

records = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="  Lab stats"):
    img_path = str(row["image_path"])
    image_id = str(row["image_id"])
    ita_val  = float(row.get("ita_value", float("nan")))
    ita_grp  = str(row["ita_group"])

    if not Path(img_path).exists():
        records.append({"image_id": image_id, "mean_L": float("nan"),
                         "mean_a_star": float("nan"), "mean_b_star": float("nan"),
                         "read_ok": False})
        continue

    img = cv2.imread(img_path)
    if img is None:
        records.append({"image_id": image_id, "mean_L": float("nan"),
                         "mean_a_star": float("nan"), "mean_b_star": float("nan"),
                         "read_ok": False})
        continue

    h, w = img.shape[:2]
    Ls, a_stars, b_stars = [], [], []
    for (r0, r1, c0, c1) in corner_boxes(h, w, PATCH_SZ):
        patch = img[r0:r1, c0:c1]
        if patch.size == 0:
            continue
        L, a, b = bgr_patch_to_Lab_means(patch)
        Ls.append(L); a_stars.append(a); b_stars.append(b)

    if not Ls:
        records.append({"image_id": image_id, "mean_L": float("nan"),
                         "mean_a_star": float("nan"), "mean_b_star": float("nan"),
                         "read_ok": False})
        continue

    records.append({
        "image_id":    image_id,
        "mean_L":      round(float(np.mean(Ls)), 3),
        "mean_a_star": round(float(np.mean(a_stars)), 3),
        "mean_b_star": round(float(np.mean(b_stars)), 3),
        "read_ok":     True,
    })

lab_df = pd.DataFrame(records)
df = df.merge(lab_df, on="image_id", how="left")

# Flag formula instability
df["abs_mean_b_star"] = df["mean_b_star"].abs()
df["ita_formula_unstable"] = df["abs_mean_b_star"] < B_STAR_THRESH

# ── Per-group statistics ──────────────────────────────────────────────────────
print("\n  Per-group |b*| and formula-instability rate:")
grp_stats = []
for grp in ["light", "intermediate", "dark", "unknown"]:
    g = df[df["ita_group"] == grp]
    if len(g) == 0:
        continue
    mean_abs_b = round(float(g["abs_mean_b_star"].mean()), 3)
    mean_L_val = round(float(g["mean_L"].mean()), 3)
    n_unstable = int(g["ita_formula_unstable"].sum())
    pct_unstable = round(100.0 * n_unstable / len(g), 1)
    mean_ita_abs = round(float(g["ita_value"].abs().mean()), 2)
    grp_stats.append({
        "ita_group":          grp,
        "n":                  len(g),
        "mean_L":             mean_L_val,
        "mean_abs_b_star":    mean_abs_b,
        "mean_abs_ITA":       mean_ita_abs,
        "n_formula_unstable": n_unstable,
        "pct_formula_unstable": pct_unstable,
    })

grp_stats_df = pd.DataFrame(grp_stats)
print(grp_stats_df.to_string(index=False))

# ── Pearson correlation |b*| vs |ITA| ─────────────────────────────────────────
valid = df[df["read_ok"] & df["ita_value"].notna() & df["mean_b_star"].notna()].copy()
valid = valid[valid["ita_group"] != "unknown"]  # exclude unknown (degenerate)

abs_b   = valid["abs_mean_b_star"].values
abs_ita = valid["ita_value"].abs().values

r, p_val = pearsonr(abs_b, abs_ita)
print(f"\n  Pearson r( |b*| , |ITA| ) = {r:.4f}  (p={p_val:.6f})")
if r < -0.3:
    verdict = "NEGATIVE correlation — small |b*| drives large |ITA|: DEGENERACY CONFIRMED"
elif r < 0:
    verdict = "weak negative correlation — mild degeneracy signal"
else:
    verdict = "non-negative — b*-degeneracy NOT the primary driver of extreme ITA"
print(f"  Interpretation: {verdict}")

# Also report per source
for src in ["rosendahl", "vienna"]:
    sub = valid[valid["source"] == src]
    if len(sub) < 10:
        continue
    r_s, p_s = pearsonr(sub["abs_mean_b_star"].values, sub["ita_value"].abs().values)
    print(f"  Pearson r ({src:<10s}): {r_s:.4f}  (p={p_s:.6f})")

# ── Save per-image CSV ────────────────────────────────────────────────────────
formula_csv = OUT_DIR / "ita_formula_diagnostic.csv"
df[[
    "image_id", "ita_group", "ita_value", "mean_L",
    "mean_a_star", "mean_b_star", "abs_mean_b_star", "ita_formula_unstable",
]].to_csv(formula_csv, index=False)
print(f"\n  Formula diagnostic CSV -> {formula_csv}")

# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC 2 — Source-Stratified Brier Comparison (nv class)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("  DIAGNOSTIC 2 — Source-Stratified Brier Comparison (nv)")
print("="*72)

# Run inference (full test set — needed for Brier)
print("\nRunning model inference …")
model_name      = "efficientnet_b0"
checkpoint_path = CHECKPOINT_TEMPLATE.format(model_name=model_name)
num_classes     = len(class_to_idx)

probs, preds, labels = run_inference(df, model_name, checkpoint_path, num_classes, batch_size=32)

rng = np.random.default_rng(SEED)
df_idx = df.reset_index(drop=True)  # align with probs/preds/labels

def run_bootstrap_brier(
    probs_a, labels_a, probs_b, labels_b, n_boot, rng, num_classes
) -> dict:
    """Matched-size with-replacement bootstrap for Brier diff (a - b)."""
    n_min = min(len(labels_a), len(labels_b))
    iter_a, iter_b = [], []
    for _ in range(n_boot):
        idx_a = rng.choice(len(labels_a), size=n_min, replace=True)
        idx_b = rng.choice(len(labels_b), size=n_min, replace=True)
        iter_a.append(brier_score_multiclass(probs_a[idx_a], labels_a[idx_a], num_classes))
        iter_b.append(brier_score_multiclass(probs_b[idx_b], labels_b[idx_b], num_classes))
    diff_dist = np.array(iter_a) - np.array(iter_b)
    obs       = float(diff_dist.mean())
    ci_lo     = float(np.percentile(diff_dist, 2.5))
    ci_hi     = float(np.percentile(diff_dist, 97.5))
    sig       = bool((ci_lo > 0) or (ci_hi < 0))
    return {
        "n_min":         n_min,
        "brier_a_raw":   round(brier_score_multiclass(probs_a, labels_a, num_classes), 4),
        "brier_b_raw":   round(brier_score_multiclass(probs_b, labels_b, num_classes), 4),
        "diff":          round(obs, 4),
        "ci_lower":      round(ci_lo, 4),
        "ci_upper":      round(ci_hi, 4),
        "significant":   sig,
    }

result_rows = []

print()
for src in ["rosendahl", "vienna"]:
    print(f"\n  ── {src.upper()} — nv dark vs light ─────────────────────────────────")

    for grp_a, grp_b in [("dark", "light"), ("dark", "intermediate"), ("light", "intermediate")]:
        mask_a = (
            (df_idx["dx"] == "nv") &
            (df_idx["source"] == src) &
            (df_idx["ita_group"] == grp_a)
        ).values
        mask_b = (
            (df_idx["dx"] == "nv") &
            (df_idx["source"] == src) &
            (df_idx["ita_group"] == grp_b)
        ).values

        n_a, n_b = int(mask_a.sum()), int(mask_b.sum())
        brier_a_raw = round(brier_score_multiclass(probs[mask_a], labels[mask_a], num_classes), 4) if n_a > 0 else float("nan")
        brier_b_raw = round(brier_score_multiclass(probs[mask_b], labels[mask_b], num_classes), 4) if n_b > 0 else float("nan")

        print(f"    {grp_a} (n={n_a}, Brier={brier_a_raw}) vs {grp_b} (n={n_b}, Brier={brier_b_raw})")

        if n_a < MIN_GROUP_N or n_b < MIN_GROUP_N:
            reason = []
            if n_a < MIN_GROUP_N: reason.append(f"{grp_a}={n_a}")
            if n_b < MIN_GROUP_N: reason.append(f"{grp_b}={n_b}")
            print(f"    => SKIPPED — too few samples: {', '.join(reason)} (threshold={MIN_GROUP_N})")
            result_rows.append({
                "source": src, "group_a": grp_a, "group_b": grp_b,
                "n_a": n_a, "n_b": n_b,
                "brier_a": brier_a_raw, "brier_b": brier_b_raw,
                "diff": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"),
                "significant": None,
                "note": f"underpowered (min {MIN_GROUP_N})",
            })
            continue

        boot = run_bootstrap_brier(
            probs[mask_a], labels[mask_a],
            probs[mask_b], labels[mask_b],
            N_BOOT, rng, num_classes,
        )
        sig_str = "YES — gap PERSISTS" if boot["significant"] else "NO — gap DISAPPEARS"
        print(
            f"    => diff={boot['diff']:.4f}  "
            f"CI=[{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]  "
            f"{sig_str}"
        )
        result_rows.append({
            "source": src, "group_a": grp_a, "group_b": grp_b,
            "n_a": n_a, "n_b": n_b,
            "brier_a": brier_a_raw, "brier_b": brier_b_raw,
            "diff": boot["diff"], "ci_lower": boot["ci_lower"], "ci_upper": boot["ci_upper"],
            "significant": boot["significant"],
            "note": "",
        })

results_df = pd.DataFrame(result_rows)
ss_path = OUT_DIR / "source_stratified_brier.csv"
results_df.to_csv(ss_path, index=False)
print(f"\n  Source-stratified Brier CSV -> {ss_path}")
print(results_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Append section to calibration_fairness_report.txt
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("  Appending to calibration_fairness_report.txt …")
print("="*72)

# Build report lines
new_section = [
    "",
    "="*72,
    "  ADDENDUM: Formula-Degeneracy & Source-Stratified Analysis",
    "="*72,
    "",
    f"── B* FORMULA-DEGENERACY CHECK ────────────────────────────────────────",
    f"  ITA = atan( (L-50) / b* ) × (180/π) — becomes extreme when b* ≈ 0.",
    f"  Threshold for instability: |b*| < {B_STAR_THRESH}",
    "",
    f"  {'Group':13s} {'n':>5s}  {'mean_L':>8s}  {'mean|b*|':>10s}  {'mean|ITA|':>10s}  {'pct_unstable':>12s}",
    "  " + "-" * 65,
]
for _, r in grp_stats_df.iterrows():
    new_section.append(
        f"  {str(r['ita_group']):13s} {int(r['n']):>5d}"
        f"  {r['mean_L']:>8.2f}  {r['mean_abs_b_star']:>10.3f}"
        f"  {r['mean_abs_ITA']:>10.2f}  {r['pct_formula_unstable']:>11.1f}%"
    )

new_section += [
    "",
    f"  Pearson r( |b*| , |ITA| ) = {r_val:.4f}  (p={p_val_r:.6f})"
    if False else  # placeholder — filled below
    f"  Pearson r( |b*| , |ITA| ) = {r:.4f}  (p={p_val:.6f})",
    f"  Interpretation: {verdict}",
    "",
]

# Degeneracy verdict in plain English
total_unstable = int(df["ita_formula_unstable"].sum())
dark_unstable_pct = grp_stats_df.loc[grp_stats_df["ita_group"]=="dark", "pct_formula_unstable"].values
dark_unstable_pct = float(dark_unstable_pct[0]) if len(dark_unstable_pct) > 0 else 0.0
light_unstable_pct = grp_stats_df.loc[grp_stats_df["ita_group"]=="light", "pct_formula_unstable"].values
light_unstable_pct = float(light_unstable_pct[0]) if len(light_unstable_pct) > 0 else 0.0

new_section += [
    f"  PLAIN ENGLISH: {total_unstable} of {len(df)} images ({100*total_unstable/len(df):.1f}%)"
    f" have |b*| < {B_STAR_THRESH} (formula-unstable zone).",
    f"  Dark group: {dark_unstable_pct:.1f}% unstable vs light: {light_unstable_pct:.1f}% unstable.",
]
if r < -0.3:
    new_section += [
        "  The strong negative correlation between |b*| and |ITA| confirms that",
        "  b*-degeneracy is a contributing driver of extreme negative ITA values",
        "  in the dark group — these may represent pink/erythematous skin being",
        "  mislabelled as 'dark' ITA due to near-zero b* denominator instability,",
        "  not true skin darkness.",
    ]
else:
    new_section += [
        "  b*-degeneracy is NOT the primary driver: the correlation between |b*|",
        "  and |ITA| is weak/non-negative, meaning extreme ITA values arise from",
        "  genuinely low L* (dark corners) rather than formula instability.",
    ]

new_section += [
    "",
    "── SOURCE-STRATIFIED BRIER COMPARISON (nv class only) ──────────────────",
    "  Restricting nv images to single camera source to isolate skin-tone",
    "  effect from institution/equipment confound.",
    "",
    f"  {'Source':12s} {'Pair':25s} {'n_a':>5s} {'n_b':>5s}"
    f" {'Brier_a':>8s} {'Brier_b':>8s} {'diff':>8s}  {'95% CI':>22s}  {'Sig?':>5s}",
    "  " + "-" * 100,
]

for _, r in results_df.iterrows():
    pair_str = f"{r['group_a']} vs {r['group_b']}"
    if r["significant"] is None:
        sig_str = "n/a"
        diff_str = "n/a"
        ci_str = f"[{r['note']}]"
    else:
        sig_str = "YES *" if r["significant"] else "no"
        diff_str = f"{r['diff']:>8.4f}"
        ci_str   = f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"
    brier_a_str = f"{r['brier_a']:.4f}" if not (isinstance(r['brier_a'], float) and math.isnan(r['brier_a'])) else "n/a"
    brier_b_str = f"{r['brier_b']:.4f}" if not (isinstance(r['brier_b'], float) and math.isnan(r['brier_b'])) else "n/a"
    new_section.append(
        f"  {str(r['source']):12s} {pair_str:25s} {int(r['n_a']) if r['n_a']==r['n_a'] else 0:>5d}"
        f" {int(r['n_b']) if r['n_b']==r['n_b'] else 0:>5d}"
        f" {brier_a_str:>8s} {brier_b_str:>8s}"
        f" {diff_str:>8s}  {ci_str:>24s}  {sig_str:>5s}"
    )

# Source-stratified verdict
dark_light_sig = {
    src: None for src in ["rosendahl", "vienna"]
}
for _, r in results_df.iterrows():
    if r["group_a"] == "dark" and r["group_b"] == "light":
        dark_light_sig[r["source"]] = r["significant"]

both_sig = all(v is True  for v in dark_light_sig.values())
none_sig = all(v is False for v in dark_light_sig.values())
one_underpowered = any(v is None for v in dark_light_sig.values())

new_section += ["", "  PLAIN ENGLISH VERDICT:"]
if one_underpowered:
    new_section += [
        "  One or more source subsets were underpowered (<15 samples per group).",
        "  Interpret available results cautiously.",
    ]
if both_sig:
    new_section += [
        "  The dark-vs-light Brier gap in nv PERSISTS in BOTH Rosendahl-only and",
        "  Vienna-only subsets. This strongly suggests a genuine skin-tone/image-",
        "  characteristic effect that is NOT merely an institution/camera artifact.",
        "  The disparity appears across both acquisition sources.",
    ]
elif none_sig:
    new_section += [
        "  The dark-vs-light Brier gap DISAPPEARS when restricting to a single source.",
        "  This suggests the gap is largely explained by the institution/camera",
        "  confound rather than being a genuine skin-tone effect.",
    ]
else:
    new_section += [
        "  The dark-vs-light Brier gap is significant in one source but not the other,",
        "  or one source is underpowered. The finding is SOURCE-DEPENDENT, suggesting",
        "  the institution/equipment confound partially explains the disparity.",
        "  Further source-matched analysis is needed before drawing causal conclusions.",
    ]

for src, sig_val in dark_light_sig.items():
    if sig_val is True:
        new_section.append(f"  ⚠  {src}: dark vs light gap IS significant (gap PERSISTS within this source)")
    elif sig_val is False:
        new_section.append(f"     {src}: dark vs light gap not significant (disappears within this source)")
    else:
        new_section.append(f"     {src}: underpowered — cannot conclude")

new_section += [
    "",
    "  NOTE ON INTERMEDIATE GROUP: intermediate nv subsets within single sources",
    "  may be too small (<15) for reliable comparison — see source_stratified_brier.csv",
    "  for exact counts.",
    "=" * 72,
]

# Append to report
with open(REPORT_PATH, "a", encoding="utf-8") as fh:
    fh.write("\n".join(new_section) + "\n")
print(f"\n  Appended to {REPORT_PATH}")
print("Done.")
