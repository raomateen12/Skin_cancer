"""
DermaLens AI — ITA Vignette/Border Artifact Diagnostic
src/ita_vignette_check.py

Checks whether extreme negative ITA values in the 'dark' group are driven
by dermatoscope vignetting (black borders hitting corner patches) rather
than genuine dark skin tone.

Tasks:
  1. Per-image corner-patch L* statistics for all test images
  2. Vignette flags + group/source aggregation -> vignette_diagnostic.csv
  3. Cleaned Brier comparison (nv dark vs light) excluding flagged images
  4. Corner-crop zoomed visualizations for the 12 validation images

Run from project root:
    python src/ita_vignette_check.py
"""

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

# ─── Import shared utilities from calibration_fairness ────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.calibration_fairness import (
    run_inference,
    brier_score_multiclass,
    class_to_idx,
    CHECKPOINT_TEMPLATE,
    N_BOOTSTRAP,
)

# ─── Paths / constants ────────────────────────────────────────────────────────
ITA_CSV   = Path("data/processed/test_with_ita.csv")
OUT_DIR   = Path("results/calibration_fairness")
CROPS_DIR = OUT_DIR / "corner_crops_zoomed"
PATCH_SZ  = 20                  # same as used when computing ITA
VIGNETTE_L_THRESH   = 30.0      # avg_corner_mean_L < 30 => likely vignette
INTER_CORNER_THRESH = 40.0      # max_corner - min_corner L* > 40 => inconsistent
N_BOOT    = N_BOOTSTRAP         # 1000
SEED      = 42
CROP_SIZE = 150                 # zoomed patch display size (px)

OUT_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def source_from_path(p: str) -> str:
    p = str(p)
    if "part_1" in p:
        return "rosendahl"
    if "part_2" in p:
        return "vienna"
    return "unknown"


def corner_boxes(h: int, w: int, p: int) -> list[tuple]:
    """(r0, r1, c0, c1) for each of the 4 corners."""
    return [
        (0,    p,    0,    p),
        (0,    p,    w-p,  w),
        (h-p,  h,    0,    p),
        (h-p,  h,    w-p,  w),
    ]


def bgr_patch_to_L_mean(patch_bgr: np.ndarray) -> float:
    """Mean L* (0-100 scale) of a BGR patch."""
    lab = cv2.cvtColor(
        patch_bgr.reshape(1, -1, 3).astype(np.uint8),
        cv2.COLOR_BGR2LAB,
    )[0]
    # OpenCV Lab: L in [0, 255], scale to [0, 100]
    return float(lab[:, 0].mean()) / 255.0 * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load ITA CSV + add source
# ─────────────────────────────────────────────────────────────────────────────

print("Loading ITA CSV …")
df = pd.read_csv(ITA_CSV)
df["source"] = df["image_path"].apply(source_from_path)
print(f"  {len(df)} rows  groups={dict(df['ita_group'].value_counts())}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Per-image corner L* statistics (all 1002 images)
# ─────────────────────────────────────────────────────────────────────────────

print("\nComputing corner-patch L* statistics for all images …")
records = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Corner stats"):
    img_path = str(row["image_path"])
    if not Path(img_path).exists():
        records.append({
            "image_id":           row["image_id"],
            "corner_L_means":     [float("nan")] * 4,
            "avg_corner_mean_L":  float("nan"),
            "min_corner_mean_L":  float("nan"),
            "max_corner_mean_L":  float("nan"),
            "inter_corner_L_gap": float("nan"),
            "read_ok":            False,
        })
        continue

    img = cv2.imread(img_path)
    if img is None:
        records.append({
            "image_id":           row["image_id"],
            "corner_L_means":     [float("nan")] * 4,
            "avg_corner_mean_L":  float("nan"),
            "min_corner_mean_L":  float("nan"),
            "max_corner_mean_L":  float("nan"),
            "inter_corner_L_gap": float("nan"),
            "read_ok":            False,
        })
        continue

    h, w = img.shape[:2]
    boxes = corner_boxes(h, w, PATCH_SZ)
    L_means = []
    for (r0, r1, c0, c1) in boxes:
        patch = img[r0:r1, c0:c1]
        if patch.size == 0:
            L_means.append(float("nan"))
        else:
            L_means.append(bgr_patch_to_L_mean(patch))

    valid = [x for x in L_means if not math.isnan(x)]
    avg_L = float(np.mean(valid))  if valid else float("nan")
    min_L = float(min(valid))      if valid else float("nan")
    max_L = float(max(valid))      if valid else float("nan")
    gap_L = (max_L - min_L)        if valid else float("nan")

    records.append({
        "image_id":           row["image_id"],
        "corner_L_means":     L_means,
        "avg_corner_mean_L":  round(avg_L, 2),
        "min_corner_mean_L":  round(min_L, 2),
        "max_corner_mean_L":  round(max_L, 2),
        "inter_corner_L_gap": round(gap_L, 2),
        "read_ok":            True,
    })

stat_df = pd.DataFrame(records).drop(columns=["corner_L_means"])
df = df.merge(stat_df, on="image_id", how="left")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Vignette flags
# ─────────────────────────────────────────────────────────────────────────────

df["flag_low_L"]     = df["avg_corner_mean_L"] < VIGNETTE_L_THRESH
df["flag_gap"]       = df["inter_corner_L_gap"] > INTER_CORNER_THRESH
df["likely_vignette_artifact"] = df["flag_low_L"] | df["flag_gap"]

print(f"\nVignette flags:")
print(f"  flag_low_L  (avg_L < {VIGNETTE_L_THRESH}): {df['flag_low_L'].sum()}")
print(f"  flag_gap    (gap  > {INTER_CORNER_THRESH}): {df['flag_gap'].sum()}")
print(f"  combined    (either flag):          {df['likely_vignette_artifact'].sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Aggregate by ita_group and source
# ─────────────────────────────────────────────────────────────────────────────

def group_summary(df_sub: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for grp, g in df_sub.groupby(group_col):
        n        = len(g)
        n_flag   = int(g["likely_vignette_artifact"].sum())
        pct_flag = round(100.0 * n_flag / n, 1) if n > 0 else 0.0
        mean_avg = round(float(g["avg_corner_mean_L"].mean()), 2)
        mean_gap = round(float(g["inter_corner_L_gap"].mean()), 2)
        rows.append({
            group_col:             grp,
            "n":                   n,
            "n_flagged":           n_flag,
            "pct_flagged":         pct_flag,
            "mean_avg_corner_L":   mean_avg,
            "mean_inter_corner_gap": mean_gap,
        })
    return pd.DataFrame(rows)

by_ita    = group_summary(df, "ita_group")
by_source = group_summary(df, "source")

print("\n  By ITA group:")
print(by_ita.to_string(index=False))
print("\n  By source:")
print(by_source.to_string(index=False))

# Also cross-tabulate: ita_group × source
cross_rows = []
for ita_grp, g1 in df.groupby("ita_group"):
    for src, g2 in g1.groupby("source"):
        n      = len(g2)
        n_flag = int(g2["likely_vignette_artifact"].sum())
        cross_rows.append({
            "ita_group": ita_grp, "source": src,
            "n": n, "n_flagged": n_flag,
            "pct_flagged": round(100.0 * n_flag / n, 1) if n > 0 else 0.0,
            "mean_avg_corner_L":   round(float(g2["avg_corner_mean_L"].mean()), 2),
            "mean_inter_corner_gap": round(float(g2["inter_corner_L_gap"].mean()), 2),
        })
cross_df = pd.DataFrame(cross_rows)
print("\n  Cross (ita_group x source):")
print(cross_df.to_string(index=False))

# Save full diagnostic CSV
diag_path = OUT_DIR / "vignette_diagnostic.csv"
df[[
    "image_id", "dx", "ita_group", "ita_value", "source",
    "avg_corner_mean_L", "min_corner_mean_L", "max_corner_mean_L",
    "inter_corner_L_gap", "flag_low_L", "flag_gap", "likely_vignette_artifact",
]].to_csv(diag_path, index=False)
print(f"\n  Full diagnostic CSV -> {diag_path}")

# Save summary CSVs
(OUT_DIR / "vignette_summary_by_ita_group.csv").write_text(
    by_ita.to_csv(index=False), encoding="utf-8"
)
(OUT_DIR / "vignette_summary_by_source.csv").write_text(
    by_source.to_csv(index=False), encoding="utf-8"
)
(OUT_DIR / "vignette_summary_cross.csv").write_text(
    cross_df.to_csv(index=False), encoding="utf-8"
)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: NV dark group — flagged fraction + cleaned Brier comparison
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("  Cleaned Brier comparison: nv dark vs light (excluding flagged)")
print("=" * 72)

df_nv = df[df["dx"] == "nv"].copy()
dark_nv  = df_nv[df_nv["ita_group"] == "dark"]
light_nv = df_nv[df_nv["ita_group"] == "light"]

n_dark_total   = len(dark_nv)
n_dark_flagged = int(dark_nv["likely_vignette_artifact"].sum())
n_light_total  = len(light_nv)
n_light_flagged= int(light_nv["likely_vignette_artifact"].sum())

print(f"\n  NV dark : {n_dark_total} total, {n_dark_flagged} flagged"
      f" ({100*n_dark_flagged/n_dark_total:.1f}%)")
print(f"  NV light: {n_light_total} total, {n_light_flagged} flagged"
      f" ({100*n_light_flagged/n_light_total:.1f}%)")

dark_clean  = dark_nv[~dark_nv["likely_vignette_artifact"]]
light_clean = light_nv[~light_nv["likely_vignette_artifact"]]
print(f"\n  After cleaning — dark: {len(dark_clean)}, light: {len(light_clean)}")

if len(dark_clean) < 10 or len(light_clean) < 10:
    print("  Too few samples after cleaning — skipping cleaned bootstrap.")
else:
    # Run inference for full test set
    print("\n  Running model inference (needed for cleaned Brier scores) …")
    model_name      = "efficientnet_b0"
    checkpoint_path = CHECKPOINT_TEMPLATE.format(model_name=model_name)
    num_classes     = len(class_to_idx)

    probs, preds, labels = run_inference(df, model_name, checkpoint_path, num_classes, batch_size=32)

    # Original (unfiltered) Brier for nv dark vs light
    dark_mask_orig  = df["ita_group"].values == "dark"
    light_mask_orig = df["ita_group"].values == "light"
    nv_mask         = df["dx"].values == "nv"

    dark_nv_mask_orig  = dark_mask_orig  & nv_mask
    light_nv_mask_orig = light_mask_orig & nv_mask

    brier_dark_orig  = brier_score_multiclass(probs[dark_nv_mask_orig],  labels[dark_nv_mask_orig],  num_classes)
    brier_light_orig = brier_score_multiclass(probs[light_nv_mask_orig], labels[light_nv_mask_orig], num_classes)
    print(f"\n  Original (all nv, unfiltered):")
    print(f"    dark  Brier = {brier_dark_orig:.4f}  (n={dark_nv_mask_orig.sum()})")
    print(f"    light Brier = {brier_light_orig:.4f}  (n={light_nv_mask_orig.sum()})")
    print(f"    diff  = {brier_dark_orig - brier_light_orig:.4f}")

    # Cleaned masks
    clean_image_ids_dark  = set(dark_clean["image_id"].values)
    clean_image_ids_light = set(light_clean["image_id"].values)
    dark_clean_mask  = nv_mask & dark_mask_orig  & df["image_id"].isin(clean_image_ids_dark).values
    light_clean_mask = nv_mask & light_mask_orig & df["image_id"].isin(clean_image_ids_light).values

    probs_dark_c  = probs[dark_clean_mask]
    labels_dark_c = labels[dark_clean_mask]
    probs_light_c = probs[light_clean_mask]
    labels_light_c= labels[light_clean_mask]

    brier_dark_c  = brier_score_multiclass(probs_dark_c,  labels_dark_c,  num_classes)
    brier_light_c = brier_score_multiclass(probs_light_c, labels_light_c, num_classes)
    print(f"\n  Cleaned (vignette-flagged images excluded):")
    print(f"    dark  Brier = {brier_dark_c:.4f}  (n={dark_clean_mask.sum()})")
    print(f"    light Brier = {brier_light_c:.4f}  (n={light_clean_mask.sum()})")
    print(f"    diff  = {brier_dark_c - brier_light_c:.4f}")

    # Matched-size bootstrap for cleaned comparison
    rng      = np.random.default_rng(SEED + 10)
    n_min_c  = min(len(labels_dark_c), len(labels_light_c))
    iter_b_dark, iter_b_light = [], []

    for _ in range(N_BOOT):
        idx_d = rng.choice(len(labels_dark_c),  size=n_min_c, replace=True)
        idx_l = rng.choice(len(labels_light_c), size=n_min_c, replace=True)
        iter_b_dark.append( brier_score_multiclass(probs_dark_c[idx_d],  labels_dark_c[idx_d],  num_classes))
        iter_b_light.append(brier_score_multiclass(probs_light_c[idx_l], labels_light_c[idx_l], num_classes))

    diff_dist = np.array(iter_b_dark) - np.array(iter_b_light)
    obs_diff  = float(diff_dist.mean())
    ci_lo     = float(np.percentile(diff_dist, 2.5))
    ci_hi     = float(np.percentile(diff_dist, 97.5))
    sig       = (ci_lo > 0) or (ci_hi < 0)

    print(f"\n  Cleaned matched-bootstrap (N_min={n_min_c}, 1000 iters, with replacement):")
    print(f"    dark mean Brier  = {np.mean(iter_b_dark):.4f}")
    print(f"    light mean Brier = {np.mean(iter_b_light):.4f}")
    print(f"    diff (dark-light) = {obs_diff:.4f}")
    print(f"    95% CI            = [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"    Significant?      = {'YES — gap PERSISTS' if sig else 'NO  — gap DISAPPEARS'}")

    # Also run original bootstrap for direct comparison
    n_min_o = min(dark_nv_mask_orig.sum(), light_nv_mask_orig.sum())
    probs_dark_o  = probs[dark_nv_mask_orig]
    labels_dark_o = labels[dark_nv_mask_orig]
    probs_light_o = probs[light_nv_mask_orig]
    labels_light_o= labels[light_nv_mask_orig]
    iter_o_dark, iter_o_light = [], []
    for _ in range(N_BOOT):
        idx_d = rng.choice(len(labels_dark_o),  size=n_min_o, replace=True)
        idx_l = rng.choice(len(labels_light_o), size=n_min_o, replace=True)
        iter_o_dark.append( brier_score_multiclass(probs_dark_o[idx_d],  labels_dark_o[idx_d],  num_classes))
        iter_o_light.append(brier_score_multiclass(probs_light_o[idx_l], labels_light_o[idx_l], num_classes))

    diff_o    = np.array(iter_o_dark) - np.array(iter_o_light)
    obs_o     = float(diff_o.mean())
    ci_lo_o   = float(np.percentile(diff_o, 2.5))
    ci_hi_o   = float(np.percentile(diff_o, 97.5))
    sig_o     = (ci_lo_o > 0) or (ci_hi_o < 0)
    print(f"\n  Original (unfiltered) bootstrap for direct comparison:")
    print(f"    diff = {obs_o:.4f}  CI=[{ci_lo_o:.4f}, {ci_hi_o:.4f}]  sig={sig_o}")

    # Save cleaned CSV
    cleaned_df = pd.DataFrame([{
        "dx_class":    "nv",
        "comparison":  "dark vs light",
        "filter":      "none (original)",
        "n_dark":       dark_nv_mask_orig.sum(),
        "n_light":      light_nv_mask_orig.sum(),
        "n_min":        n_min_o,
        "brier_dark":   round(brier_dark_orig, 6),
        "brier_light":  round(brier_light_orig, 6),
        "brier_diff":   round(obs_o, 6),
        "ci_lower":     round(ci_lo_o, 6),
        "ci_upper":     round(ci_hi_o, 6),
        "significant":  sig_o,
    }, {
        "dx_class":    "nv",
        "comparison":  "dark vs light",
        "filter":      f"exclude_vignette (low_L<{VIGNETTE_L_THRESH} OR gap>{INTER_CORNER_THRESH})",
        "n_dark":       dark_clean_mask.sum(),
        "n_light":      light_clean_mask.sum(),
        "n_min":        n_min_c,
        "brier_dark":   round(brier_dark_c, 6),
        "brier_light":  round(brier_light_c, 6),
        "brier_diff":   round(obs_diff, 6),
        "ci_lower":     round(ci_lo, 6),
        "ci_upper":     round(ci_hi, 6),
        "significant":  sig,
    }])
    cleaned_path = OUT_DIR / "class_stratified_analysis_cleaned.csv"
    cleaned_df.to_csv(cleaned_path, index=False)
    print(f"\n  Cleaned comparison CSV -> {cleaned_path}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Corner-crop zoomed visualizations (same 12 images as before)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("  Corner-crop zoomed visualizations")
print("=" * 72)

rng_vis  = np.random.default_rng(SEED)
dark_nv_r  = df[(df["dx"] == "nv") & (df["ita_group"] == "dark")].reset_index(drop=True)
light_nv_r = df[(df["dx"] == "nv") & (df["ita_group"] == "light")].reset_index(drop=True)

di_v = rng_vis.choice(len(dark_nv_r),  size=min(6, len(dark_nv_r)),  replace=False)
li_v = rng_vis.choice(len(light_nv_r), size=min(6, len(light_nv_r)), replace=False)

vis_samples = (
    [("dark",  i+1, dark_nv_r.iloc[x])  for i, x in enumerate(di_v)] +
    [("light", i+1, light_nv_r.iloc[x]) for i, x in enumerate(li_v)]
)

CORNER_NAMES = ["top-left", "top-right", "bot-left", "bot-right"]
THUMB_SIZE   = (200, 200)

for grp_lbl, idx, row in vis_samples:
    img_path = str(row["image_path"])
    image_id = str(row["image_id"])
    ita_val  = float(row.get("ita_value", float("nan")))
    flagged  = bool(row.get("likely_vignette_artifact", False))
    avg_L    = row.get("avg_corner_mean_L", float("nan"))
    gap_L    = row.get("inter_corner_L_gap", float("nan"))

    if not Path(img_path).exists():
        print(f"  MISSING: {img_path}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    boxes = corner_boxes(h, w, PATCH_SZ)

    # Compute per-corner L* for labelling
    corner_L = []
    for (r0, r1, c0, c1) in boxes:
        patch = img[r0:r1, c0:c1]
        corner_L.append(bgr_patch_to_L_mean(patch) if patch.size > 0 else float("nan"))

    # Layout: 1 thumbnail (left) + 2×2 grid of zoomed patches (right)
    fig = plt.figure(figsize=(9, 5))
    fig.patch.set_facecolor("#1a1a1a")

    # Main thumbnail (left half)
    ax_thumb = fig.add_axes([0.02, 0.08, 0.38, 0.84])
    ax_thumb.imshow(img_rgb)
    # Draw corner boxes on thumbnail
    for (r0, r1, c0, c1) in boxes:
        ax_thumb.add_patch(patches.Rectangle(
            (c0, r0), c1-c0, r1-r0,
            linewidth=2, edgecolor="lime", facecolor="none"
        ))
    flag_color = "#ff4444" if flagged else "#44ff44"
    ax_thumb.set_title(
        f"{image_id}\n"
        f"ITA={ita_val:.1f}  group={grp_lbl}\n"
        f"avg_L={avg_L:.1f}  gap={gap_L:.1f}  "
        f"{'[FLAGGED]' if flagged else '[OK]'}",
        color=flag_color, fontsize=7.5, fontweight="bold",
    )
    ax_thumb.axis("off")

    # 2×2 grid of zoomed corner patches (right half)
    positions = [(0.44, 0.50), (0.72, 0.50), (0.44, 0.06), (0.72, 0.06)]
    for ci, ((r0, r1, c0, c1), name, l_val, pos) in enumerate(
        zip(boxes, CORNER_NAMES, corner_L, positions)
    ):
        patch_bgr = img[r0:r1, c0:c1]
        patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
        # Resize to CROP_SIZE x CROP_SIZE
        patch_big = cv2.resize(patch_rgb, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_NEAREST)
        ax_p = fig.add_axes([pos[0], pos[1], 0.25, 0.42])
        ax_p.imshow(patch_big, aspect="auto")
        l_color = "#ff6666" if l_val < VIGNETTE_L_THRESH else "#ffffff"
        ax_p.set_title(f"{name}\nL*={l_val:.1f}", color=l_color, fontsize=8, pad=2)
        ax_p.axis("off")
        # Red border if this corner's L* < threshold
        if l_val < VIGNETTE_L_THRESH:
            for spine in ax_p.spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(3)

    out_name = f"{grp_lbl}_{idx:02d}_corners.png"
    fig.savefig(CROPS_DIR / out_name, dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    flagged_str = " [FLAGGED]" if flagged else ""
    print(f"  {out_name}  ITA={ita_val:.1f}  avg_L={avg_L:.1f}  gap={gap_L:.1f}{flagged_str}")

print(f"\n  Corner crop images -> {CROPS_DIR}")
print("\nDone.")
