"""
DermaLens AI — Calibration Fairness Diagnostic Script
Tasks:
  1. Acquisition-source confound check (part_1=Rosendahl, part_2=Vienna/ViDir)
  2. ITA visual validation spot-check (12 annotated images + flagged-pixel CSV)

Run from d:/fair-medical-ai:
    python src/calibration_diagnostics.py
"""

import math
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import chi2_contingency

ITA_CSV  = Path("data/processed/test_with_ita.csv")
OUT_DIR  = Path("results/calibration_fairness")
VIS_DIR  = OUT_DIR / "ita_validation_samples"
PATCH_SZ = 20
SEED     = 42

OUT_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ITA_CSV)
print(f"Loaded {len(df)} rows  columns={list(df.columns)}")

# ─── TASK 1 ───────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("  TASK 1 - Acquisition-source confound check")
print("="*72)
print("\n  NOTE: HAM10000_metadata.csv has NO 'dataset' source column.")
print("  Its columns are: lesion_id, image_id, dx, dx_type, age, sex, localization")
print("  Source is derived from image_path: 'part_1' => Rosendahl, 'part_2' => Vienna (ViDir)")

def source_from_path(p):
    p = str(p)
    if "part_1" in p:
        return "rosendahl"
    if "part_2" in p:
        return "vienna"
    return "unknown"

df["source"] = df["image_path"].apply(source_from_path)
print("\n  Source counts:", dict(df["source"].value_counts()))


def run_chi2_and_save(df_sub, row_col, col_col, save_path, label):
    ct = pd.crosstab(df_sub[row_col], df_sub[col_col])
    ct_pct = ct.div(ct.sum(axis=1), axis=0).mul(100).round(1)
    combined = ct.astype(str) + " (" + ct_pct.astype(str) + "%)"
    combined["row_total"] = ct.sum(axis=1).astype(str)
    chi2, p, dof, _ = chi2_contingency(ct)
    sep = ["--- CHI-SQUARE ---"] + [""] * (len(combined.columns) - 1)
    chi_row  = ["chi2"] + [""] * (len(combined.columns) - 2) + [f"{chi2:.4f}"]
    p_row    = ["p_value"] + [""] * (len(combined.columns) - 2) + [f"{p:.6f}"]
    dof_row  = ["dof"] + [""] * (len(combined.columns) - 2) + [str(dof)]
    footer = pd.DataFrame([sep, chi_row, p_row, dof_row], columns=combined.columns)
    out = pd.concat([combined.reset_index(), footer.reset_index(drop=True)])
    out.to_csv(save_path, index=False)
    sig = "SIGNIFICANT" if p < 0.05 else "not significant"
    print(f"\n  [{label}]")
    print(ct.to_string())
    print(f"  chi2={chi2:.4f}  p={p:.6f}  dof={dof}  => {sig}")
    print(f"  -> {save_path}")
    return {"chi2": chi2, "p_value": p, "dof": dof, "significant": p < 0.05}


r1 = run_chi2_and_save(df,                            "ita_group", "source",
     OUT_DIR/"source_distribution_by_ita_group.csv",
     "OVERALL  - source vs ITA group (all classes)")

df_nv = df[df["dx"] == "nv"].copy()
r2 = run_chi2_and_save(df_nv,                         "ita_group", "source",
     OUT_DIR/"source_distribution_by_ita_group_nv_only.csv",
     "WITHIN NV - source vs ITA group (nv only)")

df_nv_dl = df_nv[df_nv["ita_group"].isin(["dark","light"])].copy()
r3 = run_chi2_and_save(df_nv_dl,                      "ita_group", "source",
     OUT_DIR/"source_distribution_nv_dark_vs_light.csv",
     "WITHIN NV (dark vs light only) - 2x2 chi-square")

# ─── TASK 2 ───────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("  TASK 2 - ITA Visual Validation Spot-Check")
print("="*72)

rng = np.random.default_rng(SEED)
dark_nv  = df_nv[df_nv["ita_group"] == "dark"].reset_index(drop=True)
light_nv = df_nv[df_nv["ita_group"] == "light"].reset_index(drop=True)

di = rng.choice(len(dark_nv),  size=min(6, len(dark_nv)),  replace=False)
li = rng.choice(len(light_nv), size=min(6, len(light_nv)), replace=False)

samples = (
    [("dark",  i+1, dark_nv.iloc[x])  for i, x in enumerate(di)] +
    [("light", i+1, light_nv.iloc[x]) for i, x in enumerate(li)]
)

def corner_boxes(h, w, p):
    return [
        (0,    p,    0,    p),
        (0,    p,    w-p,  w),
        (h-p,  h,    0,    p),
        (h-p,  h,    w-p,  w),
    ]

summary_rows = []

for grp_lbl, idx, row in samples:
    img_path = str(row["image_path"])
    image_id = str(row["image_id"])
    ita_val  = float(row.get("ita_value", float("nan")))

    if not Path(img_path).exists():
        print(f"  MISSING: {img_path}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"  UNREADABLE: {img_path}")
        continue

    h, w = img.shape[:2]
    boxes = corner_boxes(h, w, PATCH_SZ)

    flagged_total = 0
    pixel_total   = 0
    for (r0, r1, c0, c1) in boxes:
        patch_gray = cv2.cvtColor(img[r0:r1, c0:c1], cv2.COLOR_BGR2GRAY).astype(float)
        mu, sigma = patch_gray.mean(), patch_gray.std()
        flagged_total += int((patch_gray < (mu - 2*sigma)).sum())
        pixel_total   += patch_gray.size

    pct = 100.0 * flagged_total / pixel_total if pixel_total > 0 else 0.0
    summary_rows.append({
        "image_id":                       image_id,
        "ita_group":                      grp_lbl,
        "ita_value":                      round(ita_val, 4),
        "pct_corner_pixels_flagged_dark": round(pct, 2),
    })

    # Annotated figure
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.imshow(img_rgb)
    for (r0, r1, c0, c1) in boxes:
        ax.add_patch(patches.Rectangle(
            (c0, r0), c1-c0, r1-r0,
            linewidth=2.5, edgecolor="lime", facecolor="none"
        ))
    title_color = "darkgreen" if grp_lbl == "light" else "saddlebrown"
    ax.set_title(
        f"{image_id}\nITA = {ita_val:.2f}  |  group = {grp_lbl}  |  flagged = {pct:.1f}%",
        fontsize=9, fontweight="bold", color=title_color,
    )
    ax.axis("off")
    fig.tight_layout()
    out_name = f"{grp_lbl}_{idx:02d}.png"
    fig.savefig(VIS_DIR / out_name, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_name}  ITA={ita_val:.2f}  flagged={pct:.1f}%  ({img_path})")

summary_df = pd.DataFrame(summary_rows)
sp = OUT_DIR / "ita_validation_summary.csv"
summary_df.to_csv(sp, index=False)
print(f"\n  Summary CSV -> {sp}")
print(summary_df.to_string(index=False))
print("\nDone.")
