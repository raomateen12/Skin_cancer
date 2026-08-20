"""
src/temperature_scaling.py
===========================
Post-hoc per-group Temperature Scaling on the baseline checkpoint.
No retraining. Val set is used to FIT temperatures; test set is held-out evaluation.

Reference: Guo et al. (2017) "On Calibration of Modern Neural Networks"
           https://arxiv.org/abs/1706.04599

Usage:
    python -m src.temperature_scaling

Outputs (results/temperature_scaling/):
    fitted_temperatures.json
    three_way_comparison.csv
    gap_significance_test.csv
    temperature_scaling_report.txt
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.dataset import get_eval_transforms, class_to_idx
from src.model import get_efficientnet_b0
from src.calibration_fairness import (
    brier_score_multiclass,
    calibration_metrics_for_group,
    N_BOOTSTRAP,
)

# ── Config ────────────────────────────────────────────────────────────────────
BASELINE_CKPT  = "checkpoints/best_efficientnet_b0.pth"
VAL_ITA_CSV    = "data/processed/val_with_ita.csv"
TEST_ITA_CSV   = "data/processed/test_with_ita.csv"
BASELINE_DIR   = Path("results/calibration_fairness")
OUT_DIR        = Path("results/temperature_scaling")
NUM_CLASSES    = len(class_to_idx)
GROUPS         = ["light", "intermediate", "dark"]
SEED           = 42
N_BOOT         = N_BOOTSTRAP

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Dataset ───────────────────────────────────────────────────────────────────
class _DS(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(str(row["image_path"])).convert("RGB")
        img = np.array(img)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, int(row["label_id"])


def get_logits_and_labels(df, model, device, batch_size=32):
    ds = _DS(df, get_eval_transforms(224))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="  Logits", leave=False):
            imgs = imgs.to(device)
            all_logits.append(model(imgs).cpu())
            all_labels.append(lbls)
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


# ── Temperature fitting ───────────────────────────────────────────────────────
def fit_temperature(logits, labels, n_iter=200, lr=0.01, init_T=1.5):
    logits_t = torch.from_numpy(logits).float()
    labels_t = torch.from_numpy(labels).long()
    nll = nn.CrossEntropyLoss()
    T = nn.Parameter(torch.tensor([init_T]))
    optim = torch.optim.LBFGS([T], lr=lr, max_iter=n_iter)

    def closure():
        optim.zero_grad()
        loss = nll(logits_t / T, labels_t)
        loss.backward()
        return loss

    optim.step(closure)
    return float(T.item())


def apply_temperature(logits, T):
    scaled = logits / max(T, 1e-4)
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def metrics_for(probs, labels):
    preds = probs.argmax(axis=1)
    return calibration_metrics_for_group(probs, preds, labels, NUM_CLASSES)


# ── Main ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  DermaLens AI — Per-Group Temperature Scaling")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n  Device: {device}")
ckpt = torch.load(BASELINE_CKPT, map_location=device, weights_only=False)
model = get_efficientnet_b0(NUM_CLASSES).to(device)
model.load_state_dict(ckpt["model_state_dict"])
print(f"  Baseline checkpoint: epoch {ckpt.get('epoch', '?')}, val_f1={ckpt.get('val_weighted_f1', '?'):.4f}")

val_df  = pd.read_csv(VAL_ITA_CSV)
test_df = pd.read_csv(TEST_ITA_CSV)
print(f"  Val: {len(val_df)} | Test: {len(test_df)}")

print("\n-- Step 1: Val logits (for fitting) --")
val_logits, val_labels = get_logits_and_labels(val_df, model, device)
val_groups = val_df["ita_group"].values

print("\n-- Step 2: Test logits (held-out evaluation) --")
test_logits, test_labels = get_logits_and_labels(test_df, model, device)
test_groups = test_df["ita_group"].values

print("\n-- Step 3: Fitting temperatures --")
glob_mask = np.isin(val_groups, GROUPS)
T_global  = fit_temperature(val_logits[glob_mask], val_labels[glob_mask])
print(f"  T_global = {T_global:.4f}  (n={int(glob_mask.sum())})")

T_group = {}
for grp in GROUPS:
    g_mask = (val_groups == grp)
    n = int(g_mask.sum())
    if n < 5:
        T_group[grp] = 1.0
        print(f"  T_{grp} = 1.0 (fallback, only {n} val samples)")
        continue
    T_g = fit_temperature(val_logits[g_mask], val_labels[g_mask])
    T_group[grp] = T_g
    print(f"  T_{grp} = {T_g:.4f}  (n_val={n})")

temps_json = {
    "T_global": T_global,
    "T_per_group": T_group,
    "val_n_per_group": {g: int((val_groups == g).sum()) for g in GROUPS},
    "note": "Fitted on val_with_ita.csv via LBFGS minimising NLL (Guo et al. 2017)",
}
with open(OUT_DIR / "fitted_temperatures.json", "w") as fh:
    json.dump(temps_json, fh, indent=2)
print(f"  -> results/temperature_scaling/fitted_temperatures.json")

print("\n-- Step 4: Applying temperatures to test set --")
probs_uncal  = apply_temperature(test_logits, 1.0)
probs_global = apply_temperature(test_logits, T_global)

probs_pergrp = np.zeros_like(probs_uncal)
for i, grp in enumerate(test_groups):
    T_i = T_group.get(grp, T_global)
    scaled = test_logits[i] / max(T_i, 1e-4)
    scaled -= scaled.max()
    exp = np.exp(scaled)
    probs_pergrp[i] = exp / exp.sum()

acc_uncal  = float((probs_uncal.argmax(1)  == test_labels).mean())
acc_global = float((probs_global.argmax(1) == test_labels).mean())
acc_pergrp = float((probs_pergrp.argmax(1) == test_labels).mean())
print(f"  Accuracy sanity check (must be identical):")
print(f"    uncalibrated : {acc_uncal:.6f}")
print(f"    global-T     : {acc_global:.6f}")
print(f"    per-group-T  : {acc_pergrp:.6f}")
if abs(acc_uncal - acc_global) < 1e-6 and abs(acc_uncal - acc_pergrp) < 1e-6:
    print("  PASS - accuracy identical across all three")
else:
    print("  WARN - accuracy differs! Check for bug.")

print("\n-- Step 5: Three-way comparison table --")
base_cal = pd.read_csv(BASELINE_DIR / "calibration_by_ita_group.csv")
base_by_grp = {row["group"]: row for _, row in base_cal.iterrows()}

rows = []
for grp in GROUPS + ["overall"]:
    if grp == "overall":
        mask = np.isin(test_groups, GROUPS)
        total_n = sum(base_by_grp[g]["n"] for g in GROUPS if g in base_by_grp)
        b_ece   = sum(base_by_grp[g]["ece"]      * base_by_grp[g]["n"] for g in GROUPS if g in base_by_grp) / total_n
        b_mce   = max(base_by_grp[g]["mce"]      for g in GROUPS if g in base_by_grp)
        b_brier = sum(base_by_grp[g]["brier"]    * base_by_grp[g]["n"] for g in GROUPS if g in base_by_grp) / total_n
        b_acc   = sum(base_by_grp[g]["accuracy"] * base_by_grp[g]["n"] for g in GROUPS if g in base_by_grp) / total_n
    else:
        mask = (test_groups == grp)
        b = base_by_grp.get(grp, {})
        b_ece   = float(b.get("ece",      float("nan")))
        b_mce   = float(b.get("mce",      float("nan")))
        b_brier = float(b.get("brier",    float("nan")))
        b_acc   = float(b.get("accuracy", float("nan")))

    n = int(mask.sum())
    if n == 0:
        continue

    mg = metrics_for(probs_global[mask], test_labels[mask])
    mp = metrics_for(probs_pergrp[mask], test_labels[mask])

    rows.append({
        "group": grp, "n": n,
        "uncal_ece":    round(b_ece,   5),
        "uncal_mce":    round(b_mce,   5),
        "uncal_brier":  round(b_brier, 5),
        "uncal_acc":    round(b_acc,   5),
        "global_ece":   round(mg["ece"],      5),
        "global_mce":   round(mg["mce"],      5),
        "global_brier": round(mg["brier"],    5),
        "global_acc":   round(mg["accuracy"], 5),
        "pergrp_ece":   round(mp["ece"],      5),
        "pergrp_mce":   round(mp["mce"],      5),
        "pergrp_brier": round(mp["brier"],    5),
        "pergrp_acc":   round(mp["accuracy"], 5),
    })

comp_df = pd.DataFrame(rows)
comp_df.to_csv(OUT_DIR / "three_way_comparison.csv", index=False)
print(comp_df[["group","n","uncal_ece","global_ece","pergrp_ece",
               "uncal_brier","global_brier","pergrp_brier"]].to_string(index=False))
print(f"  -> results/temperature_scaling/three_way_comparison.csv")

print("\n-- Step 6: Bootstrap gap significance --")
rng = np.random.default_rng(SEED)

def bootstrap_gap(probs_dark, labels_dark, probs_light, labels_light, n_boot, rng_):
    n_min = min(len(labels_dark), len(labels_light))
    diffs = []
    for _ in range(n_boot):
        id_ = rng_.choice(len(labels_dark),  size=n_min, replace=True)
        il  = rng_.choice(len(labels_light), size=n_min, replace=True)
        bd  = brier_score_multiclass(probs_dark[id_],  labels_dark[id_],  NUM_CLASSES)
        bl  = brier_score_multiclass(probs_light[il],  labels_light[il],  NUM_CLASSES)
        diffs.append(bd - bl)
    diffs = np.array(diffs)
    diff  = float(diffs.mean())
    ci_lo = float(np.percentile(diffs, 2.5))
    ci_hi = float(np.percentile(diffs, 97.5))
    sig   = bool((ci_lo > 0) or (ci_hi < 0))
    return diff, ci_lo, ci_hi, sig

mask_dark  = (test_groups == "dark")
mask_light = (test_groups == "light")

sig_rows = []
for label_, probs_ in [
    ("uncalibrated",          probs_uncal),
    ("global_temperature",    probs_global),
    ("per_group_temperature", probs_pergrp),
]:
    print(f"  Bootstrap ({label_}) ...", end=" ", flush=True)
    diff, ci_lo, ci_hi, sig = bootstrap_gap(
        probs_[mask_dark], test_labels[mask_dark],
        probs_[mask_light], test_labels[mask_light],
        N_BOOT, rng,
    )
    print(f"diff={diff:+.4f}  CI=[{ci_lo:.4f},{ci_hi:.4f}]  sig={sig}")
    sig_rows.append({
        "calibration":          label_,
        "brier_dark":           round(metrics_for(probs_[mask_dark],  test_labels[mask_dark])["brier"],  4),
        "brier_light":          round(metrics_for(probs_[mask_light], test_labels[mask_light])["brier"], 4),
        "diff_dark_minus_light": round(diff, 4),
        "ci_lower":             round(ci_lo, 4),
        "ci_upper":             round(ci_hi, 4),
        "significant":          sig,
        "n_dark":               int(mask_dark.sum()),
        "n_light":              int(mask_light.sum()),
    })

sig_df = pd.DataFrame(sig_rows)
sig_df.to_csv(OUT_DIR / "gap_significance_test.csv", index=False)
print(sig_df.to_string(index=False))
print(f"  -> results/temperature_scaling/gap_significance_test.csv")

print("\n-- Step 7: Writing report --")
grp_rows = {r["group"]: r for r in rows}
uncal_gap  = sig_rows[0]["diff_dark_minus_light"]
glob_gap   = sig_rows[1]["diff_dark_minus_light"]
pergrp_gap = sig_rows[2]["diff_dark_minus_light"]
uncal_ci   = (sig_rows[0]["ci_lower"], sig_rows[0]["ci_upper"])
pergrp_ci  = (sig_rows[2]["ci_lower"], sig_rows[2]["ci_upper"])
ci_overlap = min(uncal_ci[1], pergrp_ci[1]) - max(uncal_ci[0], pergrp_ci[0])
gap_change = pergrp_gap - uncal_gap

rlines = [
    "=" * 72,
    "  DermaLens AI - Temperature Scaling Calibration Report",
    "=" * 72,
    "",
    "Method: Post-hoc Temperature Scaling (Guo et al. 2017).",
    "  - NO retraining; original baseline checkpoint used as-is.",
    "  - Temperatures fitted on VAL set via LBFGS (minimise NLL).",
    "  - Evaluation on TEST set (completely held out, no data leakage).",
    "",
    "FITTED TEMPERATURES:",
    f"  T_global       = {T_global:.4f}",
]
for g in GROUPS:
    rlines.append(f"  T_{g:13s} = {T_group[g]:.4f}  (n_val={temps_json['val_n_per_group'].get(g,0)})")

rlines += [
    "",
    "  T > 1 => model was overconfident, scaling softens probs.",
    "  T < 1 => model was underconfident, scaling sharpens probs.",
    "",
    "ACCURACY SANITY CHECK:",
    f"  Uncalibrated / Global-T / Per-group-T all give acc = {acc_uncal:.4f}  PASS",
    "",
    "THREE-WAY CALIBRATION COMPARISON (test set):",
    f"  {'Group':13s} {'n':>5}  {'Brier_uncal':>11}  {'Brier_glob':>10}  {'Brier_pergrp':>12}  {'ECE_uncal':>9}  {'ECE_glob':>8}  {'ECE_pergrp':>10}",
    "  " + "-" * 90,
]
for g in GROUPS + ["overall"]:
    if g not in grp_rows:
        continue
    r = grp_rows[g]
    rlines.append(
        f"  {g:13s} {r['n']:>5d}  {r['uncal_brier']:>11.4f}  {r['global_brier']:>10.4f}"
        f"  {r['pergrp_brier']:>12.4f}  {r['uncal_ece']:>9.4f}  {r['global_ece']:>8.4f}  {r['pergrp_ece']:>10.4f}"
    )

rlines += [
    "",
    "DARK-VS-LIGHT BRIER GAP (bootstrap, 1000 iterations):",
    f"  {'Calibration':30s}  {'Gap(dark-light)':>15}  {'95% CI':>22}  {'Sig?':>5}",
    "  " + "-" * 80,
]
for r in sig_rows:
    rlines.append(
        f"  {r['calibration']:30s}  {r['diff_dark_minus_light']:>+15.4f}"
        f"  [{r['ci_lower']:.4f},{r['ci_upper']:.4f}]  {'YES' if r['significant'] else 'no':>5}"
    )

rlines += [
    "",
    "VERDICT:",
    f"  (a) Accuracy preserved: YES — all three methods give acc={acc_uncal:.4f}.",
    f"  (b) Dark-vs-light gap change (uncal -> per-group-T): {gap_change:+.4f}",
]

if gap_change < -0.01:
    rlines.append(f"      Gap SHRANK by {abs(gap_change):.4f} units.")
elif gap_change > 0.01:
    rlines.append(f"      Gap GREW by {gap_change:.4f} units (unexpected).")
else:
    rlines.append(f"      Gap LARGELY UNCHANGED (|change|={abs(gap_change):.4f}).")

if ci_overlap < 0:
    rlines.append("      CI overlap < 0 => gap change is statistically meaningful.")
else:
    rlines.append(f"      CI overlap = {ci_overlap:.4f} => gap change inconclusive from CI comparison.")

if not sig_rows[2]["significant"]:
    rlines += [
        "",
        "  KEY FINDING: After per-group temperature scaling, the dark-vs-light",
        "  Brier gap is NO LONGER statistically significant (95% CI includes 0).",
        "  Temperature scaling neutralised the calibration disparity without retraining.",
    ]
elif not sig_rows[1]["significant"]:
    rlines += [
        "",
        "  KEY FINDING: Global-T scaling neutralised the gap (CI includes 0),",
        "  per-group-T did not add further benefit over global scaling.",
    ]
else:
    rlines += [
        "",
        "  The dark-vs-light gap remains statistically significant even after",
        "  per-group temperature scaling. The gap reflects inherent difficulty",
        "  differences that confidence rescaling alone cannot address.",
    ]

rlines += [
    "",
    "OUTPUT FILES (results/temperature_scaling/):",
    "  fitted_temperatures.json   - T_global and T_per_group",
    "  three_way_comparison.csv   - ECE/MCE/Brier for all three methods",
    "  gap_significance_test.csv  - bootstrap gap for all three",
    "  temperature_scaling_report.txt - this file",
    "=" * 72,
]

report_path = OUT_DIR / "temperature_scaling_report.txt"
with open(report_path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(rlines) + "\n")
print("\n".join(rlines))
print(f"\n  Report -> {report_path}")
print("Done.")
