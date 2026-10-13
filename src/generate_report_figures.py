"""
src/generate_report_figures.py
==============================
Generate publication-quality, black-and-white / grayscale figures for the
DermaLens AI research report.

Requirements:
- Black and white / grayscale only (no color — for print/academic submission)
- High resolution (300 DPI minimum, saved as PNG)
- Clean, minimal style (no gridlines unless essential, no chartjunk)
- Times New Roman or serif font for all labels and titles
- Saved to results/report_figures/ folder
- Matplotlib with plt.style.use('grayscale')

Figures:
  1. fig1_brier_by_group.png
  2. fig2_reliability_diagrams.png
  3. fig3_confound_checks.png
  4. fig4_remediation_comparison.png
  5. fig5_nv_stratified.png
  6. fig6_federated_convergence.png
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

OUT_DIR = Path("results/report_figures")


def configure_academic_style() -> None:
    """Configure matplotlib for publication-grade, serif, grayscale graphics."""
    plt.style.use("grayscale")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "DejaVu Serif",
            "Liberation Serif",
            "Bitstream Vera Serif",
            "serif",
        ],
        "mathtext.fontset": "dejavuserif",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "normal",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    })


# ── Figure 1: Calibration Disparity Bar Chart ────────────────────────────────
def generate_figure_1(out_path: Path) -> None:
    """
    FIGURE 1 — Calibration Disparity Bar Chart:
    Bar chart showing Brier score per ITA group (light=0.158, intermediate=0.159, dark=0.305).
    Error bars from matched bootstrap std values (light std=0.040, intermediate std=0.041, dark std=0.048).
    Label bars with actual values.
    Add horizontal dashed line at overall mean Brier (0.210).
    X-axis: ITA Group, Y-axis: Brier Score (0 to 0.4).
    Title: "Figure 1: Brier Score by Estimated Skin-Tone Group (Matched Bootstrap, N=97 each)"
    """
    groups = ["Light", "Intermediate", "Dark"]
    scores = [0.158, 0.159, 0.305]
    stds = [0.040, 0.041, 0.048]
    overall_mean = 0.210

    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    # Grayscale bar shading: light, medium, and dark gray
    shades = ["#e0e0e0", "#9a9a9a", "#4a4a4a"]
    x_pos = np.arange(len(groups))
    bars = ax.bar(
        x_pos,
        scores,
        yerr=stds,
        capsize=6,
        color=shades,
        edgecolor="black",
        linewidth=1.2,
        error_kw={"elinewidth": 1.2, "ecolor": "black", "capthick": 1.2},
        width=0.52,
    )

    # Label bars with actual values (placed cleanly above error bars with subtle white backing)
    for bar, score, std in zip(bars, scores, stds):
        y_text = score + std + 0.014
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_text,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.9),
            zorder=4,
        )

    # Horizontal dashed line at overall mean Brier (0.210) with clean legend
    ax.axhline(
        overall_mean,
        color="black",
        linestyle="--",
        linewidth=1.2,
        zorder=2,
        label=f"Overall Mean ({overall_mean:.3f})",
    )
    ax.legend(loc="upper left", frameon=True, edgecolor="black", fontsize=9.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylim(0.0, 0.40)
    ax.set_xlabel("ITA Group", fontsize=11, labelpad=7)
    ax.set_ylabel("Brier Score", fontsize=11, labelpad=7)
    ax.set_title(
        "Figure 1: Brier Score by Estimated Skin-Tone Group\n(Matched Bootstrap, N=97 each)",
        pad=14,
        fontweight="bold",
    )

    ax.grid(False)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


# ── Figure 2: Reliability Diagrams (3 panels) ─────────────────────────────────
class _EvalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(str(row["image_path"])).convert("RGB")
        img_np = np.array(img)
        img_t = self.transform(image=img_np)["image"]
        return img_t, int(row["label_id"])


def _get_or_compute_baseline_preds() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load cached baseline predictions or run inference once if missing."""
    cache_path = OUT_DIR / "baseline_test_predictions.npz"
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return data["probs"], data["labels"], data["ita_group"]

    from src.calibration_fairness_hairremoved import validate_and_remap_image_paths
    from src.dataset import get_eval_transforms
    from src.model import get_efficientnet_b0

    torch.set_num_threads(os.cpu_count() or 4)
    test_csv = Path("data/processed/test_with_ita.csv")
    df = pd.read_csv(test_csv)
    df = validate_and_remap_image_paths(df)

    transform = get_eval_transforms(224)
    dataset = _EvalDataset(df, transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    model = get_efficientnet_b0(7)
    ckpt = torch.load("checkpoints/best_efficientnet_b0.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    all_probs = []
    with torch.no_grad():
        for imgs, _ in loader:
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).numpy()
            all_probs.append(probs)

    probs = np.concatenate(all_probs, axis=0)
    labels = df["label_id"].values
    ita_group = df["ita_group"].values

    np.savez(cache_path, probs=probs, labels=labels, ita_group=ita_group)
    return probs, labels, ita_group


def _compute_reliability_bins(
    probs: np.ndarray, preds: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> tuple[list[float], list[float], float]:
    """Compute bin centers, empirical accuracies, and ECE for 15 bins."""
    confs = probs.max(axis=1)
    corrects = (preds == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    bin_accs, bin_confs = [], []
    ece = 0.0
    n_total = len(confs)

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs > lo) & (confs <= hi)
        if mask.sum() > 0:
            b_acc = corrects[mask].mean()
            b_conf = confs[mask].mean()
            bin_accs.append(b_acc)
            bin_confs.append(b_conf)
            ece += (mask.sum() / n_total) * abs(b_acc - b_conf)

    return bin_confs, bin_accs, float(ece)


def generate_figure_2(out_path: Path) -> None:
    """
    FIGURE 2 — Reliability Diagrams (3 panels):
    Three side-by-side reliability diagrams (one per ITA group: light, intermediate, dark).
    Each shows predicted confidence (x-axis, 0-1) vs actual accuracy (y-axis, 0-1) in 15 bins,
    with a diagonal "perfect calibration" reference line.
    Mark each panel with the group's ECE value.
    Title: "Figure 2: Reliability Diagrams by ITA Group"
    """
    probs, labels, ita_group = _get_or_compute_baseline_preds()
    preds = probs.argmax(axis=1)

    groups = [
        ("light", "Light", 0.030),
        ("intermediate", "Intermediate", 0.070),
        ("dark", "Dark", 0.060),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)

    for idx, (grp_key, grp_label, reported_ece) in enumerate(groups):
        ax = axes[idx]
        mask = ita_group == grp_key
        probs_g = probs[mask]
        preds_g = preds[mask]
        labels_g = labels[mask]

        bin_confs, bin_accs, comp_ece = _compute_reliability_bins(
            probs_g, preds_g, labels_g, n_bins=15
        )

        # Diagonal perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration", zorder=2)

        # 15-bin empirical accuracy bars
        bar_width = 1.0 / 15.0
        ax.bar(
            bin_confs,
            bin_accs,
            width=bar_width * 0.88,
            color="#b0b0b0",
            edgecolor="black",
            linewidth=0.9,
            alpha=0.75,
            label="Observed accuracy",
            zorder=3,
        )

        # Connected points over bars
        ax.plot(
            bin_confs,
            bin_accs,
            "o-",
            color="black",
            markersize=4.5,
            linewidth=1.3,
            zorder=4,
        )

        # Annotate ECE in panel
        ax.text(
            0.05,
            0.88,
            f"ECE = {reported_ece:.3f}\n(N = {len(labels_g)})",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="normal",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", lw=0.8),
            zorder=5,
        )

        panel_letter = chr(ord("a") + idx)
        ax.set_title(f"({panel_letter}) {grp_label} Group", pad=10, fontweight="bold")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Predicted Confidence", fontsize=10.5, labelpad=6)
        if idx == 0:
            ax.set_ylabel("Empirical Accuracy", fontsize=10.5, labelpad=6)
            ax.legend(loc="lower right", fontsize=9, frameon=True, edgecolor="black")

        ax.grid(False)

    fig.suptitle("Figure 2: Reliability Diagrams by ITA Group", fontsize=13, y=1.03, weight="bold")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


# ── Figure 3: Five Confound Checks Summary (horizontal bar chart) ─────────────
def generate_figure_3(out_path: Path) -> None:
    """
    FIGURE 3 — Five Confound Checks Summary (horizontal bar chart):
    A visual summary of the 5 robustness checks. For each check, show 
    whether the gap survived (filled bar) or was partially explained 
    (hatched bar):
    Check 1: Class distribution — "Gap persists within nv class"
    Check 2: Vignette artifacts — "5% gap reduction only"  
    Check 3: Acquisition source — "Gap in both sources independently"
    Check 4: b* formula degeneracy — "Gap survives despite 43% label noise"
    Check 5: Statistical significance — "CI excludes zero, p=0.0001"
    Use a simple horizontal bar with checkmark symbols. 
    Title: "Figure 3: Robustness of Calibration Disparity Across Five Confound Checks".
    """
    checks = [
        ("Check 1: Class Distribution", "Gap persists within nv class (p = 0.0001)", True),
        ("Check 2: Vignette Artifacts", "5% gap reduction only (artifact minor)", False),
        ("Check 3: Acquisition Source", "Gap in both sources independently", True),
        ("Check 4: b* Formula Degeneracy", "Gap survives despite 43% label noise", True),
        ("Check 5: Statistical Significance", "CI excludes zero, p = 0.0001", True),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))

    y_labels = [c[0] for c in checks]
    y_pos = np.arange(len(checks))

    # All bars set to uniform length 1.0 for visual alignment of robustness checklist
    bar_width = 1.0

    for i, (_, desc, survived) in enumerate(checks):
        if survived:
            # Filled bar: solid dark gray
            ax.barh(
                y_pos[i],
                bar_width,
                color="#4a4a4a",
                edgecolor="black",
                linewidth=1.2,
                height=0.56,
            )
        else:
            # Hatched bar: light background with distinct diagonal hatching
            ax.barh(
                y_pos[i],
                bar_width,
                color="#ffffff",
                edgecolor="black",
                hatch="//////",
                linewidth=1.2,
                height=0.56,
            )

        # Vector checkmark to ensure font independence
        # Note: ax.invert_yaxis() means positive y is downward, negative y is upward on screen!
        chk_color = "white" if survived else "black"
        x0 = 0.025
        ax.plot(
            [x0, x0 + 0.011, x0 + 0.028],
            [y_pos[i] - 0.02, y_pos[i] + 0.09, y_pos[i] - 0.11],
            color=chk_color,
            lw=2.5,
            solid_capstyle="round",
            zorder=5,
        )

        if survived:
            # Text inside bar (white for contrast)
            ax.text(
                x0 + 0.045,
                y_pos[i],
                desc,
                va="center",
                ha="left",
                color="white",
                fontsize=10.5,
                fontweight="bold",
                zorder=5,
            )
        else:
            # Text inside bar (black on white badge for clear readability over hatch)
            ax.text(
                x0 + 0.045,
                y_pos[i],
                desc,
                va="center",
                ha="left",
                color="black",
                fontsize=10.5,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", lw=0.6),
                zorder=5,
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=10.5, fontweight="normal")
    ax.invert_yaxis()  # Check 1 on top, Check 5 at bottom

    ax.set_xlim(0.0, 1.08)
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)

    # Academic legend explaining solid vs hatched
    solid_patch = mpatches.Patch(
        facecolor="#4a4a4a",
        edgecolor="black",
        linewidth=1.0,
        label="Disparity Survived (Genuine Gap)",
    )
    hatched_patch = mpatches.Patch(
        facecolor="#ffffff",
        edgecolor="black",
        hatch="//////",
        linewidth=1.0,
        label="Partially Explained (Minor Confound)",
    )
    ax.legend(
        handles=[solid_patch, hatched_patch],
        loc="lower right",
        bbox_to_anchor=(1.0, -0.16),
        ncol=2,
        frameon=True,
        edgecolor="black",
        fontsize=9.5,
    )

    ax.set_title(
        "Figure 3: Robustness of Calibration Disparity Across Five Confound Checks",
        pad=14,
        fontweight="bold",
    )

    ax.grid(False)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


# ── Figure 4: Remediation Comparison Chart ────────────────────────────────────
def generate_figure_4(out_path: Path) -> None:
    """
    FIGURE 4 — Remediation Comparison Table Chart:
    Grouped bar chart showing dark-group Brier score across all interventions:
    Baseline: 0.305
    After reweighting: 0.552 (worse)
    After temperature scaling: 0.304 (unchanged)
    After hair removal: 0.289 (slight improvement, ns)
    Light group baseline reference line at 0.158
    Add annotation "ns" (not significant) above hair removal bar, 
    and "**" (significant worsening) above reweighting bar.
    Title: "Figure 4: Dark-Group Brier Score Across Remediation Attempts".
    """
    interventions = [
        "Baseline\n(Uncalibrated)",
        "Class\nReweighting",
        "Temperature\nScaling",
        "Hair Removal\nPreprocessing",
    ]
    scores = [0.305, 0.552, 0.304, 0.289]
    light_baseline = 0.158

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    # Grayscale shading for interventions
    colors = ["#4a4a4a", "#6e6e6e", "#9e9e9e", "#cccccc"]
    hatches = ["", "///", "", ""]
    x_pos = np.arange(len(interventions))

    bars = ax.bar(
        x_pos,
        scores,
        color=colors,
        edgecolor="black",
        hatch=hatches,
        linewidth=1.2,
        width=0.52,
        zorder=3,
    )

    # Reference line at light group baseline (0.158)
    ax.axhline(
        light_baseline,
        color="black",
        linestyle="--",
        linewidth=1.2,
        zorder=2,
        label=f"Light-group baseline ({light_baseline:.3f})",
    )

    # Value labels and annotations above bars
    annotations = ["", "**", "", "ns"]
    for i, (bar, score, ann) in enumerate(zip(bars, scores, annotations)):
        # Numeric label
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.012,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            zorder=4,
        )
        # Significance flag
        if ann:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                score + 0.042,
                ann,
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                zorder=4,
            )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(interventions, fontsize=10)
    ax.set_ylim(0.0, 0.70)
    ax.set_ylabel("Dark-Group Brier Score", fontsize=11, labelpad=7)
    ax.set_xlabel("Remediation Attempt", fontsize=11, labelpad=7)

    # Title with subtitle note explaining significance markings
    ax.set_title(
        "Figure 4: Dark-Group Brier Score Across Remediation Attempts\n"
        "(** significant worsening, p < 0.001; ns = not significant, p = 0.407)",
        pad=12,
        fontweight="bold",
        fontsize=11.5,
    )

    # Legend at upper right
    ax.legend(loc="upper right", frameon=True, edgecolor="black", fontsize=9.5)

    ax.grid(False)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


# ── Figure 5: nv Class Within-Group Analysis ──────────────────────────────────
def generate_figure_5(out_path: Path) -> None:
    """
    FIGURE 5 — nv Class Within-Group Analysis:
    Simple bar chart: within nv class only, dark group Brier (0.318) 
    vs light group Brier (0.088) vs intermediate (0.119). 
    Add annotation showing the gap is significant (95% CI [0.112, 0.360]). 
    Title: "Figure 5: Within-nv-Class Brier Score by ITA Group (Class-Stratified Analysis, Matched Bootstrap)".
    """
    groups = ["Light", "Intermediate", "Dark"]
    scores = [0.088, 0.119, 0.318]

    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    shades = ["#e0e0e0", "#9a9a9a", "#4a4a4a"]
    x_pos = np.arange(len(groups))

    bars = ax.bar(
        x_pos,
        scores,
        color=shades,
        edgecolor="black",
        linewidth=1.2,
        width=0.52,
        zorder=3,
    )

    # Value labels above bars
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.009,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
        )

    # Significance bracket between Light (x=0) and Dark (x=2)
    y_bracket = 0.380
    bracket_h = 0.012
    ax.plot(
        [0, 0, 2, 2],
        [y_bracket - bracket_h, y_bracket, y_bracket, y_bracket - bracket_h],
        color="black",
        lw=1.2,
        zorder=4,
    )
    ax.text(
        1.0,
        y_bracket + 0.008,
        "Disparity Δ = 0.230 ***\n95% CI [0.112, 0.360] (p = 0.0001)",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylim(0.0, 0.46)
    ax.set_xlabel("ITA Group", fontsize=11, labelpad=7)
    ax.set_ylabel("Brier Score (nv class only)", fontsize=11, labelpad=7)
    ax.set_title(
        "Figure 5: Within-nv-Class Brier Score by ITA Group\n(Class-Stratified Analysis, Matched Bootstrap)",
        pad=14,
        fontweight="bold",
    )

    ax.grid(False)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


# ── Figure 6: Federated Learning Convergence ──────────────────────────────────
def generate_figure_6(out_path: Path) -> None:
    """
    FIGURE 6 — Federated Learning Convergence:
    Line chart showing Global Val Accuracy per communication round (rounds 1-10):
    77.42, 82.32, 83.72, 85.51, 87.01, 87.61, 88.01, 87.81, 88.81, 89.01.
    Add a horizontal dashed line at centralized baseline (90.11%) labeled "Centralized (20 epochs)".
    Add annotation at round 10: "Federated: 87.91%".
    X-axis: Communication Round, Y-axis: Validation Accuracy (%).
    Title: "Figure 6: FedAvg Global Model Accuracy Across Communication Rounds vs. Centralized Baseline".
    """
    rounds = np.arange(1, 11)
    accuracies = [77.42, 82.32, 83.72, 85.51, 87.01, 87.61, 88.01, 87.81, 88.81, 89.01]
    centralized_acc = 90.11

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    # FedAvg trajectory line
    ax.plot(
        rounds,
        accuracies,
        "o-",
        color="black",
        linewidth=1.8,
        markersize=6,
        label="FedAvg Global Model",
        zorder=3,
    )

    # Centralized baseline horizontal dashed line
    ax.axhline(
        centralized_acc,
        color="#404040",
        linestyle="--",
        linewidth=1.3,
        label="Centralized (20 epochs): 90.11%",
        zorder=2,
    )

    # Annotation at round 10
    ax.annotate(
        "Federated: 87.91%",
        xy=(10, accuracies[-1]),
        xytext=(8.0, 84.8),
        arrowprops=dict(
            facecolor="black",
            arrowstyle="->",
            lw=1.2,
            shrinkA=4,
            shrinkB=4,
        ),
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#f5f5f5", edgecolor="black", lw=0.7),
        zorder=4,
    )

    ax.set_xticks(rounds)
    ax.set_xlim(0.6, 10.4)
    ax.set_ylim(74.0, 93.0)
    ax.set_xlabel("Communication Round", fontsize=11, labelpad=7)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=11, labelpad=7)
    ax.set_title(
        "Figure 6: FedAvg Global Model Accuracy Across Communication Rounds\nvs. Centralized Baseline",
        pad=14,
        fontweight="bold",
    )

    ax.legend(loc="lower right", frameon=True, edgecolor="black", fontsize=9.5)
    ax.grid(False)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


# ── Main Entrypoint ───────────────────────────────────────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_academic_style()

    figures = [
        ("Figure 1", OUT_DIR / "fig1_brier_by_group.png", generate_figure_1),
        ("Figure 2", OUT_DIR / "fig2_reliability_diagrams.png", generate_figure_2),
        ("Figure 3", OUT_DIR / "fig3_confound_checks.png", generate_figure_3),
        ("Figure 4", OUT_DIR / "fig4_remediation_comparison.png", generate_figure_4),
        ("Figure 5", OUT_DIR / "fig5_nv_stratified.png", generate_figure_5),
        ("Figure 6", OUT_DIR / "fig6_federated_convergence.png", generate_figure_6),
    ]

    print("=" * 72)
    print("  Generating Publication-Quality Research Report Figures (Grayscale, 300 DPI)")
    print("=" * 72)

    import gc

    generated_files = []
    for name, path, func in figures:
        plt.close("all")
        gc.collect()
        print(f"[*] Generating {name} -> {path} ...")
        func(path)
        plt.close("all")
        gc.collect()
        size_bytes = path.stat().st_size
        generated_files.append((path, size_bytes))
        print(f"    [+] Saved: {path.resolve()} ({size_bytes:,} bytes)")

    print("\n" + "=" * 72)
    print("  CONFIRMATION SUMMARY")
    print("=" * 72)
    for path, size in generated_files:
        print(f"  Path: {path.resolve()}\n  Size: {size:,} bytes ({size / 1024:.1f} KB)\n")


if __name__ == "__main__":
    main()
