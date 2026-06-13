"""
Compare trained models using results/metrics.csv.
Prints a comparison table, saves model_comparison.csv and a bar chart.

Usage:
    python -m src.compare_models
"""

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS      = Path("results")
METRICS_CSV  = RESULTS / "metrics.csv"
COMPARE_CSV  = RESULTS / "model_comparison.csv"
COMPARE_PNG  = RESULTS / "model_comparison_bar.png"

METRIC_COLS = ["accuracy", "weighted_precision", "weighted_recall", "weighted_f1"]


def print_table(df):
    print("\n=== Model Comparison ===\n")
    col_w = 22
    header = f"{'model':<{col_w}}" + "".join(f"{c:<{col_w}}" for c in METRIC_COLS)
    print(header)
    print("-" * len(header))
    for _, row in df.iterrows():
        line = f"{row['model_name']:<{col_w}}"
        for c in METRIC_COLS:
            val = row.get(c, "")
            line += f"{val:<{col_w}}"
        print(line)
    print()


def save_bar_chart(df):
    x = range(len(df))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, col in enumerate(METRIC_COLS):
        vals = df[col].tolist()
        offset = (i - len(METRIC_COLS) / 2) * width + width / 2
        bars = ax.bar([xi + offset for xi in x], vals, width, label=col)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["model_name"].tolist())
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("ResNet50 vs EfficientNet-B0 — Test Set Metrics")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(COMPARE_PNG, dpi=150)
    plt.close()
    print(f"Bar chart saved → {COMPARE_PNG}")


def main():
    if not METRICS_CSV.exists():
        print(f"[ERROR] {METRICS_CSV} not found.")
        print("  Run: python -m src.evaluate --model_name resnet50")
        print("  Run: python -m src.evaluate --model_name efficientnet_b0")
        return

    df = pd.read_csv(METRICS_CSV)
    if df.empty:
        print("metrics.csv is empty. Run evaluation for at least one model first.")
        return

    # Keep only the metric columns we need for comparison
    display_cols = ["model_name"] + METRIC_COLS
    available = [c for c in display_cols if c in df.columns]
    df_display = df[available].copy()

    print_table(df_display)

    RESULTS.mkdir(exist_ok=True)
    df_display.to_csv(COMPARE_CSV, index=False)
    print(f"Comparison CSV saved → {COMPARE_CSV}")

    if len(df_display) >= 1:
        save_bar_chart(df_display)


if __name__ == "__main__":
    main()
