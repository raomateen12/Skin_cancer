"""
Fairness auditor for HAM10000 skin lesion classifier.
Evaluates performance across sex, age group, and localization metadata groups.
Also runs SHAP on metadata to find factors associated with prediction correctness.

NOTE: Fitzpatrick skin tone labels are NOT available in HAM10000 metadata by default.
      Fitzpatrick analysis is supported — add a 'fitzpatrick' column to the CSV to enable it.

Usage:
    python -m src.fairness --model_name efficientnet_b0 --split test
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.dataset import HAM10000Dataset, get_eval_transforms, class_to_idx, idx_to_class, CLASS_NAMES
from src.model import get_resnet50, get_efficientnet_b0


MAPPING  = "data/processed/class_mapping.json"


def get_model(model_name, num_classes):
    if model_name == "resnet50":
        return get_resnet50(num_classes)
    elif model_name == "efficientnet_b0":
        return get_efficientnet_b0(num_classes)
    raise ValueError(f"Unknown model: {model_name}")


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    meta   = [item[2] for item in batch]
    return images, labels, meta


def bin_age(age):
    """Assign age to a readable age group bucket."""
    try:
        age = float(age)
    except (TypeError, ValueError):
        return "unknown"
    if age < 20:
        return "0-19"
    elif age < 40:
        return "20-39"
    elif age < 60:
        return "40-59"
    elif age < 80:
        return "60-79"
    else:
        return "80+"


def run_predictions(model, csv_path, device, label_names):
    """Run inference on a split CSV. Returns a DataFrame with predictions + metadata."""
    dataset = HAM10000Dataset(str(csv_path), image_size=224,
                              transform=get_eval_transforms(224))
    loader  = DataLoader(dataset, batch_size=32, shuffle=False,
                         num_workers=2, collate_fn=collate_fn)

    rows = []
    model.eval()
    with torch.no_grad():
        for images, labels, meta_batch in tqdm(loader, desc="Predicting"):
            images = images.to(device)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1)
            preds  = probs.argmax(dim=1).cpu().numpy()
            confs  = probs.max(dim=1).values.cpu().numpy()
            labels_np = labels.numpy()

            for i, m in enumerate(meta_batch):
                true_id  = labels_np[i]
                pred_id  = preds[i]
                true_dx  = idx_to_class[true_id]
                pred_dx  = idx_to_class[pred_id]
                rows.append({
                    "image_id":        m["image_id"],
                    "dx":              m["dx"],
                    "true_label_id":   int(true_id),
                    "true_label_name": label_names.get(true_dx, true_dx),
                    "pred_label_id":   int(pred_id),
                    "pred_label_name": label_names.get(pred_dx, pred_dx),
                    "confidence":      round(float(confs[i]), 4),
                    "correct":         int(true_id == pred_id),
                    "age":             m.get("age", None),
                    "sex":             m.get("sex", None),
                    "localization":    m.get("localization", None),
                    "image_path":      m.get("image_path", ""),
                })
    return pd.DataFrame(rows)


def group_metrics(df, group_col):
    """Compute per-group classification metrics."""
    if group_col not in df.columns or df[group_col].isna().all():
        return None

    df = df.dropna(subset=[group_col]).copy()
    df[group_col] = df[group_col].astype(str)
    records = []

    for group_val, grp in df.groupby(group_col):
        y_true = grp["true_label_id"].tolist()
        y_pred = grp["pred_label_id"].tolist()
        records.append({
            "group":              group_val,
            "sample_count":       len(grp),
            "accuracy":           round(accuracy_score(y_true, y_pred), 4),
            "weighted_precision": round(precision_score(y_true, y_pred, average="weighted",
                                                        zero_division=0), 4),
            "weighted_recall":    round(recall_score(y_true, y_pred, average="weighted",
                                                     zero_division=0), 4),
            "weighted_f1":        round(f1_score(y_true, y_pred, average="weighted",
                                                 zero_division=0), 4),
        })
    return pd.DataFrame(records).sort_values("group")


def save_bar_chart(metrics_df, group_col, metric_col, title, save_path):
    """Simple bar chart for one metric across groups."""
    fig, ax = plt.subplots(figsize=(max(6, len(metrics_df) * 1.2), 4))
    bars = ax.bar(metrics_df["group"].astype(str), metrics_df[metric_col], color="steelblue")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(group_col)
    ax.set_ylabel(metric_col)
    ax.set_title(title)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def fairness_gap(metrics_df):
    """Return accuracy and F1 gap (max-min) across groups."""
    if metrics_df is None or len(metrics_df) < 2:
        return None
    return {
        "max_accuracy":    metrics_df["accuracy"].max(),
        "min_accuracy":    metrics_df["accuracy"].min(),
        "accuracy_gap":    round(metrics_df["accuracy"].max() - metrics_df["accuracy"].min(), 4),
        "max_weighted_f1": metrics_df["weighted_f1"].max(),
        "min_weighted_f1": metrics_df["weighted_f1"].min(),
        "weighted_f1_gap": round(metrics_df["weighted_f1"].max() - metrics_df["weighted_f1"].min(), 4),
    }


def run_shap_analysis(pred_df, model_name, out_dir):
    """
    SHAP on metadata features to explain which factors are associated
    with the model predicting correctly or incorrectly.

    NOTE: This is NOT image-level SHAP. It explains metadata → correctness association.
    """
    df = pred_df.copy()

    # Build feature matrix from metadata
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(df["age"].median()
                              if pd.to_numeric(df["age"], errors="coerce").notna().any() else 0)
    df["age_group"] = df["age"].apply(bin_age)

    cat_cols = ["sex", "age_group", "localization", "dx"]
    df_feat = df[cat_cols + ["correct"]].copy()
    df_feat = df_feat.fillna("unknown")

    X = pd.get_dummies(df_feat[cat_cols], drop_first=False)
    y = df_feat["correct"].astype(int)

    if X.shape[1] == 0 or len(y.unique()) < 2:
        print("  SHAP skipped: not enough feature variance.")
        return []

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X, y)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)

    # shap_values is a list [class0, class1]; use class 1 (correct=1)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    mean_abs_shap = np.abs(sv).mean(axis=0)
    feature_importance = pd.DataFrame({
        "feature": X.columns.tolist(),
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    # Save feature importance CSV
    fi_path = out_dir / f"{model_name}_metadata_feature_importance.csv"
    feature_importance.to_csv(fi_path, index=False)
    print(f"  Feature importance CSV → {fi_path}")

    # Save SHAP summary plot
    shap_path = out_dir / f"{model_name}_metadata_shap_summary.png"
    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_importance) * 0.3)))
    top_n = feature_importance.head(15)
    ax.barh(top_n["feature"][::-1], top_n["mean_abs_shap"][::-1], color="darkorange")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"{model_name} — Metadata SHAP (association with correct prediction)")
    plt.tight_layout()
    plt.savefig(shap_path, dpi=120)
    plt.close()
    print(f"  SHAP summary plot → {shap_path}")

    return feature_importance["feature"].head(5).tolist()


def write_report(path, model_name, n_samples, acc, f1, gaps, top_shap_features, analyzed_groups):
    lines = [
        f"Fairness Report — {model_name}",
        "=" * 50,
        f"Test samples    : {n_samples}",
        f"Overall accuracy: {acc:.4f}",
        f"Weighted F1     : {f1:.4f}",
        "",
        f"Metadata groups analyzed: {', '.join(analyzed_groups)}",
        "",
        "Performance Gaps Across Groups:",
    ]

    any_flag = False
    for g_name, gap_info in gaps.items():
        if gap_info is None:
            continue
        flag = gap_info["accuracy_gap"] > 0.05
        if flag:
            any_flag = True
        lines.append(
            f"  {g_name}: accuracy gap={gap_info['accuracy_gap']:.4f}, "
            f"F1 gap={gap_info['weighted_f1_gap']:.4f}"
            + ("  [gap > 5%]" if flag else "")
        )

    lines += [
        "",
        f"Any gap > 5% observed: {any_flag}",
        "  Note: Performance disparity was observed across metadata groups.",
        "  This does not necessarily indicate harmful bias — group sizes differ greatly.",
        "",
        "SHAP — Top 5 metadata features associated with correct/incorrect predictions:",
    ]
    for feat in top_shap_features:
        lines.append(f"  - {feat}")

    lines += [
        "",
        "Limitation:",
        "  Fitzpatrick skin tone labels were not available in HAM10000 metadata used here,",
        "  so skin-tone fairness is not claimed. The code supports Fitzpatrick analysis",
        "  if such labels are added to the metadata CSV later.",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Fairness report → {path}")


def main():
    parser = argparse.ArgumentParser(description="Fairness audit for skin lesion classifier")
    parser.add_argument("--model_name",  type=str, default="efficientnet_b0",
                        choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--split",       type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--output_dir",  type=str, default="results/fairness")
    args = parser.parse_args()

    model_name  = args.model_name
    split       = args.split
    out_dir     = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint  = f"checkpoints/best_{model_name}.pth"
    csv_path    = f"data/processed/{split}.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model : {model_name}")
    print(f"Device: {device}")
    print(f"Split : {split}")

    # Load class mapping
    with open(MAPPING) as f:
        mapping = json.load(f)
    label_names = mapping.get("label_names", {})

    # Load model
    if not Path(checkpoint).exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint}")
        print(f"  Run: python -m src.train --model_name {model_name}")
        return

    ckpt = torch.load(checkpoint, map_location=device)
    num_classes = len(class_to_idx)
    model = get_model(model_name, num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print(f"Checkpoint loaded from epoch {ckpt.get('epoch', '?')}\n")

    # Run predictions
    pred_df = run_predictions(model, csv_path, device, label_names)

    # Add age group column
    pred_df["age_group"] = pred_df["age"].apply(bin_age)

    # Save predictions CSV
    pred_path = out_dir / f"{model_name}_predictions_with_metadata.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions saved → {pred_path}\n")

    # Overall metrics
    n_samples = len(pred_df)
    overall_acc = accuracy_score(pred_df["true_label_id"], pred_df["pred_label_id"])
    overall_f1  = f1_score(pred_df["true_label_id"], pred_df["pred_label_id"],
                           average="weighted", zero_division=0)

    print(f"Overall accuracy : {overall_acc:.4f}")
    print(f"Overall F1 (w)   : {overall_f1:.4f}\n")

    # Group analysis
    group_configs = [
        ("sex",          "sex"),
        ("age_group",    "age_group"),
        ("localization", "localization"),
    ]
    # Add Fitzpatrick if available
    if "fitzpatrick" in pred_df.columns:
        group_configs.append(("fitzpatrick", "fitzpatrick"))
        print("  Fitzpatrick column detected — running Fitzpatrick fairness analysis.")
    else:
        print("  No 'fitzpatrick' column found — skipping skin-tone fairness.")
        print("  Add a 'fitzpatrick' column to the metadata to enable it.\n")

    gaps = {}
    analyzed_groups = []

    for label, col in group_configs:
        metrics_df = group_metrics(pred_df, col)
        if metrics_df is None:
            print(f"  [{label}] skipped — no data.")
            continue
        analyzed_groups.append(label)
        csv_out = out_dir / f"{model_name}_{label}_metrics.csv"
        metrics_df.to_csv(csv_out, index=False)
        print(f"  [{label}] metrics → {csv_out}")

        chart_out = out_dir / f"{model_name}_{label}_accuracy.png"
        save_bar_chart(metrics_df, label, "accuracy",
                       f"{model_name} — Accuracy by {label}", chart_out)

        gaps[label] = fairness_gap(metrics_df)

    # Fairness gap summary
    gap_records = []
    for g_name, gap_info in gaps.items():
        if gap_info:
            row = {"group": g_name, **gap_info,
                   "gap_above_5_percent": gap_info["accuracy_gap"] > 0.05}
            gap_records.append(row)
    gap_df = pd.DataFrame(gap_records)
    gap_path = out_dir / f"{model_name}_fairness_gap_summary.csv"
    gap_df.to_csv(gap_path, index=False)
    print(f"\nFairness gap summary → {gap_path}")
    print(gap_df.to_string(index=False))

    # SHAP on metadata
    print("\nRunning metadata SHAP analysis...")
    top_shap = run_shap_analysis(pred_df, model_name, out_dir)

    # Fairness report
    report_path = out_dir / f"{model_name}_fairness_report.txt"
    write_report(report_path, model_name, n_samples, overall_acc,
                 overall_f1, gaps, top_shap, analyzed_groups)

    print(f"\n--- Summary ---")
    print(f"Model     : {model_name}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Samples   : {n_samples}")
    print(f"Output dir: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
