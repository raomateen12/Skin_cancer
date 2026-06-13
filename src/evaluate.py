"""
Evaluate a trained model checkpoint on the HAM10000 test set.
Saves metrics JSON, classification report, and confusion matrix PNG.
Updates results/metrics.csv (one row per model, no duplicates).

Usage:
    python -m src.evaluate --model_name resnet50
    python -m src.evaluate --model_name efficientnet_b0
"""

import argparse
import json
import csv
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.dataset import HAM10000Dataset, get_eval_transforms, class_to_idx
from src.model import get_resnet50, get_efficientnet_b0


TEST_CSV = "data/processed/test.csv"
MAPPING  = "data/processed/class_mapping.json"
RESULTS  = Path("results")

METRICS_COLS = [
    "model_name", "accuracy", "weighted_precision", "weighted_recall",
    "weighted_f1", "checkpoint_epoch", "saved_val_accuracy", "saved_val_weighted_f1"
]


def get_model(model_name, num_classes):
    if model_name == "resnet50":
        return get_resnet50(num_classes)
    elif model_name == "efficientnet_b0":
        return get_efficientnet_b0(num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    return images, labels


def load_class_names():
    if Path(MAPPING).exists():
        with open(MAPPING) as f:
            m = json.load(f)
        return m.get("label_names", {})
    return {}


def save_confusion_matrix(cm, class_names, save_path, model_name):
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_name} — Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {save_path}")


def update_metrics_csv(row, summary_path):
    """Write or update one row in metrics.csv — no duplicate model rows."""
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        # Drop existing row for this model if any
        df = df[df["model_name"] != row["model_name"]]
    else:
        df = pd.DataFrame(columns=METRICS_COLS)

    new_row = pd.DataFrame([row])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(summary_path, index=False)
    print(f"metrics.csv updated → {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained skin lesion classifier")
    parser.add_argument("--model_name", type=str, default="resnet50",
                        choices=["resnet50", "efficientnet_b0"],
                        help="Model architecture to evaluate")
    args = parser.parse_args()
    model_name = args.model_name

    checkpoint_path = f"checkpoints/best_{model_name}.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model : {model_name}")
    print(f"Device: {device}")

    if not Path(checkpoint_path).exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        print(f"  Run: python -m src.train --model_name {model_name}")
        return

    ckpt = torch.load(checkpoint_path, map_location=device)
    num_classes = len(class_to_idx)
    model = get_model(model_name, num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    saved_epoch = ckpt.get("epoch", "?")
    saved_f1    = ckpt.get("val_weighted_f1", ckpt.get("val_f1", None))  # backward compat
    saved_acc   = ckpt.get("val_accuracy", None)
    print(f"Loaded checkpoint from epoch {saved_epoch}")
    if saved_f1 is not None:
        print(f"  Saved val_weighted_f1 : {saved_f1:.4f}")
    if saved_acc is not None:
        print(f"  Saved val_accuracy    : {saved_acc:.4f}")

    test_dataset = HAM10000Dataset(TEST_CSV, image_size=224, transform=get_eval_transforms(224))
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False,
                              num_workers=2, collate_fn=collate_fn)
    print(f"\nTest samples: {len(test_dataset)}\n")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    rec  = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1   = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 (w)   : {f1:.4f}")

    label_names = load_class_names()
    sorted_classes = sorted(class_to_idx.keys())
    target_names = [label_names.get(c, c) for c in sorted_classes]

    report = classification_report(all_labels, all_preds,
                                   target_names=target_names, zero_division=0)
    print("\nClassification Report:")
    print(report)

    RESULTS.mkdir(exist_ok=True)

    # Save per-model metrics JSON
    metrics = {
        "model": model_name,
        "checkpoint": checkpoint_path,
        "checkpoint_epoch": saved_epoch,
        "test_samples": len(test_dataset),
        "accuracy": round(acc, 4),
        "weighted_precision": round(prec, 4),
        "weighted_recall": round(rec, 4),
        "weighted_f1": round(f1, 4),
    }
    metrics_path = RESULTS / f"{model_name}_test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved → {metrics_path}")

    # Save classification report
    report_path = RESULTS / f"{model_name}_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved → {report_path}")

    # Update summary metrics.csv (no duplicate rows per model)
    summary_row = {
        "model_name":           model_name,
        "accuracy":             round(acc, 4),
        "weighted_precision":   round(prec, 4),
        "weighted_recall":      round(rec, 4),
        "weighted_f1":          round(f1, 4),
        "checkpoint_epoch":     saved_epoch,
        "saved_val_accuracy":   round(saved_acc, 4) if saved_acc is not None else "",
        "saved_val_weighted_f1": round(saved_f1, 4) if saved_f1 is not None else "",
    }
    update_metrics_csv(summary_row, RESULTS / "metrics.csv")

    # Save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_path = RESULTS / f"{model_name}_confusion_matrix.png"
    save_confusion_matrix(cm, target_names, cm_path, model_name)


if __name__ == "__main__":
    main()
