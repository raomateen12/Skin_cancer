"""
Evaluate the best saved ResNet50 checkpoint on the HAM10000 test set.
Saves metrics JSON, classification report, and confusion matrix PNG.
"""

import json
import csv
from pathlib import Path

import numpy as np
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

from src.dataset import HAM10000Dataset, get_eval_transforms, class_to_idx, idx_to_class
from src.model import get_resnet50


CHECKPOINT = "checkpoints/best_resnet50.pth"
TEST_CSV   = "data/processed/test.csv"
MAPPING    = "data/processed/class_mapping.json"
RESULTS    = Path("results")


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


def save_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("ResNet50 — Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    if not Path(CHECKPOINT).exists():
        print(f"[ERROR] Checkpoint not found: {CHECKPOINT}")
        print("  Run training first: python -m src.train")
        return

    ckpt = torch.load(CHECKPOINT, map_location=device)
    num_classes = len(class_to_idx)
    model = get_resnet50(num_classes)
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

    # Test dataloader
    test_dataset = HAM10000Dataset(TEST_CSV, image_size=224, transform=get_eval_transforms(224))
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False,
                              num_workers=2, collate_fn=collate_fn)
    print(f"Test samples: {len(test_dataset)}\n")

    # Run inference
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Compute metrics
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    rec  = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1   = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 (w)   : {f1:.4f}")

    # Class names for report
    label_names = load_class_names()
    sorted_classes = sorted(class_to_idx.keys())
    target_names = [label_names.get(c, c) for c in sorted_classes]

    report = classification_report(
        all_labels, all_preds,
        target_names=target_names,
        zero_division=0
    )
    print("\nClassification Report:")
    print(report)

    RESULTS.mkdir(exist_ok=True)

    # Save metrics JSON
    metrics = {
        "model": "resnet50",
        "checkpoint": CHECKPOINT,
        "test_samples": len(test_dataset),
        "accuracy": round(acc, 4),
        "precision_weighted": round(prec, 4),
        "recall_weighted": round(rec, 4),
        "f1_weighted": round(f1, 4),
    }
    metrics_path = RESULTS / "resnet50_test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved → {metrics_path}")

    # Save classification report
    report_path = RESULTS / "resnet50_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved → {report_path}")

    # Save/append to summary metrics.csv
    summary_path = RESULTS / "metrics.csv"
    write_header = not summary_path.exists()
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model_name", "accuracy", "precision", "recall", "f1_weighted"])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "model_name": "resnet50",
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_weighted": round(f1, 4),
        })
    print(f"Summary appended → {summary_path}")

    # Save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_path = RESULTS / "resnet50_confusion_matrix.png"
    save_confusion_matrix(cm, target_names, cm_path)


if __name__ == "__main__":
    main()
