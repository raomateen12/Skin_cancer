"""
Train Multimodal Metadata Fusion Model (EfficientNet-B0 + Tabular Patient Metadata) on HAM10000.
Logs per-epoch metrics and compares against image-only baseline (best_efficientnet_b0.pth).

Usage:
    python -m src.train_metadata_fusion
    python -m src.train_metadata_fusion --freeze_backbone
    python -m src.train_metadata_fusion --epochs 2 --train_csv_override <path>
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import (
    CLASS_NAMES,
    CLASSES,
    HAM10000Dataset,
    collate_image_only,
    collate_with_metadata,
    get_eval_transforms,
    get_train_transforms,
)
from src.metadata_fusion_model import MetadataFusionModel, get_metadata_fusion_model
from src.model import get_efficientnet_b0


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multimodal Metadata Fusion Model on HAM10000")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--freeze_backbone", action="store_true", default=False, help="Freeze EfficientNet backbone")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path to processed CSVs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Path to save checkpoints")
    parser.add_argument(
        "--baseline_checkpoint",
        type=str,
        default="checkpoints/best_efficientnet_b0.pth",
        help="Path to image-only baseline checkpoint for comparison",
    )
    parser.add_argument("--train_csv_override", type=str, default=None, help="Override train CSV path for testing")
    parser.add_argument("--val_csv_override", type=str, default=None, help="Override val CSV path for testing")
    parser.add_argument("--stats_path", type=str, default="data/processed/metadata_stats.json", help="Metadata stats JSON")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use automatic mixed precision")
    return parser.parse_args()


def compute_class_weights(train_csv: str, num_classes: int = 7) -> torch.Tensor:
    df = pd.read_csv(train_csv)
    if "label_id" in df.columns:
        counts = df["label_id"].value_counts().sort_index()
    elif "dx" in df.columns:
        counts = df["dx"].map({cls: i for i, cls in enumerate(CLASSES)}).value_counts().sort_index()
    else:
        return torch.ones(num_classes, dtype=torch.float)

    total = len(df)
    weights = [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float)


def train_one_epoch_fusion(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
    scaler: GradScaler,
) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels, metadata in tqdm(loader, desc="  train (fusion)", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if use_amp and device.type == "cuda":
            with autocast():
                outputs = model(images, metadata)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images, metadata)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / max(1, total), correct / max(1, total)


def eval_one_epoch_fusion(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels, metadata in tqdm(loader, desc="  val (fusion)  ", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, metadata)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)

    val_f1 = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))
    probs_arr = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 7))
    return total_loss / max(1, total), correct / max(1, total), val_f1, probs_arr, np.array(all_labels)


def eval_baseline_image_only(
    baseline_model: Optional[nn.Module],
    val_csv: str,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[Optional[float], Optional[float]]:
    """Evaluates the image-only baseline on the validation dataset."""
    if baseline_model is None:
        return None, None

    baseline_model.eval()
    val_dataset = HAM10000Dataset(val_csv, transform=get_eval_transforms(224), use_metadata=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_image_only,
    )

    all_preds, all_labels = [], []
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="  val (baseline)", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = baseline_model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = correct / max(1, total)
    f1 = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))
    return acc, f1


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Multimodal Metadata Fusion Model on device: {device}")
    print(f"[*] Backbone: EfficientNet-B0 (Frozen: {args.freeze_backbone})")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    train_csv = args.train_csv_override or os.path.join(args.data_dir, "train.csv")
    val_csv = args.val_csv_override or os.path.join(args.data_dir, "val.csv")

    print(f"[*] Train dataset: {train_csv}")
    print(f"[*] Val dataset:   {val_csv}")

    # 1. Datasets & Loaders
    train_dataset = HAM10000Dataset(
        train_csv,
        transform=get_train_transforms(224),
        use_metadata=True,
    )
    val_dataset = HAM10000Dataset(
        val_csv,
        transform=get_eval_transforms(224),
        use_metadata=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_with_metadata,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_with_metadata,
        pin_memory=(device.type == "cuda"),
    )

    # 2. Model, Loss, Optimizer
    model = get_metadata_fusion_model(
        num_classes=7,
        freeze_backbone=args.freeze_backbone,
        pretrained_weights=True,
        stats_path=args.stats_path,
    ).to(device)

    class_weights = compute_class_weights(train_csv, num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer with different learning rates for backbone vs fusion head if not frozen
    if args.freeze_backbone:
        trainable_params = list(model.metadata_branch.parameters()) + list(model.fusion_head.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": model.features.parameters(), "lr": args.lr * 0.5},
                {"params": model.metadata_branch.parameters(), "lr": args.lr},
                {"params": model.fusion_head.parameters(), "lr": args.lr},
            ],
            weight_decay=args.weight_decay,
        )

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=(args.use_amp and device.type == "cuda"))

    # 3. Load baseline model for live comparison
    baseline_model = None
    baseline_path = Path(args.baseline_checkpoint)
    if baseline_path.exists():
        try:
            baseline_model = get_efficientnet_b0(num_classes=7).to(device)
            ckpt = torch.load(baseline_path, map_location=device, weights_only=False)
            baseline_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
            baseline_model.eval()
            print(f"[+] Loaded baseline image-only model from {baseline_path}")
        except Exception as e:
            print(f"[!] Warning: Could not load baseline model: {e}")
            baseline_model = None

    # Compute baseline metric once if static
    baseline_acc, baseline_f1 = None, None
    if baseline_model is not None:
        baseline_acc, baseline_f1 = eval_baseline_image_only(
            baseline_model,
            val_csv=val_csv,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print(f"[*] Baseline (Image-Only) Val Acc: {baseline_acc * 100.0:.2f}% | Val F1: {baseline_f1:.4f}")

    # 4. Training Loop
    best_val_f1 = -1.0
    best_val_acc = -1.0
    history = []

    print("\n" + "=" * 90)
    print(f"{'Epoch':^6} | {'Train Loss':^10} | {'Train Acc':^10} | {'Fusion Val Acc':^15} | {'Fusion Val F1':^14} | {'Base Acc':^10} | {'Delta':^8}")
    print("=" * 90)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch_fusion(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            use_amp=args.use_amp,
            scaler=scaler,
        )

        val_loss, val_acc, val_f1, _, _ = eval_one_epoch_fusion(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step()

        delta_str = "N/A"
        if baseline_acc is not None:
            delta = (val_acc - baseline_acc) * 100.0
            delta_str = f"{delta:+.2f}%"

        base_acc_str = f"{baseline_acc * 100.0:.2f}%" if baseline_acc is not None else "N/A"

        print(
            f"{epoch:^6d} | {train_loss:^10.4f} | {train_acc * 100.0:^9.2f}% | "
            f"{val_acc * 100.0:^14.2f}% | {val_f1:^14.4f} | {base_acc_str:^10} | {delta_str:^8}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "baseline_val_acc": baseline_acc,
            "baseline_val_f1": baseline_f1,
            "delta_acc": (val_acc - baseline_acc) if baseline_acc is not None else None,
        })

        # Save best model checkpoint
        if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and val_acc > best_val_acc):
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_ckpt_path = checkpoint_dir / "best_metadata_fusion.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "baseline_val_acc": baseline_acc,
                    "freeze_backbone": args.freeze_backbone,
                    "num_classes": 7,
                    "metadata_dim": model.metadata_dim,
                },
                best_ckpt_path,
            )

    print("=" * 90)
    print(f"[+] Training complete. Best Fusion Val F1: {best_val_f1:.4f} (Val Acc: {best_val_acc * 100.0:.2f}%)")
    print(f"[+] Saved best checkpoint to: {checkpoint_dir / 'best_metadata_fusion.pth'}")

    # 5. Save History & Training Curves
    history_df = pd.DataFrame(history)
    history_path = results_dir / "training_history_metadata_fusion.csv"
    history_df.to_csv(history_path, index=False)
    print(f"[+] Saved training history to: {history_path}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss", marker="o")
    ax1.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss", marker="s")
    ax1.set_title("Loss Curves (Metadata Fusion)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(history_df["epoch"], history_df["val_acc"] * 100.0, label="Fusion Val Acc", marker="o", color="#0B7FEA")
    if baseline_acc is not None:
        ax2.axhline(y=baseline_acc * 100.0, color="#DC2626", linestyle="--", label=f"Baseline Image-Only ({baseline_acc * 100.0:.1f}%)")
    ax2.set_title("Validation Accuracy Comparison")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    curves_path = results_dir / "training_curves_metadata_fusion.png"
    plt.tight_layout()
    plt.savefig(curves_path, dpi=140)
    plt.close()
    print(f"[+] Saved training curves to: {curves_path}")


if __name__ == "__main__":
    main()
