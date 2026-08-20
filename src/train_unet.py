"""
src/train_unet.py
Training pipeline for U-Net binary skin lesion segmentation on ISIC 2018 Task 1 data.

Features:
  - Combined BCE + Soft-Dice loss for balanced boundary optimization
  - Validation tracking of Dice coefficient (F1) and IoU (Jaccard Index)
  - Saves best checkpoint to checkpoints/best_unet.pth based on validation Dice
  - Generates training history CSV and convergence plot in results/
  - Mixed-precision (AMP) acceleration on CUDA GPU

Usage:
  python -m src.train_unet --epochs 20 --batch_size 16 --lr 1e-4
"""

from __future__ import annotations
import argparse
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.unet_model import get_unet, count_parameters
from src.segmentation_dataset import get_segmentation_loaders


# ── Loss Functions & Evaluation Metrics ───────────────────────────────────────

class SoftDiceLoss(nn.Module):
    """Soft Dice loss operating directly on sigmoid probabilities."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """Combined Binary Cross Entropy + Soft Dice Loss."""

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss


def compute_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> tuple[float, float]:
    """
    Compute binary Dice coefficient and IoU (Jaccard Index) for batch.

    Returns
    -------
    dice_score : float (0.0 to 1.0)
    iou_score  : float (0.0 to 1.0)
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum().item()
    total_p = preds_flat.sum().item()
    total_t = targets_flat.sum().item()
    union = total_p + total_t - intersection

    dice = (2.0 * intersection + eps) / (total_p + total_t + eps)
    iou = (intersection + eps) / (union + eps)

    return float(dice), float(iou)


# ── Training & Evaluation Loops ──────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> tuple[float, float, float]:
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    total_batches = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            with autocast():
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        dice, iou = compute_segmentation_metrics(logits.detach(), masks)

        running_loss += loss.item()
        running_dice += dice
        running_iou += iou
        total_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{dice:.4f}"})

    n = max(1, total_batches)
    return running_loss / n, running_dice / n, running_iou / n


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    total_batches = 0

    pbar = tqdm(loader, desc="Val", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)
        dice, iou = compute_segmentation_metrics(logits, masks)

        running_loss += loss.item()
        running_dice += dice
        running_iou += iou
        total_batches += 1

    n = max(1, total_batches)
    return running_loss / n, running_dice / n, running_iou / n


# ── Plotting Utilities ────────────────────────────────────────────────────────

def plot_training_curves(history_df: pd.DataFrame, out_path: Path):
    """Plot Loss, Dice, and IoU curves across training epochs."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = history_df["epoch"].values

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=120)

    # 1. Loss
    axes[0].plot(epochs, history_df["train_loss"], label="Train Loss", color="#0B7FEA", lw=2)
    axes[0].plot(epochs, history_df["val_loss"], label="Val Loss", color="#EF4444", lw=2, linestyle="--")
    axes[0].set_title("BCE + Dice Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 2. Dice Score
    axes[1].plot(epochs, history_df["train_dice"], label="Train Dice", color="#0B7FEA", lw=2)
    axes[1].plot(epochs, history_df["val_dice"], label="Val Dice", color="#10B981", lw=2, linestyle="--")
    axes[1].set_title("Dice Coefficient (F1 Score)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # 3. IoU Score
    axes[2].plot(epochs, history_df["train_iou"], label="Train IoU", color="#0B7FEA", lw=2)
    axes[2].plot(epochs, history_df["val_iou"], label="Val IoU", color="#8B5CF6", lw=2, linestyle="--")
    axes[2].set_title("IoU (Jaccard Index)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("IoU")
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved training curves plot to: {out_path}")


# ── Main Training Runner ──────────────────────────────────────────────────────

def run_training(args):
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    print(f"Using device: {device}")

    # Build DataLoaders
    train_loader, val_loader = get_segmentation_loaders(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        batch_size=args.batch_size,
        img_size=(args.img_size, args.img_size),
        num_workers=args.num_workers,
    )
    print(f"Loaded {len(train_loader.dataset):,} train images, {len(val_loader.dataset):,} val images.")

    # Build U-Net Model
    model = get_unet(
        in_channels=3,
        out_channels=1,
        base_channels=args.base_channels,
        bilinear=args.bilinear,
    ).to(device)

    params = count_parameters(model)
    print(f"Initialized U-Net model: {params['total']:,} total parameters ({params['trainable']:,} trainable).")

    # Optimization
    criterion = BCEDiceLoss(bce_weight=args.bce_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Setup directories
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = save_dir / "best_unet.pth"
    history_csv_path = results_dir / "unet_training_history.csv"
    plot_png_path = results_dir / "unet_training_curves.png"

    best_val_dice = -1.0
    history: list[dict] = []

    print(f"\nStarting U-Net training for {args.epochs} epoch(s)...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_dice, tr_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_dice, val_iou = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()
        elapsed = time.time() - t0

        record = {
            "epoch": epoch,
            "train_loss": round(tr_loss, 4),
            "train_dice": round(tr_dice, 4),
            "train_iou": round(tr_iou, 4),
            "val_loss": round(val_loss, 4),
            "val_dice": round(val_dice, 4),
            "val_iou": round(val_iou, 4),
            "lr": round(scheduler.get_last_lr()[0], 6),
            "time_sec": round(elapsed, 2),
        }
        history.append(record)

        is_best = val_dice > best_val_dice
        marker = " 🌟 [NEW BEST]" if is_best else ""
        if is_best:
            best_val_dice = val_dice
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_dice,
                "val_iou": val_iou,
                "val_loss": val_loss,
                "base_channels": args.base_channels,
                "img_size": args.img_size,
            }, best_checkpoint_path)

        print(
            f"Epoch {epoch:2d}/{args.epochs:2d} | "
            f"Train Loss: {tr_loss:.4f}, Dice: {tr_dice:.4f}, IoU: {tr_iou:.4f} | "
            f"Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f} | "
            f"Time: {elapsed:.1f}s{marker}"
        )

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time / 60:.2f} mins.")
    print(f"Best Validation Dice: {best_val_dice:.4f} (Saved to {best_checkpoint_path})")

    # Save history and plots
    history_df = pd.DataFrame(history)
    history_df.to_csv(history_csv_path, index=False)
    print(f"Saved training history to: {history_csv_path}")

    plot_training_curves(history_df, plot_png_path)

    return history_df, best_val_dice


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net on lesion segmentation data.")
    parser.add_argument("--train_csv", "--train-csv", type=str, default="data/processed/segmentation_train.csv")
    parser.add_argument("--val_csv", "--val-csv", type=str, default="data/processed/segmentation_val.csv")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", "--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay", "--weight-decay", type=float, default=1e-4)
    parser.add_argument("--img_size", "--img-size", type=int, default=224, help="Square image size")
    parser.add_argument("--base_channels", "--base-channels", type=int, default=32, help="U-Net base channel width")
    parser.add_argument("--bilinear", action="store_true", help="Use bilinear upsampling instead of transposed conv")
    parser.add_argument("--bce_weight", "--bce-weight", type=float, default=0.5, help="BCE weight in combined loss")
    parser.add_argument("--num_workers", "--num-workers", type=int, default=0)
    parser.add_argument("--save_dir", "--save-dir", type=str, default="checkpoints")
    parser.add_argument("--results_dir", "--results-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
