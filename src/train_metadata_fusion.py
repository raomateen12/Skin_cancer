"""
Train Multimodal Metadata Fusion Model (EfficientNet-B0 + Tabular Patient Metadata) on HAM10000.

v2: Fixed differential learning rates with backbone warmup freeze strategy.
  - Backbone (EfficientNet-B0 features): backbone_lr = base_lr * 0.1 (default 1e-5)
  - Metadata MLP + fusion head: base_lr (default 1e-4)
  - First `warmup_freeze_epochs` epochs: backbone frozen entirely (metadata branch bootstraps)
  - From epoch warmup_freeze_epochs+1: backbone unfrozen with its lower lr

Logs per-epoch metrics and compares against image-only baseline (best_efficientnet_b0.pth).
Saves both checkpoints/best_metadata_fusion.pth AND checkpoints/best_metadata_fusion_v2.pth.
Prints metadata branch weight norms at end to confirm non-trivial learning.

Usage:
    # Full training (GPU recommended):
    python -m src.train_metadata_fusion

    # Quick smoke test (CPU, 2 epochs, small CSV):
    python -m src.train_metadata_fusion --epochs 2 \\
        --train_csv_override data/processed/smoke_train_100.csv \\
        --val_csv_override data/processed/smoke_val_50.csv \\
        --batch_size 16 --no_amp
"""

from __future__ import annotations

import argparse
import copy
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
from torch.cuda.amp import autocast
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
    parser = argparse.ArgumentParser(description="Train Multimodal Metadata Fusion Model on HAM10000 (v2: differential LR + warmup)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate for metadata branch + fusion head")
    parser.add_argument("--backbone_lr_ratio", type=float, default=0.1,
                        help="Ratio of backbone lr to base lr (default 0.1 → backbone_lr = lr * 0.1)")
    parser.add_argument("--warmup_freeze_epochs", type=int, default=2,
                        help="Number of initial epochs to freeze backbone entirely for metadata branch warmup")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--freeze_backbone", action="store_true", default=False,
                        help="Freeze EfficientNet backbone for entire training (overrides warmup)")
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
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use automatic mixed precision (GPU only)")
    parser.add_argument("--no_amp", action="store_true", default=False, help="Disable AMP (use on CPU)")
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


def _set_backbone_grad(model: nn.Module, requires_grad: bool) -> None:
    """Enable or disable gradient computation for the EfficientNet backbone."""
    for param in model.features.parameters():
        param.requires_grad = requires_grad
    for param in model.avgpool.parameters():
        param.requires_grad = requires_grad


def build_optimizer(
    model: nn.Module,
    base_lr: float,
    backbone_lr: float,
    weight_decay: float,
    backbone_frozen: bool,
) -> torch.optim.AdamW:
    """
    Build an AdamW optimizer with differential learning rates:
      - Backbone (features + avgpool): backbone_lr
      - Metadata branch + fusion head: base_lr

    When backbone_frozen=True, backbone params are excluded from optimizer entirely.
    """
    new_params = list(model.metadata_branch.parameters()) + list(model.fusion_head.parameters())

    if backbone_frozen:
        param_groups = [{"params": new_params, "lr": base_lr}]
    else:
        backbone_params = list(model.features.parameters()) + list(model.avgpool.parameters())
        param_groups = [
            {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
            {"params": new_params, "lr": base_lr, "name": "new_branches"},
        ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    return optimizer


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


def print_optimizer_lr_groups(optimizer: torch.optim.Optimizer, label: str = "") -> None:
    """Diagnostic: print learning rate of each parameter group."""
    prefix = f"[LR Check{' ' + label if label else ''}]"
    for i, pg in enumerate(optimizer.param_groups):
        name = pg.get("name", f"group_{i}")
        n_params = sum(p.numel() for p in pg["params"])
        print(f"  {prefix} param_group[{i}] name={name!r:20s}  lr={pg['lr']:.2e}  n_params={n_params:,}")


def print_metadata_branch_weight_norms(model: nn.Module) -> None:
    """
    Prints the weight norms of each trainable layer in the metadata branch and fusion head.
    If norms are near zero -> branch was degenerate (ignored). If well above zero -> branch learned.
    Random-init expected norm for Linear(11, 64) ~= sqrt(11) * kaiming_uniform ~= 3-6
    Trained norms should differ (higher or lower depending on regularization).
    """
    print("\n" + "=" * 75)
    print("  METADATA BRANCH & FUSION HEAD WEIGHT NORMS (post-training analysis)")
    print("=" * 75)
    print("  Interpretation:")
    print("    ~0.0  -> branch degenerate (gradients vanished / was ignored)")
    print("    >0.1  -> branch has learned non-trivial weights [OK]")
    print("    >1.0  -> strong signal absorbed by branch [GOOD]")
    print()

    for name, module in model.metadata_branch.named_modules():
        if isinstance(module, nn.Linear):
            w_norm = float(torch.norm(module.weight).item())
            b_norm = float(torch.norm(module.bias).item()) if module.bias is not None else 0.0
            print(f"  metadata_branch.{name}  weight_norm={w_norm:.4f}  bias_norm={b_norm:.4f}")

    for name, module in model.fusion_head.named_modules():
        if isinstance(module, nn.Linear):
            w_norm = float(torch.norm(module.weight).item())
            b_norm = float(torch.norm(module.bias).item()) if module.bias is not None else 0.0
            print(f"  fusion_head.{name}      weight_norm={w_norm:.4f}  bias_norm={b_norm:.4f}")

    print("=" * 75 + "\n")


def main():
    args = parse_args()
    use_amp = args.use_amp and not args.no_amp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_lr = args.lr * args.backbone_lr_ratio

    print(f"[*] Training Multimodal Metadata Fusion Model (v2 — Differential LR) on: {device}")
    print(f"[*] Base LR (metadata branch + fusion head): {args.lr:.2e}")
    print(f"[*] Backbone LR (EfficientNet-B0 features):  {backbone_lr:.2e}  (ratio={args.backbone_lr_ratio})")
    print(f"[*] Warmup freeze epochs (backbone frozen):  {args.warmup_freeze_epochs}")
    print(f"[*] Fully frozen backbone (--freeze_backbone): {args.freeze_backbone}")
    print(f"[*] AMP: {use_amp}")

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

    # 2. Model — backbone starts frozen for warmup
    model = get_metadata_fusion_model(
        num_classes=7,
        freeze_backbone=False,   # we manage freezing manually below
        pretrained_weights=True,
        stats_path=args.stats_path,
    ).to(device)

    # Phase 1: Backbone frozen during warmup
    warmup_active = (not args.freeze_backbone) and (args.warmup_freeze_epochs > 0)
    backbone_currently_frozen = True

    if args.freeze_backbone:
        # Fully frozen for entire run — classic frozen-backbone mode
        _set_backbone_grad(model, requires_grad=False)
        print("[*] Backbone: FROZEN for entire training (--freeze_backbone mode)")
    else:
        # Start with backbone frozen for warmup
        _set_backbone_grad(model, requires_grad=False)
        print(f"[*] Backbone: FROZEN for first {args.warmup_freeze_epochs} warmup epoch(s)")

    class_weights = compute_class_weights(train_csv, num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Build optimizer for Phase 1 (metadata branch + fusion head only) ──
    optimizer = build_optimizer(
        model=model,
        base_lr=args.lr,
        backbone_lr=backbone_lr,
        weight_decay=args.weight_decay,
        backbone_frozen=True,   # warmup: only new params
    )
    print("\n[*] Initial optimizer parameter groups (backbone frozen warmup):")
    print_optimizer_lr_groups(optimizer, label="warmup init")

    # Scheduler covers full training; we re-build optimizer after warmup so scheduler resets then
    try:
        # PyTorch ≥ 2.3 API
        scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))
    except TypeError:
        # Older PyTorch fallback
        from torch.cuda.amp import GradScaler as _GradScaler
        scaler = _GradScaler(enabled=(use_amp and device.type == "cuda"))

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

    # Compute baseline metric once (static reference)
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

    # Cosine annealing over remaining epochs after warmup
    remaining_epochs = args.epochs - args.warmup_freeze_epochs if warmup_active else args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, remaining_epochs), eta_min=1e-6)

    print("\n" + "=" * 98)
    print(
        f"{'Epoch':^6} | {'Phase':^12} | {'Train Loss':^10} | {'Train Acc':^10} | "
        f"{'Fusion Val Acc':^15} | {'Fusion Val F1':^14} | {'Base Acc':^10} | {'Delta':^8}"
    )
    print("=" * 98)

    for epoch in range(1, args.epochs + 1):

        # ── Phase transition: unfreeze backbone after warmup ──────────────────
        if warmup_active and epoch == args.warmup_freeze_epochs + 1 and backbone_currently_frozen:
            print(f"\n[*] Epoch {epoch}: UNFREEZING backbone (warmup complete). Rebuilding optimizer with differential LRs.")
            _set_backbone_grad(model, requires_grad=True)
            backbone_currently_frozen = False

            # Rebuild full optimizer with 2 param groups
            optimizer = build_optimizer(
                model=model,
                base_lr=args.lr,
                backbone_lr=backbone_lr,
                weight_decay=args.weight_decay,
                backbone_frozen=False,
            )
            print("[*] Post-warmup optimizer parameter groups:")
            print_optimizer_lr_groups(optimizer, label="post-warmup")

            # New scheduler for remaining epochs
            remaining = args.epochs - args.warmup_freeze_epochs
            scheduler = CosineAnnealingLR(optimizer, T_max=max(1, remaining), eta_min=1e-6)

        phase_label = "backbone-off" if backbone_currently_frozen else "full-train "

        train_loss, train_acc = train_one_epoch_fusion(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            use_amp=use_amp,
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
            f"{epoch:^6d} | {phase_label:^12s} | {train_loss:^10.4f} | {train_acc * 100.0:^9.2f}% | "
            f"{val_acc * 100.0:^14.2f}% | {val_f1:^14.4f} | {base_acc_str:^10} | {delta_str:^8}"
        )

        history.append({
            "epoch": epoch,
            "phase": phase_label.strip(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "baseline_val_acc": baseline_acc,
            "baseline_val_f1": baseline_f1,
            "delta_acc": (val_acc - baseline_acc) if baseline_acc is not None else None,
        })

        # Save best model — both canonical + versioned checkpoint
        is_best = val_f1 > best_val_f1 or (val_f1 == best_val_f1 and val_acc > best_val_acc)
        if is_best:
            best_val_f1 = val_f1
            best_val_acc = val_acc

            ckpt_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_f1": val_f1,
                "baseline_val_acc": baseline_acc,
                "freeze_backbone": args.freeze_backbone,
                "warmup_freeze_epochs": args.warmup_freeze_epochs,
                "backbone_lr_ratio": args.backbone_lr_ratio,
                "base_lr": args.lr,
                "backbone_lr": backbone_lr,
                "num_classes": 7,
                "metadata_dim": model.metadata_dim,
                "training_version": "v2_differential_lr",
            }

            best_ckpt_path = checkpoint_dir / "best_metadata_fusion.pth"
            torch.save(ckpt_payload, best_ckpt_path)

            v2_ckpt_path = checkpoint_dir / "best_metadata_fusion_v2.pth"
            torch.save(ckpt_payload, v2_ckpt_path)

    print("=" * 98)
    print(f"[+] Training complete. Best Fusion Val F1: {best_val_f1:.4f} (Val Acc: {best_val_acc * 100.0:.2f}%)")
    print(f"[+] Saved best checkpoint to:         {checkpoint_dir / 'best_metadata_fusion.pth'}")
    print(f"[+] Saved versioned checkpoint to:    {checkpoint_dir / 'best_metadata_fusion_v2.pth'}")

    # 5. Post-training: Metadata branch weight norm analysis
    print_metadata_branch_weight_norms(model)

    # 6. Save History & Training Curves
    history_df = pd.DataFrame(history)
    history_path = results_dir / "training_history_metadata_fusion_v2.csv"
    history_df.to_csv(history_path, index=False)
    print(f"[+] Saved training history to: {history_path}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss", marker="o")
    ax1.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss", marker="s")

    # Shade warmup zone
    if args.warmup_freeze_epochs > 0 and not args.freeze_backbone:
        ax1.axvspan(0.5, args.warmup_freeze_epochs + 0.5, alpha=0.08, color="#F59E0B", label="Backbone-frozen warmup")
        ax2.axvspan(0.5, args.warmup_freeze_epochs + 0.5, alpha=0.08, color="#F59E0B", label="Backbone-frozen warmup")

    ax1.set_title("Loss Curves (Metadata Fusion v2)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(history_df["epoch"], history_df["val_acc"] * 100.0, label="Fusion Val Acc (v2)", marker="o", color="#0B7FEA")
    if baseline_acc is not None:
        ax2.axhline(
            y=baseline_acc * 100.0,
            color="#DC2626",
            linestyle="--",
            label=f"Baseline Image-Only ({baseline_acc * 100.0:.1f}%)",
        )
    ax2.set_title("Validation Accuracy (v2: differential LR + warmup)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    curves_path = results_dir / "training_curves_metadata_fusion_v2.png"
    plt.tight_layout()
    plt.savefig(curves_path, dpi=140, facecolor="white")
    plt.close()
    print(f"[+] Saved training curves to: {curves_path}")


if __name__ == "__main__":
    main()
