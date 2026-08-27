"""
Federated Learning Simulation for Skin Lesion Classification on HAM10000.
========================================================================
Implements Federated Averaging (FedAvg; McMahan et al., 2017) across 3 simulated
hospital nodes (Rosendahl Hospital node and 2 Vienna General Hospital sub-nodes).

Demonstrates decentralized, privacy-preserving training without raw image sharing,
and benchmarks against a centralized pooled-data training baseline with identical
compute budget (R x E = 10 rounds x 2 local epochs = 20 total epochs).

Usage:
    # Full experiment (GPU / Colab / Kaggle):
    python -m src.federated_simulation --rounds 10 --local_epochs 2

    # Quick smoke test (CPU, 2 rounds, tiny synthetic/mini subset):
    python -m src.federated_simulation --smoke_test
"""

from __future__ import annotations

import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.dataset import (
    CLASS_NAMES,
    CLASSES,
    HAM10000Dataset,
    collate_image_only,
    get_eval_transforms,
    get_train_transforms,
)
from src.model import get_efficientnet_b0


# ── Synthetic Dataset for Smoke Testing ───────────────────────────────────────

class SyntheticLesionDataset(Dataset):
    """Generates synthetic RGB tensors with random labels for rapid smoke testing."""

    def __init__(self, num_samples: int = 30, image_size: int = 224, num_classes: int = 7):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        np.random.seed(42)
        self.labels = np.random.randint(0, num_classes, size=num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Deterministic synthetic image tensor for reproducibility
        torch.manual_seed(idx + 1000)
        img = torch.randn(3, self.image_size, self.image_size)
        label = int(self.labels[idx])
        return img, label


# ── Data Partitioning ────────────────────────────────────────────────────────

def partition_ham10000_nodes(
    train_csv_path: str,
) -> Tuple[Dict[int, pd.DataFrame], Dict[str, any]]:
    """
    Partitions HAM10000 training dataset into 3 simulated clinical hospital nodes:
      - Node 0 (Rosendahl Hospital node): HAM10000 part_2 images (~50% of data)
      - Node 1 (Vienna General Hospital - Clinic A): HAM10000 part_1 images (first half, ~25%)
      - Node 2 (Vienna General Hospital - Clinic B): HAM10000 part_1 images (second half, ~25%)
    """
    df = pd.read_csv(train_csv_path)

    # Check for explicit 'source' or infer from image_path
    if "source" in df.columns:
        rosendahl_mask = df["source"].astype(str).str.lower().str.contains("rosendahl")
        vienna_mask = ~rosendahl_mask
    else:
        # Infer from part_1 / part_2 in image_path
        rosendahl_mask = df["image_path"].astype(str).str.contains("part_2")
        vienna_mask = df["image_path"].astype(str).str.contains("part_1")

    df_rosendahl = df[rosendahl_mask].copy().reset_index(drop=True)
    df_vienna = df[vienna_mask].copy().reset_index(drop=True)

    mid_v = len(df_vienna) // 2
    df_vienna_1 = df_vienna.iloc[:mid_v].copy().reset_index(drop=True)
    df_vienna_2 = df_vienna.iloc[mid_v:].copy().reset_index(drop=True)

    node_dfs = {
        0: df_rosendahl,
        1: df_vienna_1,
        2: df_vienna_2,
    }

    node_names = {
        0: "Hospital Node 0 (Rosendahl Clinic)",
        1: "Hospital Node 1 (Vienna General - Cohort 1)",
        2: "Hospital Node 2 (Vienna General - Cohort 2)",
    }

    print("\n" + "=" * 80)
    print("  FEDERATED HOSPITAL NODE DATASET PARTITIONS & CLASS DISTRIBUTIONS")
    print("=" * 80)

    partition_stats = {}
    for node_id, node_df in node_dfs.items():
        total_samples = len(node_df)
        counts = node_df["dx"].value_counts().to_dict() if "dx" in node_df.columns else {}
        partition_stats[node_id] = {
            "name": node_names[node_id],
            "total_samples": total_samples,
            "class_distribution": counts,
        }
        print(f"\n[*] {node_names[node_id]}: Total {total_samples:,} images ({total_samples / len(df) * 100:.1f}%)")
        for cls_code in CLASSES:
            c = counts.get(cls_code, 0)
            pct = (c / max(1, total_samples)) * 100.0
            print(f"    - {cls_code.upper():<5} ({CLASS_NAMES.get(cls_code, cls_code):<24}): {c:>5d} ({pct:>5.1f}%)")

    print("=" * 80 + "\n")
    return node_dfs, partition_stats


# ── FedAvg Core Algorithm ────────────────────────────────────────────────────

def federated_average(
    local_weights_list: List[Dict[str, torch.Tensor]],
    sample_weights: List[int],
) -> Dict[str, torch.Tensor]:
    """
    Computes FedAvg weighted average of client weights:
      w_global = sum( (n_k / N) * w_k )
    """
    total_samples = sum(sample_weights)
    if total_samples == 0:
        raise ValueError("Total sample count cannot be zero for FedAvg aggregation.")

    aggregated_weights = {}
    # Initialize with zeros in matching tensor shapes & dtypes
    for key in local_weights_list[0].keys():
        first_tensor = local_weights_list[0][key]
        if first_tensor.dtype.is_floating_point:
            aggregated_weights[key] = torch.zeros_like(first_tensor)
        else:
            # For non-floating point buffers (e.g. num_batches_tracked in BatchNorm)
            aggregated_weights[key] = first_tensor.clone()

    # Accumulate weighted sums for floating point weights
    for weights, count in zip(local_weights_list, sample_weights):
        fraction = float(count) / float(total_samples)
        for key in weights.keys():
            if weights[key].dtype.is_floating_point:
                aggregated_weights[key] += fraction * weights[key]

    return aggregated_weights


# ── Local Training & Evaluation Functions ───────────────────────────────────

def train_local_client(
    client_id: int,
    initial_weights: Dict[str, torch.Tensor],
    loader: DataLoader,
    local_epochs: int,
    lr: float,
    device: torch.device,
    weight_decay: float = 1e-4,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Simulates a client hospital node training locally for `local_epochs`
    starting from broadcasted global model weights.
    Returns:
      updated_weights: state_dict after local SGD/Adam training
      avg_local_loss: average loss across local epochs
    """
    model = get_efficientnet_b0(num_classes=7, pretrained=False).to(device)
    model.load_state_dict(initial_weights)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    total_loss, total_steps = 0.0, 0

    for epoch in range(local_epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_steps += 1

    avg_loss = total_loss / max(1, total_steps)
    # Move updated weights to CPU for communication
    updated_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    return updated_weights, avg_loss


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluates model on validation DataLoader. Returns (val_loss, val_acc, val_f1)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = total_loss / max(1, total)
    val_acc = correct / max(1, total)
    val_f1 = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))
    return val_loss, val_acc, val_f1


# ── Main Simulation Routine ──────────────────────────────────────────────────

def run_federated_simulation(
    train_csv: str,
    val_csv: str,
    rounds: int = 10,
    local_epochs: int = 2,
    batch_size: int = 32,
    lr: float = 1e-4,
    device_str: str = "auto",
    checkpoint_dir: str = "checkpoints",
    results_dir: str = "results",
    smoke_test: bool = False,
):
    # Device setup
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"[*] Starting Federated Learning Simulation (FedAvg) on device: {device}")
    print(f"[*] Total Communication Rounds (R): {rounds} | Local Epochs per Round (E): {local_epochs}")
    print(f"[*] Total Equivalent Compute Budget: {rounds * local_epochs} epochs")

    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    res_path = Path(results_dir)
    res_path.mkdir(parents=True, exist_ok=True)

    # 1. Prepare Datasets & Loaders
    if smoke_test:
        print("\n[!] SMOKE TEST MODE ACTIVATED: Using synthetic datasets (30 samples per node).")
        node_loaders = {
            0: DataLoader(SyntheticLesionDataset(30), batch_size=8, shuffle=True),
            1: DataLoader(SyntheticLesionDataset(30), batch_size=8, shuffle=True),
            2: DataLoader(SyntheticLesionDataset(30), batch_size=8, shuffle=True),
        }
        val_loader = DataLoader(SyntheticLesionDataset(30), batch_size=8, shuffle=False)
        node_sample_counts = {0: 30, 1: 30, 2: 30}
        pooled_dataset = SyntheticLesionDataset(90)
        pooled_loader = DataLoader(pooled_dataset, batch_size=8, shuffle=True)
    else:
        node_dfs, _ = partition_ham10000_nodes(train_csv)
        node_loaders = {}
        node_sample_counts = {}

        # Create temporary subset CSVs or Datasets for each node
        for node_id, df_node in node_dfs.items():
            temp_csv = res_path / f"temp_node_{node_id}.csv"
            df_node.to_csv(temp_csv, index=False)
            ds = HAM10000Dataset(str(temp_csv), transform=get_train_transforms(224), use_metadata=False)
            node_loaders[node_id] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=collate_image_only,
                pin_memory=(device.type == "cuda"),
            )
            node_sample_counts[node_id] = len(df_node)

        # Centralized pooled training loader (all 8,012 images)
        pooled_ds = HAM10000Dataset(train_csv, transform=get_train_transforms(224), use_metadata=False)
        pooled_loader = DataLoader(
            pooled_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_image_only,
            pin_memory=(device.type == "cuda"),
        )

        # Centralized validation loader
        val_ds = HAM10000Dataset(val_csv, transform=get_eval_transforms(224), use_metadata=False)
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_image_only,
            pin_memory=(device.type == "cuda"),
        )

    # 2. Seed and Initialize Global Model
    torch.manual_seed(42)
    global_model = get_efficientnet_b0(num_classes=7, pretrained=not smoke_test).to(device)
    initial_weights = copy.deepcopy({k: v.cpu() for k, v in global_model.state_dict().items()})

    # Initial zero-shot evaluation
    init_loss, init_acc, init_f1 = evaluate_model(global_model, val_loader, device)
    print(f"[*] Initial Model State: Val Acc={init_acc * 100:.2f}%, Val F1={init_f1:.4f}")

    # 3. Federated Learning Execution (FedAvg)
    print("\n" + "=" * 92)
    print(f"{'Round':^7} | {'Node 0 Loss':^13} | {'Node 1 Loss':^13} | {'Node 2 Loss':^13} | {'Global Val Acc':^16} | {'Global Val F1':^15}")
    print("=" * 92)

    current_global_weights = copy.deepcopy(initial_weights)
    fed_history = []
    best_fed_f1 = -1.0
    best_fed_acc = -1.0

    for r in range(1, rounds + 1):
        local_weights_list = []
        sample_counts = []
        node_losses = {}

        # a. Broadcast and Local Training on Each Node
        for node_id in sorted(node_loaders.keys()):
            loader = node_loaders[node_id]
            updated_w, local_loss = train_local_client(
                client_id=node_id,
                initial_weights=current_global_weights,
                loader=loader,
                local_epochs=local_epochs,
                lr=lr,
                device=device,
            )
            local_weights_list.append(updated_w)
            sample_counts.append(node_sample_counts[node_id])
            node_losses[node_id] = local_loss

        # b. Server-side FedAvg Aggregation
        current_global_weights = federated_average(local_weights_list, sample_counts)

        # c. Global Model Evaluation on Centralized Val Set
        global_model.load_state_dict({k: v.to(device) for k, v in current_global_weights.items()})
        g_loss, g_acc, g_f1 = evaluate_model(global_model, val_loader, device)

        print(
            f"{r:^7d} | {node_losses[0]:^13.4f} | {node_losses[1]:^13.4f} | {node_losses[2]:^13.4f} | "
            f"{g_acc * 100:^15.2f}% | {g_f1:^15.4f}"
        )

        fed_history.append({
            "round": r,
            "node_0_loss": node_losses[0],
            "node_1_loss": node_losses[1],
            "node_2_loss": node_losses[2],
            "global_val_loss": g_loss,
            "global_val_acc": g_acc,
            "global_val_f1": g_f1,
        })

        # Save checkpoint every round (Kaggle disconnect prevention & best model tracking)
        is_best = g_f1 > best_fed_f1
        if is_best:
            best_fed_f1 = g_f1
            best_fed_acc = g_acc

        fed_ckpt_file = ckpt_path / "best_federated_efficientnet_b0.pth"
        torch.save(
            {
                "round": r,
                "model_state_dict": current_global_weights,
                "global_val_acc": g_acc,
                "global_val_f1": g_f1,
                "best_fed_acc": best_fed_acc,
                "best_fed_f1": best_fed_f1,
                "num_classes": 7,
            },
            fed_ckpt_file,
        )

    print("=" * 92)
    print(f"[+] Federated Training Complete. Best Global Val Acc: {best_fed_acc * 100:.2f}% (F1: {best_fed_f1:.4f})")
    print(f"[+] Saved Federated Checkpoint to: {fed_ckpt_file}")

    # 4. Centralized Comparison Baseline (Total Epochs = rounds * local_epochs)
    total_centralized_epochs = rounds * local_epochs
    print(f"\n[*] Training Centralized Comparison Model ({total_centralized_epochs} pooled epochs)...")

    central_model = get_efficientnet_b0(num_classes=7, pretrained=False).to(device)
    central_model.load_state_dict({k: v.to(device) for k, v in copy.deepcopy(initial_weights).items()})
    central_optimizer = torch.optim.AdamW(central_model.parameters(), lr=lr, weight_decay=1e-4)
    central_criterion = nn.CrossEntropyLoss()

    central_history = []
    best_cent_acc, best_cent_f1 = -1.0, -1.0

    for epoch in range(1, total_centralized_epochs + 1):
        central_model.train()
        c_loss_total, c_steps = 0.0, 0
        for images, labels in pooled_loader:
            images, labels = images.to(device), labels.to(device)
            central_optimizer.zero_grad()
            out = central_model(images)
            loss = central_criterion(out, labels)
            loss.backward()
            central_optimizer.step()
            c_loss_total += loss.item()
            c_steps += 1

        c_val_loss, c_val_acc, c_val_f1 = evaluate_model(central_model, val_loader, device)
        if c_val_f1 > best_cent_f1:
            best_cent_f1 = c_val_f1
            best_cent_acc = c_val_acc

        central_history.append({
            "epoch": epoch,
            "train_loss": c_loss_total / max(1, c_steps),
            "val_loss": c_val_loss,
            "val_acc": c_val_acc,
            "val_f1": c_val_f1,
        })

    print(f"[+] Centralized Baseline Complete. Final Val Acc: {best_cent_acc * 100:.2f}% (F1: {best_cent_f1:.4f})")

    # 5. Save Simulation Results CSV
    results_data = []
    for f_item in fed_history:
        r = f_item["round"]
        # Corresponding centralized epoch is r * local_epochs
        cent_equiv = central_history[(r * local_epochs) - 1] if (r * local_epochs) <= len(central_history) else central_history[-1]
        results_data.append({
            "communication_round": r,
            "equiv_central_epoch": r * local_epochs,
            "fed_val_loss": f_item["global_val_loss"],
            "fed_val_acc": f_item["global_val_acc"],
            "fed_val_f1": f_item["global_val_f1"],
            "centralized_val_loss": cent_equiv["val_loss"],
            "centralized_val_acc": cent_equiv["val_acc"],
            "centralized_val_f1": cent_equiv["val_f1"],
            "accuracy_gap": (cent_equiv["val_acc"] - f_item["global_val_acc"]),
        })

    res_df = pd.DataFrame(results_data)
    csv_file = res_path / "federated_simulation_results.csv"
    res_df.to_csv(csv_file, index=False)
    print(f"[+] Saved Detailed Metric Logs to: {csv_file}")

    # 6. Plot Learning Curves Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=140)

    # Plot Accuracy
    rounds_axis = res_df["communication_round"]
    ax1.plot(rounds_axis, res_df["fed_val_acc"] * 100.0, marker="o", color="#0B7FEA", label="Federated (FedAvg)")
    ax1.plot(rounds_axis, res_df["centralized_val_acc"] * 100.0, marker="s", color="#10B981", linestyle="--", label="Centralized (Pooled)")
    ax1.set_title("Validation Accuracy vs. Communication Rounds", fontweight="bold")
    ax1.set_xlabel("Communication Round (2 local epochs / round)")
    ax1.set_ylabel("Validation Accuracy (%)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Plot F1
    ax2.plot(rounds_axis, res_df["fed_val_f1"], marker="o", color="#0B7FEA", label="Federated (FedAvg)")
    ax2.plot(rounds_axis, res_df["centralized_val_f1"], marker="s", color="#10B981", linestyle="--", label="Centralized (Pooled)")
    ax2.set_title("Validation Weighted F1 vs. Communication Rounds", fontweight="bold")
    ax2.set_xlabel("Communication Round (2 local epochs / round)")
    ax2.set_ylabel("Validation Weighted F1")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.suptitle("DermaLens AI: Federated Learning vs. Centralized Training Simulation", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    curve_path = res_path / "federated_vs_centralized.png"
    plt.savefig(curve_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[+] Saved Comparison Curves to: {curve_path}")

    # Clean temporary files if any
    for node_id in range(3):
        t_csv = res_path / f"temp_node_{node_id}.csv"
        if t_csv.exists():
            t_csv.unlink()

    # 7. Print Final Publication-Ready Summary Table
    print("\n" + "=" * 80)
    print("                      RESEARCH COMPARISON SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Method':<40} | {'Final Val Acc':^14} | {'Final Val F1':^13} | {'Notes':<18}")
    print("-" * 80)
    print(f"{f'Centralized ({total_centralized_epochs} epochs pooled)':<40} | {best_cent_acc * 100:^13.2f}% | {best_cent_f1:^13.4f} | {'standard training':<18}")
    print(f"{f'Federated ({rounds} rounds x {local_epochs} local epochs)':<40} | {best_fed_acc * 100:^13.2f}% | {best_fed_f1:^13.4f} | {'no data sharing':<18}")
    print("=" * 80 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning Simulation (FedAvg) on HAM10000")
    parser.add_argument("--train_csv", type=str, default="data/processed/train.csv", help="Path to train CSV")
    parser.add_argument("--val_csv", type=str, default="data/processed/val.csv", help="Path to val CSV")
    parser.add_argument("--rounds", type=int, default=10, help="Number of communication rounds")
    parser.add_argument("--local_epochs", type=int, default=2, help="Local epochs per round per node")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda, cpu, auto)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save simulation logs and plots")
    parser.add_argument("--smoke_test", action="store_true", default=False, help="Run quick 2-round synthetic smoke test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke_test:
        run_federated_simulation(
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            rounds=2,
            local_epochs=1,
            batch_size=8,
            device_str="cpu",
            smoke_test=True,
        )
    else:
        run_federated_simulation(
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device_str=args.device,
            checkpoint_dir=args.checkpoint_dir,
            results_dir=args.results_dir,
            smoke_test=False,
        )
