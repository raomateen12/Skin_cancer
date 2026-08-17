"""
Train ResNet50 or EfficientNet-B0 on HAM10000 (run on Google Colab T4 GPU).
Reads config from configs/config.yaml.
Saves best checkpoint, training history CSV, and training curves.

Usage:
    python -m src.train --model_name efficientnet_b0
    python -m src.train --model_name efficientnet_b0 --use_ita_reweighting
        Loads train_with_ita.csv, combines class + ITA-group inverse-frequency
        weights into a WeightedRandomSampler.  Checkpoint saved as
        checkpoints/best_efficientnet_b0_reweighted.pth.
        Logs per-ITA-group validation Brier score each epoch.
"""

import argparse
import math
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset import HAM10000Dataset, get_train_transforms, get_eval_transforms, class_to_idx
from src.model import get_resnet50, get_efficientnet_b0


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def get_model(model_name, num_classes):
    if model_name == "resnet50":
        return get_resnet50(num_classes)
    elif model_name == "efficientnet_b0":
        return get_efficientnet_b0(num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def compute_class_weights(train_csv, num_classes):
    """Inverse frequency weights to handle class imbalance."""
    df = pd.read_csv(train_csv)
    counts = df["label_id"].value_counts().sort_index()
    total = len(df)
    weights = [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float)


def compute_ita_sample_weights(
    df: pd.DataFrame,
    class_weights_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Combine per-class inverse-frequency weight with per-ITA-group
    inverse-frequency weight into a single per-sample weight for
    WeightedRandomSampler.

    Design:
      - class_weight[i]     = total / (num_classes * count_of_class_i)
      - ita_weight[g]       = total_stable / (num_stable_groups * count_stable_in_g)
        where 'stable' means ita_formula_unstable == False
      - formula-unstable images get ita_weight = 1.0 (neutral; don't penalise
        or reward mislabelled ITA groups)
      - final per-sample weight = class_weight * ita_weight

    Parameters
    ----------
    df : DataFrame with columns label_id, ita_group, ita_formula_unstable
    class_weights_tensor : 1-D tensor of length num_classes

    Returns
    -------
    1-D FloatTensor of length len(df)
    """
    df = df.reset_index(drop=True)
    required = {"label_id", "ita_group", "ita_formula_unstable"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"compute_ita_sample_weights: missing columns {missing}")

    # ── ITA group weights (stable images only) ────────────────────────────────
    stable_mask = ~df["ita_formula_unstable"].astype(bool)
    stable_df   = df[stable_mask]
    stable_groups = [g for g in ["light", "intermediate", "dark"] if g != "unknown"]
    grp_counts = {
        g: int((stable_df["ita_group"] == g).sum()) for g in stable_groups
    }
    # Only weight groups that actually appear
    grp_counts = {g: n for g, n in grp_counts.items() if n > 0}
    total_stable     = int(stable_mask.sum())
    num_stable_groups = len(grp_counts)

    ita_weight_map: dict[str, float] = {}
    for g, n in grp_counts.items():
        ita_weight_map[g] = total_stable / (num_stable_groups * n)
    # Normalise so median stable weight = 1.0
    if ita_weight_map:
        med = float(np.median(list(ita_weight_map.values())))
        if med > 0:
            ita_weight_map = {g: w / med for g, w in ita_weight_map.items()}

    print("  ITA group weights (normalised, stable images only):")
    for g, w in ita_weight_map.items():
        print(f"    {g:15s}: {w:.4f}  (n_stable={grp_counts[g]})")
    print(f"    formula-unstable : 1.0000  (neutral, n={int((~stable_mask).sum())})")

    # ── Per-sample weight ─────────────────────────────────────────────────────
    sample_weights = []
    for _, row in df.iterrows():
        cw  = float(class_weights_tensor[int(row["label_id"])])
        grp = str(row["ita_group"])
        if bool(row["ita_formula_unstable"]) or grp not in ita_weight_map:
            iw = 1.0
        else:
            iw = ita_weight_map[grp]
        sample_weights.append(cw * iw)

    return torch.tensor(sample_weights, dtype=torch.float)


def collate_fn(batch):
    """Handles both (img, label) and (img, label, metadata) tuples."""
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    return images, labels


def train_one_epoch(model, loader, criterion, optimizer, device, use_amp, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  val  ", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)

    val_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    probs_arr = np.concatenate(all_probs, axis=0)  # (N, C)
    return total_loss / total, correct / total, val_f1, probs_arr, np.array(all_labels)


def brier_multiclass(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """Mean multiclass Brier score = mean( sum_c (p_c - y_c)^2 )."""
    one_hot = np.eye(num_classes)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def log_ita_group_brier(
    probs: np.ndarray,
    labels: np.ndarray,
    ita_groups: np.ndarray,
    num_classes: int,
) -> None:
    """Print per-ITA-group accuracy and Brier score for the validation set."""
    print("  Val Brier by ITA group:")
    for grp in ["light", "intermediate", "dark", "unknown"]:
        mask = ita_groups == grp
        n = int(mask.sum())
        if n == 0:
            continue
        acc   = float((probs[mask].argmax(axis=1) == labels[mask]).mean())
        brier = brier_multiclass(probs[mask], labels[mask], num_classes)
        flag  = " ⚠" if n < 30 else ""
        print(f"    {grp:13s}: n={n:>4d}  acc={acc:.4f}  Brier={brier:.4f}{flag}")


def save_curves(history, save_path, model_name):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title(f"{model_name} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Val")
    axes[1].set_title(f"{model_name} — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(epochs, history["val_f1"], color="green", label="Val F1")
    axes[2].set_title(f"{model_name} — Validation Weighted F1")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Curves saved → {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train skin lesion classifier on HAM10000")
    parser.add_argument("--model_name", type=str, default="resnet50",
                        choices=["resnet50", "efficientnet_b0"],
                        help="Model architecture to train")
    parser.add_argument("--use_ita_reweighting", action="store_true", default=False,
                        help="Combine class + ITA-group inverse-frequency weights via "
                             "WeightedRandomSampler.  Requires train_with_ita.csv. "
                             "Saves checkpoint as best_<model>_reweighted.pth.")
    parser.add_argument("--train_csv_override", type=str, default=None,
                        help="Override training CSV path (e.g. for smoke-test subset).")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override num_epochs from config.yaml.")
    args = parser.parse_args()
    model_name          = args.model_name
    use_ita_reweighting = args.use_ita_reweighting

    cfg = load_config()
    seed        = cfg.get("seed", 42)
    image_size  = cfg.get("image_size", 224)
    batch_size  = cfg.get("batch_size", 32)
    num_epochs  = cfg.get("num_epochs", 20)
    lr          = cfg.get("learning_rate", 1e-4)
    num_workers = cfg.get("num_workers", 2)
    patience    = 5

    # CLI overrides (useful for smoke-test / Colab)
    if args.epochs is not None:
        num_epochs = args.epochs

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model              : {model_name}")
    print(f"ITA reweighting    : {use_ita_reweighting}")
    print(f"Device             : {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    use_amp = device.type == "cuda"

    num_classes = len(class_to_idx)

    # ── Choose training CSV ───────────────────────────────────────────────────
    if args.train_csv_override:
        train_csv = args.train_csv_override
        print(f"  [override] train_csv = {train_csv}")
    elif use_ita_reweighting:
        train_csv = "data/processed/train_with_ita.csv"
        if not Path(train_csv).exists():
            raise FileNotFoundError(
                f"{train_csv} not found. "
                "Run: python -m src.label_ita_splits  first."
            )
    else:
        train_csv = "data/processed/train.csv"
    val_csv = "data/processed/val.csv"

    # ── Checkpoint path ───────────────────────────────────────────────────────
    if use_ita_reweighting:
        checkpoint_path = f"checkpoints/best_{model_name}_reweighted.pth"
    else:
        checkpoint_path = f"checkpoints/best_{model_name}.pth"

    train_dataset = HAM10000Dataset(train_csv, image_size, get_train_transforms(image_size))
    val_dataset   = HAM10000Dataset(val_csv,   image_size, get_eval_transforms(image_size))

    # ── Build DataLoaders ─────────────────────────────────────────────────────
    class_weights_cpu = compute_class_weights(train_csv, num_classes)

    if use_ita_reweighting:
        print("\n  Computing ITA-combined sample weights …")
        train_df_full = pd.read_csv(train_csv)
        sample_weights = compute_ita_sample_weights(train_df_full, class_weights_cpu)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=(device.type == "cuda"),
            collate_fn=collate_fn,
        )
        # Build val ITA group array for per-epoch Brier logging
        val_csv_ita = "data/processed/val_with_ita.csv"
        if Path(val_csv_ita).exists():
            val_ita_groups = pd.read_csv(val_csv_ita)["ita_group"].values
        else:
            val_ita_groups = None
            print("  WARNING: val_with_ita.csv not found — per-group Brier logging disabled.")
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=(device.type == "cuda"),
            collate_fn=collate_fn,
        )
        val_ita_groups = None

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    class_weights = class_weights_cpu.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = get_model(model_name, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler() if use_amp else None

    Path("checkpoints").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    suffix          = "_reweighted" if use_ita_reweighting else ""
    history_path    = f"results/{model_name}{suffix}_training_history.csv"
    curves_path     = f"results/{model_name}{suffix}_training_curves.png"

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_val_f1 = 0.0
    epochs_no_improve = 0

    print(f"\nStarting training: {num_epochs} epochs, batch_size={batch_size}, lr={lr}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}\n")

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_amp, scaler
        )
        val_loss, val_acc, val_f1, val_probs, val_labels = eval_one_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}")
        print(f"  val_loss={val_loss:.4f}    val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")

        # Per-ITA-group Brier logging (only when reweighting or val ITA labels available)
        if use_ita_reweighting and val_ita_groups is not None:
            log_ita_group_brier(val_probs, val_labels, val_ita_groups, num_classes)

        # Save best model based on highest val weighted F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save({
                "epoch":                epoch,
                "model_name":           model_name,
                "use_ita_reweighting":  use_ita_reweighting,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":             val_loss,
                "val_accuracy":         val_acc,
                "val_weighted_f1":      val_f1,
                "class_to_idx":         class_to_idx,
            }, checkpoint_path)
            print(f"  ✓ Best model saved (val_f1={val_f1:.4f}) → {checkpoint_path}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{patience} epochs")
            if epochs_no_improve >= patience:
                print("  Early stopping triggered.")
                break

    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"\nTraining history saved → {history_path}")

    save_curves(history, curves_path, model_name)
    print("Training complete.")


if __name__ == "__main__":
    main()
