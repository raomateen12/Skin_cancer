"""
Train ResNet50 on HAM10000 (designed to run on Google Colab T4 GPU).
Reads config from configs/config.yaml.
Saves best checkpoint, training history CSV, and loss/accuracy curves.
"""

import os
import json
import csv
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # no display needed on Colab/server
import matplotlib.pyplot as plt

from src.dataset import HAM10000Dataset, get_train_transforms, get_eval_transforms, class_to_idx
from src.model import get_resnet50


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def compute_class_weights(train_csv, num_classes):
    """Inverse frequency weights to handle class imbalance."""
    df = pd.read_csv(train_csv)
    counts = df["label_id"].value_counts().sort_index()
    total = len(df)
    weights = [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float)


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    return images, labels  # metadata not needed during training


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

    val_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return total_loss / total, correct / total, val_f1


def save_curves(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(epochs, history["val_f1"], color="green", label="Val F1")
    axes[2].set_title("Validation Weighted F1")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Curves saved → {save_path}")


def main():
    cfg = load_config()
    seed = cfg.get("seed", 42)
    image_size = cfg.get("image_size", 224)
    batch_size = cfg.get("batch_size", 32)
    num_epochs = cfg.get("num_epochs", 20)
    lr = cfg.get("learning_rate", 1e-4)
    num_workers = cfg.get("num_workers", 2)
    patience = 5

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    use_amp = device.type == "cuda"

    num_classes = len(class_to_idx)
    train_csv = "data/processed/train.csv"
    val_csv   = "data/processed/val.csv"

    train_dataset = HAM10000Dataset(train_csv, image_size, get_train_transforms(image_size))
    val_dataset   = HAM10000Dataset(val_csv,   image_size, get_eval_transforms(image_size))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=(device.type == "cuda"),
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=(device.type == "cuda"),
                              collate_fn=collate_fn)

    # Class-weighted loss for imbalanced HAM10000
    class_weights = compute_class_weights(train_csv, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = get_resnet50(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler() if use_amp else None

    Path("checkpoints").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    checkpoint_path = "checkpoints/best_resnet50.pth"
    history_path    = "results/resnet50_training_history.csv"
    curves_path     = "results/resnet50_training_curves.png"

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0

    print(f"\nStarting training: {num_epochs} epochs, batch_size={batch_size}, lr={lr}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}\n")

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_amp, scaler
        )
        val_loss, val_acc, val_f1 = eval_one_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}")
        print(f"  val_loss={val_loss:.4f}    val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_f1": val_f1,
                "class_to_idx": class_to_idx,
            }, checkpoint_path)
            print(f"  ✓ Best model saved (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{patience} epochs")
            if epochs_no_improve >= patience:
                print("  Early stopping triggered.")
                break

    # Save training history CSV
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"\nTraining history saved → {history_path}")

    save_curves(history, curves_path)
    print("Training complete.")


if __name__ == "__main__":
    main()
