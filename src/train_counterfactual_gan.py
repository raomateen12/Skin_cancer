"""
src/train_counterfactual_gan.py
================================
Training pipeline for Counterfactual Explanation GAN (Progressive Exaggeration).
Learns continuous melanoma probability exaggeration guided by frozen EfficientNet-B0.

Features:
- Precomputes & caches baseline melanoma probabilities P(mel) for all training samples
- Periodic interval checkpointing (checkpoints/counterfactual_gan_epoch_{epoch}.pth)
- Tracks Generator loss, Discriminator loss, Reconstruction loss, Classifier consistency error
- Includes fast --smoke_test mode on synthetic data
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# Add project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.counterfactual_gan import (
    CounterfactualGAN,
    CounterfactualLoss,
    get_frozen_classifier,
    get_melanoma_probability,
    MELANOMA_CLASS_IDX,
)

logger = logging.getLogger("dermalens.cf_train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── Dataset for Counterfactual Training ───────────────────────────────────────

class CounterfactualDataset(Dataset):
    """
    Loads training dermoscopy images at 128x128 normalized to [-1, 1],
    paired with precomputed baseline classifier melanoma probabilities.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        mel_probs: np.ndarray,
        img_size: int = 128,
        augment: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.mel_probs = mel_probs.astype(np.float32)
        self.img_size = img_size
        self.augment = augment

        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # maps to [-1, 1]
        ])

        self.aug_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]) if augment else self.base_transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = str(row["image_path"])

        # Handle path separators
        if not Path(img_path).exists():
            img_path = str(ROOT / img_path.replace("\\", "/"))

        image = Image.open(img_path).convert("RGB")
        tensor = self.aug_transform(image)
        orig_prob = torch.tensor([self.mel_probs[idx]], dtype=torch.float32)

        return {
            "image": tensor,          # [3, 128, 128] in [-1, 1]
            "orig_prob": orig_prob,   # [1] in [0, 1]
        }


# ── Synthetic Dataset for Smoke Testing ───────────────────────────────────────

class SyntheticCounterfactualDataset(Dataset):
    """Generates synthetic random images and probabilities for fast pipeline testing."""
    def __init__(self, num_samples: int = 20, img_size: int = 128):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Random image in [-1, 1]
        tensor = torch.randn(3, self.img_size, self.img_size).clamp(-1.0, 1.0)
        orig_prob = torch.rand(1, dtype=torch.float32)
        return {
            "image": tensor,
            "orig_prob": orig_prob,
        }


# ── Probability Precomputation & Caching ──────────────────────────────────────

def precompute_melanoma_probabilities(
    df: pd.DataFrame,
    classifier: nn.Module,
    cache_path: Path,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Precompute and cache baseline melanoma probabilities for all training images."""
    if cache_path.exists():
        logger.info("Loading cached melanoma probabilities from: %s", cache_path)
        return np.load(cache_path)

    logger.info("Precomputing baseline melanoma probabilities for %d images...", len(df))
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    probs_list = []
    classifier.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(df), batch_size), desc="Precomputing P(mel)"):
            batch_df = df.iloc[i : i + batch_size]
            tensors = []
            for _, row in batch_df.iterrows():
                p = str(row["image_path"])
                if not Path(p).exists():
                    p = str(ROOT / p.replace("\\", "/"))
                img = Image.open(p).convert("RGB")
                tensors.append(eval_transform(img))

            batch_tensor = torch.stack(tensors).to(device)
            logits = classifier(batch_tensor)
            batch_probs = torch.softmax(logits, dim=1)[:, MELANOMA_CLASS_IDX].cpu().numpy()
            probs_list.extend(batch_probs.tolist())

    probs_arr = np.array(probs_list, dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, probs_arr)
    logger.info("Saved precomputed probabilities to: %s", cache_path)
    return probs_arr


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_counterfactual_gan(
    train_csv: Path = ROOT / "data/processed/train.csv",
    classifier_checkpoint: Path = ROOT / "checkpoints/best_efficientnet_b0.pth",
    checkpoint_dir: Path = ROOT / "checkpoints",
    epochs: int = 50,
    batch_size: int = 32,
    lr_g: float = 2e-4,
    lr_d: float = 1e-4,
    img_size: int = 128,
    save_every_epochs: int = 5,
    device_str: str = "cuda",
    smoke_test: bool = False,
) -> dict:
    """Run full Counterfactual GAN training pipeline."""
    device = torch.device(device_str if torch.cuda.is_available() and "cuda" in device_str else "cpu")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("      DERMALENS COUNTERFACTUAL GAN TRAINING (Progressive Exaggeration)")
    print("=" * 70)
    print(f"Device               : {device}")
    print(f"Image Resolution     : {img_size}x{img_size}")
    print(f"Batch Size           : {batch_size}")
    print(f"Epochs               : {epochs}")
    print(f"Learning Rate (G/D)  : {lr_g} / {lr_d}")
    print(f"Classifier Guidance  : {classifier_checkpoint.name}")
    print("=" * 70)

    # 1. Load Frozen Classifier
    if not classifier_checkpoint.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {classifier_checkpoint}")
    classifier = get_frozen_classifier(classifier_checkpoint, device)

    # 2. Build GAN Model & Loss
    gan = CounterfactualGAN(latent_dim=128, base_channels=64).to(device)
    loss_module = CounterfactualLoss(classifier=classifier).to(device)

    param_counts = gan.get_parameter_count()
    print("Model Parameter Counts:")
    for k, v in param_counts.items():
        print(f"  {k:15s}: {v:,} ({v/1e6:.2f}M)")
    print("=" * 70)

    # 3. Setup Optimizers
    opt_g = torch.optim.Adam(
        list(gan.encoder.parameters()) + list(gan.generator.parameters()),
        lr=lr_g,
        betas=(0.5, 0.999),
    )
    opt_d = torch.optim.Adam(
        gan.discriminator.parameters(),
        lr=lr_d,
        betas=(0.5, 0.999),
    )

    # 4. Dataset Setup
    if smoke_test:
        print("[SMOKE TEST MODE] Initializing synthetic dataset with 20 samples...")
        dataset = SyntheticCounterfactualDataset(num_samples=20, img_size=img_size)
    else:
        df_train = pd.read_csv(train_csv)
        cache_path = ROOT / "data/processed/train_melanoma_probs.npy"
        mel_probs = precompute_melanoma_probabilities(df_train, classifier, cache_path, device)
        dataset = CounterfactualDataset(df_train, mel_probs, img_size=img_size, augment=True)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    print(f"Loaded {len(dataset):,} training samples ({len(loader)} batches per epoch).")

    # 5. Training Epochs
    best_loss_cls = float("inf")
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        gan.encoder.train()
        gan.generator.train()
        gan.discriminator.train()

        total_loss_g = 0.0
        total_loss_d = 0.0
        total_loss_recon = 0.0
        total_loss_cls = 0.0
        total_loss_adv = 0.0
        num_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{epochs:02d}")
        for batch in pbar:
            real_imgs = batch["image"].to(device)       # [B, 3, 128, 128]
            orig_probs = batch["orig_prob"].to(device)  # [B, 1]
            B = real_imgs.size(0)

            # Random target melanoma probabilities in [0, 1] for progressive exaggeration
            target_probs = torch.rand(B, 1, device=device)

            # ── Step A: Forward Generator ────────────────────────────────────
            _, bottleneck = gan.encoder(real_imgs)
            fake_target_imgs = gan.generator(bottleneck, target_probs)
            recon_imgs = gan.generator(bottleneck, orig_probs)

            # ── Step B: Train Discriminator ──────────────────────────────────
            opt_d.zero_grad()
            pred_real_d = gan.discriminator(real_imgs)
            pred_fake_d = gan.discriminator(fake_target_imgs.detach())
            loss_d = loss_module.compute_discriminator_loss(pred_real_d, pred_fake_d)
            loss_d.backward()
            opt_d.step()

            # ── Step C: Train Generator & Encoder ────────────────────────────
            opt_g.zero_grad()
            pred_fake_d_for_g = gan.discriminator(fake_target_imgs)
            g_losses = loss_module.compute_generator_loss(
                real_img=real_imgs,
                orig_prob=orig_probs,
                target_prob=target_probs,
                fake_target_img=fake_target_imgs,
                recon_img=recon_imgs,
                pred_fake_disc=pred_fake_d_for_g,
            )
            loss_g = g_losses["loss_g"]
            loss_g.backward()
            opt_g.step()

            # Accumulate metrics
            total_loss_g += loss_g.item()
            total_loss_d += loss_d.item()
            total_loss_recon += g_losses["loss_recon"].item()
            total_loss_cls += g_losses["loss_cls"].item()
            total_loss_adv += g_losses["loss_adv"].item()
            num_batches += 1

            pbar.set_postfix({
                "L_G": f"{loss_g.item():.3f}",
                "L_D": f"{loss_d.item():.3f}",
                "L_cls": f"{g_losses['loss_cls'].item():.4f}",
                "L_rec": f"{g_losses['loss_recon'].item():.3f}",
            })

        avg_loss_g = total_loss_g / max(num_batches, 1)
        avg_loss_d = total_loss_d / max(num_batches, 1)
        avg_loss_cls = total_loss_cls / max(num_batches, 1)
        avg_loss_recon = total_loss_recon / max(num_batches, 1)

        print(
            f"[Epoch {epoch:02d}/{epochs:02d}] "
            f"Loss_G: {avg_loss_g:.4f} | Loss_D: {avg_loss_d:.4f} | "
            f"Loss_Cls: {avg_loss_cls:.4f} | Loss_Recon: {avg_loss_recon:.4f}"
        )

        epoch_stats = {
            "epoch": epoch,
            "loss_g": round(avg_loss_g, 4),
            "loss_d": round(avg_loss_d, 4),
            "loss_cls": round(avg_loss_cls, 4),
            "loss_recon": round(avg_loss_recon, 4),
        }
        history.append(epoch_stats)

        # Checkpoint Saving
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": gan.state_dict(),
            "opt_g_state_dict": opt_g.state_dict(),
            "opt_d_state_dict": opt_d.state_dict(),
            "loss_g": avg_loss_g,
            "loss_d": avg_loss_d,
            "loss_cls": avg_loss_cls,
            "param_counts": param_counts,
            "img_size": img_size,
        }

        # Save latest best
        if avg_loss_cls < best_loss_cls:
            best_loss_cls = avg_loss_cls
            best_path = checkpoint_dir / "best_counterfactual_gan.pth"
            torch.save(checkpoint_data, best_path)
            print(f"  * Saved new best counterfactual GAN (Loss_Cls: {avg_loss_cls:.4f}) -> {best_path.name}")

        # Periodic checkpoint
        if epoch % save_every_epochs == 0 or epoch == epochs:
            ckpt_path = checkpoint_dir / f"counterfactual_gan_epoch_{epoch:02d}.pth"
            torch.save(checkpoint_data, ckpt_path)
            print(f"  * Periodic checkpoint saved -> {ckpt_path.name}")

    print("=" * 70)
    print("Training finished successfully.")
    print("=" * 70)
    return {"history": history, "best_loss_cls": best_loss_cls, "param_counts": param_counts}


# ── CLI Interface ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train Counterfactual GAN (Progressive Exaggeration)")
    parser.add_argument("--train_csv", type=Path, default=ROOT / "data/processed/train.csv")
    parser.add_argument("--classifier_checkpoint", type=Path, default=ROOT / "checkpoints/best_efficientnet_b0.pth")
    parser.add_argument("--checkpoint_dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_g", type=float, default=2e-4)
    parser.add_argument("--lr_d", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--save_every_epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke_test", action="store_true", help="Run 2 fast epochs on synthetic data")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_counterfactual_gan(
        train_csv=args.train_csv,
        classifier_checkpoint=args.classifier_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        epochs=2 if args.smoke_test else args.epochs,
        batch_size=4 if args.smoke_test else args.batch_size,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        img_size=args.img_size,
        save_every_epochs=args.save_every_epochs,
        device_str=args.device,
        smoke_test=args.smoke_test,
    )
