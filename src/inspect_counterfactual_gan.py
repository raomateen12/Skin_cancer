"""
src/inspect_counterfactual_gan.py
===================================
Inspection and validation script for Counterfactual Explanation GAN.
Evaluates whether the generator produces genuine semantic visual modifications
(lesion border, color, morphology) or has collapsed to an adversarial noise shortcut.

Tasks performed:
1. Loads best_counterfactual_gan.pth and frozen EfficientNet-B0 classifier.
2. Selects 5 validation images with low baseline melanoma probability P(mel) < 0.2 (nv/bkl).
3. Generates counterfactuals at targets t in [0.1, 0.5, 0.9].
4. Measures actual classifier response P(mel) on generated counterfactual images.
5. Computes pixel-level L1 and L2 (RMSE) differences in [0, 1] and [0, 255] ranges.
6. Saves 4-panel visual comparison figures to results/counterfactual_inspection/.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Set project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.counterfactual_gan import (
    CounterfactualGAN,
    get_frozen_classifier,
    get_melanoma_probability,
    MELANOMA_CLASS_IDX,
)


def load_image_tensor(image_path: Path, img_size: int = 128) -> tuple[torch.Tensor, np.ndarray]:
    """
    Load image, resize to img_size x img_size, and return:
    - tensor: [1, 3, img_size, img_size] in range [-1, 1]
    - rgb_np: [img_size, img_size, 3] in range [0, 1]
    """
    pil_img = Image.open(image_path).convert("RGB")
    pil_img_resized = pil_img.resize((img_size, img_size), Image.Resampling.BILINEAR)
    rgb_np = np.array(pil_img_resized, dtype=np.float32) / 255.0

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    tensor = transform(pil_img_resized).unsqueeze(0)  # [1, 3, H, W]
    return tensor, rgb_np


def tensor_to_rgb_np(img_tensor_tanh: torch.Tensor) -> np.ndarray:
    """Convert [1, 3, H, W] in [-1, 1] to numpy [H, W, 3] in [0, 1]."""
    img_01 = (img_tensor_tanh.detach().cpu().squeeze(0) + 1.0) / 2.0
    img_01 = torch.clamp(img_01, 0.0, 1.0).permute(1, 2, 0).numpy()
    return img_01


def inspect_counterfactual_gan(
    gan_checkpoint: Path = ROOT / "checkpoints/best_counterfactual_gan.pth",
    classifier_checkpoint: Path = ROOT / "checkpoints/best_efficientnet_b0.pth",
    val_csv: Path = ROOT / "data/processed/val.csv",
    output_dir: Path = ROOT / "results/counterfactual_inspection",
    target_probs: list[float] = [0.1, 0.5, 0.9],
    num_samples: int = 5,
    seed: int = 42,
    device_str: str = "cpu",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str)

    print("=" * 80)
    print("      COUNTERFACTUAL GAN INSPECTION & ADVERSARIAL SHORTCUT ANALYSIS")
    print("=" * 80)
    print(f"Loading GAN checkpoint       : {gan_checkpoint}")
    print(f"Loading Classifier checkpoint: {classifier_checkpoint}")
    print(f"Validation dataset           : {val_csv}")

    # 1. Load Frozen Classifier
    classifier = get_frozen_classifier(classifier_checkpoint, device)
    print("Frozen EfficientNet-B0 classifier loaded successfully.")

    # 2. Load Counterfactual GAN
    gan_ckpt = torch.load(gan_checkpoint, map_location=device, weights_only=False)
    img_size = gan_ckpt.get("img_size", 128)
    gan = CounterfactualGAN(latent_dim=128, base_channels=64).to(device)
    gan.load_state_dict(gan_ckpt["model_state_dict"])
    gan.eval()
    print(f"Counterfactual GAN loaded successfully (Epoch {gan_ckpt.get('epoch', '?')}, img_size={img_size}).")

    # 3. Select 5 validation samples with low initial P(mel) (< 0.2)
    df_val = pd.read_csv(val_csv)
    print(f"Scanning validation set ({len(df_val)} images) for low-melanoma-risk samples...")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Filter to benign classes (nv, bkl) for clear counterfactual progression
    candidate_df = df_val[df_val["dx"].isin(["nv", "bkl", "df", "vasc"])].copy()
    candidate_df = candidate_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    selected_samples = []
    with torch.no_grad():
        for _, row in candidate_df.iterrows():
            img_path = Path(str(row["image_path"]).replace("\\", "/"))
            if not img_path.exists():
                img_path = ROOT / img_path
            if not img_path.exists():
                continue

            tensor, rgb_np = load_image_tensor(img_path, img_size)
            tensor = tensor.to(device)
            orig_p_mel = get_melanoma_probability(tensor, classifier).item()

            if orig_p_mel < 0.20:
                selected_samples.append({
                    "image_id": row["image_id"],
                    "dx": row["dx"],
                    "label_name": row.get("label_name", row["dx"]),
                    "image_path": img_path,
                    "tensor": tensor,
                    "rgb_np": rgb_np,
                    "orig_p_mel": orig_p_mel,
                })
                if len(selected_samples) >= num_samples:
                    break

    print(f"Selected {len(selected_samples)} validation samples with baseline P(mel) < 0.20.\n")

    # 4. Generate counterfactuals, measure classifier response & pixel differences
    summary_records = []

    print("-" * 80)
    print(f"{'Image ID':<14} | {'Target':<6} | {'P(mel)_orig':<11} | {'P(mel)_actual':<13} | {'L1 Diff [0-1]':<13} | {'L2 (RMSE)':<10} | {'L1 (0-255)':<10}")
    print("-" * 80)

    for idx, sample in enumerate(selected_samples):
        img_id = sample["image_id"]
        dx = sample["dx"]
        orig_tensor = sample["tensor"]
        orig_rgb = sample["rgb_np"]
        orig_p_mel = sample["orig_p_mel"]

        # Run Encoder once for this image
        with torch.no_grad():
            _, bottleneck = gan.encoder(orig_tensor)

        cf_results = []
        for target_p in target_probs:
            target_t = torch.tensor([[target_p]], dtype=torch.float32, device=device)
            with torch.no_grad():
                cf_tensor = gan.generator(bottleneck, target_t)
                actual_p_mel = get_melanoma_probability(cf_tensor, classifier).item()

            cf_rgb = tensor_to_rgb_np(cf_tensor)

            # Compute pixel-level differences (in [0, 1] range)
            diff_abs = np.abs(cf_rgb - orig_rgb)
            l1_diff = float(np.mean(diff_abs))
            l2_diff = float(np.sqrt(np.mean((cf_rgb - orig_rgb) ** 2)))
            l1_diff_255 = l1_diff * 255.0
            max_diff_255 = float(np.max(diff_abs) * 255.0)

            cf_results.append({
                "target_p": target_p,
                "actual_p_mel": actual_p_mel,
                "cf_rgb": cf_rgb,
                "l1_diff": l1_diff,
                "l2_diff": l2_diff,
                "l1_diff_255": l1_diff_255,
                "max_diff_255": max_diff_255,
            })

            print(
                f"{img_id:<14} | {target_p:<6.1f} | {orig_p_mel:<11.4f} | {actual_p_mel:<13.4f} | "
                f"{l1_diff:<13.6f} | {l2_diff:<10.6f} | {l1_diff_255:<10.2f}"
            )

            summary_records.append({
                "image_id": img_id,
                "diagnosis": dx,
                "target_p": target_p,
                "orig_p_mel": round(orig_p_mel, 4),
                "actual_p_mel": round(actual_p_mel, 4),
                "prob_error": round(abs(actual_p_mel - target_p), 4),
                "l1_diff_01": round(l1_diff, 6),
                "l2_rmse_01": round(l2_diff, 6),
                "l1_diff_255": round(l1_diff_255, 2),
                "max_diff_255": round(max_diff_255, 2),
            })

        # 5. Plot 4-panel figure: [Original | Target=0.1 | Target=0.5 | Target=0.9]
        fig, axes = plt.subplots(1, 4, figsize=(18, 5.0), dpi=140)

        # Panel 1: Original
        axes[0].imshow(orig_rgb)
        axes[0].set_title(
            f"Original ({img_id})\nDx: {dx} | P(mel) = {orig_p_mel:.3f}",
            fontsize=11,
            fontweight="bold",
            pad=8,
        )
        axes[0].axis("off")

        # Panels 2-4: Counterfactuals
        for p_idx, res in enumerate(cf_results):
            ax = axes[p_idx + 1]
            ax.imshow(res["cf_rgb"])
            target_val = res["target_p"]
            actual_val = res["actual_p_mel"]
            l1_val_255 = res["l1_diff_255"]
            l2_val = res["l2_diff"]

            title_color = "#0B7FEA" if target_val == 0.1 else ("#D97706" if target_val == 0.5 else "#DC2626")
            ax.set_title(
                f"Target P(mel) = {target_val:.1f}\nActual P(mel) = {actual_val:.3f}\n"
                f"L1: {l1_val_255:.1f}/255 | RMSE: {l2_val:.4f}",
                fontsize=10.5,
                fontweight="bold",
                pad=8,
                color=title_color,
            )
            ax.axis("off")

        plt.suptitle(
            f"Counterfactual GAN Inspection: {img_id} ({sample['label_name']})\n"
            f"Testing for Semantic Progression vs. Adversarial Shortcut",
            fontsize=13,
            fontweight="bold",
            y=1.04,
        )
        plt.tight_layout()

        out_figure_path = output_dir / f"cf_inspection_{img_id}.png"
        plt.savefig(out_figure_path, bbox_inches="tight", facecolor="white")
        plt.close()

    print("-" * 80)
    print(f"Saved {len(selected_samples)} comparison figures to: {output_dir}")

    # 6. Save summary CSV
    df_summary = pd.DataFrame(summary_records)
    csv_path = output_dir / "counterfactual_inspection_metrics.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"Saved metrics summary table to: {csv_path}")
    print("=" * 80)

    # 7. Overall diagnostic analysis
    mean_l1_01 = df_summary["l1_diff_01"].mean()
    mean_l1_255 = df_summary["l1_diff_255"].mean()
    mean_l2_01 = df_summary["l2_rmse_01"].mean()
    mean_prob_err = df_summary["prob_error"].mean()

    print("\n" + "=" * 80)
    print("                     DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print(f"Mean Classifier Tracking Error |Actual - Target| : {mean_prob_err:.4f}")
    print(f"Mean L1 Pixel Difference (0 to 1 scale)         : {mean_l1_01:.6f}")
    print(f"Mean L1 Pixel Difference (0 to 255 RGB scale)    : {mean_l1_255:.2f} intensity units (out of 255)")
    print(f"Mean L2 / RMSE Pixel Difference (0 to 1 scale)  : {mean_l2_01:.6f}")
    print("=" * 80)

    return df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Counterfactual GAN for Adversarial Shortcuts.")
    parser.add_argument("--gan_checkpoint", type=Path, default=ROOT / "checkpoints/best_counterfactual_gan.pth")
    parser.add_argument("--classifier_checkpoint", type=Path, default=ROOT / "checkpoints/best_efficientnet_b0.pth")
    parser.add_argument("--val_csv", type=Path, default=ROOT / "data/processed/val.csv")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "results/counterfactual_inspection")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    inspect_counterfactual_gan(
        gan_checkpoint=args.gan_checkpoint,
        classifier_checkpoint=args.classifier_checkpoint,
        val_csv=args.val_csv,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        device_str=args.device,
    )
