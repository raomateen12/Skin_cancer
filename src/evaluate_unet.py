"""
src/evaluate_unet.py
Evaluation script for U-Net lesion segmentation model.

Features:
  - Computes global & per-image Dice, IoU, Sensitivity, Specificity, and Pixel Accuracy
  - Generates multi-panel visual overlay comparisons (Image, GT, Prediction, Contour Overlay)
  - Saves visualizations to results/unet_sample_predictions/

Usage:
  python -m src.evaluate_unet --checkpoint checkpoints/best_unet.pth --data_csv data/processed/segmentation_val.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.unet_model import get_unet
from src.segmentation_dataset import SegmentationDataset, IMAGENET_MEAN, IMAGENET_STD


def compute_detailed_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray, eps: float = 1e-7) -> dict[str, float]:
    """Compute binary classification metrics between 2D boolean/binary masks."""
    p = (pred_bin > 0.5).astype(np.float32)
    y = (gt_bin > 0.5).astype(np.float32)

    tp = float(np.sum(p * y))
    fp = float(np.sum(p * (1.0 - y)))
    fn = float(np.sum((1.0 - p) * y))
    tn = float(np.sum((1.0 - p) * (1.0 - y)))

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    sensitivity = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
    }


def draw_contour_overlay(
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
) -> np.ndarray:
    """
    Draw Ground Truth (Green) and Predicted (Red) boundary contours on original image.
    """
    overlay = image_rgb.copy()

    # Ground Truth contours in Bright Green (0, 255, 0)
    gt_uint8 = (gt_mask > 0.5).astype(np.uint8) * 255
    contours_gt, _ = cv2.findContours(gt_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours_gt, -1, (0, 255, 0), 2)

    # Predicted contours in Bright Red / Coral (255, 50, 50)
    pred_uint8 = (pred_mask > 0.5).astype(np.uint8) * 255
    contours_pred, _ = cv2.findContours(pred_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours_pred, -1, (255, 50, 50), 2)

    return overlay


def save_prediction_visualization(
    img_tensor: torch.Tensor,
    gt_tensor: torch.Tensor,
    pred_probs: torch.Tensor,
    image_id: str,
    metrics: dict[str, float],
    out_path: Path,
):
    """Save 4-panel visual comparison: Original, Ground Truth, Predicted Mask, Contour Overlay."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Un-normalize image tensor for display
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)
    img_np = img_tensor.cpu().numpy() * std + mean
    img_np = np.clip(img_np, 0.0, 1.0).transpose(1, 2, 0)  # [H, W, 3]
    img_uint8 = (img_np * 255.0).astype(np.uint8)

    gt_np = gt_tensor.squeeze().cpu().numpy()
    pred_prob_np = pred_probs.squeeze().cpu().numpy()
    pred_bin_np = (pred_prob_np > 0.5).astype(np.float32)

    overlay_img = draw_contour_overlay(img_uint8, gt_np, pred_bin_np)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), dpi=120)

    # 1. Original
    axes[0].imshow(img_uint8)
    axes[0].set_title(f"Image: {image_id}", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # 2. Ground Truth
    axes[1].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth Mask", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # 3. Predicted Probability Map
    im = axes[2].imshow(pred_prob_np, cmap="magma", vmin=0, vmax=1)
    axes[2].set_title(f"Prediction (Dice: {metrics['dice']:.3f})", fontsize=12, fontweight="bold")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Contour Overlay
    axes[3].imshow(overlay_img)
    axes[3].set_title("Boundary Overlay\n(Green: GT, Red: Pred)", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def evaluate_unet(
    checkpoint_path: Path,
    data_csv: Path,
    output_dir: Path,
    num_samples: int = 10,
    threshold: float = 0.5,
    device_str: str = "cuda",
) -> pd.DataFrame:
    """Run full evaluation on dataset and save metrics summary & sample visualizations."""
    device = torch.device(device_str if torch.cuda.is_available() and "cuda" in device_str else "cpu")
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Evaluation dataset: {data_csv}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    base_channels = checkpoint.get("base_channels", 32)
    img_size = checkpoint.get("img_size", 224)

    # Load Model
    model = get_unet(in_channels=3, out_channels=1, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded U-Net (base_channels={base_channels}, img_size={img_size}).")

    # Load Dataset
    dataset = SegmentationDataset(data_csv, img_size=(img_size, img_size), is_train=False)
    print(f"Evaluating {len(dataset):,} samples...")

    sample_save_dir = output_dir / "unet_sample_predictions"
    sample_save_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Evaluating"):
            item = dataset[i]
            img = item["image"].unsqueeze(0).to(device)  # [1, 3, H, W]
            gt = item["mask"]                             # [1, H, W]
            image_id = item["image_id"]

            logits = model(img)
            probs = torch.sigmoid(logits).squeeze(0)      # [1, H, W]
            preds_bin = (probs > threshold).float()

            pred_np = preds_bin.squeeze().cpu().numpy()
            gt_np = gt.squeeze().cpu().numpy()

            metrics = compute_detailed_metrics(pred_np, gt_np)
            records.append({
                "image_id": image_id,
                "dice": round(metrics["dice"], 4),
                "iou": round(metrics["iou"], 4),
                "sensitivity": round(metrics["sensitivity"], 4),
                "specificity": round(metrics["specificity"], 4),
                "accuracy": round(metrics["accuracy"], 4),
            })

            # Save visual sample overlays for the first `num_samples`
            if i < num_samples:
                out_png = sample_save_dir / f"pred_{image_id}.png"
                save_prediction_visualization(
                    img_tensor=item["image"],
                    gt_tensor=gt,
                    pred_probs=probs,
                    image_id=image_id,
                    metrics=metrics,
                    out_path=out_png,
                )

    results_df = pd.DataFrame(records)
    summary_csv = output_dir / "unet_evaluation_summary.csv"
    results_df.to_csv(summary_csv, index=False)

    print("\n" + "=" * 60)
    print("           U-NET SEGMENTATION EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Evaluated Samples : {len(results_df):,}")
    print(f"Mean Dice Score (F1)   : {results_df['dice'].mean():.4f} ± {results_df['dice'].std():.4f}")
    print(f"Mean IoU (Jaccard Index): {results_df['iou'].mean():.4f} ± {results_df['iou'].std():.4f}")
    print(f"Mean Sensitivity        : {results_df['sensitivity'].mean():.4f}")
    print(f"Mean Specificity        : {results_df['specificity'].mean():.4f}")
    print(f"Mean Pixel Accuracy     : {results_df['accuracy'].mean():.4f}")
    print(f"Saved sample plots to   : {sample_save_dir}")
    print(f"Saved metrics summary to: {summary_csv}")
    print("=" * 60)

    return results_df


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained U-Net segmentation model.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_unet.pth"))
    parser.add_argument("--data_csv", "--data-csv", type=Path, default=Path("data/processed/segmentation_val.csv"))
    parser.add_argument("--output_dir", "--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--num_samples", "--num-samples", type=int, default=10, help="Number of visual overlay samples to save")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary classification threshold")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_unet(
        checkpoint_path=args.checkpoint,
        data_csv=args.data_csv,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        threshold=args.threshold,
        device_str=args.device,
    )
