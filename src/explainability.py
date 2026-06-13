"""
Generate Grad-CAM and EigenCAM heatmap overlays for trained skin lesion models.
Saves combined 3-panel figures (Original | Grad-CAM | EigenCAM) per test sample.

Usage:
    python -m src.explainability --model_name efficientnet_b0 --num_per_class 5
    python -m src.explainability --model_name resnet50 --num_per_class 5
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.dataset import class_to_idx, idx_to_class, get_eval_transforms
from src.model import get_resnet50, get_efficientnet_b0


TEST_CSV = "data/processed/test.csv"
MAPPING  = "data/processed/class_mapping.json"


def get_model_and_target(model_name, num_classes):
    """Load model and pick the target layer for CAM."""
    if model_name == "resnet50":
        model = get_resnet50(num_classes)
        target_layers = [model.layer4[-1]]
    elif model_name == "efficientnet_b0":
        model = get_efficientnet_b0(num_classes)
        # Last feature block before the classifier
        target_layers = [model.features[-1]]
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model, target_layers


def load_rgb_224(image_path):
    """Load image as float32 RGB numpy array normalized to [0,1], resized to 224x224."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img_np = np.array(img, dtype=np.float32) / 255.0
    return img_np


def preprocess_for_model(img_np):
    """Apply ImageNet normalization and convert to tensor (1, 3, H, W)."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = (img_np - mean) / std
    tensor = torch.tensor(tensor).permute(2, 0, 1).unsqueeze(0)  # NCHW
    return tensor.float()


def make_combined_figure(original, gradcam_overlay, eigencam_overlay, title_info, save_path):
    """Save a 3-panel figure: Original | Grad-CAM | EigenCAM."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(gradcam_overlay)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    axes[2].imshow(eigencam_overlay)
    axes[2].set_title("EigenCAM")
    axes[2].axis("off")

    pred_str = title_info.get("pred", "")
    true_str = title_info.get("true", "")
    correct  = title_info.get("correct", False)
    conf_str = title_info.get("conf", "")
    status   = "✓" if correct else "✗"

    fig.suptitle(
        f"{status}  True: {true_str}  |  Pred: {pred_str}  ({conf_str})",
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def select_samples(test_df, num_per_class):
    """Pick up to num_per_class rows per dx class from test dataframe."""
    samples = []
    for dx, group in test_df.groupby("dx"):
        chosen = group.sample(n=min(num_per_class, len(group)), random_state=42)
        samples.append(chosen)
    return pd.concat(samples).reset_index(drop=True)


def run_xai(model, target_layers, input_tensor, target_class, device):
    """Run GradCAM and EigenCAM, return numpy heatmaps."""
    input_tensor = input_tensor.to(device)
    targets = [ClassifierOutputTarget(target_class)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        gradcam_map = cam(input_tensor=input_tensor, targets=targets)[0]

    with EigenCAM(model=model, target_layers=target_layers) as cam:
        eigencam_map = cam(input_tensor=input_tensor, targets=targets)[0]

    return gradcam_map, eigencam_map


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM and EigenCAM heatmaps")
    parser.add_argument("--model_name",   type=str, default="efficientnet_b0",
                        choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--num_per_class", type=int, default=5,
                        help="Number of test images to process per class")
    parser.add_argument("--output_dir",   type=str, default="results/gradcam_samples")
    args = parser.parse_args()

    model_name    = args.model_name
    num_per_class = args.num_per_class
    checkpoint    = f"checkpoints/best_{model_name}.pth"
    out_dir       = Path(args.output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model : {model_name}")
    print(f"Device: {device}")

    # Load class mapping
    with open(MAPPING) as f:
        mapping = json.load(f)
    label_names = mapping.get("label_names", {})

    # Load checkpoint
    if not Path(checkpoint).exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint}")
        print(f"  Run: python -m src.train --model_name {model_name}")
        return

    ckpt = torch.load(checkpoint, map_location=device)
    num_classes = len(class_to_idx)
    model, target_layers = get_model_and_target(model_name, num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Checkpoint loaded from epoch {ckpt.get('epoch', '?')}")

    # Load and sample test data
    test_df = pd.read_csv(TEST_CSV)
    samples = select_samples(test_df, num_per_class)
    print(f"\nSelected {len(samples)} samples ({num_per_class} per class max)\n")

    metadata_rows = []
    correct_count = 0

    for i, row in samples.iterrows():
        image_id   = row["image_id"]
        true_dx    = row["dx"]
        true_label = class_to_idx[true_dx]
        img_path   = row["image_path"]

        # Load image in two forms
        img_np    = load_rgb_224(img_path)               # float32 [0,1] for overlay
        input_ten = preprocess_for_model(img_np)          # normalized tensor

        # Model prediction
        with torch.no_grad():
            logits = model(input_ten.to(device))
            probs  = torch.softmax(logits, dim=1)[0]
            pred_label = probs.argmax().item()
            confidence = probs[pred_label].item()

        pred_dx = idx_to_class[pred_label]
        correct = pred_label == true_label
        if correct:
            correct_count += 1

        # Run CAM methods
        gradcam_map, eigencam_map = run_xai(model, target_layers, input_ten, pred_label, device)

        # Build overlays (expects float32 RGB [0,1])
        gradcam_overlay  = show_cam_on_image(img_np, gradcam_map,  use_rgb=True)
        eigencam_overlay = show_cam_on_image(img_np, eigencam_map, use_rgb=True)

        # File names
        prefix = f"{true_dx}_{image_id}"
        gradcam_path  = out_dir / f"{prefix}_gradcam.png"
        eigencam_path = out_dir / f"{prefix}_eigencam.png"
        combined_path = out_dir / f"{prefix}_combined.png"

        # Save individual overlays
        Image.fromarray(gradcam_overlay).save(gradcam_path)
        Image.fromarray(eigencam_overlay).save(eigencam_path)

        # Save combined 3-panel
        title_info = {
            "true":    label_names.get(true_dx, true_dx),
            "pred":    label_names.get(pred_dx, pred_dx),
            "conf":    f"{confidence:.2%}",
            "correct": correct,
        }
        make_combined_figure(img_np, gradcam_overlay, eigencam_overlay, title_info, combined_path)

        metadata_rows.append({
            "image_id":      image_id,
            "true_label":    true_dx,
            "predicted_label": pred_dx,
            "confidence":    round(confidence, 4),
            "correct":       correct,
            "original_path": img_path,
            "gradcam_path":  str(gradcam_path),
            "eigencam_path": str(eigencam_path),
            "combined_path": str(combined_path),
        })

        status = "✓" if correct else "✗"
        print(f"  {status} [{i+1}/{len(samples)}] {image_id} | true={true_dx} pred={pred_dx} conf={confidence:.2%}")

    # Save metadata CSV
    meta_path = out_dir / "xai_samples_metadata.csv"
    pd.DataFrame(metadata_rows).to_csv(meta_path, index=False)

    print(f"\n--- Summary ---")
    print(f"Model          : {model_name}")
    print(f"Checkpoint     : {checkpoint}")
    print(f"Samples        : {len(samples)}")
    print(f"Correct        : {correct_count}/{len(samples)}")
    print(f"Output dir     : {out_dir}")
    print(f"Metadata CSV   : {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
