"""
DermaLens AI — Prediction CLI
==============================
Run inference on a single image using a trained EfficientNet-B0 checkpoint.

Usage:
    python -m src.predict --image_path path/to/image.jpg --model_name efficientnet_b0

If the checkpoint is missing, the script will report the expected path clearly.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent

CHECKPOINT_CANDIDATES = [
    ROOT / "checkpoints" / "best_efficientnet_b0.pth",
    ROOT / "checkpoints" / "efficientnet_b0_best.pth",
]
CLASS_MAPPING_PATH = ROOT / "data" / "processed" / "class_mapping.json"

LABEL_NAMES: dict[str, str] = {
    "akiec": "Actinic Keratosis",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion",
}

CONCERN_MAP: dict[str, str] = {
    "akiec": "Moderate concern — schedule a dermatologist visit",
    "bcc": "Higher concern — seek prompt dermatologist evaluation",
    "bkl": "Lower concern — monitor for changes",
    "df": "Lower concern — monitor for changes",
    "mel": "Higher concern — seek urgent dermatologist evaluation",
    "nv": "Lower concern — monitor for changes",
    "vasc": "Lower concern — monitor for changes",
}

FALLBACK_CLASS_LABELS: dict[int, str] = {
    0: "akiec",
    1: "bcc",
    2: "bkl",
    3: "df",
    4: "mel",
    5: "nv",
    6: "vasc",
}

DISCLAIMER = (
    "Educational support only. This is not a medical diagnosis. "
    "Consult a qualified dermatologist for any skin concerns."
)


def _find_checkpoint() -> Path | None:
    for p in CHECKPOINT_CANDIDATES:
        if p.exists():
            return p
    return None


def _load_class_mapping() -> dict[int, str]:
    if CLASS_MAPPING_PATH.exists():
        try:
            raw = json.loads(CLASS_MAPPING_PATH.read_text(encoding="utf-8"))
            if "idx_to_class" in raw:
                return {int(k): v for k, v in raw["idx_to_class"].items()}
            first_key = next(iter(raw))
            try:
                return {int(k): v for k, v in raw.items()}
            except ValueError:
                return {int(v): k for k, v in raw.items()}
        except Exception:
            pass
    return FALLBACK_CLASS_LABELS


def _load_label_names() -> dict[str, str]:
    if CLASS_MAPPING_PATH.exists():
        try:
            raw = json.loads(CLASS_MAPPING_PATH.read_text(encoding="utf-8"))
            if "label_names" in raw:
                return raw["label_names"]
        except Exception:
            pass
    return LABEL_NAMES


def predict(image_path: str, model_name: str = "efficientnet_b0") -> dict:
    """
    Run inference on a single image.
    Returns a structured result dict or an error dict.
    Never raises; always returns a printable result.
    """
    # Validate image path
    img_path = Path(image_path)
    if not img_path.exists():
        return {
            "ok": False,
            "error": f"Image file not found: {image_path}",
        }

    # Check checkpoint
    checkpoint_path = _find_checkpoint()
    if checkpoint_path is None:
        candidates = "\n".join(f"  - {p}" for p in CHECKPOINT_CANDIDATES)
        return {
            "ok": False,
            "error": (
                "Model checkpoint not found. Expected one of:\n"
                f"{candidates}\n\n"
                "Train the model first: python -m src.train"
            ),
            "missing_path": str(CHECKPOINT_CANDIDATES[0]),
        }

    # Load model
    try:
        import torch
        from torchvision import transforms
        from PIL import Image
        from src.model import get_efficientnet_b0  # type: ignore

        device = torch.device("cpu")
        model = get_efficientnet_b0(num_classes=7)
        state = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        state_dict = state.get("model_state_dict", state)
        model.load_state_dict(state_dict)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    except Exception as e:
        return {
            "ok": False,
            "error": f"Failed to load model from {checkpoint_path}: {e}",
        }

    # Run inference
    try:
        img = Image.open(str(img_path)).convert("RGB")
        tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze()

        class_labels = _load_class_mapping()
        label_names = _load_label_names()
        top3_idx = probs.argsort(descending=True)[:3].tolist()
        predicted_idx = top3_idx[0]
        predicted_code = class_labels.get(predicted_idx, f"class_{predicted_idx}")
        predicted_name = label_names.get(predicted_code, predicted_code)
        confidence = float(probs[predicted_idx])

        top_3 = [
            {
                "code": class_labels.get(i, f"class_{i}"),
                "name": label_names.get(class_labels.get(i, ""), class_labels.get(i, "")),
                "confidence": round(float(probs[i]), 4),
            }
            for i in top3_idx
        ]

        return {
            "ok": True,
            "image_path": str(img_path),
            "checkpoint_used": str(checkpoint_path),
            "predicted_code": predicted_code,
            "predicted_name": predicted_name,
            "confidence": round(confidence, 4),
            "concern_level": CONCERN_MAP.get(predicted_code, "Unknown"),
            "top_3": top_3,
            "disclaimer": DISCLAIMER,
        }

    except Exception as e:
        return {
            "ok": False,
            "error": f"Inference failed: {e}",
        }


def main():
    parser = argparse.ArgumentParser(
        description="DermaLens AI — Skin lesion pattern prediction CLI"
    )
    parser.add_argument(
        "--image_path",
        required=True,
        help="Path to the skin lesion image (JPG, PNG, WEBP)",
    )
    parser.add_argument(
        "--model_name",
        default="efficientnet_b0",
        choices=["efficientnet_b0"],
        help="Model architecture to use (default: efficientnet_b0)",
    )
    args = parser.parse_args()

    result = predict(args.image_path, args.model_name)

    if not result.get("ok"):
        print(f"\n[ERROR] {result.get('error', 'Unknown error')}")
        if "missing_path" in result:
            print(f"Expected checkpoint: {result['missing_path']}")
        sys.exit(1)

    print("\n" + "=" * 52)
    print("  DermaLens AI — Prediction Result")
    print("=" * 52)
    print(f"  Image          : {result['image_path']}")
    print(f"  Checkpoint     : {result['checkpoint_used']}")
    print("-" * 52)
    print(f"  Predicted Code : {result['predicted_code']}")
    print(f"  Predicted Name : {result['predicted_name']}")
    print(f"  Confidence     : {result['confidence'] * 100:.1f}%")
    print(f"  Concern Level  : {result['concern_level']}")
    print("-" * 52)
    print("  Top 3 Predictions:")
    for i, p in enumerate(result["top_3"], 1):
        print(f"    {i}. {p['name']} ({p['code']}) — {p['confidence'] * 100:.1f}%")
    print("-" * 52)
    print(f"\n  ⚠  {result['disclaimer']}\n")


if __name__ == "__main__":
    main()
