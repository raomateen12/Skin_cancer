"""
DermaLens AI — FastAPI Backend
================================
Minimal API bridge between the Next.js frontend and the Python ML/RAG stack.

Endpoints:
    GET  /health   — system status
    POST /predict  — image classification (EfficientNet-B0)
    POST /ask      — RAG-based Q&A assistant

Run:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import io
import os
import json
import sys
from pathlib import Path
from typing import Optional

# ── Add project root to path so src.* imports work ──────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError as e:
    raise ImportError(
        "FastAPI dependencies missing. Run: pip install fastapi uvicorn python-multipart"
    ) from e

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DermaLens AI API",
    description="AI-assisted skin lesion analysis API bridge.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ── Paths ────────────────────────────────────────────────────────────────────

# Accept both naming conventions for the checkpoint
CHECKPOINT_CANDIDATES = [
    ROOT / "checkpoints" / "best_efficientnet_b0.pth",
    ROOT / "checkpoints" / "efficientnet_b0_best.pth",
]
CLASS_MAPPING_PATH = ROOT / "data" / "processed" / "class_mapping.json"
RAG_INDEX_PATH = ROOT / "vectorstore" / "faiss_index"

# ── Lazy-loaded model state ──────────────────────────────────────────────────

_model = None
_transform = None

# ── Fallback class labels (used when class_mapping.json is absent) ───────────

FALLBACK_CLASS_LABELS: dict[int, str] = {
    0: "akiec",
    1: "bcc",
    2: "bkl",
    3: "df",
    4: "mel",
    5: "nv",
    6: "vasc",
}

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
    "akiec": "moderate",
    "bcc": "high",
    "bkl": "low",
    "df": "low",
    "mel": "high",
    "nv": "low",
    "vasc": "low",
}

NEXT_STEPS: dict[str, list[str]] = {
    "low": [
        "Continue monitoring for any changes over the next few months.",
        "Photograph the lesion monthly to track evolution.",
        "Consult a dermatologist if you notice any changes.",
    ],
    "moderate": [
        "Schedule a dermatologist appointment within the next 2–4 weeks.",
        "Document the lesion with photos and note any symptoms.",
        "Avoid prolonged UV exposure and use SPF 50+ sunscreen.",
    ],
    "high": [
        "Seek a dermatologist consultation as soon as possible.",
        "Do not delay — early evaluation is critical for better outcomes.",
        "Bring a timeline of how and when the lesion has changed.",
    ],
}

DISCLAIMER = (
    "This is an educational insight only. "
    "It is not a medical diagnosis. "
    "Always consult a qualified dermatologist for skin concerns."
)


def _find_checkpoint() -> Optional[Path]:
    """Return the first existing checkpoint path, or None."""
    for p in CHECKPOINT_CANDIDATES:
        if p.exists():
            return p
    return None


def _load_class_mapping() -> dict[int, str]:
    """
    Load idx→class_code mapping from class_mapping.json.

    The file has the structure:
        {
          "class_to_idx": {"akiec": 0, ...},
          "idx_to_class": {"0": "akiec", ...},
          "label_names":  {"akiec": "Actinic keratoses", ...}
        }
    Falls back to built-in labels if the file is absent or malformed.
    """
    if CLASS_MAPPING_PATH.exists():
        try:
            raw = json.loads(CLASS_MAPPING_PATH.read_text(encoding="utf-8"))
            # Preferred: use the explicit idx_to_class sub-key
            if "idx_to_class" in raw:
                return {int(k): v for k, v in raw["idx_to_class"].items()}
            # Fallback: flat mapping with string-int keys
            first_key = next(iter(raw))
            try:
                return {int(k): v for k, v in raw.items()}
            except ValueError:
                # Keys are class codes — invert to idx→code
                return {int(v): k for k, v in raw.items()}
        except Exception:
            pass
    return FALLBACK_CLASS_LABELS


def _load_label_names() -> dict[str, str]:
    """
    Load human-readable label names from class_mapping.json.
    Falls back to the hardcoded LABEL_NAMES dict.
    """
    if CLASS_MAPPING_PATH.exists():
        try:
            raw = json.loads(CLASS_MAPPING_PATH.read_text(encoding="utf-8"))
            if "label_names" in raw:
                return raw["label_names"]
        except Exception:
            pass
    return LABEL_NAMES


def _load_model() -> bool:
    """Attempt to load EfficientNet-B0 checkpoint lazily. Returns True if successful."""
    global _model, _transform
    if _model is not None:
        return True

    checkpoint_path = _find_checkpoint()
    if checkpoint_path is None:
        return False

    try:
        import torch
        from torchvision import transforms
        from src.model import get_efficientnet_b0  # type: ignore

        device = torch.device("cpu")
        model = get_efficientnet_b0(num_classes=7)
        state = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        # Support both raw state_dict and wrapped checkpoint dicts
        state_dict = state.get("model_state_dict", state)
        model.load_state_dict(state_dict)
        model.eval()

        _model = model
        _transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        return True
    except Exception:
        return False


def _check_vectorstore() -> bool:
    """Check if FAISS vectorstore index exists on disk."""
    return (RAG_INDEX_PATH / "index.faiss").exists()


def _check_rag_dependencies() -> bool:
    """Return True if all RAG/langchain dependencies are importable."""
    try:
        import langchain_community  # noqa: F401
        import langchain_huggingface  # noqa: F401
        import faiss  # noqa: F401
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _generate_gradcam(image_bytes: bytes, predicted_idx: int) -> dict:
    """
    Generate Grad-CAM and EigenCAM overlays for the predicted class.
    Returns a dict:
      {
        "available": True,
        "images": [original_b64, gradcam_b64, eigencam_b64]
      }
    Falls back to {"available": False, "images": []} if grad-cam libs missing.
    """
    try:
        import base64
        import numpy as np
        import torch
        from PIL import Image as PILImage
        from pytorch_grad_cam import GradCAM, EigenCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        if _model is None:
            return {"available": False, "images": []}

        # Target layer: last features block of EfficientNet-B0
        target_layers = [_model.features[-1]]  # type: ignore[attr-defined]

        # Load and resize original image
        pil_img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
        img_np = np.array(pil_img, dtype=np.float32) / 255.0

        # Build normalized tensor
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = torch.tensor((img_np - mean) / std).permute(2, 0, 1).unsqueeze(0).float()

        targets = [ClassifierOutputTarget(predicted_idx)]

        with GradCAM(model=_model, target_layers=target_layers) as cam:  # type: ignore[arg-type]
            gradcam_map = cam(input_tensor=tensor, targets=targets)[0]

        with EigenCAM(model=_model, target_layers=target_layers) as cam:  # type: ignore[arg-type]
            eigencam_map = cam(input_tensor=tensor, targets=targets)[0]

        gradcam_overlay  = show_cam_on_image(img_np, gradcam_map,  use_rgb=True)
        eigencam_overlay = show_cam_on_image(img_np, eigencam_map, use_rgb=True)

        def to_b64(arr: np.ndarray) -> str:
            buf = io.BytesIO()
            PILImage.fromarray(arr).save(buf, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        # Also encode the original resized image for the frontend
        orig_buf = io.BytesIO()
        pil_img.save(orig_buf, format="PNG")
        orig_b64 = "data:image/png;base64," + base64.b64encode(orig_buf.getvalue()).decode()

        return {
            "available": True,
            "images": [
                orig_b64,
                to_b64(gradcam_overlay),
                to_b64(eigencam_overlay),
            ],
        }

    except ImportError:
        # pytorch-grad-cam not installed — silent graceful fallback
        return {"available": False, "images": []}
    except Exception:
        # Any other error — don't crash the prediction
        return {"available": False, "images": []}


# ── Request/Response models ──────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    language: str = "auto"


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Return full system status matching the documented health schema."""
    checkpoint_path = _find_checkpoint()
    model_available = checkpoint_path is not None
    class_mapping_available = CLASS_MAPPING_PATH.exists()
    rag_index_available = _check_vectorstore()
    rag_deps_available = _check_rag_dependencies()
    rag_available = rag_index_available and rag_deps_available

    missing = []
    if not model_available:
        missing.append(str(CHECKPOINT_CANDIDATES[0].relative_to(ROOT)))
    if not class_mapping_available:
        missing.append(str(CLASS_MAPPING_PATH.relative_to(ROOT)))
    if not rag_index_available:
        missing.append(str((RAG_INDEX_PATH / "index.faiss").relative_to(ROOT)))
    if not rag_deps_available:
        missing.append("RAG Python dependencies (langchain-community, faiss-cpu, sentence-transformers)")

    if missing:
        message = f"Missing artifacts or dependencies: {', '.join(missing)}"
    else:
        message = "All systems operational."

    return {
        "status": "ok",
        "model_available": model_available,
        "checkpoint_path": str(CHECKPOINT_CANDIDATES[0].relative_to(ROOT)),
        "class_mapping_available": class_mapping_available,
        "rag_available": rag_available,
        "rag_index_available": rag_index_available,
        "rag_dependencies_available": rag_deps_available,
        "rag_index_path": str(RAG_INDEX_PATH.relative_to(ROOT)),
        "message": message,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept a skin lesion image and return an AI-assisted educational classification.

    Returns { ok: false, error: ..., missing_path: ... } if model is unavailable.
    Never returns a fake result as a real one.
    """
    checkpoint_path = _find_checkpoint()

    if checkpoint_path is None:
        return {
            "ok": False,
            "available": False,
            "error": "Model weights are not available in this environment.",
            "missing_path": str(CHECKPOINT_CANDIDATES[0].relative_to(ROOT)),
        }

    model_loaded = _load_model()
    if not model_loaded:
        return {
            "ok": False,
            "available": False,
            "error": "Model checkpoint could not be loaded. The file may be corrupt or incompatible.",
            "missing_path": str(checkpoint_path.relative_to(ROOT)),
        }

    try:
        import torch
        from PIL import Image

        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = _transform(img).unsqueeze(0)  # type: ignore[operator]

        with torch.no_grad():
            logits = _model(tensor)  # type: ignore[operator]
            probs = torch.softmax(logits, dim=1).squeeze()

        class_labels = _load_class_mapping()
        label_names = _load_label_names()
        top3_idx = probs.argsort(descending=True)[:3].tolist()
        predicted_idx = top3_idx[0]
        predicted_code = class_labels.get(predicted_idx, f"class_{predicted_idx}")
        predicted_name = label_names.get(predicted_code, predicted_code)
        confidence = float(probs[predicted_idx])
        concern = CONCERN_MAP.get(predicted_code, "unknown")

        top_predictions = [
            {
                "code": class_labels.get(i, f"class_{i}"),
                "name": label_names.get(class_labels.get(i, ""), class_labels.get(i, f"class_{i}")),
                "confidence": round(float(probs[i]), 4),
            }
            for i in top3_idx
        ]

        # Keep backward-compatible fields for the frontend ResultPanel
        top_3 = [
            {"label": class_labels.get(i, f"class_{i}"), "probability": round(float(probs[i]), 4)}
            for i in top3_idx
        ]

        # Generate Grad-CAM / EigenCAM overlays
        xai = _generate_gradcam(contents, predicted_idx)

        return {
            "ok": True,
            "available": True,
            "predicted_code": predicted_code,
            "predicted_class": predicted_code,
            "predicted_name": predicted_name,
            "predicted_label": predicted_name,
            "confidence": round(confidence, 4),
            "concern_level": concern,
            "concern_message": NEXT_STEPS.get(concern, ["Consult a dermatologist."])[0],
            "top_predictions": top_predictions,
            "top_3": top_3,
            "next_steps": NEXT_STEPS.get(concern, []),
            "gradcam_available": xai["available"],
            "gradcam_images": xai["images"],
            "disclaimer": DISCLAIMER,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(exc)}",
        )


@app.post("/ask")
async def ask(request: AskRequest):
    """
    Answer a medical question using the RAG pipeline.
    Returns a clean JSON response in all error states — never exposes raw exceptions.
    """
    # Check vectorstore presence
    if not _check_vectorstore():
        return {
            "ok": False,
            "answer": "The medical knowledge assistant is not fully connected in this environment.",
            "sources": [],
            "language": "english",
            "language_detected": "english",
            "reason": "index_missing",
            "disclaimer": DISCLAIMER,
        }

    # Check RAG dependencies
    if not _check_rag_dependencies():
        return {
            "ok": False,
            "answer": "The medical knowledge assistant is not fully connected in this environment.",
            "sources": [],
            "language": "english",
            "language_detected": "english",
            "reason": "dependency_missing",
            "disclaimer": DISCLAIMER,
        }

    try:
        from src.rag import answer_question  # type: ignore

        result = answer_question(
            question=request.question,
            language=request.language if request.language != "auto" else "auto",
        )

        # rag.py returned an error dict (e.g. library missing, index load failed)
        if result.get("error"):
            return {
                "ok": False,
                "answer": "The medical knowledge assistant is not fully connected in this environment.",
                "sources": [],
                "language": result.get("language", "english"),
                "language_detected": result.get("language", "english"),
                "reason": "rag_error",
                "disclaimer": DISCLAIMER,
            }

        return {
            "ok": True,
            "answer": result.get("answer", "No answer found."),
            "sources": result.get("sources", []),
            "language": result.get("language", request.language),
            "language_detected": result.get("language", request.language),
            "disclaimer": DISCLAIMER,
        }

    except (ImportError, ModuleNotFoundError):
        # Dependency missing at call time — return clean user-facing message
        return {
            "ok": False,
            "answer": "The medical knowledge assistant is not fully connected in this environment.",
            "sources": [],
            "language": "english",
            "language_detected": "english",
            "reason": "dependency_missing",
            "disclaimer": DISCLAIMER,
        }

    except Exception as exc:
        # Unexpected error — return clean message, log detail server-side
        import logging
        logging.getLogger("dermalens").error("Assistant error: %s", exc, exc_info=True)
        return {
            "ok": False,
            "answer": "The medical knowledge assistant encountered an unexpected error.",
            "sources": [],
            "language": "english",
            "language_detected": "english",
            "reason": "unexpected_error",
            "disclaimer": DISCLAIMER,
        }
