# DermaLens AI

DermaLens AI is an open-source dermatology research project that pairs skin lesion classification with an evidence-grounded clinical assistant. It combines an EfficientNet-B0 classifier trained on the HAM10000 dataset, U-Net lesion boundary segmentation, Grad-CAM and EigenCAM explainability, individual typology angle (ITA) skin-tone fairness audits, and a bilingual (English and Roman Urdu) retrieval-augmented generation (RAG) assistant.

The project includes both a FastAPI backend (deployed on Hugging Face Spaces) and a Next.js 16 web application (deployed on Vercel).

---

## Architecture Overview

```text
                                [ User / Browser ]
                                        │
                                        ▼
                   Next.js 16 Web Application (Vercel)
               ├── Interactive Dermoscopy Workspace
               ├── Grad-CAM & U-Net Visualizer
               ├── ABCD Counterfactual Simulator
               └── Evidence-Grounded Chat (with inline citations)
                                        │
                                        ▼ HTTPS
                       FastAPI Gateway (Hugging Face Spaces)
                                        │
              ┌─────────────────────────┴─────────────────────────┐
              ▼                                                   ▼
   Vision & Diagnostics Pipeline                       Clinical RAG Pipeline
   ├── Heuristic Skin Validator                        ├── PyPDFLoader (medical PDFs)
   ├── EfficientNet-B0 (7 classes)                     ├── RecursiveCharacterTextSplitter (800 / 120)
   ├── U-Net Lesion Boundary Segmentation              ├── all-MiniLM-L6-v2 Embeddings (384d)
   ├── PyTorch Grad-CAM & EigenCAM                     ├── Local FAISS Index (IndexFlatL2)
   ├── Multimodal Metadata Fusion (Age/Sex/Site)       ├── Out-of-domain L2 Gate (distance > 1.25)
   ├── ABCD Morphological Perturbations                ├── Gemini Generation (Dual-key fallback)
   └── ITA Skin-Tone Calibration Audit                 ├── Inline Citation Parser ([1], [2])
                                                       └── 2nd-Pass LLM Critic Fact-Checker
```

---

## Core Systems

### 1. Skin Lesion Classification
The vision pipeline classifies dermatoscopic images into 7 diagnostic categories from the HAM10000 dataset:
- `akiec` — Actinic Keratoses and Intraepithelial Carcinoma
- `bcc` — Basal Cell Carcinoma
- `bkl` — Benign Keratosis (solar lentigines / seborrheic keratoses)
- `df` — Dermatofibroma
- `mel` — Melanoma
- `nv` — Melanocytic Nevi
- `vasc` — Vascular Lesions

On the test split, EfficientNet-B0 outperforms the ResNet50 baseline:

| Model | Test Accuracy | Weighted Precision | Weighted Recall | Weighted F1 |
|---|---|---|---|---|
| ResNet50 (Baseline) | 0.8154 | 0.8579 | 0.8154 | 0.8279 |
| **EfficientNet-B0** | **0.8603** | **0.8689** | **0.8603** | **0.8667** |

### 2. Lesion Boundary Segmentation (U-Net)
A trained U-Net (`src/unet_model.py`, weights in `checkpoints/best_unet.pth`) generates binary lesion masks and extracts clinical morphology parameters:
- Boundary contours overlaid directly onto original dermoscopy images.
- Quantitative metrics: lesion area percentage, perimeter, compactness, solidity, and border irregularity index.

### 3. Explainability and Counterfactuals
- **Grad-CAM and EigenCAM (`src/explainability.py`):** Visual attention heatmaps generated from the final convolutional block (`model.features[-1]`) of EfficientNet-B0.
- **ABCD Counterfactual Simulation (`src/counterfactual_explainer.py`):** Modifies the segmented U-Net lesion mask using OpenCV morphological operations (border irregularity smoothing, asymmetry alignment, diameter scaling) and re-runs the classifier to show how risk confidence shifts under clinical ABCD changes.

### 4. Multimodal Patient Metadata Fusion
`src/metadata_fusion_model.py` combines the 1280-dimensional image representation from EfficientNet-B0 with structured demographic metadata (normalized patient age, binary sex, one-hot anatomical site). A dedicated fusion head outputs refined predictions and alerts clinicians when image-only and multimodal models disagree.

### 5. Fairness, Calibration, and Hair-Removal Sensitivity
HAM10000 lacks self-reported skin-tone labels. We estimate pigmentation using the Individual Typology Angle (ITA) in the CIELAB color space:

$$\text{ITA} = \frac{\arctan\left(\frac{L^* - 50}{b^*}\right) \times 180}{\pi}$$

Images are categorized into Light ($>28^\circ$), Intermediate ($10^\circ \text{ to } 28^\circ$), and Dark ($<10^\circ$) groups.
- **Baseline Disparity:** Baseline testing showed an Expected Calibration Error (ECE) and Brier-score disparity between estimated dark and light skin tones (dark Brier = 0.305 vs light Brier = 0.158).
- **Hair-Removal Experiment (`src/calibration_fairness_hairremoved.py`):** Applying classical Black-Hat morphological filtering with Telea inpainting before inference reduced the dark-vs-light Brier gap from 0.147 to 0.133 (a reduction of 0.014 confirmed through paired bootstrap testing, $N=1000$).

### 6. Evidence-Grounded Dermatology Assistant (RAG)
`src/rag.py` answers clinical and patient queries in English or Roman Urdu using authoritative medical PDFs (`data/medical_pdfs/`):
- **Chunking:** `RecursiveCharacterTextSplitter` with 800-character chunks and 120-character overlap.
- **Vector Index:** `sentence-transformers/all-MiniLM-L6-v2` generates 384-dimensional embeddings stored in a local FAISS `IndexFlatL2` index (`vectorstore/faiss_index/`).
- **Domain Gate:** If the nearest chunk has an L2 distance $> 1.25$, the assistant refuses to answer rather than guessing on non-dermatological topics.
- **Generation:** Google Gemini generates answers strictly from retrieved chunks, inserting inline `[1]`, `[2]` markers.
- **Citation Parser:** Regex extracts markers and links them to source filenames, page numbers, and text snippets.
- **Critic Fact-Checker:** A second LLM pass evaluates each sentence against the retrieved chunks, labeling it `SUPPORTED`, `PARTIAL`, or `UNSUPPORTED`.

---

## Repository Structure

```text
fair-medical-ai/
├── api/
│   └── main.py                     # FastAPI application endpoints
├── checkpoints/                    # Trained PyTorch weights (.pth)
├── configs/                        # Class mappings and runtime configs
├── data/
│   ├── medical_pdfs/               # Reference PDFs for RAG indexing
│   ├── processed/                  # Cleaned CSVs and metadata statistics
│   └── raw/                        # Raw HAM10000 image archives
├── frontend/                       # Next.js 16 web application
│   ├── app/                        # App router pages (/ and /app)
│   ├── components/                 # Workspace panels, upload cards, modals
│   ├── lib/api.ts                  # Resilient API client with cloud failover
│   └── public/samples/             # 5 pre-loaded clinical dataset samples
├── results/                        # Generated figures, calibration metrics, reports
├── src/
│   ├── calibration_fairness.py     # ECE, MCE, and Brier evaluation by ITA group
│   ├── calibration_fairness_hairremoved.py # Hair-removal fairness experiment
│   ├── counterfactual_explainer.py # ABCD morphological perturbation engine
│   ├── dataset.py                  # PyTorch Dataset and Albumentations pipelines
│   ├── evaluate.py                 # Classification metrics and confusion matrices
│   ├── explainability.py           # Grad-CAM and EigenCAM implementations
│   ├── fairness.py                 # Metadata group metrics and SHAP analysis
│   ├── hair_removal.py             # Black-Hat morphology + Telea inpainting
│   ├── image_validator.py          # HSV skin-color ratio validation gate
│   ├── ita_utils.py                # CIELAB ITA calculation helpers
│   ├── metadata_fusion_model.py    # Joint image + patient tabular model
│   ├── model.py                    # EfficientNet-B0 and ResNet50 architectures
│   ├── rag.py                      # FAISS indexing, Gemini RAG, and critic verifier
│   ├── segmentation_inference.py   # U-Net contour and morphology extractor
│   └── unet_model.py               # PyTorch U-Net architecture
├── .github/workflows/
│   └── keep_alive.yml              # 15-minute cron ping to keep HF space awake
├── requirements.txt                # Full Python dependencies
└── README.md
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ and npm (for frontend)
- CUDA-compatible GPU recommended for full training runs; CPU works for evaluation and API serving.

### 2. Environment Setup

```bash
git clone https://github.com/raomateen12/Skin_cancer.git
cd Skin_cancer

# Python virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the root directory for Gemini and Kaggle API keys:

```env
GEMINI_API_KEY_1=your_google_gemini_api_key
GEMINI_API_KEY_2=your_backup_gemini_api_key  # Optional fallback
```

### 3. Dataset Download (HAM10000)
Place your `kaggle.json` credentials in `~/.kaggle/kaggle.json` (or `C:\Users\<username>\.kaggle\kaggle.json` on Windows) and run:

```bash
python -m src.download_kaggle_dataset
python -m src.prepare_data
python -m src.check_dataloaders
```

### 4. Running the FastAPI Backend

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API exposes:
- `GET /` — Service status
- `GET /health` — Verifies model checkpoints and FAISS index presence
- `POST /predict` — Image classification, U-Net contouring, Grad-CAM, and fusion
- `POST /ask` — RAG Q&A query endpoint

API documentation is accessible at `http://localhost:8000/docs`.

### 5. Running the Next.js Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` in your browser. The application runs against the local backend when accessed on `localhost` and automatically falls back to the cloud backend on remote hosts.

---

## Reproducing Key Experiments

### Model Training & Evaluation
```bash
# Train baseline ResNet50
python -m src.train --model_name resnet50

# Train EfficientNet-B0
python -m src.train --model_name efficientnet_b0

# Evaluate and generate comparison tables
python -m src.evaluate --model_name efficientnet_b0
python -m src.compare_models
```

### U-Net Lesion Segmentation
```bash
python -m src.prepare_segmentation_data
python -m src.train_unet
python -m src.evaluate_unet
```

### Explainability (Grad-CAM & EigenCAM)
```bash
python -m src.explainability --model_name efficientnet_b0 --num_per_class 5
```

### Fairness and Hair-Removal Calibration Audit
```bash
# Label test set with ITA angles
python -m src.label_ita_splits

# Run hair-removal sensitivity experiment
python -m src.calibration_fairness_hairremoved

# Run paired bootstrap hypothesis test on Brier score gap
python -m src.gap_significance_test

# Generate high-resolution publication figures
python -m src.generate_report_figures
```

### Building the RAG Knowledge Index
Add reference PDFs to `data/medical_pdfs/` and build the FAISS index:

```bash
python -m src.rag --build_index
```

Test queries directly from the command line:

```bash
# English
python -m src.rag --ask "What are the warning signs of melanoma?"

# Roman Urdu
python -m src.rag --ask "Melanoma ki pehchan kaise hoti hai?" --language roman_urdu
```

---

## Deployment Details

- **Frontend:** Hosted on Vercel at [dermaaibymateendev.vercel.app](https://dermaaibymateendev.vercel.app).
- **Backend:** Hosted on Hugging Face Spaces at `rmas12-dermalens-api.hf.space`.
- **Uptime Automation:** Free-tier Hugging Face Spaces enter sleep mode after 48 hours of inactivity. A GitHub Actions workflow (`.github/workflows/keep_alive.yml`) runs every 15 minutes, pinging the `/health` endpoint to keep the container awake.

---

## Medical Disclaimer

DermaLens AI is developed for research and educational purposes only. It is not a certified diagnostic device (SaMD) and should not replace clinical judgment by a licensed medical professional or board-certified dermatologist.
