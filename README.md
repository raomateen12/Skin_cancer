# Skin Cancer Diagnosis — Fair Medical AI

A research project for skin lesion classification using deep learning on the HAM10000 dataset.
Built as part of a study on model explainability and fairness in medical AI.

## What it does

- Classifies dermoscopic images into 7 skin disease categories using ResNet50 and EfficientNet
- Uses Grad-CAM to visualize what the model focuses on for each prediction
- Runs a fairness audit to check performance across demographic groups
- Includes a RAG-based Q&A system for querying medical literature


## Modules

- `src/dataset.py` — HAM10000 dataset class and transforms
- `src/model.py` — ResNet50 and EfficientNet model definitions
- `src/train.py` — training loop with early stopping and class weighting
- `src/evaluate.py` — test set evaluation and metrics
- `src/compare_models.py` — comparison table and bar chart across models
- `src/explainability.py` — Grad-CAM and EigenCAM (in progress)
- `src/fairness.py` — group metrics and SHAP analysis (in progress)
- `src/rag.py` — PDF loading, FAISS index, and Q&A (in progress)
- `app/app.py` — Streamlit dashboard

## Dataset

HAM10000 — not included in this repo.
Download it from Kaggle: `kmader/skin-cancer-mnist-ham10000`

Place your `kaggle.json` at `C:\Users\<username>\.kaggle\kaggle.json` (Windows) or `~/.kaggle/kaggle.json` (Linux/Colab), then run:

```bash
python -m src.download_kaggle_dataset
python -m src.prepare_data
python -m src.check_dataloaders
```

## Training (run on Google Colab T4)

Open `notebooks/02_cnn_training.ipynb` in Colab, or run:

```bash
pip install -r requirements.txt
# Train ResNet50 (baseline)
python -m src.train --model_name resnet50
python -m src.evaluate --model_name resnet50
# Train EfficientNet-B0 (comparison)
python -m src.train --model_name efficientnet_b0
python -m src.evaluate --model_name efficientnet_b0
# Compare results
python -m src.compare_models
```

For local UI only:
```bash
pip install -r requirements-local.txt
streamlit run app/app.py
```

### ResNet50 baseline (Colab T4)

| metric | value |
|---|---|
| test accuracy | 0.8154 |
| weighted precision | 0.8579 |
| weighted recall | 0.8154 |
| weighted F1 | 0.8279 |

### EfficientNet-B0 (Colab T4)

| metric | value |
|---|---|
| test accuracy | 0.8603 |
| weighted precision | 0.8689 |
| weighted recall | 0.8603 |
| weighted F1 | 0.8667 |

EfficientNet-B0 outperforms ResNet50 on all metrics and is used as the default model for explainability.
Full comparison saved to `results/model_comparison.csv` (generated after training, not committed).

## Explainability

Grad-CAM and EigenCAM are used to visualize which regions of a skin lesion image the model focuses on.

- Default model for explainability: **EfficientNet-B0** (best test F1: 0.8667)
- Generates 3-panel figures per test sample: Original | Grad-CAM | EigenCAM
- Up to 5 samples per class (~35 total)
- Generated heatmaps are **not committed** to GitHub

To run on Colab:
```bash
python -m src.explainability --model_name efficientnet_b0 --num_per_class 5
```

## Status

- [x] Project structure and dataset pipeline
- [x] ResNet50 training + evaluation (baseline)
- [x] EfficientNet-B0 training + evaluation (best model)
- [x] Model comparison pipeline
- [x] Grad-CAM & EigenCAM explainability (run on Colab)
- [x] Fairness analysis and metadata SHAP
- [x] RAG chatbot for educational Q&A (English & Roman Urdu)

## Fairness & SHAP

Analyzes model performance across sex, age, and localization groups.
Uses SHAP on metadata features to find factors associated with correct predictions.

```bash
python -m src.fairness --model_name efficientnet_b0
```

## RAG Chatbot (Educational Q&A)

A Retrieval-Augmented Generation module for skin disease Q&A using medical PDFs.

- **Storage:** PDFs are placed in `data/medical_pdfs/` (not committed).
- **Index:** Vector store saved in `vectorstore/faiss_index/` (not committed).
- **Languages:** Supports English and Roman Urdu answers.
- **Citations:** Provides source filename and page numbers.
- **Disclaimer:** Educational information only; not a medical diagnosis.

### Commands

1. **Build FAISS Index:**
   ```bash
   python -m src.rag --build_index
   ```

2. **Ask Questions:**
   ```bash
   # English
   python -m src.rag --ask "What are common warning signs of melanoma?"
   # Roman Urdu
   python -m src.rag --ask "melanoma ki warning signs kya hain?" --language roman_urdu
   ```
