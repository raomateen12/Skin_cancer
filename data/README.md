# Data Directory

## Structure

- `data/raw/` — Original files downloaded from Kaggle (images + metadata CSV).
- `data/processed/` — Generated train/val/test CSV splits and class mapping JSON.

## Dataset

**HAM10000** (Human Against Machine with 10000 training images)
- Source: [Kaggle — Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- Slug: `kmader/skin-cancer-mnist-ham10000`
- Contains ~10,000 dermoscopic images across 7 skin disease classes.

## Kaggle Credentials

> **Important:** Never commit `kaggle.json` to the repository.

Place your `kaggle.json` at:

**Windows:**
```
C:\Users\<WindowsUserName>\.kaggle\kaggle.json
```

**Linux/Colab:**
```
~/.kaggle/kaggle.json
```

The file must have the format:
```json
{"username": "your_kaggle_username", "key": "your_api_key"}
```

## How to Download

**Local (Windows):**
```bash
python -m src.download_kaggle_dataset
python -m src.prepare_data
python -m src.check_dataloaders
```

**Google Colab:**
Upload `kaggle.json` securely using Colab's file upload, then run the same scripts.

## Notes

- `data/raw/` and `data/processed/` contents are gitignored.
- Only `.gitkeep` placeholder files are tracked.
- Download can be done locally or in Colab before training.
