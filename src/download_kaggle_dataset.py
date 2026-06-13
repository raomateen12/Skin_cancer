"""
Download the HAM10000 dataset from Kaggle using the Kaggle CLI.

Prerequisites:
  - Place your kaggle.json at:
      Windows: C:\\Users\\<WindowsUserName>\\.kaggle\\kaggle.json
      Linux/Colab: ~/.kaggle/kaggle.json
  - Never commit kaggle.json to the repository.
"""

import subprocess
import sys
from pathlib import Path

DATASET_NAME = "kmader/skin-cancer-mnist-ham10000"
RAW_DATA_DIR = Path("data/raw")
METADATA_FILE = RAW_DATA_DIR / "HAM10000_metadata.csv"
KAGGLE_JSON = Path.home() / ".kaggle" / "kaggle.json"


def check_credentials():
    if not KAGGLE_JSON.exists():
        print("\n[ERROR] kaggle.json not found.")
        print("  Place your Kaggle API credentials at:")
        print("  Windows: C:\\Users\\<WindowsUserName>\\.kaggle\\kaggle.json")
        print("  Linux/Colab: ~/.kaggle/kaggle.json")
        print("\n  Download kaggle.json from:")
        print("  https://www.kaggle.com/settings -> API -> Create New Token")
        return False
    return True


def dataset_already_downloaded():
    return METADATA_FILE.exists()


def download_dataset():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset: {DATASET_NAME}")
    print(f"Destination: {RAW_DATA_DIR.resolve()}")
    print("This may take several minutes (~3 GB)...\n")

    # Use kaggle CLI directly (works on both Windows and Colab)
    result = subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", DATASET_NAME,
            "-p", str(RAW_DATA_DIR),
            "--unzip",
        ],
        text=True,
    )

    if result.returncode != 0:
        print("[ERROR] Download failed. Check your Kaggle credentials and internet connection.")
        sys.exit(1)

    print("\nDownload complete.")


if __name__ == "__main__":
    if dataset_already_downloaded():
        print(f"Dataset already present at {RAW_DATA_DIR}. Skipping download.")
        sys.exit(0)

    if not check_credentials():
        sys.exit(1)

    download_dataset()
