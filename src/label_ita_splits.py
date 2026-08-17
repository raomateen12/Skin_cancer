"""
src/label_ita_splits.py
Compute ITA labels for train.csv and val.csv, saving to
  data/processed/train_with_ita.csv
  data/processed/val_with_ita.csv

Run from project root:
    python -m src.label_ita_splits
"""

from pathlib import Path
import pandas as pd

from src.ita_utils import label_split

TRAIN_CSV = Path("data/processed/train.csv")
VAL_CSV   = Path("data/processed/val.csv")
TRAIN_OUT = Path("data/processed/train_with_ita.csv")
VAL_OUT   = Path("data/processed/val_with_ita.csv")

print("=" * 60)
print("  ITA labeling — train + val splits")
print("=" * 60)

train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)

label_split(train_df, TRAIN_OUT, split_name="train")
label_split(val_df,   VAL_OUT,   split_name="val")

print("\nDone.")
