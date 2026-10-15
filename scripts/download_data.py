"""Download the raw UCI data and write the cleaned CSV.

Usage: python scripts/download_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.data_processing import build_clean_dataset  # noqa: E402

if __name__ == "__main__":
    df = build_clean_dataset()
    print(f"cleaned dataset: {df.shape} -> {config.CLEAN_PATH}")
