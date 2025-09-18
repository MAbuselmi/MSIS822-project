# utils.py
# Helper functions (e.g., load/save data, logging)

import pandas as pd

def load_dataset_csv(path: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    return pd.read_csv(path)

def save_dataset_csv(df: pd.DataFrame, path: str):
    """Save dataset to CSV."""
    df.to_csv(path, index=False, encoding="utf-8-sig")
