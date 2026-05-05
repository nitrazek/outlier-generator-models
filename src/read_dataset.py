import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

AVAILABLE = ("HAR", "ODDS", "DFVE")


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    meta: dict

def load(name: str) -> Dataset:
    """Load a processed dataset by name (HAR, ODDS, DFVE)."""
    if name not in AVAILABLE:
        raise ValueError(f"Unknown dataset {name!r}. Available: {AVAILABLE}")

    ds_dir = PROCESSED / name
    csv_path = ds_dir / "data.csv"
    meta_path = ds_dir / "meta.json"
    if not csv_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Processed data for {name} not found in {ds_dir}. "
            f"Run `python src/prepare_dataset.py` first."
        )

    df = pd.read_csv(csv_path)
    meta = json.loads(meta_path.read_text())

    feature_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy()

    print(f"[{name}] loaded {X.shape[0]} samples, {X.shape[1]} features")
    return Dataset(name=name, X=X, y=y, feature_names=feature_cols, meta=meta)


def load_all() -> dict[str, Dataset]:
    return {name: load(name) for name in AVAILABLE}

if __name__ == "__main__":
    load_all()
