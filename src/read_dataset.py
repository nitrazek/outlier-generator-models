import json
from enum import StrEnum
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


class Dataset(StrEnum):
    HAR = "HAR"
    ODDS = "ODDS"
    DFVE = "DFVE"


def load(name: Dataset | str) -> tuple[pd.DataFrame, dict]:
    """Load a processed dataset. Returns (dataframe, meta).

    The dataframe contains feature columns (f0..fN), a 'label' column, and
    optionally extra columns (e.g. 'label_ratio' for DFVE).
    """
    name = Dataset(name)

    ds_dir = PROCESSED / name.value
    csv_path = ds_dir / "data.csv"
    meta_path = ds_dir / "meta.json"
    if not csv_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Processed data for {name} not found in {ds_dir}. "
            f"Run `python src/prepare_dataset.py` first."
        )

    df = pd.read_csv(csv_path)
    meta = json.loads(meta_path.read_text())

    print(f"[{name}] loaded {df.shape[0]} samples, {df.shape[1]} columns")
    return df, meta


def load_all() -> dict[Dataset, tuple[pd.DataFrame, dict]]:
    return {name: load(name) for name in Dataset}