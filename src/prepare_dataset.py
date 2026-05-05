import json
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from scipy.io import loadmat
from sklearn.datasets import load_svmlight_files

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"


def _save(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    label_desc: str,
    extra_columns: dict[str, np.ndarray] | None = None,
    extra_meta: dict | None = None,
) -> None:
    out_dir = OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    if extra_columns:
        for col_name, col_values in extra_columns.items():
            df[col_name] = col_values
    df["label"] = y
    csv_path = out_dir / "data.csv"
    df.to_csv(csv_path, index=False)
    meta = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "label": label_desc,
    }
    if extra_meta:
        meta.update(extra_meta)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  -> {csv_path.relative_to(ROOT)} ({X.shape[0]}x{X.shape[1]})")


def prepare_har() -> None:
    print("[HAR] preparing...")
    src = RAW / "HAR"
    if not (src / "train" / "X_train.txt").exists():
        raise FileNotFoundError(f"Missing HAR raw files in {src}")

    X_train = pd.read_csv(src / "train" / "X_train.txt", sep=r"\s+", header=None).to_numpy(dtype=np.float32)
    y_train = pd.read_csv(src / "train" / "y_train.txt", header=None).to_numpy(dtype=np.int64).ravel()
    X_test = pd.read_csv(src / "test" / "X_test.txt", sep=r"\s+", header=None).to_numpy(dtype=np.float32)
    y_test = pd.read_csv(src / "test" / "y_test.txt", header=None).to_numpy(dtype=np.int64).ravel()

    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    # HAR features are already normalized to [-1, 1]; keep multi-class labels (1..6)
    # and let the downstream code pick a "normal" class for one-class classification.
    _save("HAR", X, y, "activity class id (1..6) from activity_labels.txt")


def prepare_odds() -> None:
    print("[ODDS/http] preparing...")
    src = RAW / "ODDS" / "http.mat"
    if not src.exists():
        raise FileNotFoundError(f"Missing {src}")

    with h5py.File(src, "r") as f:
        X = np.asarray(f["X"], dtype=np.float32).T
        y = np.asarray(f["y"], dtype=np.int64).ravel()
    _save("ODDS", X, y, "0 = inlier, 1 = outlier (anomaly)")


def prepare_dfve() -> None:
    print("[DFVE] preparing...")
    src = RAW / "DFVE"
    files = sorted(src.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"Missing DFVE files in {src}")

    # All monthly files use libsvm-like sparse format with float "label"
    # equal to the fraction of antivirus engines that flagged the sample.
    # load_svmlight_files unifies the feature dimension across all files.
    Xs, ys = [], []
    loaded = load_svmlight_files([str(f) for f in files])
    # load_svmlight_files returns [X1, y1, X2, y2, ...]
    for i in range(0, len(loaded), 2):
        Xs.append(loaded[i])
        ys.append(loaded[i + 1])

    from scipy.sparse import vstack as sp_vstack

    X_sparse = sp_vstack(Xs).tocsr()
    ratio = np.concatenate(ys).astype(np.float32)
    X = X_sparse.toarray().astype(np.float32)

    # Binary label: 1 if any antivirus flagged the file (malware), 0 otherwise.
    label = (ratio > 0).astype(np.int64)

    _save(
        "DFVE",
        X,
        label,
        "1 = malware (any AV detection), 0 = clean",
        extra_columns={"label_ratio": ratio},
        extra_meta={"label_ratio": "fraction of AV engines that flagged the sample (float in [0,1])"},
    )


def prepare_all() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prepare_dfve()
    prepare_har()
    prepare_odds()
    print("Done.")


if __name__ == "__main__":
    prepare_all()
