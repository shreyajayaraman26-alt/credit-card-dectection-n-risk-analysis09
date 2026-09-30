"""
Preprocessing module with high-performance NumPy loading and caching.
"""
from __future__ import annotations
from pathlib import Path
import csv
import numpy as np

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "creditcard.csv"
CACHE_PATH = Path(__file__).resolve().parents[1] / "models" / "creditcard_cache.npy"

RAW_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount", "Hour", "LogAmount", "Night"]


class SimpleStandardScaler:
    """StandardScaler implementation using NumPy."""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray) -> "SimpleStandardScaler":
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # Avoid division by zero
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if len(X.shape) == 1:
            return (X - self.mean_) / self.scale_
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def load_raw_data(path: str | Path | None = None, use_cache: bool = True) -> tuple[np.ndarray, list[str]]:
    """Load credit card dataset using cached NumPy array or robust CSV parser."""
    csv_path = Path(path) if path else DATA_PATH
    cache_path = CACHE_PATH

    if use_cache and cache_path.exists():
        data = np.load(cache_path)
        if not np.any(np.isnan(data[:, -1])):
            return data, RAW_COLUMNS

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    print("   Parsing raw CSV and building high-performance cache...")
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row:
                continue
            # Strip quotes and cast
            row_floats = [float(val.replace('"', '').strip()) for val in row]
            rows.append(row_floats)

    data = np.asarray(rows, dtype=np.float64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, data)
    return data, RAW_COLUMNS


def preprocess_numpy(data: np.ndarray) -> np.ndarray:
    """Clean data: remove NaNs/Infs and clip negative amounts."""
    valid_mask = np.all(np.isfinite(data), axis=1)
    clean = data[valid_mask].copy()
    # Amount column is index 29
    clean[:, 29] = np.maximum(clean[:, 29], 0.0)
    return clean


def stratified_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/test split with pure NumPy."""
    rng = np.random.RandomState(random_state)
    
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    
    rng.shuffle(idx_0)
    rng.shuffle(idx_1)
    
    n_test_0 = int(len(idx_0) * test_size)
    n_test_1 = int(len(idx_1) * test_size)
    
    test_idx = np.concatenate([idx_0[:n_test_0], idx_1[:n_test_1]])
    train_idx = np.concatenate([idx_0[n_test_0:], idx_1[n_test_1:]])
    
    rng.shuffle(test_idx)
    rng.shuffle(train_idx)
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
