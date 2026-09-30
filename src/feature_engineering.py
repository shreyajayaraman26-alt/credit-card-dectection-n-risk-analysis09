"""
Feature engineering module.
Calculates Hour, LogAmount, and Night indicators.
"""
from __future__ import annotations
import numpy as np


def engineer_features_numpy(raw_matrix: np.ndarray) -> np.ndarray:
    """
    Given raw matrix [Time, V1...V28, Amount] (or with Class at the end),
    computes engineered features and returns matrix [V1...V28, Amount, Hour, LogAmount, Night].
    """
    time_col = raw_matrix[:, 0]
    v_cols = raw_matrix[:, 1:29] # V1 to V28
    amount_col = raw_matrix[:, 29:30] # Amount

    # Hour of transaction in day (0.0 to 24.0)
    hour = ((time_col % 86400) / 3600.0)[:, np.newaxis]
    
    # LogAmount
    log_amount = np.log1p(np.maximum(amount_col, 0.0))
    
    # Night flag (1 if Hour < 6 or Hour >= 22)
    night = ((hour < 6.0) | (hour >= 22.0)).astype(np.float64)

    return np.hstack([v_cols, amount_col, hour, log_amount, night])


def engineer_single_row(features: dict[str, float]) -> dict[str, float]:
    """Calculate engineered features for a single transaction dictionary."""
    time_val = float(features.get("Time", 0.0))
    amount_val = max(float(features.get("Amount", 0.0)), 0.0)
    
    hour_val = (time_val % 86400.0) / 3600.0
    log_amount_val = float(np.log1p(amount_val))
    night_val = 1.0 if (hour_val < 6.0 or hour_val >= 22.0) else 0.0

    res = dict(features)
    res["Hour"] = round(hour_val, 2)
    res["LogAmount"] = round(log_amount_val, 4)
    res["Night"] = int(night_val)
    return res
