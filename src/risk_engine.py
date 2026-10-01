"""
Adaptive Risk Engine.

Combines Random Forest fraud probability with Isolation Forest anomaly
score using dynamic weights, then maps a risk score to an action:

    LOW    -> APPROVE  (< 0.30)
    MEDIUM -> REVIEW   (0.30 - 0.70)
    HIGH   -> FLAG     (>= 0.70)
"""
from __future__ import annotations
import numpy as np

LOW_THRESHOLD = 0.30
HIGH_THRESHOLD = 0.70


def adaptive_weights(
    rf_prob: np.ndarray,
    if_score: np.ndarray | None = None,
    night: np.ndarray | None = None,
    log_amount: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dynamic RF / IF weighting.

    - RF is trusted more when it is confident (probability near 0 or 1).
    - Isolation Forest is trusted more when RF is uncertain (~0.5), because
      that is where novel fraud patterns often sit on an imbalanced dataset.
    - Night-time and unusually large amounts nudge extra weight onto IF.

    Weights always sum to 1.
    """
    rf_prob = np.asarray(rf_prob, dtype=np.float64)
    uncertainty = 1.0 - np.abs(rf_prob - 0.5) * 2.0
    w_if = 0.25 + 0.45 * uncertainty

    if night is not None:
        w_if = w_if + 0.05 * np.asarray(night, dtype=np.float64)

    if log_amount is not None:
        amt = np.asarray(log_amount, dtype=np.float64)
        amt_shift = np.clip((amt - 3.0) / 6.0, 0.0, 1.0)
        w_if = w_if + 0.08 * amt_shift

    if if_score is not None:
        ifs = np.asarray(if_score, dtype=np.float64)
        w_if = w_if + 0.05 * np.clip(ifs - 0.6, 0.0, 1.0)

    w_if = np.clip(w_if, 0.15, 0.80)
    w_rf = 1.0 - w_if
    return w_rf, w_if


def combine_scores(rf_prob: np.ndarray, if_score: np.ndarray, w_rf: np.ndarray, w_if: np.ndarray) -> np.ndarray:
    return w_rf * np.asarray(rf_prob, dtype=np.float64) + w_if * np.asarray(if_score, dtype=np.float64)


def decide_single(score: float, low: float = LOW_THRESHOLD, high: float = HIGH_THRESHOLD) -> tuple[str, str]:
    if score >= high:
        return "HIGH", "FLAG"
    elif score >= low:
        return "MEDIUM", "REVIEW"
    return "LOW", "APPROVE"


def decide_array(score: np.ndarray, low: float = LOW_THRESHOLD, high: float = HIGH_THRESHOLD) -> tuple[np.ndarray, np.ndarray]:
    score = np.asarray(score, dtype=np.float64)
    levels = np.where(score >= high, "HIGH", np.where(score >= low, "MEDIUM", "LOW"))
    actions = np.where(levels == "HIGH", "FLAG", np.where(levels == "MEDIUM", "REVIEW", "APPROVE"))
    return levels, actions


def explain_row(
    rf_prob: float,
    if_score: float,
    w_rf: float,
    w_if: float,
    risk: float,
    action: str,
    hour: float | None = None,
    amount: float | None = None,
    night: int | None = None,
) -> list[str]:
    notes = [
        f"Supervised RF fraud probability is {rf_prob * 100:.1f}% (model weight: {w_rf * 100:.0f}%)",
        f"Unsupervised IF anomaly score is {if_score * 100:.1f}% (model weight: {w_if * 100:.0f}%)",
        f"Blended adaptive risk score is {risk * 100:.1f}% -> Recommended Action: {action}",
    ]
    if rf_prob >= 0.5:
        notes.append("Supervised pattern matcher detected strong similarity with historical confirmed frauds.")
    else:
        notes.append("Supervised model shows low similarity to known fraud signatures.")

    if if_score >= 0.6:
        notes.append("Anomaly detector flagged significant geometric deviation from normal spending clusters.")
    else:
        notes.append("Anomaly score lies within the nominal spending distribution.")

    if w_if > w_rf:
        notes.append("High model uncertainty shifted decision weight towards unsupervised outlier detection.")

    if night == 1 or (hour is not None and (hour < 6 or hour >= 22)):
        notes.append("Transaction occurred during high-risk overnight hours (22:00 - 06:00).")

    if amount is not None and amount > 500:
        notes.append(f"Elevated transaction amount ({amount:,.2f}) increased risk sensitivity.")

    if action == "APPROVE":
        notes.append("Risk indicators are well below the low threshold (0.30). Safe to approve instantly.")
    elif action == "REVIEW":
        notes.append("Borderline risk profile (0.30 - 0.70). Recommended for secondary auth (SMS/OTP) or manual review.")
    else:
        notes.append("High risk threshold exceeded (>= 0.70). Recommended immediate block and fraud alert.")

    return notes


def score_transactions(
    rf_prob: np.ndarray,
    if_score: np.ndarray,
    night: np.ndarray | None = None,
    log_amount: np.ndarray | None = None,
    low: float = LOW_THRESHOLD,
    high: float = HIGH_THRESHOLD,
) -> dict[str, np.ndarray]:
    w_rf, w_if = adaptive_weights(rf_prob, if_score, night=night, log_amount=log_amount)
    risk = combine_scores(rf_prob, if_score, w_rf, w_if)
    levels, actions = decide_array(risk, low=low, high=high)
    return {
        "rf_fraud_probability": np.asarray(rf_prob, dtype=np.float64),
        "if_anomaly_score": np.asarray(if_score, dtype=np.float64),
        "rf_weight": w_rf,
        "if_weight": w_if,
        "risk_score": risk,
        "risk_level": levels,
        "action": actions,
    }
