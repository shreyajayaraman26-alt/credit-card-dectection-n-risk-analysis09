"""
FastAPI Backend Server for Credit Card Fraud & Risk Analyser.
Serves REST API and mounts the interactive React Single Page Application.
"""
from __future__ import annotations
from pathlib import Path
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.feature_engineering import engineer_single_row
from src.random_forest import fraud_probability
from src.isolation_forest import anomaly_score
from src.risk_engine import adaptive_weights, combine_scores, decide_single, explain_row, LOW_THRESHOLD, HIGH_THRESHOLD

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "trained_models.pkl"
STATIC_DIR = ROOT / "static"

app = FastAPI(
    title="SentinelPay AI - Fraud Detection & Risk Analysis API",
    description="Adaptive Supervised + Unsupervised Hybrid Risk Engine API",
    version="2.0.0",
)

# CORS middleware for open integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Trained Model Bundle
bundle = None
if MODEL_PATH.exists():
    try:
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        print(f"[FastAPI] Loaded model bundle from {MODEL_PATH}")
    except Exception as e:
        print(f"[FastAPI] Warning: Failed to load bundle: {e}")
else:
    print("[FastAPI] Warning: Model bundle not found. Run 'python train.py' first.")


# Pydantic Request Models
class TransactionInput(BaseModel):
    Time: float = Field(default=0.0, description="Seconds elapsed since first transaction")
    Amount: float = Field(default=100.0, ge=0.0, description="Transaction monetary amount")
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0


class BatchScoreRequest(BaseModel):
    count: int = Field(default=60, ge=1, le=500, description="Number of sample transactions to score")


@app.get("/", response_class=HTMLResponse)
async def serve_react_app():
    """Serves the React Single Page Application."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="React frontend index.html not found.")
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health_check():
    """Health check status endpoint."""
    return {
        "status": "healthy",
        "model_loaded": bundle is not None,
        "version": "2.0.0",
        "architecture": "FastAPI + React 18 + NumPy ML",
    }


@app.get("/api/model-info")
async def get_model_info():
    """Returns model metadata, evaluation metrics, and feature lists."""
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model bundle not loaded. Run train.py first.")

    return {
        "metrics": bundle.get("metrics", {}),
        "feature_columns": bundle.get("feature_columns", []),
        "raw_feature_columns": bundle.get("raw_feature_columns", []),
        "thresholds": {
            "low": bundle.get("low_threshold", LOW_THRESHOLD),
            "high": bundle.get("high_threshold", HIGH_THRESHOLD),
        },
        "score_ranges": {
            "if_min": bundle.get("if_score_min", 0.0),
            "if_max": bundle.get("if_score_max", 1.0),
        },
    }


@app.get("/api/sample-transactions")
async def get_sample_transactions():
    """Returns pre-cached representative fraud and legitimate sample transactions."""
    if bundle is None or "samples" not in bundle:
        raise HTTPException(status_code=503, detail="Sample transactions not available.")
    return bundle["samples"]


@app.post("/api/score")
async def score_single_transaction(tx: TransactionInput):
    """Scores a single credit card transaction through the dual-engine risk system."""
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model bundle not loaded. Run train.py first.")

    raw_dict = tx.model_dump()
    featured_dict = engineer_single_row(raw_dict)

    feat_cols = bundle["feature_columns"]
    feat_vector = np.array([featured_dict[col] for col in feat_cols], dtype=np.float64).reshape(1, -1)
    
    # Scale for Isolation Forest
    scaler = bundle["scaler"]
    feat_scaled = scaler.transform(feat_vector)

    # 1. Supervised Random Forest Fraud Probability
    rf_model = bundle["rf_model"]
    rf_prob = float(fraud_probability(rf_model, feat_vector)[0])

    # 2. Unsupervised Isolation Forest Anomaly Score
    if_model = bundle["if_model"]
    if_score = float(anomaly_score(
        if_model,
        feat_scaled,
        bundle["if_score_min"],
        bundle["if_score_max"],
    )[0])

    # 3. Adaptive Risk Engine Combination
    night_val = featured_dict.get("Night", 0)
    log_amt = featured_dict.get("LogAmount", 0.0)

    w_rf, w_if = adaptive_weights(
        rf_prob=np.array([rf_prob]),
        if_score=np.array([if_score]),
        night=np.array([night_val]),
        log_amount=np.array([log_amt]),
    )
    w_rf = float(w_rf[0])
    w_if = float(w_if[0])

    risk_score = float(combine_scores(np.array([rf_prob]), np.array([if_score]), np.array([w_rf]), np.array([w_if]))[0])
    level, action = decide_single(risk_score, low=bundle["low_threshold"], high=bundle["high_threshold"])

    # Explainability details
    explanations = explain_row(
        rf_prob=rf_prob,
        if_score=if_score,
        w_rf=w_rf,
        w_if=w_if,
        risk=risk_score,
        action=action,
        hour=featured_dict.get("Hour"),
        amount=featured_dict.get("Amount"),
        night=night_val,
    )

    return {
        "rf_fraud_probability": round(rf_prob, 4),
        "if_anomaly_score": round(if_score, 4),
        "rf_weight": round(w_rf, 4),
        "if_weight": round(w_if, 4),
        "risk_score": round(risk_score, 4),
        "risk_level": level,
        "action": action,
        "engineered_features": {
            "Hour": featured_dict.get("Hour"),
            "LogAmount": featured_dict.get("LogAmount"),
            "Night": featured_dict.get("Night"),
        },
        "explanations": explanations,
    }


@app.post("/api/batch-score")
async def score_batch_transactions(req: BatchScoreRequest):
    """Generates an evaluated batch stream from sample dataset for live monitoring."""
    if bundle is None or "samples" not in bundle:
        raise HTTPException(status_code=503, detail="Model bundle not loaded.")

    samples_fraud = bundle["samples"]["fraud"]
    samples_legit = bundle["samples"]["legit"]

    # Combine a realistic mix
    total = req.count
    n_fraud = min(len(samples_fraud), max(2, int(total * 0.25)))
    n_legit = min(len(samples_legit), total - n_fraud)

    batch_pool = samples_fraud[:n_fraud] + samples_legit[:n_legit]
    np.random.seed(42)
    np.random.shuffle(batch_pool)

    results = []
    summary = {"APPROVE": 0, "REVIEW": 0, "FLAG": 0}

    for item in batch_pool:
        tx_in = TransactionInput(**{k: item[k] for k in item if k in TransactionInput.model_fields})
        scored = await score_single_transaction(tx_in)
        action = scored["action"]
        summary[action] = summary.get(action, 0) + 1

        results.append({
            "amount": float(item.get("Amount", 0.0)),
            "hour": float(item.get("Hour", 0.0)),
            "rf_prob": scored["rf_fraud_probability"],
            "if_score": scored["if_anomaly_score"],
            "risk_score": scored["risk_score"],
            "action": action,
            "true_class": item.get("Class", 0),
        })

    return {
        "total": len(results),
        "summary": summary,
        "transactions": results,
    }


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
