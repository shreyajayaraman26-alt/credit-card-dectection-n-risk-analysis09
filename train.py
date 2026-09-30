"""
Training script for Credit Card Fraud & Risk Analyser.
Trains both the Supervised Random Forest and Unsupervised Isolation Forest.
"""
from __future__ import annotations
from pathlib import Path
import csv
import pickle
import time
import numpy as np

from src.preprocessing import SimpleStandardScaler, load_raw_data, preprocess_numpy, stratified_train_test_split
from src.feature_engineering import engineer_features_numpy
from src.random_forest import train_random_forest, fraud_probability
from src.isolation_forest import train_isolation_forest, anomaly_score, fit_score_range
from src.risk_engine import LOW_THRESHOLD, HIGH_THRESHOLD, score_transactions

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "trained_models.pkl"


def main() -> None:
    print("=" * 60)
    print("Starting Model Training Pipeline")
    print("=" * 60)

    t0 = time.time()
    print("1. Loading transaction dataset...")
    data, header = load_raw_data()
    print(f"   Loaded {len(data):,} rows. Total columns: {len(header)}")

    print("2. Preprocessing & Feature Engineering...")
    clean = preprocess_numpy(data)
    
    # Split raw features and Class label
    X_raw = clean[:, :-1]
    y = clean[:, -1].astype(int)

    n_fraud = int(np.sum(y == 1))
    n_legit = int(np.sum(y == 0))
    print(f"   Cleaned rows: {len(clean):,} | Fraud: {n_fraud:,} ({n_fraud/len(clean)*100:.3f}%) | Legit: {n_legit:,}")

    # Compute engineered features: [V1...V28, Amount, Hour, LogAmount, Night]
    X_featured = engineer_features_numpy(X_raw)
    feature_columns = [f"V{i}" for i in range(1, 29)] + ["Amount", "Hour", "LogAmount", "Night"]
    
    print(f"   Engineered {X_featured.shape[1]} features: {feature_columns[-4:]}")

    print("3. Stratified Train/Test Split (80/20)...")
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X_featured, y, test_size=0.2, random_state=42
    )
    print(f"   Train set: {X_train.shape[0]:,} rows (Fraud: {int(np.sum(y_train==1)):,})")
    print(f"   Test set:  {X_test.shape[0]:,} rows (Fraud: {int(np.sum(y_test==1)):,})")

    # Fit Scaler
    print("4. Standardizing features for Isolation Forest...")
    scaler = SimpleStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    print("\n5. Training Supervised Random Forest Classifier...")
    rf_start = time.time()
    rf_model = train_random_forest(X_train, y_train)
    rf_dur = time.time() - rf_start
    print(f"   Random Forest trained in {rf_dur:.2f}s ({rf_model.n_estimators} trees)")

    rf_prob_test = fraud_probability(rf_model, X_test)
    rf_preds_test = (rf_prob_test >= 0.5).astype(int)

    # Compute Evaluation Metrics
    tp = int(np.sum((rf_preds_test == 1) & (y_test == 1)))
    fp = int(np.sum((rf_preds_test == 1) & (y_test == 0)))
    tn = int(np.sum((rf_preds_test == 0) & (y_test == 0)))
    fn = int(np.sum((rf_preds_test == 0) & (y_test == 1)))

    accuracy = (tp + tn) / max(len(y_test), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * (precision * recall) / max(precision + recall, 1e-12)

    print("\n   --- Supervised Performance on Test Set ---")
    print(f"   Accuracy:  {accuracy * 100:.2f}%")
    print(f"   Precision: {precision * 100:.2f}%")
    print(f"   Recall:    {recall * 100:.2f}% (Detected {tp}/{tp+fn} frauds)")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   Confusion Matrix: [TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}]")

    # Train Isolation Forest on legitimate rows
    print("\n6. Training Unsupervised Isolation Forest (legitimate transactions)...")
    if_start = time.time()
    legit_mask = y_train == 0
    if_model = train_isolation_forest(X_train_scaled[legit_mask])
    lo, hi = fit_score_range(if_model, X_train_scaled)
    if_scores_test = anomaly_score(if_model, X_test_scaled, lo, hi)
    if_dur = time.time() - if_start
    print(f"   Isolation Forest trained in {if_dur:.2f}s. Score range: [{lo:.4f}, {hi:.4f}]")

    # Evaluate Adaptive Risk Engine
    print("\n7. Evaluating Adaptive Risk Engine...")
    night_idx = feature_columns.index("Night")
    log_amt_idx = feature_columns.index("LogAmount")
    amount_idx = feature_columns.index("Amount")
    hour_idx = feature_columns.index("Hour")

    decisions = score_transactions(
        rf_prob=rf_prob_test,
        if_score=if_scores_test,
        night=X_test[:, night_idx],
        log_amount=X_test[:, log_amt_idx],
        low=LOW_THRESHOLD,
        high=HIGH_THRESHOLD,
    )

    actions = decisions["action"]
    levels = decisions["risk_level"]
    risk_scores = decisions["risk_score"]

    print("   Action Distribution across Test Set:")
    for act in ["APPROVE", "REVIEW", "FLAG"]:
        cnt = int(np.sum(actions == act))
        print(f"     {act:8s}: {cnt:6,} ({cnt / len(actions) * 100:.2f}%)")

    fraud_test_mask = y_test == 1
    print("\n   True Frauds Breakdown by Risk Engine Action:")
    for act in ["APPROVE", "REVIEW", "FLAG"]:
        cnt = int(np.sum((actions == act) & fraud_test_mask))
        print(f"     {act:8s}: {cnt:4,} / {int(np.sum(fraud_test_mask))} ({cnt / max(np.sum(fraud_test_mask), 1) * 100:.1f}%)")

    # Sample representative rows for quick interactive UI presets
    print("\n8. Caching sample rows for API & UI...")
    fraud_indices = np.where(y == 1)[0]
    legit_indices = np.where(y == 0)[0]

    sample_frauds = []
    for idx in fraud_indices[:15]:
        row_feat = {col: float(val) for col, val in zip(feature_columns, X_featured[idx])}
        row_feat["Class"] = 1
        sample_frauds.append(row_feat)

    sample_legits = []
    for idx in legit_indices[:25]:
        row_feat = {col: float(val) for col, val in zip(feature_columns, X_featured[idx])}
        row_feat["Class"] = 0
        sample_legits.append(row_feat)

    # Save Bundle
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "rf_model": rf_model,
        "if_model": if_model,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "raw_feature_columns": [c for c in header if c != "Class"],
        "if_score_min": lo,
        "if_score_max": hi,
        "low_threshold": LOW_THRESHOLD,
        "high_threshold": HIGH_THRESHOLD,
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "total_transactions": len(clean),
            "total_frauds": n_fraud,
            "total_legit": n_legit,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "samples": {
            "fraud": sample_frauds,
            "legit": sample_legits,
        },
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"9. Successfully saved model bundle to {MODEL_PATH}")

    # Write risk_decisions.csv for test records
    out_csv = ROOT / "risk_decisions.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index", "true_class", "rf_prob", "if_score", "rf_weight", "if_weight",
            "risk_score", "risk_level", "action", "amount", "hour", "night"
        ])
        for i in range(min(5000, len(y_test))):
            writer.writerow([
                i,
                int(y_test[i]),
                f"{decisions['rf_fraud_probability'][i]:.4f}",
                f"{decisions['if_anomaly_score'][i]:.4f}",
                f"{decisions['rf_weight'][i]:.4f}",
                f"{decisions['if_weight'][i]:.4f}",
                f"{decisions['risk_score'][i]:.4f}",
                decisions["risk_level"][i],
                decisions["action"][i],
                f"{X_test[i, amount_idx]:.2f}",
                f"{X_test[i, hour_idx]:.2f}",
                int(X_test[i, night_idx]),
            ])
    print(f"   Wrote sample test evaluations to {out_csv}")
    print(f"\nCompleted total training workflow in {time.time() - t0:.2f}s.")
    print("=" * 60)


if __name__ == "__main__":
    main()
