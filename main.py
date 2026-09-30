"""
Main Entry Point for Credit Card Fraud & Risk Analyser.
Starts the FastAPI backend and serves the React frontend at http://127.0.0.1:8000.
"""
from pathlib import Path
import uvicorn

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "trained_models.pkl"

if __name__ == "__main__":
    if not MODEL_PATH.exists():
        print("[INFO] No trained model bundle found. Running training pipeline...")
        from train import main as run_training
        run_training()

    print("=" * 65)
    print("[SERVER] Starting Credit Card Fraud & Risk Analyser (FastAPI + React)")
    print("[SERVER] Web Application: http://127.0.0.1:8000")
    print("[SERVER] API Docs (Swagger): http://127.0.0.1:8000/docs")
    print("=" * 65)

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
