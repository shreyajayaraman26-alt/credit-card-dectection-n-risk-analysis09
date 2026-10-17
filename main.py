"""
Main Entry Point for Credit Card Fraud & Risk Analyser.
Starts the FastAPI backend and serves the React frontend.
"""
from pathlib import Path
import socket
import uvicorn

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "trained_models.pkl"


def get_available_port(preferred_ports: tuple[int, ...] = (80, 8000, 8080)) -> int:
    """Finds the first available port from the preferred list."""
    for port in preferred_ports:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("0.0.0.0", port))
                return port
        except OSError:
            continue
    return 8000


if __name__ == "__main__":
    if not MODEL_PATH.exists():
        print("[INFO] No trained model bundle found. Running training pipeline...")
        from train import main as run_training
        run_training()

    port = get_available_port((80, 8000, 8080))
    url = "http://localhost" if port == 80 else f"http://localhost:{port}"

    print("=" * 65)
    print("[SERVER] Starting Credit Card Fraud & Risk Analyser (FastAPI + React)")
    print(f"[SERVER] Web Dashboard: {url}")
    print(f"[SERVER] API Docs:      {url}/docs")
    print("=" * 65)

    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
