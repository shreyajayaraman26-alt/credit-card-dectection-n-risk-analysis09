"""
Convenience launcher for Credit Card Fraud & Risk Analyser.
Can be executed with: python run.py (or python main.py)
"""
import threading
import time
import webbrowser
from main import get_available_port, MODEL_PATH
import uvicorn

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

    def open_browser():
        time.sleep(1.5)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
