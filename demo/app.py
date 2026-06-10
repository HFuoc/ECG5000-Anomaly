from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.model import ECG_VAE

INPUT_DIM = 140
HIDDEN_DIM = 64
LATENT_DIM = 2
THRESHOLD = 0.037435

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "results" / "checkpoints" / "best_model.pth"
DIST_DIR = Path(__file__).resolve().parent / "static_dist"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Debug paths
print(f"DEBUG: __file__ = {__file__}")
print(f"DEBUG: PROJECT_ROOT = {PROJECT_ROOT}")
print(f"DEBUG: DATA_DIR = {DATA_DIR}")
print(f"DEBUG: DATA_DIR exists = {DATA_DIR.exists()}")
print(f"DEBUG: test_data.npy exists = {(DATA_DIR / 'test_data.npy').exists()}")
print(f"DEBUG: test_labels.npy exists = {(DATA_DIR / 'test_labels.npy').exists()}")


class PredictRequest(BaseModel):
    signal: list[float]


class SampleResponse(BaseModel):
    signal: list[float]
    preview: str
    sample_type: str
    index: int


class PredictionResponse(BaseModel):
    status: Literal["normal", "anomaly"]
    title: str
    subtitle: str
    reconstruction_error: float
    threshold: float
    severity_index: float
    latent_mu: list[float]
    signal: list[float]
    reconstruction: list[float]
    step_errors: list[float]


def load_dataset() -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        test_data = np.load(DATA_DIR / "test_data.npy")
        test_labels = np.load(DATA_DIR / "test_labels.npy")
        return test_data, test_labels
    except Exception:
        return None, None


def load_model() -> ECG_VAE | None:
    model = ECG_VAE(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
    ).to(DEVICE)

    if not MODEL_PATH.exists():
        return None

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


test_data, test_labels = load_dataset()
model = load_model()

app = FastAPI(title="ECG Anomaly Detection Demo")

if DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="assets")


def validate_signal(values: list[float] | np.ndarray) -> np.ndarray:
    signal = np.asarray(values, dtype=np.float32).flatten()
    if signal.size != INPUT_DIM:
        raise HTTPException(
            status_code=400,
            detail=f"Expected exactly {INPUT_DIM} values, received {signal.size}.",
        )
    if not np.isfinite(signal).all():
        raise HTTPException(status_code=400, detail="Signal contains non-finite values.")
    return signal


def run_prediction(values: list[float] | np.ndarray) -> PredictionResponse:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model checkpoint not found: {MODEL_PATH.relative_to(PROJECT_ROOT)}",
        )

    signal = validate_signal(values)
    tensor = torch.FloatTensor(signal).reshape(1, -1).to(DEVICE)

    with torch.no_grad():
        reconstruction_error = float(model.reconstruction_error(tensor).item())
        reconstruction, mu, _ = model(tensor)
        reconstruction_np = reconstruction.cpu().numpy()[0]
        mu_np = mu.cpu().numpy()[0]

    is_anomaly = reconstruction_error > THRESHOLD
    severity_index = min(
        100.0,
        (reconstruction_error / max(THRESHOLD * 2, reconstruction_error)) * 100,
    )
    step_errors = np.square(signal - reconstruction_np)

    return PredictionResponse(
        status="anomaly" if is_anomaly else "normal",
        title="Anomaly Detected" if is_anomaly else "Normal Rhythm Profile",
        subtitle=(
            "The reconstruction pattern deviates from the learned normal ECG manifold."
            if is_anomaly
            else "The signal remains inside the expected reconstruction safety band."
        ),
        reconstruction_error=reconstruction_error,
        threshold=THRESHOLD,
        severity_index=severity_index,
        latent_mu=np.round(mu_np, 4).astype(float).tolist(),
        signal=np.round(signal, 6).astype(float).tolist(),
        reconstruction=np.round(reconstruction_np, 6).astype(float).tolist(),
        step_errors=np.round(step_errors, 8).astype(float).tolist(),
    )


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "model_loaded": model is not None,
        "dataset_loaded": test_data is not None and test_labels is not None,
        "device": str(DEVICE),
        "latent_dim": LATENT_DIM,
        "input_dim": INPUT_DIM,
        "threshold": THRESHOLD,
    }


@app.get("/api/sample/{sample_type}", response_model=SampleResponse)
def sample(sample_type: Literal["normal", "anomaly"]) -> SampleResponse:
    if test_data is None or test_labels is None:
        raise HTTPException(
            status_code=503,
            detail=f"Dataset not found in {DATA_DIR.relative_to(PROJECT_ROOT)}.",
        )

    target_label = 0 if sample_type == "normal" else 1
    indexes = np.where(test_labels == target_label)[0]
    if indexes.size == 0:
        raise HTTPException(status_code=404, detail=f"No {sample_type} samples found.")

    index = int(np.random.choice(indexes))
    signal = test_data[index].astype(float)
    preview = ", ".join(f"{value:.4f}" for value in signal)

    return SampleResponse(
        signal=np.round(signal, 6).tolist(),
        preview=preview,
        sample_type=sample_type,
        index=index,
    )


@app.post("/api/predict", response_model=PredictionResponse)
def predict(request: PredictRequest) -> PredictionResponse:
    return run_prediction(request.signal)


@app.post("/api/upload", response_model=SampleResponse)
async def upload_csv(file: UploadFile = File(...)) -> SampleResponse:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a .csv file.")

    try:
        frame = pd.read_csv(file.file, header=None)
        signal = validate_signal(frame.values.flatten())
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"CSV parsing error: {exc}") from exc

    return SampleResponse(
        signal=np.round(signal, 6).astype(float).tolist(),
        preview=", ".join(f"{value:.4f}" for value in signal),
        sample_type="csv",
        index=-1,
    )


@app.get("/{full_path:path}")
def serve_frontend(full_path: str) -> FileResponse:
    index_file = DIST_DIR / "index.html"
    requested_file = DIST_DIR / full_path

    if full_path and requested_file.is_file():
        return FileResponse(requested_file)
    if index_file.exists():
        return FileResponse(index_file)

    raise HTTPException(
        status_code=404,
        detail="React build not found. Run npm install and npm run build in demo/frontend.",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "demo.app:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "7860")),
        reload=False,
    )
