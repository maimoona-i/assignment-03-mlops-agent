"""FastAPI inference service that is deployed on Kubernetes.

  /health  -> liveness  (process is alive)
  /ready   -> readiness (model loaded and usable)  -> 503 otherwise, so K8s stops routing traffic
  /version -> deployed application / model / dataset version (from the ConfigMap)
  /predict -> prediction
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("inference-api")

state: dict = {"model": None, "error": None}


def load_model() -> None:
    path = os.environ.get("MODEL_PATH", "/app/models/model.joblib")
    try:
        import joblib

        state["model"] = joblib.load(path)
        state["error"] = None
        log.info("Model loaded from %s", path)
    except Exception as exc:
        state["model"], state["error"] = None, f"{type(exc).__name__}: {exc}"
        log.error("MODEL LOAD FAILED from %s: %s", path, state["error"])


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(title="ML Inference API", lifespan=lifespan)


class PredictRequest(BaseModel):
    features: list[float] = Field(min_length=1, max_length=500)


@app.get("/health")
def health() -> dict:
    return {"status": "alive"}


@app.get("/ready")
def ready() -> dict:
    if state["model"] is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {state['error']}")
    return {"status": "ready"}


@app.get("/version")
def version() -> dict:
    return {"app_version": os.environ.get("APP_VERSION", "dev"), "model_run_id": os.environ.get("MODEL_RUN_ID", "unknown"),
            "dataset_version": os.environ.get("DATASET_VERSION", "unknown")}


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    expected = getattr(model, "n_features_in_", len(req.features))
    if len(req.features) != expected:
        raise HTTPException(status_code=422, detail=f"expected {expected} features, got {len(req.features)}")
    pred = int(model.predict([req.features])[0])
    proba = float(model.predict_proba([req.features])[0][pred]) if hasattr(model, "predict_proba") else None
    return {"prediction": pred, "probability": proba, "model_run_id": os.environ.get("MODEL_RUN_ID", "unknown")}
