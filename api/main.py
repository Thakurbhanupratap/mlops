"""FastAPI service exposing the heart-disease model with logging and Prometheus metrics."""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Histogram,
                               generate_latest)
from pydantic import BaseModel, Field

from src.inference import load_model, predict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("heart-api")

REQUESTS = Counter("api_requests_total", "Total API requests",
                   ["method", "endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency in seconds",
                    ["endpoint"])
PREDICTIONS = Counter("model_predictions_total", "Predictions by class", ["label"])


@asynccontextmanager
async def lifespan(_app):
    load_model()
    logger.info("model loaded")
    yield


app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan)


class Patient(BaseModel):
    age: float = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=1, le=4, description="chest pain type")
    trestbps: float = Field(..., ge=50, le=300)
    chol: float = Field(..., ge=50, le=700)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=40, le=250)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=10)
    slope: int = Field(..., ge=1, le=3)
    ca: int = Field(..., ge=0, le=3)
    thal: int = Field(..., description="3 normal, 6 fixed defect, 7 reversible defect")


class Prediction(BaseModel):
    prediction: int
    confidence: float
    probability_disease: float


@app.middleware("http")
async def track_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    path = request.url.path
    REQUESTS.labels(request.method, path, response.status_code).inc()
    LATENCY.labels(path).observe(elapsed)
    logger.info("%s %s -> %s (%.1f ms)", request.method, path,
                response.status_code, elapsed * 1000)
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict_endpoint(patient: Patient):
    if patient.thal not in (3, 6, 7):
        raise HTTPException(status_code=422, detail="thal must be 3, 6 or 7")
    result = predict(patient.model_dump())
    PREDICTIONS.labels(str(result["prediction"])).inc()
    logger.info("prediction=%s confidence=%s", result["prediction"], result["confidence"])
    return result


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
