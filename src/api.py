"""
REST API (FastAPI) exposing a /predict endpoint for real-time
fake news classification.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import ModelNotTrainedError, is_model_ready, predict_text

app = FastAPI(
    title="Fake News Detector API",
    description="Classifies news article text as real or fake.",
    version="0.1.0",
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw article title/body to classify")


class PredictResponse(BaseModel):
    label: str
    label_id: int
    confidence: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=is_model_ready())


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        result = predict_text(payload.text)
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictResponse(label=result.label, label_id=result.label_id, confidence=result.confidence)
