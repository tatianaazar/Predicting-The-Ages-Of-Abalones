"""
api.py
------
FastAPI REST API for serving Abalone age predictions.

Endpoints:
    GET  /             — health check
    GET  /model/info   — model metadata and feature info
    POST /predict      — predict age from abalone measurements
    POST /predict/batch — predict age for multiple abalones at once

To run locally:
    cd src
    uvicorn api:app --reload --port 8000

Then open:
    http://localhost:8000/docs  — interactive Swagger UI (try it live)
    http://localhost:8000/redoc — ReDoc documentation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Literal
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import os
import time

from preprocess import preprocess_inference

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Abalone Age Predictor",
    description=(
        "Predicts the age of abalone from physical measurements. "
        "Uses a Neural Network with K-means cluster features (R²=0.67, MAE=1.36). "
        "Built as a production ML system on top of MSc Data Science coursework."
    ),
    version="1.0.0",
)

# Allow requests from React frontend (localhost:3000) and any deployed domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = "../models"

# ── Model loading ─────────────────────────────────────────────────────────────

# Load once at startup, reuse for all requests
_stats = None
_kmeans = None
_model = None


def get_artifacts():
    """Lazy-load model artifacts on first request."""
    global _stats, _kmeans, _model
    if _model is None:
        try:
            _stats = joblib.load(f"{MODELS_DIR}/stats.pkl")
            _kmeans = joblib.load(f"{MODELS_DIR}/kmeans.pkl")
            _model = load_model(f"{MODELS_DIR}/neural_network.keras")
            print("Model artifacts loaded successfully.")
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Model artifacts not found. Run train.py first. Details: {e}"
            )
    return _stats, _kmeans, _model


# ── Request / Response schemas ────────────────────────────────────────────────

class AbaloneInput(BaseModel):
    """
    Physical measurements of a single abalone.
    All continuous features are in mm (dimensions) or grams (weights).
    Note: Rings are NOT included — the whole point is to predict age without them.
    """
    sex: Literal["M", "F", "I"] = Field(
        ...,
        description="Sex of the abalone: M=Male, F=Female, I=Infant"
    )
    length: float = Field(
        ..., gt=0, le=1.0,
        description="Longest shell measurement (mm), typically 0.07–0.82"
    )
    diameter: float = Field(
        ..., gt=0, le=1.0,
        description="Diameter perpendicular to length (mm), typically 0.05–0.65"
    )
    height: float = Field(
        ..., gt=0, le=1.5,
        description="Height with meat in shell (mm), typically 0.0–0.52"
    )
    whole_weight: float = Field(
        ..., gt=0,
        description="Whole abalone weight (grams), typically 0.002–2.83"
    )
    shucked_weight: float = Field(
        ..., gt=0,
        description="Weight of meat (grams), typically 0.001–1.49"
    )
    viscera_weight: float = Field(
        ..., gt=0,
        description="Gut weight after bleeding (grams), typically 0.001–0.76"
    )
    shell_weight: float = Field(
        ..., gt=0,
        description="Shell weight after drying (grams), typically 0.002–1.005"
    )

    @field_validator("length", "diameter", "height")
    @classmethod
    def dimensions_positive(cls, v):
        if v <= 0:
            raise ValueError("Dimensions must be positive")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "sex": "M",
                "length": 0.455,
                "diameter": 0.365,
                "height": 0.095,
                "whole_weight": 0.514,
                "shucked_weight": 0.2245,
                "viscera_weight": 0.101,
                "shell_weight": 0.15
            }
        }
    }


class PredictionResponse(BaseModel):
    predicted_age_years: float = Field(..., description="Predicted age in years")
    model: str = Field(..., description="Model used for prediction")
    note: str = Field(..., description="Context on prediction accuracy")


class BatchInput(BaseModel):
    abalones: list[AbaloneInput] = Field(..., min_length=1, max_length=100)


class BatchResponse(BaseModel):
    predictions: list[float]
    count: int
    model: str


class ModelInfo(BaseModel):
    model_type: str
    architecture: str
    preprocessing: str
    test_r2: float
    test_mae: float
    test_mse: float
    feature_importance_note: str
    training_data: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Health check — confirms the API is running."""
    return {
        "status": "ok",
        "service": "Abalone Age Predictor",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
def model_info():
    """
    Returns metadata about the trained model, including architecture,
    preprocessing steps, and test set performance.
    """
    return ModelInfo(
        model_type="Neural Network (Keras/TensorFlow)",
        architecture="Input → Dense(64, ReLU, L1) → Dense(32, ReLU) → Dropout(0.2) → Dense(1)",
        preprocessing=(
            "1. Sex encoded: M=0, F=1, I=2  "
            "2. Continuous features manually z-score standardised  "
            "3. K-means (k=3) cluster labels added as one-hot features"
        ),
        test_r2=0.67,
        test_mae=1.36,
        test_mse=3.85,
        feature_importance_note=(
            "Shell weight and shucked weight are the strongest predictors "
            "(Gini importance: 0.497). K-means clustering as a feature engineering "
            "step improved R² from 0.57 to 0.67."
        ),
        training_data="UCI Abalone Dataset — 4,177 samples, 8 physical features"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(input_data: AbaloneInput):
    """
    Predict the age of a single abalone from its physical measurements.

    Returns predicted age in years. Typical MAE is ±1.36 years on the test set.
    """
    try:
        stats, kmeans, model = get_artifacts()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    raw = {
        "Sex": input_data.sex,
        "Length": input_data.length,
        "Diameter": input_data.diameter,
        "Height": input_data.height,
        "Whole_weight": input_data.whole_weight,
        "Shucked_weight": input_data.shucked_weight,
        "Viscera_weight": input_data.viscera_weight,
        "Shell_weight": input_data.shell_weight,
    }

    try:
        X = preprocess_inference(raw, stats, kmeans)
        prediction = float(model.predict(X, verbose=0).flatten()[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return PredictionResponse(
        predicted_age_years=round(prediction, 2),
        model="Neural Network (R²=0.67, MAE=±1.36 years)",
        note="Age is estimated from physical measurements. Typical error is ±1–2 years."
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(batch: BatchInput):
    """
    Predict ages for multiple abalones in a single request.
    Accepts up to 100 abalones per call.
    """
    try:
        stats, kmeans, model = get_artifacts()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    predictions = []
    for item in batch.abalones:
        raw = {
            "Sex": item.sex,
            "Length": item.length,
            "Diameter": item.diameter,
            "Height": item.height,
            "Whole_weight": item.whole_weight,
            "Shucked_weight": item.shucked_weight,
            "Viscera_weight": item.viscera_weight,
            "Shell_weight": item.shell_weight,
        }
        try:
            X = preprocess_inference(raw, stats, kmeans)
            pred = float(model.predict(X, verbose=0).flatten()[0])
            predictions.append(round(pred, 2))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed for item: {e}")

    return BatchResponse(
        predictions=predictions,
        count=len(predictions),
        model="Neural Network (R²=0.67, MAE=±1.36 years)"
    )
