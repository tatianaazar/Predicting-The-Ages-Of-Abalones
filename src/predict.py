"""
predict.py
----------
Inference logic for the Abalone age prediction pipeline.
Loads saved models and preprocessing artifacts, runs prediction on new input.

Usage:
    python predict.py --sex M --length 0.45 --diameter 0.35 --height 0.11 \
                      --whole_weight 0.45 --shucked_weight 0.19 \
                      --viscera_weight 0.10 --shell_weight 0.14
"""

import argparse
import joblib
import numpy as np
from tensorflow.keras.models import load_model

from preprocess import preprocess_inference

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # silences TensorFlow logs

MODELS_DIR = "../models"


def load_artifacts():
    """Load all saved preprocessing and model artifacts."""
    stats = joblib.load(f"{MODELS_DIR}/stats.pkl")
    kmeans = joblib.load(f"{MODELS_DIR}/kmeans.pkl")
    model = load_model(f"{MODELS_DIR}/neural_network.keras")
    return stats, kmeans, model


def predict(raw_input: dict) -> float:
    """
    Run full inference pipeline on a single input.
    
    Args:
        raw_input: dict with keys:
            Sex (str: M/F/I), Length, Diameter, Height,
            Whole_weight, Shucked_weight, Viscera_weight, Shell_weight
    
    Returns:
        Predicted age in years (float)
    """
    stats, kmeans, model = load_artifacts()
    X = preprocess_inference(raw_input, stats, kmeans)
    prediction = model.predict(X, verbose=0).flatten()[0]
    return float(prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict abalone age")
    parser.add_argument("--sex", type=str, required=True, choices=["M", "F", "I"])
    parser.add_argument("--length", type=float, required=True)
    parser.add_argument("--diameter", type=float, required=True)
    parser.add_argument("--height", type=float, required=True)
    parser.add_argument("--whole_weight", type=float, required=True)
    parser.add_argument("--shucked_weight", type=float, required=True)
    parser.add_argument("--viscera_weight", type=float, required=True)
    parser.add_argument("--shell_weight", type=float, required=True)
    args = parser.parse_args()

    raw_input = {
        "Sex": args.sex,
        "Length": args.length,
        "Diameter": args.diameter,
        "Height": args.height,
        "Whole_weight": args.whole_weight,
        "Shucked_weight": args.shucked_weight,
        "Viscera_weight": args.viscera_weight,
        "Shell_weight": args.shell_weight,
    }

    age = predict(raw_input)
    print(f"\nPredicted Abalone Age: {age:.2f} years")
