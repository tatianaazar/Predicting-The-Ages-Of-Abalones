"""
train.py
--------
Model training for the Abalone age prediction pipeline.
Supports Neural Network (primary) and XGBoost (comparison).
MLflow is used for experiment tracking — every run logs params, metrics,
and model artifacts automatically.

Key design decisions (from experimental report):
- Neural Network: 2 hidden layers (64 -> 32), ReLU, Adam, L1 regularisation,
  Dropout(0.2), 100 epochs, batch_size=32
- XGBoost: n_estimators=175, learning_rate=0.07, max_depth=4
- NN outperforms XGBoost on this dataset (R2: 0.67 vs 0.61, MAE: 1.36 vs 1.49)

To view experiment results:
    mlflow ui
Then open http://localhost:8000 in your browser.
"""

import numpy as np
import os
import joblib
import mlflow
import mlflow.keras
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l1
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import tensorflow as tf

from src.preprocess import preprocess

RANDOM_STATE = 42
TEST_SIZE = 0.10
VAL_SIZE = 0.10
MLFLOW_EXPERIMENT_NAME = "abalone-age-prediction"


def set_seeds():
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)


def split_data(X, y):
    """
    Two-stage split:
    1. Hold out 10% as final test set
    2. Hold out 10% of remaining as validation set
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


class MLflowEpochLogger(Callback):
    """
    Keras callback that logs training and validation loss
    to MLflow at the end of each epoch.
    This gives you the full loss curve visible in the MLflow UI.
    """
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            mlflow.log_metric("train_loss", logs.get("loss", 0), step=epoch)
            mlflow.log_metric("train_mae", logs.get("mae", 0), step=epoch)
            mlflow.log_metric("val_loss", logs.get("val_loss", 0), step=epoch)
            mlflow.log_metric("val_mae", logs.get("val_mae", 0), step=epoch)


def build_neural_network(input_dim: int) -> Sequential:
    """
    Build neural network architecture.

    Architecture choices (experimentally validated):
    - 2 hidden layers: 64 -> 32 neurons (halving pattern worked best)
    - ReLU activation: appropriate for positive target (age), less expensive than LeakyReLU
    - L1 regularisation on first layer: reduces overfitting from outlier noise
    - Dropout(0.2): additional regularisation
    - Adam optimiser: outperformed SGD on this dataset
    - MSE loss: standard for regression
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation="relu", kernel_regularizer=l1(0.01)),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(loss="mean_squared_error", optimizer=Adam(), metrics=["mae"])
    return model


def train_neural_network(X_train, y_train, X_val, y_val,
                          epochs: int = 100, batch_size: int = 32,
                          save_path: str = None):
    """
    Train neural network with MLflow experiment tracking.
    Logs: all hyperparameters, per-epoch loss curves, final metrics, model artifact.
    """
    set_seeds()

    params = {
        "model_type": "neural_network",
        "layers": "64-32-1",
        "activation": "relu",
        "optimizer": "adam",
        "loss": "mse",
        "regularisation": "l1(0.01) on layer 1",
        "dropout": 0.2,
        "epochs": epochs,
        "batch_size": batch_size,
        "n_clusters_feature": 3,
        "random_state": RANDOM_STATE,
    }

    with mlflow.start_run(run_name="neural_network"):
        mlflow.log_params(params)

        model = build_neural_network(input_dim=X_train.shape[1])
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[MLflowEpochLogger()]
        )

        if save_path:
            model.save(save_path)
            mlflow.log_artifact(save_path, artifact_path="model")
            print(f"Neural network saved and logged to MLflow: {save_path}")

        print("Neural network run logged to MLflow.")

    return model


def train_xgboost(X_train, y_train, X_test, y_test,
                  save_path: str = None):
    """
    Train XGBoost with MLflow experiment tracking.
    Logs: hyperparameters, test metrics, model artifact.
    Hyperparameters tuned experimentally: n_estimators=175, learning_rate=0.07, max_depth=4
    """
    params = {
        "model_type": "xgboost",
        "objective": "reg:squarederror",
        "learning_rate": 0.07,
        "max_depth": 4,
        "n_estimators": 175,
        "n_clusters_feature": 3,
        "random_state": RANDOM_STATE,
    }

    with mlflow.start_run(run_name="xgboost"):
        mlflow.log_params(params)

        model = XGBRegressor(
            objective="reg:squarederror",
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            n_estimators=params["n_estimators"],
            random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)

        # Log test metrics inside the run
        y_pred = model.predict(X_test)
        mlflow.log_metric("test_r2", r2_score(y_test, y_pred))
        mlflow.log_metric("test_mae", mean_absolute_error(y_test, y_pred))
        mlflow.log_metric("test_mse", mean_squared_error(y_test, y_pred))

        if save_path:
            joblib.dump(model, save_path)
            mlflow.log_artifact(save_path, artifact_path="model")
            print(f"XGBoost saved and logged to MLflow: {save_path}")

        print("XGBoost run logged to MLflow.")

    return model


def evaluate(model, X_test, y_test, model_name: str = "Model") -> dict:
    """Evaluate model, print metrics, and log to active MLflow run if open."""
    y_pred = model.predict(X_test)
    if hasattr(y_pred, "flatten"):
        y_pred = y_pred.flatten()

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{model_name} Results:")
    print(f"  R2:  {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  MSE: {mse:.4f}")

    # Log to MLflow if there's an active run
    if mlflow.active_run():
        mlflow.log_metric("test_r2", r2)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_mse", mse)

    return {"r2": r2, "mae": mae, "mse": mse}


if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "../data/abalone.csv"

    os.makedirs("../models", exist_ok=True)

    # Set MLflow experiment — all runs will be grouped here
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"MLflow experiment: '{MLFLOW_EXPERIMENT_NAME}'")
    print("Run 'mlflow ui' after training to view results at http://localhost:5000\n")

    # Preprocess
    X, y, stats, kmeans = preprocess(
        data_path,
        kmeans_save_path="../models/kmeans.pkl"
    )
    joblib.dump(stats, "../models/stats.pkl")
    print("Standardisation stats saved.")

    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Train Neural Network — logs to MLflow automatically
    nn_model = train_neural_network(
        X_train, y_train, X_val, y_val,
        save_path="../models/neural_network.keras"
    )
    nn_metrics = evaluate(nn_model, X_test, y_test, "Neural Network")

    # Train XGBoost — logs to MLflow automatically
    xgb_model = train_xgboost(
        X_train, y_train, X_test, y_test,
        save_path="../models/xgboost.pkl"
    )
    xgb_metrics = evaluate(xgb_model, X_test, y_test, "XGBoost")

    # Summary
    print("\n--- Model Comparison ---")
    print(f"{'Model':<20} {'R2':>8} {'MAE':>8} {'MSE':>8}")
    print(f"{'Neural Network':<20} {nn_metrics['r2']:>8.4f} {nn_metrics['mae']:>8.4f} {nn_metrics['mse']:>8.4f}")
    print(f"{'XGBoost':<20} {xgb_metrics['r2']:>8.4f} {xgb_metrics['mae']:>8.4f} {xgb_metrics['mse']:>8.4f}")
    print("\nAll runs logged. View with: mlflow ui")
