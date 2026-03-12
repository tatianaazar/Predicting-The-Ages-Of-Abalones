"""
preprocess.py
-------------
All data loading, cleaning, feature engineering, and preprocessing logic
for the Abalone age prediction pipeline.

Key design decisions (from experimental report):
- Outliers are retained: removing them consistently worsened model performance
- Manual standardisation outperformed sklearn's StandardScaler
- K-means clustering (k=3, elbow method) used as a feature engineering step:
  adding cluster labels as one-hot features improved R2 from 0.57 to 0.67
- PCA and autoencoders were tested and discarded: dimensionality is important here
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import joblib
import os

# Column names for the raw dataset (no header in source file)
COLUMN_NAMES = [
    "Sex", "Length", "Diameter", "Height",
    "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"
]

# Features to standardise (all continuous physical measurements)
COLS_TO_STANDARDIZE = [
    "Length", "Diameter", "Height",
    "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight"
]

# Final feature set fed to the model
FEATURE_COLS = [
    "Sex",
    "Length_Standardized", "Diameter_Standardized", "Height_Standardized",
    "Whole_weight_Standardized", "Shucked_weight_Standardized",
    "Viscera_weight_Standardized", "Shell_weight_Standardized"
]

N_CLUSTERS = 3  # Optimal k determined via elbow method
RANDOM_STATE = 42


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw abalone dataset from CSV."""
    df = pd.read_csv(filepath, header=None, names=COLUMN_NAMES)
    print(f"Loaded {len(df)} records from {filepath}")
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive Age from Rings.
    Age = Rings + 1.5 (biological convention for abalone).
    Rings column is then dropped from features.
    """
    df = df.copy()
    df["Age"] = df["Rings"] + 1.5
    return df


def encode_sex(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical Sex feature numerically.
    M -> 0, F -> 1, I -> 2
    
    Note: Male and Female distributions overlap heavily (confirmed via boxplots).
    Infant is distinct but the feature still marginally improves predictions.
    """
    df = df.copy()
    df["Sex"] = df["Sex"].map({"M": 0, "F": 1, "I": 2})
    return df


def standardize_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Manually standardise continuous features using z-score normalisation.
    
    Manual standardisation was used rather than sklearn's StandardScaler
    as it yielded better model performance experimentally.
    
    Returns:
        df: DataFrame with added *_Standardized columns
        stats: dict of {col: (mean, std)} for use at inference time
    """
    df = df.copy()
    stats = {}
    for col in COLS_TO_STANDARDIZE:
        mean = df[col].mean()
        std = df[col].std()
        df[f"{col}_Standardized"] = (df[col] - mean) / std
        stats[col] = (mean, std)
    return df, stats


def standardize_features_inference(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """
    Apply pre-computed standardisation stats to new data at inference time.
    Uses training set mean/std to avoid data leakage.
    """
    df = df.copy()
    for col, (mean, std) in stats.items():
        df[f"{col}_Standardized"] = (df[col] - mean) / std
    return df


def fit_kmeans(X: pd.DataFrame, save_path: str = None) -> KMeans:
    """
    Fit K-means clustering on feature matrix.
    k=3 determined via elbow method on sum of squared distances.
    
    Cluster labels are added as one-hot features, which improved R2 from
    0.57 to 0.67 by capturing latent age-related groupings in the data.
    """
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
    kmeans.fit(X)
    if save_path:
        joblib.dump(kmeans, save_path)
        print(f"K-means model saved to {save_path}")
    return kmeans


def add_cluster_features(X: pd.DataFrame, kmeans: KMeans) -> pd.DataFrame:
    """
    Predict cluster labels and append as one-hot encoded columns.
    Cluster_0, Cluster_1, Cluster_2 are always present (even at inference
    with a single row) because we use pd.Categorical with fixed categories.
    """
    clusters = kmeans.predict(X)
    cluster_series = pd.Categorical(clusters, categories=range(N_CLUSTERS))
    cluster_dummies = pd.get_dummies(cluster_series, prefix="Cluster").astype(int)
    cluster_dummies.index = X.reset_index(drop=True).index
    return pd.concat([X.reset_index(drop=True), cluster_dummies.reset_index(drop=True)], axis=1)


def preprocess(filepath: str, kmeans_save_path: str = None) -> tuple:
    """
    Full preprocessing pipeline for training.
    
    Steps:
        1. Load raw data
        2. Derive Age target from Rings
        3. Encode Sex feature
        4. Standardise continuous features (manual z-score)
        5. Fit K-means and add cluster one-hot features
    
    Returns:
        X: feature DataFrame (standardised + cluster features)
        y: target Series (Age)
        stats: standardisation stats for inference
        kmeans: fitted KMeans object for inference
    """
    df = load_data(filepath)
    df = add_target(df)
    df = encode_sex(df)
    df, stats = standardize_features(df)

    X = df[FEATURE_COLS].copy()
    y = df["Age"]

    kmeans = fit_kmeans(X, save_path=kmeans_save_path)
    X = add_cluster_features(X, kmeans)

    print(f"Preprocessing complete. Feature shape: {X.shape}")
    return X, y, stats, kmeans


def preprocess_inference(raw_input: dict, stats: dict, kmeans: KMeans) -> pd.DataFrame:
    """
    Preprocess a single inference input using training-time stats.
    
    Args:
        raw_input: dict with keys matching COLUMN_NAMES (excluding Rings/Age)
        stats: standardisation stats from training
        kmeans: fitted KMeans from training
    
    Returns:
        X: single-row DataFrame ready for model prediction
    """
    df = pd.DataFrame([raw_input])
    df = encode_sex(df)
    df = standardize_features_inference(df, stats)

    X = df[FEATURE_COLS].copy()
    X = add_cluster_features(X, kmeans)
    return X
