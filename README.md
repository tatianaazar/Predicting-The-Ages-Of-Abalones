# Abalone Age Predictor — Production ML System

A production-grade machine learning system that predicts the age of abalone from physical measurements. Built on top of MSc Data Science coursework (Lancaster University Leipzig, Distinction), refactored into a deployable ML pipeline with experiment tracking, a REST API, and Docker containerisation.

**Live demo:** `http://localhost:8000/docs` (run locally — see [Quickstart](#quickstart))

---

## The Problem

Determining abalone age normally requires cutting the shell, staining it, and counting rings under a microscope — a slow, destructive process. This system predicts age from non-destructive physical measurements (weight, dimensions, sex), making it useful for aquaculture pricing and conservation monitoring.

**Age = Rings + 1.5** (biological convention). Rings are excluded from the model features — the goal is to predict without them.

---

## Results

| Model | R² | MAE | MSE |
|---|---|---|---|
| **Neural Network** | **0.67** | **1.36 years** | **3.85** |
| XGBoost | 0.61 | 1.49 years | 4.17 |
| Random Forest (baseline) | 0.58 | — | — |
| LASSO (baseline) | 0.52 | — | — |

The neural network outperforms all baselines. Typical prediction error is **±1–2 years**.

---

## Key Finding: K-Means as a Feature Engineering Step

The most impactful preprocessing decision was using K-means clustering (k=3, elbow method) to generate cluster membership features, which were one-hot encoded and appended to the feature matrix.

| Configuration | R² | MAE | MSE |
|---|---|---|---|
| Without clustering | 0.57 | 1.57 | 5.20 |
| **With clustering (k=3)** | **0.67** | **1.36** | **3.85** |

Abalones of similar age tend to cluster in the multidimensional measurement space. Adding cluster labels gives the model an explicit signal about these groupings, capturing non-linear structure that the raw features alone don't expose.

Other dimensionality reduction techniques (PCA, autoencoders) were tested and discarded — they consistently reduced performance, indicating that all 8 features carry meaningful, non-redundant information for this task.

---

## Architecture

```
Raw Input (8 measurements)
        │
        ▼
┌─────────────────────────┐
│      preprocess.py      │
│  • Encode Sex (M/F/I)   │
│  • Z-score standardise  │
│  • K-means cluster (k=3)│
│  • One-hot cluster labels│
└────────────┬────────────┘
             │  11 features
             ▼
┌─────────────────────────┐
│    Neural Network       │
│  Input(11)              │
│  → Dense(64, ReLU, L1)  │
│  → Dense(32, ReLU)      │
│  → Dropout(0.2)         │
│  → Dense(1)             │
└────────────┬────────────┘
             │
             ▼
     Predicted Age (years)
```

**Design decisions (all experimentally validated):**
- **ReLU activation** — appropriate for a positive target (age), less computationally expensive than LeakyReLU
- **L1 regularisation** on first layer — reduces overfitting driven by the dataset's many outliers
- **Adam optimiser** — outperformed SGD on this dataset
- **Manual z-score standardisation** — outperformed `sklearn.StandardScaler` in experiments
- **Outliers retained** — removing outliers (tested at k=2.0 to 4.0 IQR multipliers) consistently worsened performance; they carry real signal

---

## Project Structure

```
abalone-ml-system/
├── src/
│   ├── preprocess.py     # Data loading, encoding, standardisation, K-means
│   ├── train.py          # Model training with MLflow experiment tracking
│   ├── predict.py        # CLI inference tool
│   └── api.py            # FastAPI REST API
├── models/               # Saved artifacts (generated after training)
│   ├── neural_network.keras
│   ├── xgboost.pkl
│   ├── kmeans.pkl
│   └── stats.pkl         # Training-set standardisation stats (used at inference)
├── data/                 # Place abalone dataset here
├── notebooks/            # Exploratory analysis
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/abalone-ml-system.git
cd abalone-ml-system
pip install -r requirements.txt
```

### 2. Train the models

Place the abalone dataset (CSV, no header) in `data/abalone.csv`, then:

```bash
cd src
python train.py ../data/abalone.csv
```

This will:
- Preprocess the data and fit the K-means clusterer
- Train both the Neural Network and XGBoost models
- Save all artifacts to `models/`
- Log all parameters, metrics, and loss curves to MLflow

### 3. View experiment results

```bash
mlflow ui
```

Open `http://localhost:5000` to compare runs, inspect loss curves, and review hyperparameters.

### 4. Run the API

```bash
uvicorn api:app --reload --port 8000
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

### 5. Predict from the command line

```bash
python predict.py --sex M --length 0.45 --diameter 0.35 --height 0.11 --whole_weight 0.45 --shucked_weight 0.19 --viscera_weight 0.10 --shell_weight 0.14
# → Predicted Abalone Age: 10.97 years
```

---

## API Reference

**Base URL:** `http://localhost:8000`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/model/info` | Model metadata, architecture, test metrics |
| `POST` | `/predict` | Predict age for one abalone |
| `POST` | `/predict/batch` | Predict age for up to 100 abalones |

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sex": "M",
    "length": 0.455,
    "diameter": 0.365,
    "height": 0.095,
    "whole_weight": 0.514,
    "shucked_weight": 0.2245,
    "viscera_weight": 0.101,
    "shell_weight": 0.15
  }'
```

### Example response

```json
{
  "predicted_age_years": 10.97,
  "model": "Neural Network (R²=0.67, MAE=±1.36 years)",
  "note": "Age is estimated from physical measurements. Typical error is ±1–2 years."
}
```

---

## Docker

### Build and run

```bash
# Build the image
docker build -t abalone-predictor .

# Run the container
docker run -p 8000:8000 abalone-predictor
```

The API will be available at `http://localhost:8000/docs`.

### Notes
- The `models/` directory must exist and contain trained artifacts before building the image
- Run `python src/train.py data/abalone.csv` first to generate the model files

---

## Dataset

**UCI Abalone Dataset** — 4,177 samples, 9 features (8 predictors + rings target)

| Feature | Type | Unit | Description |
|---|---|---|---|
| Sex | Categorical | M/F/I | Male, Female, Infant |
| Length | Continuous | mm | Longest shell measurement |
| Diameter | Continuous | mm | Perpendicular to length |
| Height | Continuous | mm | With meat in shell |
| Whole weight | Continuous | g | Whole abalone |
| Shucked weight | Continuous | g | Weight of meat |
| Viscera weight | Continuous | g | Gut weight after bleeding |
| Shell weight | Continuous | g | After drying |
| Rings | Integer | — | **Target proxy** (Age = Rings + 1.5) |

**Key observations from EDA:**
- Features are highly intercorrelated (Pearson r up to 0.99 between dimensions)
- Weak correlation with target (highest: shell weight at r=0.63)
- Significant outliers present — retained after systematic testing showed removal worsened performance
- Male and Female distributions overlap heavily; Infant is a distinct sub-group

---

## Stack

| Layer | Technology |
|---|---|
| Model training | TensorFlow / Keras, XGBoost, scikit-learn |
| Experiment tracking | MLflow |
| API | FastAPI + Uvicorn |
| Containerisation | Docker |
| Language | Python 3.11 |

---

## What's Next

- [ ] GitHub Actions CI/CD — auto-retrain and redeploy on data push
- [ ] Evidently drift monitoring — detect when incoming data shifts from training distribution
- [ ] React frontend — browser-based prediction form
- [ ] Cloud deployment — Railway or Render (free tier)
