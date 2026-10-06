# Customer Churn Prediction

## Overview

Customer Churn Prediction is an end-to-end machine learning project designed to build, validate, compare, improve, and deploy predictive models that identify whether a customer is likely to churn (cancel their service or subscription) using historical customer demographics, account characteristics, and service usage data.

## ML Lifecycle Architecture

```text
Step 1: Codebase Setup & Automated Testing
  ↓
Step 2: Data Loading & Quality Validation
  ↓
Step 3: Exploratory Data Analysis (EDA)
  ↓
Step 4: Data Preprocessing & Leakage-Safe 70/15/15 Split
  ↓
Step 5: Baseline Model (Logistic Regression)
  ↓
Step 6: Multiple Model Training & Validation Comparison (DT & RF)
  ↓
Step 7: Model Improvement (Tuning, Class Balancing & Thresholds)
  ↓
Step 8: Final Model Selection & Unbiased Test Evaluation  <-- [COMPLETED]
  ↓
Production Deployment & API Serving (Future)
```

---

## Current Status: Step 8 — Final Model Selection & Test Evaluation

### 1. Final Model Selection (Based Strictly on Validation Performance)
We reviewed all validation results from Steps 5, 6, and 7 and selected **Tuned Random Forest** as the final champion model:
- **Highest Validation F1 Score**: **0.6407** (surpassing baseline Random Forest of 0.5344 and baseline Logistic Regression of 0.6182).
- **Highest Churn Detection Sensitivity**: **82.14% Recall** (catching 230 out of 280 validation churners).
- **65.3% Reduction in Missed Churners**: False Negatives dropped from 144 down to 50 on validation data.
- **Overfitting Resolved**: Tree regularization (`max_depth=12`, `min_samples_leaf=8`, `min_samples_split=20`, `class_weight='balanced'`) reduced the generalization gap from $-0.4618$ to $-0.0754$.

---

### 2. Final Test Set Results (Unbiased Out-of-Sample Evaluation)

The untouched test partition (`data/processed/test.csv`: 1,057 samples, 281 actual churners / 776 retained, 26.58% churn rate) was unlocked for the first and only time.

| Partition | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Validation Set** | 0.7557 | 0.5251 | 0.8214 | 0.6407 | **0.8421** |
| **Test Set (Unbiased)** | **0.7597** | **0.5330** | **0.7758** | **0.6319** | **0.8410** |

#### Test Confusion Matrix Breakdown (1,057 Test Samples)

| Metric | Count | Percentage | Business Significance |
|:---|:---:|:---:|:---|
| **True Negatives (TN)** | **585** | 55.35% | Correctly identified loyal customers (no retention cost wasted) |
| **False Positives (FP)** | **191** | 18.07% | Retained customers flagged for retention check-in |
| **False Negatives (FN)** | **63** | 5.96% | Missed churners (reduced from 144 in baseline to only 63!) |
| **True Positives (TP)** | **218** | 20.62% | Churners successfully identified and saved (**77.58% caught**) |

*Generalization Confirmation*: The ROC-AUC difference between validation (0.8421) and test (0.8410) is less than **0.0011**, and F1 degradation is only **0.0088**, confirming zero data leakage and outstanding out-of-sample stability.

---

### 3. Production Prediction Function

The project provides an end-to-end inference function:

```python
from src.churn_prediction.predict import predict_churn

# Single customer prediction
customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 90.00,
    "TotalCharges": 90.00,
}

result = predict_churn(customer)
print(result.churn_prediction)  # 1 (Likely to Churn)
print(result.churn_probability)  # 0.6798 (67.98% Churn Risk)

# Batch DataFrame inference
import pandas as pd

df_batch = pd.DataFrame([customer, customer])
batch_results = predict_churn(df_batch)
```

---

## Project Structure

```text
customer-churn-prediction/
│
├── data/
│   ├── raw/                      # Raw dataset (telco_customer_churn.csv)
│   └── processed/                # Preprocessed partitions (train.csv, val.csv, test.csv)
│
├── notebooks/                    # Interactive research notebooks
│   ├── 01_data_understanding.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_baseline_logistic_regression.ipynb
│   ├── 05_model_comparison.ipynb
│   ├── 06_model_improvement.ipynb
│   └── 07_final_evaluation.ipynb # Step 8 final test evaluation & prediction demo
│
├── src/                          # Modular production source code
│   ├── config.py                 # Centralized configuration paths & hyperparameter defaults
│   ├── data/                     # Ingestion and validation scripts
│   ├── visualization/            # Visualization and plotting utilities
│   ├── features/                 # Preprocessing pipeline and feature encoders
│   ├── models/                   # Estimator factories, training, tuning, serialization
│   ├── evaluation/               # Metrics, comparison tables, confusion matrices, plots
│   ├── predict.py                # Production predict_churn() inference interface
│   └── churn_prediction/api/     # FastAPI application & Pydantic request/response schemas
│       ├── main.py               # Endpoints: GET /health, POST /predict
│       └── schemas.py            # Pydantic schemas with data validation
│
├── models/                       # Serialized model artifacts
│   ├── preprocessor.joblib       # Fitted ColumnTransformer preprocessing pipeline
│   ├── final_model.joblib        # Final production-ready model (Tuned Random Forest)
│   ├── final_candidate_model.joblib
│   ├── logistic_regression_baseline.joblib
│   ├── decision_tree_baseline.joblib
│   ├── random_forest_baseline.joblib
│   ├── logistic_regression_tuned.joblib
│   └── random_forest_tuned.joblib
│
├── reports/
│   └── figures/                  # Publication-ready figure exports
│       ├── final_confusion_matrix.png
│       ├── final_test_performance_comparison.png
│       ├── model_improvement_comparison.png
│       ├── improved_model_confusion_matrix.png
│       ├── threshold_tuning_curve.png
│       ├── model_validation_comparison.png
│       └── random_forest_feature_importance.png
│
├── scripts/
│   ├── train_and_compare_models.py
│   ├── train_and_tune_models.py
│   └── evaluate_final_model.py   # Final test evaluation execution script
│
├── tests/                        # Automated unit test suite (138 passing tests)
│   ├── test_api.py               # FastAPI test suite (/health, /predict, validation)
│   ├── test_data_loader.py
│   ├── test_data_validation.py
│   ├── test_preprocessing.py
│   ├── test_model.py
│   ├── test_model_comparison.py
│   ├── test_tuning.py
│   ├── test_prediction.py        # Prediction interface test suite
│   ├── test_metrics.py
│   └── test_visualization.py
│
├── requirements.txt
└── README.md
```

---

## FastAPI REST API Service (Step 9)

The churn prediction service exposes a production-ready REST API built with **FastAPI** and **Uvicorn**. It provides real-time single-customer churn inference and health monitoring without duplicating preprocessing or model logic.

### 1. Start the API Server

From the project root directory, run:

```bash
uvicorn churn_prediction.api.main:app --reload
```

Or alternatively:

```bash
uvicorn src.churn_prediction.api.main:app --reload
```

The server starts locally at `http://localhost:8000`.

### 2. Interactive API Documentation (Swagger)

FastAPI automatically generates interactive OpenAPI/Swagger documentation accessible in your browser at:

* **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
* **OpenAPI Schema**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

### 3. API Endpoints

| Method | Endpoint | Request Body | Response | Description |
|---|---|---|---|---|
| `GET` | `/health` | None | `{"status": "healthy"}` | Service health & liveness check |
| `POST` | `/predict` | `CustomerDataRequest` (JSON) | `{"prediction": "...", "probability": ...}` | Real-time customer churn prediction |

### 4. Example Requests & Responses

#### Health Check

```bash
curl -X GET http://localhost:8000/health
```

**Response:**

```json
{
  "status": "healthy"
}
```

#### Churn Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customerID": "7590-VHVEG",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
  }'
```

**Response:**

```json
{
  "prediction": "Churn",
  "probability": 0.82
}
```

---

## Setup & Execution

### 1. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Execute Final Evaluation Pipeline

```bash
python scripts/evaluate_final_model.py
```

### 3. Launch FastAPI Service

```bash
uvicorn churn_prediction.api.main:app --reload
```

### 4. Run Automated Tests & Code Quality Suite

```bash
pytest tests/
ruff check .
ruff format --check .
mypy src
```

### 5. Interactive Final Evaluation Notebook

```bash
jupyter notebook notebooks/07_final_evaluation.ipynb
```

