# Customer Churn Prediction

## Overview

**Customer Churn Prediction** is a comprehensive, production-grade machine learning project designed to identify, understand, and predict customer churn in telecommunications. By analyzing historical customer demographics, subscription account characteristics, and service usage patterns, this project builds and evaluates multiple machine learning classifiers, applies rigorous optimization and class balancing techniques, and deploys the champion model through a containerized FastAPI REST service.

The project strictly adheres to machine learning engineering best practices, including:
- **Zero Data Leakage**: Complete isolation of the test dataset until final model selection, with preprocessing transformers fitted solely on training data.
- **Cost-Sensitive Optimization**: Prioritizing churn detection sensitivity (Recall) and harmonic mean (F1 Score) to minimize missed churners and protect customer lifetime value.
- **Production Readiness**: Full test coverage, strict static typing (`mypy`), code formatting (`ruff`), OpenAPI/Swagger API documentation, and Docker Compose orchestration with automated health checking.

---

## Machine Learning Lifecycle Architecture

```text
Step 1: Codebase Setup, Tooling & Automated Testing
  ↓
Step 2: Data Loading & Quality Validation (7,043 samples, schema checks)
  ↓
Step 3: Exploratory Data Analysis (EDA) & Visualizations
  ↓
Step 4: Data Preprocessing & Leakage-Safe 70/15/15 Stratified Split
  ↓
Step 5: Baseline Classification Model (Logistic Regression)
  ↓
Step 6: Multiple Model Training & Validation Comparison (DT vs. RF vs. LR)
  ↓
Step 7: Model Improvement (GridSearchCV, Class Balancing & Threshold Tuning)
  ↓
Step 8: Final Model Selection & Unbiased Test Evaluation (Tuned RF)
  ↓
Step 9: FastAPI REST API Service (Endpoints: /health, /predict, /docs)
  ↓
Step 10: Dockerization, Health Monitoring & Integration (Docker Compose)
```

---

## Dataset

The project uses the standard **Telco Customer Churn** dataset (`data/raw/telco_customer_churn.csv`):
* **Sample Size**: 7,043 customer records.
* **Feature Dimensionality**: 21 columns (1 identifier `customerID`, 1 target label `Churn`, and 19 feature attributes).
* **Feature Categories**:
  - *Demographics*: `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
  - *Subscribed Services*: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.
  - *Account & Billing Information*: `tenure` (months), `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`.
* **Data Hygiene & Missing Value Handling**:
  - The `TotalCharges` column contained 11 whitespace string values (`" "`) corresponding to new customers with `tenure == 0`.
  - These values were coerced to numeric `NaN` and imputed with the median of training data during preprocessing to prevent data leakage.
  - Zero duplicate rows were present.

---

## Exploratory Data Analysis (EDA)

Key patterns uncovered during exploratory data analysis:

1. **Target Class Imbalance**:
   - Retained Customers (`Churn = "No"`): **5,174** (73.46%)
   - Churned Customers (`Churn = "Yes"`): **1,869** (26.54%)
   - Class imbalance ratio: **2.77 : 1**. Evaluating models using accuracy alone would be misleading; Recall and F1 Score are essential.
2. **Contract Type Impact**:
   - Customers on **Month-to-month contracts** experience a churn rate exceeding **42%**, compared to **~11%** for One-year and **<3%** for Two-year contracts.
3. **Tenure & Customer Lifespan**:
   - The highest concentration of churn occurs during the first 1 to 6 months of customer tenure. Churn probability decreases substantially after year 1.
4. **Internet Service & Tech Support**:
   - Customers with **Fiber optic** internet exhibit higher churn rates than DSL or no-internet users, strongly correlated with higher monthly charges and the absence of supplementary services like `TechSupport` or `OnlineSecurity`.
5. **Payment Method**:
   - Customers paying via **Electronic check** churn at significantly higher rates (~45%) than those using credit cards, bank transfers, or mailed checks.

### Exported Visualizations
All figures are generated programmatically and saved in `reports/figures/`:
* `churn_distribution.png`: Target distribution and class imbalance breakdown.
* `tenure_vs_churn.png`: KDE and histogram analysis of tenure across churn classes.
* `monthly_charges_vs_churn.png`: Distribution of monthly charges vs. churn outcome.
* `contract_type_vs_churn.png`: Churn rate segmented by contract duration.
* `numerical_correlation_matrix.png`: Pearson correlation matrix of continuous variables.

---

## Data Preprocessing & Leakage-Safe Splitting

To guarantee zero information leakage, all preprocessing transformations were fitted exclusively on the training partition:

1. **Stratified 70 / 15 / 15 Partitioning**:
   - **Training Set (`train.csv`)**: 4,930 samples (1,308 churners, 26.53%)
   - **Validation Set (`val.csv`)**: 1,056 samples (280 churners, 26.52%)
   - **Test Set (`test.csv`)**: 1,057 samples (281 churners, 26.58%)
   - Stratified sampling ensures identical churn proportions across all splits.
2. **ColumnTransformer Preprocessing Pipeline**:
   - **Numeric Pipeline**: Handled via `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
   - **Categorical Pipeline**: Handled via `OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)`.
   - The combined transformer expands input features to 45 encoded features.
   - Fitted pipeline is saved to `models/preprocessor.joblib`.

---

## Models Evaluated

Three distinct classification algorithm families were trained on `X_train` and benchmarked on `X_val`:

1. **Logistic Regression (Baseline)**: Linear classifier optimizing log-loss with $L_2$ regularization (`max_iter=1000`, `random_state=42`).
2. **Decision Tree (`DecisionTreeClassifier`)**: Non-linear hierarchical tree partitioning (`random_state=42`).
3. **Random Forest (`RandomForestClassifier`)**: Bagging ensemble of 100 decorrelated decision trees (`n_estimators=100`, `random_state=42`).

---

## Model Comparison (Validation Benchmark)

Validation results on the 1,056 validation samples (`val.csv`):

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Missed Churners (FN) | Overfitting (F1 Gap) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression (Baseline)** | **0.8059** | **0.6459** | **0.5929** | **0.6182** | **0.8454** | **114 (10.8%)** | **+0.0202 (Stable)** |
| **Decision Tree** | 0.7178 | 0.4690 | 0.4857 | 0.4772 | 0.6430 | 144 (13.6%) | -0.5190 (Severe) |
| **Random Forest (Baseline)** | 0.7756 | 0.5939 | 0.4857 | 0.5344 | 0.8131 | 144 (13.6%) | -0.4618 (High) |

*Key Diagnostic Finding*: Without tree depth and leaf constraints, baseline Decision Tree and Random Forest memorized training data (Train F1 = 0.9962), resulting in poor validation recall (48.57%). Logistic Regression provided a stable baseline but still missed 114 actual churners (40.7% miss rate) due to default thresholding.

---

## Model Improvement Techniques

To address the high false negative rate and tree overfitting, three techniques were applied **strictly on training data**:

1. **Technique 1: Hyperparameter Tuning via 5-Fold StratifiedKFold GridSearchCV**:
   - Random Forest grid searched over `n_estimators`, `max_depth` (6–12), `min_samples_split` (5–20), and `min_samples_leaf` (2–8).
   - Logistic Regression grid searched over inverse regularization $C \in [0.01, 5.0]$.
2. **Technique 2: Cost-Sensitive Class Balancing**:
   - Tested `class_weight='balanced'` and asymmetric cost ratios to penalize False Negatives.
3. **Technique 3: Out-of-Fold Decision Threshold Optimization**:
   - Optimized decision threshold $\tau \in [0.1, 0.9]$ using out-of-fold training probability cross-validation to maximize F1 score.

### Comparison of Improved Models on Validation Set

| Model Architecture | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Missed Churners (FN) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression (Baseline)** | 0.8059 | **0.6459** | 0.5929 | 0.6182 | **0.8454** | 114 |
| **Tuned Logistic Regression** | 0.7519 | 0.5222 | 0.7571 | 0.6181 | 0.8451 | 68 |
| **LR + Optimal Threshold ($\tau=0.290$)** | 0.7472 | 0.5149 | 0.8000 | 0.6266 | **0.8454** | 56 |
| **Tuned Random Forest (Champion)** | **0.7557** | 0.5251 | **0.8214** | **0.6407** | 0.8421 | **50** |

---

## Final Model Selection & Business Rationale

**Tuned Random Forest** was selected as the champion production model based strictly on validation performance:
* **Best F1 Score**: **0.6407** (surpassing baseline Random Forest of 0.5344 and baseline Logistic Regression of 0.6182).
* **Highest Churn Detection Sensitivity**: **82.14% Recall** (capturing 230 out of 280 validation churners).
* **65.3% Reduction in Missed Churners**: False Negatives dropped from 144 down to 50.
* **Overfitting Resolved**: Tree regularization (`max_depth=12`, `min_samples_leaf=8`, `min_samples_split=20`, `class_weight='balanced'`) reduced the generalization gap from $-0.4618$ to $-0.0754$.

---

## Final Test Results (Unbiased Out-of-Sample Evaluation)

After finalizing model selection, the untouched test partition (`data/processed/test.csv`: 1,057 samples, 281 actual churners / 776 retained) was evaluated for the first and only time.

| Partition | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Validation Set** | 0.7557 | 0.5251 | 0.8214 | 0.6407 | **0.8421** |
| **Test Set (Unbiased Final)** | **0.7597** | **0.5330** | **0.7758** | **0.6319** | **0.8410** |

### Test Confusion Matrix Breakdown (1,057 Test Samples)

| Outcome Metric | Count | Percentage | Business Significance |
|:---|:---:|:---:|:---|
| **True Negatives (TN)** | **585** | 55.35% | Correctly identified loyal customers (no retention intervention needed) |
| **False Positives (FP)** | **191** | 18.07% | Retained customers flagged for proactive retention offers |
| **False Negatives (FN)** | **63** | 5.96% | Missed churners (drastically reduced from 144 in baseline) |
| **True Positives (TP)** | **218** | 20.62% | Churners successfully identified and targeted (**77.58% caught**) |

*Generalization Confirmation*: Test ROC-AUC (0.8410) matches validation ROC-AUC (0.8421) within **0.0011**, and F1 degradation is only **0.0088**, confirming zero data leakage and outstanding generalization stability.

---

## Production Prediction Function

The project provides an inference interface supporting both single-customer dictionaries and batch pandas DataFrames:

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
print(result.churn_probability)  # 0.6798 (67.98% Churn Probability)

# Batch DataFrame inference
import pandas as pd

df_batch = pd.DataFrame([customer, customer])
batch_results = predict_churn(df_batch)
```

---

## FastAPI REST API Service

The churn prediction service exposes a production REST API built with **FastAPI** and **Uvicorn**, using Pydantic for strict request validation without duplicating model or preprocessing logic.

### 1. Start the API Server

From the project root directory:

```bash
uvicorn churn_prediction.api.main:app --reload
```

The server starts at `http://localhost:8000`.

### 2. Interactive Documentation (Swagger UI)

* **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
* **OpenAPI Schema**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

### 3. API Endpoints

| Method | Endpoint | Request Body | Response | Description |
|---|---|---|---|---|
| `GET` | `/health` | None | `{"status": "healthy"}` | Service health & liveness check |
| `POST` | `/predict` | `CustomerDataRequest` (JSON) | `{"prediction": "...", "probability": ...}` | Real-time customer churn prediction |

### 4. Example Requests

#### Health Check
```bash
curl -X GET http://localhost:8000/health
```
```json
{"status": "healthy"}
```

#### Predict Churn
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "TotalCharges": 90.00
  }'
```
```json
{
  "prediction": "Churn",
  "probability": 0.68
}
```

---

## Dockerization & Container Execution

The service is fully containerized using multi-stage builds, non-root execution, and Docker Compose with automatic health monitoring.

### 1. Build and Run with Docker Compose

```bash
# Build the image
docker compose build

# Start the service in detached mode
docker compose up -d

# Check container status and health
docker compose ps

# View real-time logs
docker compose logs -f

# Stop container
docker compose down
```

### 2. Standalone Docker Run

```bash
docker build -t customer-churn-prediction-api:latest .

docker run -d \
  --name customer_churn_api \
  -p 8000:8000 \
  --env-file .env.example \
  customer-churn-prediction-api:latest
```

### 3. Automated Container Health Check

The container includes a health check running every 15 seconds against `GET /health`:

```bash
docker inspect --format='{{json .State.Health.Status}}' customer_churn_api
```
Expected output: `"healthy"`.

---

## Testing & Code Quality Instructions

The project includes an automated test suite (137 tests) covering data loading, validation, preprocessing, model training, tuning, evaluation, inference, and the FastAPI endpoints.

### Run Unit & Integration Tests

```bash
pytest tests/
```

### Run Static Analysis & Linter Suite

```bash
# Check code style and formatting
ruff check .
ruff format --check .

# Static type checking
mypy src

# Run all pre-commit hooks
pre-commit run --all-files
```

---

## Project Structure

```text
customer-churn-prediction/
│
├── data/
│   ├── raw/                      # Raw dataset (telco_customer_churn.csv)
│   └── processed/                # Preprocessed splits (train.csv, val.csv, test.csv)
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
│   ├── config.py                 # Central configuration and file path definitions
│   ├── data/                     # Ingestion and schema validation
│   ├── visualization/            # EDA plotting utilities
│   ├── features/                 # Preprocessing pipeline and ColumnTransformer
│   ├── models/                   # Estimator factories, training, and tuning
│   ├── evaluation/               # Metrics, comparison tables, confusion matrices
│   ├── predict.py                # Production predict_churn() inference interface
│   └── churn_prediction/api/     # FastAPI application & Pydantic schemas
│       ├── main.py               # Endpoints: GET /health, POST /predict
│       └── schemas.py            # Pydantic validation schemas
│
├── models/                       # Serialized model artifacts
│   ├── preprocessor.joblib       # Fitted ColumnTransformer pipeline
│   ├── final_model.joblib        # Champion model (Tuned Random Forest)
│   ├── final_candidate_model.joblib
│   ├── logistic_regression_baseline.joblib
│   ├── decision_tree_baseline.joblib
│   ├── random_forest_baseline.joblib
│   ├── logistic_regression_tuned.joblib
│   └── random_forest_tuned.joblib
│
├── reports/
│   └── figures/                  # Exported visualization figures
│       ├── churn_distribution.png
│       ├── tenure_vs_churn.png
│       ├── monthly_charges_vs_churn.png
│       ├── contract_type_vs_churn.png
│       ├── numerical_correlation_matrix.png
│       ├── model_validation_comparison.png
│       ├── threshold_tuning_curve.png
│       ├── model_improvement_comparison.png
│       ├── final_confusion_matrix.png
│       └── final_test_performance_comparison.png
│
├── scripts/
│   ├── train_and_compare_models.py
│   ├── train_and_tune_models.py
│   └── evaluate_final_model.py   # Final test evaluation execution script
│
├── tests/                        # Automated test suite (137 passing tests)
│   ├── test_api.py               # FastAPI test suite (/health, /predict, validation)
│   ├── test_data_loader.py
│   ├── test_data_validation.py
│   ├── test_preprocessing.py
│   ├── test_model.py
│   ├── test_model_comparison.py
│   ├── test_tuning.py
│   ├── test_prediction.py
│   ├── test_metrics.py
│   └── test_visualization.py
│
├── Dockerfile                    # Multi-stage production Docker image
├── .dockerignore                 # Container build exclusions
├── docker-compose.yml            # Docker Compose service definition
├── .env.example                  # Environment configuration template
├── .pre-commit-config.yaml       # Pre-commit hook configuration
├── requirements.txt              # Production and development dependencies
└── README.md                     # Project documentation
```
