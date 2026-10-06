# Customer Churn Prediction

## Overview

Customer Churn Prediction is an end-to-end machine learning project designed to build, validate, and compare predictive models that identify whether a customer is likely to churn (cancel their service or subscription) using historical customer demographics, account characteristics, and service usage data.

## ML Lifecycle Architecture

```text
Step 1: Codebase Setup
  ↓
Step 2: Data Loading & Validation
  ↓
Step 3: Exploratory Data Analysis (EDA)
  ↓
Step 4: Data Preprocessing & 70/15/15 Split
  ↓
Step 5: Baseline Model (Logistic Regression)
  ↓
Step 6: Multiple Model Training & Validation Comparison  <-- [COMPLETED]
  ↓
Step 7: Hyperparameter Tuning, Class Balancing & Final Test Evaluation (Upcoming)
  ↓
Step 8: Serving & Model Deployment
```

---

## Current Status: Step 6 — Multiple Model Training and Validation Comparison

Three candidate models were trained strictly on `data/processed/train.csv` (4,930 samples, 45 preprocessed features) and evaluated on `data/processed/val.csv` (1,056 samples):

1. **Logistic Regression (Baseline)**: $L_2$ regularized linear classification (`max_iter=1000`, `random_state=42`)
2. **Decision Tree**: Unconstrained recursive tree partitioner (`DecisionTreeClassifier(random_state=42)`)
3. **Random Forest**: 100-tree bagging ensemble (`RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)`)

> **Strict Test Set Protection**: `data/processed/test.csv` remains completely locked and untouched. All model comparisons and selections are performed strictly on the validation partition.

---

### Validation Performance Comparison Table

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression (Baseline)** | **0.8059** | **0.6459** | **0.5929** | **0.6182** | **0.8454** |
| **Decision Tree** | 0.7178 | 0.4690 | 0.4857 | 0.4772 | 0.6430 |
| **Random Forest** | 0.7756 | 0.5939 | 0.4857 | 0.5344 | 0.8131 |

### Confusion Matrix Breakdown (1,056 Validation Samples)

| Model | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) |
|:---|:---:|:---:|:---:|:---:|
| **Logistic Regression** | **685 (64.87%)** | **91 (8.62%)** | **114 (10.80%)** | **166 (15.72%)** |
| **Decision Tree** | 622 (58.90%) | 154 (14.58%) | 144 (13.64%) | 136 (12.88%) |
| **Random Forest** | 683 (64.68%) | 93 (8.81%) | 144 (13.64%) | 136 (12.88%) |

*Key Takeaway*: Logistic Regression catches **30 more actual churners** (166 vs 136, 59.29% recall vs 48.57%) and generates fewer false alarms than the unconstrained tree models.

---

### Overfitting & Generalization Diagnostics

| Model | Train Accuracy | Val Accuracy | Train F1 | Val F1 | Generalization Gap (F1) | Diagnostic |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Logistic Regression** | 0.8037 | 0.8059 | 0.5980 | 0.6182 | **+0.0202** | **Stable (Zero variance error)** |
| **Decision Tree** | 0.9980 | 0.7178 | 0.9962 | 0.4772 | **-0.5190** | **Severe Overfitting (Memorization)** |
| **Random Forest** | 0.9980 | 0.7756 | 0.9962 | 0.5344 | **-0.4618** | **High Overfitting (Needs pruning)** |

- **Decision Tree** completely memorizes training patterns (Train F1 = 0.9962) but degrades sharply on validation data (Val F1 = 0.4772).
- **Random Forest** benefits from bagging variance reduction, lifting ROC-AUC to 0.8131 (vs 0.6430 for DT), but unpruned deep trees still exhibit high variance.
- **Logistic Regression** emerges as the clear **validation champion** because linear inductive bias prevents noise memorization in the absence of hyperparameter tuning.

---

### Top Predictive Features (Random Forest Gini Importance)

1. `TotalCharges`: **16.0%**
2. `tenure`: **13.6%**
3. `MonthlyCharges`: **13.5%**
4. `Contract_Month-to-month`: **5.0%**
5. `PaymentMethod_Electronic check`: **3.4%**
6. `InternetService_Fiber optic`: **3.1%**

> **Non-Causal Interpretation**: Gini feature importance measures **predictive partitioning utility** across tree splits, not causal mechanisms. High tenure and TotalCharges signify account maturity and duration of service, not a direct causal intervention to prevent churn.

---

## Project Structure

```text
customer-churn-prediction/
│
├── data/
│   ├── raw/                      # Raw, unmodified dataset files
│   └── processed/                # Preprocessed partitions (train.csv, val.csv, test.csv)
│
├── notebooks/                    # Interactive analysis notebooks
│   ├── 01_data_understanding.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_baseline_logistic_regression.ipynb
│   └── 05_model_comparison.ipynb # Step 6 multi-model training and validation comparison
│
├── src/                          # Reusable project source code
│   ├── config.py                 # Configuration paths and hyperparameter defaults
│   ├── data/                     # Data loading and ingestion scripts
│   ├── visualization/            # Data visualization and plotting utilities
│   ├── features/                 # Preprocessing pipeline and feature encoders
│   ├── models/                   # Estimator factories, training, inference, serialization
│   └── evaluation/               # Metric computers, comparison tables, confusion matrices
│
├── models/                       # Serialized model artifacts
│   ├── preprocessor.joblib
│   ├── logistic_regression_baseline.joblib
│   ├── decision_tree_baseline.joblib
│   └── random_forest_baseline.joblib
│
├── reports/
│   └── figures/                  # Publication-quality figure exports
│       ├── model_validation_comparison.png
│       ├── random_forest_feature_importance.png
│       ├── logistic_regression_validation_confusion_matrix.png
│       ├── decision_tree_validation_confusion_matrix.png
│       └── random_forest_validation_confusion_matrix.png
│
├── scripts/
│   └── train_and_compare_models.py # Executable training and comparison script
│
├── tests/                        # Comprehensive automated test suite (pytest)
│   ├── test_data_loader.py
│   ├── test_data_validation.py
│   ├── test_preprocessing.py
│   ├── test_model.py
│   ├── test_model_comparison.py
│   ├── test_metrics.py
│   └── test_visualization.py
│
├── pyproject.toml / requirements.txt
└── README.md
```

---

## Setup & Execution

### 1. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train and Compare Models

Execute the training script to fit all candidate models on `train.csv`, evaluate on `val.csv`, and export serialized models and figures:

```bash
python scripts/train_and_compare_models.py
```

### 3. Run Automated Tests

```bash
pytest tests/
```

### 4. Open Step 6 Notebook

```bash
jupyter notebook notebooks/05_model_comparison.ipynb
```
