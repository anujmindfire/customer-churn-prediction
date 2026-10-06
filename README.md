# Customer Churn Prediction

## Overview

Customer Churn Prediction is an end-to-end machine learning project designed to build, validate, compare, and improve predictive models that identify whether a customer is likely to churn (cancel their service or subscription) using historical customer demographics, account characteristics, and service usage data.

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
Step 6: Multiple Model Training & Validation Comparison
  ↓
Step 7: Model Improvement (Tuning, Class Balancing & Thresholds)  <-- [COMPLETED]
  ↓
Step 8: Final Test Evaluation & Production Artifacts (Upcoming)
```

---

## Current Status: Step 7 — Model Improvement

In Step 7, we applied systematic improvement techniques to address the high rate of missed churners (False Negatives) in the baseline models. All hyperparameter searches and threshold discoveries were conducted **strictly on training data** using 5-fold Stratified Cross-Validation.

> **Strict Test Set Protection**: `data/processed/test.csv` remains completely locked, unopened, and untouched. All evaluations and comparisons are performed solely on `data/processed/val.csv` (1,056 samples, 280 churners / 776 retained).

---

### Improvement Techniques Applied

1. **Hyperparameter Tuning (`GridSearchCV`)**:
   - Logistic Regression: Tuned regularization inverse $C \in [0.01, 5.0]$ and solver parameters.
   - Random Forest: Regularized tree architecture (`max_depth=12`, `min_samples_leaf=8`, `min_samples_split=20`, `n_estimators=100`) to resolve the severe overfitting observed in Step 6.
2. **Class Balancing (Cost-Sensitive Weighting)**:
   - Evaluated `class_weight='balanced'` and custom cost ratios (`{0: 1.0, 1: 2.0}`) to penalize false negatives and boost churner detection.
3. **Out-of-Fold Decision Threshold Optimization**:
   - Evaluated candidate cutoffs $\tau \in [0.1, 0.9]$ on cross-validated training probability predictions without data leakage. Found optimal cutoff $\tau = 0.290$ (peak CV F1 = 0.6359).

---

### Validation Performance Comparison Table

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Missed Churners (FN) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression (Baseline)** | **0.8059** | **0.6459** | 0.5929 | 0.6182 | **0.8454** | 114 |
| **Tuned Logistic Regression** | 0.7519 | 0.5222 | 0.7571 | 0.6181 | 0.8451 | 68 (-40.4%) |
| **Baseline LR + Optimal Threshold ($\tau=0.290$)** | 0.7472 | 0.5149 | 0.8000 | 0.6266 | **0.8454** | 56 (-50.9%) |
| **Tuned Random Forest (Champion)** | 0.7557 | 0.5251 | **0.8214** | **0.6407** | 0.8421 | **50 (-65.3%)** |

### Confusion Matrix Breakdown (1,056 Validation Samples)

| Model | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Churn Detection Rate |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Baseline LR** | **685 (64.87%)** | **91 (8.62%)** | 114 (10.80%) | 166 (15.72%) | 59.29% |
| **Tuned LR** | 582 (55.11%) | 194 (18.37%) | 68 (6.44%) | 212 (20.08%) | 75.71% |
| **LR + Optimal Threshold** | 565 (53.50%) | 211 (19.98%) | 56 (5.30%) | 224 (21.21%) | 80.00% |
| **Tuned RF (Champion)** | 568 (53.79%) | 208 (19.70%) | **50 (4.73%)** | **230 (21.78%)** | **82.14%** |

---

### Key Business & Technical Takeaways

1. **Massive Reduction in Missed Churners**: Tuned Random Forest catches **230 out of 280 churners** (Recall 82.14%), slashing missed churners from 144 down to 50—a **65.3% reduction in lost retention opportunities**.
2. **Overfitting Successfully Resolved**: By regularizing maximum depth and minimum leaf samples, Random Forest training-to-validation F1 gap shrank from $-0.4618$ to $-0.0754$, turning a severely overfitted baseline into the highest-performing generalizer.
3. **Current Best Model**: **Tuned Random Forest** is serialized as `models/final_candidate_model.joblib`.

---

## Project Structure

```text
customer-churn-prediction/
│
├── data/
│   ├── raw/                      # Raw dataset files
│   └── processed/                # Preprocessed partitions (train.csv, val.csv, test.csv)
│
├── notebooks/                    # Interactive analysis notebooks
│   ├── 01_data_understanding.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_baseline_logistic_regression.ipynb
│   ├── 05_model_comparison.ipynb
│   └── 06_model_improvement.ipynb # Step 7 hyperparameter tuning & class balancing
│
├── src/                          # Reusable project source code
│   ├── config.py                 # Configuration paths and hyperparameter defaults
│   ├── data/                     # Data loading and ingestion scripts
│   ├── visualization/            # Data visualization and plotting utilities
│   ├── features/                 # Preprocessing pipeline and feature encoders
│   ├── models/                   # Estimator factories, training, inference, tuning
│   │   ├── model.py
│   │   └── tuning.py             # GridSearchCV and threshold optimization utilities
│   └── evaluation/               # Metrics, comparison tables, confusion matrices, plots
│
├── models/                       # Serialized model artifacts
│   ├── preprocessor.joblib
│   ├── logistic_regression_baseline.joblib
│   ├── decision_tree_baseline.joblib
│   ├── random_forest_baseline.joblib
│   ├── logistic_regression_tuned.joblib
│   ├── random_forest_tuned.joblib
│   └── final_candidate_model.joblib # Current best model (Tuned Random Forest)
│
├── reports/
│   └── figures/                  # Publication-quality figure exports
│       ├── model_validation_comparison.png
│       ├── model_improvement_comparison.png
│       ├── improved_model_confusion_matrix.png
│       ├── threshold_tuning_curve.png
│       └── random_forest_feature_importance.png
│
├── scripts/
│   ├── train_and_compare_models.py # Step 6 execution script
│   └── train_and_tune_models.py    # Step 7 tuning and improvement script
│
├── tests/                        # Comprehensive automated test suite (115 passing tests)
│   ├── test_data_loader.py
│   ├── test_data_validation.py
│   ├── test_preprocessing.py
│   ├── test_model.py
│   ├── test_model_comparison.py
│   ├── test_tuning.py            # Tuning & threshold optimization test suite
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

### 2. Run Model Improvement Pipeline

```bash
python scripts/train_and_tune_models.py
```

### 3. Run Automated Tests & Quality Checks

```bash
pytest tests/
ruff check .
ruff format --check .
mypy src
```

### 4. Interactive Notebook

```bash
jupyter notebook notebooks/06_model_improvement.ipynb
```
