# Data Directory Documentation

This directory contains the dataset assets for the **Customer Churn Prediction** project.

## Directory Structure

```text
data/
├── README.md                          # Data documentation and policies (this file)
├── raw/
│   ├── .gitkeep
│   └── telco_customer_churn.csv       # Raw immutable IBM Telco Customer Churn dataset
└── processed/
    └── .gitkeep                       # Destination for cleaned, transformed, and split data
```

---

## Dataset Provenance & Overview

- **Source**: IBM Sample Data Sets / Telco Customer Churn repository (`https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`)
- **Dataset Name**: IBM Telco Customer Churn
- **Format**: CSV (Comma-Separated Values)
- **Number of Records**: 7,043
- **Number of Columns**: 21
- **Target Variable**: `Churn` (Binary: `Yes` / `No`)

---

## Target Variable

- **Column**: `Churn`
- **Definition**: Indicates whether the customer churned (left the company within the last month).
- **Class Breakdown**:
  - `No`: 5,174 records (~73.46%)
  - `Yes`: 1,869 records (~26.54%)
- **Class Balance**: Moderate imbalance (~2.77:1 ratio). Evaluation metrics should emphasize PR-AUC, ROC-AUC, Precision, Recall, and F1-score over raw accuracy alone.

---

## Feature Dictionary

The 21 columns are grouped into 4 distinct categories:

### 1. Customer Identification
- `customerID` (*string*): Unique identifier assigned to each telecom customer. Not used directly as a predictive feature.

### 2. Demographics
- `gender` (*categorical*): Gender of the customer (`Male`, `Female`).
- `SeniorCitizen` (*numeric binary*): Whether the customer is a senior citizen (`1`: Yes, `0`: No).
- `Partner` (*categorical*): Whether the customer has a partner (`Yes`, `No`).
- `Dependents` (*categorical*): Whether the customer lives with dependents (`Yes`, `No`).

### 3. Account & Financial Information
- `tenure` (*integer*): Number of months the customer has stayed with the company (0 to 72).
- `Contract` (*categorical*): Contract term (`Month-to-month`, `One year`, `Two year`).
- `PaperlessBilling` (*categorical*): Whether paperless billing is enabled (`Yes`, `No`).
- `PaymentMethod` (*categorical*): Payment method (`Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)`).
- `MonthlyCharges` (*float*): The monthly amount charged to the customer in USD.
- `TotalCharges` (*object / numeric*): Total amount charged to the customer over their tenure in USD. Contains 11 records with whitespace strings (`" "`) corresponding to customers with `tenure == 0`.

### 4. Subscribed Services
- `PhoneService` (*categorical*): Whether the customer has phone service (`Yes`, `No`).
- `MultipleLines` (*categorical*): Whether customer has multiple phone lines (`Yes`, `No`, `No phone service`).
- `InternetService` (*categorical*): Internet service provider type (`DSL`, `Fiber optic`, `No`).
- `OnlineSecurity` (*categorical*): Whether online security add-on is enabled (`Yes`, `No`, `No internet service`).
- `OnlineBackup` (*categorical*): Whether online backup add-on is enabled (`Yes`, `No`, `No internet service`).
- `DeviceProtection` (*categorical*): Whether device protection plan is enabled (`Yes`, `No`, `No internet service`).
- `TechSupport` (*categorical*): Whether premium technical support is enabled (`Yes`, `No`, `No internet service`).
- `StreamingTV` (*categorical*): Whether streaming TV service is enabled (`Yes`, `No`, `No internet service`).
- `StreamingMovies` (*categorical*): Whether streaming movie service is enabled (`Yes`, `No`, `No internet service`).

---

## Known Data Quality Nuances

1. **Whitespace in `TotalCharges`**:
   - 11 rows contain `" "` (whitespace string) instead of a valid float.
   - All 11 records have `tenure == 0`.
   - In preprocessing, these values should be converted to numeric (NaN) and imputed (e.g. set to `0.0` or dropped based on analysis).
2. **Compound Service Categories**:
   - Several service features have `"No internet service"` or `"No phone service"` in addition to `"No"`. These should be handled carefully during one-hot or ordinal encoding.

---

## Raw Storage Policy

1. **Immutability**: Raw files placed in `data/raw/` must never be altered, overwritten, edited in-place, or manually scrubbed.
2. **Reproducibility**: Any data cleaning, type conversion, missing value handling, or feature extraction must be executed via automated code in `src/` and saved to `data/processed/`.
3. **Tracking**: Version-control code scripts and configuration, and keep raw dataset artifacts documented with provenance and checksums.
