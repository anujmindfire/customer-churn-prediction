# Customer Churn Prediction

## Overview

Customer Churn Prediction is a machine learning project designed to build a predictive model that identifies whether a customer is likely to churn (cancel their service or subscription) based on historical customer data.

## Objective

This project spans the complete machine learning workflow:

```text
Data Preparation
→ EDA
→ Preprocessing
→ Train/Validation/Test
→ Model Training
→ Model Comparison
→ Model Improvement
→ Evaluation
→ Prediction
→ Optional API Deployment
```

## Current Status

**Status:** Step 1 — Basic Codebase Setup complete.

The project structure, environment configuration, initial dependencies, and test framework have been set up. Dataset loading, exploratory data analysis (EDA), preprocessing, and modeling are planned for subsequent steps.

## Project Structure

```text
customer-churn-prediction/
│
├── data/
│   ├── raw/          # Raw, unmodified dataset files
│   └── processed/    # Cleaned and feature-engineered datasets
│
├── notebooks/        # Jupyter Notebooks for EDA and experimentation
│
├── src/              # Source code for the project
│   ├── __init__.py
│   ├── data/         # Data loading and ingestion scripts
│   ├── features/     # Feature engineering and transformation
│   ├── models/       # Model training, inference, and serialization
│   └── evaluation/   # Evaluation metrics and performance visualizers
│
├── models/           # Trained and serialized model binaries
│
├── reports/
│   └── figures/      # Generated plots and figure outputs
│
├── tests/            # Automated test suite
│
├── requirements.txt  # Project dependencies
├── README.md         # Project documentation
├── .gitignore        # Git ignore specifications
└── .python-version   # Python environment specification
```

## Setup & Virtual Environment

### 1. Clone the repository and navigate to the project directory

```bash
git clone <repository-url>
cd customer-churn-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

- **Linux / macOS:**

  ```bash
  source .venv/bin/activate
  ```

- **Windows (Command Prompt):**

  ```cmd
  .venv\Scripts\activate.bat
  ```

- **Windows (PowerShell):**

  ```powershell
  .venv\Scripts\Activate.ps1
  ```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Running Jupyter Notebook

To start the Jupyter Notebook server:

```bash
jupyter notebook
```

Then open `notebooks/01_data_understanding.ipynb` from the Jupyter web interface.

## Running Tests

To verify that the workspace and package imports are working correctly:

```bash
pytest tests/
```
