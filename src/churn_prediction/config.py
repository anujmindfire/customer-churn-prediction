"""Configuration settings and project constants."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "telco_customer_churn.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Random state for reproducible experiments
RANDOM_STATE = 42

# Dataset split proportions
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Target and identifier column definitions
TARGET_COL = "Churn"
ID_COLS = ["customerID"]
TARGET_MAPPING = {"No": 0, "Yes": 1}
POSITIVE_CLASS = 1
NEGATIVE_CLASS = 0
