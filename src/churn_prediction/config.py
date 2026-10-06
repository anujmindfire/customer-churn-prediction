"""Configuration settings and project constants."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "telco_customer_churn.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_DATA_PATH = PROCESSED_DATA_DIR / "val.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"
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

# Baseline Model settings
BASELINE_MODEL_PATH = MODELS_DIR / "logistic_regression_baseline.joblib"
LOGISTIC_REGRESSION_MODEL_PATH = BASELINE_MODEL_PATH
DECISION_TREE_MODEL_PATH = MODELS_DIR / "decision_tree_baseline.joblib"
RANDOM_FOREST_MODEL_PATH = MODELS_DIR / "random_forest_baseline.joblib"

# Improved & Final Model settings
FINAL_CANDIDATE_MODEL_PATH = MODELS_DIR / "final_candidate_model.joblib"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.joblib"
TUNED_LOGISTIC_REGRESSION_MODEL_PATH = MODELS_DIR / "logistic_regression_tuned.joblib"
TUNED_RANDOM_FOREST_MODEL_PATH = MODELS_DIR / "random_forest_tuned.joblib"

DEFAULT_LOGISTIC_REGRESSION_PARAMS = {
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
}
DEFAULT_DECISION_TREE_PARAMS = {
    "random_state": RANDOM_STATE,
}
DEFAULT_RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Visualization artifact paths
CONFUSION_MATRIX_FIGURE_PATH = FIGURES_DIR / "logistic_regression_confusion_matrix.png"
LOGISTIC_REGRESSION_CM_PATH = (
    FIGURES_DIR / "logistic_regression_validation_confusion_matrix.png"
)
DECISION_TREE_CM_PATH = FIGURES_DIR / "decision_tree_validation_confusion_matrix.png"
RANDOM_FOREST_CM_PATH = FIGURES_DIR / "random_forest_validation_confusion_matrix.png"
MODEL_COMPARISON_FIGURE_PATH = FIGURES_DIR / "model_validation_comparison.png"
FEATURE_IMPORTANCE_FIGURE_PATH = FIGURES_DIR / "random_forest_feature_importance.png"
MODEL_IMPROVEMENT_FIGURE_PATH = FIGURES_DIR / "model_improvement_comparison.png"
IMPROVED_MODEL_CM_PATH = FIGURES_DIR / "improved_model_confusion_matrix.png"
THRESHOLD_TUNING_FIGURE_PATH = FIGURES_DIR / "threshold_tuning_curve.png"
FINAL_CONFUSION_MATRIX_PATH = FIGURES_DIR / "final_confusion_matrix.png"
FINAL_COMPARISON_FIGURE_PATH = FIGURES_DIR / "final_test_performance_comparison.png"
