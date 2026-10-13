"""Compatibility alias for src.churn_prediction.models."""

from src.churn_prediction.models.model import (
    create_all_candidate_models,
    create_baseline_pipeline,
    create_decision_tree_model,
    create_logistic_regression_model,
    create_random_forest_model,
    get_model_feature_importances,
    load_model,
    predict_churn,
    predict_churn_probabilities,
    save_model,
    train_model,
)
from src.churn_prediction.models.tuning import (
    find_optimal_threshold,
    find_optimal_threshold_cv,
    predict_with_threshold,
    tune_logistic_regression,
    tune_random_forest,
)

__all__ = [
    "create_all_candidate_models",
    "create_baseline_pipeline",
    "create_decision_tree_model",
    "create_logistic_regression_model",
    "create_random_forest_model",
    "find_optimal_threshold",
    "find_optimal_threshold_cv",
    "get_model_feature_importances",
    "load_model",
    "predict_churn",
    "predict_churn_probabilities",
    "predict_with_threshold",
    "save_model",
    "train_model",
    "tune_logistic_regression",
    "tune_random_forest",
]
