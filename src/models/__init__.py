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

__all__ = [
    "create_all_candidate_models",
    "create_baseline_pipeline",
    "create_decision_tree_model",
    "create_logistic_regression_model",
    "create_random_forest_model",
    "get_model_feature_importances",
    "load_model",
    "predict_churn",
    "predict_churn_probabilities",
    "save_model",
    "train_model",
]
