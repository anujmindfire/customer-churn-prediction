"""Compatibility alias for src.churn_prediction.evaluation.plots."""

from src.churn_prediction.evaluation.plots import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
)

__all__ = [
    "plot_confusion_matrix",
    "plot_feature_importance",
    "plot_model_comparison",
]
