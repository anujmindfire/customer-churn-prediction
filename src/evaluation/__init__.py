"""Model evaluation and metrics modules."""

from src.churn_prediction.evaluation.metrics import (
    calculate_classification_metrics,
    compare_models,
    compute_confusion_matrix,
    format_metrics_summary,
    get_classification_report,
    get_confusion_matrix_breakdown,
)
from src.churn_prediction.evaluation.plots import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
    plot_threshold_tuning_curve,
)

__all__ = [
    "calculate_classification_metrics",
    "compare_models",
    "compute_confusion_matrix",
    "format_metrics_summary",
    "get_classification_report",
    "get_confusion_matrix_breakdown",
    "plot_confusion_matrix",
    "plot_feature_importance",
    "plot_model_comparison",
    "plot_threshold_tuning_curve",
]
