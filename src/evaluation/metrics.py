"""Compatibility alias for src.churn_prediction.evaluation.metrics."""

from src.churn_prediction.evaluation.metrics import (
    calculate_classification_metrics,
    compare_models,
    compute_confusion_matrix,
    format_metrics_summary,
    get_classification_report,
    get_confusion_matrix_breakdown,
)

__all__ = [
    "calculate_classification_metrics",
    "compare_models",
    "compute_confusion_matrix",
    "format_metrics_summary",
    "get_classification_report",
    "get_confusion_matrix_breakdown",
]
