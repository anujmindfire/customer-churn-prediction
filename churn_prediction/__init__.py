"""Compatibility root package for churn_prediction."""

from src.churn_prediction import ChurnPredictionResult, __version__, predict_churn

__all__ = ["ChurnPredictionResult", "__version__", "predict_churn"]
