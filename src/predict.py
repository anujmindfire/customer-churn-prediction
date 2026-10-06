"""Compatibility alias for src.churn_prediction.predict."""

from src.churn_prediction.predict import (
    ChurnPredictionResult,
    get_inference_artifacts,
    predict_churn,
)

__all__ = [
    "ChurnPredictionResult",
    "get_inference_artifacts",
    "predict_churn",
]
