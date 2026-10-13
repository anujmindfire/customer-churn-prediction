"""Compatibility re-export of API schemas."""

from src.churn_prediction.api.schemas import (
    ChurnPredictionResponse,
    CustomerDataRequest,
    HealthResponse,
)

__all__ = [
    "ChurnPredictionResponse",
    "CustomerDataRequest",
    "HealthResponse",
]
