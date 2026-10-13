"""Compatibility re-export of FastAPI main application."""

from src.churn_prediction.api.main import app, health_check, lifespan, predict

__all__ = ["app", "health_check", "lifespan", "predict"]
