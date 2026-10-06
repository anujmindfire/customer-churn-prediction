"""FastAPI application for Customer Churn Prediction service.

Exposes RESTful endpoints for health checks and real-time customer churn inference.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, status

from src.churn_prediction.api.schemas import (
    ChurnPredictionResponse,
    CustomerDataRequest,
    HealthResponse,
)
from src.churn_prediction.predict import (
    ChurnPredictionResult,
    get_inference_artifacts,
    predict_churn,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager to preload model artifacts during startup."""
    try:
        get_inference_artifacts()
        logger.info("Inference model and preprocessor artifacts loaded successfully.")
    except (FileNotFoundError, OSError, ValueError) as exc:  # pragma: no cover
        logger.warning(
            "Artifact preloading on startup failed: %s. Artifacts will be loaded on demand.",
            exc,
        )
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Production-ready FastAPI service providing real-time customer churn predictions "
        "and probability scores using a tuned Random Forest classification model."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns the operational health status of the service.",
    tags=["Health"],
)
def health_check() -> HealthResponse:
    """Return API operational status."""
    return HealthResponse(status="healthy")


@app.post(
    "/predict",
    response_model=ChurnPredictionResponse,
    summary="Predict Customer Churn",
    description=(
        "Accepts raw customer attributes, runs preprocessing and model inference, "
        "and returns churn classification and probability score."
    ),
    tags=["Prediction"],
    responses={
        200: {
            "description": "Successful prediction response.",
            "model": ChurnPredictionResponse,
        },
        422: {
            "description": "Validation error for missing or invalid input attributes."
        },
        500: {"description": "Internal server error during prediction execution."},
    },
)
def predict(request: CustomerDataRequest) -> ChurnPredictionResponse:
    """Execute churn prediction for a single customer.

    Args:
        request: CustomerDataRequest containing customer demographics, service plans,
            and billing records.

    Returns:
        ChurnPredictionResponse: Churn classification ('Churn' or 'No Churn')
            and calibrated churn probability score.

    Raises:
        HTTPException: 422 if input data is rejected by inference logic,
            or 500 if an internal prediction failure occurs.
    """
    try:
        payload = request.model_dump()
        result = predict_churn(payload)

        if not isinstance(result, ChurnPredictionResult):
            # In case a DataFrame was unexpectedly returned for single dict
            churn_pred = int(result["churn_prediction"].iloc[0])
            churn_prob = float(result["churn_probability"].iloc[0])
        else:
            churn_pred = result.churn_prediction
            churn_prob = result.churn_probability

        label: Literal["Churn", "No Churn"] = "Churn" if churn_pred == 1 else "No Churn"
        return ChurnPredictionResponse(
            prediction=label,
            probability=round(churn_prob, 4),
        )
    except ValueError as exc:
        logger.warning("Validation error during churn prediction: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Internal prediction error occurred.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing the prediction request.",
        ) from exc
