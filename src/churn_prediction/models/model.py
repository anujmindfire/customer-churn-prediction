"""Logistic Regression baseline model utilities for Customer Churn Prediction.

This module provides factory functions to create, train, evaluate, serialize,
and deserialize the baseline Logistic Regression model for customer churn prediction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.exceptions import NotFittedError  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]

from src.churn_prediction.config import (
    BASELINE_MODEL_PATH,
    DEFAULT_LOGISTIC_REGRESSION_PARAMS,
    RANDOM_STATE,
)

logger = logging.getLogger(__name__)


def create_logistic_regression_model(
    random_state: int = RANDOM_STATE,
    max_iter: int = 1000,
    **kwargs: Any,
) -> LogisticRegression:
    """Create and configure a baseline Logistic Regression estimator.

    Args:
        random_state: Pseudo-random seed for reproducibility. Defaults to RANDOM_STATE (42).
        max_iter: Maximum number of solver convergence iterations. Defaults to 1000.
        **kwargs: Additional keyword arguments passed to LogisticRegression.

    Returns:
        LogisticRegression: Unfitted scikit-learn LogisticRegression instance.

    Raises:
        ValueError: If max_iter is non-positive.
    """
    if max_iter <= 0:
        raise ValueError(f"max_iter must be greater than 0, got {max_iter}")

    params = {
        **DEFAULT_LOGISTIC_REGRESSION_PARAMS,
        "random_state": random_state,
        "max_iter": max_iter,
        **kwargs,
    }

    logger.info("Initializing LogisticRegression with parameters: %s", params)
    return LogisticRegression(**params)


def train_model(
    model: LogisticRegression,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
) -> LogisticRegression:
    """Train the Logistic Regression estimator strictly on training data.

    Args:
        model: Unfitted LogisticRegression instance.
        X_train: Preprocessed training feature matrix (shape: [n_samples, n_features]).
        y_train: Training binary target labels (shape: [n_samples]).

    Returns:
        LogisticRegression: Fitted LogisticRegression instance.

    Raises:
        TypeError: If inputs are invalid types.
        ValueError: If sample counts do not match or if training data is empty.
    """
    if not isinstance(model, LogisticRegression):
        raise TypeError(
            f"Expected LogisticRegression model, got {type(model).__name__}"
        )

    n_samples_X = len(X_train)
    n_samples_y = len(y_train)

    if n_samples_X == 0 or n_samples_y == 0:
        raise ValueError("Cannot train model on empty training data.")

    if n_samples_X != n_samples_y:
        msg = (
            f"Feature count ({n_samples_X}) does not match label count ({n_samples_y})."
        )
        logger.error(msg)
        raise ValueError(msg)

    logger.info(
        "Fitting LogisticRegression baseline model on %d training samples...",
        n_samples_X,
    )
    model.fit(X_train, y_train)
    logger.info(
        "Model training complete. Intercept: %s, Number of features: %d.",
        getattr(model, "intercept_", None),
        getattr(model, "n_features_in_", 0),
    )
    return model


def predict_churn(
    model: LogisticRegression,
    X: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """Generate binary churn predictions (0 = Retained, 1 = Churned).

    Uses the standard decision threshold of 0.5.

    Args:
        model: Fitted LogisticRegression instance.
        X: Preprocessed feature matrix (shape: [n_samples, n_features]).

    Returns:
        np.ndarray: 1D array of binary predictions (dtype int64).

    Raises:
        TypeError: If model is not a LogisticRegression.
        NotFittedError: If model has not been fitted yet.
        ValueError: If input feature matrix is empty.
    """
    if not isinstance(model, LogisticRegression):
        raise TypeError(
            f"Expected LogisticRegression model, got {type(model).__name__}"
        )
    if len(X) == 0:
        raise ValueError("Cannot predict on empty feature matrix.")

    try:
        predictions: np.ndarray = model.predict(X)
    except NotFittedError as err:
        logger.error("Prediction failed: model has not been fitted.")
        raise NotFittedError(
            "LogisticRegression model must be fitted before predicting."
        ) from err

    logger.debug("Generated %d predictions.", len(predictions))
    return predictions.astype(np.int64)


def predict_churn_probabilities(
    model: LogisticRegression,
    X: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """Generate predicted probabilities of customer churn: P(Churn = 1 | X).

    Args:
        model: Fitted LogisticRegression instance.
        X: Preprocessed feature matrix (shape: [n_samples, n_features]).

    Returns:
        np.ndarray: 1D array of churn probabilities between 0.0 and 1.0.

    Raises:
        TypeError: If model is not a LogisticRegression.
        NotFittedError: If model has not been fitted yet.
        ValueError: If input feature matrix is empty.
    """
    if not isinstance(model, LogisticRegression):
        raise TypeError(
            f"Expected LogisticRegression model, got {type(model).__name__}"
        )
    if len(X) == 0:
        raise ValueError("Cannot predict probabilities on empty feature matrix.")

    try:
        proba: np.ndarray = model.predict_proba(X)
    except NotFittedError as err:
        logger.error("Probability prediction failed: model has not been fitted.")
        raise NotFittedError(
            "LogisticRegression model must be fitted before predicting probabilities."
        ) from err

    # Extract column corresponding to positive class (Churn = 1)
    churn_probabilities: np.ndarray = proba[:, 1]
    logger.debug("Generated %d churn probability scores.", len(churn_probabilities))
    return churn_probabilities.astype(np.float64)


def save_model(
    model: LogisticRegression,
    filepath: str | Path = BASELINE_MODEL_PATH,
) -> Path:
    """Serialize and save the trained Logistic Regression model using joblib.

    Args:
        model: Fitted LogisticRegression model.
        filepath: Destination file path. Defaults to BASELINE_MODEL_PATH.

    Returns:
        Path: Absolute path to the saved model artifact.

    Raises:
        TypeError: If model is not a LogisticRegression instance.
        NotFittedError: If model has not been fitted.
    """
    if not isinstance(model, LogisticRegression):
        raise TypeError(
            f"Expected LogisticRegression model, got {type(model).__name__}"
        )

    # Ensure model is fitted before saving
    if not hasattr(model, "coef_"):
        msg = "Refusing to save an unfitted LogisticRegression model artifact."
        logger.error(msg)
        raise NotFittedError(msg)

    dest_path = Path(filepath).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, dest_path)
    logger.info("Successfully serialized model artifact to: %s", dest_path)
    return dest_path


def load_model(filepath: str | Path = BASELINE_MODEL_PATH) -> LogisticRegression:
    """Deserialize and load a trained Logistic Regression model from disk.

    Args:
        filepath: Source file path. Defaults to BASELINE_MODEL_PATH.

    Returns:
        LogisticRegression: Fitted LogisticRegression instance.

    Raises:
        FileNotFoundError: If the model artifact file does not exist.
        TypeError: If loaded object is not a LogisticRegression.
    """
    src_path = Path(filepath).resolve()
    if not src_path.exists():
        msg = f"Model artifact not found at: {src_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    loaded = joblib.load(src_path)
    if not isinstance(loaded, LogisticRegression):
        msg = f"Loaded object is of type {type(loaded).__name__}, expected LogisticRegression."
        logger.error(msg)
        raise TypeError(msg)

    logger.info("Successfully loaded model artifact from: %s", src_path)
    return loaded


def create_baseline_pipeline(
    preprocessor: ColumnTransformer,
    model: LogisticRegression | None = None,
) -> Pipeline:
    """Construct an end-to-end scikit-learn Pipeline linking preprocessor and model.

    Args:
        preprocessor: Configured ColumnTransformer preprocessor.
        model: Optional LogisticRegression instance. If None, a default
            baseline model will be created.

    Returns:
        Pipeline: End-to-end pipeline with steps [('preprocessor', ...), ('classifier', ...)].
    """
    clf = model if model is not None else create_logistic_regression_model()
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )
    logger.info("Constructed end-to-end baseline Pipeline.")
    return pipeline
