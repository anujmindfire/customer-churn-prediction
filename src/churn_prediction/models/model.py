"""Model training, prediction, serialization, and factory utilities for Customer Churn Prediction.

This module provides factory functions to create candidate models (Logistic Regression,
Decision Tree, Random Forest), train and evaluate estimators, extract feature importances,
and serialize/deserialize trained models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator  # type: ignore[import-untyped]
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]
from sklearn.exceptions import NotFittedError  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.tree import DecisionTreeClassifier  # type: ignore[import-untyped]
from sklearn.utils.validation import check_is_fitted  # type: ignore[import-untyped]

from src.churn_prediction.config import (
    BASELINE_MODEL_PATH,
    DEFAULT_DECISION_TREE_PARAMS,
    DEFAULT_LOGISTIC_REGRESSION_PARAMS,
    DEFAULT_RANDOM_FOREST_PARAMS,
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


def create_decision_tree_model(
    random_state: int = RANDOM_STATE,
    **kwargs: Any,
) -> DecisionTreeClassifier:
    """Create and configure a baseline Decision Tree classifier.

    Args:
        random_state: Pseudo-random seed for reproducibility. Defaults to RANDOM_STATE (42).
        **kwargs: Additional keyword arguments passed to DecisionTreeClassifier.

    Returns:
        DecisionTreeClassifier: Unfitted scikit-learn DecisionTreeClassifier instance.
    """
    params = {
        **DEFAULT_DECISION_TREE_PARAMS,
        "random_state": random_state,
        **kwargs,
    }

    logger.info("Initializing DecisionTreeClassifier with parameters: %s", params)
    return DecisionTreeClassifier(**params)


def create_random_forest_model(
    n_estimators: int = 100,
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
    **kwargs: Any,
) -> RandomForestClassifier:
    """Create and configure a baseline Random Forest classifier.

    Args:
        n_estimators: Number of trees in the forest. Defaults to 100.
        random_state: Pseudo-random seed for reproducibility. Defaults to RANDOM_STATE (42).
        n_jobs: Number of CPU workers for parallel fitting. Defaults to -1 (all cores).
        **kwargs: Additional keyword arguments passed to RandomForestClassifier.

    Returns:
        RandomForestClassifier: Unfitted scikit-learn RandomForestClassifier instance.

    Raises:
        ValueError: If n_estimators is non-positive.
    """
    if n_estimators <= 0:
        raise ValueError(f"n_estimators must be greater than 0, got {n_estimators}")

    params = {
        **DEFAULT_RANDOM_FOREST_PARAMS,
        "n_estimators": n_estimators,
        "random_state": random_state,
        "n_jobs": n_jobs,
        **kwargs,
    }

    logger.info("Initializing RandomForestClassifier with parameters: %s", params)
    return RandomForestClassifier(**params)


def create_all_candidate_models(
    random_state: int = RANDOM_STATE,
) -> dict[str, BaseEstimator]:
    """Instantiate all candidate models configured with standard baseline parameters.

    Args:
        random_state: Pseudo-random seed for reproducibility. Defaults to RANDOM_STATE (42).

    Returns:
        dict[str, BaseEstimator]: Mapping of model names to unfitted estimator instances:
            - 'Logistic Regression': Baseline LogisticRegression estimator
            - 'Decision Tree': Unconstrained DecisionTreeClassifier
            - 'Random Forest': Ensemble RandomForestClassifier with 100 estimators
    """
    return {
        "Logistic Regression": create_logistic_regression_model(
            random_state=random_state
        ),
        "Decision Tree": create_decision_tree_model(random_state=random_state),
        "Random Forest": create_random_forest_model(random_state=random_state),
    }


def train_model(
    model: BaseEstimator,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
) -> BaseEstimator:
    """Train an estimator strictly on training data.

    Args:
        model: Unfitted scikit-learn estimator instance.
        X_train: Preprocessed training feature matrix (shape: [n_samples, n_features]).
        y_train: Training binary target labels (shape: [n_samples]).

    Returns:
        BaseEstimator: Fitted estimator instance.

    Raises:
        TypeError: If model does not provide a fit method.
        ValueError: If sample counts do not match or if training data is empty.
    """
    if not hasattr(model, "fit"):
        raise TypeError(
            f"Expected estimator with 'fit' method, got {type(model).__name__}"
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

    model_name = type(model).__name__
    logger.info("Fitting %s model on %d training samples...", model_name, n_samples_X)
    model.fit(X_train, y_train)
    logger.info("Training complete for %s.", model_name)
    return model


def predict_churn(
    model: BaseEstimator,
    X: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """Generate binary churn predictions (0 = Retained, 1 = Churned).

    Uses the standard decision threshold of 0.5.

    Args:
        model: Fitted scikit-learn estimator instance.
        X: Preprocessed feature matrix (shape: [n_samples, n_features]).

    Returns:
        np.ndarray: 1D array of binary predictions (dtype int64).

    Raises:
        TypeError: If model does not have a predict method.
        NotFittedError: If model has not been fitted yet.
        ValueError: If input feature matrix is empty.
    """
    if not hasattr(model, "predict"):
        raise TypeError(
            f"Expected estimator with 'predict' method, got {type(model).__name__}"
        )
    if len(X) == 0:
        raise ValueError("Cannot predict on empty feature matrix.")

    try:
        check_is_fitted(model)
    except NotFittedError as err:
        logger.error("Prediction failed: model has not been fitted.")
        raise NotFittedError(
            f"{type(model).__name__} model must be fitted before predicting."
        ) from err

    predictions: np.ndarray = model.predict(X)
    logger.debug("Generated %d predictions.", len(predictions))
    return predictions.astype(np.int64)


def predict_churn_probabilities(
    model: BaseEstimator,
    X: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """Generate predicted probabilities of customer churn: P(Churn = 1 | X).

    Args:
        model: Fitted scikit-learn estimator instance with predict_proba method.
        X: Preprocessed feature matrix (shape: [n_samples, n_features]).

    Returns:
        np.ndarray: 1D array of churn probabilities between 0.0 and 1.0.

    Raises:
        TypeError: If model does not provide predict_proba method.
        NotFittedError: If model has not been fitted yet.
        ValueError: If input feature matrix is empty.
    """
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            f"Expected estimator with 'predict_proba' method, got {type(model).__name__}"
        )
    if len(X) == 0:
        raise ValueError("Cannot predict probabilities on empty feature matrix.")

    try:
        check_is_fitted(model)
    except NotFittedError as err:
        logger.error("Probability prediction failed: model has not been fitted.")
        raise NotFittedError(
            f"{type(model).__name__} model must be fitted before predicting probabilities."
        ) from err

    proba: np.ndarray = model.predict_proba(X)
    # Extract column corresponding to positive class (Churn = 1)
    churn_probabilities: np.ndarray = proba[:, 1]
    logger.debug("Generated %d churn probability scores.", len(churn_probabilities))
    return churn_probabilities.astype(np.float64)


def get_model_feature_importances(
    model: BaseEstimator,
    feature_names: list[str],
) -> pd.Series:
    """Extract and sort feature importances from a tree-based model.

    Args:
        model: Fitted tree-based model (e.g. DecisionTreeClassifier, RandomForestClassifier).
        feature_names: Ordered list of feature names corresponding to model inputs.

    Returns:
        pd.Series: Feature importances sorted in descending order with feature names as index.

    Raises:
        AttributeError: If model does not expose feature_importances_.
        NotFittedError: If model has not been fitted.
        ValueError: If number of feature names does not match model feature importances length.
    """
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            f"Model {type(model).__name__} does not expose feature_importances_. "
            "Feature importances are typically available for tree-based estimators."
        )

    check_is_fitted(model)

    importances = model.feature_importances_
    if len(importances) != len(feature_names):
        raise ValueError(
            f"Mismatch between number of feature names ({len(feature_names)}) "
            f"and model feature importances ({len(importances)})."
        )

    series = pd.Series(importances, index=feature_names, name="importance").sort_values(
        ascending=False
    )
    return series


def save_model(
    model: BaseEstimator,
    filepath: str | Path = BASELINE_MODEL_PATH,
) -> Path:
    """Serialize and save a trained model estimator using joblib.

    Args:
        model: Fitted scikit-learn estimator.
        filepath: Destination file path. Defaults to BASELINE_MODEL_PATH.

    Returns:
        Path: Absolute path to the saved model artifact.

    Raises:
        TypeError: If model is not a valid scikit-learn estimator.
        NotFittedError: If model has not been fitted.
    """
    if not hasattr(model, "predict"):
        raise TypeError(
            f"Expected estimator with predict method, got {type(model).__name__}"
        )

    try:
        check_is_fitted(model)
    except NotFittedError as err:
        msg = f"Refusing to save an unfitted {type(model).__name__} model artifact."
        logger.error(msg)
        raise NotFittedError(msg) from err

    dest_path = Path(filepath).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, dest_path)
    logger.info("Successfully serialized model artifact to: %s", dest_path)
    return dest_path


def load_model(
    filepath: str | Path = BASELINE_MODEL_PATH,
    expected_type: type | None = None,
) -> Any:
    """Deserialize and load a trained model from disk.

    Args:
        filepath: Source file path. Defaults to BASELINE_MODEL_PATH.
        expected_type: Optional expected model class type to validate against.

    Returns:
        Any: Fitted scikit-learn estimator instance.

    Raises:
        FileNotFoundError: If the model artifact file does not exist.
        TypeError: If loaded object does not match expected_type.
    """
    src_path = Path(filepath).resolve()
    if not src_path.exists():
        msg = f"Model artifact not found at: {src_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    loaded = joblib.load(src_path)
    if expected_type is not None and not isinstance(loaded, expected_type):
        msg = f"Loaded object is of type {type(loaded).__name__}, expected {expected_type.__name__}."
        logger.error(msg)
        raise TypeError(msg)

    logger.info("Successfully loaded model artifact from: %s", src_path)
    return loaded


def create_baseline_pipeline(
    preprocessor: ColumnTransformer,
    model: BaseEstimator | None = None,
) -> Pipeline:
    """Construct an end-to-end scikit-learn Pipeline linking preprocessor and model.

    Args:
        preprocessor: Configured ColumnTransformer preprocessor.
        model: Optional estimator instance. If None, a default
            baseline LogisticRegression model will be created.

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
    logger.info("Constructed end-to-end baseline Pipeline with %s.", type(clf).__name__)
    return pipeline
