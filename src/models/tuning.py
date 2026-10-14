"""Hyperparameter tuning and decision threshold optimization utilities.

This module provides functions to perform cross-validated hyperparameter optimization
with class weighting, find optimal decision thresholds on out-of-fold training probabilities,
and apply thresholded decision rules to avoid data leakage.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator  # type: ignore[import-untyped]
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (  # type: ignore[import-untyped]
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
)

from src.config import RANDOM_STATE

logger = logging.getLogger(__name__)


def tune_logistic_regression(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    param_grid: dict[str, list[Any]] | None = None,
    cv_splits: int = 5,
    scoring: str = "f1",
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
) -> GridSearchCV:
    """Perform cross-validated hyperparameter tuning for Logistic Regression.

    Explores regularization strengths (C) and class weighting options to optimize
    classification performance on imbalanced churn data strictly using training data.

    Args:
        X_train: Preprocessed training feature matrix.
        y_train: Binary training target labels.
        param_grid: Custom parameter grid. If None, uses default search space.
        cv_splits: Number of stratified cross-validation folds. Defaults to 5.
        scoring: Metric to optimize. Defaults to 'f1'.
        random_state: Random state for reproducibility. Defaults to RANDOM_STATE.
        n_jobs: Number of parallel jobs. Defaults to -1.

    Returns:
        GridSearchCV: Fitted GridSearchCV object containing best estimator and results.

    Raises:
        ValueError: If training data is empty or dimensions mismatch.
    """
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Cannot perform tuning on empty training data.")
    if len(X_train) != len(y_train):
        raise ValueError("Mismatch between X_train and y_train sample lengths.")

    grid = (
        param_grid
        if param_grid is not None
        else {
            "C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            "class_weight": [None, "balanced", {0: 1.0, 1: 1.5}, {0: 1.0, 1: 2.0}],
            "solver": ["lbfgs"],
            "max_iter": [1000],
            "random_state": [random_state],
        }
    )

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    estimator = LogisticRegression()

    logger.info(
        "Starting Logistic Regression GridSearchCV with %d CV splits optimizing '%s'...",
        cv_splits,
        scoring,
    )
    search = GridSearchCV(
        estimator=estimator,
        param_grid=grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(X_train, y_train)

    logger.info(
        "Tuning complete. Best params: %s (Best CV %s: %.4f)",
        search.best_params_,
        scoring,
        search.best_score_,
    )
    return search


def tune_random_forest(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    param_grid: dict[str, list[Any]] | None = None,
    cv_splits: int = 5,
    scoring: str = "f1",
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
) -> GridSearchCV:
    """Perform cross-validated hyperparameter tuning for Random Forest.

    Regularizes tree depth and leaf size to control variance and applies class weighting
    to improve minority class sensitivity strictly on training data.

    Args:
        X_train: Preprocessed training feature matrix.
        y_train: Binary training target labels.
        param_grid: Custom parameter grid. If None, uses default search space.
        cv_splits: Number of stratified cross-validation folds. Defaults to 5.
        scoring: Metric to optimize. Defaults to 'f1'.
        random_state: Random state for reproducibility. Defaults to RANDOM_STATE.
        n_jobs: Number of parallel jobs. Defaults to -1.

    Returns:
        GridSearchCV: Fitted GridSearchCV object containing best estimator and results.

    Raises:
        ValueError: If training data is empty or dimensions mismatch.
    """
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Cannot perform tuning on empty training data.")
    if len(X_train) != len(y_train):
        raise ValueError("Mismatch between X_train and y_train sample lengths.")

    grid = (
        param_grid
        if param_grid is not None
        else {
            "n_estimators": [100, 200],
            "max_depth": [6, 8, 10, 12],
            "min_samples_split": [5, 10, 20],
            "min_samples_leaf": [2, 4, 8],
            "class_weight": [None, "balanced"],
            "random_state": [random_state],
        }
    )

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    estimator = RandomForestClassifier()

    logger.info(
        "Starting Random Forest GridSearchCV with %d CV splits optimizing '%s'...",
        cv_splits,
        scoring,
    )
    search = GridSearchCV(
        estimator=estimator,
        param_grid=grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(X_train, y_train)

    logger.info(
        "Random Forest tuning complete. Best params: %s (Best CV %s: %.4f)",
        search.best_params_,
        scoring,
        search.best_score_,
    )
    return search


def find_optimal_threshold(
    y_true: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
    metric: str = "f1",
    num_thresholds: int = 81,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find decision threshold that maximizes a specified classification metric.

    Evaluates candidate thresholds in [0.1, 0.9] and computes corresponding
    precision, recall, and F1 scores.

    Args:
        y_true: Ground truth binary target labels (0 or 1).
        y_proba: Predicted probabilities for the positive class (1).
        metric: Target optimization metric ('f1', 'recall', or 'precision'). Defaults to 'f1'.
        num_thresholds: Number of evenly spaced thresholds to evaluate. Defaults to 81.

    Returns:
        tuple containing:
            - optimal_threshold (float): Threshold yielding highest metric score.
            - best_score (float): Highest score achieved.
            - thresholds (np.ndarray): Array of tested thresholds.
            - f1_scores (np.ndarray): F1 scores across thresholds.
            - precisions (np.ndarray): Precision scores across thresholds.
            - recalls (np.ndarray): Recall scores across thresholds.

    Raises:
        ValueError: If input arrays are empty or lengths do not match.
    """
    y_true_arr = np.asarray(y_true)
    y_proba_arr = np.asarray(y_proba)

    if len(y_true_arr) == 0 or len(y_proba_arr) == 0:
        raise ValueError("Cannot search thresholds on empty arrays.")
    if len(y_true_arr) != len(y_proba_arr):
        raise ValueError("Length mismatch between y_true and y_proba.")

    thresholds = np.linspace(0.1, 0.9, num_thresholds)
    f1_list: list[float] = []
    prec_list: list[float] = []
    rec_list: list[float] = []

    best_threshold = 0.5
    best_score = -1.0

    for th in thresholds:
        preds = (y_proba_arr >= th).astype(int)
        f1 = float(f1_score(y_true_arr, preds, zero_division=0))
        prec = float(precision_score(y_true_arr, preds, zero_division=0))
        rec = float(recall_score(y_true_arr, preds, zero_division=0))

        f1_list.append(f1)
        prec_list.append(prec)
        rec_list.append(rec)

        target_score = f1 if metric == "f1" else (rec if metric == "recall" else prec)
        if target_score > best_score:
            best_score = target_score
            best_threshold = float(th)

    logger.info(
        "Optimal threshold for '%s': %.3f (Best score: %.4f)",
        metric,
        best_threshold,
        best_score,
    )
    return (
        round(best_threshold, 3),
        round(best_score, 4),
        thresholds,
        np.array(f1_list),
        np.array(prec_list),
        np.array(rec_list),
    )


def find_optimal_threshold_cv(
    estimator: BaseEstimator,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    cv_splits: int = 5,
    metric: str = "f1",
    random_state: int = RANDOM_STATE,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find optimal decision threshold using out-of-fold training predictions to prevent data leakage.

    Args:
        estimator: Unfitted or fitted scikit-learn estimator with predict_proba method.
        X_train: Preprocessed training feature matrix.
        y_train: Training binary labels.
        cv_splits: Number of CV folds for out-of-fold probability estimation. Defaults to 5.
        metric: Optimization metric. Defaults to 'f1'.
        random_state: Random state for CV split reproducibility. Defaults to RANDOM_STATE.

    Returns:
        tuple containing (optimal_threshold, best_score, thresholds, f1s, precisions, recalls).
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    logger.info("Generating out-of-fold probability predictions on training data...")
    oof_proba: np.ndarray = cross_val_predict(
        estimator,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    return find_optimal_threshold(
        y_true=y_train,
        y_proba=oof_proba,
        metric=metric,
    )


def predict_with_threshold(
    y_proba: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Generate binary predictions by applying a custom probability threshold.

    Points with P(Churn = 1 | X) >= threshold are classified as 1, otherwise 0.

    Args:
        y_proba: 1D array or Series of predicted churn probabilities.
        threshold: Classification cutoff threshold. Defaults to 0.5. Must be in (0.0, 1.0).

    Returns:
        np.ndarray: Binary predictions (0 or 1).

    Raises:
        ValueError: If threshold is not within (0.0, 1.0) or if y_proba is empty.
    """
    if not (0.0 < threshold < 1.0):
        raise ValueError(f"Threshold must be strictly between 0 and 1, got {threshold}")

    proba_arr = np.asarray(y_proba)
    if len(proba_arr) == 0:
        raise ValueError("Cannot apply threshold to empty probability array.")

    return (proba_arr >= threshold).astype(np.int64)
