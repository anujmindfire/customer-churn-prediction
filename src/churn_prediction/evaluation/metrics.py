"""Model evaluation metrics utilities for binary churn classification.

This module provides reusable functions to compute standard classification metrics
(Accuracy, Precision, Recall, F1, ROC-AUC), extract confusion matrix components,
and generate formatted reports and comparison summaries.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def calculate_classification_metrics(
    y_true: pd.Series | np.ndarray | list[int],
    y_pred: pd.Series | np.ndarray | list[int],
    y_proba: pd.Series | np.ndarray | list[float] | None = None,
) -> dict[str, float]:
    """Calculate core classification metrics for binary classification.

    Computes Accuracy, Precision, Recall, F1 Score, and ROC-AUC (if probability scores
    are provided). Safely handles edge cases like zero division when a model predicts
    only one class.

    Args:
        y_true: Ground truth binary target labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_proba: Optional predicted probabilities of the positive class (Churn = 1).

    Returns:
        dict[str, float]: Dictionary containing computed metrics rounded to 4 decimals.

    Raises:
        ValueError: If array lengths mismatch or if arrays are empty.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if len(y_true_arr) == 0 or len(y_pred_arr) == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")

    if len(y_true_arr) != len(y_pred_arr):
        msg = f"Length mismatch: y_true has {len(y_true_arr)} items, y_pred has {len(y_pred_arr)} items."
        logger.error(msg)
        raise ValueError(msg)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    metrics: dict[str, float] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }

    if y_proba is not None:
        y_proba_arr = np.asarray(y_proba)
        if len(y_proba_arr) != len(y_true_arr):
            raise ValueError(
                f"Length mismatch: y_true ({len(y_true_arr)}) vs y_proba ({len(y_proba_arr)})."
            )

        unique_classes = np.unique(y_true_arr)
        if len(unique_classes) < 2:
            logger.warning(
                "ROC-AUC cannot be calculated: y_true contains only one class (%s).",
                unique_classes,
            )
            metrics["roc_auc"] = float("nan")
        else:
            roc_auc = float(roc_auc_score(y_true_arr, y_proba_arr))
            metrics["roc_auc"] = round(roc_auc, 4)

    logger.info("Classification metrics calculated: %s", metrics)
    return metrics


def compute_confusion_matrix(
    y_true: pd.Series | np.ndarray | list[int],
    y_pred: pd.Series | np.ndarray | list[int],
) -> np.ndarray:
    """Compute 2x2 confusion matrix for binary classification.

    Matrix layout:
        [[TN, FP],
         [FN, TP]]

    Args:
        y_true: Ground truth binary target labels.
        y_pred: Predicted binary labels.

    Returns:
        np.ndarray: 2x2 integer confusion matrix.

    Raises:
        ValueError: If input arrays are empty or lengths do not match.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if len(y_true_arr) == 0 or len(y_pred_arr) == 0:
        raise ValueError("Cannot compute confusion matrix on empty arrays.")
    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError("Length mismatch between y_true and y_pred.")

    cm: np.ndarray = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    logger.debug("Computed confusion matrix:\n%s", cm)
    return cm


def get_confusion_matrix_breakdown(
    y_true: pd.Series | np.ndarray | list[int],
    y_pred: pd.Series | np.ndarray | list[int],
) -> dict[str, int]:
    """Extract individual True Negative, False Positive, False Negative, and True Positive counts.

    Args:
        y_true: Ground truth binary target labels.
        y_pred: Predicted binary labels.

    Returns:
        dict[str, int]: Breakdown with keys 'tn', 'fp', 'fn', 'tp'.
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    breakdown = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    logger.info(
        "Confusion matrix breakdown: TN=%d, FP=%d, FN=%d, TP=%d.",
        tn,
        fp,
        fn,
        tp,
    )
    return breakdown


def get_classification_report(
    y_true: pd.Series | np.ndarray | list[int],
    y_pred: pd.Series | np.ndarray | list[int],
    target_names: list[str] | None = None,
    output_dict: bool = False,
) -> str | dict[str, Any]:
    """Generate a comprehensive precision, recall, f1-score report per class.

    Args:
        y_true: Ground truth binary target labels.
        y_pred: Predicted binary labels.
        target_names: Display names for the classes. Defaults to ['No Churn', 'Churn'].
        output_dict: If True, returns report as nested dictionary. Defaults to False.

    Returns:
        str | dict[str, Any]: Text report or dictionary of metrics.
    """
    names = target_names if target_names is not None else ["No Churn", "Churn"]
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    report = classification_report(
        y_true_arr,
        y_pred_arr,
        target_names=names,
        zero_division=0,
        output_dict=output_dict,
    )
    return report


def format_metrics_summary(
    metrics: dict[str, float],
    model_name: str = "Logistic Regression",
) -> pd.DataFrame:
    """Format evaluation metrics into a clean single-row pandas DataFrame.

    Args:
        metrics: Dictionary returned by calculate_classification_metrics.
        model_name: Name of the evaluated model. Defaults to "Logistic Regression".

    Returns:
        pd.DataFrame: Formatted DataFrame suitable for table display and model comparison.
    """
    row = {"Model": model_name}
    for k, v in metrics.items():
        # Clean column name formatting: 'roc_auc' -> 'ROC-AUC', 'f1' -> 'F1 Score'
        col_name = "ROC-AUC" if k == "roc_auc" else k.capitalize()
        if k == "f1":
            col_name = "F1 Score"
        row[col_name] = v

    return pd.DataFrame([row])
