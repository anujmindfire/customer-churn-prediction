"""Unit tests for classification metrics, confusion matrix, and evaluation visualizations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.evaluation.metrics import (
    calculate_classification_metrics,
    compute_confusion_matrix,
    format_metrics_summary,
    get_classification_report,
    get_confusion_matrix_breakdown,
)
from src.evaluation.plots import plot_confusion_matrix


def test_calculate_classification_metrics_perfect():
    """Verify metrics for perfect predictions."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    y_proba = [0.1, 0.2, 0.8, 0.9]

    metrics = calculate_classification_metrics(y_true, y_pred, y_proba)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_calculate_classification_metrics_realistic():
    """Verify metrics calculation on realistic predictions."""
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 0, 1, 1]
    y_proba = [0.1, 0.2, 0.6, 0.4, 0.8, 0.9]

    metrics = calculate_classification_metrics(y_true, y_pred, y_proba)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_calculate_classification_metrics_without_proba():
    """Verify metrics calculation when probability is not supplied."""
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]

    metrics = calculate_classification_metrics(y_true, y_pred)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" not in metrics


def test_calculate_classification_metrics_zero_division():
    """Verify zero division is safely handled when model predicts all zeros."""
    y_true = [1, 1, 1]
    y_pred = [0, 0, 0]  # No positive predictions

    metrics = calculate_classification_metrics(y_true, y_pred)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_calculate_classification_metrics_single_class_roc():
    """Verify single class ground truth does not crash ROC-AUC calculation."""
    y_true = [1, 1, 1]  # Only class 1 present
    y_pred = [1, 1, 1]
    y_proba = [0.8, 0.9, 0.95]

    metrics = calculate_classification_metrics(y_true, y_pred, y_proba)
    assert np.isnan(metrics["roc_auc"])


def test_calculate_classification_metrics_mismatch():
    """Verify length mismatch raises ValueError."""
    with pytest.raises(ValueError, match="Length mismatch"):
        calculate_classification_metrics([0, 1], [0, 1, 0])


def test_calculate_classification_metrics_empty():
    """Verify empty input raises ValueError."""
    with pytest.raises(ValueError, match="Cannot compute metrics on empty arrays"):
        calculate_classification_metrics([], [])


def test_compute_confusion_matrix_shape_and_values():
    """Verify 2x2 shape and proper quadrant mapping."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]

    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm.shape == (2, 2)
    assert cm[0, 0] == 1  # TN
    assert cm[0, 1] == 1  # FP
    assert cm[1, 0] == 1  # FN
    assert cm[1, 1] == 1  # TP


def test_get_confusion_matrix_breakdown():
    """Verify dictionary extraction of TN, FP, FN, TP."""
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 1]

    breakdown = get_confusion_matrix_breakdown(y_true, y_pred)
    assert breakdown["tn"] == 2
    assert breakdown["fp"] == 0
    assert breakdown["fn"] == 1
    assert breakdown["tp"] == 2


def test_get_classification_report():
    """Verify string and dictionary classification reports."""
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]

    report_str = get_classification_report(y_true, y_pred, output_dict=False)
    assert isinstance(report_str, str)
    assert "No Churn" in report_str
    assert "Churn" in report_str

    report_dict = get_classification_report(y_true, y_pred, output_dict=True)
    assert isinstance(report_dict, dict)
    assert "No Churn" in report_dict
    assert "accuracy" in report_dict


def test_format_metrics_summary():
    """Verify formatting metrics dictionary into DataFrame."""
    metrics = {
        "accuracy": 0.85,
        "precision": 0.75,
        "recall": 0.65,
        "f1": 0.70,
        "roc_auc": 0.88,
    }
    df = format_metrics_summary(metrics, model_name="Baseline Model")

    assert len(df) == 1
    assert df.loc[0, "Model"] == "Baseline Model"
    assert df.loc[0, "Accuracy"] == 0.85
    assert df.loc[0, "F1 Score"] == 0.70
    assert df.loc[0, "ROC-AUC"] == 0.88


def test_plot_confusion_matrix_and_save(tmp_path: Path):
    """Verify confusion matrix plotting and disk export."""
    cm = np.array([[685, 91], [114, 166]])
    out_file = tmp_path / "cm.png"

    fig = plot_confusion_matrix(cm, save_path=out_file)
    assert isinstance(fig, plt.Figure)
    assert out_file.exists()
    plt.close(fig)


def test_plot_confusion_matrix_invalid_shape():
    """Verify non-2x2 matrix raises ValueError."""
    cm = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="Expected 2x2 confusion matrix"):
        plot_confusion_matrix(cm)
