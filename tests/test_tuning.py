"""Unit tests for model tuning, threshold optimization, and associated plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from src.evaluation.plots import plot_threshold_tuning_curve
from src.models.tuning import (
    find_optimal_threshold,
    find_optimal_threshold_cv,
    predict_with_threshold,
    tune_logistic_regression,
    tune_random_forest,
)


@pytest.fixture
def synthetic_tuning_data() -> tuple[np.ndarray, np.ndarray]:
    """Provide a small synthetic binary classification dataset."""
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 4)
    # Slight imbalance: ~30% positive
    y = ((X[:, 0] + 0.5 * X[:, 1]) > 0.5).astype(int)
    return X, y


def test_tune_logistic_regression_synthetic(
    synthetic_tuning_data: tuple[np.ndarray, np.ndarray],
):
    """Verify tune_logistic_regression returns fitted GridSearchCV with valid best params."""
    X, y = synthetic_tuning_data
    param_grid = {
        "C": [0.1, 1.0],
        "class_weight": [None, "balanced"],
        "max_iter": [500],
    }

    search = tune_logistic_regression(
        X_train=X,
        y_train=y,
        param_grid=param_grid,
        cv_splits=2,
    )

    assert isinstance(search, GridSearchCV)
    assert hasattr(search, "best_estimator_")
    assert isinstance(search.best_estimator_, LogisticRegression)
    assert search.best_params_["C"] in [0.1, 1.0]


def test_tune_logistic_regression_empty_raises():
    """Verify empty input raises ValueError."""
    with pytest.raises(ValueError, match="empty training data"):
        tune_logistic_regression(X_train=np.empty((0, 4)), y_train=np.empty(0))


def test_tune_random_forest_synthetic(
    synthetic_tuning_data: tuple[np.ndarray, np.ndarray],
):
    """Verify tune_random_forest returns fitted GridSearchCV."""
    X, y = synthetic_tuning_data
    param_grid = {
        "n_estimators": [10],
        "max_depth": [3],
        "min_samples_leaf": [2],
    }

    search = tune_random_forest(
        X_train=X,
        y_train=y,
        param_grid=param_grid,
        cv_splits=2,
    )

    assert isinstance(search, GridSearchCV)
    assert hasattr(search, "best_estimator_")
    assert search.best_params_["max_depth"] == 3


def test_find_optimal_threshold():
    """Verify find_optimal_threshold returns peak threshold and arrays."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    best_th, best_f1, ths, f1s, precs, recs = find_optimal_threshold(
        y_true, y_proba, metric="f1"
    )

    assert 0.4 <= best_th <= 0.6
    assert best_f1 == 1.0
    assert len(ths) == len(f1s) == len(precs) == len(recs)


def test_find_optimal_threshold_dimension_mismatch():
    """Verify length mismatch raises ValueError."""
    with pytest.raises(ValueError, match="Length mismatch"):
        find_optimal_threshold([0, 1], [0.5])


def test_find_optimal_threshold_cv(
    synthetic_tuning_data: tuple[np.ndarray, np.ndarray],
):
    """Verify find_optimal_threshold_cv finds threshold on out-of-fold probabilities."""
    X, y = synthetic_tuning_data
    clf = LogisticRegression(max_iter=500, random_state=42)

    best_th, best_f1, ths, f1s, _precs, _recs = find_optimal_threshold_cv(
        estimator=clf,
        X_train=X,
        y_train=y,
        cv_splits=2,
    )

    assert 0.1 <= best_th <= 0.9
    assert 0.0 <= best_f1 <= 1.0
    assert len(ths) == len(f1s)


def test_predict_with_threshold():
    """Verify predict_with_threshold applies decision cutoff."""
    y_proba = np.array([0.2, 0.45, 0.55, 0.8])
    preds_05 = predict_with_threshold(y_proba, threshold=0.5)
    np.testing.assert_array_equal(preds_05, [0, 0, 1, 1])

    preds_04 = predict_with_threshold(y_proba, threshold=0.4)
    np.testing.assert_array_equal(preds_04, [0, 1, 1, 1])


def test_predict_with_threshold_invalid():
    """Verify invalid thresholds raise ValueError."""
    y_proba = np.array([0.2, 0.7])
    with pytest.raises(ValueError, match="Threshold must be strictly between 0 and 1"):
        predict_with_threshold(y_proba, threshold=1.0)
    with pytest.raises(ValueError, match="Threshold must be strictly between 0 and 1"):
        predict_with_threshold(y_proba, threshold=0.0)


def test_plot_threshold_tuning_curve(tmp_path: Path):
    """Verify plot_threshold_tuning_curve creates and saves plot."""
    thresholds = np.linspace(0.1, 0.9, 10)
    f1s = np.linspace(0.5, 0.7, 10)
    precs = np.linspace(0.4, 0.8, 10)
    recs = np.linspace(0.9, 0.3, 10)

    save_path = tmp_path / "thresh_curve.png"
    fig = plot_threshold_tuning_curve(
        thresholds=thresholds,
        f1_scores=f1s,
        precisions=precs,
        recalls=recs,
        optimal_threshold=0.5,
        optimal_f1=0.7,
        save_path=save_path,
    )

    assert isinstance(fig, plt.Figure)
    assert save_path.exists()
    assert save_path.stat().st_size > 0
    plt.close(fig)
