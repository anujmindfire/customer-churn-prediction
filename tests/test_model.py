"""Unit tests for Logistic Regression baseline model creation, training, and persistence."""

from pathlib import Path

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.churn_prediction.models.model import (
    create_baseline_pipeline,
    create_logistic_regression_model,
    load_model,
    predict_churn,
    predict_churn_probabilities,
    save_model,
    train_model,
)


@pytest.fixture
def synthetic_data() -> tuple[np.ndarray, np.ndarray]:
    """Provide a small synthetic linearly separable binary dataset."""
    np.random.seed(42)
    n = 60
    # Two features
    X = np.random.randn(n, 4)
    # Simple threshold target
    y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
    return X, y


def test_create_logistic_regression_model_defaults():
    """Verify factory returns an unfitted model with default configuration."""
    model = create_logistic_regression_model()
    assert isinstance(model, LogisticRegression)
    assert model.max_iter == 1000
    assert model.random_state == 42
    assert not hasattr(model, "coef_")


def test_create_logistic_regression_model_custom():
    """Verify custom hyperparameters are respected."""
    model = create_logistic_regression_model(random_state=123, max_iter=500, C=0.5)
    assert model.random_state == 123
    assert model.max_iter == 500
    assert model.C == 0.5


def test_create_logistic_regression_invalid_max_iter():
    """Verify non-positive max_iter raises ValueError."""
    with pytest.raises(ValueError, match="max_iter must be greater than 0"):
        create_logistic_regression_model(max_iter=0)


def test_train_model_synthetic(synthetic_data: tuple[np.ndarray, np.ndarray]):
    """Verify model can be trained on synthetic data."""
    X, y = synthetic_data
    model = create_logistic_regression_model()
    fitted_model = train_model(model, X, y)

    assert fitted_model is model
    assert hasattr(fitted_model, "coef_")
    assert fitted_model.coef_.shape == (1, 4)
    assert hasattr(fitted_model, "intercept_")


def test_train_model_dimension_mismatch():
    """Verify length mismatch between features and labels raises ValueError."""
    X = np.zeros((10, 2))
    y = np.zeros(8)
    model = create_logistic_regression_model()
    with pytest.raises(ValueError, match="does not match label count"):
        train_model(model, X, y)


def test_train_model_empty_raises():
    """Verify empty input raises ValueError."""
    X = np.empty((0, 2))
    y = np.empty(0)
    model = create_logistic_regression_model()
    with pytest.raises(ValueError, match="empty training data"):
        train_model(model, X, y)


def test_predict_churn_unfitted_raises():
    """Verify predicting with an unfitted model raises NotFittedError."""
    model = create_logistic_regression_model()
    X = np.ones((5, 4))
    with pytest.raises(NotFittedError):
        predict_churn(model, X)


def test_predict_churn_values_and_shape(synthetic_data: tuple[np.ndarray, np.ndarray]):
    """Verify predictions have valid binary values and match sample length."""
    X, y = synthetic_data
    model = train_model(create_logistic_regression_model(), X, y)

    preds = predict_churn(model, X)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X)
    assert set(np.unique(preds)).issubset({0, 1})


def test_predict_churn_probabilities_bounds(
    synthetic_data: tuple[np.ndarray, np.ndarray],
):
    """Verify probability predictions are bounded in [0.0, 1.0]."""
    X, y = synthetic_data
    model = train_model(create_logistic_regression_model(), X, y)

    probs = predict_churn_probabilities(model, X)
    assert isinstance(probs, np.ndarray)
    assert len(probs) == len(X)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_predict_empty_raises(synthetic_data: tuple[np.ndarray, np.ndarray]):
    """Verify predicting on empty array raises ValueError."""
    X, y = synthetic_data
    model = train_model(create_logistic_regression_model(), X, y)
    with pytest.raises(ValueError, match="empty feature matrix"):
        predict_churn(model, np.empty((0, 4)))


def test_save_and_load_model(
    tmp_path: Path, synthetic_data: tuple[np.ndarray, np.ndarray]
):
    """Verify serializing and deserializing fitted model preserves predictions."""
    X, y = synthetic_data
    model = train_model(create_logistic_regression_model(), X, y)

    save_path = tmp_path / "baseline.joblib"
    saved = save_model(model, save_path)
    assert saved.exists()

    loaded = load_model(saved)
    assert isinstance(loaded, LogisticRegression)

    orig_preds = predict_churn(model, X)
    loaded_preds = predict_churn(loaded, X)
    np.testing.assert_array_equal(orig_preds, loaded_preds)

    orig_probs = predict_churn_probabilities(model, X)
    loaded_probs = predict_churn_probabilities(loaded, X)
    np.testing.assert_allclose(orig_probs, loaded_probs)


def test_save_unfitted_model_raises(tmp_path: Path):
    """Verify attempting to save an unfitted model raises NotFittedError."""
    model = create_logistic_regression_model()
    with pytest.raises(NotFittedError):
        save_model(model, tmp_path / "model.joblib")


def test_load_nonexistent_model_raises():
    """Verify loading non-existent model raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_model("non_existent_file.joblib")


def test_create_baseline_pipeline():
    """Verify baseline pipeline combines preprocessor dummy and model."""
    from sklearn.preprocessing import StandardScaler

    pipeline = create_baseline_pipeline(StandardScaler())  # type: ignore[arg-type]
    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps
    assert isinstance(pipeline.named_steps["classifier"], LogisticRegression)
