"""Unit tests for multi-model training, comparison, feature importance, and plotting."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.evaluation.metrics import (
    compare_models,
)
from src.evaluation.plots import (
    plot_feature_importance,
    plot_model_comparison,
)
from src.models.model import (
    create_all_candidate_models,
    create_decision_tree_model,
    create_logistic_regression_model,
    create_random_forest_model,
    get_model_feature_importances,
    load_model,
    predict_churn,
    predict_churn_probabilities,
    save_model,
    train_model,
)


@pytest.fixture
def synthetic_data() -> tuple[np.ndarray, np.ndarray]:
    """Provide synthetic binary classification dataset."""
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 5)
    y = ((X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2]) > 0).astype(int)
    return X, y


def test_create_decision_tree_model_defaults():
    """Verify Decision Tree factory defaults."""
    model = create_decision_tree_model()
    assert isinstance(model, DecisionTreeClassifier)
    assert model.random_state == 42
    assert not hasattr(model, "tree_")


def test_create_decision_tree_model_custom():
    """Verify custom hyperparameters for Decision Tree."""
    model = create_decision_tree_model(
        random_state=99, max_depth=5, min_samples_leaf=10
    )
    assert model.random_state == 99
    assert model.max_depth == 5
    assert model.min_samples_leaf == 10


def test_create_random_forest_model_defaults():
    """Verify Random Forest factory defaults."""
    model = create_random_forest_model()
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 100
    assert model.random_state == 42
    assert model.n_jobs == -1
    assert not hasattr(model, "estimators_")


def test_create_random_forest_model_invalid_n_estimators():
    """Verify non-positive n_estimators raises ValueError."""
    with pytest.raises(ValueError, match="n_estimators must be greater than 0"):
        create_random_forest_model(n_estimators=0)


def test_create_all_candidate_models():
    """Verify candidate models dictionary contains all 3 models with correct types."""
    models = create_all_candidate_models()
    assert set(models.keys()) == {
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
    }
    assert isinstance(models["Logistic Regression"], LogisticRegression)
    assert isinstance(models["Decision Tree"], DecisionTreeClassifier)
    assert isinstance(models["Random Forest"], RandomForestClassifier)


def test_train_and_predict_decision_tree(synthetic_data: tuple[np.ndarray, np.ndarray]):
    """Verify training and predicting with Decision Tree."""
    X, y = synthetic_data
    dt = create_decision_tree_model()
    fitted = train_model(dt, X, y)
    assert hasattr(fitted, "tree_")

    preds = predict_churn(fitted, X)
    assert len(preds) == len(X)
    assert set(np.unique(preds)).issubset({0, 1})

    probs = predict_churn_probabilities(fitted, X)
    assert len(probs) == len(X)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_train_and_predict_random_forest(synthetic_data: tuple[np.ndarray, np.ndarray]):
    """Verify training and predicting with Random Forest."""
    X, y = synthetic_data
    rf = create_random_forest_model(n_estimators=10)
    fitted = train_model(rf, X, y)
    assert hasattr(fitted, "estimators_")

    preds = predict_churn(fitted, X)
    assert len(preds) == len(X)
    assert set(np.unique(preds)).issubset({0, 1})

    probs = predict_churn_probabilities(fitted, X)
    assert len(probs) == len(X)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_get_model_feature_importances(synthetic_data: tuple[np.ndarray, np.ndarray]):
    """Verify feature importances are extracted and sorted descending."""
    X, y = synthetic_data
    feature_names = ["feat_A", "feat_B", "feat_C", "feat_D", "feat_E"]
    rf = train_model(create_random_forest_model(n_estimators=10), X, y)

    importances = get_model_feature_importances(rf, feature_names)
    assert isinstance(importances, pd.Series)
    assert len(importances) == 5
    assert list(importances.index) == list(
        importances.sort_values(ascending=False).index
    )
    assert pytest.approx(float(importances.sum()), rel=1e-3) == 1.0


def test_get_model_feature_importances_dimension_mismatch(
    synthetic_data: tuple[np.ndarray, np.ndarray],
):
    """Verify feature names count mismatch raises ValueError."""
    X, y = synthetic_data
    rf = train_model(create_random_forest_model(n_estimators=10), X, y)
    with pytest.raises(ValueError, match="Mismatch between number of feature names"):
        get_model_feature_importances(rf, ["f1", "f2"])


def test_get_model_feature_importances_unsupported_model(
    synthetic_data: tuple[np.ndarray, np.ndarray],
):
    """Verify extracting feature importances from Logistic Regression raises AttributeError."""
    X, y = synthetic_data
    lr = train_model(create_logistic_regression_model(), X, y)
    with pytest.raises(AttributeError, match="does not expose feature_importances_"):
        get_model_feature_importances(lr, ["f1", "f2", "f3", "f4", "f5"])


def test_save_and_load_candidate_models(
    tmp_path: Path, synthetic_data: tuple[np.ndarray, np.ndarray]
):
    """Verify serialization and deserialization for DT and RF models."""
    X, y = synthetic_data
    dt = train_model(create_decision_tree_model(), X, y)
    rf = train_model(create_random_forest_model(n_estimators=10), X, y)

    dt_path = tmp_path / "dt.joblib"
    rf_path = tmp_path / "rf.joblib"

    save_model(dt, dt_path)
    save_model(rf, rf_path)

    loaded_dt = load_model(dt_path, expected_type=DecisionTreeClassifier)
    loaded_rf = load_model(rf_path, expected_type=RandomForestClassifier)

    assert isinstance(loaded_dt, DecisionTreeClassifier)
    assert isinstance(loaded_rf, RandomForestClassifier)

    np.testing.assert_array_equal(predict_churn(dt, X), predict_churn(loaded_dt, X))
    np.testing.assert_array_equal(predict_churn(rf, X), predict_churn(loaded_rf, X))


def test_load_model_expected_type_mismatch(
    tmp_path: Path, synthetic_data: tuple[np.ndarray, np.ndarray]
):
    """Verify TypeError raised when loaded model doesn't match expected_type."""
    X, y = synthetic_data
    dt = train_model(create_decision_tree_model(), X, y)
    dt_path = tmp_path / "dt.joblib"
    save_model(dt, dt_path)

    with pytest.raises(TypeError, match="expected RandomForestClassifier"):
        load_model(dt_path, expected_type=RandomForestClassifier)


def test_compare_models_consolidation():
    """Verify compare_models formats metric dictionaries into expected DataFrame."""
    metrics_dict = {
        "Logistic Regression": {
            "accuracy": 0.8059,
            "precision": 0.6459,
            "recall": 0.5929,
            "f1": 0.6182,
            "roc_auc": 0.8454,
        },
        "Decision Tree": {
            "accuracy": 0.7178,
            "precision": 0.4690,
            "recall": 0.4857,
            "f1": 0.4772,
            "roc_auc": 0.6430,
        },
        "Random Forest": {
            "accuracy": 0.7756,
            "precision": 0.5939,
            "recall": 0.4857,
            "f1": 0.5344,
            "roc_auc": 0.8131,
        },
    }

    df = compare_models(metrics_dict)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df["Model"]) == [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
    ]
    assert "F1 Score" in df.columns
    assert "ROC-AUC" in df.columns
    assert df.loc[df["Model"] == "Logistic Regression", "Accuracy"].values[0] == 0.8059


def test_compare_models_empty_raises():
    """Verify empty dictionary raises ValueError in compare_models."""
    with pytest.raises(ValueError, match="empty metrics dictionary"):
        compare_models({})


def test_plot_model_comparison(tmp_path: Path):
    """Verify plot_model_comparison produces figure and saves image file."""
    comparison_df = pd.DataFrame(
        [
            {
                "Model": "Model A",
                "Accuracy": 0.80,
                "Precision": 0.65,
                "Recall": 0.60,
                "F1 Score": 0.62,
                "ROC-AUC": 0.85,
            },
            {
                "Model": "Model B",
                "Accuracy": 0.75,
                "Precision": 0.55,
                "Recall": 0.50,
                "F1 Score": 0.52,
                "ROC-AUC": 0.78,
            },
        ]
    )

    save_path = tmp_path / "comparison.png"
    fig = plot_model_comparison(comparison_df, save_path=save_path)
    assert isinstance(fig, plt.Figure)
    assert save_path.exists()
    assert save_path.stat().st_size > 0
    plt.close(fig)


def test_plot_model_comparison_missing_model_column():
    """Verify ValueError is raised if Model column is missing."""
    df = pd.DataFrame([{"Accuracy": 0.80, "Precision": 0.65}])
    with pytest.raises(ValueError, match="must contain a 'Model' column"):
        plot_model_comparison(df)


def test_plot_feature_importance(tmp_path: Path):
    """Verify plot_feature_importance produces figure and saves image file."""
    importances = pd.Series(
        [0.4, 0.3, 0.2, 0.1],
        index=["feature_1", "feature_2", "feature_3", "feature_4"],
    )

    save_path = tmp_path / "feat_imp.png"
    fig = plot_feature_importance(importances, top_n=3, save_path=save_path)
    assert isinstance(fig, plt.Figure)
    assert save_path.exists()
    assert save_path.stat().st_size > 0
    plt.close(fig)


def test_plot_feature_importance_empty_raises():
    """Verify ValueError is raised for empty series."""
    with pytest.raises(ValueError, match="empty series"):
        plot_feature_importance(pd.Series(dtype=float))
