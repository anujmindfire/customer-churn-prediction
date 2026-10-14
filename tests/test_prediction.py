"""Unit tests for production predict_churn function and inference pipeline."""

import pandas as pd
import pytest

from src.config import FINAL_MODEL_PATH, PREPROCESSOR_PATH
from src.predict import ChurnPredictionResult, predict_churn


@pytest.fixture
def sample_customer_raw() -> dict[str, object]:
    """Provide a valid single customer raw record."""
    return {
        "customerID": "TEST-0001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }


def test_predict_churn_single_dict(sample_customer_raw: dict[str, object]):
    """Verify predict_churn returns valid ChurnPredictionResult for single dict."""
    result = predict_churn(
        sample_customer_raw,
        model_path=FINAL_MODEL_PATH,
        preprocessor_path=PREPROCESSOR_PATH,
    )

    assert isinstance(result, ChurnPredictionResult)
    assert result.churn_prediction in [0, 1]
    assert 0.0 <= result.churn_probability <= 1.0

    # Dict key access
    assert "churn_prediction" in result
    assert "churn_probability" in result
    assert "prediction" in result
    assert "probability" in result

    # Object attribute access
    assert hasattr(result, "churn_prediction")
    assert hasattr(result, "churn_probability")

    # Unpacking access
    pred, prob = result
    assert pred == result.churn_prediction
    assert prob == result.churn_probability


def test_predict_churn_batch_dataframe(sample_customer_raw: dict[str, object]):
    """Verify predict_churn returns augmented DataFrame for multi-row input."""
    row1 = sample_customer_raw.copy()
    row2 = sample_customer_raw.copy()
    row2["tenure"] = 65
    row2["Contract"] = "Two year"

    df_batch = pd.DataFrame([row1, row2])
    result_df = predict_churn(df_batch)

    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 2
    assert "churn_prediction" in result_df.columns
    assert "churn_probability" in result_df.columns
    assert set(result_df["churn_prediction"].unique()).issubset({0, 1})
    assert (result_df["churn_probability"] >= 0.0).all()
    assert (result_df["churn_probability"] <= 1.0).all()


def test_predict_churn_list_of_dicts(sample_customer_raw: dict[str, object]):
    """Verify predict_churn handles list of customer dictionaries."""
    result = predict_churn([sample_customer_raw, sample_customer_raw])
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


def test_predict_churn_string_total_charges(sample_customer_raw: dict[str, object]):
    """Verify TotalCharges passed as string with whitespace is handled safely."""
    customer = sample_customer_raw.copy()
    customer["TotalCharges"] = " "  # Whitespace edge case
    result = predict_churn(customer)
    assert isinstance(result, ChurnPredictionResult)
    assert result.churn_prediction in [0, 1]


def test_predict_churn_empty_raises():
    """Verify empty inputs raise ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        predict_churn([])
    with pytest.raises(ValueError, match="cannot be empty"):
        predict_churn(pd.DataFrame())


def test_predict_churn_invalid_type_raises():
    """Verify invalid input types raise TypeError."""
    with pytest.raises(TypeError, match="Expected customer_data"):
        predict_churn("invalid_string")  # type: ignore[arg-type]


def test_predict_churn_custom_threshold(sample_customer_raw: dict[str, object]):
    """Verify custom threshold alters binary decision cutoffs."""
    res_low = predict_churn(sample_customer_raw, threshold=0.01)
    res_high = predict_churn(sample_customer_raw, threshold=0.99)

    assert res_low.churn_prediction == 1
    assert res_high.churn_prediction == 0
