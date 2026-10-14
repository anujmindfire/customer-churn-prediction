"""Unit tests for dataset validation functionality."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.loader import load_dataset
from src.data.validation import (
    check_duplicates,
    check_missing_values,
    check_suspicious_values,
    generate_data_quality_report,
    get_column_info,
    get_dataset_shape,
    identify_feature_types,
    validate_target,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame with various data anomalies for testing."""
    return pd.DataFrame(
        {
            "customerID": ["C01", "C02", "C03", "C04", "C04"],  # C04 is duplicate row
            "gender": ["Female", "Male", "Female", "Male", "Male"],
            "SeniorCitizen": [0, 1, 0, 0, 0],
            "tenure": [1, 24, 0, 12, 12],
            "MonthlyCharges": [29.85, 56.95, np.nan, 80.0, 80.0],
            "TotalCharges": ["29.85", "1366.8", " ", "960.0", "960.0"],  # row 2 has " "
            "Churn": ["No", "No", "Yes", "No", "No"],
        }
    )


def test_get_dataset_shape(sample_df: pd.DataFrame):
    """Test get_dataset_shape returns correct (rows, cols)."""
    assert get_dataset_shape(sample_df) == (5, 7)


def test_get_dataset_shape_invalid_type():
    """Test get_dataset_shape raises TypeError on non-DataFrame."""
    with pytest.raises(TypeError, match="Expected pd.DataFrame"):
        get_dataset_shape([1, 2, 3])  # type: ignore[arg-type]


def test_get_column_info(sample_df: pd.DataFrame):
    """Test column metadata retrieval."""
    info = get_column_info(sample_df)
    assert info["total_columns"] == 7
    assert "customerID" in info["column_names"]
    assert "Churn" in info["column_names"]
    assert "MonthlyCharges" in info["dtypes"]
    assert info["memory_usage_mb"] > 0


def test_check_missing_values(sample_df: pd.DataFrame):
    """Test missing value detection."""
    missing = check_missing_values(sample_df)
    assert isinstance(missing, pd.Series)
    assert missing["MonthlyCharges"] == 1
    assert missing["gender"] == 0
    # TotalCharges contains " ", which is not standard NaN
    assert missing["TotalCharges"] == 0


def test_check_duplicates(sample_df: pd.DataFrame):
    """Test duplicate row detection."""
    dup_count = check_duplicates(sample_df)
    assert dup_count == 1  # 5th row is identical to 4th


def test_validate_target(sample_df: pd.DataFrame):
    """Test target column validation."""
    target_info = validate_target(sample_df, target_col="Churn")
    assert target_info["target_column"] == "Churn"
    assert target_info["is_present"] is True
    assert set(target_info["unique_values"]) == {"No", "Yes"}
    assert target_info["value_counts"]["No"] == 4
    assert target_info["value_counts"]["Yes"] == 1
    assert target_info["is_imbalanced"] is True


def test_validate_target_missing_column(sample_df: pd.DataFrame):
    """Test KeyError raised if target column is absent."""
    with pytest.raises(KeyError, match="Target column 'NonExistent' not found"):
        validate_target(sample_df, target_col="NonExistent")


def test_identify_feature_types(sample_df: pd.DataFrame):
    """Test feature classification into numerical, categorical, id, target."""
    types = identify_feature_types(
        sample_df, target_col="Churn", id_cols=["customerID"]
    )
    assert types["identifier"] == ["customerID"]
    assert types["target"] == ["Churn"]
    assert "SeniorCitizen" in types["numerical"]
    assert "tenure" in types["numerical"]
    assert "MonthlyCharges" in types["numerical"]
    assert "gender" in types["categorical"]
    assert "TotalCharges" in types["categorical"]  # object dtype due to string


def test_check_suspicious_values(sample_df: pd.DataFrame):
    """Test detection of whitespace strings and placeholder tokens."""
    suspicious = check_suspicious_values(sample_df)
    assert suspicious["has_suspicious_values"] is True
    assert "TotalCharges" in suspicious["columns_with_suspicious_values"]
    tc_info = suspicious["columns_with_suspicious_values"]["TotalCharges"]
    assert tc_info["whitespace_count"] == 1
    assert 2 in tc_info["sample_indices"]  # row index 2 has " "


def test_generate_data_quality_report(sample_df: pd.DataFrame):
    """Test end-to-end report generation."""
    report = generate_data_quality_report(sample_df, target_col="Churn")
    assert report["rows"] == 5
    assert report["columns"] == 7
    assert report["duplicate_rows"] == 1
    assert report["total_standard_nulls"] == 1
    assert report["suspicious_values"]["has_suspicious_values"] is True
    assert report["target_validation"]["target_column"] == "Churn"


def test_real_dataset_quality():
    """Test validation functions on the actual Telco dataset."""
    raw_path = Path("data/raw/telco_customer_churn.csv")
    if not raw_path.exists():
        pytest.skip("Raw dataset not found.")

    df = load_dataset(raw_path)
    report = generate_data_quality_report(df, target_col="Churn")

    assert report["rows"] == 7043
    assert report["columns"] == 21
    assert report["duplicate_rows"] == 0
    assert report["total_standard_nulls"] == 0

    # Real data has 11 whitespace strings in TotalCharges
    suspicious = report["suspicious_values"]
    assert suspicious["has_suspicious_values"] is True
    assert "TotalCharges" in suspicious["columns_with_suspicious_values"]
    assert (
        suspicious["columns_with_suspicious_values"]["TotalCharges"]["whitespace_count"]
        == 11
    )

    # Real target balance
    target = report["target_validation"]
    assert target["value_counts"]["No"] == 5174
    assert target["value_counts"]["Yes"] == 1869
    assert target["is_imbalanced"] is True
