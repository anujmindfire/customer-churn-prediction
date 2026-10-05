"""Unit tests for dataset loading functionality."""

from pathlib import Path

import pandas as pd
import pytest

from src.churn_prediction.data.loader import load_dataset


def test_load_dataset_success(tmp_path: Path):
    """Test loading a valid CSV file."""
    csv_file = tmp_path / "sample.csv"
    sample_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    sample_df.to_csv(csv_file, index=False)

    df = load_dataset(csv_file)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 2)
    assert list(df.columns) == ["col1", "col2"]


def test_load_dataset_accepts_string_path(tmp_path: Path):
    """Test loading with string path."""
    csv_file = tmp_path / "sample_str.csv"
    pd.DataFrame({"x": [10]}).to_csv(csv_file, index=False)

    df = load_dataset(str(csv_file))
    assert df.shape == (1, 1)
    assert df.iloc[0, 0] == 10


def test_load_dataset_file_not_found(tmp_path: Path):
    """Test that a non-existent file raises FileNotFoundError."""
    missing_file = tmp_path / "non_existent.csv"
    with pytest.raises(FileNotFoundError, match="not found"):
        load_dataset(missing_file)


def test_load_dataset_not_a_csv(tmp_path: Path):
    """Test that non-CSV file raises ValueError."""
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("hello,world")

    with pytest.raises(ValueError, match="Expected a CSV file"):
        load_dataset(txt_file)


def test_load_dataset_directory_raises_error(tmp_path: Path):
    """Test that passing a directory raises ValueError."""
    with pytest.raises(ValueError, match="not a file"):
        load_dataset(tmp_path)


def test_load_dataset_empty_file_raises_error(tmp_path: Path):
    """Test that an empty CSV file raises ValueError."""
    empty_file = tmp_path / "empty.csv"
    empty_file.touch()

    with pytest.raises(ValueError, match="empty"):
        load_dataset(empty_file)


def test_load_dataset_invalid_type():
    """Test that invalid filepath types raise TypeError."""
    with pytest.raises(TypeError, match="filepath must be a str or Path"):
        load_dataset(123)  # type: ignore[arg-type]


def test_load_dataset_real_data():
    """Test loading the real raw telco customer churn dataset if present."""
    raw_path = Path("data/raw/telco_customer_churn.csv")
    if raw_path.exists():
        df = load_dataset(raw_path)
        assert df.shape == (7043, 21)
        assert "Churn" in df.columns
        assert "customerID" in df.columns
