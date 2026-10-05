"""Unit tests for data preprocessing and train/validation/test splitting."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError

from src.churn_prediction.features.preprocessing import (
    build_preprocessing_pipeline,
    clean_dataset,
    fit_preprocessor,
    get_feature_names,
    identify_feature_types,
    load_preprocessor,
    prepare_target,
    save_preprocessor,
    save_processed_datasets,
    separate_features_target,
    split_data,
    transform_data,
)


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Fixture providing a synthetic raw churn dataset."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "customerID": [f"CUST-{i:04d}" for i in range(n)],
            "gender": np.random.choice(["Male", "Female"], size=n),
            "SeniorCitizen": np.random.choice([0, 1], size=n, p=[0.8, 0.2]),
            "tenure": np.random.randint(0, 72, size=n),
            "MonthlyCharges": np.round(np.random.uniform(20.0, 110.0, size=n), 2),
            "TotalCharges": [
                " " if i < 5 else str(round(np.random.uniform(50.0, 5000.0), 2))
                for i in range(n)
            ],
            "Contract": np.random.choice(
                ["Month-to-month", "One year", "Two year"], size=n
            ),
            "Churn": np.random.choice(["No", "Yes"], size=n, p=[0.7, 0.3]),
        }
    )


# ---------------------------------------------------------------------------
# 1. Data Cleaning Tests
# ---------------------------------------------------------------------------


def test_clean_dataset_whitespace_total_charges(sample_raw_df: pd.DataFrame):
    """Test that whitespace in TotalCharges is coerced to 0.0 and converted to float."""
    # Ensure there are whitespace values
    assert (sample_raw_df["TotalCharges"] == " ").sum() == 5
    cleaned = clean_dataset(sample_raw_df)

    assert pd.api.types.is_float_dtype(cleaned["TotalCharges"].dtype)
    assert (cleaned["TotalCharges"] == 0.0).sum() >= 5
    assert cleaned["TotalCharges"].isnull().sum() == 0


def test_clean_dataset_immutability(sample_raw_df: pd.DataFrame):
    """Test that clean_dataset creates a copy and does not mutate the original DataFrame."""
    original_type = sample_raw_df["TotalCharges"].dtype
    _ = clean_dataset(sample_raw_df)
    assert sample_raw_df["TotalCharges"].dtype == original_type


def test_clean_dataset_duplicate_handling():
    """Test that duplicate records are detected and dropped."""
    df = pd.DataFrame(
        {
            "customerID": ["C1", "C1"],
            "tenure": [10, 10],
            "MonthlyCharges": [50.0, 50.0],
            "TotalCharges": ["500.0", "500.0"],
            "Churn": ["No", "No"],
        }
    )
    cleaned = clean_dataset(df)
    assert len(cleaned) == 1


def test_clean_dataset_empty_raises():
    """Test that clean_dataset raises ValueError on empty DataFrame."""
    with pytest.raises(ValueError, match="Cannot clean an empty DataFrame"):
        clean_dataset(pd.DataFrame())


# ---------------------------------------------------------------------------
# 2. Target Preparation Tests
# ---------------------------------------------------------------------------


def test_prepare_target_valid(sample_raw_df: pd.DataFrame):
    """Test converting 'No'/'Yes' target to 0/1 integers."""
    y = prepare_target(sample_raw_df, target_col="Churn")
    assert isinstance(y, pd.Series)
    assert set(y.unique()).issubset({0, 1})
    assert y.dtype == np.int64
    assert len(y) == len(sample_raw_df)
    assert y.name == "Churn"


def test_prepare_target_missing_column(sample_raw_df: pd.DataFrame):
    """Test that missing target column raises KeyError."""
    with pytest.raises(KeyError, match="Target column 'NonExistent' not found"):
        prepare_target(sample_raw_df, target_col="NonExistent")


def test_prepare_target_unexpected_classes():
    """Test that unexpected target classes raise ValueError."""
    df = pd.DataFrame({"Churn": ["No", "Yes", "Maybe", "Unknown"]})
    with pytest.raises(ValueError, match="unexpected classes"):
        prepare_target(df, target_col="Churn")


def test_prepare_target_null_values():
    """Test that target with unresolved nulls raises ValueError."""
    df = pd.DataFrame({"Churn": ["No", "Yes", None]})
    with pytest.raises(ValueError, match="unresolved missing values"):
        prepare_target(df, target_col="Churn")


def test_prepare_target_invalid_mapping():
    """Test that non-binary target mapping raises ValueError."""
    df = pd.DataFrame({"Churn": ["No", "Yes"]})
    with pytest.raises(ValueError, match="Target mapping values must be binary"):
        prepare_target(df, target_col="Churn", mapping={"No": 1, "Yes": 2})


# ---------------------------------------------------------------------------
# 3. Feature / Target Separation Tests
# ---------------------------------------------------------------------------


def test_separate_features_target(sample_raw_df: pd.DataFrame):
    """Test separating X and y, dropping target and customer identifier."""
    cleaned = clean_dataset(sample_raw_df)
    X, y = separate_features_target(cleaned, target_col="Churn", id_cols=["customerID"])

    assert "Churn" not in X.columns
    assert "customerID" not in X.columns
    assert len(X) == len(y)
    assert isinstance(y, pd.Series)
    assert set(y.unique()).issubset({0, 1})


def test_separate_features_target_empty_features_raises():
    """Test that dropping all columns raises ValueError."""
    df = pd.DataFrame({"customerID": ["C1"], "Churn": ["No"]})
    with pytest.raises(ValueError, match="zero columns"):
        separate_features_target(df, target_col="Churn", id_cols=["customerID"])


# ---------------------------------------------------------------------------
# 4. Feature Identification Tests
# ---------------------------------------------------------------------------


def test_identify_feature_types(sample_raw_df: pd.DataFrame):
    """Test programmatic classification into numerical and categorical columns."""
    cleaned = clean_dataset(sample_raw_df)
    X, _ = separate_features_target(cleaned)
    num_cols, cat_cols = identify_feature_types(X)

    assert set(num_cols) == {
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    }
    assert set(cat_cols) == {"gender", "Contract"}
    assert len(num_cols) + len(cat_cols) == X.shape[1]


# ---------------------------------------------------------------------------
# 5. Train/Validation/Test Split Tests
# ---------------------------------------------------------------------------


def test_split_data_proportions_and_shapes(sample_raw_df: pd.DataFrame):
    """Test 70/15/15 stratified split generates correct partitions."""
    cleaned = clean_dataset(sample_raw_df)
    X, y = separate_features_target(cleaned)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42
    )

    n_total = len(cleaned)
    assert len(X_train) == 70
    assert len(X_val) == 15
    assert len(X_test) == 15
    assert len(X_train) + len(X_val) + len(X_test) == n_total
    assert len(y_train) == 70
    assert len(y_val) == 15
    assert len(y_test) == 15


def test_split_data_stratification(sample_raw_df: pd.DataFrame):
    """Test that churn proportions are approximately equal across splits."""
    cleaned = clean_dataset(sample_raw_df)
    X, y = separate_features_target(cleaned)

    _, _, _, y_train, y_val, y_test = split_data(
        X, y, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42
    )

    base_rate = float(y.mean())
    assert abs(float(y_train.mean()) - base_rate) < 0.05
    assert abs(float(y_val.mean()) - base_rate) < 0.08
    assert abs(float(y_test.mean()) - base_rate) < 0.08


def test_split_data_reproducibility(sample_raw_df: pd.DataFrame):
    """Test that the same random seed produces identical splits."""
    cleaned = clean_dataset(sample_raw_df)
    X, y = separate_features_target(cleaned)

    split1 = split_data(X, y, random_state=42)
    split2 = split_data(X, y, random_state=42)

    pd.testing.assert_frame_equal(split1[0], split2[0])
    pd.testing.assert_series_equal(split1[3], split2[3])


def test_split_data_invalid_proportions(sample_raw_df: pd.DataFrame):
    """Test that split proportions not summing to 1.0 raise ValueError."""
    cleaned = clean_dataset(sample_raw_df)
    X, y = separate_features_target(cleaned)

    with pytest.raises(ValueError, match="Split proportions must sum to 1.0"):
        split_data(X, y, train_size=0.5, val_size=0.2, test_size=0.2)


# ---------------------------------------------------------------------------
# 6. Preprocessing Pipeline & Transformers Tests
# ---------------------------------------------------------------------------


def test_build_pipeline_and_transform(sample_raw_df: pd.DataFrame):
    """Test constructing and executing ColumnTransformer pipeline."""
    cleaned = clean_dataset(sample_raw_df)
    X, y = separate_features_target(cleaned)
    num_cols, cat_cols = identify_feature_types(X)

    X_train, X_val, X_test, _, _, _ = split_data(X, y, random_state=42)

    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)
    assert isinstance(preprocessor, ColumnTransformer)

    # Transform before fit should raise NotFittedError
    with pytest.raises(NotFittedError):
        transform_data(preprocessor, X_val)

    # Fit only on training data
    fit_preprocessor(preprocessor, X_train)

    X_train_proc = transform_data(preprocessor, X_train, as_dataframe=True)
    X_val_proc = transform_data(preprocessor, X_val, as_dataframe=True)
    X_test_proc = transform_data(preprocessor, X_test, as_dataframe=True)

    assert isinstance(X_train_proc, pd.DataFrame)
    assert isinstance(X_val_proc, pd.DataFrame)
    assert isinstance(X_test_proc, pd.DataFrame)

    assert X_train_proc.shape[0] == len(X_train)
    assert X_val_proc.shape[0] == len(X_val)
    assert X_test_proc.shape[0] == len(X_test)

    assert X_train_proc.shape[1] == X_val_proc.shape[1] == X_test_proc.shape[1]

    # Verify feature names match
    names = get_feature_names(preprocessor)
    assert list(X_train_proc.columns) == names


def test_pipeline_imputation_and_unknown_categories():
    """Test imputer handles missing values and OneHotEncoder ignores unknown categories."""
    train_data = pd.DataFrame(
        {
            "num_feat": [10.0, 20.0, 30.0, np.nan],
            "cat_feat": ["A", "B", "A", None],
        }
    )
    val_data = pd.DataFrame(
        {
            "num_feat": [np.nan, 25.0],
            # "C" is an unseen category not present during fit
            "cat_feat": ["C", "B"],
        }
    )

    preprocessor = build_preprocessing_pipeline(
        numerical_cols=["num_feat"], categorical_cols=["cat_feat"]
    )
    fit_preprocessor(preprocessor, train_data)

    # Should transform without error despite missing values and unseen category "C"
    val_proc = transform_data(preprocessor, val_data, as_dataframe=True)
    assert isinstance(val_proc, pd.DataFrame)
    assert val_proc.isnull().sum().sum() == 0
    assert len(val_proc) == 2


# ---------------------------------------------------------------------------
# 7. Data Leakage Prevention Tests
# ---------------------------------------------------------------------------


def test_data_leakage_scaler_fitted_on_train_only():
    """Verify scaler statistics are computed exclusively on training data.

    If training data has mean 10 and validation data has an extreme outlier (1,000,000),
    the preprocessor's stored mean_ MUST remain 10.0, demonstrating that validation
    data does not bleed into the preprocessing parameters.
    """
    X_train = pd.DataFrame({"num": [8.0, 10.0, 12.0], "cat": ["A", "B", "A"]})
    X_val = pd.DataFrame({"num": [1_000_000.0, 2_000_000.0], "cat": ["A", "B"]})

    preprocessor = build_preprocessing_pipeline(
        numerical_cols=["num"], categorical_cols=["cat"]
    )
    fit_preprocessor(preprocessor, X_train)

    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
    expected_train_mean = 10.0
    assert np.isclose(scaler.mean_[0], expected_train_mean)

    # Transform validation set
    _ = transform_data(preprocessor, X_val)

    # Scaler mean must still be 10.0, not affected by the extreme validation numbers
    assert np.isclose(scaler.mean_[0], expected_train_mean)


# ---------------------------------------------------------------------------
# 8. Serialization and Artifact Saving Tests
# ---------------------------------------------------------------------------


def test_save_and_load_preprocessor(tmp_path: Path, sample_raw_df: pd.DataFrame):
    """Test serializing and loading the ColumnTransformer artifact."""
    cleaned = clean_dataset(sample_raw_df)
    X, y = separate_features_target(cleaned)
    num_cols, cat_cols = identify_feature_types(X)
    X_train, _, X_test, _, _, _ = split_data(X, y, random_state=42)

    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)

    # Saving unfitted preprocessor should raise NotFittedError
    save_file = tmp_path / "preprocessor.joblib"
    with pytest.raises(NotFittedError):
        save_preprocessor(preprocessor, save_file)

    fit_preprocessor(preprocessor, X_train)
    saved_path = save_preprocessor(preprocessor, save_file)
    assert saved_path.exists()

    loaded = load_preprocessor(saved_path)
    assert isinstance(loaded, ColumnTransformer)

    # Assert loaded preprocessor produces identical output
    original_out = transform_data(preprocessor, X_test, as_dataframe=False)
    loaded_out = transform_data(loaded, X_test, as_dataframe=False)
    np.testing.assert_allclose(original_out, loaded_out)


def test_load_preprocessor_missing_file_raises():
    """Test that loading a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_preprocessor("non_existent_file.joblib")


def test_save_processed_datasets(tmp_path: Path):
    """Test saving processed train, val, and test datasets with metadata."""
    X_train = pd.DataFrame({"f1": [1.0, 2.0], "f2": [0.5, 0.8]})
    y_train = pd.Series([0, 1], name="Churn")
    X_val = pd.DataFrame({"f1": [3.0], "f2": [0.1]})
    y_val = pd.Series([0], name="Churn")
    X_test = pd.DataFrame({"f1": [4.0], "f2": [0.9]})
    y_test = pd.Series([1], name="Churn")

    saved_files = save_processed_datasets(
        X_train, y_train, X_val, y_val, X_test, y_test, output_dir=tmp_path
    )

    for path in saved_files.values():
        assert path.exists()

    train_read = pd.read_csv(saved_files["train"])
    assert "Churn" in train_read.columns
    assert len(train_read) == 2
