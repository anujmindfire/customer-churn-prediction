"""Data preprocessing, cleaning, transformation, and splitting pipelines.

This module provides reusable, modular components to clean raw customer churn data,
perform stratified train/validation/test splits, and apply scikit-learn preprocessing
pipelines (imputation, scaling, one-hot encoding) strictly fitted on training data
to prevent data leakage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.exceptions import NotFittedError  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import (  # type: ignore[import-untyped]
    OneHotEncoder,
    StandardScaler,
)

from src.churn_prediction.config import (
    ID_COLS,
    PREPROCESSOR_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TARGET_COL,
    TARGET_MAPPING,
    TEST_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
)

logger = logging.getLogger(__name__)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Perform deterministic data cleaning based on exploratory findings.

    Specific cleaning steps:
    1. Makes a defensive copy to preserve the original DataFrame.
    2. Cleans 'TotalCharges': converts whitespace strings (found in zero-tenure records)
       to 0.0 and casts the column to float64.
    3. Handles duplicate rows by logging and dropping exact duplicates.

    Args:
        df: Input raw pandas DataFrame.

    Returns:
        pd.DataFrame: Cleaned pandas DataFrame.

    Raises:
        TypeError: If df is not a pandas DataFrame.
        ValueError: If df is empty.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")
    if df.empty:
        raise ValueError("Cannot clean an empty DataFrame.")

    cleaned_df = df.copy()

    # 1. Clean TotalCharges whitespace anomalies
    if "TotalCharges" in cleaned_df.columns:
        original_dtype = cleaned_df["TotalCharges"].dtype
        if not pd.api.types.is_numeric_dtype(original_dtype):
            whitespace_mask = cleaned_df["TotalCharges"].astype(str).str.strip() == ""
            whitespace_count = int(whitespace_mask.sum())
            if whitespace_count > 0:
                logger.info(
                    "Found %d records with whitespace in 'TotalCharges'. Coercing to 0.0.",
                    whitespace_count,
                )
            cleaned_series = (
                cleaned_df["TotalCharges"].astype(str).str.strip().replace("", "0.0")
            )
            cleaned_df["TotalCharges"] = pd.to_numeric(
                cleaned_series, errors="coerce"
            ).fillna(0.0)
            logger.info("Converted 'TotalCharges' from %s to float64.", original_dtype)

    # 2. Check and handle duplicate rows
    duplicate_count = int(cleaned_df.duplicated().sum())
    if duplicate_count > 0:
        logger.warning(
            "Found %d exact duplicate rows. Removing duplicates, keeping first occurrence.",
            duplicate_count,
        )
        cleaned_df = cleaned_df.drop_duplicates(keep="first").reset_index(drop=True)
    else:
        logger.info("No duplicate rows found in dataset.")

    logger.info("Data cleaning complete. Cleaned shape: %s", cleaned_df.shape)
    return cleaned_df


def prepare_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    mapping: dict[str, int] | None = None,
) -> pd.Series:
    """Validate and convert the churn target column into a binary numeric Series.

    Args:
        df: Input pandas DataFrame.
        target_col: Column name of the churn target variable. Defaults to TARGET_COL ("Churn").
        mapping: Dictionary mapping original labels to binary values (0/1).
            Defaults to TARGET_MAPPING ({"No": 0, "Yes": 1}).

    Returns:
        pd.Series: Binary target Series (dtype int64).

    Raises:
        TypeError: If df is not a pandas DataFrame.
        KeyError: If target_col is not present in df.
        ValueError: If target contains nulls, non-binary, or unexpected classes.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")
    if target_col not in df.columns:
        msg = f"Target column '{target_col}' not found in DataFrame columns: {list(df.columns)}"
        logger.error(msg)
        raise KeyError(msg)

    target_mapping = mapping if mapping is not None else TARGET_MAPPING
    series = df[target_col]

    # Validate no missing values
    null_count = int(series.isnull().sum())
    if null_count > 0:
        msg = f"Target column '{target_col}' contains {null_count} unresolved missing values."
        logger.error(msg)
        raise ValueError(msg)

    # Validate all unique values are accounted for in mapping
    unique_vals = set(series.unique())
    expected_vals = set(target_mapping.keys())
    unexpected_vals = unique_vals - expected_vals

    if unexpected_vals:
        msg = (
            f"Target column '{target_col}' contains unexpected classes: {unexpected_vals}. "
            f"Expected classes are: {expected_vals}."
        )
        logger.error(msg)
        raise ValueError(msg)

    # Validate mapped values are strictly binary {0, 1}
    mapped_targets = set(target_mapping.values())
    if mapped_targets != {0, 1}:
        msg = f"Target mapping values must be binary {{0, 1}}, got {mapped_targets}."
        logger.error(msg)
        raise ValueError(msg)

    binary_target: pd.Series = series.map(target_mapping).astype("int64")
    binary_target.name = target_col

    logger.info(
        "Target preparation successful for '%s': 0=%d, 1=%d (churn rate: %.2f%%).",
        target_col,
        int((binary_target == 0).sum()),
        int((binary_target == 1).sum()),
        float(binary_target.mean() * 100),
    )
    return binary_target


def separate_features_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    id_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate input features and binary target, excluding non-predictive identifiers.

    Excludes the target column and non-predictive identifiers (such as 'customerID')
    from the feature matrix X to prevent high-cardinality noise or data leakage.

    Args:
        df: Input DataFrame.
        target_col: Column name of the churn target variable. Defaults to TARGET_COL ("Churn").
        id_cols: List of identifier column names to drop. Defaults to ID_COLS (["customerID"]).

    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature matrix (X) and binary target series (y).

    Raises:
        TypeError: If df is not a pandas DataFrame.
        KeyError: If target_col is missing from df.
        ValueError: If dropping columns results in empty feature set.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

    # Prepare binary target first
    y = prepare_target(df, target_col=target_col)

    # Determine identifier columns to drop
    cols_to_drop = [target_col]
    identifiers_to_drop = id_cols if id_cols is not None else ID_COLS
    for col in identifiers_to_drop:
        if col in df.columns and col not in cols_to_drop:
            cols_to_drop.append(col)

    logger.info(
        "Separating features and target: dropping %s from feature set X.",
        cols_to_drop,
    )

    X = df.drop(columns=cols_to_drop).copy()

    if X.empty or X.shape[1] == 0:
        msg = "Feature matrix X has zero columns after dropping target and identifier columns."
        logger.error(msg)
        raise ValueError(msg)

    logger.info(
        "Separation completed: X shape=%s, y length=%d.",
        X.shape,
        len(y),
    )
    return X, y


def identify_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Programmatically classify feature columns into numerical and categorical lists.

    Args:
        X: Feature matrix DataFrame (without target or identifier).

    Returns:
        tuple[list[str], list[str]]: (numerical_features, categorical_features).

    Raises:
        TypeError: If X is not a pandas DataFrame.
        ValueError: If X has no columns.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(X).__name__}")
    if X.empty or X.shape[1] == 0:
        raise ValueError("Feature matrix X has no columns to classify.")

    numerical_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col].dtype):
            numerical_cols.append(str(col))
        else:
            categorical_cols.append(str(col))

    logger.info(
        "Identified feature types: %d numerical features, %d categorical features.",
        len(numerical_cols),
        len(categorical_cols),
    )
    logger.debug("Numerical columns: %s", numerical_cols)
    logger.debug("Categorical columns: %s", categorical_cols)

    return numerical_cols, categorical_cols


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = TRAIN_SIZE,
    val_size: float = VAL_SIZE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Perform a stratified two-stage train/validation/test split.

    Preserves target churn proportion across all partitions using scikit-learn
    stratification. Default proportions are 70% Train, 15% Validation, 15% Test.

    Args:
        X: Feature matrix DataFrame.
        y: Binary target Series.
        train_size: Proportion of dataset for training (default 0.70).
        val_size: Proportion of dataset for validation (default 0.15).
        test_size: Proportion of dataset for testing (default 0.15).
        random_state: Seed for reproducible pseudo-random splitting (default 42).

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test).

    Raises:
        ValueError: If split proportions do not sum to 1.0 or if dataset length mismatches.
    """
    total_prop = train_size + val_size + test_size
    if not np.isclose(total_prop, 1.0, atol=1e-5):
        msg = f"Split proportions must sum to 1.0, got {train_size} + {val_size} + {test_size} = {total_prop}"
        logger.error(msg)
        raise ValueError(msg)

    if len(X) != len(y):
        msg = f"Length mismatch: X has {len(X)} rows but y has {len(y)} rows."
        logger.error(msg)
        raise ValueError(msg)

    if len(X) < 10:
        msg = f"Dataset too small to split meaningfully: {len(X)} rows."
        logger.error(msg)
        raise ValueError(msg)

    temp_size = val_size + test_size

    # Stage 1: Split into Train (70%) and Temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=temp_size,
        random_state=random_state,
        stratify=y,
    )

    # Stage 2: Split Temp into Validation (50% of Temp = 15%) and Test (50% of Temp = 15%)
    relative_test_size = test_size / temp_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=y_temp,
    )

    n_total = len(X)
    logger.info(
        "Split complete: Train=%d (%.2f%%), Val=%d (%.2f%%), Test=%d (%.2f%%).",
        len(X_train),
        len(X_train) / n_total * 100,
        len(X_val),
        len(X_val) / n_total * 100,
        len(X_test),
        len(X_test) / n_total * 100,
    )
    logger.info(
        "Churn rates: Train=%.2f%%, Val=%.2f%%, Test=%.2f%% (Overall=%.2f%%).",
        float(y_train.mean() * 100),
        float(y_val.mean() * 100),
        float(y_test.mean() * 100),
        float(y.mean() * 100),
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessing_pipeline(
    numerical_cols: list[str],
    categorical_cols: list[str],
    verbose_feature_names: bool = False,
) -> ColumnTransformer:
    """Construct an unfitted scikit-learn ColumnTransformer preprocessing pipeline.

    Numerical pipeline:
        1. SimpleImputer(strategy='median') for robust outlier-resistant imputation.
        2. StandardScaler() to standardize features to zero mean and unit variance.

    Categorical pipeline:
        1. SimpleImputer(strategy='most_frequent') for mode imputation.
        2. OneHotEncoder(handle_unknown='ignore', sparse_output=False) to ensure
           unseen categories at inference time do not trigger crashes, and produce
           dense array output for compatibility with downstream classifiers.

    Args:
        numerical_cols: List of column names to treat as numerical.
        categorical_cols: List of column names to treat as categorical.
        verbose_feature_names: If False, omits transformer prefixes ('num__', 'cat__')
            when calling get_feature_names_out(). Defaults to False.

    Returns:
        ColumnTransformer: Configured, unfitted ColumnTransformer object.

    Raises:
        ValueError: If both numerical_cols and categorical_cols are empty.
    """
    if not numerical_cols and not categorical_cols:
        raise ValueError(
            "At least one numerical or categorical column must be provided."
        )

    transformers: list[tuple[str, Any, list[str]]] = []

    if numerical_cols:
        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numerical_pipeline, numerical_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=verbose_feature_names,
    )

    logger.info(
        "Built ColumnTransformer with %d numerical columns and %d categorical columns.",
        len(numerical_cols),
        len(categorical_cols),
    )
    return preprocessor


def fit_preprocessor(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
) -> ColumnTransformer:
    """Fit the preprocessing pipeline strictly on training feature data.

    CRITICAL LEAKAGE RULE:
    The pipeline MUST ONLY be fitted on X_train. It must never see X_val, X_test,
    or the full un-split feature matrix X during fitting.

    Args:
        preprocessor: Unfitted ColumnTransformer instance.
        X_train: Training feature DataFrame.

    Returns:
        ColumnTransformer: Fitted ColumnTransformer instance.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If X_train is empty.
    """
    if not isinstance(preprocessor, ColumnTransformer):
        raise TypeError(
            f"Expected ColumnTransformer, got {type(preprocessor).__name__}"
        )
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(X_train).__name__}")
    if X_train.empty:
        raise ValueError("Cannot fit preprocessor on an empty DataFrame.")

    logger.info(
        "Fitting preprocessor exclusively on training data (rows: %d, columns: %d) "
        "to guarantee zero data leakage.",
        X_train.shape[0],
        X_train.shape[1],
    )
    preprocessor.fit(X_train)
    logger.info("Preprocessor successfully fitted on training data.")
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Retrieve output feature names from a fitted ColumnTransformer.

    Args:
        preprocessor: Fitted ColumnTransformer instance.

    Returns:
        list[str]: Transformed output feature names.

    Raises:
        NotFittedError: If preprocessor has not been fitted yet.
    """
    try:
        names = preprocessor.get_feature_names_out()
        return [str(name) for name in names]
    except NotFittedError as err:
        logger.error("Cannot retrieve feature names from an unfitted preprocessor.")
        raise NotFittedError(
            "Preprocessor must be fitted before retrieving feature names."
        ) from err


def transform_data(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    as_dataframe: bool = True,
) -> np.ndarray | pd.DataFrame:
    """Transform a feature matrix using an already fitted ColumnTransformer.

    Args:
        preprocessor: Fitted ColumnTransformer instance.
        X: Feature matrix DataFrame to transform.
        as_dataframe: If True, returns a pandas DataFrame with preserved index
            and transformed feature names. If False, returns raw numpy ndarray.
            Defaults to True.

    Returns:
        np.ndarray | pd.DataFrame: Transformed features.

    Raises:
        TypeError: If inputs are invalid.
        NotFittedError: If preprocessor is not fitted.
        ValueError: If transformation fails.
    """
    if not isinstance(preprocessor, ColumnTransformer):
        raise TypeError(
            f"Expected ColumnTransformer, got {type(preprocessor).__name__}"
        )
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(X).__name__}")

    try:
        transformed: np.ndarray = preprocessor.transform(X)
    except NotFittedError as err:
        logger.error("Transform failed: preprocessor is not fitted.")
        raise NotFittedError(
            "Preprocessor must be fitted before transforming data."
        ) from err

    logger.debug(
        "Transformed input matrix %s into feature shape %s.",
        X.shape,
        transformed.shape,
    )

    if as_dataframe:
        feature_names = get_feature_names(preprocessor)
        return pd.DataFrame(
            transformed,
            columns=feature_names,
            index=X.index,
        )

    return transformed


def save_preprocessor(
    preprocessor: ColumnTransformer,
    filepath: str | Path = PREPROCESSOR_PATH,
) -> Path:
    """Serialize and save the fitted preprocessor artifact using joblib.

    Args:
        preprocessor: Fitted ColumnTransformer instance.
        filepath: Destination file path. Defaults to PREPROCESSOR_PATH (models/preprocessor.joblib).

    Returns:
        Path: Path where the preprocessor was saved.

    Raises:
        NotFittedError: If preprocessor has not been fitted.
        TypeError: If preprocessor is not a ColumnTransformer.
    """
    if not isinstance(preprocessor, ColumnTransformer):
        raise TypeError(
            f"Expected ColumnTransformer, got {type(preprocessor).__name__}"
        )

    # Ensure preprocessor has been fitted
    try:
        _ = preprocessor.get_feature_names_out()
    except NotFittedError as err:
        msg = "Refusing to save an unfitted preprocessor artifact."
        logger.error(msg)
        raise NotFittedError(msg) from err

    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, path)
    logger.info("Successfully serialized preprocessor artifact to: %s", path)
    return path


def load_preprocessor(filepath: str | Path = PREPROCESSOR_PATH) -> ColumnTransformer:
    """Deserialize and load a preprocessor artifact from disk using joblib.

    Args:
        filepath: Source file path. Defaults to PREPROCESSOR_PATH (models/preprocessor.joblib).

    Returns:
        ColumnTransformer: Loaded fitted ColumnTransformer instance.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        TypeError: If loaded object is not a ColumnTransformer.
    """
    path = Path(filepath).resolve()
    if not path.exists():
        msg = f"Preprocessor artifact not found at: {path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    loaded = joblib.load(path)
    if not isinstance(loaded, ColumnTransformer):
        msg = f"Loaded artifact is of type {type(loaded).__name__}, expected ColumnTransformer."
        logger.error(msg)
        raise TypeError(msg)

    logger.info("Successfully loaded preprocessor artifact from: %s", path)
    return loaded


def save_processed_datasets(
    X_train_proc: pd.DataFrame,
    y_train: pd.Series,
    X_val_proc: pd.DataFrame,
    y_val: pd.Series,
    X_test_proc: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str | Path = PROCESSED_DATA_DIR,
    target_col: str = TARGET_COL,
) -> dict[str, Path]:
    """Save transformed train, validation, and test datasets with target columns.

    Outputs:
    - train.csv: Transformed training features + target
    - val.csv: Transformed validation features + target
    - test.csv: Transformed test features + target
    - metadata.json: Partition metadata and column schema

    Args:
        X_train_proc: Transformed training feature DataFrame.
        y_train: Training binary target Series.
        X_val_proc: Transformed validation feature DataFrame.
        y_val: Validation binary target Series.
        X_test_proc: Transformed test feature DataFrame.
        y_test: Test binary target Series.
        output_dir: Destination directory path. Defaults to PROCESSED_DATA_DIR.
        target_col: Target column name. Defaults to TARGET_COL ("Churn").

    Returns:
        dict[str, Path]: Dictionary mapping partition names to saved file paths.
    """
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    # Combine features and target for each partition
    train_df = X_train_proc.copy()
    train_df[target_col] = y_train.values

    val_df = X_val_proc.copy()
    val_df[target_col] = y_val.values

    test_df = X_test_proc.copy()
    test_df[target_col] = y_test.values

    train_path = out_path / "train.csv"
    val_path = out_path / "val.csv"
    test_path = out_path / "test.csv"
    meta_path = out_path / "metadata.json"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    metadata = {
        "partitions": {
            "train": {
                "rows": len(train_df),
                "columns": int(train_df.shape[1]),
                "churn_rate": round(float(y_train.mean()), 4),
                "churn_positive_count": int(y_train.sum()),
                "churn_negative_count": int((y_train == 0).sum()),
            },
            "validation": {
                "rows": len(val_df),
                "columns": int(val_df.shape[1]),
                "churn_rate": round(float(y_val.mean()), 4),
                "churn_positive_count": int(y_val.sum()),
                "churn_negative_count": int((y_val == 0).sum()),
            },
            "test": {
                "rows": len(test_df),
                "columns": int(test_df.shape[1]),
                "churn_rate": round(float(y_test.mean()), 4),
                "churn_positive_count": int(y_test.sum()),
                "churn_negative_count": int((y_test == 0).sum()),
            },
        },
        "feature_count": int(X_train_proc.shape[1]),
        "feature_names": list(X_train_proc.columns),
        "target_column": target_col,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "Saved processed datasets to %s (train: %d, val: %d, test: %d).",
        out_path,
        len(train_df),
        len(val_df),
        len(test_df),
    )

    return {
        "train": train_path,
        "validation": val_path,
        "test": test_path,
        "metadata": meta_path,
    }
