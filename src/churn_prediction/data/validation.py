"""Data validation and quality assessment utilities for Customer Churn Prediction."""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _validate_is_dataframe(df: pd.DataFrame) -> None:
    """Validate that the input object is a pandas DataFrame.

    Args:
        df: Input object to validate.

    Raises:
        TypeError: If df is not an instance of pd.DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")


def get_dataset_shape(df: pd.DataFrame) -> tuple[int, int]:
    """Return the dimensions (rows, columns) of the DataFrame.

    Args:
        df: Input pandas DataFrame.

    Returns:
        Tuple[int, int]: Number of rows and number of columns.

    Raises:
        TypeError: If df is not a pandas DataFrame.
    """
    _validate_is_dataframe(df)
    shape: tuple[int, int] = (int(df.shape[0]), int(df.shape[1]))
    logger.debug("Dataset shape: %d rows, %d columns", shape[0], shape[1])
    return shape


def get_column_info(df: pd.DataFrame) -> dict[str, Any]:
    """Retrieve column names, data types, and total memory usage.

    Args:
        df: Input pandas DataFrame.

    Returns:
        Dict[str, Any]: Dictionary containing column summary metadata.

    Raises:
        TypeError: If df is not a pandas DataFrame.
    """
    _validate_is_dataframe(df)
    memory_bytes = float(df.memory_usage(deep=True).sum())
    memory_mb = round(memory_bytes / (1024 * 1024), 4)

    return {
        "total_columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
        "memory_usage_mb": memory_mb,
    }


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Count missing (standard null/NaN) values for each column.

    Args:
        df: Input pandas DataFrame.

    Returns:
        pd.Series: Count of null values indexed by column name.

    Raises:
        TypeError: If df is not a pandas DataFrame.
    """
    _validate_is_dataframe(df)
    missing = df.isnull().sum()
    total_missing = int(missing.sum())
    logger.info("Found %d standard null values across all columns.", total_missing)
    return missing


def check_duplicates(df: pd.DataFrame) -> int:
    """Count completely duplicated rows in the dataset.

    Args:
        df: Input pandas DataFrame.

    Returns:
        int: Number of duplicate rows.

    Raises:
        TypeError: If df is not a pandas DataFrame.
    """
    _validate_is_dataframe(df)
    dup_count = int(df.duplicated().sum())
    logger.info("Duplicate check complete: %d duplicate rows found.", dup_count)
    return dup_count


def validate_target(df: pd.DataFrame, target_col: str = "Churn") -> dict[str, Any]:
    """Validate presence, distributions, and class balance of target column.

    Args:
        df: Input pandas DataFrame.
        target_col: Name of the target variable column. Defaults to "Churn".

    Returns:
        Dict[str, Any]: Target validation metrics and distributions.

    Raises:
        TypeError: If df is not a pandas DataFrame.
        KeyError: If target_col is not present in df.
    """
    _validate_is_dataframe(df)
    if target_col not in df.columns:
        msg = f"Target column '{target_col}' not found in DataFrame columns."
        logger.error(msg)
        raise KeyError(msg)

    series = df[target_col]
    counts_dict = series.value_counts(dropna=False).to_dict()
    proportions_dict = (
        series.value_counts(normalize=True, dropna=False).round(4).to_dict()
    )
    missing_count = int(series.isnull().sum())
    unique_vals = list(series.dropna().unique())

    # Detect imbalance (if ratio of max to min class is greater than 1.5)
    counts = list(series.value_counts(dropna=True))
    is_imbalanced = False
    imbalance_ratio = 1.0
    if len(counts) > 1 and min(counts) > 0:
        imbalance_ratio = round(max(counts) / min(counts), 3)
        is_imbalanced = imbalance_ratio > 1.5

    result: dict[str, Any] = {
        "target_column": target_col,
        "is_present": True,
        "unique_values": unique_vals,
        "value_counts": {str(k): int(v) for k, v in counts_dict.items()},
        "proportions": {str(k): float(v) for k, v in proportions_dict.items()},
        "missing_count": missing_count,
        "is_imbalanced": is_imbalanced,
        "imbalance_ratio": imbalance_ratio,
    }
    logger.info(
        "Target validation complete for '%s': %d unique values, imbalanced=%s",
        target_col,
        len(unique_vals),
        is_imbalanced,
    )
    return result


def identify_feature_types(
    df: pd.DataFrame,
    target_col: str | None = "Churn",
    id_cols: list[str] | None = None,
) -> dict[str, list[str]]:
    """Classify DataFrame columns into numerical, categorical, ID, and target.

    Args:
        df: Input pandas DataFrame.
        target_col: Name of the target variable (optional). Defaults to "Churn".
        id_cols: List of column names representing customer IDs. Defaults to ["customerID"].

    Returns:
        Dict[str, List[str]]: Mapping of column categories to column name lists.

    Raises:
        TypeError: If df is not a pandas DataFrame.
    """
    _validate_is_dataframe(df)

    if id_cols is None:
        id_cols = ["customerID"] if "customerID" in df.columns else []

    excluded = set(id_cols)
    if target_col and target_col in df.columns:
        excluded.add(target_col)

    numerical_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col].dtype):
            numerical_cols.append(str(col))
        else:
            categorical_cols.append(str(col))

    target_list = [target_col] if target_col and target_col in df.columns else []

    feature_types: dict[str, list[str]] = {
        "numerical": numerical_cols,
        "categorical": categorical_cols,
        "identifier": [c for c in id_cols if c in df.columns],
        "target": target_list,
    }

    logger.debug(
        "Identified feature types: %d numerical, %d categorical, %d id, %d target",
        len(numerical_cols),
        len(categorical_cols),
        len(feature_types["identifier"]),
        len(target_list),
    )
    return feature_types


def check_suspicious_values(df: pd.DataFrame) -> dict[str, Any]:
    """Scan columns for hidden missing values such as whitespace strings or placeholders.

    In datasets like Telco Customer Churn, features like 'TotalCharges' contain
    whitespace strings (" ") for zero-tenure accounts which bypass standard NaN checks.

    Args:
        df: Input pandas DataFrame.

    Returns:
        Dict[str, Any]: Summary of suspicious values detected per column.

    Raises:
        TypeError: If df is not a pandas DataFrame.
    """
    _validate_is_dataframe(df)

    suspicious_by_column: dict[str, dict[str, Any]] = {}
    known_placeholders = {"?", "na", "n/a", "null", "none", "nan"}

    for col in df.columns:
        series = df[col]
        # Only inspect non-numeric or object-like series
        if series.dtype == object or isinstance(series.dtype, pd.StringDtype):
            # Check for pure whitespace or empty strings
            whitespace_mask = series.apply(
                lambda val: isinstance(val, str) and val.strip() == ""
            )
            whitespace_count = int(whitespace_mask.sum())

            # Check for string placeholders
            placeholder_mask = series.apply(
                lambda val: (
                    isinstance(val, str) and val.strip().lower() in known_placeholders
                )
            )
            placeholder_count = int(placeholder_mask.sum())

            total_suspicious = whitespace_count + placeholder_count
            if total_suspicious > 0:
                indices = list(df[whitespace_mask | placeholder_mask].index[:10])
                pct = round((total_suspicious / len(df)) * 100, 4)
                suspicious_by_column[str(col)] = {
                    "total_suspicious": total_suspicious,
                    "whitespace_count": whitespace_count,
                    "placeholder_count": placeholder_count,
                    "percentage": pct,
                    "sample_indices": [int(i) for i in indices],
                }
                logger.warning(
                    "Column '%s' contains %d suspicious values (%.2f%%)",
                    col,
                    total_suspicious,
                    pct,
                )

    return {
        "columns_with_suspicious_values": suspicious_by_column,
        "has_suspicious_values": len(suspicious_by_column) > 0,
        "total_affected_columns": len(suspicious_by_column),
    }


def generate_data_quality_report(
    df: pd.DataFrame, target_col: str = "Churn"
) -> dict[str, Any]:
    """Generate a comprehensive data quality and validation report.

    Args:
        df: Input pandas DataFrame.
        target_col: Name of the target variable. Defaults to "Churn".

    Returns:
        Dict[str, Any]: Aggregated dictionary containing complete health check results.

    Raises:
        TypeError: If df is not a pandas DataFrame.
    """
    _validate_is_dataframe(df)

    shape = get_dataset_shape(df)
    duplicates = check_duplicates(df)
    missing = check_missing_values(df).to_dict()
    column_info = get_column_info(df)
    feature_types = identify_feature_types(df, target_col=target_col)
    suspicious = check_suspicious_values(df)

    target_info: dict[str, Any] | None = None
    if target_col in df.columns:
        target_info = validate_target(df, target_col=target_col)

    report: dict[str, Any] = {
        "rows": shape[0],
        "columns": shape[1],
        "duplicate_rows": duplicates,
        "standard_null_counts": {str(k): int(v) for k, v in missing.items()},
        "total_standard_nulls": sum(int(v) for v in missing.values()),
        "suspicious_values": suspicious,
        "feature_types": feature_types,
        "memory_usage_mb": column_info["memory_usage_mb"],
        "target_validation": target_info,
    }

    logger.info("Data quality report generated successfully.")
    return report
