"""Data loading and validation module for Customer Churn Prediction."""

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

__all__ = [
    "check_duplicates",
    "check_missing_values",
    "check_suspicious_values",
    "generate_data_quality_report",
    "get_column_info",
    "get_dataset_shape",
    "identify_feature_types",
    "load_dataset",
    "validate_target",
]
