"""Compatibility alias for src.churn_prediction.data.validation."""

from src.churn_prediction.data.validation import (
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
    "validate_target",
]
