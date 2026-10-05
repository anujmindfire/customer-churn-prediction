"""Compatibility alias for src.churn_prediction.features.preprocessing."""

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

__all__ = [
    "build_preprocessing_pipeline",
    "clean_dataset",
    "fit_preprocessor",
    "get_feature_names",
    "identify_feature_types",
    "load_preprocessor",
    "prepare_target",
    "save_preprocessor",
    "save_processed_datasets",
    "separate_features_target",
    "split_data",
    "transform_data",
]
