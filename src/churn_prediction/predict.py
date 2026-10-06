"""End-to-end inference and prediction interface for Customer Churn Prediction.

This module provides the production prediction function `predict_churn(customer_data)`
which handles raw customer records, applies the saved preprocessing pipeline, and
generates binary churn predictions and probability scores using the final model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator  # type: ignore[import-untyped]
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]

from src.churn_prediction.config import (
    FINAL_CANDIDATE_MODEL_PATH,
    FINAL_MODEL_PATH,
    PREPROCESSOR_PATH,
)

logger = logging.getLogger(__name__)

# Cached model and preprocessor instances to avoid reloading disk artifacts on every call
_CACHED_MODEL: BaseEstimator | None = None
_CACHED_PREPROCESSOR: ColumnTransformer | None = None


class ChurnPredictionResult(dict):
    """Container for single-customer churn prediction outputs.

    Supports dictionary key access, object attribute access, and tuple unpacking:
        >>> result = predict_churn(customer)
        >>> result["churn_prediction"], result["churn_probability"]
        >>> result.churn_prediction, result.churn_probability
        >>> pred, prob = result
    """

    def __init__(
        self,
        churn_prediction: int,
        churn_probability: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            churn_prediction=churn_prediction,
            churn_probability=round(float(churn_probability), 4),
            prediction=churn_prediction,
            probability=round(float(churn_probability), 4),
            **kwargs,
        )
        self.churn_prediction: int = churn_prediction
        self.churn_probability: float = round(float(churn_probability), 4)

    def __iter__(self) -> Any:
        yield self.churn_prediction
        yield self.churn_probability

    def __repr__(self) -> str:
        return (
            f"ChurnPredictionResult(churn_prediction={self.churn_prediction}, "
            f"churn_probability={self.churn_probability:.4f})"
        )


def get_inference_artifacts(
    model_path: Path | str | None = None,
    preprocessor_path: Path | str | None = None,
    force_reload: bool = False,
) -> tuple[BaseEstimator, ColumnTransformer]:
    """Retrieve or load cached model and preprocessing artifacts.

    Args:
        model_path: Path to serialized model artifact. If None, resolves to
            FINAL_MODEL_PATH, falling back to FINAL_CANDIDATE_MODEL_PATH.
        preprocessor_path: Path to preprocessor artifact. Defaults to PREPROCESSOR_PATH.
        force_reload: If True, bypasses in-memory cache and reloads from disk.

    Returns:
        tuple[BaseEstimator, ColumnTransformer]: Loaded (model, preprocessor).

    Raises:
        FileNotFoundError: If model or preprocessor file is missing.
    """
    global _CACHED_MODEL, _CACHED_PREPROCESSOR

    if force_reload or _CACHED_MODEL is None or _CACHED_PREPROCESSOR is None:
        # Determine model path
        m_path: Path
        if model_path is not None:
            m_path = Path(model_path).resolve()
        elif FINAL_MODEL_PATH.exists():
            m_path = FINAL_MODEL_PATH.resolve()
        elif FINAL_CANDIDATE_MODEL_PATH.exists():
            m_path = FINAL_CANDIDATE_MODEL_PATH.resolve()
        else:
            raise FileNotFoundError(
                f"No final model artifact found at {FINAL_MODEL_PATH} or {FINAL_CANDIDATE_MODEL_PATH}."
            )

        p_path = (
            Path(preprocessor_path).resolve()
            if preprocessor_path is not None
            else PREPROCESSOR_PATH.resolve()
        )

        if not m_path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {m_path}")
        if not p_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at: {p_path}")

        logger.info(
            "Loading inference artifacts: model=%s, preprocessor=%s", m_path, p_path
        )
        _CACHED_MODEL = joblib.load(m_path)
        _CACHED_PREPROCESSOR = joblib.load(p_path)

    return _CACHED_MODEL, _CACHED_PREPROCESSOR


def predict_churn(
    customer_data: dict[str, Any] | list[dict[str, Any]] | pd.DataFrame,
    threshold: float = 0.5,
    model: BaseEstimator | None = None,
    preprocessor: ColumnTransformer | None = None,
    model_path: Path | str | None = None,
    preprocessor_path: Path | str | None = None,
) -> ChurnPredictionResult | pd.DataFrame:
    """Predict customer churn and probability from raw customer data.

    Accepts raw demographic and service features, executes the trained
    ColumnTransformer preprocessing pipeline, and performs inference using
    the finalized machine learning model.

    Args:
        customer_data: Single customer dictionary, list of customer dictionaries,
            or a pandas DataFrame containing raw Telco churn features.
        threshold: Decision threshold for positive classification (Churn = 1). Defaults to 0.5.
        model: Optional pre-loaded estimator instance.
        preprocessor: Optional pre-loaded ColumnTransformer instance.
        model_path: Optional explicit path to model artifact on disk.
        preprocessor_path: Optional explicit path to preprocessor artifact on disk.

    Returns:
        ChurnPredictionResult: For single customer dictionary input.
            Contains keys 'churn_prediction' and 'churn_probability'.
        pd.DataFrame: For multi-customer batch DataFrame input.
            Contains original features plus columns 'churn_prediction' and 'churn_probability'.

    Raises:
        ValueError: If input is empty or missing required feature columns.
        TypeError: If customer_data is of an unsupported type.
    """
    is_single_dict = False

    if isinstance(customer_data, dict):
        is_single_dict = True
        df = pd.DataFrame([customer_data])
    elif isinstance(customer_data, list):
        if len(customer_data) == 0:
            raise ValueError("Input customer list cannot be empty.")
        df = pd.DataFrame(customer_data)
    elif isinstance(customer_data, pd.DataFrame):
        if customer_data.empty:
            raise ValueError("Input DataFrame cannot be empty.")
        df = customer_data.copy()
    else:
        raise TypeError(
            f"Expected customer_data as dict, list[dict], or pd.DataFrame; got {type(customer_data).__name__}"
        )

    # Clean TotalCharges numeric conversion if present as string
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop identifier and target columns if accidentally passed in inference data
    features_to_drop = [c for c in ["customerID", "Churn"] if c in df.columns]
    inference_features = df.drop(columns=features_to_drop)

    # Retrieve artifacts
    if model is None or preprocessor is None:
        loaded_model, loaded_preprocessor = get_inference_artifacts(
            model_path=model_path,
            preprocessor_path=preprocessor_path,
        )
        clf = model if model is not None else loaded_model
        prep = preprocessor if preprocessor is not None else loaded_preprocessor
    else:
        clf = model
        prep = preprocessor

    # Transform raw features using ColumnTransformer
    X_transformed = prep.transform(inference_features)
    feature_names = prep.get_feature_names_out()
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

    # Generate probabilities and binary predictions
    probabilities = clf.predict_proba(X_transformed_df)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    if is_single_dict or len(df) == 1:
        return ChurnPredictionResult(
            churn_prediction=int(predictions[0]),
            churn_probability=float(probabilities[0]),
        )

    # Return augmented DataFrame for batch inferences
    result_df = df.copy()
    result_df["churn_prediction"] = predictions
    result_df["churn_probability"] = np.round(probabilities, 4)
    return result_df
