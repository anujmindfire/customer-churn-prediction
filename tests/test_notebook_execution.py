"""Unit integration test for preprocessing workflow using synthetic test fixture."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.features.preprocessing import (
    build_preprocessing_pipeline,
    clean_dataset,
    fit_preprocessor,
    identify_feature_types,
    load_preprocessor,
    save_preprocessor,
    save_processed_datasets,
    separate_features_target,
    split_data,
    transform_data,
)


def test_preprocessing_workflow_end_to_end(tmp_path):
    """Verify complete preprocessing workflow on synthetic dataset fixture."""
    np.random.seed(42)
    n = 100
    df_raw = pd.DataFrame(
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

    df_clean = clean_dataset(df_raw)
    assert df_clean["TotalCharges"].isnull().sum() == 0

    X, y = separate_features_target(df_clean)
    assert "customerID" not in X.columns
    assert "Churn" not in X.columns

    num_cols, cat_cols = identify_feature_types(X)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, random_state=42)

    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)
    fit_preprocessor(preprocessor, X_train)

    X_train_proc = transform_data(preprocessor, X_train, as_dataframe=True)
    X_val_proc = transform_data(preprocessor, X_val, as_dataframe=True)
    X_test_proc = transform_data(preprocessor, X_test, as_dataframe=True)

    assert len(X_train_proc) == 70
    assert len(X_val_proc) == 15
    assert len(X_test_proc) == 15

    artifact_path = tmp_path / "preprocessor.joblib"
    save_preprocessor(preprocessor, artifact_path)
    loaded = load_preprocessor(artifact_path)
    assert isinstance(loaded, ColumnTransformer)

    saved_datasets = save_processed_datasets(
        X_train_proc,
        y_train,
        X_val_proc,
        y_val,
        X_test_proc,
        y_test,
        output_dir=tmp_path / "processed",
    )
    for path in saved_datasets.values():
        assert path.exists()
