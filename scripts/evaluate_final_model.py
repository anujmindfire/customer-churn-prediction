"""Script to execute final model selection, save final_model.joblib, and perform unbiased evaluation on test.csv."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    FINAL_CANDIDATE_MODEL_PATH,
    FINAL_COMPARISON_FIGURE_PATH,
    FINAL_CONFUSION_MATRIX_PATH,
    FINAL_MODEL_PATH,
    PREPROCESSOR_PATH,
    TARGET_COL,
    TEST_DATA_PATH,
    VAL_DATA_PATH,
)
from src.evaluation.metrics import (
    calculate_classification_metrics,
    compare_models,
    compute_confusion_matrix,
    get_classification_report,
    get_confusion_matrix_breakdown,
)
from src.evaluation.plots import (
    plot_confusion_matrix,
    plot_model_comparison,
)
from src.models.model import (
    load_model,
    predict_churn,
    predict_churn_probabilities,
    save_model,
)
from src.predict import predict_churn as predict_customer_churn


def main() -> None:
    print("=== Step 8: Final Model Selection & Test Evaluation ===")

    # 1. Final Model Selection (based strictly on Validation performance from Step 7)
    print("\n1. Loading selected final candidate model (Tuned Random Forest)...")
    selected_model = load_model(FINAL_CANDIDATE_MODEL_PATH)
    print(f"Loaded selected model: {type(selected_model).__name__}")
    print(f"Model parameters: {selected_model.get_params()}")

    # Save as final_model.joblib
    save_model(selected_model, FINAL_MODEL_PATH)
    print(f"Saved final production-ready model artifact to: {FINAL_MODEL_PATH}")

    # 2. Validation performance review (for baseline comparison)
    val_df = pd.read_csv(VAL_DATA_PATH)
    X_val = val_df.drop(columns=[TARGET_COL])
    y_val = val_df[TARGET_COL]

    p_val = predict_churn_probabilities(selected_model, X_val)
    preds_val = predict_churn(selected_model, X_val)
    val_metrics = calculate_classification_metrics(y_val, preds_val, p_val)
    print(f"\nSelected Model Validation Metrics: {val_metrics}")

    # 3. Unlocking and evaluating on the UNTOUCHED Test Dataset (First and Only time!)
    print(f"\n2. Unlocking test dataset from: {TEST_DATA_PATH}")
    test_df = pd.read_csv(TEST_DATA_PATH)
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(
        f"Test churn distribution: Churn=1: {(y_test == 1).sum():,}, Churn=0: {(y_test == 0).sum():,} ({y_test.mean() * 100:.2f}% churn rate)"
    )

    p_test = predict_churn_probabilities(selected_model, X_test)
    preds_test = predict_churn(selected_model, X_test)
    test_metrics = calculate_classification_metrics(y_test, preds_test, p_test)
    cm_test = compute_confusion_matrix(y_test, preds_test)
    breakdown_test = get_confusion_matrix_breakdown(y_test, preds_test)

    print("\n=== FINAL TEST EVALUATION RESULTS ===")
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1 Score:  {test_metrics['f1']:.4f}")
    print(f"ROC-AUC:   {test_metrics.get('roc_auc', float('nan')):.4f}")
    print(f"Confusion Matrix:\n{cm_test}")
    print(
        f"Breakdown: TN={breakdown_test['tn']}, FP={breakdown_test['fp']}, FN={breakdown_test['fn']}, TP={breakdown_test['tp']}"
    )

    print("\nClassification Report (Test Partition):")
    report = get_classification_report(
        y_test, preds_test, target_names=["No Churn", "Churn"]
    )
    print(report)

    # 4. Final Results Comparison Table
    results_summary = {
        "Final Model (Validation Set)": val_metrics,
        "Final Model (Test Set — Unbiased)": test_metrics,
    }
    final_table = compare_models(results_summary)
    print("\n=== Final Model Performance Comparison (Val vs. Test) ===")
    print(final_table.to_string(index=False))

    # 5. Generate and Save Figures
    print("\nGenerating final figures...")
    fig_cm = plot_confusion_matrix(
        cm_test,
        title="Final Test Confusion Matrix — Tuned Random Forest",
        save_path=FINAL_CONFUSION_MATRIX_PATH,
    )
    plt.close(fig_cm)
    print(f"Saved final confusion matrix to: {FINAL_CONFUSION_MATRIX_PATH}")

    fig_comp = plot_model_comparison(
        final_table,
        title="Final Generalization Evaluation: Validation vs. Unbiased Test Set",
        save_path=FINAL_COMPARISON_FIGURE_PATH,
    )
    plt.close(fig_comp)
    print(f"Saved final comparison plot to: {FINAL_COMPARISON_FIGURE_PATH}")

    # 6. Verify High-Level predict_churn() function on raw customer data
    print("\n6. Verifying high-level predict_churn() on sample raw customer data...")
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }

    result = predict_customer_churn(
        sample_customer,
        model_path=FINAL_MODEL_PATH,
        preprocessor_path=PREPROCESSOR_PATH,
    )
    print(f"Sample prediction result: {result}")
    print(
        f"Prediction: {result.churn_prediction}, Probability: {result.churn_probability:.4f}"
    )
    assert result.churn_prediction in [0, 1]
    assert 0.0 <= result.churn_probability <= 1.0

    print("\nStep 8 Final Model Selection and Test Evaluation completed successfully!")


if __name__ == "__main__":
    main()
