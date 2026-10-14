"""Script to run hyperparameter tuning, class balancing, threshold optimization, and export improved models and figures."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    FINAL_CANDIDATE_MODEL_PATH,
    IMPROVED_MODEL_CM_PATH,
    MODEL_IMPROVEMENT_FIGURE_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TARGET_COL,
    THRESHOLD_TUNING_FIGURE_PATH,
    TUNED_LOGISTIC_REGRESSION_MODEL_PATH,
    TUNED_RANDOM_FOREST_MODEL_PATH,
)
from src.evaluation.metrics import (
    calculate_classification_metrics,
    compare_models,
    compute_confusion_matrix,
    get_confusion_matrix_breakdown,
)
from src.evaluation.plots import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_threshold_tuning_curve,
)
from src.models.model import (
    create_logistic_regression_model,
    predict_churn,
    predict_churn_probabilities,
    save_model,
    train_model,
)
from src.models.tuning import (
    find_optimal_threshold_cv,
    predict_with_threshold,
    tune_logistic_regression,
    tune_random_forest,
)


def main() -> None:
    print("Loading preprocessed training and validation data...")
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_val = val_df.drop(columns=[TARGET_COL])
    y_val = val_df[TARGET_COL]

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

    # 1. Baseline Logistic Regression (Step 5/6 benchmark)
    print("\n--- Evaluating Baseline Logistic Regression ---")
    base_lr = create_logistic_regression_model(random_state=RANDOM_STATE)
    train_model(base_lr, X_train, y_train)
    p_val_base = predict_churn_probabilities(base_lr, X_val)
    preds_base = predict_churn(base_lr, X_val)
    metrics_base = calculate_classification_metrics(y_val, preds_base, p_val_base)
    print(f"Baseline LR Metrics: {metrics_base}")

    # 2. Technique 1 & 2: Hyperparameter Tuning + Class Balancing for Logistic Regression
    print("\n--- Tuning Logistic Regression with Class Balancing (GridSearchCV) ---")
    lr_grid = {
        "C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
        "class_weight": [None, "balanced", {0: 1.0, 1: 1.5}, {0: 1.0, 1: 2.0}],
        "solver": ["lbfgs"],
        "max_iter": [1000],
        "random_state": [RANDOM_STATE],
    }
    lr_search = tune_logistic_regression(
        X_train=X_train,
        y_train=y_train,
        param_grid=lr_grid,
        cv_splits=5,
        scoring="f1",
        random_state=RANDOM_STATE,
    )
    tuned_lr = lr_search.best_estimator_
    print(f"Best LR Parameters: {lr_search.best_params_}")
    print(f"Best LR CV F1 Score: {lr_search.best_score_:.4f}")

    p_val_tuned_lr = predict_churn_probabilities(tuned_lr, X_val)
    preds_tuned_lr = predict_churn(tuned_lr, X_val)
    metrics_tuned_lr = calculate_classification_metrics(
        y_val, preds_tuned_lr, p_val_tuned_lr
    )
    print(f"Tuned LR Validation Metrics: {metrics_tuned_lr}")

    # 3. Technique 1 & 2: Hyperparameter Tuning + Class Balancing for Random Forest
    print("\n--- Tuning Random Forest with Regularization & Class Balancing ---")
    rf_grid = {
        "n_estimators": [100, 200],
        "max_depth": [6, 8, 10, 12],
        "min_samples_split": [5, 10, 20],
        "min_samples_leaf": [2, 4, 8],
        "class_weight": [None, "balanced"],
        "random_state": [RANDOM_STATE],
    }
    rf_search = tune_random_forest(
        X_train=X_train,
        y_train=y_train,
        param_grid=rf_grid,
        cv_splits=5,
        scoring="f1",
        random_state=RANDOM_STATE,
    )
    tuned_rf = rf_search.best_estimator_
    print(f"Best RF Parameters: {rf_search.best_params_}")
    print(f"Best RF CV F1 Score: {rf_search.best_score_:.4f}")

    p_val_tuned_rf = predict_churn_probabilities(tuned_rf, X_val)
    preds_tuned_rf = predict_churn(tuned_rf, X_val)
    metrics_tuned_rf = calculate_classification_metrics(
        y_val, preds_tuned_rf, p_val_tuned_rf
    )
    print(f"Tuned RF Validation Metrics: {metrics_tuned_rf}")

    # 4. Technique 3: Out-of-fold Decision Threshold Optimization (on training data only!)
    print("\n--- Finding Optimal Decision Threshold via Training Cross-Validation ---")
    opt_th, best_cv_f1, ths, f1s, precs, recs = find_optimal_threshold_cv(
        estimator=base_lr,
        X_train=X_train,
        y_train=y_train,
        cv_splits=5,
        metric="f1",
        random_state=RANDOM_STATE,
    )
    print(
        f"Optimal OOF Training Threshold: {opt_th:.3f} (OOF Peak F1: {best_cv_f1:.4f})"
    )

    preds_th_opt = predict_with_threshold(p_val_base, threshold=opt_th)
    metrics_th_opt = calculate_classification_metrics(y_val, preds_th_opt, p_val_base)
    print(
        f"Baseline LR + Optimal Threshold ({opt_th:.3f}) Val Metrics: {metrics_th_opt}"
    )

    # 5. Consolidation and Comparative Summary
    all_metrics = {
        "Logistic Regression (Baseline)": metrics_base,
        "LR + Optimal Threshold": metrics_th_opt,
        "Tuned Random Forest": metrics_tuned_rf,
        "Tuned Logistic Regression": metrics_tuned_lr,
    }
    comp_df = compare_models(all_metrics)
    print("\n=== Comprehensive Model Improvement Comparison ===")
    print(comp_df.to_string(index=False))

    # Confusion matrix breakdown for champion model (Tuned RF)
    cm_improved = compute_confusion_matrix(y_val, preds_tuned_rf)
    breakdown_improved = get_confusion_matrix_breakdown(y_val, preds_tuned_rf)
    print(f"\nTuned RF Confusion Matrix Breakdown: {breakdown_improved}")

    # 6. Save Artifacts
    print("\nSaving improved model artifacts...")
    save_model(tuned_rf, FINAL_CANDIDATE_MODEL_PATH)
    save_model(tuned_lr, TUNED_LOGISTIC_REGRESSION_MODEL_PATH)
    save_model(tuned_rf, TUNED_RANDOM_FOREST_MODEL_PATH)
    print(f"Saved final candidate model to: {FINAL_CANDIDATE_MODEL_PATH}")

    # 7. Generate Figures
    print("Generating and saving figures...")
    fig_comp = plot_model_comparison(
        comp_df,
        title="Model Performance Improvement vs. Baseline on Validation Set",
        save_path=MODEL_IMPROVEMENT_FIGURE_PATH,
    )
    plt.close(fig_comp)

    fig_cm = plot_confusion_matrix(
        cm_improved,
        title="Validation Confusion Matrix — Tuned Random Forest (Champion Model)",
        save_path=IMPROVED_MODEL_CM_PATH,
    )
    plt.close(fig_cm)

    fig_th = plot_threshold_tuning_curve(
        thresholds=ths,
        f1_scores=f1s,
        precisions=precs,
        recalls=recs,
        optimal_threshold=opt_th,
        optimal_f1=best_cv_f1,
        title="Cross-Validation Decision Threshold Optimization Curve",
        save_path=THRESHOLD_TUNING_FIGURE_PATH,
    )
    plt.close(fig_th)

    print("Step 7 Model Improvement completed successfully!")


if __name__ == "__main__":
    main()
