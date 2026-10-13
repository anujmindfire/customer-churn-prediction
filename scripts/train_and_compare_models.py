import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import matplotlib.pyplot as plt
import pandas as pd

from src.churn_prediction.config import (
    DECISION_TREE_CM_PATH,
    DECISION_TREE_MODEL_PATH,
    FEATURE_IMPORTANCE_FIGURE_PATH,
    LOGISTIC_REGRESSION_CM_PATH,
    MODEL_COMPARISON_FIGURE_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_FOREST_CM_PATH,
    RANDOM_FOREST_MODEL_PATH,
    RANDOM_STATE,
    TARGET_COL,
)
from src.churn_prediction.evaluation.metrics import (
    calculate_classification_metrics,
    compare_models,
    compute_confusion_matrix,
    get_confusion_matrix_breakdown,
)
from src.churn_prediction.evaluation.plots import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
)
from src.churn_prediction.models.model import (
    create_all_candidate_models,
    get_model_feature_importances,
    predict_churn,
    predict_churn_probabilities,
    save_model,
    train_model,
)


def main() -> None:
    print("Loading preprocessed training and validation data...")
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_val = val_df.drop(columns=[TARGET_COL])
    y_val = val_df[TARGET_COL]
    feature_names = list(X_train.columns)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

    models = create_all_candidate_models(random_state=RANDOM_STATE)
    metrics_summary = {}

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        train_model(model, X_train, y_train)

        # Train metrics (for overfitting check)
        train_preds = predict_churn(model, X_train)
        train_metrics = calculate_classification_metrics(y_train, train_preds)
        print(f"{name} Train Metrics: {train_metrics}")

        # Val metrics
        val_preds = predict_churn(model, X_val)
        val_probs = predict_churn_probabilities(model, X_val)
        val_metrics = calculate_classification_metrics(y_val, val_preds, val_probs)
        metrics_summary[name] = val_metrics
        print(f"{name} Val Metrics: {val_metrics}")

        # Confusion Matrix
        cm = compute_confusion_matrix(y_val, val_preds)
        breakdown = get_confusion_matrix_breakdown(y_val, val_preds)
        print(f"{name} CM Breakdown: {breakdown}")

        # Save model
        if name == "Decision Tree":
            save_model(model, DECISION_TREE_MODEL_PATH)
            fig_cm = plot_confusion_matrix(
                cm,
                title="Confusion Matrix — Decision Tree Baseline",
                save_path=DECISION_TREE_CM_PATH,
            )
            plt.close(fig_cm)
        elif name == "Random Forest":
            save_model(model, RANDOM_FOREST_MODEL_PATH)
            fig_cm = plot_confusion_matrix(
                cm,
                title="Confusion Matrix — Random Forest Baseline",
                save_path=RANDOM_FOREST_CM_PATH,
            )
            plt.close(fig_cm)
            # Feature Importance
            importances = get_model_feature_importances(model, feature_names)
            fig_fi = plot_feature_importance(
                importances,
                top_n=15,
                title="Top Feature Importances — Random Forest (Validation Comparison)",
                save_path=FEATURE_IMPORTANCE_FIGURE_PATH,
            )
            plt.close(fig_fi)
        elif name == "Logistic Regression":
            fig_cm = plot_confusion_matrix(
                cm,
                title="Confusion Matrix — Logistic Regression Baseline",
                save_path=LOGISTIC_REGRESSION_CM_PATH,
            )
            plt.close(fig_cm)

    comparison_df = compare_models(metrics_summary)
    print("\n--- Model Comparison Summary ---")
    print(comparison_df.to_string(index=False))

    fig_comp = plot_model_comparison(
        comparison_df,
        save_path=MODEL_COMPARISON_FIGURE_PATH,
    )
    plt.close(fig_comp)
    print("\nAll candidate models and comparison figures generated successfully.")


if __name__ == "__main__":
    main()
