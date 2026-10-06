"""Evaluation visualization utilities for Customer Churn Prediction."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str] | None = None,
    title: str = "Confusion Matrix — Logistic Regression Baseline",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot an annotated confusion matrix heatmap for binary classification.

    Displays cell names (TN, FP, FN, TP), absolute observation counts,
    and proportion of total validation samples in each cell.

    Args:
        cm: 2x2 integer confusion matrix [[TN, FP], [FN, TP]].
        labels: Class display labels. Defaults to ['No Churn (0)', 'Churn (1)'].
        title: Title for the plot.
        save_path: Optional path to save the generated figure.

    Returns:
        plt.Figure: Matplotlib Figure object.

    Raises:
        ValueError: If confusion matrix shape is not (2, 2).
    """
    if cm.shape != (2, 2):
        raise ValueError(f"Expected 2x2 confusion matrix, got shape {cm.shape}")

    class_names = labels if labels is not None else ["No Churn (0)", "Churn (1)"]
    total_samples = float(cm.sum())

    # Quadrant descriptive annotations
    quadrant_labels = [
        ["True Neg (TN)", "False Pos (FP)"],
        ["False Neg (FN)", "True Pos (TP)"],
    ]

    annot_matrix = np.empty((2, 2), dtype=object)
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = (count / total_samples) * 100 if total_samples > 0 else 0.0
            annot_matrix[i, j] = f"{quadrant_labels[i][j]}\n{count:,}\n({pct:.1f}%)"

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        cm,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"fontsize": 11, "fontweight": "bold"},
        ax=ax,
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="semibold", labelpad=10)
    ax.set_ylabel("Actual Class", fontsize=11, fontweight="semibold", labelpad=10)

    plt.tight_layout()

    if save_path:
        out = Path(save_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        logger.info("Saved confusion matrix heatmap to %s", out)

    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric_cols: list[str] | None = None,
    title: str = "Model Performance Comparison on Validation Set",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot a grouped bar chart comparing multiple classification models across metrics.

    Args:
        comparison_df: DataFrame containing model metrics with a 'Model' column.
        metric_cols: List of metric columns to plot. Defaults to
            ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'].
        title: Title for the chart.
        save_path: Optional destination path to save the generated plot.

    Returns:
        plt.Figure: Matplotlib Figure object.

    Raises:
        ValueError: If 'Model' column is missing or if no valid metric columns exist.
    """
    if "Model" not in comparison_df.columns:
        raise ValueError("comparison_df must contain a 'Model' column.")

    default_cols = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    target_cols = (
        metric_cols
        if metric_cols is not None
        else [c for c in default_cols if c in comparison_df.columns]
    )

    if not target_cols:
        raise ValueError("No valid metric columns found in comparison_df.")

    melted = comparison_df.melt(
        id_vars=["Model"],
        value_vars=target_cols,
        var_name="Metric",
        value_name="Score",
    )

    palette = ["#2563eb", "#ea580c", "#16a34a", "#9333ea", "#0891b2"]
    num_models = len(comparison_df["Model"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=melted,
        x="Metric",
        y="Score",
        hue="Model",
        palette=palette[:num_models],
        ax=ax,
    )

    ax.set_ylim(0.0, 1.05)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Evaluation Metric", fontsize=11, fontweight="semibold", labelpad=8)
    ax.set_ylabel("Validation Score", fontsize=11, fontweight="semibold", labelpad=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for p in ax.patches:
        h = float(getattr(p, "get_height", lambda: 0.0)())
        if not np.isnan(h) and h > 0:
            x = float(getattr(p, "get_x", lambda: 0.0)())
            w = float(getattr(p, "get_width", lambda: 0.0)())
            ax.annotate(
                f"{h:.3f}",
                (x + w / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                xytext=(0, 2),
                textcoords="offset points",
            )

    ax.legend(title="Model", frameon=True, loc="upper right")
    plt.tight_layout()

    if save_path:
        out = Path(save_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        logger.info("Saved model comparison plot to %s", out)

    return fig


def plot_feature_importance(
    importances: pd.Series,
    top_n: int = 15,
    title: str = "Top Feature Importances — Random Forest",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot a horizontal bar chart displaying top feature importances.

    Args:
        importances: pandas Series of feature importances indexed by feature name.
        top_n: Number of top features to display. Defaults to 15.
        title: Title for the chart.
        save_path: Optional destination path to save the generated plot.

    Returns:
        plt.Figure: Matplotlib Figure object.

    Raises:
        ValueError: If importances series is empty.
    """
    if len(importances) == 0:
        raise ValueError("Cannot plot feature importance for empty series.")

    top = importances.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, max(5.0, len(top) * 0.38)))
    bars = ax.barh(
        top.index.tolist(),
        top.to_numpy(),
        color="#2563eb",
        edgecolor="#1d4ed8",
        alpha=0.85,
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel(
        "Gini Feature Importance", fontsize=11, fontweight="semibold", labelpad=8
    )
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    max_val = float(top.max()) if len(top) > 0 else 0.1
    ax.set_xlim(0, max_val * 1.25)

    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            f"{width:.4f} ({width * 100:.1f}%)",
            (width, bar.get_y() + bar.get_height() / 2.0),
            ha="left",
            va="center",
            fontsize=9,
            fontweight="semibold",
            xytext=(5, 0),
            textcoords="offset points",
        )

    ax.text(
        0.0,
        -0.12,
        "Note: Feature importance indicates predictive utility in tree splits, not causal determination.",
        transform=ax.transAxes,
        fontsize=9,
        fontstyle="italic",
        color="#4b5563",
    )

    plt.tight_layout()

    if save_path:
        out = Path(save_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        logger.info("Saved feature importance plot to %s", out)

    return fig
