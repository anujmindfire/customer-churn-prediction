"""Evaluation visualization utilities for Customer Churn Prediction."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
