"""Visualization package compatibility alias."""

from src.churn_prediction.visualization.plots import (
    plot_categorical_churn_rates,
    plot_churn_distribution,
    plot_correlation_heatmap,
    plot_numerical_distribution,
    set_plot_style,
)

__all__ = [
    "plot_categorical_churn_rates",
    "plot_churn_distribution",
    "plot_correlation_heatmap",
    "plot_numerical_distribution",
    "set_plot_style",
]
