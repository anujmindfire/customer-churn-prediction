"""Plotting and visualization utilities for Customer Churn Prediction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]
from matplotlib.axes import Axes
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

DEFAULT_PALETTE = {"No": "#2b5c8f", "Yes": "#d95f02"}
DEFAULT_COLORS = ["#2b5c8f", "#d95f02", "#2ca02c", "#9467bd", "#8c564b"]


def set_plot_style(
    style: str = "whitegrid",
    palette: str | list[str] | dict[str, str] | None = None,
    font_scale: float = 1.05,
) -> None:
    """Configure global matplotlib and seaborn aesthetic styling.

    Args:
        style: Seaborn plot style name (e.g. 'whitegrid', 'white', 'darkgrid').
        palette: Default color palette to configure.
        font_scale: Scaling factor for font sizes.
    """
    sns.set_theme(style=style, font_scale=font_scale)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.labelweight"] = "normal"
    plt.rcParams["figure.titlesize"] = 15
    plt.rcParams["figure.titleweight"] = "bold"
    if palette is not None and isinstance(palette, (str, list)):
        sns.set_palette(palette)


def _validate_is_dataframe(df: pd.DataFrame) -> None:
    """Validate that the input object is a pandas DataFrame.

    Args:
        df: Input object to validate.

    Raises:
        TypeError: If df is not an instance of pd.DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")


def _save_figure(fig: Figure, save_path: str | Path | None) -> None:
    """Save matplotlib figure to disk if save_path is provided.

    Args:
        fig: Matplotlib Figure instance.
        save_path: Filepath where the plot should be saved.
    """
    if save_path is None:
        return

    path = Path(save_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    logger.info("Saved figure to %s", path)


def plot_churn_distribution(
    df: pd.DataFrame,
    target_col: str = "Churn",
    palette: dict[str, str] | list[str] | None = None,
    title: str = "Customer Churn Distribution",
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Plot frequency and percentage distribution of the target churn column.

    Displays count bar chart annotated with total frequency and class percentage.

    Args:
        df: Input pandas DataFrame containing customer data.
        target_col: Name of the churn target column. Defaults to "Churn".
        palette: Color palette mapping or list. Defaults to standard theme.
        title: Title of the visualization.
        save_path: Optional path to save the generated figure.

    Returns:
        tuple[Figure, Axes]: Generated matplotlib Figure and Axes.

    Raises:
        TypeError: If df is not a pandas DataFrame.
        KeyError: If target_col is missing from df.
        ValueError: If df is empty.
    """
    _validate_is_dataframe(df)
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")
    if df.empty:
        raise ValueError("Cannot plot churn distribution for an empty DataFrame.")

    set_plot_style()
    colors = palette if palette is not None else DEFAULT_PALETTE

    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df[target_col].value_counts(dropna=False)
    total = len(df)

    order = list(counts.index)
    plot_palette = (
        colors
        if isinstance(colors, dict)
        else {cat: colors[i % len(colors)] for i, cat in enumerate(order)}
    )

    bars = sns.barplot(
        x=order,
        y=counts.values,
        palette=plot_palette,
        hue=order,
        legend=False,
        ax=ax,
        edgecolor="black",
        linewidth=0.8,
    )

    max_count = counts.max()
    ax.set_ylim(0, max_count * 1.15)
    ax.set_title(title, pad=15)
    ax.set_xlabel(f"{target_col} Status")
    ax.set_ylabel("Number of Customers")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, count in zip(bars.patches, counts.values, strict=False):
        pct = (count / total) * 100
        text = f"{count:,}\n({pct:.1f}%)"
        ax.annotate(
            text,
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
            fontweight="bold",
            fontsize=10,
        )

    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig, ax


def plot_numerical_distribution(
    df: pd.DataFrame,
    column: str,
    target_col: str | None = "Churn",
    bins: int = 30,
    palette: dict[str, str] | list[str] | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> tuple[Figure, Any]:
    """Plot distribution of a numerical column via histogram/KDE and boxplot.

    When target_col is provided, displays distributions and box plots conditioned
    on the target class to highlight differences between churners and non-churners.

    Args:
        df: Input pandas DataFrame.
        column: Numerical column name to visualize.
        target_col: Target column name for grouped analysis. Defaults to "Churn".
        bins: Number of histogram bins. Defaults to 30.
        palette: Color palette mapping or list.
        title: Main supertitle for the visualization.
        save_path: Optional path to save the generated figure.

    Returns:
        tuple[Figure, Any]: Matplotlib Figure and array of Axes.

    Raises:
        TypeError: If df is not a pandas DataFrame.
        KeyError: If column or target_col is not in df.
        ValueError: If column contains no valid numeric values or df is empty.
    """
    _validate_is_dataframe(df)
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")
    if target_col is not None and target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")
    if df.empty:
        raise ValueError("Cannot plot distribution for an empty DataFrame.")

    # Prepare numeric series
    series = pd.to_numeric(df[column], errors="coerce")
    if series.dropna().empty:
        raise ValueError(f"Column '{column}' contains no numeric values.")

    df_clean = df.copy()
    df_clean[column] = series
    df_clean = df_clean.dropna(subset=[column])

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_hist, ax_box = axes[0], axes[1]

    colors = palette if palette is not None else DEFAULT_PALETTE

    if target_col is not None:
        sns.histplot(
            data=df_clean,
            x=column,
            hue=target_col,
            kde=True,
            bins=bins,
            palette=colors,
            element="step",
            common_norm=False,
            ax=ax_hist,
            alpha=0.5,
        )
        ax_hist.set_title(f"{column} Histogram & KDE by {target_col}")
        ax_hist.set_ylabel("Density / Count")

        sns.boxplot(
            data=df_clean,
            x=target_col,
            y=column,
            hue=target_col,
            palette=colors,
            legend=False,
            notch=True,
            ax=ax_box,
        )
        ax_box.set_title(f"{column} Box Plot by {target_col}")
        ax_box.set_xlabel(f"{target_col} Status")

        # Display median stats
        medians = df_clean.groupby(target_col)[column].median()
        stats_text = "Medians:\n" + "\n".join(
            f"{cat}: {val:.1f}" for cat, val in medians.items()
        )
        ax_box.text(
            0.95,
            0.95,
            stats_text,
            transform=ax_box.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.8},
            fontsize=9,
        )
    else:
        sns.histplot(
            data=df_clean,
            x=column,
            kde=True,
            bins=bins,
            color="#2b5c8f",
            ax=ax_hist,
        )
        ax_hist.set_title(f"{column} Histogram & KDE")
        ax_hist.set_ylabel("Count")

        sns.boxplot(
            data=df_clean,
            y=column,
            color="#2b5c8f",
            notch=True,
            ax=ax_box,
        )
        ax_box.set_title(f"{column} Box Plot")

    ax_hist.grid(True, linestyle="--", alpha=0.5)
    ax_box.grid(True, linestyle="--", alpha=0.5)

    main_title = title or f"Distribution Analysis of {column}"
    fig.suptitle(main_title, y=1.02)

    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig, axes


def plot_categorical_churn_rates(
    df: pd.DataFrame,
    categorical_cols: list[str] | str,
    target_col: str = "Churn",
    positive_val: Any = "Yes",
    top_n: int | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> tuple[Figure, Any]:
    """Plot churn rates across categories with overall churn reference line.

    Calculates the proportion of positive churn instances for each category and
    displays bar charts annotated with exact percentages and sample sizes.

    Args:
        df: Input pandas DataFrame.
        categorical_cols: Single column name or list of categorical column names.
        target_col: Target column name. Defaults to "Churn".
        positive_val: Target positive class label. Defaults to "Yes".
        top_n: Optional max number of categories to display per feature.
        title: Optional custom figure title.
        save_path: Optional path to save the generated figure.

    Returns:
        tuple[Figure, Any]: Matplotlib Figure and Axes (or array of Axes).

    Raises:
        TypeError: If df is not a pandas DataFrame.
        KeyError: If target_col or any categorical column is missing from df.
        ValueError: If df is empty or positive_val not in target_col.
    """
    _validate_is_dataframe(df)
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")
    if df.empty:
        raise ValueError("Cannot compute churn rates for an empty DataFrame.")

    cols = (
        [categorical_cols]
        if isinstance(categorical_cols, str)
        else list(categorical_cols)
    )
    for col in cols:
        if col not in df.columns:
            raise KeyError(f"Categorical column '{col}' not found in DataFrame.")

    set_plot_style()
    overall_churn_rate = float((df[target_col] == positive_val).mean() * 100)

    n_plots = len(cols)
    if n_plots == 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        axes_list = [ax]
    else:
        ncols = 2 if n_plots <= 4 else 3
        nrows = int(np.ceil(n_plots / ncols))
        fig, grid_axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.5 * nrows))
        axes_list = list(np.array(grid_axes).ravel())

    for idx, col in enumerate(cols):
        ax = axes_list[idx]
        group = df.groupby(col, observed=False)
        total_counts = group.size()
        churn_counts = (
            df[df[target_col] == positive_val]
            .groupby(col, observed=False)
            .size()
            .reindex(total_counts.index, fill_value=0)
        )

        rates = (churn_counts / total_counts) * 100
        summary_df = pd.DataFrame({"rate": rates, "count": total_counts}).sort_values(
            by="rate", ascending=False
        )

        if top_n is not None and top_n > 0:
            summary_df = summary_df.head(top_n)

        categories = list(summary_df.index.astype(str))
        rates_vals = list(summary_df["rate"].values)
        counts_vals = list(summary_df["count"].values)

        # Highlight categories with churn higher than baseline
        bar_colors = [
            "#d95f02" if r > overall_churn_rate else "#2b5c8f" for r in rates_vals
        ]

        bars = ax.bar(
            categories,
            rates_vals,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.8,
        )

        # Baseline reference line
        ax.axhline(
            overall_churn_rate,
            color="firebrick",
            linestyle="--",
            linewidth=1.2,
            label=f"Baseline ({overall_churn_rate:.1f}%)",
        )

        max_rate = max(rates_vals) if rates_vals else 0
        ax.set_ylim(0, max(max_rate * 1.25, overall_churn_rate * 1.25, 20))
        ax.set_title(f"Churn Rate by {col}", pad=10)
        ax.set_ylabel("Churn Rate (%)")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.tick_params(axis="x", rotation=25)

        for bar, rate, count in zip(bars, rates_vals, counts_vals, strict=False):
            ax.annotate(
                f"{rate:.1f}%\n(n={count:,})",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
            )

        if idx == 0:
            ax.legend(loc="upper right", frameon=True)

    # Hide unused subplots
    for unused_idx in range(n_plots, len(axes_list)):
        axes_list[unused_idx].set_visible(False)

    main_title = title or f"Churn Rate by Category (Overall: {overall_churn_rate:.1f}%)"
    fig.suptitle(main_title, y=1.02)
    plt.tight_layout()
    _save_figure(fig, save_path)

    return fig, ax if n_plots == 1 else grid_axes


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    cmap: str = "coolwarm",
    annot: bool = True,
    title: str = "Numerical Feature Correlation Matrix",
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Plot correlation heatmap for numerical features with triangular mask.

    Automatically identifies numeric columns or utilizes user-specified columns,
    coercing string numerics (e.g. TotalCharges) if required.

    Args:
        df: Input pandas DataFrame.
        columns: Specific list of numeric columns. If None, auto-selects numeric.
        method: Correlation method ('pearson', 'spearman', 'kendall').
        cmap: Colormap name for seaborn heatmap. Defaults to 'coolwarm'.
        annot: Whether to annotate cells with numerical correlation values.
        title: Title of the heatmap visualization.
        save_path: Optional path to save the generated figure.

    Returns:
        tuple[Figure, Axes]: Matplotlib Figure and Axes.

    Raises:
        TypeError: If df is not a pandas DataFrame.
        ValueError: If fewer than 2 numeric columns are available.
        KeyError: If specified columns are not found in df.
    """
    _validate_is_dataframe(df)

    if columns is not None:
        for col in columns:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")
        sub_df = df[columns].copy()
    else:
        # Check all columns, attempting coercion for potential numeric strings (e.g. TotalCharges)
        candidate_cols: list[str] = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col].dtype):
                candidate_cols.append(str(col))
            else:
                # Test if column is mostly numeric strings
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.dropna().shape[0] > len(df) * 0.9:
                    candidate_cols.append(str(col))
        sub_df = df[candidate_cols].copy()

    # Ensure all columns in sub_df are numeric
    for col in sub_df.columns:
        sub_df[col] = pd.to_numeric(sub_df[col], errors="coerce")

    sub_df = sub_df.dropna(how="all")
    numeric_cols = [col for col in sub_df.columns if sub_df[col].dropna().nunique() > 1]

    if len(numeric_cols) < 2:
        raise ValueError(
            f"At least 2 numeric features required for correlation heatmap, found {len(numeric_cols)}."
        )

    corr = sub_df[numeric_cols].corr(method=method)

    set_plot_style()
    fig, ax = plt.subplots(
        figsize=(max(7, len(numeric_cols) * 1.6), max(6, len(numeric_cols) * 1.4))
    )

    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        mask=mask,
        annot=annot,
        fmt=".2f",
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        square=True,
        linewidths=0.8,
        cbar_kws={"shrink": 0.8, "label": f"{method.capitalize()} Correlation"},
        ax=ax,
    )

    ax.set_title(title, pad=15)
    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig, ax
