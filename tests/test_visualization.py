"""Tests for visualization and plotting utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.churn_prediction.visualization.plots import (
    plot_categorical_churn_rates,
    plot_churn_distribution,
    plot_correlation_heatmap,
    plot_numerical_distribution,
    set_plot_style,
)
from src.visualization.plots import (
    plot_categorical_churn_rates as alias_cat,
)
from src.visualization.plots import (
    plot_churn_distribution as alias_churn,
)
from src.visualization.plots import (
    plot_correlation_heatmap as alias_corr,
)
from src.visualization.plots import (
    plot_numerical_distribution as alias_num,
)
from src.visualization.plots import (
    set_plot_style as alias_style,
)


@pytest.fixture(autouse=True)
def close_figures():
    """Ensure all matplotlib figures are closed after each test to free memory."""
    yield
    plt.close("all")


@pytest.fixture
def sample_churn_df() -> pd.DataFrame:
    """Provide a small synthetic DataFrame representative of customer churn data."""
    return pd.DataFrame(
        {
            "customerID": [f"ID_{i:03d}" for i in range(1, 11)],
            "gender": [
                "Female",
                "Male",
                "Female",
                "Male",
                "Female",
                "Male",
                "Female",
                "Male",
                "Female",
                "Male",
            ],
            "SeniorCitizen": [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            "tenure": [1, 24, 3, 48, 12, 60, 2, 36, 6, 72],
            "Contract": [
                "Month-to-month",
                "One year",
                "Month-to-month",
                "Two year",
                "Month-to-month",
                "Two year",
                "Month-to-month",
                "One year",
                "Month-to-month",
                "Two year",
            ],
            "InternetService": [
                "DSL",
                "Fiber optic",
                "DSL",
                "No",
                "Fiber optic",
                "DSL",
                "Fiber optic",
                "DSL",
                "Fiber optic",
                "No",
            ],
            "MonthlyCharges": [
                29.85,
                70.35,
                55.20,
                20.10,
                85.00,
                64.50,
                95.50,
                45.00,
                80.10,
                25.00,
            ],
            "TotalCharges": [
                "29.85",
                "1688.4",
                "165.6",
                "964.8",
                "1020.0",
                "3870.0",
                "191.0",
                "1620.0",
                "480.6",
                "1800.0",
            ],
            "Churn": [
                "No",
                "No",
                "Yes",
                "No",
                "Yes",
                "No",
                "Yes",
                "No",
                "Yes",
                "No",
            ],
        }
    )


def test_set_plot_style():
    """Verify that set_plot_style executes and configures styling parameters."""
    set_plot_style(style="whitegrid", font_scale=1.0)
    assert plt.rcParams["font.family"] == ["sans-serif"]


def test_plot_churn_distribution(sample_churn_df: pd.DataFrame):
    """Test generating churn distribution bar chart."""
    fig, ax = plot_churn_distribution(
        sample_churn_df,
        target_col="Churn",
        title="Custom Churn Distribution",
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert ax.get_title() == "Custom Churn Distribution"
    assert len(ax.patches) == 2


def test_plot_churn_distribution_save(sample_churn_df: pd.DataFrame, tmp_path: Path):
    """Verify plot_churn_distribution saves an image file to disk."""
    save_file = tmp_path / "figures" / "churn_dist_test.png"
    _fig, _ax = plot_churn_distribution(sample_churn_df, save_path=save_file)
    assert save_file.exists()
    assert save_file.stat().st_size > 0


def test_plot_churn_distribution_invalid(sample_churn_df: pd.DataFrame):
    """Verify error handling on invalid DataFrame or missing target."""
    with pytest.raises(TypeError):
        plot_churn_distribution("not_a_df")  # type: ignore

    with pytest.raises(KeyError):
        plot_churn_distribution(sample_churn_df, target_col="NonExistent")

    with pytest.raises(ValueError):
        plot_churn_distribution(pd.DataFrame({"Churn": []}))


def test_plot_numerical_distribution_single(sample_churn_df: pd.DataFrame):
    """Test numerical distribution plotting without target breakdown."""
    fig, axes = plot_numerical_distribution(
        sample_churn_df, column="MonthlyCharges", target_col=None
    )
    assert isinstance(fig, Figure)
    assert len(axes) == 2


def test_plot_numerical_distribution_with_target(sample_churn_df: pd.DataFrame):
    """Test numerical distribution plotting split by target churn."""
    fig, axes = plot_numerical_distribution(
        sample_churn_df,
        column="tenure",
        target_col="Churn",
        title="Tenure by Churn",
    )
    assert isinstance(fig, Figure)
    assert len(axes) == 2
    assert "tenure" in axes[0].get_title().lower()


def test_plot_numerical_distribution_with_string_numeric(
    sample_churn_df: pd.DataFrame,
):
    """Test numerical distribution handles string representations of numbers."""
    fig, axes = plot_numerical_distribution(
        sample_churn_df,
        column="TotalCharges",
        target_col="Churn",
    )
    assert isinstance(fig, Figure)
    assert len(axes) == 2


def test_plot_numerical_distribution_save(
    sample_churn_df: pd.DataFrame, tmp_path: Path
):
    """Verify plot_numerical_distribution saves image to disk."""
    save_file = tmp_path / "numerical_test.png"
    plot_numerical_distribution(sample_churn_df, column="tenure", save_path=save_file)
    assert save_file.exists()
    assert save_file.stat().st_size > 0


def test_plot_numerical_distribution_invalid(sample_churn_df: pd.DataFrame):
    """Verify error handling for invalid columns or non-numeric values."""
    with pytest.raises(KeyError):
        plot_numerical_distribution(sample_churn_df, column="DoesNotExist")

    with pytest.raises(KeyError):
        plot_numerical_distribution(
            sample_churn_df, column="tenure", target_col="InvalidTarget"
        )

    with pytest.raises(ValueError):
        plot_numerical_distribution(sample_churn_df, column="Contract")


def test_plot_categorical_churn_rates_single(sample_churn_df: pd.DataFrame):
    """Test categorical churn rates with a single column."""
    fig, ax = plot_categorical_churn_rates(
        sample_churn_df,
        categorical_cols="Contract",
        target_col="Churn",
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert "Contract" in ax.get_title()


def test_plot_categorical_churn_rates_multiple(sample_churn_df: pd.DataFrame):
    """Test categorical churn rates across multiple columns in subplots."""
    fig, axes = plot_categorical_churn_rates(
        sample_churn_df,
        categorical_cols=["Contract", "InternetService"],
        target_col="Churn",
    )
    assert isinstance(fig, Figure)
    assert hasattr(axes, "shape") or isinstance(axes, (list, tuple))


def test_plot_categorical_churn_rates_save(
    sample_churn_df: pd.DataFrame, tmp_path: Path
):
    """Verify plot_categorical_churn_rates saves figure to disk."""
    save_file = tmp_path / "cat_rates.png"
    plot_categorical_churn_rates(
        sample_churn_df,
        categorical_cols="Contract",
        save_path=save_file,
    )
    assert save_file.exists()
    assert save_file.stat().st_size > 0


def test_plot_categorical_churn_rates_invalid(sample_churn_df: pd.DataFrame):
    """Verify error handling on invalid categorical columns or targets."""
    with pytest.raises(KeyError):
        plot_categorical_churn_rates(sample_churn_df, categorical_cols="InvalidCol")

    with pytest.raises(KeyError):
        plot_categorical_churn_rates(
            sample_churn_df,
            categorical_cols="Contract",
            target_col="InvalidTarget",
        )


def test_plot_correlation_heatmap_auto(sample_churn_df: pd.DataFrame):
    """Test correlation heatmap with automatic numeric column detection."""
    fig, ax = plot_correlation_heatmap(sample_churn_df)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert "Correlation Matrix" in ax.get_title()


def test_plot_correlation_heatmap_specified_cols(sample_churn_df: pd.DataFrame):
    """Test correlation heatmap with user specified columns."""
    fig, ax = plot_correlation_heatmap(
        sample_churn_df,
        columns=["tenure", "MonthlyCharges"],
        title="Custom Correlation",
    )
    assert isinstance(fig, Figure)
    assert ax.get_title() == "Custom Correlation"


def test_plot_correlation_heatmap_save(sample_churn_df: pd.DataFrame, tmp_path: Path):
    """Verify plot_correlation_heatmap saves image to disk."""
    save_file = tmp_path / "corr_heatmap.png"
    plot_correlation_heatmap(sample_churn_df, save_path=save_file)
    assert save_file.exists()
    assert save_file.stat().st_size > 0


def test_plot_correlation_heatmap_invalid(sample_churn_df: pd.DataFrame):
    """Verify error handling when insufficient numeric features exist."""
    with pytest.raises(ValueError):
        plot_correlation_heatmap(sample_churn_df[["Contract", "gender"]])

    with pytest.raises(KeyError):
        plot_correlation_heatmap(sample_churn_df, columns=["missing_col"])


def test_visualization_alias_parity():
    """Verify src.visualization alias exposes exact same functions."""
    assert alias_churn is plot_churn_distribution
    assert alias_num is plot_numerical_distribution
    assert alias_cat is plot_categorical_churn_rates
    assert alias_corr is plot_correlation_heatmap
    assert alias_style is set_plot_style
