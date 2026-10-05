"""Dataset loading utilities for customer churn prediction."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_dataset(filepath: str | Path) -> pd.DataFrame:
    """Load a CSV dataset into a pandas DataFrame with validation.

    Args:
        filepath: Path to the CSV file as a string or Path object.

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.

    Raises:
        TypeError: If filepath is not a string or Path object.
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is not a CSV or is empty.

    Example:
        >>> from pathlib import Path
        >>> df = load_dataset(Path("data/raw/telco_customer_churn.csv"))
        >>> df.shape
        (7043, 21)
    """
    if not isinstance(filepath, (str, Path)):
        raise TypeError(
            f"filepath must be a str or Path, got {type(filepath).__name__}"
        )

    path = Path(filepath).resolve()
    logger.info("Attempting to load dataset from: %s", path)

    if not path.exists():
        msg = f"Dataset file not found at: {path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    if not path.is_file():
        msg = f"Path exists but is not a file: {path}"
        logger.error(msg)
        raise ValueError(msg)

    if path.suffix.lower() != ".csv":
        msg = f"Expected a CSV file (.csv), but got: {path.name}"
        logger.error(msg)
        raise ValueError(msg)

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as err:
        msg = f"The dataset file is empty: {path}"
        logger.error(msg)
        raise ValueError(msg) from err
    except Exception:
        logger.exception("Failed to load dataset from %s", path)
        raise

    if df.empty:
        msg = f"Loaded DataFrame is empty (zero rows): {path}"
        logger.warning(msg)

    logger.info(
        "Successfully loaded dataset from %s with %d rows and %d columns.",
        path.name,
        df.shape[0],
        df.shape[1],
    )
    return df
