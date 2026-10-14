"""Basic import tests to verify project structure."""

import importlib
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_src_imports():
    """Verify src subpackages can be imported."""
    packages = [
        "src",
        "src.churn_prediction",
        "src.churn_prediction.config",
        "src.churn_prediction.predict",
        "src.churn_prediction.data",
        "src.churn_prediction.data.loader",
        "src.churn_prediction.data.validation",
        "src.churn_prediction.features",
        "src.churn_prediction.features.preprocessing",
        "src.churn_prediction.models",
        "src.churn_prediction.models.model",
        "src.churn_prediction.models.tuning",
        "src.churn_prediction.evaluation",
        "src.churn_prediction.evaluation.metrics",
        "src.churn_prediction.evaluation.plots",
        "src.churn_prediction.visualization",
        "src.churn_prediction.visualization.plots",
        "src.churn_prediction.api",
        "src.churn_prediction.api.main",
        "src.churn_prediction.api.schemas",
    ]
    for pkg in packages:
        mod = importlib.import_module(pkg)
        assert mod is not None, f"Failed to import {pkg}"
