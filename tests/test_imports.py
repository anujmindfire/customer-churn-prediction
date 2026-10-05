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
        "src.data",
        "src.features",
        "src.models",
        "src.evaluation",
    ]
    for pkg in packages:
        mod = importlib.import_module(pkg)
        assert mod is not None, f"Failed to import {pkg}"
