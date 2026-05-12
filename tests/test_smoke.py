"""Smoke tests: verify the package and core dependencies are importable."""

def test_risklens_imports():
    import risklens  # noqa: F401

def test_core_deps_import():
    import pandas  # noqa: F401
    import numpy  # noqa: F401
    import sklearn  # noqa: F401
    import xgboost  # noqa: F401
    import lifelines  # noqa: F401
    import pymc  # noqa: F401
    import shap  # noqa: F401