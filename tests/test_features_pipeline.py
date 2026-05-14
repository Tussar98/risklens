"""Tests for the feature pipeline: leakage protection, determinism, output shape."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from risklens.features.leakage_blacklist import POST_ORIGINATION_COLUMNS
from risklens.features.pipeline import (
    assert_no_leakage_columns,
    build_feature_pipeline,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "interim" / "loans_filtered.parquet"


@pytest.fixture(scope="module")
def sample_data() -> pd.DataFrame:
    """Load a small sample of the filtered dataset for pipeline tests.

    Skipped if the Parquet file doesn't exist (e.g., in CI where data isn't downloaded).
    """
    if not DATA_PATH.exists():
        pytest.skip(f"Data file not available at {DATA_PATH}; skipping data-dependent tests.")
    return pd.read_parquet(DATA_PATH).sample(2_000, random_state=42)


def test_pipeline_builds() -> None:
    """The pipeline can be constructed without error."""
    pipe = build_feature_pipeline()
    assert pipe is not None
    assert "preprocess" in pipe.named_steps
    assert "encode" in pipe.named_steps


def test_pipeline_fit_transform_shape(sample_data: pd.DataFrame) -> None:
    """Fit + transform produces a 2D array with the expected row count."""
    X = sample_data.drop(columns=["target"])
    y = sample_data["target"]

    pipe = build_feature_pipeline()
    X_out = pipe.fit_transform(X, y)

    assert X_out.ndim == 2
    assert X_out.shape[0] == X.shape[0]
    assert X_out.shape[1] > 0


def test_pipeline_no_leakage_columns(sample_data: pd.DataFrame) -> None:
    """No post-origination column may appear in the output feature names."""
    X = sample_data.drop(columns=["target"])
    y = sample_data["target"]

    pipe = build_feature_pipeline()
    pipe.fit(X, y)
    feature_names = list(pipe.named_steps["encode"].get_feature_names_out())

    leaked = [f for f in feature_names if f in POST_ORIGINATION_COLUMNS]
    assert not leaked, f"Leakage columns found in pipeline output: {leaked}"


def test_pipeline_output_no_nan(sample_data: pd.DataFrame) -> None:
    """Pipeline output must contain no NaN values (imputation should handle all)."""
    X = sample_data.drop(columns=["target"])
    y = sample_data["target"]

    pipe = build_feature_pipeline()
    X_out = pipe.fit_transform(X, y)

    assert not np.isnan(np.asarray(X_out, dtype=float)).any(), (
        "Pipeline output contains NaN values; imputation logic is incomplete."
    )


def test_pipeline_deterministic(sample_data: pd.DataFrame) -> None:
    """Fitting on the same data twice produces identical output matrices."""
    X = sample_data.drop(columns=["target"])
    y = sample_data["target"]

    pipe1 = build_feature_pipeline()
    pipe2 = build_feature_pipeline()
    X_out_1 = np.asarray(pipe1.fit_transform(X, y), dtype=float)
    X_out_2 = np.asarray(pipe2.fit_transform(X, y), dtype=float)

    np.testing.assert_allclose(X_out_1, X_out_2, rtol=1e-10, atol=1e-10)


def test_assert_no_leakage_columns_raises() -> None:
    """The standalone safety helper raises when a blacklist column is present."""
    leakage_col = next(iter(POST_ORIGINATION_COLUMNS))
    with pytest.raises(ValueError, match="Leakage column"):
        assert_no_leakage_columns(["loan_amnt", leakage_col, "fico_range_low"])


def test_assert_no_leakage_columns_passes() -> None:
    """The standalone safety helper passes when no blacklist column is present."""
    assert_no_leakage_columns(["loan_amnt", "fico_range_low", "dti"])