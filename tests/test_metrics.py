"""Tests for the PD metrics module."""

from __future__ import annotations

import numpy as np
import pytest

from risklens.evaluation.metrics import (
    PDMetrics,
    compute_ks,
    compute_pd_metrics,
)


def test_compute_pd_metrics_perfect_predictions() -> None:
    """Perfect predictions: AUC and KS are 1.0; Brier is small."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.05, 0.10, 0.15, 0.85, 0.90, 0.95])

    m = compute_pd_metrics(y_true, y_score)

    assert m.auc == pytest.approx(1.0)
    assert m.ks == pytest.approx(1.0)
    assert m.brier < 0.05
    assert m.n_samples == 6
    assert m.base_rate == pytest.approx(0.5)


def test_compute_pd_metrics_random_predictions() -> None:
    """Random predictions: AUC near 0.5."""
    rng = np.random.default_rng(42)
    y_true = rng.binomial(1, 0.2, size=5_000)
    y_score = rng.uniform(0, 1, size=5_000)

    m = compute_pd_metrics(y_true, y_score)

    assert 0.45 < m.auc < 0.55
    assert m.ks < 0.10


def test_compute_pd_metrics_returns_dataclass() -> None:
    """Output is a PDMetrics dataclass with all expected fields."""
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, 0.8, 0.3, 0.7])

    m = compute_pd_metrics(y_true, y_score)

    assert isinstance(m, PDMetrics)
    d = m.to_dict()
    expected_keys = {
        "auc", "ks", "ks_threshold", "brier",
        "log_loss", "n_samples", "base_rate",
    }
    assert set(d.keys()) == expected_keys


def test_compute_pd_metrics_rejects_invalid_scores() -> None:
    """Scores outside [0, 1] raise ValueError."""
    y_true = np.array([0, 1])
    y_score = np.array([0.5, 1.5])

    with pytest.raises(ValueError, match=r"y_score must be in \[0, 1\]"):
        compute_pd_metrics(y_true, y_score)


def test_compute_pd_metrics_rejects_shape_mismatch() -> None:
    """Mismatched shapes raise ValueError."""
    y_true = np.array([0, 1, 0])
    y_score = np.array([0.5, 0.7])

    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_pd_metrics(y_true, y_score)


def test_compute_ks_single_class_raises() -> None:
    """KS is undefined when one class is missing from y_true."""
    y_true = np.array([0, 0, 0, 0])
    y_score = np.array([0.1, 0.2, 0.3, 0.4])

    with pytest.raises(ValueError, match="KS undefined"):
        compute_ks(y_true, y_score)


def test_compute_ks_returns_score_at_max_separation() -> None:
    """KS function returns both the value and the score where it occurs."""
    rng = np.random.default_rng(0)
    y_true = rng.binomial(1, 0.3, size=2_000)
    noise = rng.normal(0, 0.4, size=2_000)
    logits = -1.0 + 2.5 * y_true + noise
    y_score = 1 / (1 + np.exp(-logits))

    ks, threshold = compute_ks(y_true, y_score)

    assert 0.5 < ks <= 1.0
    assert 0.0 < threshold < 1.0

