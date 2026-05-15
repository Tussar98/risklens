"""Calibration analysis and correction for PD models.

Tools for:
  - Computing reliability curves (mean predicted PD vs observed default rate
    by quantile bin)
  - Computing Expected Calibration Error (ECE), the weighted mean absolute
    difference between predicted and observed rates
  - Fitting an isotonic regression calibrator on a held-out set (typically val)
    to map raw scores -> calibrated probabilities

Isotonic regression is preferred over Platt scaling for PD models because it
is non-parametric: it can correct an arbitrary monotone miscalibration without
assuming a specific functional form. The cost is needing more calibration data
(thousands of points minimum). With ~169k val rows we have plenty.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


@dataclass
class ReliabilityCurve:
    """Reliability curve data: one row per bin."""

    bin_lower: np.ndarray
    bin_upper: np.ndarray
    mean_predicted: np.ndarray
    observed_rate: np.ndarray
    bin_count: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "bin_lower": self.bin_lower,
            "bin_upper": self.bin_upper,
            "mean_predicted": self.mean_predicted,
            "observed_rate": self.observed_rate,
            "bin_count": self.bin_count,
        })


def compute_reliability_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> ReliabilityCurve:
    """Compute the reliability curve for a binary classifier.

    Args:
        y_true: 1D binary array of true labels.
        y_score: 1D array of predicted probabilities in [0, 1].
        n_bins: number of bins to partition the predictions into.
        strategy: "quantile" for equal-frequency bins (recommended for PD;
            ensures every bin has enough data), or "uniform" for equal-width.

    Returns:
        ReliabilityCurve dataclass with one row per bin.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score).astype(float).ravel()

    if strategy == "quantile":
        bin_edges = np.quantile(y_score, np.linspace(0.0, 1.0, n_bins + 1))
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
        bin_edges = np.unique(bin_edges)
    elif strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    bin_idx = np.clip(np.searchsorted(bin_edges, y_score, side="right") - 1, 0, len(bin_edges) - 2)

    lowers, uppers, means, rates, counts = [], [], [], [], []
    for b in range(len(bin_edges) - 1):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        lowers.append(float(bin_edges[b]))
        uppers.append(float(bin_edges[b + 1]))
        means.append(float(y_score[mask].mean()))
        rates.append(float(y_true[mask].mean()))
        counts.append(n)

    return ReliabilityCurve(
        bin_lower=np.array(lowers),
        bin_upper=np.array(uppers),
        mean_predicted=np.array(means),
        observed_rate=np.array(rates),
        bin_count=np.array(counts),
    )


def expected_calibration_error(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> float:
    """Expected Calibration Error (ECE).

    Weighted mean of |mean_predicted - observed_rate| across bins, weighted by
    bin size. Zero for a perfectly calibrated model.
    """
    curve = compute_reliability_curve(y_true, y_score, n_bins=n_bins, strategy=strategy)
    total = curve.bin_count.sum()
    if total == 0:
        return 0.0
    weights = curve.bin_count / total
    gaps = np.abs(curve.mean_predicted - curve.observed_rate)
    return float(np.sum(weights * gaps))


def fit_isotonic_calibrator(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> IsotonicRegression:
    """Fit a monotone isotonic regression mapping raw scores -> calibrated probs.

    Fit on the validation set; apply to val and test for evaluation.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score).astype(float).ravel()
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(y_score, y_true)
    logger.info(f"Fitted isotonic calibrator on {len(y_true):,} val rows")
    return iso


def apply_calibrator(calibrator: IsotonicRegression, y_score: np.ndarray) -> np.ndarray:
    """Map raw scores to calibrated probabilities."""
    return calibrator.predict(np.asarray(y_score).astype(float).ravel())


__all__ = [
    "ReliabilityCurve",
    "compute_reliability_curve",
    "expected_calibration_error",
    "fit_isotonic_calibrator",
    "apply_calibrator",
]
