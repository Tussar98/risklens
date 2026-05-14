"""Classification metrics for PD model evaluation.

Computes the four metrics every PD report shows: AUC, KS, Brier, log-loss.
KS is the credit-risk specific one - it measures the maximum gap between
the cumulative distributions of predicted probabilities for defaulters vs
non-defaulters. Risk teams care about KS more than they care about AUC because
it directly answers "at what score do bad loans separate from good ones?"
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


@dataclass
class PDMetrics:
    """Container for PD model performance metrics on a single dataset."""

    auc: float
    ks: float
    ks_threshold: float
    brier: float
    log_loss: float
    n_samples: int
    base_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def report(self, label: str = "") -> str:
        """Multi-line formatted report for logging or saving."""
        header = f"=== PD Metrics{': ' + label if label else ''} ===\n"
        return (
            header
            + f"  Samples:    {self.n_samples:,}\n"
            + f"  Base rate:  {self.base_rate:.4f}\n"
            + f"  AUC:        {self.auc:.4f}\n"
            + f"  KS:         {self.ks:.4f}  (at score {self.ks_threshold:.4f})\n"
            + f"  Brier:      {self.brier:.4f}\n"
            + f"  Log-loss:   {self.log_loss:.4f}"
        )


def compute_ks(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Compute the Kolmogorov-Smirnov statistic and the score at which it occurs.

    KS = max | F_1(p) - F_0(p) |, where F_1 / F_0 are the empirical CDFs of
    predicted scores among true positives / negatives respectively. The score
    at which the max gap occurs is useful for cutoff selection.

    Returns:
        (ks_value, threshold_at_max_separation)
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    scores_pos = y_score[y_true == 1]
    scores_neg = y_score[y_true == 0]

    if len(scores_pos) == 0 or len(scores_neg) == 0:
        raise ValueError("KS undefined when one class is missing from y_true.")

    result = ks_2samp(scores_pos, scores_neg)
    ks_value = float(result.statistic)

    # Find the score at which the max separation occurs
    all_scores = np.sort(np.unique(y_score))
    cdf_pos = np.searchsorted(np.sort(scores_pos), all_scores, side="right") / len(scores_pos)
    cdf_neg = np.searchsorted(np.sort(scores_neg), all_scores, side="right") / len(scores_neg)
    gap = np.abs(cdf_pos - cdf_neg)
    threshold = float(all_scores[np.argmax(gap)])

    return ks_value, threshold


def compute_pd_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> PDMetrics:
    """Compute the full metrics suite on a single (y_true, y_score) pair.

    Args:
        y_true: 1D binary array of true labels (0 or 1).
        y_score: 1D array of predicted probabilities in [0, 1].

    Returns:
        PDMetrics dataclass with auc, ks, brier, log_loss, etc.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score).astype(float).ravel()

    if y_true.shape != y_score.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_score {y_score.shape}"
        )
    if not np.all((y_score >= 0.0) & (y_score <= 1.0)):
        raise ValueError("y_score must be in [0, 1].")

    ks, ks_threshold = compute_ks(y_true, y_score)

    return PDMetrics(
        auc=float(roc_auc_score(y_true, y_score)),
        ks=ks,
        ks_threshold=ks_threshold,
        brier=float(brier_score_loss(y_true, y_score)),
        log_loss=float(log_loss(y_true, y_score, labels=[0, 1])),
        n_samples=int(y_true.size),
        base_rate=float(y_true.mean()),
    )


__all__ = [
    "PDMetrics",
    "compute_ks",
    "compute_pd_metrics",
]
