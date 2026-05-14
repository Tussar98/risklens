"""Logistic regression PD model: interpretable baseline.

Wraps sklearn LogisticRegression with project conventions:
  - L2 regularization, lbfgs solver, 1000 max_iter (converges on ~1M rows)
  - No class balancing (PD outputs must be calibrated probabilities; reweighting
    breaks calibration)
  - Disk persistence: saves the full Pipeline (preprocess + encode + model)
    as a single joblib artifact for inference reuse
  - Coefficient inspection with feature names attached
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from risklens.features.pipeline import build_feature_pipeline

logger = logging.getLogger(__name__)


@dataclass
class LogisticPDConfig:
    """Hyperparameters for the logistic PD model."""

    C: float = 1.0
    max_iter: int = 1000
    solver: str = "lbfgs"
    random_state: int = 42

    def to_dict(self) -> dict[str, Any]:
        return {
            "C": self.C,
            "max_iter": self.max_iter,
            "solver": self.solver,
            "random_state": self.random_state,
        }


def build_logistic_pipeline(config: LogisticPDConfig | None = None) -> Pipeline:
    """Construct full pipeline: feature pipeline + logistic regression classifier."""
    config = config or LogisticPDConfig()
    feature_pipeline = build_feature_pipeline()
    classifier = LogisticRegression(
        C=config.C,
        max_iter=config.max_iter,
        solver=config.solver,
        random_state=config.random_state,
        n_jobs=-1,
    )
    return Pipeline(steps=[
        ("features", feature_pipeline),
        ("classifier", classifier),
    ])


def fit_logistic_pd(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: LogisticPDConfig | None = None,
) -> Pipeline:
    """Fit the full pipeline (features + logistic) on training data."""
    config = config or LogisticPDConfig()
    pipe = build_logistic_pipeline(config)
    logger.info(f"Fitting logistic PD model on {len(X_train):,} rows...")
    pipe.fit(X_train, y_train)
    logger.info("Fit complete.")
    return pipe


def predict_proba_default(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Predict probability of default (class 1) for each row.

    Returns a 1D numpy array of shape (n_samples,) with values in [0, 1].
    """
    proba = pipe.predict_proba(X)
    return proba[:, 1]


def save_pipeline(pipe: Pipeline, path: Path) -> Path:
    """Serialize the fitted pipeline to disk via joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved pipeline to {path} ({size_mb:.2f} MB)")
    return path


def load_pipeline(path: Path) -> Pipeline:
    """Deserialize a fitted pipeline from disk."""
    pipe = joblib.load(path)
    logger.info(f"Loaded pipeline from {path}")
    return pipe


def get_coefficients(pipe: Pipeline) -> pd.DataFrame:
    """Return logistic coefficients alongside their feature names.

    Useful for inspecting model behavior and producing the MRM appendix.
    """
    encoder = pipe.named_steps["features"].named_steps["encode"]
    feature_names = list(encoder.get_feature_names_out())
    classifier = pipe.named_steps["classifier"]

    if classifier.coef_.shape[0] != 1:
        raise ValueError("Expected binary classifier with a single coef row.")

    coef = classifier.coef_[0]
    intercept = float(classifier.intercept_[0])

    df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coef,
        "abs_coefficient": np.abs(coef),
    })
    df = df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    logger.info(f"Logistic intercept: {intercept:.4f}")
    logger.info(f"Number of features: {len(feature_names)}")

    return df


__all__ = [
    "LogisticPDConfig",
    "build_logistic_pipeline",
    "fit_logistic_pd",
    "predict_proba_default",
    "save_pipeline",
    "load_pipeline",
    "get_coefficients",
]
