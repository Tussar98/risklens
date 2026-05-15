"""XGBoost PD model: gradient-boosted trees baseline.

Wraps xgboost.XGBClassifier with project conventions:
  - Sensible defaults for tabular credit-risk data
  - Early stopping on validation AUC
  - No class balancing (preserve probability calibration)
  - Disk persistence via joblib
  - Feature importance + SHAP-ready interface
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from risklens.features.pipeline import build_feature_pipeline

logger = logging.getLogger(__name__)


@dataclass
class XGBoostPDConfig:
    """Hyperparameters for the XGBoost PD model."""

    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    min_child_weight: int = 10
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50
    random_state: int = 42

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "early_stopping_rounds": self.early_stopping_rounds,
            "random_state": self.random_state,
        }


def build_xgboost_classifier(config: XGBoostPDConfig | None = None) -> XGBClassifier:
    """Construct an XGBClassifier with the project defaults."""
    config = config or XGBoostPDConfig()
    return XGBClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        min_child_weight=config.min_child_weight,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        early_stopping_rounds=config.early_stopping_rounds,
        objective="binary:logistic",
        eval_metric=["auc", "logloss"],
        tree_method="hist",
        random_state=config.random_state,
        n_jobs=-1,
        verbosity=0,
    )


def fit_xgboost_pd(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: XGBoostPDConfig | None = None,
) -> Pipeline:
    """Fit feature pipeline + XGBoost on train, with early stopping on val.

    The feature pipeline is fit only on the training set, then frozen and applied
    to val. This prevents leakage of val statistics into the training transform.
    """
    config = config or XGBoostPDConfig()

    # Fit feature pipeline on train, freeze, transform both
    feature_pipeline = build_feature_pipeline()
    logger.info(f"Fitting feature pipeline on {len(X_train):,} train rows...")
    X_train_t = feature_pipeline.fit_transform(X_train, y_train)
    X_val_t = feature_pipeline.transform(X_val)
    logger.info(f"Feature matrix shape: {X_train_t.shape}")

    # Fit XGBoost with early stopping
    clf = build_xgboost_classifier(config)
    logger.info(f"Fitting XGBoost: {config.to_dict()}")
    clf.fit(
        X_train_t,
        y_train,
        eval_set=[(X_val_t, y_val)],
        verbose=False,
    )
    best_iter = clf.best_iteration if hasattr(clf, "best_iteration") else None
    if best_iter is not None:
        logger.info(f"XGBoost converged at iteration {best_iter} (of {config.n_estimators})")

    # Wrap in a Pipeline for consistent interface with logistic baseline
    return Pipeline(steps=[
        ("features", feature_pipeline),
        ("classifier", clf),
    ])


def predict_proba_default(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Predict probability of default (class 1) for each row."""
    proba = pipe.predict_proba(X)
    return proba[:, 1]


def save_pipeline(pipe: Pipeline, path: Path) -> Path:
    """Serialize the fitted pipeline to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved XGBoost pipeline to {path} ({size_mb:.2f} MB)")
    return path


def load_pipeline(path: Path) -> Pipeline:
    """Deserialize a fitted XGBoost pipeline from disk."""
    pipe = joblib.load(path)
    logger.info(f"Loaded XGBoost pipeline from {path}")
    return pipe


def get_feature_importance(pipe: Pipeline) -> pd.DataFrame:
    """Return feature importances ranked by gain.

    Gain = avg loss reduction when a feature is used for splitting. The standard
    XGBoost importance metric for model interpretation.
    """
    encoder = pipe.named_steps["features"].named_steps["encode"]
    feature_names = list(encoder.get_feature_names_out())
    clf = pipe.named_steps["classifier"]

    booster = clf.get_booster()
    importance_dict = booster.get_score(importance_type="gain")

    # XGBoost names features as f0, f1, ... mapped by column order
    df_rows = []
    for i, name in enumerate(feature_names):
        key = f"f{i}"
        gain = importance_dict.get(key, 0.0)
        df_rows.append({"feature": name, "importance_gain": gain})

    df = pd.DataFrame(df_rows)
    df = df.sort_values("importance_gain", ascending=False).reset_index(drop=True)
    return df


__all__ = [
    "XGBoostPDConfig",
    "build_xgboost_classifier",
    "fit_xgboost_pd",
    "predict_proba_default",
    "save_pipeline",
    "load_pipeline",
    "get_feature_importance",
]
