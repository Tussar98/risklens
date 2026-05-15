"""Two-stage hurdle LGD model.

LGD modeling is complicated by the bimodal distribution of recovery rates:
~31% of charged-off loans recover $0 (LGD = 1.0), while the remaining 69%
recover a continuous distribution of principal. A single-stage regressor
handles this poorly.

The hurdle model decomposes the problem:
  Stage 1 (hurdle): P(any recovery | default) -- logistic regression
  Stage 2 (conditional): E[recovery_rate | default, recovered] -- XGBoost
                         regressor on logit-transformed target
  Composition: E[recovery_rate] = P(recovered) * E[rate | recovered]
               LGD = 1 - E[recovery_rate]

Inputs match the PD model's feature pipeline so features are encoded
consistently across the EL framework.
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
from xgboost import XGBRegressor

from risklens.features.pipeline import build_feature_pipeline

logger = logging.getLogger(__name__)

# Numerical guards for the logit transform on (0, 1)-bounded targets.
_LOGIT_EPS = 1e-3


@dataclass
class HurdleLGDConfig:
    """Hyperparameters for the two-stage hurdle LGD model."""

    # Stage 1 (logistic hurdle)
    hurdle_C: float = 1.0
    hurdle_max_iter: int = 1000

    # Stage 2 (XGBoost regressor on logit-recovery_rate)
    reg_n_estimators: int = 500
    reg_max_depth: int = 5
    reg_learning_rate: float = 0.05
    reg_min_child_weight: int = 10
    reg_subsample: float = 0.8
    reg_colsample_bytree: float = 0.8
    reg_early_stopping_rounds: int = 50

    random_state: int = 42

    def to_dict(self) -> dict[str, Any]:
        return {
            "hurdle_C": self.hurdle_C,
            "hurdle_max_iter": self.hurdle_max_iter,
            "reg_n_estimators": self.reg_n_estimators,
            "reg_max_depth": self.reg_max_depth,
            "reg_learning_rate": self.reg_learning_rate,
            "reg_min_child_weight": self.reg_min_child_weight,
            "reg_subsample": self.reg_subsample,
            "reg_colsample_bytree": self.reg_colsample_bytree,
            "reg_early_stopping_rounds": self.reg_early_stopping_rounds,
            "random_state": self.random_state,
        }


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _LOGIT_EPS, 1.0 - _LOGIT_EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class HurdleLGDModel:
    """Two-stage hurdle model for LGD prediction.

    Workflow:
      1. Fit a shared feature pipeline on all charge-offs in the training set.
      2. Fit a logistic regression hurdle: P(recovered_flag = 1).
      3. Fit an XGBoost regressor on logit(recovery_rate) using ONLY the
         subset of training rows where recovered_flag == 1.
    """

    def __init__(self, config: HurdleLGDConfig | None = None) -> None:
        self.config = config or HurdleLGDConfig()
        self.feature_pipeline: Pipeline | None = None
        self.hurdle: LogisticRegression | None = None
        self.regressor: XGBRegressor | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        recovered_flag_train: pd.Series,
        recovery_rate_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        recovered_flag_val: pd.Series | None = None,
        recovery_rate_val: pd.Series | None = None,
    ) -> HurdleLGDModel:
        """Fit both stages. X_val and val targets enable early stopping on stage 2."""

        # 1. Fit feature pipeline on all training rows.
        self.feature_pipeline = build_feature_pipeline()
        logger.info(f"Fitting feature pipeline on {len(X_train):,} train rows")
        X_train_t = self.feature_pipeline.fit_transform(X_train, recovered_flag_train)

        # 2. Stage 1: hurdle classifier.
        self.hurdle = LogisticRegression(
            C=self.config.hurdle_C,
            max_iter=self.config.hurdle_max_iter,
            solver="lbfgs",
            random_state=self.config.random_state,
        )
        logger.info(f"Fitting hurdle (logistic) on {len(X_train_t):,} rows, "
                    f"base rate {recovered_flag_train.mean():.4f}")
        self.hurdle.fit(X_train_t, recovered_flag_train)

        # 3. Stage 2: conditional regressor on logit(recovery_rate) | recovered.
        mask_train = recovered_flag_train.to_numpy() == 1
        X_train_recovered = X_train_t[mask_train]
        y_train_logit = _logit(recovery_rate_train.to_numpy()[mask_train])
        logger.info(f"Fitting conditional regressor on {mask_train.sum():,} recovered loans")

        eval_set = None
        if X_val is not None and recovered_flag_val is not None and recovery_rate_val is not None:
            X_val_t = self.feature_pipeline.transform(X_val)
            mask_val = recovered_flag_val.to_numpy() == 1
            X_val_recovered = X_val_t[mask_val]
            y_val_logit = _logit(recovery_rate_val.to_numpy()[mask_val])
            eval_set = [(X_val_recovered, y_val_logit)]
            logger.info(f"  Stage 2 early stopping eval set: {mask_val.sum():,} rows")

        self.regressor = XGBRegressor(
            n_estimators=self.config.reg_n_estimators,
            max_depth=self.config.reg_max_depth,
            learning_rate=self.config.reg_learning_rate,
            min_child_weight=self.config.reg_min_child_weight,
            subsample=self.config.reg_subsample,
            colsample_bytree=self.config.reg_colsample_bytree,
            early_stopping_rounds=self.config.reg_early_stopping_rounds,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=self.config.random_state,
            n_jobs=-1,
            verbosity=0,
        )
        self.regressor.fit(
            X_train_recovered,
            y_train_logit,
            eval_set=eval_set,
            verbose=False,
        )
        if eval_set is not None and hasattr(self.regressor, "best_iteration"):
            logger.info(f"  Stage 2 converged at iteration {self.regressor.best_iteration}")

        return self

    def predict_components(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (p_recovered, conditional_recovery_rate) for each row."""
        if self.feature_pipeline is None or self.hurdle is None or self.regressor is None:
            raise RuntimeError("Model has not been fitted yet.")

        X_t = self.feature_pipeline.transform(X)
        p_recovered = self.hurdle.predict_proba(X_t)[:, 1]
        logit_rate = self.regressor.predict(X_t)
        conditional_rate = _sigmoid(logit_rate)
        return p_recovered, conditional_rate

    def predict_recovery_rate(self, X: pd.DataFrame) -> np.ndarray:
        """E[recovery_rate] = P(recovered) * E[rate | recovered]."""
        p_recovered, conditional_rate = self.predict_components(X)
        return p_recovered * conditional_rate

    def predict_lgd(self, X: pd.DataFrame) -> np.ndarray:
        """LGD = 1 - E[recovery_rate]."""
        return 1.0 - self.predict_recovery_rate(X)


def save_lgd_model(model: HurdleLGDModel, path: Path) -> Path:
    """Serialize the fitted LGD model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved LGD model to {path} ({size_mb:.2f} MB)")
    return path


def load_lgd_model(path: Path) -> HurdleLGDModel:
    """Deserialize a fitted LGD model."""
    model = joblib.load(path)
    logger.info(f"Loaded LGD model from {path}")
    return model


__all__ = [
    "HurdleLGDConfig",
    "HurdleLGDModel",
    "save_lgd_model",
    "load_lgd_model",
]

