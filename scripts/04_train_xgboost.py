"""Train the XGBoost PD model.

Loads filtered data, splits by vintage, fits the pipeline + XGBoost on train
with early stopping on val, evaluates on all three splits, and saves:
  - artifacts/models/pd_xgboost.joblib
  - artifacts/reports/pd_xgboost_metrics.json
  - artifacts/reports/pd_xgboost_importance.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from risklens.data.splits import split_by_vintage, split_features_and_target
from risklens.evaluation.metrics import compute_pd_metrics
from risklens.models.pd_xgboost import (
    XGBoostPDConfig,
    fit_xgboost_pd,
    get_feature_importance,
    predict_proba_default,
    save_pipeline,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "interim" / "loans_filtered.parquet"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "pd_xgboost.joblib"
METRICS_PATH = PROJECT_ROOT / "artifacts" / "reports" / "pd_xgboost_metrics.json"
IMPORTANCE_PATH = PROJECT_ROOT / "artifacts" / "reports" / "pd_xgboost_importance.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost PD model")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--min-child-weight", type=int, default=10)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional: train on a random sample of N rows for quick runs",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Loading {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    logger.info(f"Full dataset: {len(df):,} loans")

    splits = split_by_vintage(df)

    X_train, y_train = split_features_and_target(splits.train)
    X_val, y_val = split_features_and_target(splits.val)
    X_test, y_test = split_features_and_target(splits.test)

    if args.sample is not None and args.sample < len(X_train):
        logger.info(f"Sampling {args.sample:,} train rows for quick run")
        idx = X_train.sample(args.sample, random_state=42).index
        X_train = X_train.loc[idx]
        y_train = y_train.loc[idx]

    config = XGBoostPDConfig(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
    )

    pipe = fit_xgboost_pd(X_train, y_train, X_val, y_val, config)

    results: dict[str, dict] = {"config": config.to_dict()}
    for split_name, X, y in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        logger.info(f"Scoring {split_name} ({len(X):,} rows)...")
        y_score = predict_proba_default(pipe, X)
        metrics = compute_pd_metrics(y.to_numpy(), y_score)
        logger.info(metrics.report(split_name))
        results[split_name] = metrics.to_dict()

    save_pipeline(pipe, MODEL_PATH)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(f"Saved metrics report to {METRICS_PATH}")

    importance_df = get_feature_importance(pipe)
    importance_df.to_csv(IMPORTANCE_PATH, index=False)
    logger.info(f"Saved feature importance to {IMPORTANCE_PATH}")
    logger.info("Top 15 features by gain:")
    for _, row in importance_df.head(15).iterrows():
        logger.info(f"  {row['feature']:<35} {row['importance_gain']:>10.2f}")

    print(f"\nSUCCESS: XGBoost model + reports written under {MODEL_PATH.parent.parent}")


if __name__ == "__main__":
    main()
