"""Train the logistic regression PD baseline.

Loads the filtered dataset, splits by vintage, fits the full pipeline
(features + logistic) on train, evaluates on val and test, and saves:
  - artifacts/models/pd_logistic.joblib
  - artifacts/reports/pd_logistic_metrics.json
  - artifacts/reports/pd_logistic_coefficients.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from risklens.data.splits import split_by_vintage, split_features_and_target
from risklens.evaluation.metrics import compute_pd_metrics
from risklens.models.pd_logistic import (
    LogisticPDConfig,
    fit_logistic_pd,
    get_coefficients,
    predict_proba_default,
    save_pipeline,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "interim" / "loans_filtered.parquet"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "pd_logistic.joblib"
METRICS_PATH = PROJECT_ROOT / "artifacts" / "reports" / "pd_logistic_metrics.json"
COEF_PATH = PROJECT_ROOT / "artifacts" / "reports" / "pd_logistic_coefficients.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train logistic PD baseline")
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse regularization strength (smaller = stronger regularization)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum solver iterations",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional: train on a random sample of N rows (for quick smoke runs)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Load and split
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

    # Fit
    config = LogisticPDConfig(C=args.C, max_iter=args.max_iter)
    logger.info(f"Logistic config: {config.to_dict()}")
    pipe = fit_logistic_pd(X_train, y_train, config)

    # Evaluate on all three splits
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

    # Save model
    save_pipeline(pipe, MODEL_PATH)

    # Save metrics report
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(f"Saved metrics report to {METRICS_PATH}")

    # Save coefficients for MRM appendix
    coef_df = get_coefficients(pipe)
    coef_df.to_csv(COEF_PATH, index=False)
    logger.info(f"Saved coefficients to {COEF_PATH}")
    logger.info("Top 15 features by absolute coefficient:")
    for _, row in coef_df.head(15).iterrows():
        logger.info(f"  {row['feature']:<35} {row['coefficient']:>+8.4f}")

    print(f"\nSUCCESS: model + reports written under {MODEL_PATH.parent.parent}")


if __name__ == "__main__":
    main()
