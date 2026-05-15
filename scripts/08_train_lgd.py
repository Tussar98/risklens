"""Train the two-stage hurdle LGD model.

Loads the LGD dataset, splits by vintage (same policy as PD: train 2013-2016,
val 2017, test 2018Q1-Q2), fits both stages with early stopping on stage 2,
evaluates each stage and the composed LGD on all three splits, and saves:
  - artifacts/models/lgd_hurdle.joblib
  - artifacts/reports/lgd_metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from risklens.data.splits import TEST_QUARTERS_EXCLUDED, TEST_YEARS, TRAIN_YEARS, VAL_YEARS
from risklens.models.lgd import HurdleLGDConfig, HurdleLGDModel, save_lgd_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "interim" / "lgd_data.parquet"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "lgd_hurdle.joblib"
METRICS_PATH = PROJECT_ROOT / "artifacts" / "reports" / "lgd_metrics.json"


def split_lgd_by_vintage(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Apply the same vintage policy as PD but to the LGD dataframe."""
    df = df.dropna(subset=["vintage_year"]).copy()
    df["vintage_year"] = df["vintage_year"].astype(int)

    train = df[df["vintage_year"].isin(TRAIN_YEARS)].reset_index(drop=True)
    val = df[df["vintage_year"].isin(VAL_YEARS)].reset_index(drop=True)
    test = df[
        df["vintage_year"].isin(TEST_YEARS)
        & ~df["vintage_quarter"].isin(TEST_QUARTERS_EXCLUDED)
    ].reset_index(drop=True)

    return {"train": train, "val": val, "test": test}


def split_lgd_features_and_targets(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separate borrower features from the two LGD targets.

    Drops: recovery target columns, derived recovery fields, vintage helpers,
    and the loan_status column (used only for filtering, not modeling).
    """
    drop_cols = [
        "recoveries", "collection_recovery_fee", "total_pymnt", "total_rec_prncp",
        "funded_amnt", "recovery_amount", "recovery_rate", "recovered_flag",
        "vintage_year", "vintage_quarter", "issue_d", "loan_status",
    ]
    drop_present = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_present)
    y_flag = df["recovered_flag"].astype(int)
    y_rate = df["recovery_rate"].astype(float)
    return X, y_flag, y_rate


def evaluate_split(
    model: HurdleLGDModel,
    X: pd.DataFrame,
    y_flag: pd.Series,
    y_rate: pd.Series,
    label: str,
    logger: logging.Logger,
) -> dict[str, float]:
    """Evaluate hurdle accuracy, conditional MAE, and composed-LGD calibration."""
    p_recovered, cond_rate = model.predict_components(X)
    expected_rate = p_recovered * cond_rate

    # Hurdle: AUC for the binary recovered flag
    hurdle_auc = float(roc_auc_score(y_flag, p_recovered))

    # Conditional regressor: MAE on recovered subset
    mask = y_flag.to_numpy() == 1
    if mask.any():
        cond_mae = float(np.mean(np.abs(cond_rate[mask] - y_rate.to_numpy()[mask])))
    else:
        cond_mae = float("nan")

    # Composed LGD: how close is mean predicted recovery to mean observed?
    mean_observed_rate = float(y_rate.mean())
    mean_predicted_rate = float(expected_rate.mean())
    mean_observed_lgd = 1.0 - mean_observed_rate
    mean_predicted_lgd = 1.0 - mean_predicted_rate
    composed_mae = float(np.mean(np.abs(expected_rate - y_rate.to_numpy())))

    metrics = {
        "n_samples": int(len(y_flag)),
        "hurdle_base_rate": float(y_flag.mean()),
        "hurdle_auc": hurdle_auc,
        "conditional_mae": cond_mae,
        "mean_observed_recovery_rate": mean_observed_rate,
        "mean_predicted_recovery_rate": mean_predicted_rate,
        "mean_observed_lgd": mean_observed_lgd,
        "mean_predicted_lgd": mean_predicted_lgd,
        "composed_mae": composed_mae,
    }

    logger.info(f"=== LGD metrics: {label} ===")
    logger.info(f"  Samples:                 {metrics['n_samples']:,}")
    logger.info(f"  Hurdle base rate:        {metrics['hurdle_base_rate']:.4f}")
    logger.info(f"  Hurdle AUC:              {metrics['hurdle_auc']:.4f}")
    logger.info(f"  Conditional MAE:         {metrics['conditional_mae']:.4f}")
    logger.info(f"  Mean observed LGD:       {metrics['mean_observed_lgd']:.4f}")
    logger.info(f"  Mean predicted LGD:      {metrics['mean_predicted_lgd']:.4f}")
    logger.info(f"  Composed MAE (rate):     {metrics['composed_mae']:.4f}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train two-stage hurdle LGD model")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Loading {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    logger.info(f"LGD dataset: {len(df):,} charged-off loans")

    splits = split_lgd_by_vintage(df)
    logger.info(
        f"Vintage split: train={len(splits['train']):,}, "
        f"val={len(splits['val']):,}, test={len(splits['test']):,}"
    )

    X_train, y_flag_train, y_rate_train = split_lgd_features_and_targets(splits["train"])
    X_val, y_flag_val, y_rate_val = split_lgd_features_and_targets(splits["val"])
    X_test, y_flag_test, y_rate_test = split_lgd_features_and_targets(splits["test"])

    config = HurdleLGDConfig(
        reg_n_estimators=args.n_estimators,
        reg_max_depth=args.max_depth,
        reg_learning_rate=args.learning_rate,
    )
    logger.info(f"Config: {config.to_dict()}")

    model = HurdleLGDModel(config)
    model.fit(
        X_train, y_flag_train, y_rate_train,
        X_val=X_val, recovered_flag_val=y_flag_val, recovery_rate_val=y_rate_val,
    )

    results: dict[str, dict] = {"config": config.to_dict()}
    for label, X, y_flag, y_rate in [
        ("train", X_train, y_flag_train, y_rate_train),
        ("val", X_val, y_flag_val, y_rate_val),
        ("test", X_test, y_flag_test, y_rate_test),
    ]:
        results[label] = evaluate_split(model, X, y_flag, y_rate, label, logger)

    save_lgd_model(model, MODEL_PATH)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(f"Saved LGD metrics to {METRICS_PATH}")

    print("\nSUCCESS: LGD model + metrics written")


if __name__ == "__main__":
    main()


