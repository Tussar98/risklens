"""Calibration analysis for the PD models.

Loads the logistic and XGBoost models, computes reliability curves on val and
test sets, fits an isotonic calibrator on val for XGBoost, and produces:
  - docs/figures/calibration/01_reliability_logistic_test.png
  - docs/figures/calibration/02_reliability_xgboost_test_uncalibrated.png
  - docs/figures/calibration/03_reliability_xgboost_test_calibrated.png
  - docs/figures/calibration/04_calibration_comparison.png
  - artifacts/models/xgboost_isotonic_calibrator.joblib
  - artifacts/reports/calibration_summary.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from risklens.data.splits import split_by_vintage, split_features_and_target
from risklens.evaluation.metrics import compute_pd_metrics
from risklens.models.calibration import (
    apply_calibrator,
    compute_reliability_curve,
    expected_calibration_error,
    fit_isotonic_calibrator,
)
from risklens.models.pd_logistic import load_pipeline as load_logistic
from risklens.models.pd_logistic import predict_proba_default as predict_logistic
from risklens.models.pd_xgboost import load_pipeline as load_xgb
from risklens.models.pd_xgboost import predict_proba_default as predict_xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "interim" / "loans_filtered.parquet"
LOGISTIC_PATH = PROJECT_ROOT / "artifacts" / "models" / "pd_logistic.joblib"
XGB_PATH = PROJECT_ROOT / "artifacts" / "models" / "pd_xgboost.joblib"
CALIBRATOR_PATH = (
    PROJECT_ROOT / "artifacts" / "models" / "xgboost_isotonic_calibrator.joblib"
)
SUMMARY_PATH = PROJECT_ROOT / "artifacts" / "reports" / "calibration_summary.json"
FIG_DIR = PROJECT_ROOT / "docs" / "figures" / "calibration"

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
})


def plot_reliability(
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    title: str,
    out_path: Path,
) -> None:
    """Plot one or more reliability curves on the same axes.

    Each entry in `curves` is (label, (mean_predicted, observed_rate, bin_count)).
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")

    for label, (mean_pred, obs_rate, bin_count) in curves.items():
        sizes = 20 + 80 * (bin_count / bin_count.max())
        ax.plot(mean_pred, obs_rate, marker="o", linewidth=1.5, label=label, alpha=0.85)
        ax.scatter(mean_pred, obs_rate, s=sizes, alpha=0.6)

    ax.set_xlabel("Mean predicted PD (bin)")
    ax.set_ylabel("Observed default rate (bin)")
    ax.set_title(title)
    ax.set_xlim(0, max(0.6, ax.get_xlim()[1]))
    ax.set_ylim(0, max(0.6, ax.get_ylim()[1]))
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger = logging.getLogger(__name__)
    logger.info(f"  saved {out_path.name}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    splits = split_by_vintage(df)
    X_val, y_val = split_features_and_target(splits.val)
    X_test, y_test = split_features_and_target(splits.test)

    logger.info("Loading saved models...")
    logistic = load_logistic(LOGISTIC_PATH)
    xgb = load_xgb(XGB_PATH)

    logger.info("Scoring val and test on both models...")
    p_logistic_test = predict_logistic(logistic, X_test)
    p_xgb_val = predict_xgb(xgb, X_val)
    p_xgb_test = predict_xgb(xgb, X_test)

    logger.info("Fitting isotonic calibrator on val (XGBoost)...")
    calibrator = fit_isotonic_calibrator(y_val.to_numpy(), p_xgb_val)
    p_xgb_test_cal = apply_calibrator(calibrator, p_xgb_test)

    rc_logistic = compute_reliability_curve(y_test.to_numpy(), p_logistic_test, n_bins=10)
    rc_xgb_raw = compute_reliability_curve(y_test.to_numpy(), p_xgb_test, n_bins=10)
    rc_xgb_cal = compute_reliability_curve(y_test.to_numpy(), p_xgb_test_cal, n_bins=10)

    logistic_curve = (
        rc_logistic.mean_predicted,
        rc_logistic.observed_rate,
        rc_logistic.bin_count,
    )
    xgb_raw_curve = (
        rc_xgb_raw.mean_predicted,
        rc_xgb_raw.observed_rate,
        rc_xgb_raw.bin_count,
    )
    xgb_cal_curve = (
        rc_xgb_cal.mean_predicted,
        rc_xgb_cal.observed_rate,
        rc_xgb_cal.bin_count,
    )

    plot_reliability(
        {"Logistic baseline": logistic_curve},
        title="Reliability: Logistic baseline (test, n=40,979)",
        out_path=FIG_DIR / "01_reliability_logistic_test.png",
    )
    plot_reliability(
        {"XGBoost (raw)": xgb_raw_curve},
        title="Reliability: XGBoost uncalibrated (test, n=40,979)",
        out_path=FIG_DIR / "02_reliability_xgboost_test_uncalibrated.png",
    )
    plot_reliability(
        {"XGBoost (isotonic-calibrated)": xgb_cal_curve},
        title="Reliability: XGBoost after isotonic calibration (test, n=40,979)",
        out_path=FIG_DIR / "03_reliability_xgboost_test_calibrated.png",
    )
    plot_reliability(
        {
            "Logistic": logistic_curve,
            "XGBoost raw": xgb_raw_curve,
            "XGBoost calibrated": xgb_cal_curve,
        },
        title="Reliability comparison: test set (n=40,979)",
        out_path=FIG_DIR / "04_calibration_comparison.png",
    )

    summary = {
        "test_metrics": {
            "logistic": {
                "ece": expected_calibration_error(y_test.to_numpy(), p_logistic_test),
                **compute_pd_metrics(y_test.to_numpy(), p_logistic_test).to_dict(),
            },
            "xgboost_raw": {
                "ece": expected_calibration_error(y_test.to_numpy(), p_xgb_test),
                **compute_pd_metrics(y_test.to_numpy(), p_xgb_test).to_dict(),
            },
            "xgboost_calibrated": {
                "ece": expected_calibration_error(y_test.to_numpy(), p_xgb_test_cal),
                **compute_pd_metrics(y_test.to_numpy(), p_xgb_test_cal).to_dict(),
            },
        }
    }

    logger.info("\n=== Calibration summary (test) ===")
    for name, m in summary["test_metrics"].items():
        logger.info(
            f"  {name:<22}  AUC={m['auc']:.4f}  KS={m['ks']:.4f}  "
            f"Brier={m['brier']:.4f}  ECE={m['ece']:.4f}"
        )

    joblib.dump(calibrator, CALIBRATOR_PATH)
    logger.info(f"Saved isotonic calibrator to {CALIBRATOR_PATH}")

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Saved calibration summary to {SUMMARY_PATH}")

    print("\nSUCCESS: 4 figures + calibrator + summary written")


if __name__ == "__main__":
    main()
