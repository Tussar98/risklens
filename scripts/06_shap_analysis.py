"""SHAP feature attribution for the XGBoost PD model.

Loads the trained XGBoost pipeline, samples 10,000 test loans, computes SHAP
values, and produces:
  - docs/figures/shap/01_summary_beeswarm.png
  - docs/figures/shap/02_summary_bar.png
  - docs/figures/shap/03_dependence_int_rate.png
  - docs/figures/shap/04_dependence_term_months.png
  - docs/figures/shap/05_dependence_fico_range_low.png
  - docs/figures/shap/06_dependence_dti.png
  - artifacts/reports/shap_summary.csv

SHAP values are computed on a sample for speed; results are stable at this size.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap

from risklens.data.splits import split_by_vintage, split_features_and_target
from risklens.models.pd_xgboost import load_pipeline as load_xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "interim" / "loans_filtered.parquet"
XGB_PATH = PROJECT_ROOT / "artifacts" / "models" / "pd_xgboost.joblib"
FIG_DIR = PROJECT_ROOT / "docs" / "figures" / "shap"
SUMMARY_PATH = PROJECT_ROOT / "artifacts" / "reports" / "shap_summary.csv"

SAMPLE_SIZE = 10_000

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    splits = split_by_vintage(df)
    X_test, y_test = split_features_and_target(splits.test)

    logger.info(f"Sampling {SAMPLE_SIZE:,} test rows for SHAP")
    sample = X_test.sample(SAMPLE_SIZE, random_state=42)

    logger.info("Loading XGBoost pipeline...")
    pipe = load_xgb(XGB_PATH)
    feature_pipeline = pipe.named_steps["features"]
    classifier = pipe.named_steps["classifier"]

    logger.info("Transforming sample through feature pipeline...")
    X_sample_transformed = feature_pipeline.transform(sample)
    feature_names = list(feature_pipeline.named_steps["encode"].get_feature_names_out())
    X_sample_df = pd.DataFrame(X_sample_transformed, columns=feature_names)

    logger.info("Computing SHAP values (Explainer with predict_proba)...")
    background = X_sample_df.sample(100, random_state=0)
    explainer = shap.Explainer(classifier.predict_proba, background)
    shap_explanation = explainer(X_sample_df)
    # predict_proba returns 2 columns; take class-1 SHAP values
    shap_values = shap_explanation.values[:, :, 1]
    logger.info(f"SHAP matrix shape: {shap_values.shape}")

    # 1. Beeswarm summary plot
    logger.info("Generating beeswarm summary plot...")
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample_df,
        max_display=20,
        show=False,
        plot_size=(10, 8),
    )
    plt.title("SHAP feature impact on PD predictions (test sample, n=10,000)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_summary_beeswarm.png")
    plt.close()

    # 2. Bar plot of mean |SHAP|
    logger.info("Generating mean |SHAP| bar plot...")
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample_df,
        plot_type="bar",
        max_display=20,
        show=False,
        plot_size=(10, 8),
    )
    plt.title("Mean |SHAP| feature importance (test sample, n=10,000)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_summary_bar.png")
    plt.close()

    # 3-6. Dependence plots for top features
    dependence_features = [
        ("int_rate", "03_dependence_int_rate.png"),
        ("term_months", "04_dependence_term_months.png"),
        ("fico_range_low", "05_dependence_fico_range_low.png"),
        ("dti", "06_dependence_dti.png"),
    ]
    for feature, filename in dependence_features:
        if feature not in feature_names:
            logger.warning(f"Feature '{feature}' not found, skipping dependence plot")
            continue
        logger.info(f"Generating dependence plot for {feature}...")
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            feature,
            shap_values,
            X_sample_df,
            interaction_index="auto",
            show=False,
        )
        plt.title(f"SHAP dependence: {feature}")
        plt.tight_layout()
        plt.savefig(FIG_DIR / filename)
        plt.close()

    # Summary table
    import numpy as np
    abs_shap = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "mean_shap": shap_values.mean(axis=0),
    })
    abs_shap = abs_shap.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    abs_shap.to_csv(SUMMARY_PATH, index=False)
    logger.info(f"Saved SHAP summary table to {SUMMARY_PATH}")

    logger.info("\n=== Top 15 features by mean |SHAP| ===")
    for _, row in abs_shap.head(15).iterrows():
        direction = "+" if row["mean_shap"] > 0 else "-"
        logger.info(
            f"  {row['feature']:<35}  mean|SHAP|={row['mean_abs_shap']:.4f}  "
            f"(mean SHAP {direction}{abs(row['mean_shap']):.4f})"
        )

    print("\nSUCCESS: 6 SHAP figures + summary table written")


if __name__ == "__main__":
    main()



