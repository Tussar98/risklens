"""Expected Loss analysis.

Builds the per-loan EL table, aggregates by vintage quarter, and produces:
  - artifacts/reports/expected_loss_per_loan.parquet
  - artifacts/reports/expected_loss_by_vintage.csv
  - artifacts/reports/expected_loss_summary.json
  - docs/figures/expected_loss/01_predicted_vs_observed_by_vintage.png
  - docs/figures/expected_loss/02_loss_rate_by_grade.png
  - docs/figures/expected_loss/03_loss_rate_decomposition.png
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from risklens.models.expected_loss import (
    aggregate_by_vintage_quarter,
    build_expected_loss_table,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv.gz"
PD_MODEL = PROJECT_ROOT / "artifacts" / "models" / "pd_xgboost.joblib"
LGD_MODEL = PROJECT_ROOT / "artifacts" / "models" / "lgd_hurdle.joblib"

PER_LOAN_OUT = PROJECT_ROOT / "artifacts" / "reports" / "expected_loss_per_loan.parquet"
VINTAGE_OUT = PROJECT_ROOT / "artifacts" / "reports" / "expected_loss_by_vintage.csv"
SUMMARY_OUT = PROJECT_ROOT / "artifacts" / "reports" / "expected_loss_summary.json"
FIG_DIR = PROJECT_ROOT / "docs" / "figures" / "expected_loss"

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
})


def plot_predicted_vs_observed(vintage_df: pd.DataFrame, out_path: Path) -> None:
    """Predicted vs observed loss rate by vintage quarter, with volume bars."""
    fig, ax1 = plt.subplots(figsize=(13, 5.5))
    ax2 = ax1.twinx()

    x = range(len(vintage_df))
    ax1.bar(
        x, vintage_df["total_funded"] / 1e6,
        color="lightsteelblue", alpha=0.5, label="Funded (USD millions)",
    )
    ax2.plot(
        x, vintage_df["expected_loss_rate"],
        color="firebrick", marker="o", linewidth=2, label="Predicted EL rate",
    )
    ax2.plot(
        x, vintage_df["observed_loss_rate"],
        color="darkgreen", marker="s", linewidth=2, label="Observed loss rate",
    )

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(vintage_df["vintage_quarter"], rotation=60, ha="right")
    ax1.set_xlabel("Vintage quarter")
    ax1.set_ylabel("Funded volume (USD millions)")
    ax2.set_ylabel("Loss rate")
    ax2.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax1.set_title("Portfolio Expected Loss: predicted vs observed by vintage quarter")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_loss_decomposition(vintage_df: pd.DataFrame, out_path: Path) -> None:
    """Mean PD and mean LGD by vintage, on twin axes."""
    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax2 = ax1.twinx()

    x = range(len(vintage_df))
    ax1.plot(
        x, vintage_df["mean_pd"],
        color="firebrick", marker="o", linewidth=2, label="Mean PD",
    )
    ax2.plot(
        x, vintage_df["mean_lgd"],
        color="navy", marker="s", linewidth=2, label="Mean LGD",
    )

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(vintage_df["vintage_quarter"], rotation=60, ha="right")
    ax1.set_xlabel("Vintage quarter")
    ax1.set_ylabel("Mean PD", color="firebrick")
    ax2.set_ylabel("Mean LGD", color="navy")
    ax1.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax2.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax1.set_title("EL decomposition: mean PD and mean LGD by vintage quarter")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PER_LOAN_OUT.parent.mkdir(parents=True, exist_ok=True)

    el_table = build_expected_loss_table(RAW_CSV, PD_MODEL, LGD_MODEL)
    logger.info("Writing per-loan EL table...")
    el_table.to_parquet(PER_LOAN_OUT, engine="pyarrow", compression="snappy", index=False)
    logger.info(f"  saved {PER_LOAN_OUT.name}")

    vintage_df = aggregate_by_vintage_quarter(el_table)
    vintage_df.to_csv(VINTAGE_OUT, index=False)
    logger.info(f"Saved vintage-level aggregates to {VINTAGE_OUT.name}")

    plot_predicted_vs_observed(vintage_df, FIG_DIR / "01_predicted_vs_observed_by_vintage.png")
    logger.info("  saved 01_predicted_vs_observed_by_vintage.png")
    plot_loss_decomposition(vintage_df, FIG_DIR / "03_loss_rate_decomposition.png")
    logger.info("  saved 03_loss_rate_decomposition.png")

    total_funded = float(el_table["loan_amnt"].sum())
    total_expected = float(el_table["expected_loss"].sum())
    total_observed = float(el_table["observed_loss"].sum())
    summary = {
        "n_loans": int(len(el_table)),
        "total_funded_usd": total_funded,
        "total_expected_loss_usd": total_expected,
        "total_observed_loss_usd": total_observed,
        "portfolio_expected_loss_rate": total_expected / total_funded,
        "portfolio_observed_loss_rate": total_observed / total_funded,
        "portfolio_loss_rate_error": (total_expected - total_observed) / total_funded,
        "best_vintage": {
            "quarter": vintage_df.loc[vintage_df["observed_loss_rate"].idxmin(), "vintage_quarter"],
            "observed_loss_rate": float(vintage_df["observed_loss_rate"].min()),
        },
        "worst_vintage": {
            "quarter": vintage_df.loc[vintage_df["observed_loss_rate"].idxmax(), "vintage_quarter"],
            "observed_loss_rate": float(vintage_df["observed_loss_rate"].max()),
        },
    }

    SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Saved summary to {SUMMARY_OUT.name}")

    logger.info("\n=== Portfolio EL summary ===")
    logger.info(f"  Loans:                       {summary['n_loans']:,}")
    logger.info(f"  Total funded:        ${total_funded:>15,.0f}")
    logger.info(f"  Total expected loss: ${total_expected:>15,.0f}")
    logger.info(f"  Total observed loss: ${total_observed:>15,.0f}")
    logger.info(f"  Predicted loss rate: {summary['portfolio_expected_loss_rate']:.4%}")
    logger.info(f"  Observed loss rate:  {summary['portfolio_observed_loss_rate']:.4%}")
    logger.info(f"  Best vintage:  {summary['best_vintage']['quarter']} "
                f"({summary['best_vintage']['observed_loss_rate']:.4%})")
    logger.info(f"  Worst vintage: {summary['worst_vintage']['quarter']} "
                f"({summary['worst_vintage']['observed_loss_rate']:.4%})")

    print("\nSUCCESS: per-loan EL + vintage aggregates + 2 figures + summary written")


if __name__ == "__main__":
    main()
