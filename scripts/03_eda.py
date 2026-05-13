"""Exploratory data analysis: produce six figures + summary stats for the filtered dataset.

Reads data/interim/loans_filtered.parquet and writes:
  - docs/figures/eda/01_target_rate_by_grade.png
  - docs/figures/eda/02_target_rate_by_fico_band.png
  - docs/figures/eda/03_target_rate_by_purpose.png
  - docs/figures/eda/04_target_rate_by_term.png
  - docs/figures/eda/05_vintage_volume_and_default_rate.png
  - docs/figures/eda/06_missingness_heatmap.png
  - docs/figures/eda/summary_stats.txt
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "interim" / "loans_filtered.parquet"
FIG_DIR = PROJECT_ROOT / "docs" / "figures" / "eda"

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})


def load_data() -> pd.DataFrame:
    """Load the filtered dataset and parse a few raw columns for analysis."""
    logger.info(f"Loading {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)

    df["fico_mid"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df["fico_band"] = pd.cut(
        df["fico_mid"],
        bins=[0, 660, 700, 740, 780, 850],
        labels=["<660", "660-699", "700-739", "740-779", "780+"],
    )
    df["term_months"] = df["term"].str.extract(r"(\d+)").astype("Int16")

    logger.info(f"Loaded {len(df):,} loans, {df['target'].mean():.2%} target rate")
    return df


def plot_target_rate_by_grade(df: pd.DataFrame, out: Path) -> None:
    """Default rate by LC grade, with volume bars on a secondary axis."""
    g = df.groupby("grade").agg(
        n=("target", "size"),
        default_rate=("target", "mean"),
    ).sort_index()

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.bar(g.index, g["n"], color="lightsteelblue", alpha=0.6, label="Loan count")
    ax2.plot(
        g.index, g["default_rate"],
        color="firebrick", marker="o", linewidth=2, label="Default rate",
    )

    ax1.set_xlabel("LC Grade")
    ax1.set_ylabel("Loan count")
    ax2.set_ylabel("Default rate")
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.set_title("Default rate by LC grade (volume bars + rate line)")

    for i, rate in enumerate(g["default_rate"]):
        ax2.annotate(
            f"{rate:.1%}", (i, rate),
            textcoords="offset points", xytext=(0, 8),
            ha="center", fontsize=9, color="firebrick",
        )

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    logger.info(f"  saved {out.name}")


def plot_target_rate_by_fico(df: pd.DataFrame, out: Path) -> None:
    """Default rate by FICO band."""
    g = df.groupby("fico_band", observed=True).agg(
        n=("target", "size"),
        default_rate=("target", "mean"),
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(g.index.astype(str), g["default_rate"], color="steelblue")
    ax.set_xlabel("FICO band (origination)")
    ax.set_ylabel("Default rate")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Default rate by FICO band")

    for bar, n, rate in zip(bars, g["n"], g["default_rate"], strict=True):
        ax.annotate(
            f"{rate:.1%}\n(n={n:,})",
            (bar.get_x() + bar.get_width() / 2, rate),
            textcoords="offset points", xytext=(0, 5),
            ha="center", fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    logger.info(f"  saved {out.name}")


def plot_target_rate_by_purpose(df: pd.DataFrame, out: Path) -> None:
    """Default rate by loan purpose, sorted, top 12."""
    g = df.groupby("purpose").agg(
        n=("target", "size"),
        default_rate=("target", "mean"),
    ).sort_values("n", ascending=False).head(12).sort_values("default_rate")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(g.index, g["default_rate"], color="steelblue")
    ax.set_xlabel("Default rate")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Default rate by loan purpose (top 12 by volume)")

    for i, (rate, n) in enumerate(zip(g["default_rate"], g["n"], strict=True)):
        ax.annotate(
            f"{rate:.1%} (n={n:,})", (rate, i),
            textcoords="offset points", xytext=(5, 0),
            va="center", fontsize=9,
        )

    ax.set_xlim(0, max(g["default_rate"]) * 1.25)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    logger.info(f"  saved {out.name}")


def plot_target_rate_by_term(df: pd.DataFrame, out: Path) -> None:
    """Default rate by loan term (36 vs 60 months)."""
    g = df.groupby("term_months").agg(
        n=("target", "size"),
        default_rate=("target", "mean"),
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        g.index.astype(str) + "mo",
        g["default_rate"],
        color=["steelblue", "indianred"],
    )
    ax.set_xlabel("Loan term")
    ax.set_ylabel("Default rate")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Default rate by loan term")

    for bar, n, rate in zip(bars, g["n"], g["default_rate"], strict=True):
        ax.annotate(
            f"{rate:.1%}\n(n={n:,})",
            (bar.get_x() + bar.get_width() / 2, rate),
            textcoords="offset points", xytext=(0, 5),
            ha="center", fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    logger.info(f"  saved {out.name}")


def plot_vintage_volume_and_rate(df: pd.DataFrame, out: Path) -> None:
    """Origination volume and default rate by vintage quarter."""
    g = df.dropna(subset=["vintage_quarter"]).groupby("vintage_quarter").agg(
        n=("target", "size"),
        default_rate=("target", "mean"),
    ).sort_index()

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    ax1.bar(range(len(g)), g["n"], color="lightsteelblue", alpha=0.6, label="Volume")
    ax2.plot(
        range(len(g)), g["default_rate"],
        color="firebrick", marker="o", linewidth=2, label="Default rate",
    )

    ax1.set_xticks(range(len(g)))
    ax1.set_xticklabels(g.index, rotation=60, ha="right")
    ax1.set_xlabel("Vintage quarter")
    ax1.set_ylabel("Loan count")
    ax2.set_ylabel("Default rate")
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.set_title("Origination volume and default rate by vintage quarter")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    logger.info(f"  saved {out.name}")


def plot_missingness(df: pd.DataFrame, out: Path) -> None:
    """Bar chart of missingness by column (columns with >0% missing only)."""
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(miss))))
    ax.barh(miss.index[::-1], miss.values[::-1], color="darkorange")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Missing fraction")
    ax.set_title(f"Column missingness ({len(miss)} columns with any missing values)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    logger.info(f"  saved {out.name}")


def write_summary_stats(df: pd.DataFrame, out: Path) -> None:
    """Write a plain-text summary of dataset shape, target, vintage, dtypes."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("EDA Summary — RiskLens filtered dataset")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    lines.append(f"Target rate: {df['target'].mean():.4%}")
    lines.append(
        f"Charge-offs/Defaults: {int(df['target'].sum()):,} of {len(df):,}"
    )
    lines.append("")
    lines.append("Vintage range:")
    vy = df["vintage_year"].dropna()
    lines.append(f"  {int(vy.min())} – {int(vy.max())}")
    lines.append("")
    lines.append("Loans per vintage year:")
    for year, count in df.groupby("vintage_year").size().items():
        lines.append(f"  {int(year)}: {int(count):>10,}")
    lines.append("")
    lines.append("Loan status distribution (filtered):")
    for status, count in df["loan_status"].value_counts().items():
        lines.append(f"  {status:<15} {int(count):>10,}")
    lines.append("")
    lines.append("Memory usage (MB):")
    lines.append(f"  {df.memory_usage(deep=True).sum() / 1024**2:,.1f}")
    lines.append("")
    lines.append("Columns with any missing values:")
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    for col, frac in miss.items():
        lines.append(f"  {col:<35} {frac:>7.2%}")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"  saved {out.name}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    plot_target_rate_by_grade(df, FIG_DIR / "01_target_rate_by_grade.png")
    plot_target_rate_by_fico(df, FIG_DIR / "02_target_rate_by_fico_band.png")
    plot_target_rate_by_purpose(df, FIG_DIR / "03_target_rate_by_purpose.png")
    plot_target_rate_by_term(df, FIG_DIR / "04_target_rate_by_term.png")
    plot_vintage_volume_and_rate(
        df, FIG_DIR / "05_vintage_volume_and_default_rate.png"
    )
    plot_missingness(df, FIG_DIR / "06_missingness_heatmap.png")
    write_summary_stats(df, FIG_DIR / "summary_stats.txt")

    print(f"\nSUCCESS: 6 figures + summary stats written to {FIG_DIR}")


if __name__ == "__main__":
    main()