"""Expected Loss framework: combine PD, LGD, and EAD into per-loan EL.

For each loan in scope, computes:
  EL = PD x LGD x EAD
where PD comes from the XGBoost classifier, LGD from the hurdle model, and
EAD is approximated by loan_amnt. Also computes observed (realized) loss
for ground-truth comparison.

Inputs:
  - raw_csv_path: the original Lending Club CSV. Re-read here so we can keep
    the loan id field, which was excluded from the filtered Parquet.
  - pd_model_path: serialized PD pipeline (Day 4 artifact).
  - lgd_model_path: serialized hurdle LGD model (Day 5 artifact).

Outputs:
  A pandas DataFrame with one row per loan and columns:
    id, vintage_year, vintage_quarter, loan_status, loan_amnt,
    pd, lgd, expected_loss, observed_loss
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from risklens.data.lgd_data import BORROWER_FEATURE_COLUMNS, RECOVERY_COLUMNS
from risklens.data.load import TERMINAL_STATUSES
from risklens.models.lgd import HurdleLGDModel, load_lgd_model
from risklens.models.pd_xgboost import load_pipeline as load_pd_pipeline
from risklens.models.pd_xgboost import predict_proba_default as predict_pd

logger = logging.getLogger(__name__)


# Loan id + the union of columns needed for PD features, LGD features, and
# observed loss calculation.
_EL_COLUMNS: list[str] = sorted(
    set(BORROWER_FEATURE_COLUMNS) | set(RECOVERY_COLUMNS) | {"id"}
)


def _load_raw_for_el(
    raw_csv_path: Path,
    chunksize: int = 200_000,
) -> Iterator[pd.DataFrame]:
    """Stream raw CSV with only the columns needed for the EL framework."""
    logger.info(f"Reading {raw_csv_path} for EL aggregation")
    cumulative = 0
    reader = pd.read_csv(
        raw_csv_path,
        compression="gzip",
        low_memory=False,
        chunksize=chunksize,
        usecols=_EL_COLUMNS,
    )
    for i, chunk in enumerate(reader):
        cumulative += len(chunk)
        logger.info(f"  chunk {i}: {len(chunk):,} rows (cumulative: {cumulative:,})")
        yield chunk


def _parse_issue_date(df: pd.DataFrame) -> pd.DataFrame:
    """Parse issue_d into issue_dt + vintage fields."""
    df = df.copy()
    df["issue_dt"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["vintage_year"] = df["issue_dt"].dt.year.astype("Int16")
    df["vintage_quarter"] = (
        df["issue_dt"].dt.year.astype("Int16").astype(str)
        + "Q"
        + df["issue_dt"].dt.quarter.astype("Int8").astype(str)
    )
    return df


def _compute_observed_loss(df: pd.DataFrame) -> pd.Series:
    """Per-loan observed (realized) loss in dollars.

    For Fully Paid: 0 (no loss).
    For Charged Off / Default: funded_amnt - net_recovery, floored at 0.
                                  where net_recovery = recoveries - collection_recovery_fee
    """
    is_loss = df["loan_status"].isin({"Charged Off", "Default"})
    net_recovery = (
        df["recoveries"].fillna(0.0) - df["collection_recovery_fee"].fillna(0.0)
    ).clip(lower=0.0)
    loss = (df["funded_amnt"].fillna(0.0) - net_recovery).clip(lower=0.0)
    return loss.where(is_loss, 0.0)


def build_expected_loss_table(
    raw_csv_path: Path,
    pd_model_path: Path,
    lgd_model_path: Path,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """Compute per-loan EL by joining PD, LGD, EAD, and observed-loss columns.

    Loads both models once, then streams the raw CSV in chunks. For each chunk:
      - filter to terminal-state loans (Fully Paid, Charged Off, Default)
      - parse vintage fields
      - score with PD model -> p_default
      - score with LGD model -> lgd
      - compute expected_loss = p_default * lgd * loan_amnt
      - compute observed_loss
    Concatenates and returns the full dataframe.
    """
    logger.info("Loading PD pipeline...")
    pd_pipe = load_pd_pipeline(pd_model_path)
    logger.info("Loading LGD model...")
    lgd_model: HurdleLGDModel = load_lgd_model(lgd_model_path)

    chunks: list[pd.DataFrame] = []
    for chunk in _load_raw_for_el(raw_csv_path, chunksize=chunksize):
        chunk = chunk[chunk["loan_status"].isin(TERMINAL_STATUSES)].copy()
        if chunk.empty:
            continue
        chunk = _parse_issue_date(chunk)

        # PD model expects the same feature schema as the PD training data,
        # which kept loan_status, issue_d, issue_dt. The pipeline drops them
        # internally via the leakage blacklist + remainder='drop'.
        p_default = predict_pd(pd_pipe, chunk)

        # LGD model uses the same feature pipeline so X structure matches.
        lgd_per_loan = lgd_model.predict_lgd(chunk)

        ead = chunk["loan_amnt"].astype(float).to_numpy()
        expected_loss = p_default * lgd_per_loan * ead
        observed_loss = _compute_observed_loss(chunk).to_numpy()

        result = pd.DataFrame({
            "id": chunk["id"].astype(str).to_numpy(),
            "vintage_year": chunk["vintage_year"].to_numpy(),
            "vintage_quarter": chunk["vintage_quarter"].to_numpy(),
            "loan_status": chunk["loan_status"].to_numpy(),
            "loan_amnt": ead,
            "pd": p_default,
            "lgd": lgd_per_loan,
            "expected_loss": expected_loss,
            "observed_loss": observed_loss,
        })
        chunks.append(result)

    if not chunks:
        raise ValueError("No terminal-state loans found for EL aggregation.")

    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"EL table built: {len(df):,} loans")
    logger.info(f"  Total funded:    ${df['loan_amnt'].sum():>15,.0f}")
    logger.info(f"  Total expected:  ${df['expected_loss'].sum():>15,.0f}")
    logger.info(f"  Total observed:  ${df['observed_loss'].sum():>15,.0f}")
    return df


def aggregate_by_vintage_quarter(el_table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-loan EL to vintage-quarter cohorts."""
    df = el_table.dropna(subset=["vintage_quarter"]).copy()
    grouped = df.groupby("vintage_quarter").agg(
        n_loans=("id", "size"),
        total_funded=("loan_amnt", "sum"),
        total_expected_loss=("expected_loss", "sum"),
        total_observed_loss=("observed_loss", "sum"),
        mean_pd=("pd", "mean"),
        mean_lgd=("lgd", "mean"),
    ).reset_index()
    grouped["expected_loss_rate"] = grouped["total_expected_loss"] / grouped["total_funded"]
    grouped["observed_loss_rate"] = grouped["total_observed_loss"] / grouped["total_funded"]
    grouped["loss_rate_error"] = grouped["expected_loss_rate"] - grouped["observed_loss_rate"]
    return grouped.sort_values("vintage_quarter").reset_index(drop=True)


__all__ = [
    "build_expected_loss_table",
    "aggregate_by_vintage_quarter",
]

