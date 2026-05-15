"""Build the LGD-specific dataset from the raw Lending Club CSV.

The leakage-blacklisted PD dataset deliberately strips recovery and funded_amnt
columns since they are post-origination and would leak into the PD target.
LGD modeling needs them as the *target*, so we read the raw CSV again with a
different column whitelist.

Output:
  data/interim/lgd_data.parquet -- one row per Charged Off loan, with borrower
  features (at origination) and recovery fields (post-default).

Target derivation:
  recovery_amount = recoveries - collection_recovery_fee  (net recovery)
  recovery_rate   = recovery_amount / funded_amnt          clipped to [0, 1]
  recovered_flag  = 1 if recovery_amount > 0 else 0        (hurdle target)
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# Loan status values we keep for LGD modeling.
LGD_STATUSES: frozenset[str] = frozenset({"Charged Off"})

# Recovery / EAD columns -- the LGD target source.
RECOVERY_COLUMNS: list[str] = [
    "loan_amnt",
    "funded_amnt",
    "recoveries",
    "collection_recovery_fee",
    "total_pymnt",
    "total_rec_prncp",
]

# Borrower-feature columns -- the LGD model inputs. Same shape as the PD model
# input columns (everything known at origination), so we can reuse the existing
# feature pipeline downstream.
BORROWER_FEATURE_COLUMNS: list[str] = [
    # Loan terms
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade", "purpose",
    # Borrower financials
    "annual_inc", "dti", "emp_length", "home_ownership", "verification_status",
    "annual_inc_joint", "dti_joint", "verification_status_joint", "revol_bal_joint",
    # Credit history
    "fico_range_low", "fico_range_high", "earliest_cr_line",
    "delinq_2yrs", "inq_last_6mths",
    "mths_since_last_delinq", "mths_since_last_record", "mths_since_last_major_derog",
    "open_acc", "total_acc", "pub_rec", "pub_rec_bankruptcies", "tax_liens",
    "acc_open_past_24mths", "revol_bal", "revol_util",
    # Listing
    "initial_list_status", "addr_state",
    # Pipeline / vintage
    "issue_d", "loan_status",
]


def _columns_to_load() -> list[str]:
    """Union of recovery and borrower columns, deduplicated."""
    return sorted(set(BORROWER_FEATURE_COLUMNS) | set(RECOVERY_COLUMNS))


def load_raw_chunks(
    raw_csv_path: Path,
    chunksize: int = 200_000,
) -> Iterator[pd.DataFrame]:
    """Yield chunks of the raw gzipped CSV, with only the columns we need."""
    logger.info(f"Reading {raw_csv_path} in chunks of {chunksize:,} rows")
    cumulative = 0
    reader = pd.read_csv(
        raw_csv_path,
        compression="gzip",
        low_memory=False,
        chunksize=chunksize,
        usecols=_columns_to_load(),
    )
    for i, chunk in enumerate(reader):
        cumulative += len(chunk)
        logger.info(f"  chunk {i}: {len(chunk):,} rows (cumulative: {cumulative:,})")
        yield chunk


def filter_to_charged_off(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows whose loan_status is in LGD_STATUSES."""
    mask = df["loan_status"].isin(LGD_STATUSES)
    return df.loc[mask].copy()


def add_recovery_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Derive recovery_amount, recovery_rate, and recovered_flag.

    recovery_amount   = recoveries - collection_recovery_fee  (net of collection cost)
    recovery_rate     = recovery_amount / funded_amnt          clipped to [0, 1]
    recovered_flag    = 1 if recovery_amount > 0 else 0
    """
    df = df.copy()
    df["recovery_amount"] = df["recoveries"].fillna(0.0) - df["collection_recovery_fee"].fillna(0.0)
    df["recovery_amount"] = df["recovery_amount"].clip(lower=0.0)

    funded = df["funded_amnt"].replace(0, pd.NA)
    rate = (df["recovery_amount"] / funded).astype(float)
    df["recovery_rate"] = rate.clip(lower=0.0, upper=1.0).fillna(0.0)

    df["recovered_flag"] = (df["recovery_amount"] > 0).astype("int8")
    return df


def parse_issue_date(df: pd.DataFrame) -> pd.DataFrame:
    """Parse issue_d into datetime + vintage fields, matching the PD pipeline."""
    df = df.copy()
    df["issue_dt"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["vintage_year"] = df["issue_dt"].dt.year.astype("Int16")
    df["vintage_quarter"] = (
        df["issue_dt"].dt.year.astype("Int16").astype(str)
        + "Q"
        + df["issue_dt"].dt.quarter.astype("Int8").astype(str)
    )
    return df


def build_lgd_dataset(
    raw_csv_path: Path,
    output_path: Path,
    chunksize: int = 200_000,
) -> Path:
    """End-to-end: raw CSV -> filtered to Charged Off -> recovery targets -> Parquet."""
    chunks: list[pd.DataFrame] = []
    for chunk in load_raw_chunks(raw_csv_path, chunksize=chunksize):
        chunk = filter_to_charged_off(chunk)
        if chunk.empty:
            continue
        chunk = add_recovery_targets(chunk)
        chunk = parse_issue_date(chunk)
        chunks.append(chunk)

    if not chunks:
        raise ValueError("No Charged Off loans found in the raw data.")

    df = pd.concat(chunks, ignore_index=True)

    logger.info(f"LGD dataset shape: {df.shape}")
    logger.info(f"Recovered (any amount) rate: {df['recovered_flag'].mean():.4f}")
    logger.info(f"Mean recovery_rate:          {df['recovery_rate'].mean():.4f}")
    logger.info(f"Mean recovery_rate | recovered>0: "
                f"{df.loc[df['recovered_flag'] == 1, 'recovery_rate'].mean():.4f}")
    logger.info(
        f"Vintage range: {int(df['vintage_year'].dropna().min())} - "
        f"{int(df['vintage_year'].dropna().max())}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
    logger.info(f"Wrote LGD Parquet to {output_path}")

    return output_path


__all__ = [
    "LGD_STATUSES",
    "RECOVERY_COLUMNS",
    "BORROWER_FEATURE_COLUMNS",
    "load_raw_chunks",
    "filter_to_charged_off",
    "add_recovery_targets",
    "parse_issue_date",
    "build_lgd_dataset",
]
