"""Transform the raw Lending Club CSV into a filtered Parquet of completed loans."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from risklens.features.leakage_blacklist import POST_ORIGINATION_COLUMNS

logger = logging.getLogger(__name__)

TERMINAL_STATUSES: frozenset[str] = frozenset({"Fully Paid", "Charged Off", "Default"})
CHARGED_OFF_STATUSES: frozenset[str] = frozenset({"Charged Off", "Default"})

_KEEP_FOR_PIPELINE: frozenset[str] = frozenset({"loan_status", "issue_d"})


def load_raw_chunks(
    raw_csv_path: Path,
    chunksize: int = 200_000,
) -> Iterator[pd.DataFrame]:
    """Yield chunks of the raw gzipped Lending Club CSV."""
    logger.info(f"Reading {raw_csv_path} in chunks of {chunksize:,} rows")
    cumulative = 0
    reader = pd.read_csv(
        raw_csv_path,
        compression="gzip",
        low_memory=False,
        chunksize=chunksize,
    )
    for i, chunk in enumerate(reader):
        cumulative += len(chunk)
        logger.info(f"  chunk {i}: {len(chunk):,} rows (cumulative: {cumulative:,})")
        yield chunk


def filter_to_terminal_loans(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows whose loan_status is in TERMINAL_STATUSES."""
    mask = df["loan_status"].isin(TERMINAL_STATUSES)
    return df.loc[mask].copy()


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add a binary `target` column: 1 if Charged Off or Default, else 0."""
    df = df.copy()
    df["target"] = df["loan_status"].isin(CHARGED_OFF_STATUSES).astype("int8")
    return df


def drop_leakage_columns(
    df: pd.DataFrame,
    keep_target_source: bool = True,
) -> pd.DataFrame:
    """Drop post-origination columns except those needed by the pipeline."""
    to_drop = POST_ORIGINATION_COLUMNS.intersection(df.columns)
    if keep_target_source:
        to_drop = to_drop - _KEEP_FOR_PIPELINE
    else:
        to_drop = to_drop - frozenset({"issue_d"})

    dropped = sorted(to_drop)
    logger.info(f"Dropping {len(dropped)} leakage columns")
    logger.debug(f"  columns dropped: {dropped}")
    return df.drop(columns=list(to_drop))


def parse_issue_date(df: pd.DataFrame) -> pd.DataFrame:
    """Parse `issue_d` (e.g. 'Dec-2015') into datetime + vintage fields."""
    df = df.copy()
    df["issue_dt"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")

    n_unparsed = df["issue_dt"].isna().sum()
    if n_unparsed:
        logger.warning(f"{n_unparsed:,} rows had unparseable issue_d values")

    df["vintage_year"] = df["issue_dt"].dt.year.astype("Int16")
    df["vintage_quarter"] = (
        df["issue_dt"].dt.year.astype("Int16").astype(str)
        + "Q"
        + df["issue_dt"].dt.quarter.astype("Int8").astype(str)
    )
    return df


def build_filtered_dataset(
    raw_csv_path: Path,
    output_path: Path,
    chunksize: int = 200_000,
) -> Path:
    """End-to-end: stream raw CSV to filtered Parquet."""
    chunks: list[pd.DataFrame] = []
    for chunk in load_raw_chunks(raw_csv_path, chunksize=chunksize):
        chunk = filter_to_terminal_loans(chunk)
        if chunk.empty:
            continue
        chunk = add_target(chunk)
        chunk = drop_leakage_columns(chunk, keep_target_source=True)
        chunk = parse_issue_date(chunk)
        chunks.append(chunk)

    if not chunks:
        raise ValueError("No terminal-state loans found in the raw data.")

    df = pd.concat(chunks, ignore_index=True)

    target_rate = df["target"].mean()
    target_count = int(df["target"].sum())
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    logger.info(f"Filtered dataset shape: {df.shape}")
    logger.info(f"Target rate: {target_rate:.4f} ({target_count:,} of {len(df):,})")
    logger.info(
        f"Vintage years: {int(df['vintage_year'].min())} - {int(df['vintage_year'].max())}"
    )
    logger.info(f"In-memory size: {memory_mb:.1f} MB")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
    logger.info(f"Wrote Parquet to {output_path}")

    return output_path
