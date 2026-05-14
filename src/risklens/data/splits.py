"""Vintage-based train/val/test splits for the PD model.

Cohorts are defined by loan origination quarter (issue_d). Random splits would
leak future information into training. Vintage splits simulate the deployment
scenario: train on what is observed, validate and test on what comes next.

Split policy (from project plan, with right-censoring adjustment):
  - Train:    2013-Q1 through 2016-Q4
  - Validate: 2017-Q1 through 2017-Q4
  - Test:     2018-Q1 through 2018-Q2

Excluded:
  - Pre-2013 vintages: cold-start era with noisy default rates, small volume.
  - 2018 Q3-Q4: right-censored - too few months of observation before the data
    ends (2018-12-31) for terminal-state labels to be reliable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


TRAIN_YEARS: tuple[int, ...] = (2013, 2014, 2015, 2016)
VAL_YEARS: tuple[int, ...] = (2017,)
TEST_YEARS: tuple[int, ...] = (2018,)
TEST_QUARTERS_EXCLUDED: frozenset[str] = frozenset({"2018Q3", "2018Q4"})


@dataclass
class SplitResult:
    """Container for the three-way split."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> str:
        lines = [
            f"  Train: {len(self.train):>10,} loans, target rate {self.train['target'].mean():.4f}",
            f"  Val:   {len(self.val):>10,} loans, target rate {self.val['target'].mean():.4f}",
            f"  Test:  {len(self.test):>10,} loans, target rate {self.test['target'].mean():.4f}",
        ]
        return "\n".join(lines)


def split_by_vintage(
    df: pd.DataFrame,
    train_years: tuple[int, ...] = TRAIN_YEARS,
    val_years: tuple[int, ...] = VAL_YEARS,
    test_years: tuple[int, ...] = TEST_YEARS,
    test_quarters_excluded: frozenset[str] = TEST_QUARTERS_EXCLUDED,
) -> SplitResult:
    """Split a filtered dataframe into train/val/test by vintage_year + vintage_quarter.

    Requires columns: vintage_year (int-like), vintage_quarter (string like 2018Q1).
    """
    required = {"vintage_year", "vintage_quarter", "target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Required columns missing for vintage split: {sorted(missing)}")

    has_year = df["vintage_year"].notna()
    df = df.loc[has_year].copy()
    df["vintage_year"] = df["vintage_year"].astype(int)

    train_mask = df["vintage_year"].isin(train_years)
    val_mask = df["vintage_year"].isin(val_years)
    test_mask = (
        df["vintage_year"].isin(test_years)
        & ~df["vintage_quarter"].isin(test_quarters_excluded)
    )

    result = SplitResult(
        train=df.loc[train_mask].reset_index(drop=True),
        val=df.loc[val_mask].reset_index(drop=True),
        test=df.loc[test_mask].reset_index(drop=True),
    )

    logger.info("Vintage split:")
    for line in result.summary().splitlines():
        logger.info(line)

    if len(result.train) == 0:
        raise ValueError("Train split is empty; check TRAIN_YEARS and input data.")
    if len(result.val) == 0:
        raise ValueError("Val split is empty; check VAL_YEARS and input data.")
    if len(result.test) == 0:
        raise ValueError("Test split is empty; check TEST_YEARS and input data.")

    return result


def split_features_and_target(
    df: pd.DataFrame,
    target_col: str = "target",
    drop_cols: tuple[str, ...] = ("target", "loan_status", "issue_d"),
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from target, dropping pipeline-only columns.

    `loan_status` is dropped to prevent it from being used as a feature
    (it derives the target). `issue_d` is dropped because it is replaced by
    the parsed `issue_dt` / `vintage_year` / `vintage_quarter` columns, and
    the parsed columns themselves should not be model features.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in dataframe")

    y = df[target_col].astype(int)

    extra_drops = ("vintage_year", "vintage_quarter", "issue_dt")
    cols_to_drop = [c for c in (*drop_cols, *extra_drops) if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    return X, y


__all__ = [
    "TRAIN_YEARS",
    "VAL_YEARS",
    "TEST_YEARS",
    "TEST_QUARTERS_EXCLUDED",
    "SplitResult",
    "split_by_vintage",
    "split_features_and_target",
]
