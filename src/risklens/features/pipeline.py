"""Feature pipeline: raw filtered dataset to model-ready numeric matrix."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from risklens.features.leakage_blacklist import POST_ORIGINATION_COLUMNS

logger = logging.getLogger(__name__)


NUMERIC_FEATURES: list[str] = [
    "loan_amnt",
    "installment",
    "annual_inc",
    "dti",
    "fico_range_low",
    "fico_range_high",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "total_acc",
    "pub_rec_bankruptcies",
    "tax_liens",
    "acc_open_past_24mths",
]

NUMERIC_INFORMATIVE_MISSING: list[str] = [
    "mths_since_last_delinq",
    "mths_since_last_record",
    "mths_since_last_major_derog",
]

PERCENT_FEATURES: list[str] = ["int_rate", "revol_util"]

CATEGORICAL_FEATURES: list[str] = [
    "home_ownership",
    "purpose",
    "verification_status",
    "initial_list_status",
    "addr_state",
]

ORDINAL_EMP_LENGTH: list[str] = [
    "n/a",
    "< 1 year",
    "1 year",
    "2 years",
    "3 years",
    "4 years",
    "5 years",
    "6 years",
    "7 years",
    "8 years",
    "9 years",
    "10+ years",
]

DERIVED_FEATURES: list[str] = [
    "term_months",
    "credit_history_months",
    "is_joint_application",
    "has_delinq",
    "has_record",
    "has_major_derog",
]


@dataclass
class PreprocessRawColumns(BaseEstimator, TransformerMixin):
    """Parse raw LC columns into clean numeric form."""

    issue_date_col: str = "issue_dt"

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> PreprocessRawColumns:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for col in PERCENT_FEATURES:
            if col in X.columns:
                X[col] = (
                    X[col]
                    .astype(str)
                    .str.replace("%", "", regex=False)
                    .str.strip()
                    .replace({"nan": np.nan, "": np.nan})
                    .astype(float)
                )

        if "term" in X.columns:
            X["term_months"] = (
                X["term"]
                .astype(str)
                .str.extract(r"(\d+)", expand=False)
                .astype("Int16")
                .astype(float)
            )

        if "earliest_cr_line" in X.columns and self.issue_date_col in X.columns:
            earliest = pd.to_datetime(
                X["earliest_cr_line"], format="%b-%Y", errors="coerce"
            )
            issue = pd.to_datetime(X[self.issue_date_col], errors="coerce")
            delta_days = (issue - earliest).dt.days
            X["credit_history_months"] = (delta_days / 30.4375).round().astype(float)

        for col in NUMERIC_INFORMATIVE_MISSING:
            indicator_col = "has_" + col.replace("mths_since_last_", "")
            if col in X.columns:
                X[indicator_col] = X[col].notna().astype("int8")
                X[col] = X[col].fillna(0.0)
            else:
                X[indicator_col] = 0

        joint_cols = [
            "annual_inc_joint",
            "dti_joint",
            "verification_status_joint",
            "revol_bal_joint",
        ]
        present_joint = [c for c in joint_cols if c in X.columns]
        if present_joint:
            X["is_joint_application"] = (
                X[present_joint].notna().any(axis=1).astype("int8")
            )
        else:
            X["is_joint_application"] = 0

        leakage_present = POST_ORIGINATION_COLUMNS.intersection(X.columns)
        pipeline_keep = {"loan_status", "issue_d", "issue_dt"}
        to_drop = leakage_present - pipeline_keep
        if to_drop:
            logger.debug(
                f"PreprocessRawColumns dropping {len(to_drop)} blacklisted columns"
            )
            X = X.drop(columns=list(to_drop))

        return X


class _EmpLengthNormalizer(BaseEstimator, TransformerMixin):
    """Fill NaN emp_length with n/a so OrdinalEncoder can place it in the order."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> _EmpLengthNormalizer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "emp_length" in X.columns:
            X["emp_length"] = X["emp_length"].fillna("n/a")
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray(["emp_length"])
        return np.asarray(input_features)


def build_column_transformer() -> ColumnTransformer:
    """Construct the ColumnTransformer that operates after PreprocessRawColumns."""

    numeric_all = (
        NUMERIC_FEATURES
        + NUMERIC_INFORMATIVE_MISSING
        + PERCENT_FEATURES
        + ["term_months", "credit_history_months"]
    )
    binary_indicators = [
        "is_joint_application",
        "has_delinq",
        "has_record",
        "has_major_derog",
    ]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            min_frequency=0.01,
        )),
    ])

    ordinal_pipeline = Pipeline(steps=[
        ("normalize", _EmpLengthNormalizer()),
        ("ordinal", OrdinalEncoder(
            categories=[ORDINAL_EMP_LENGTH],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_all),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
            ("ord", ordinal_pipeline, ["emp_length"]),
            ("bin", "passthrough", binary_indicators),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return transformer


def build_feature_pipeline() -> Pipeline:
    """Full pipeline: raw filtered DataFrame to numeric feature matrix."""
    return Pipeline(steps=[
        ("preprocess", PreprocessRawColumns()),
        ("encode", build_column_transformer()),
    ])


def assert_no_leakage_columns(feature_names: list[str]) -> None:
    """Raise if any feature name matches a blacklisted column."""
    leaked = [f for f in feature_names if f in POST_ORIGINATION_COLUMNS]
    if leaked:
        raise ValueError(
            f"Leakage column(s) present in feature set: {leaked}. "
            f"Check POST_ORIGINATION_COLUMNS and the pipeline."
        )


__all__ = [
    "NUMERIC_FEATURES",
    "NUMERIC_INFORMATIVE_MISSING",
    "PERCENT_FEATURES",
    "CATEGORICAL_FEATURES",
    "ORDINAL_EMP_LENGTH",
    "DERIVED_FEATURES",
    "PreprocessRawColumns",
    "build_column_transformer",
    "build_feature_pipeline",
    "assert_no_leakage_columns",
]
