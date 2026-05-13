# Data Dictionary

**Project:** RiskLens — Credit Risk Modeling on Lending Club Data
**Dataset:** Lending Club accepted loans, 2007–2018
**Source:** Kaggle dataset `wordsforthewise/lending-club`, file `accepted_2007_to_2018Q4.csv.gz`
**License:** CC0 1.0 Universal (Public Domain)

---

## 1. Target Definition

**Target column:** `target` (binary, int8)

| Value | loan_status                       | Meaning                            |
| ----- | --------------------------------- | ---------------------------------- |
| 0     | Fully Paid                        | Loan repaid in full                |
| 1     | Charged Off                       | Loan written off by Lending Club   |
| 1     | Default                           | 121+ days delinquent (de facto CO) |

**Universe:** Only loans in a terminal state are retained. Loans still in repayment (`Current`, `Late (16-30 days)`, `Late (31-120 days)`, `In Grace Period`, `Issued`) are excluded because their outcome is unknown.

**Horizon:** The plan specifies "Charged Off within 36 months of origination." This implementation uses the looser practitioner standard — any loan that reached a terminal state, regardless of original term. Rationale:

- LC offers 36- and 60-month products; restricting to 36 months would discard ~40% of the data.
- Charge-offs after 36 months are rare in LC data; the looser horizon does not materially change the target rate.
- The constraint is documented here for transparency and revisited in the MRM document's Limitations section.

---

## 2. Filtering Summary

| Stage                          | Rows         | Columns |
| ------------------------------ | ------------ | ------- |
| Raw CSV                        | ~2,260,000   | 151     |
| After filter to terminal state | 1,345,350    | 151     |
| After leakage column removal   | 1,345,350    | 40      |

**Final target rate:** 19.97% (268,599 charge-offs/defaults out of 1,345,350 loans)

This rate is consistent with published Lending Club performance summaries and reflects LC's mix of prime through near-prime unsecured personal loans.

---

## 3. Leakage Exclusion Policy

The PD model uses only columns observable at loan origination. Columns populated after a loan is issued (payment history, recovery, current balance, hardship flags, settlement status) are excluded to prevent target leakage. The full exclusion list and rationale are defined in:

`src/risklens/features/leakage_blacklist.py`

Of 151 raw columns, 111 are excluded as post-origination. The remaining 40 columns are retained for modeling (with two exceptions kept for the data pipeline only: `loan_status` for target derivation, `issue_d` for vintage splitting; both are dropped before training).

The blacklist is grouped by exclusion reason: payment/recovery history, current loan state, hardship/settlement, secondary listing, identifiers, target-related.

---

## 4. Retained Columns

Columns retained in `data/interim/loans_filtered.parquet`.

### 4.1 Loan terms (origination)

| Column        | Type    | Description                                                          |
| ------------- | ------- | -------------------------------------------------------------------- |
| `loan_amnt`   | float   | Loan amount requested by the borrower (USD)                          |
| `term`        | string  | Loan term: "36 months" or "60 months"                                |
| `int_rate`    | string  | Interest rate assigned by LC (string with % suffix; parse to float)  |
| `installment` | float   | Monthly payment owed by the borrower                                 |
| `grade`       | string  | LC's assigned grade: A (lowest risk) through G (highest)             |
| `sub_grade`   | string  | LC's sub-grade: A1 (lowest) through G5 (highest)                     |
| `purpose`     | string  | Loan purpose (debt_consolidation, credit_card, home_improvement, …)  |

### 4.2 Borrower financials

| Column                      | Type   | Description                                                       |
| --------------------------- | ------ | ----------------------------------------------------------------- |
| `annual_inc`                | float  | Self-reported annual income                                       |
| `dti`                       | float  | Debt-to-income ratio (excluding mortgage)                         |
| `emp_length`                | string | Employment length, bucketed: "< 1 year" through "10+ years", n/a  |
| `home_ownership`            | string | RENT, OWN, MORTGAGE, OTHER, NONE, ANY                             |
| `verification_status`       | string | Income verification: Verified, Source Verified, Not Verified      |
| `annual_inc_joint`          | float  | Joint annual income (joint applications only; mostly NaN)         |
| `dti_joint`                 | float  | Joint DTI (joint applications only; mostly NaN)                   |
| `verification_status_joint` | string | Joint verification status (joint applications only; mostly NaN)   |
| `revol_bal_joint`           | float  | Joint revolving balance (joint applications only; mostly NaN)     |

### 4.3 Credit history

| Column                         | Type   | Description                                                       |
| ------------------------------ | ------ | ----------------------------------------------------------------- |
| `fico_range_low`               | float  | FICO score range (low end) at application                         |
| `fico_range_high`              | float  | FICO score range (high end) at application                        |
| `earliest_cr_line`             | string | Month/year of earliest reported credit line                       |
| `delinq_2yrs`                  | float  | 30+ day past-due incidences in last 2 years                       |
| `inq_last_6mths`               | float  | Credit inquiries in past 6 months                                 |
| `mths_since_last_delinq`       | float  | Months since most recent delinquency (NaN if none)                |
| `mths_since_last_record`       | float  | Months since most recent public record (NaN if none)              |
| `mths_since_last_major_derog`  | float  | Months since most recent 90+ day rating (NaN if none)             |
| `open_acc`                     | float  | Number of open credit lines                                       |
| `total_acc`                    | float  | Total credit lines (historical + open)                            |
| `pub_rec`                      | float  | Derogatory public records                                         |
| `pub_rec_bankruptcies`         | float  | Public record bankruptcies                                        |
| `tax_liens`                    | float  | Tax liens                                                         |
| `acc_open_past_24mths`         | float  | Accounts opened in past 24 months                                 |
| `revol_bal`                    | float  | Total revolving credit balance                                    |
| `revol_util`                   | string | Revolving utilization (string with % suffix; parse to float)      |

### 4.4 Listing metadata

| Column                  | Type   | Description                                                |
| ----------------------- | ------ | ---------------------------------------------------------- |
| `initial_list_status`   | string | Whether loan was listed Whole (W) or Fractional (F) at LC  |
| `addr_state`            | string | Borrower's US state code (e.g., CA, NY, TX)                |

### 4.5 Pipeline / derived columns

| Column            | Type     | Description                                                       |
| ----------------- | -------- | ----------------------------------------------------------------- |
| `issue_d`         | string   | Issue month, LC format (e.g., "Dec-2015")                         |
| `loan_status`     | string   | Original loan status; retained for target derivation only         |
| `target`          | int8     | Binary target: 1 if charged off or default, else 0                |
| `issue_dt`        | datetime | Parsed `issue_d` as datetime                                      |
| `vintage_year`    | Int16    | Year of loan origination                                          |
| `vintage_quarter` | string   | Year-quarter of origination (e.g., "2015Q4")                      |

**Note:** `loan_status`, `issue_d`, `issue_dt`, `vintage_year`, and `vintage_quarter` are retained in the filtered dataset for vintage-based splitting and EDA. They are excluded from the model feature set at training time.

---

## 5. Feature Type Notes

Several string-typed columns store numeric content and require parsing before modeling:

- `int_rate`: stored as e.g. `"13.56%"` → parse to float (`0.1356` or `13.56` depending on convention)
- `revol_util`: stored as e.g. `"83.7%"` → parse to float
- `term`: stored as `"36 months"` or `"60 months"` → parse to int (months)
- `emp_length`: ordinal but encoded as strings → map to ordinal int or one-hot
- `earliest_cr_line`: date as e.g. `"Aug-2003"` → parse to datetime → derive months of credit history at loan origination

These transformations are applied in the feature pipeline (`src/risklens/features/`) and not at the data-loading stage.

---

## 6. Known Data Quality Issues

- **Mixed types on read:** Several columns are flagged with mixed dtypes when loaded without `low_memory=False`. Our loader sets `low_memory=False` to read each column with a consistent inferred dtype.
- **Joint application sparsity:** `annual_inc_joint`, `dti_joint`, `verification_status_joint`, `revol_bal_joint` are populated only for joint applications. Expect ~95%+ missing on these columns; treat as either separate features with missingness indicators, or drop entirely in a first-pass model.
- **`int_rate` and `revol_util` as strings:** LC stores these with `%` suffix, requiring explicit parsing.
- **Sample selection bias:** This dataset contains only loans that LC accepted and issued. Rejected applications are in a separate file (excluded from this project per scope). The PD model therefore predicts default among accepted loans, not approval propensity. This bias is acknowledged in the MRM document's Limitations section.

---

## 7. Data Lineage

Kaggle (wordsforthewise/lending-club)
↓  scripts/01_download_data.py
data/raw/accepted_2007_to_2018Q4.csv.gz    (374 MB compressed, 2.26M rows × 151 cols)
↓  scripts/02_build_features.py
data/interim/loans_filtered.parquet         (snappy-compressed, 1.35M rows × 40 cols)
↓  (next: feature engineering, src/risklens/features/)
data/processed/                              (model-ready features per experiment)

Each transformation is reproducible from a single command on a fresh checkout (`uv sync`, then the script).

---

*Last updated: Day 1 of project. Will be revised as feature engineering and modeling reveal new data quality findings.*