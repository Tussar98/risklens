"""Leakage column blacklist for the Lending Club PD model.

Columns in POST_ORIGINATION_COLUMNS are known only after a loan has been issued
and (partially or fully) observed. They are therefore unavailable at scoring
time and would cause catastrophic target leakage if used as features. The PD
model must use only columns known at origination.

This list is curated from the Lending Club data dictionary (LCDataDictionary.xlsx)
and follows standard credit-risk modeling practice. The list is intentionally
conservative: when a column's availability at origination is ambiguous, it is
excluded.

Grouped by reason for exclusion:

1. Payment / recovery history (populated as loan ages)
2. Current loan state (populated after origination, evolves over time)
3. Hardship / settlement (populated only if borrower enters hardship)
4. Secondary listing / joint application post-origination fields
5. Identifiers / metadata not predictive at origination
6. The target itself (loan_status) and target-derived fields
"""

from __future__ import annotations

# 1. Payment and recovery history -- populated as the loan amortizes.
_PAYMENT_HISTORY = frozenset({
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "last_credit_pull_d",
    "last_fico_range_high",
    "last_fico_range_low",
    "out_prncp",
    "out_prncp_inv",
})

# 2. Current loan state -- evolves after origination.
_CURRENT_STATE = frozenset({
    "pymnt_plan",
    "collections_12_mths_ex_med",
    "acc_now_delinq",
    "tot_coll_amt",
    "tot_cur_bal",
    "total_rev_hi_lim",
    "total_bal_ex_mort",
    "total_bc_limit",
    "total_il_high_credit_limit",
    "avg_cur_bal",
    "bc_open_to_buy",
    "bc_util",
    "chargeoff_within_12_mths",
    "delinq_amnt",
    "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl",
    "mort_acc",
    "mths_since_recent_bc",
    "mths_since_recent_bc_dlq",
    "mths_since_recent_inq",
    "mths_since_recent_revol_delinq",
    "num_accts_ever_120_pd",
    "num_actv_bc_tl",
    "num_actv_rev_tl",
    "num_bc_sats",
    "num_bc_tl",
    "num_il_tl",
    "num_op_rev_tl",
    "num_rev_accts",
    "num_rev_tl_bal_gt_0",
    "num_sats",
    "num_tl_120dpd_2m",
    "num_tl_30dpd",
    "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m",
    "pct_tl_nvr_dlq",
    "percent_bc_gt_75",
    "tot_hi_cred_lim",
    "total_bal_il",
    "il_util",
    "open_acc_6m",
    "open_act_il",
    "open_il_12m",
    "open_il_24m",
    "mths_since_rcnt_il",
    "open_rv_12m",
    "open_rv_24m",
    "max_bal_bc",
    "all_util",
    "inq_fi",
    "total_cu_tl",
    "inq_last_12m",
})

# 3. Hardship and settlement -- only populated for distressed loans.
_HARDSHIP_SETTLEMENT = frozenset({
    "hardship_flag",
    "hardship_type",
    "hardship_reason",
    "hardship_status",
    "deferral_term",
    "hardship_amount",
    "hardship_start_date",
    "hardship_end_date",
    "payment_plan_start_date",
    "hardship_length",
    "hardship_dpd",
    "hardship_loan_status",
    "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount",
    "hardship_last_payment_amount",
    "debt_settlement_flag",
    "debt_settlement_flag_date",
    "settlement_status",
    "settlement_date",
    "settlement_amount",
    "settlement_percentage",
    "settlement_term",
})

# 4. Secondary listing / post-origination joint fields.
_SECONDARY = frozenset({
    "sec_app_fico_range_low",
    "sec_app_fico_range_high",
    "sec_app_earliest_cr_line",
    "sec_app_inq_last_6mths",
    "sec_app_mort_acc",
    "sec_app_open_acc",
    "sec_app_revol_util",
    "sec_app_open_act_il",
    "sec_app_num_rev_accts",
    "sec_app_chargeoff_within_12_mths",
    "sec_app_collections_12_mths_ex_med",
    "sec_app_mths_since_last_major_derog",
})

# 5. Identifiers and metadata -- not features.
_IDENTIFIERS = frozenset({
    "id",
    "member_id",
    "url",
    "desc",
    "title",
    "emp_title",
    "zip_code",
    "policy_code",
    "application_type",
    "disbursement_method",
    "funded_amnt",
    "funded_amnt_inv",
    "issue_d",
})

# 6. Target and target-derived columns.
TARGET_COLUMN = "loan_status"
_TARGET_RELATED = frozenset({
    "loan_status",
})

POST_ORIGINATION_COLUMNS: frozenset[str] = (
    _PAYMENT_HISTORY
    | _CURRENT_STATE
    | _HARDSHIP_SETTLEMENT
    | _SECONDARY
    | _IDENTIFIERS
    | _TARGET_RELATED
)


def is_leakage_column(col: str) -> bool:
    """Return True if `col` must be excluded from the feature set."""
    return col in POST_ORIGINATION_COLUMNS


__all__ = [
    "POST_ORIGINATION_COLUMNS",
    "TARGET_COLUMN",
    "is_leakage_column",
]
