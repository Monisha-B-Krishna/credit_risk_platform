"""Cleaning and feature engineering for the Home Credit dataset.

This file has two jobs, kept in separate functions on purpose so each
step is easy to test and reason about on its own:

    1. clean_application_data()      - fixes known data quality issues
       in the main application table, based on findings from EDA
       (notebooks/eda.ipynb, Section 18).

    2. engineer_*_features()         - builds new, more useful columns,
       including summarizing the supplementary tables (bureau,
       previous applications, installments) from many rows per
       applicant down to one row per applicant, so they can be merged
       onto the main table.

build_feature_matrix() ties everything together and returns a
model-ready (X, y) pair. Class imbalance is NOT handled here - that is
a model training decision, handled in src/ml/train.py via class
weighting, not a data cleaning step.
"""

import numpy as np
import pandas as pd

# --- Constants, based on EDA findings ---

# Placeholder value found in DAYS_EMPLOYED for applicants who are not
# currently employed (mostly pensioners). Not a real value - see EDA
# Finding 5 (Section 18).
DAYS_EMPLOYED_ANOMALY = 365243

# Columns missing above this fraction are dropped rather than imputed,
# since imputing over half a column mostly manufactures signal that
# isn't really there. See EDA Finding 2.
MISSING_THRESHOLD = 0.5

# Extreme incomes are capped at this percentile rather than dropped,
# so the row (and its other information) isn't lost entirely. See EDA
# Finding 5 (the 117,000,000 income outlier).
INCOME_CAP_PERCENTILE = 0.995


def clean_application_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known data quality issues in the main application table.

    Steps, in order:
        1. DAYS_EMPLOYED placeholder (365243) converted to NaN.
        2. OWN_CAR_AGE missing values filled with 0 - this column is
           ~66% missing, but the missingness is structural (it's blank
           for applicants who don't own a car), not a genuine gap. This
           has to run BEFORE the missing-column drop below, or the
           column would be dropped entirely instead of fixed.
        3. AMT_INCOME_TOTAL capped at the 99.5th percentile, to contain
           the extreme outlier found in EDA without losing that row.
        4. CODE_GENDER "XNA" rows dropped (4 rows, negligible sample).
        5. Remaining columns above MISSING_THRESHOLD dropped, with the
           dropped column names printed so the choice is traceable.

    Parameters
    ----------
    df : pd.DataFrame
        Raw application_train.csv, as loaded by src.data.loader.

    Returns
    -------
    pd.DataFrame
        Cleaned copy of the input.
    """
    df = df.copy()

    # 1. DAYS_EMPLOYED anomaly
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_ANOMALY, np.nan)

    # 2. OWN_CAR_AGE - structural missingness, fixed before the generic drop
    if "OWN_CAR_AGE" in df.columns:
        df["OWN_CAR_AGE"] = df["OWN_CAR_AGE"].fillna(0)

    # 3. Income outlier - capped, not dropped
    income_cap = df["AMT_INCOME_TOTAL"].quantile(INCOME_CAP_PERCENTILE)
    df["AMT_INCOME_TOTAL"] = np.minimum(df["AMT_INCOME_TOTAL"], income_cap)

    # 4. Negligible-sample gender category
    df = df[df["CODE_GENDER"] != "XNA"].copy()

    # 5. Drop columns with heavy missingness (mostly redundant
    # building/apartment detail fields collected three ways)
    missing_pct = df.isnull().mean()
    heavy_missing_cols = missing_pct[missing_pct > MISSING_THRESHOLD].index.tolist()
    print(f"Dropping {len(heavy_missing_cols)} columns with "
          f">{MISSING_THRESHOLD * 100:.0f}% missing values.")
    df = df.drop(columns=heavy_missing_cols)

    return df


def engineer_application_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the year/ratio features identified as useful during EDA.

    - AGE_YEARS: DAYS_BIRTH converted from negative days to years.
    - EMPLOYMENT_YEARS: cleaned DAYS_EMPLOYED converted to years.
    - CREDIT_INCOME_RATIO: loan amount relative to income.
    - ANNUITY_INCOME_RATIO: repayment amount relative to income.

    Both ratios guard against division by zero (income = 0 would
    otherwise produce inf, which can break model training).
    """
    df = df.copy()

    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365
    df["EMPLOYMENT_YEARS"] = -df["DAYS_EMPLOYED"] / 365

    safe_income = df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / safe_income
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / safe_income

    return df


def engineer_bureau_features(bureau: pd.DataFrame) -> pd.DataFrame:
    """Summarize bureau.csv from many rows per applicant to one row per applicant.

    One applicant can have many historical credit records at other
    lenders. This aggregates that history into per-applicant totals.

    Also adds HAS_BUREAU_HISTORY, which captures the EDA finding that
    applicants with NO bureau history default more often than those
    with some history (the "thin file" effect) - worth keeping as its
    own signal, not just implied by missing aggregate values.
    """
    agg = bureau.groupby("SK_ID_CURR").agg(
        BUREAU_CREDIT_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_COUNT=("CREDIT_ACTIVE", lambda x: (x == "Active").sum()),
        BUREAU_OVERDUE_MAX=("AMT_CREDIT_MAX_OVERDUE", "max"),
        BUREAU_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
    ).reset_index()

    agg["HAS_BUREAU_HISTORY"] = 1

    return agg


def engineer_previous_application_features(prev: pd.DataFrame) -> pd.DataFrame:
    """Summarize previous_application.csv to one row per applicant.

    Captures the EDA finding that applicants with prior refused
    applications default more often than those with none.
    """
    agg = prev.groupby("SK_ID_CURR").agg(
        PREV_APPLICATION_COUNT=("SK_ID_PREV", "count"),
        PREV_REFUSED_COUNT=("NAME_CONTRACT_STATUS", lambda x: (x == "Refused").sum()),
        PREV_APPROVED_COUNT=("NAME_CONTRACT_STATUS", lambda x: (x == "Approved").sum()),
        PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
    ).reset_index()

    agg["PREV_REFUSAL_RATE"] = agg["PREV_REFUSED_COUNT"] / agg["PREV_APPLICATION_COUNT"]

    return agg


def engineer_installment_features(installments: pd.DataFrame) -> pd.DataFrame:
    """Summarize installments_payments.csv to one row per applicant.

    Captures the EDA finding that a history of late payments predicts
    higher default risk on the current loan - the most direct
    repayment-behaviour signal available in this dataset.
    """
    installments = installments.copy()
    installments["DAYS_LATE"] = (
        installments["DAYS_ENTRY_PAYMENT"] - installments["DAYS_INSTALMENT"]
    )

    agg = installments.groupby("SK_ID_CURR").agg(
        INSTALLMENT_COUNT=("SK_ID_PREV", "count"),
        AVG_DAYS_LATE=("DAYS_LATE", "mean"),
        MAX_DAYS_LATE=("DAYS_LATE", "max"),
    ).reset_index()

    return agg


def build_feature_matrix(tables: dict, source_key: str = "app_train"):
    """Build the full model-ready feature matrix from all loaded tables.

    Class imbalance is intentionally NOT handled here - that is a model
    training decision (class weighting in src/ml/train.py), not a data
    cleaning step.

    Parameters
    ----------
    tables : dict
        Output of src.data.loader.load_all_tables(), or a dict
        containing at least a single-applicant DataFrame under
        source_key for prediction use.
    source_key : str
        Which table in `tables` to use as the base applicant data.
        "app_train" for training (has TARGET). "app_test" or a
        single-applicant table for prediction (no TARGET).

    Returns
    -------
    X : pd.DataFrame
        Feature matrix, one row per applicant.
    y : pd.Series or None
        TARGET label if present in the source table, otherwise None
        (prediction mode - the outcome isn't known yet).
    """
    df = clean_application_data(tables[source_key])
    df = engineer_application_features(df)

    if "bureau" in tables:
        bureau_features = engineer_bureau_features(tables["bureau"])
        df = df.merge(bureau_features, on="SK_ID_CURR", how="left")
        df["HAS_BUREAU_HISTORY"] = df["HAS_BUREAU_HISTORY"].fillna(0)

    if "previous_application" in tables:
        prev_features = engineer_previous_application_features(tables["previous_application"])
        df = df.merge(prev_features, on="SK_ID_CURR", how="left")

    if "installments" in tables:
        installment_features = engineer_installment_features(tables["installments"])
        df = df.merge(installment_features, on="SK_ID_CURR", how="left")

    # Count-based columns: no history means zero activity, so 0 is the
    # correct fill value here (not missing data).
    count_cols = [
        "BUREAU_CREDIT_COUNT", "BUREAU_ACTIVE_COUNT",
        "PREV_APPLICATION_COUNT", "PREV_REFUSED_COUNT", "PREV_APPROVED_COUNT",
        "INSTALLMENT_COUNT",
    ]
    for col in count_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Rate/mean columns (PREV_REFUSAL_RATE, AVG_DAYS_LATE, MAX_DAYS_LATE,
    # BUREAU_OVERDUE_MAX, BUREAU_DEBT_SUM) are deliberately left as NaN
    # for applicants with no history. A 0% refusal rate or 0 days late
    # would misleadingly suggest a clean track record, when the truth is
    # there is no track record at all. LightGBM handles NaN natively
    # during split selection, so no imputation is needed here.

    has_target = "TARGET" in df.columns
    y = df["TARGET"] if has_target else None
    drop_cols = ["SK_ID_CURR"] + (["TARGET"] if has_target else [])
    X = df.drop(columns=drop_cols)

    # LightGBM reads pandas "category" dtype natively - no manual
    # one-hot or label encoding needed.
    categorical_cols = X.select_dtypes(include=["object", "string"]).columns
    for col in categorical_cols:
        X[col] = X[col].astype("category")

    return X, y


if __name__ == "__main__":
    import sys
    import os
    from datetime import datetime

    # Always resolve paths relative to the project root, regardless of
    # which folder this script is actually run from.
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    os.chdir(PROJECT_ROOT)
    sys.path.append(PROJECT_ROOT)

    from src.data.loader import load_all_tables

    tables = load_all_tables()
    X, y = build_feature_matrix(tables)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts(normalize=True) * 100}")

    # Append this run's stats to a log file, so results can be
    # compared across runs over time.
    log_path = os.path.join(PROJECT_ROOT, "outputs", "preprocessing_log.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"\n--- Run at {datetime.now().isoformat()} ---\n")
        f.write(f"Feature matrix shape: {X.shape}\n")
        f.write(f"Target distribution:\n{y.value_counts(normalize=True) * 100}\n")

    print(f"\nResults appended to {log_path}")