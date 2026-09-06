"""Scores a single loan applicant's default risk using the trained model.

Given an applicant's details as a dictionary, this:
    1. Builds the same feature matrix used during training, via
       src.data.preprocessor (with TARGET absent, since the outcome
       isn't known yet for a new applicant)
    2. Loads the saved model, its expected feature columns, and the
       exact categorical values seen during training - applying the
       same categories to a new applicant avoids silently wrong
       predictions from a single row only ever seeing one category
    3. Returns a default probability, a risk band, and which fields
       were filled with defaults because they weren't provided

Run from the project root: `python src/ml/predict.py` for a demo with
a sample applicant.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.data.preprocessor import build_feature_matrix

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# Risk band cutoffs. These are business-rule thresholds chosen for this
# demonstration, not statistically derived from the validation set -
# stated plainly rather than overclaiming precision.
LOW_RISK_THRESHOLD = 0.10
MEDIUM_RISK_THRESHOLD = 0.30

# Matches the count_cols list in preprocessor.py's build_feature_matrix,
# plus HAS_BUREAU_HISTORY (filled separately there, right after the
# bureau merge). Kept in sync manually for now - fine at this project's
# size, would move to a shared config if this grew further.
COUNT_COLUMNS = [
    "BUREAU_CREDIT_COUNT", "BUREAU_ACTIVE_COUNT", "HAS_BUREAU_HISTORY",
    "PREV_APPLICATION_COUNT", "PREV_REFUSED_COUNT", "PREV_APPROVED_COUNT",
    "INSTALLMENT_COUNT",
]


_cached_model_bundle = None


def load_model_and_features():
    """Loads the model and its metadata once, then reuses the cached
    result. Reloading a joblib model file from disk on every single
    prediction is wasted work - this is the same caching pattern used
    for the DuckDB connection in query_runner.py."""
    global _cached_model_bundle

    if _cached_model_bundle is not None:
        return _cached_model_bundle

    model_path = os.path.join(MODEL_DIR, "credit_risk_model.joblib")
    features_path = os.path.join(MODEL_DIR, "feature_columns.json")
    categories_path = os.path.join(MODEL_DIR, "categorical_categories.json")

    model = joblib.load(model_path)
    with open(features_path) as f:
        feature_columns = json.load(f)
    with open(categories_path) as f:
        categorical_categories = json.load(f)

    _cached_model_bundle = (model, feature_columns, categorical_categories)
    return _cached_model_bundle


def get_risk_band(probability: float) -> str:
    if probability < LOW_RISK_THRESHOLD:
        return "Low"
    elif probability < MEDIUM_RISK_THRESHOLD:
        return "Medium"
    else:
        return "High"


def prepare_applicant_row(applicant_data: dict, feature_columns: list,
                           categorical_categories: dict):
    """Builds a single-applicant table, runs it through the same
    cleaning/feature-engineering pipeline as training, and aligns the
    result to exactly the columns and categorical encodings the model
    expects.

    A new applicant has no bureau/previous application/installment
    history to merge in, so those tables are simply omitted - the
    resulting missing columns are added back below. Count-based history
    features are filled with 0 when unavailable (matching the training-
    time fill logic in preprocessor.py), while other missing features
    are left as NaN for LightGBM to handle as missing values.
    """
    applicant_df = pd.DataFrame([applicant_data])
    tables = {"applicant": applicant_df}

    X, _ = build_feature_matrix(tables, source_key="applicant")

    filled_with_defaults = []
    for col in feature_columns:
        if col not in X.columns:
            filled_with_defaults.append(col)
            X[col] = 0 if col in COUNT_COLUMNS else np.nan

    # Reapply the exact categorical values seen during training to
    # every categorical column - both newly-added missing ones and
    # ones the applicant actually provided. Recomputing categories
    # from a single row would only ever see one value and could
    # silently encode it differently than training did.
    for col, categories in categorical_categories.items():
        if col in X.columns:
            X[col] = pd.Categorical(X[col], categories=categories)

    X = X[feature_columns]

    return X, filled_with_defaults


def predict_risk(applicant_data: dict) -> dict:
    """Score a single applicant and return probability, risk score,
    risk band, and how complete the provided information was.

    Parameters
    ----------
    applicant_data : dict
        Raw applicant fields, using the same column names as
        application_train.csv (e.g. AMT_INCOME_TOTAL, DAYS_BIRTH,
        CODE_GENDER, etc.). SK_ID_CURR is not required.

    Returns
    -------
    dict with keys: probability, risk_score, risk_band,
    input_completeness, fields_filled_with_defaults
    """
    model, feature_columns, categorical_categories = load_model_and_features()
    X, filled_with_defaults = prepare_applicant_row(
        applicant_data, feature_columns, categorical_categories
    )

    probability = model.predict_proba(X)[0, 1]
    risk_band = get_risk_band(probability)

    total_features = len(feature_columns)
    filled_count = len(filled_with_defaults)
    provided_count = total_features - filled_count

    return {
        "probability": round(float(probability), 4),
        "risk_score": round(float(probability) * 100, 2),
        "risk_band": risk_band,
        "input_completeness": {
            "provided_features": provided_count,
            "filled_with_defaults": filled_count,
            "total_features": total_features,
        },
        # Kept for debugging/traceability, not meant for end-user display.
        "fields_filled_with_defaults": filled_with_defaults,
    }


if __name__ == "__main__":
    # Demo applicant - illustrative values, not a real Home Credit row.
    sample_applicant = {
        "CODE_GENDER": "F",
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "Y",
        "CNT_CHILDREN": 0,
        "AMT_INCOME_TOTAL": 180000.0,
        "AMT_CREDIT": 450000.0,
        "AMT_ANNUITY": 22000.0,
        "AMT_GOODS_PRICE": 450000.0,
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "DAYS_BIRTH": -12000,
        "DAYS_EMPLOYED": -2000,
        "NAME_CONTRACT_TYPE": "Cash loans",
        "NAME_TYPE_SUITE": "Unaccompanied",
        "REGION_POPULATION_RELATIVE": 0.02,
        "EXT_SOURCE_1": 0.6,
        "EXT_SOURCE_2": 0.6,
        "EXT_SOURCE_3": 0.6,
    }

    result = predict_risk(sample_applicant)

    print("=" * 50)
    print("CREDIT RISK ASSESSMENT")
    print("=" * 50)
    print(f"Default Probability: {result['probability'] * 100:.2f}%")
    print(f"Risk Score:          {result['risk_score']} / 100")
    print(f"Risk Band:           {result['risk_band'].upper()}")
    print()
    print(f"Provided features:   {result['input_completeness']['provided_features']}")
    print(f"Filled with defaults: {result['input_completeness']['filled_with_defaults']}")
    print(f"Total features used: {result['input_completeness']['total_features']}")
    if result['input_completeness']['filled_with_defaults'] > 0:
        print("\nNote: assessment generated with partial applicant information.")
    print("=" * 50)