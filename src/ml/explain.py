"""Explains individual predictions from the credit risk model using SHAP.

For a given applicant, this reports which features pushed the
prediction toward higher or lower risk, and by how much - turning a
raw probability into a plain-language explanation a non-technical
reviewer (e.g. a loan officer) can actually use.

Uses SHAP's TreeExplainer, which is fast and exact for tree-based
models like LightGBM (unlike KernelExplainer, which is model-agnostic
but much slower and only approximate).

Only features the applicant actually has a real value for are shown in
the explanation. LightGBM can treat missingness itself as informative,
so a missing field can technically influence the prediction - but
"this increased your risk (value: unknown)" isn't a useful explanation
for a human reader, so those are excluded from the ranked list and
summarized separately instead.

Protected/sensitive attributes (gender, marital status) are also
excluded from the displayed explanation, even though the model may use
them internally - showing these as a "reason" for a credit decision is
a fair-lending concern regardless of what SHAP mathematically
calculated.

Run from the project root: `python src/ml/explain.py` for a demo using
the same sample applicant as predict.py.
"""

import json
import os
import sys

import pandas as pd
import shap

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.ml.predict import (
    load_model_and_features,
    prepare_applicant_row,
    get_risk_band,
)

TOP_N_FEATURES = 5

# Plain-language labels for the most common/important features (from
# EDA's strongest predictors and the engineered features). Any feature
# not listed here falls back to a readable auto-formatted version of
# its column name.
FEATURE_LABELS = {
    "EXT_SOURCE_1": "External credit score 1",
    "EXT_SOURCE_2": "External credit score 2",
    "EXT_SOURCE_3": "External credit score 3",
    "AMT_INCOME_TOTAL": "Annual income",
    "AMT_CREDIT": "Loan amount requested",
    "AMT_ANNUITY": "Annual repayment amount",
    "AMT_GOODS_PRICE": "Price of goods being financed",
    "DAYS_BIRTH": "Age",
    "AGE_YEARS": "Age (years)",
    "DAYS_EMPLOYED": "Employment duration",
    "EMPLOYMENT_YEARS": "Employment duration (years)",
    "CODE_GENDER": "Gender",
    "FLAG_OWN_CAR": "Car ownership",
    "FLAG_OWN_REALTY": "Property ownership",
    "CNT_CHILDREN": "Number of children",
    "NAME_EDUCATION_TYPE": "Education level",
    "NAME_FAMILY_STATUS": "Family status",
    "NAME_INCOME_TYPE": "Income type",
    "NAME_HOUSING_TYPE": "Housing type",
    "CREDIT_INCOME_RATIO": "Loan-to-income ratio",
    "ANNUITY_INCOME_RATIO": "Repayment-to-income ratio",
    "BUREAU_CREDIT_COUNT": "Number of external credit records",
    "BUREAU_ACTIVE_COUNT": "Number of active external credits",
    "HAS_BUREAU_HISTORY": "Has external credit history",
    "PREV_APPLICATION_COUNT": "Number of previous applications",
    "PREV_REFUSED_COUNT": "Number of previously refused applications",
    "PREV_APPROVED_COUNT": "Number of previously approved applications",
    "PREV_REFUSAL_RATE": "Previous application refusal rate",
    "INSTALLMENT_COUNT": "Number of past installment payments",
    "AVG_DAYS_LATE": "Average days late on past payments",
    "MAX_DAYS_LATE": "Maximum days late on a past payment",
}

# Protected/sensitive attributes are excluded from the displayed
# explanation, even though the model may use them internally. Showing
# "your gender decreased your risk" or "your marital status increased
# your risk" to a loan officer is a fair-lending concern regardless of
# what SHAP mathematically calculated - several jurisdictions restrict
# using characteristics like these in credit decisions.
EXCLUDED_EXPLANATION_FEATURES = {
    "CODE_GENDER",
    "NAME_FAMILY_STATUS",
}


def humanize_feature_name(feature: str) -> str:
    """Converts a raw column name into a readable label. Falls back to
    a formatted version of the column name for anything not in
    FEATURE_LABELS, rather than requiring every one of the 97 features
    to be manually mapped."""
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    return feature.replace("_", " ").title()


def explain_prediction(applicant_data: dict, top_n: int = TOP_N_FEATURES) -> dict:
    """Score an applicant and explain the prediction with SHAP.

    Parameters
    ----------
    applicant_data : dict
        Same format as predict.predict_risk() - raw applicant fields.
    top_n : int
        How many top contributing features to return.

    Returns
    -------
    dict with keys: probability, risk_score, risk_band,
    top_risk_increasing_features, top_risk_decreasing_features,
    missing_field_influence
    """
    model, feature_columns, categorical_categories = load_model_and_features()
    X, _ = prepare_applicant_row(applicant_data, feature_columns, categorical_categories)

    probability = model.predict_proba(X)[0, 1]
    risk_band = get_risk_band(probability)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # SHAP's output shape for binary classification has changed across
    # versions - sometimes a list of two arrays (one per class),
    # sometimes a single 3D array (samples, features, classes). Either
    # way, index 1 is the positive class (TARGET = 1, default).
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    feature_impacts = list(zip(X.columns, shap_values[0], X.iloc[0].values))

    # Only explain using features the applicant actually has a real
    # value for, and exclude protected/sensitive attributes from the
    # displayed explanation - see module docstring for both reasons.
    known_impacts = [
        item for item in feature_impacts
        if pd.notna(item[2]) and item[0] not in EXCLUDED_EXPLANATION_FEATURES
    ]
    unknown_impacts = [item for item in feature_impacts if pd.isna(item[2])]

    # Sorted separately for clarity - largest positive impact first for
    # risk-increasing factors, most negative first for risk-decreasing.
    positive_impacts = sorted(
        [item for item in known_impacts if item[1] > 0],
        key=lambda item: item[1],
        reverse=True,
    )
    negative_impacts = sorted(
        [item for item in known_impacts if item[1] < 0],
        key=lambda item: item[1],
    )

    risk_increasing = [
        {
            "feature": humanize_feature_name(f),
            "impact": round(float(v), 4),
            "value": _format_value_for_display(f, _clean_value(val)),
        }
        for f, v, val in positive_impacts[:top_n]
    ]

    risk_decreasing = [
        {
            "feature": humanize_feature_name(f),
            "impact": round(float(v), 4),
            "value": _format_value_for_display(f, _clean_value(val)),
        }
        for f, v, val in negative_impacts[:top_n]
    ]

    # Aggregate influence of fields the applicant didn't provide, kept
    # separate from the main explanation rather than mixed into it.
    # This is a diagnostic magnitude, not a percentage or literal risk
    # contribution.
    missing_field_influence = round(
        float(sum(abs(v) for _, v, _ in unknown_impacts)), 4
    )

    return {
        "probability": round(float(probability), 4),
        "risk_score": round(float(probability) * 100, 2),
        "risk_band": risk_band,
        "top_risk_increasing_features": risk_increasing,
        "top_risk_decreasing_features": risk_decreasing,
        "missing_field_influence": missing_field_influence,
    }


def _clean_value(val):
    """Converts a raw feature value into something JSON-serializable
    and readable (pandas/numpy types don't always print cleanly)."""
    try:
        if isinstance(val, float):
            return round(val, 2)
        return str(val)
    except Exception:
        return str(val)

def _format_value_for_display(feature: str, value):
    """A few raw columns are stored as negative day-counts, which read
    poorly in a plain-English explanation. Converts just those to a
    readable year value for display, without changing anything about
    how the model itself was trained or predicts."""
    day_based_features = {"DAYS_BIRTH", "DAYS_EMPLOYED"}
    if feature in day_based_features:
        try:
            years = abs(float(value)) / 365
            return f"{years:.1f} years"
        except (TypeError, ValueError):
            return value
    return value


def explain_in_plain_language(explanation: dict) -> str:
    """Turns the raw SHAP output into a short, plain-English summary -
    the format a non-technical reviewer (e.g. a loan officer) actually
    wants, rather than a table of SHAP values."""
    lines = []
    lines.append(f"Risk Assessment: {explanation['risk_band'].upper()} "
                  f"({explanation['risk_score']}/100)")
    lines.append("")

    if explanation["missing_field_influence"] > 0.3:
        lines.append("Note: this assessment used partial information - "
                      "some unprovided fields also influenced the result.")
        lines.append("")

    if explanation["top_risk_increasing_features"]:
        lines.append("Factors increasing risk:")
        for item in explanation["top_risk_increasing_features"]:
            lines.append(f"  - {item['feature']}: {item['value']}")

    if explanation["top_risk_decreasing_features"]:
        lines.append("")
        lines.append("Factors decreasing risk:")
        for item in explanation["top_risk_decreasing_features"]:
            lines.append(f"  - {item['feature']}: {item['value']}")

    return "\n".join(lines)


if __name__ == "__main__":
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

    explanation = explain_prediction(sample_applicant)

    print("=" * 50)
    print("SHAP EXPLANATION (raw)")
    print("=" * 50)
    print(json.dumps(explanation, indent=2))

    print("\n" + "=" * 50)
    print("PLAIN-LANGUAGE SUMMARY")
    print("=" * 50)
    print(explain_in_plain_language(explanation))