"""Extracts simple, readable business rules from the credit risk model.

A shallow decision tree (max depth 3) is trained on the same
training data as the main model, learning to approximate the LightGBM
model's predictions rather than the raw TARGET label directly. This is
called a "surrogate model" - it doesn't need to be as accurate as the
real model, it needs to be simple enough for a non-technical credit
policy team to read, review and understand.

The real LightGBM model (src/ml/train.py) remains the one actually
used for scoring applicants (src/ml/predict.py). This module produces
a parallel, human-readable summary of the same general decision
pattern - it is not a replacement for the real model, and is not as
accurate.

Run from the project root: `python src/ml/business_rules.py`
"""

import os
import sys

from sklearn.tree import DecisionTreeClassifier, export_text

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.data.loader import load_all_tables
from src.data.preprocessor import build_feature_matrix
from src.ml.train import get_train_val_split, RANDOM_STATE
from src.ml.explain import EXCLUDED_EXPLANATION_FEATURES, humanize_feature_name

MAX_DEPTH = 3
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


# Raw day-count columns are excluded from the surrogate tree, since
# their readable counterparts (AGE_YEARS, EMPLOYMENT_YEARS) already
# exist in the feature matrix with identical information in a much
# more readable unit. Excluding the raw versions means the tree can
# only ever split on the years-based columns, so no text-based
# conversion of the printed rules is needed afterward.
DAY_BASED_RAW_FEATURES = {"DAYS_BIRTH", "DAYS_EMPLOYED"}


def get_excluded_columns(df_columns):
    """Columns to exclude from the surrogate tree: protected/sensitive
    attributes (same reasoning as explain.py) plus raw day-count
    columns (superseded by their readable year-based versions)."""
    return [
        c for c in (EXCLUDED_EXPLANATION_FEATURES | DAY_BASED_RAW_FEATURES)
        if c in df_columns
    ]


def train_surrogate_tree(model, X_train):
    """Trains a shallow decision tree to approximate the real model's
    predictions on the training data.

    Learning from the real model's predicted labels (rather than the
    true TARGET labels directly) is what makes this a surrogate model -
    it's explaining what the LightGBM model learned, not re-deriving a
    separate model from scratch.

    Note: the surrogate is trained on the real model's binary predictions
    at the default 0.5 threshold, not its underlying probability ranking -
    it explains "would this be flagged as default," not the full
    continuous risk score.

    """
    excluded_cols = get_excluded_columns(X_train.columns)
    X_train_filtered = X_train.drop(columns=excluded_cols)

    # Decision trees need numeric input - category dtype columns are
    # converted to their integer codes. This is a simplification made
    # only for the surrogate tree; the real model (train.py) handles
    # categoricals natively and is unaffected.
    X_numeric = X_train_filtered.copy()
    for col in X_numeric.select_dtypes(include="category").columns:
        X_numeric[col] = X_numeric[col].cat.codes

    model_predictions = model.predict(X_train)

    surrogate = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
    surrogate.fit(X_numeric, model_predictions)

    return surrogate, X_numeric.columns.tolist()


def get_readable_rules(surrogate, feature_names) -> str:
    """Converts the trained tree into readable IF/THEN text, using the
    same plain-language feature names as explain.py for consistency."""
    raw_rules = export_text(surrogate, feature_names=feature_names, decimals=2)

    readable_rules = raw_rules
    for raw_name in sorted(feature_names, key=len, reverse=True):
        label = humanize_feature_name(raw_name)
        readable_rules = readable_rules.replace(raw_name, label)

    return readable_rules
def get_structured_rules(surrogate, feature_names):
    """Walks the trained tree directly and returns each decision path
    as a structured rule (conditions + predicted class + confidence),
    rather than printed tree text. This is what powers a readable
    card-based display, instead of asking anyone to read raw ASCII
    tree output."""
    tree = surrogate.tree_
    paths = []

    def walk(node, conditions):
        is_leaf = tree.children_left[node] == tree.children_right[node]
        if is_leaf:
            values = tree.value[node][0]
            total = int(tree.n_node_samples[node])
            class_idx = int(values.argmax())
            predicted_class = int(surrogate.classes_[class_idx])
            confidence = float(values[class_idx] / total) if total > 0 else 0.0
            paths.append({
                "conditions": list(conditions),
                "predicted_class": predicted_class,
                "confidence": round(confidence, 3),
                "sample_count": total,
            })
            return
        feature = feature_names[tree.feature[node]]
        threshold = tree.threshold[node]
        walk(tree.children_left[node], conditions + [(feature, "<=", threshold)])
        walk(tree.children_right[node], conditions + [(feature, ">", threshold)])

    walk(0, [])
    return paths


def format_rule_as_sentence(rule: dict) -> str:
    """Turns one structured rule into a plain-English sentence, using
    the same readable feature labels as explain.py."""
    parts = []
    for feature, operator, threshold in rule["conditions"]:
        label = humanize_feature_name(feature)
        direction = "at most" if operator == "<=" else "above"
        parts.append(f"{label} is {direction} {threshold:.2f}")

    condition_text = " AND ".join(parts)
    risk_label = "HIGH RISK" if rule["predicted_class"] == 1 else "LOW RISK"
    return f"IF {condition_text} \u2192 {risk_label}"


def measure_surrogate_agreement(surrogate, X_val, model, feature_names) -> float:
    """Reports how often the simple tree agrees with the real model's
    predictions on unseen data - the surrogate's accuracy is not
    expected to match the real model, but this number should be
    disclosed honestly rather than presenting the rules as equivalent
    to the actual model."""
    excluded_cols = get_excluded_columns(X_val.columns)
    X_val_filtered = X_val.drop(columns=excluded_cols)
    X_val_numeric = X_val_filtered.copy()
    for col in X_val_numeric.select_dtypes(include="category").columns:
        X_val_numeric[col] = X_val_numeric[col].cat.codes
    X_val_numeric = X_val_numeric[feature_names]

    model_predictions = model.predict(X_val)
    surrogate_predictions = surrogate.predict(X_val_numeric)

    agreement = (model_predictions == surrogate_predictions).mean()
    return round(float(agreement), 4)


def main():
    import joblib

    model_path = os.path.join(MODEL_DIR, "credit_risk_model.joblib")
    model = joblib.load(model_path)

    tables = load_all_tables()
    X, y = build_feature_matrix(tables)
    X_train, X_val, y_train, y_val = get_train_val_split(X, y)

    surrogate, feature_names = train_surrogate_tree(model, X_train)
    agreement = measure_surrogate_agreement(surrogate, X_val, model, feature_names)

    rules_text = get_readable_rules(surrogate, feature_names)
    structured_rules = get_structured_rules(surrogate, feature_names)
    rule_sentences = [format_rule_as_sentence(r) for r in structured_rules]

    print("=" * 60)
    print("BUSINESS RULES (simplified decision tree, max depth 3)")
    print("=" * 60)
    print(rules_text)
    print(f"\nAgreement with the real model on validation data: "
          f"{agreement * 100:.1f}%")
    print("(This surrogate is a simplified, human-readable summary of "
          "the real model's behavior, not a replacement for it. The "
          "actual model used for scoring is the LightGBM model in "
          "src/ml/train.py.)")

    output_path = os.path.join(PROJECT_ROOT, "outputs", "business_rules.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("Business Rules (simplified decision tree, max depth 3)\n")
        f.write("=" * 60 + "\n")
        f.write(rules_text)
        f.write(f"\nAgreement with the real model on validation data: "
                f"{agreement * 100:.1f}%\n")
    print(f"\nSaved to {output_path}")

    import json
    structured_path = os.path.join(PROJECT_ROOT, "outputs", "business_rules_structured.json")
    with open(structured_path, "w") as f:
        json.dump({
            "rules": structured_rules,
            "sentences": rule_sentences,
            "agreement": agreement,
        }, f, indent=2)
    print(f"Structured rules saved to {structured_path}")


if __name__ == "__main__":
    main()