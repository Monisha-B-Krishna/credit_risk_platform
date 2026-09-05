"""Compares class_weight="balanced" against no class weighting, on the
identical train/validation split, to confirm the precision/recall
trade-off documented after the first training run is actually caused
by the class weighting choice, not by preprocessing or chance.

This is a supporting experiment, not part of the core pipeline -
train.py still trains and saves the model used by predict.py. This
script exists to produce evidence for the README's model rationale
section.

Run from the project root: `python src/ml/compare_models.py`
"""

import os
import sys

import lightgbm as lgb
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.data.loader import load_all_tables
from src.data.preprocessor import build_feature_matrix
from src.ml.train import RANDOM_STATE, get_train_val_split


def train_and_score(X_train, y_train, X_val, y_val, class_weight):
    """Train one LightGBM model with the given class_weight setting and
    return its evaluation metrics on the validation set."""
    model = lgb.LGBMClassifier(
        objective="binary",
        class_weight=class_weight,
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        verbose=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "roc_auc": roc_auc_score(y_val, y_prob),
        "pr_auc": average_precision_score(y_val, y_prob),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1": f1_score(y_val, y_pred, zero_division=0),
        "best_iteration": model.best_iteration_,
    }


def print_comparison(no_weight_metrics, balanced_metrics):
    print("\n" + "=" * 65)
    print(f"{'Metric':<15}{'No Weighting':>20}{'class_weight=balanced':>25}")
    print("=" * 65)
    for key, label in [
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1 Score"),
    ]:
        print(f"{label:<15}{no_weight_metrics[key]:>20.4f}{balanced_metrics[key]:>25.4f}")
    print("=" * 65)


def main():
    tables = load_all_tables()
    X, y = build_feature_matrix(tables)

    X_train, X_val, y_train, y_val = get_train_val_split(X, y)
    print(f"Train: {X_train.shape}, Validation: {X_val.shape}\n")

    print("Training model WITHOUT class weighting...")
    no_weight_metrics = train_and_score(X_train, y_train, X_val, y_val, class_weight=None)

    print("\nTraining model WITH class_weight='balanced'...")
    balanced_metrics = train_and_score(X_train, y_train, X_val, y_val, class_weight="balanced")

    print_comparison(no_weight_metrics, balanced_metrics)

    # Save the comparison for reference in the README
    output_path = os.path.join(PROJECT_ROOT, "outputs", "model_comparison.txt")
    with open(output_path, "w") as f:
        f.write("Class Weighting Comparison\n")
        f.write("=" * 65 + "\n")
        f.write(f"{'Metric':<15}{'No Weighting':>20}{'class_weight=balanced':>25}\n")
        for key, label in [
            ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"),
            ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1 Score"),
        ]:
            f.write(f"{label:<15}{no_weight_metrics[key]:>20.4f}{balanced_metrics[key]:>25.4f}\n")
    print(f"\nComparison saved to {output_path}")


if __name__ == "__main__":
    main()