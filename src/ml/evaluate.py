"""Evaluates the trained credit risk model on the held-out validation set.

Reproduces the exact same train/validation split used in train.py (same
random_state), loads the saved model, and reports both metrics relevant
to an imbalanced classification problem like this one:

    - ROC-AUC: how well the model ranks defaulters above non-defaulters
    - PR-AUC: particularly useful alongside ROC-AUC given the ~8%
      default rate found in EDA (Section 18, Finding 1) - both are
      reported, not one in place of the other
    - Precision / Recall / F1: at the default 0.5 probability threshold
    - Confusion matrix: raw counts of correct/incorrect predictions

Run from the project root: `python src/ml/evaluate.py`
"""

import json
import os
import sys
from datetime import datetime

import joblib
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.data.loader import load_all_tables
from src.data.preprocessor import build_feature_matrix
from src.ml.train import MODEL_DIR, get_train_val_split


def load_trained_model():
    model_path = os.path.join(MODEL_DIR, "credit_risk_model.joblib")
    return joblib.load(model_path)


def evaluate_model(model, X_val, y_val, threshold: float = 0.5) -> dict:
    """Compute ROC-AUC, PR-AUC, and threshold-based classification metrics."""
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_val, y_prob),
        "pr_auc": average_precision_score(y_val, y_prob),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1": f1_score(y_val, y_pred, zero_division=0),
    }

    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}  (particularly useful alongside "
          f"ROC-AUC given the ~8% default rate)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, zero_division=0))

    print("Confusion Matrix:")
    print("                Predicted No Default | Predicted Default")
    cm = confusion_matrix(y_val, y_pred)
    print(f"Actual No Default:     {cm[0][0]:>8}       |    {cm[0][1]:>8}")
    print(f"Actual Default:        {cm[1][0]:>8}       |    {cm[1][1]:>8}")

    return metrics


def log_evaluation(metrics: dict):
    """Append this evaluation's metrics to a log file, and save the
    latest result as metrics.json for the README/UI to reference."""
    log_path = os.path.join(PROJECT_ROOT, "outputs", "evaluation_log.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"\n--- Run at {datetime.now().isoformat()} ---\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")

    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


def main():
    tables = load_all_tables()
    X, y = build_feature_matrix(tables)

    # Same random_state as train.py reproduces the identical split,
    # so evaluation runs on the same held-out data the model never saw.
    _, X_val, _, y_val = get_train_val_split(X, y)

    model = load_trained_model()
    metrics = evaluate_model(model, X_val, y_val)
    log_evaluation(metrics)


if __name__ == "__main__":
    main()