"""Trains a LightGBM credit default risk model.

Pipeline:
    1. Load raw tables and build the feature matrix (src.data.preprocessor)
    2. Stratified train/validation split (application_test.csv has no
       TARGET label, so it cannot be used for evaluation - a portion of
       application_train.csv is held out instead)
    3. Train LightGBM with class_weight="balanced" to address the ~8%
       default rate found during EDA, using early stopping on the
       validation set
    4. Save the trained model, feature column list, and categorical
       category values to models/

Run from the project root: `python src/ml/train.py`
"""

import json
import os
import sys
from datetime import datetime

import joblib
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.data.loader import load_all_tables
from src.data.preprocessor import build_feature_matrix

VALIDATION_SIZE = 0.2
RANDOM_STATE = 42
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


def get_train_val_split(X, y):
    """Stratified split so both sets keep the same ~8% default rate.

    Stratification matters here specifically because of the class
    imbalance found in EDA - a plain random split could, by chance, put
    very few defaulters in the validation set, making evaluation
    metrics unreliable.
    """
    return train_test_split(
        X, y,
        test_size=VALIDATION_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def train_model(X_train, y_train, X_val, y_val):
    """Train a LightGBM classifier with class weighting for imbalance.

    class_weight="balanced" is used instead of SMOTE (oversampling) -
    simpler to justify, and avoids generating synthetic applicants from
    a feature set that mixes many categorical and engineered columns.
    This may improve recall on the minority (default) class at some
    cost to precision - a documented trade-off, not an oversight.
    """
    model = lgb.LGBMClassifier(
        objective="binary",
        class_weight="balanced",
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

    return model


def save_model(model, feature_columns):
    """Save the trained model and its feature column list to models/.

    The feature list is saved alongside the model because predict.py
    needs to know the exact column order/names to build a matching
    input row for a new applicant.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "credit_risk_model.joblib")
    joblib.dump(model, model_path)

    features_path = os.path.join(MODEL_DIR, "feature_columns.json")
    with open(features_path, "w") as f:
        json.dump(list(feature_columns), f, indent=2)

    print(f"Model saved to {model_path}")
    print(f"Feature list saved to {features_path}")


def save_categorical_categories(X):
    """Save the exact category values LightGBM saw during training for
    every categorical column, so predict.py can apply the identical
    encoding to a new applicant instead of recomputing categories from
    a single row (which would only ever see one category and could
    silently produce wrong predictions)."""
    categorical_cols = X.select_dtypes(include="category").columns
    categories = {col: X[col].cat.categories.tolist() for col in categorical_cols}

    path = os.path.join(MODEL_DIR, "categorical_categories.json")
    with open(path, "w") as f:
        json.dump(categories, f, indent=2)
    print(f"Categorical categories saved to {path}")


def log_training_run(model, X_train, X_val, val_auc):
    """Append a timestamped record of this training run for comparison
    across future runs (same pattern used in preprocessor.py)."""
    log_path = os.path.join(PROJECT_ROOT, "outputs", "training_log.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "a") as f:
        f.write(f"\n--- Run at {datetime.now().isoformat()} ---\n")
        f.write(f"Train rows: {X_train.shape[0]}, Val rows: {X_val.shape[0]}\n")
        f.write(f"Features: {X_train.shape[1]}\n")
        f.write(f"Best iteration: {model.best_iteration_}\n")
        f.write(f"Validation ROC-AUC: {val_auc:.4f}\n")


def main():
    tables = load_all_tables()
    X, y = build_feature_matrix(tables)

    X_train, X_val, y_train, y_val = get_train_val_split(X, y)
    print(f"Train: {X_train.shape}, Validation: {X_val.shape}")

    model = train_model(X_train, y_train, X_val, y_val)

    # Calculated independently rather than read from model.best_score_,
    # so the reported number doesn't depend on LightGBM's internal
    # tracking matching exactly what's computed here.
    val_pred_proba = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred_proba)

    print(f"\nValidation ROC-AUC: {val_auc:.4f}")
    print(f"Best iteration: {model.best_iteration_}")

    save_model(model, X.columns)
    save_categorical_categories(X)
    log_training_run(model, X_train, X_val, val_auc)


if __name__ == "__main__":
    main()