"""
train_model.py
---------------
Splits the engineered dataset, trains two baseline classifiers
(Logistic Regression and Random Forest) to predict `delay`, and
reports evaluation metrics so we have a benchmark before tuning.

Split strategy: TIME-BASED, not random. Trips span 2019-01-01 to
2019-02-12 (~6 weeks). A random split would let the model "see the
future" (train on a trip from week 5, test on one from week 2), which
overstates real-world performance. Splitting chronologically -- train
on the earliest ~80% of trips, test on the most recent ~20% -- mimics
how the model will actually be used: predicting delays for upcoming
trips based on patterns learned from past ones.
"""

import joblib
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

from feature_engineering import build_final_dataset
from config import BASE_DIR

MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

TARGET_COL = "delay"


def time_based_split(full_df: pd.DataFrame, model_df: pd.DataFrame, test_size: float = 0.2):
    """
    Uses full_df's departure_date purely to DETERMINE the cutoff and the
    row order -- model_df (which has no date column) is split using
    that same row order, so features and dates stay aligned.
    """
    full_sorted = full_df.sort_values("departure_date")
    model_sorted = model_df.loc[full_sorted.index]

    cutoff = int(len(full_sorted) * (1 - test_size))
    train_idx = full_sorted.index[:cutoff]
    test_idx = full_sorted.index[cutoff:]

    print(
        f"Train period: {full_sorted['departure_date'].iloc[0]} -> "
        f"{full_sorted.loc[train_idx, 'departure_date'].iloc[-1]}  ({len(train_idx)} trips)"
    )
    print(
        f"Test period:  {full_sorted.loc[test_idx, 'departure_date'].iloc[0]} -> "
        f"{full_sorted['departure_date'].iloc[-1]}  ({len(test_idx)} trips)"
    )

    X_train = model_sorted.loc[train_idx].drop(columns=[TARGET_COL])
    y_train = model_sorted.loc[train_idx, TARGET_COL]
    X_test = model_sorted.loc[test_idx].drop(columns=[TARGET_COL])
    y_test = model_sorted.loc[test_idx, TARGET_COL]
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Fit the scaler on TRAIN ONLY, then apply to both -- fitting on the
    full dataset would leak test-set statistics (mean/std) into training,
    which quietly inflates evaluation scores.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train_scaled, X_test_scaled, scaler


def evaluate(model, X_test, y_test, name: str) -> dict:
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
    }

    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        if k != "model":
            print(f"{k:10s}: {v:.4f}")
    print("\nConfusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, preds))
    print("\n", classification_report(y_test, preds, target_names=["on_time", "delayed"]))
    print(y_test.value_counts())
    print(y_test.value_counts(normalize=True))
    return metrics


def train_and_evaluate():
    full_df, model_df = build_final_dataset()
    X_train, X_test, y_train, y_test = time_based_split(full_df, model_df)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    results = []

    # Logistic Regression: simple, interpretable baseline.
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_scaled, y_train)
    results.append(
        evaluate(
            log_reg,
            X_test_scaled,
            y_test,
            "Logistic Regression",
        )
    )

    # Random Forest: handles non-linear interactions (e.g. bad weather
    # AND high traffic compounding) that logistic regression can't
    # capture on its own -- good second baseline for comparison.
    # NOTE: trained on the SCALED features (X_train_scaled), not raw
    # X_train. Trees don't strictly need scaling to split correctly, but
    # predict.py always calls scaler.transform() before model.predict()
    # for whichever model is loaded -- training RF on raw values while
    # predict.py feeds it scaled values silently collapses its
    # predictions toward ~0.5 (verified: probabilities went from a
    # spread-out 0.17-0.78 range down to a flat ~0.47-0.50 cluster when
    # this mismatch was introduced). Keeping one shared preprocessing
    # path for every model avoids this class of bug entirely.
    rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    results.append(evaluate(rf, X_test_scaled, y_test, "Random Forest"))
    train_acc = rf.score(X_train_scaled, y_train)
    test_acc = rf.score(X_test_scaled, y_test)

    print(f"RF Train Accuracy: {train_acc:.4f}")
    print(f"RF Test Accuracy : {test_acc:.4f}")

    # Feature importance from the stronger tree-based model -- useful
    # for sanity-checking that the engineered features actually matter.
    importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(
        ascending=False
    )
    print("\nTop 10 most important features (Random Forest):")
    print(importances.head(10))
    importances.to_csv(
        os.path.join(MODELS_DIR, "feature_importance.csv")
    )

    # Save everything needed to reproduce predictions later.
    joblib.dump(log_reg, os.path.join(MODELS_DIR, "logistic_regression.joblib"))
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(
        X_train.columns.tolist(),
        os.path.join(MODELS_DIR, "feature_columns.joblib")
    )
    print(f"\nModels saved to {MODELS_DIR}")

    return pd.DataFrame(results)


if __name__ == "__main__":
    results_df = train_and_evaluate()
    print("\n=== Summary ===")
    print(results_df.set_index("model").round(4))