"""
Train and compare multiple classifiers on the attrition data.
- Models: Logistic Regression, RandomForest, GradientBoosting
- Class weighting for imbalance
- Threshold tuning: best F1 and recall@precision >= PRECISION_TARGET
- Saves best model + metrics and per-model comparisons.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

PRECISION_TARGET = 0.70  # adjust if you want a stricter precision requirement
RANDOM_SEED = 42


def load_processed(cfg):
    processed_dir = Path(cfg["paths"]["processed_dir"])
    X_train = joblib.load(processed_dir / "X_train.joblib")
    X_test = joblib.load(processed_dir / "X_test.joblib")
    y_train = pd.read_csv(processed_dir / "y_train.csv")["left"].to_numpy()
    y_test = pd.read_csv(processed_dir / "y_test.csv")["left"].to_numpy()
    return X_train, X_test, y_train, y_test


def build_models():
    return {
        "log_reg": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
        "grad_boost": GradientBoostingClassifier(random_state=RANDOM_SEED),
    }


def sweep_thresholds(probas, y_true, precision_target=PRECISION_TARGET):
    best_f1 = (-1, 0.5)
    best_recall = (-1, 0.5)
    for thr in np.linspace(0.01, 0.99, 99):
        preds = (probas >= thr).astype(int)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1[0]:
            best_f1 = (f1, thr)
        if prec >= precision_target and rec > best_recall[0]:
            best_recall = (rec, thr)
    return {
        "best_f1": {"threshold": float(best_f1[1]), "f1": float(best_f1[0])},
        "recall_at_precision": {
            "threshold": float(best_recall[1]),
            "precision_target": float(precision_target),
            "recall": float(best_recall[0]),
        },
    }


def metrics_at_threshold(model, X, y, thr):
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= thr).astype(int)
    return {
        "threshold": float(thr),
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds),
        "roc_auc": roc_auc_score(y, proba),
    }


def train_and_select(cfg):
    X_train, X_test, y_train, y_test = load_processed(cfg)

    # validation split for threshold tuning
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_SEED
    )

    models = build_models()
    per_model = {}
    fitted = {}

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        val_proba = model.predict_proba(X_val)[:, 1]
        thresholds = sweep_thresholds(val_proba, y_val, PRECISION_TARGET)
        test_metrics = {
            "best_f1": metrics_at_threshold(model, X_test, y_test, thresholds["best_f1"]["threshold"]),
            "recall_at_precision": metrics_at_threshold(
                model, X_test, y_test, thresholds["recall_at_precision"]["threshold"]
            ),
        }
        per_model[name] = {
            "thresholds": thresholds,
            "test_metrics": test_metrics,
        }
        fitted[name] = model

    # choose best by F1 using the best_f1 threshold
    best_name = max(per_model, key=lambda k: per_model[k]["test_metrics"]["best_f1"]["f1"])
    best_model = fitted[best_name]
    best_thr = per_model[best_name]["thresholds"]["best_f1"]["threshold"]
    best_metrics = per_model[best_name]["test_metrics"]["best_f1"]

    model_dir = Path(cfg["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_dir / "model.joblib")

    with open(model_dir / "metrics.json", "w") as f:
        json.dump(
            {"model": best_name, "threshold": best_thr, **best_metrics, "precision_target": PRECISION_TARGET},
            f,
            indent=2,
        )
    with open(model_dir / "metrics_by_model.json", "w") as f:
        json.dump(per_model, f, indent=2)

    print(f"Best model: {best_name} @ threshold {best_thr:.2f}")
    print("Best-F1 test metrics:", best_metrics)


def main():
    import yaml

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    train_and_select(cfg)


if __name__ == "__main__":
    main()
