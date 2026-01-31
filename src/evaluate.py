"""
Evaluate the saved best model using tuned thresholds.
Reports both best-F1 and recall-at-precision regimes.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    precision_score,
    recall_score,
)

from src.utils import ensure_dir


def load_data(cfg):
    processed = Path(cfg["paths"]["processed_dir"])
    X_test = joblib.load(processed / "X_test.joblib")
    y_test = pd.read_csv(processed / "y_test.csv")["left"].to_numpy()
    return X_test, y_test


def load_thresholds(cfg):
    model_dir = Path(cfg["paths"]["model_dir"])
    thresholds = {"best_f1": 0.5, "recall_at_precision": 0.5}
    best_model_name = None
    metrics_file = model_dir / "metrics.json"
    by_model_file = model_dir / "metrics_by_model.json"

    if metrics_file.exists():
        data = json.loads(metrics_file.read_text())
        thresholds["best_f1"] = data.get("threshold", 0.5)
        best_model_name = data.get("model")

    if by_model_file.exists() and best_model_name:
        by_model = json.loads(by_model_file.read_text())
        if best_model_name in by_model:
            thresholds["recall_at_precision"] = by_model[best_model_name]["thresholds"]["recall_at_precision"]["threshold"]
    return thresholds, best_model_name


def metrics_for_threshold(model, X, y, thr):
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= thr).astype(int)
    return {
        "threshold": float(thr),
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds),
        "roc_auc": roc_auc_score(y, proba),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
        "proba": proba,  # for plotting
    }


def plot_curves(y_true, proba, figures_dir):
    ensure_dir(figures_dir)
    fpr, tpr, _ = roc_curve(y_true, proba)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    roc_path = Path(figures_dir) / "roc_curve.png"
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()

    prec, rec, _ = precision_recall_curve(y_true, proba)
    plt.figure()
    plt.plot(rec, prec, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    pr_path = Path(figures_dir) / "pr_curve.png"
    plt.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(roc_path), str(pr_path)


def evaluate(cfg):
    model_dir = Path(cfg["paths"]["model_dir"])
    model = joblib.load(model_dir / "model.joblib")
    X_test, y_test = load_data(cfg)
    thresholds, _ = load_thresholds(cfg)

    best_metrics = metrics_for_threshold(model, X_test, y_test, thresholds["best_f1"])
    recall_pref_metrics = metrics_for_threshold(model, X_test, y_test, thresholds["recall_at_precision"])

    roc_path, pr_path = plot_curves(y_test, best_metrics["proba"], cfg["paths"]["figures_dir"])
    best_metrics.pop("proba", None)
    recall_pref_metrics.pop("proba", None)

    metrics_out = {
        "best_f1": best_metrics,
        "recall_at_precision": recall_pref_metrics,
    }

    out_path = ensure_dir(cfg["paths"]["reports_dir"])
    with open(out_path / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    print("Eval metrics (best F1):", best_metrics)
    print("Eval metrics (recall@precision):", recall_pref_metrics)
    print("Plots:", roc_path, pr_path)


def main():
    import yaml
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    evaluate(cfg)


if __name__ == "__main__":
    main()
