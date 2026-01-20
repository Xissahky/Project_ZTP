from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

def _safe_roc_auc(y_true, y_proba) -> float | None:
    # roc_auc requires both classes present
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_proba))
    except Exception:
        return None

def evaluate_classifier(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)

    # Try probabilities if model supports it
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": _safe_roc_auc(y_test, y_proba) if y_proba is not None else None,
    }
    return metrics

def compare_models(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols].sort_values(by="f1", ascending=False)
