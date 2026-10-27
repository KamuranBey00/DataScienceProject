"""Shared metrics, threshold rule and prediction-file format."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from src.config import ID_COL, PREDS_DIR

PRED_COLUMNS = [ID_COL, "y_true", "y_score", "y_pred", "threshold"]


def evaluate(y_true, y_score, y_pred) -> dict:
    """Main metrics are macro-F1 and PR-AUC; accuracy is informational only."""
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def pick_threshold(y_true, y_score) -> float:
    """Threshold that maximises macro-F1. Call on the validation split only."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    candidates = np.unique(np.quantile(y_score, np.linspace(0.01, 0.99, 99)))
    scores = [f1_score(y_true, (y_score >= t).astype(int), average="macro") for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def save_preds(name: str, row_id, y_true, y_score, threshold: float):
    """Write models/preds/{name}.csv in the shared format (test split)."""
    y_score = np.asarray(y_score, dtype=float)
    preds = pd.DataFrame(
        {
            ID_COL: np.asarray(row_id),
            "y_true": np.asarray(y_true, dtype=int),
            "y_score": y_score,
            "y_pred": (y_score >= threshold).astype(int),
            "threshold": threshold,
        }
    )
    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    path = PREDS_DIR / f"{name}.csv"
    preds.to_csv(path, index=False)
    return path


def load_preds(name: str) -> pd.DataFrame:
    path = PREDS_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; its owner has not pushed it yet.")
    preds = pd.read_csv(path)
    missing = set(PRED_COLUMNS) - set(preds.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns {sorted(missing)}")
    return preds
