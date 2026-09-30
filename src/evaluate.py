"""Evaluate a saved churn model on a labelled dataframe."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"


def evaluate_on_df(df: pd.DataFrame, target_col: str = "churn") -> float:
    """Return ROC-AUC for the saved model on a labelled dataframe."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    clf = joblib.load(MODEL_PATH)
    X = df.drop(columns=[target_col, "customer_id"], errors="ignore")
    y = df[target_col]
    preds = clf.predict_proba(X)[:, 1]
    roc = roc_auc_score(y, preds)
    print(f"Test ROC-AUC: {roc:.4f}")
    return roc
