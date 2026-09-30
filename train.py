"""Train and save the customer-churn classification pipeline."""

from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from src.data_utils import generate_synthetic_churn
from src.preprocessing import build_preprocessor, split_data

RANDOM_SEED = 42
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "churn_model.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.pkl"


def train_pipeline(
    df=None,
    target_col: str = "churn",
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
):
    """Train the pipeline, print ROC-AUC, and save model artifacts."""
    if df is None:
        df = generate_synthetic_churn(n_samples=8_000)

    X_train, X_valid, y_train, y_valid = split_data(
        df, target_col, test_size, random_state
    )
    preprocessor = build_preprocessor(X_train)

    model = GradientBoostingClassifier(random_state=random_state)
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    clf.fit(X_train, y_train)
    y_valid_pred = clf.predict_proba(X_valid)[:, 1]
    roc = roc_auc_score(y_valid, y_valid_pred)
    print(f"Validation ROC-AUC: {roc:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(list(X_train.columns), FEATURE_COLUMNS_PATH)

    return clf, roc


if __name__ == "__main__":
    train_pipeline()
