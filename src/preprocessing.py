"""Data splitting and preprocessing utilities."""

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


NON_PREDICTIVE_COLUMNS = {"customer_id"}


def split_data(
    df: pd.DataFrame,
    target_col: str = "churn",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a dataframe while excluding identifiers from model features."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    drop_columns = [target_col, *[c for c in NON_PREDICTIVE_COLUMNS if c in df.columns]]
    X = df.drop(columns=drop_columns)
    y = df[target_col]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing for numerical and categorical columns."""
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
