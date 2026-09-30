import pandas as pd

from src.preprocessing import build_preprocessor


def test_preprocessor_builds():
    df = pd.DataFrame(
        {
            "tenure_months": [1, 12],
            "plan_type": ["basic", "premium"],
            "monthly_usage": [20.0, 60.0],
            "support_tickets": [0, 3],
            "churn": [1, 0],
        }
    )
    X = df.drop(columns=["churn"])
    preprocessor = build_preprocessor(X)
    X_transformed = preprocessor.fit_transform(X)
    assert X_transformed.shape[0] == len(X)


def test_customer_id_is_not_a_model_feature():
    from src.preprocessing import split_data

    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5, 6],
            "tenure_months": [1, 12, 4, 30, 8, 20],
            "plan_type": ["basic", "premium", "basic", "standard", "premium", "standard"],
            "monthly_usage": [20, 60, 25, 70, 55, 45],
            "support_tickets": [3, 0, 4, 1, 0, 2],
            "churn": [1, 0, 1, 0, 0, 1],
        }
    )
    X_train, _, _, _ = split_data(df, test_size=0.33)
    assert "customer_id" not in X_train.columns
