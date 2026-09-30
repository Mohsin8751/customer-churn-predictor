from src.data_utils import generate_synthetic_churn
from src.train import train_pipeline


def test_train_pipeline_runs():
    df = generate_synthetic_churn(n_samples=500)
    clf, roc = train_pipeline(df=df, test_size=0.2)

    assert clf is not None
    assert 0.0 <= roc <= 1.0
    assert roc >= 0.5
