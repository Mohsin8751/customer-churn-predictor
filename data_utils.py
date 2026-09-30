"""Utilities for generating the synthetic churn dataset used by the project."""

import numpy as np
import pandas as pd

RANDOM_SEED = 42


def generate_synthetic_churn(n_samples: int = 10_000) -> pd.DataFrame:
    """Generate a reproducible synthetic customer-churn dataset.

    The target is generated from the customer features with a deliberately
    visible signal so the training pipeline can be tested reliably.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")

    rng = np.random.default_rng(RANDOM_SEED)

    tenure_months = rng.integers(1, 120, size=n_samples)
    plan_type = rng.choice(
        ["basic", "standard", "premium"],
        size=n_samples,
        p=[0.5, 0.3, 0.2],
    )
    monthly_usage = rng.normal(loc=50, scale=20, size=n_samples).clip(0, None)
    support_tickets = rng.poisson(lam=1.2, size=n_samples)

    # Strong enough signal for a small test dataset while keeping the data
    # realistic enough for demonstrating a churn-classification workflow.
    churn_prob = (
        0.05
        + 0.30 * (tenure_months < 6)
        + 0.25 * (plan_type == "basic")
        + 0.20 * (monthly_usage < 30)
        + 0.35 * (support_tickets > 2)
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.95)
    churn = (rng.random(n_samples) < churn_prob).astype(int)

    return pd.DataFrame(
        {
            "customer_id": np.arange(n_samples),
            "tenure_months": tenure_months,
            "plan_type": plan_type,
            "monthly_usage": monthly_usage,
            "support_tickets": support_tickets,
            "churn": churn,
        }
    )
