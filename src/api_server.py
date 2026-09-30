"""FastAPI service for customer-churn predictions."""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")
model = joblib.load(MODEL_PATH)


class Customer(BaseModel):
    tenure_months: int = Field(ge=1)
    plan_type: str
    monthly_usage: float = Field(ge=0)
    support_tickets: int = Field(ge=0)


@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: Customer):
    data = customer.model_dump()
    df = pd.DataFrame([data])
    probability = float(model.predict_proba(df)[0, 1])

    if probability >= 0.70:
        risk = "high"
    elif probability >= 0.40:
        risk = "medium"
    else:
        risk = "low"

    return {
        "churn_probability": round(probability, 4),
        "risk_level": risk,
    }
