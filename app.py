"""Streamlit web app for the customer churn prediction project."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained model was not found. Run 'python -m src.train' first."
        )
    return joblib.load(MODEL_PATH)


def risk_details(probability: float):
    if probability >= 0.70:
        return "High", "This customer has a relatively high predicted churn probability."
    if probability >= 0.40:
        return "Medium", "This customer has a moderate predicted churn probability."
    return "Low", "This customer has a relatively low predicted churn probability."


st.title("📊 Customer Churn Predictor")
st.write(
    "Enter a customer's information below to estimate the probability that they may churn."
)

try:
    model = load_model()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

with st.form("customer_form"):
    st.subheader("Customer information")

    tenure_months = st.number_input(
        "Tenure (months)", min_value=1, max_value=120, value=12, step=1
    )
    plan_type = st.selectbox("Plan type", ["basic", "standard", "premium"])
    monthly_usage = st.number_input(
        "Monthly usage", min_value=0.0, value=50.0, step=1.0
    )
    support_tickets = st.number_input(
        "Support tickets", min_value=0, value=1, step=1
    )

    submitted = st.form_submit_button("Predict churn", type="primary")

if submitted:
    customer = pd.DataFrame(
        [
            {
                "tenure_months": tenure_months,
                "plan_type": plan_type,
                "monthly_usage": monthly_usage,
                "support_tickets": support_tickets,
            }
        ]
    )

    probability = float(model.predict_proba(customer)[0, 1])
    risk, explanation = risk_details(probability)

    st.divider()
    st.subheader("Prediction")
    st.metric("Churn probability", f"{probability:.1%}")

    if risk == "High":
        st.error(f"Risk level: {risk}")
    elif risk == "Medium":
        st.warning(f"Risk level: {risk}")
    else:
        st.success(f"Risk level: {risk}")

    st.progress(probability)
    st.caption(explanation)

st.divider()
st.caption(
    "Educational demo: the included model was trained on synthetic data. "
    "For real business use, retrain and validate it with representative historical customer data."
)
