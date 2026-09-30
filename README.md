# Customer Churn Prediction Project

A simple end-to-end machine-learning project that generates synthetic customer data, trains a churn classifier, evaluates it with ROC-AUC, saves the trained pipeline, and exposes predictions through FastAPI.

## Project structure

```text
churn_project/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── churn_model.pkl
│   └── feature_columns.pkl
├── app.py
├── src/
│   ├── __init__.py
│   ├── api_server.py
│   ├── data_utils.py
│   ├── evaluate.py
│   ├── preprocessing.py
│   └── train.py
├── tests/
│   ├── test_model.py
│   └── test_preprocessing.py
├── requirements.txt
└── README.md
```

## 1. Install dependencies

Create/activate a virtual environment in PyCharm, then run:

```bash
python -m pip install -r requirements.txt
```

## 2. Train the model

From the project root:

```bash
python -m src.train
```

The trained pipeline is saved to `models/churn_model.pkl`.

## 3. Run tests

```bash
pytest -q
```

## 4. Start the Streamlit app

From the project root:

```bash
streamlit run app.py
```

The app will open in your browser. It provides a simple form where you can enter customer information and receive a churn probability and risk level.

## 5. Start the API

From the project root:

```bash
uvicorn src.api_server:app --reload
```

Then open the FastAPI documentation at `http://127.0.0.1:8000/docs`.

## Example API prediction

Send a POST request to `/predict` with:

```json
{
  "tenure_months": 4,
  "plan_type": "basic",
  "monthly_usage": 20,
  "support_tickets": 4
}
```

The API returns a churn probability and a simple risk level.

## Important note

The included data is synthetic and is intended for learning and testing the project pipeline. A production churn system should be trained and validated on representative historical customer data.
