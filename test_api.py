"""Tests for the Churn Prediction API."""

from fastapi.testclient import TestClient
from run import app, MODEL_NAMES
import numpy as np

client = TestClient(app)

SAMPLE = {
    "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 12, "PhoneService": "Yes",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 70.0, "TotalCharges": 840.0,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_valid():
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["churn"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_all_models():
    for name in MODEL_NAMES:
        payload = {**SAMPLE, "model_name": name}
        r = client.post("/predict", json=payload)
        assert r.status_code == 200, f"{name} failed: {r.json()}"
        body = r.json()
        assert body["churn"] in (0, 1)
        assert 0.0 <= body["probability"] <= 1.0


def test_predict_invalid_type():
    r = client.post("/predict", json={"SeniorCitizen": "bad"})
    assert r.status_code == 422


def test_predict_missing_field():
    r = client.post("/predict", json={"SeniorCitizen": 0})
    assert r.status_code == 422


def test_feature_vector_length():
    from predict import build_features
    row_df = build_features(SAMPLE)
    assert row_df.shape[1] == 44


def test_model_consistency():
    r1 = client.post("/predict", json=SAMPLE)
    r2 = client.post("/predict", json=SAMPLE)
    assert r1.json() == r2.json()
