# tests/test_api.py

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_schema():
    response = client.get("/health")
    data = response.json()
    assert "status" in data
    assert "model_version" in data
    assert "test_mae" in data
    assert data["status"] == "ok"


def test_predict_endpoint():
    response = client.post("/predict", json={
        "is_weekend": 1,
        "promo_spend": 12000,
        "table_occupancy": 0.82,
        "slot_occupancy": 0.74,
        "local_event": 1,
    })
    assert response.status_code == 200


def test_predict_response_schema():
    response = client.post("/predict", json={
        "is_weekend": 1,
        "promo_spend": 12000,
        "table_occupancy": 0.82,
        "slot_occupancy": 0.74,
        "local_event": 1,
    })
    data = response.json()
    assert "predicted_revenue" in data
    assert data["predicted_revenue"] > 0


def test_predict_invalid_input():
    response = client.post("/predict", json={
        "is_weekend": 5,
        "promo_spend": -100,
        "table_occupancy": 2.0,
        "slot_occupancy": 0.74,
        "local_event": 1,
    })
    assert response.status_code == 422
