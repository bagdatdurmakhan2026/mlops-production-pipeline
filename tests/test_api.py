from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    # Проверяем наличие статуса и то, что модель загружена
    assert "status" in data
    assert data.get("model_loaded") is True

def test_predict():
    # Передаем ровно 2 признака, которые ожидает модель
    payload = {"features": [1200.0, 4.1]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data or "predicted_rating" in data