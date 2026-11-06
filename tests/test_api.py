from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data.get("model_loaded") is True

def test_predict():
    # Передаем корректные реалистичные признаки: [votes, approx_cost]
    payload = {"features": [1200.0, 1500.0]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Жестко проверяем, что API возвращает поле "prediction" со значением-числом
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))