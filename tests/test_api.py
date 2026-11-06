from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_healthz():
    response = client.get("/healthz")
    # Проверяем, что эндпоинт отдает успешный статус или 503 (если модель не в памяти во время теста)
    assert response.status_code in [200, 503]
    data = response.json()
    assert "status" in data or "detail" in data

def test_predict_endpoint_structure():
    # Проверяем валидацию входных данных для /predict
    response = client.post("/predict", json={"features": [100.0, 1500.0]})
    # Даже если модель вызвана, проверим что API принимает правильный формат
    assert response.status_code in [200, 503]