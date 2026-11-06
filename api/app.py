from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import pickle
from pathlib import Path

app = FastAPI(title="MLOps Production Pipeline", version="1.0.0")

# Определяем абсолютный путь к модели
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model.pkl"

model = None
from pydantic import BaseModel, Field
from typing import List

class PredictionInput(BaseModel):
    # Ограничиваем ожидаемое количество признаков (например, ровно 3 признака: cost, rate, votes)
    features: List[float] = Field(..., min_items=3, max_items=3, description="Список из 3 признаков: [cost, rate, votes]")
# Загружаем модель при старте приложения
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Модель успешно загружена!")
except Exception as e:
    print(f"ОШИБКА при загрузке модели: {e}")

class PredictionInput(BaseModel):
    features: list[float]

@app.get("/healthz", status_code=status.HTTP_200_OK)
def health_check():
    """Эндпоинт для проверки здоровья сервиса (liveness/readiness probe)"""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model is not loaded"
        )
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict(data: PredictionInput):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model is not loaded. Service unavailable."
        )
    
    # Превращаем входной список в формат, понятный scikit-learn (2D массив)
    import numpy as np
    features_array = np.array(data.features).reshape(1, -1)
    
    prediction = model.predict(features_array)
    return {"prediction": float(prediction[0])}