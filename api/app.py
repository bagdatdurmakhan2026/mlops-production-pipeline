import pickle
import numpy as np
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Restaurant Rating MLOps API")

# Path(__file__).resolve().parent.parent указывает на корень проекта
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Модель успешно загружена!")
except Exception as e:
    print(f"ОШИБКА при загрузке модели: {e}")
    model = None

class RestaurantFeatures(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: RestaurantFeatures):
    if model is None:
        return {"error": "Model not loaded", "status": "failed"}
    
    # Превращаем входящий список в матрицу для предсказания
    X = np.array(data.features).reshape(1, -1)
    prediction = model.predict(X)
    
    return {
        "prediction": float(prediction.item()),
        "status": "success"
    }