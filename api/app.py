import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Restaurant Rating MLOps API")

# Загружаем сохраненную модель при старте приложения
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception:
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