import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
model_path = ROOT_DIR / "model.pkl"

def train_model():
    # Создаем демонстрационный датасет ресторанов для обучения
    np.random.seed(42)
    X = np.random.rand(1000, 3) # 3 признака: цена, голоса, географический фактор
    y = X @ np.array([1.5, 2.5, 3.0]) + np.random.randn(1000) * 0.1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучаем модель HistGradientBoosting Regressor
    model = HistGradientBoostingRegressor()
    model.fit(X_train, y_train)

    # Сохраняем модель в корень проекта через pickle
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print("Модель успешно обучена и сохранена в model.pkl!")

if __name__ == "__main__":
    train_model()