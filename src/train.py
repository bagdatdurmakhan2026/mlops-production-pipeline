import json
import os
import pickle
import zipfile
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor  # <--- Импортируем Baseline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
ZIP_PATH = DATA_DIR / "zomato.csv.zip"
CSV_NAME = "zomato.csv"
MODEL_PATH = ROOT_DIR / "model.pkl"
METRICS_PATH = ROOT_DIR / "metrics.json"

def load_data():
    if not (DATA_DIR / CSV_NAME).exists() and ZIP_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
            print("Датасет успешно распакован!")

    csv_path = DATA_DIR / CSV_NAME
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print("Доступные колонки в датасете:", df.columns.tolist())
        return df
    else:
        raise FileNotFoundError(f"Датасет не найден по пути: {csv_path}")

def main():
    print("Загрузка данных...")
    df = load_data()

    df = df.dropna(subset=['votes', 'approx_cost(for two people)'])

    target_column = 'rate'
    if target_column not in df.columns:
        raise ValueError(f"❌ Ошибка: Колонка '{target_column}' обязательна для обучения модели, но отсутствует в датасете.")

    df['rate_clean'] = df[target_column].astype(str).str.replace('/5', '').str.strip()
    df['rate_clean'] = pd.to_numeric(df['rate_clean'], errors='coerce')
    df = df.dropna(subset=['rate_clean'])
    
    y = df['rate_clean'].values

    df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(str).str.replace(',', '').astype(float)
    X = df[['votes', 'approx_cost(for two people)']].values

    if len(X) == 0:
        raise ValueError("❌ Ошибка: После очистки данных не осталось объектов для обучения.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 1. Обучаем Baseline (простая модель, предсказывающая среднее)
    print("📊 Обучение Baseline модели (DummyRegressor)...")
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    y_pred_baseline = baseline.predict(X_test)

    baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
    baseline_mae = mean_absolute_error(y_test, y_pred_baseline)
    baseline_r2 = r2_score(y_test, y_pred_baseline)

    print(f"   - Baseline RMSE: {round(baseline_rmse, 4)}")

    # 2. Обучаем основную модель
    print(f"🚀 Обучение основной модели (HistGradientBoostingRegressor) на {len(X_train)} объектах...")
    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("📈 Метрики качества модели на тесте:")
    print(f"   - RMSE: {round(rmse, 4)} (против Baseline: {round(baseline_rmse, 4)})")
    print(f"   - MAE: {round(mae, 4)}")
    print(f"   - R2 Score: {round(r2, 4)}")

    # 3. Сохраняем расширенные метрики в JSON
    metrics = {
        "model": {
            "rmse": round(float(rmse), 4),
            "mae": round(float(mae), 4),
            "r2_score": round(float(r2), 4)
        },
        "baseline": {
            "rmse": round(float(baseline_rmse), 4),
            "mae": round(float(baseline_mae), 4),
            "r2_score": round(float(baseline_r2), 4)
        }
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"💾 Метрики с baseline успешно сохранены в {METRICS_PATH}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"💾 Модель успешно сохранена в {MODEL_PATH}")

if __name__ == "__main__":
    main()