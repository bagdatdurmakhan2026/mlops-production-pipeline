import json
import os
import pickle
import zipfile
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path

# Находим корень проекта и папки
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

    # 1. Очищаем данные от пропусков в реальных колонках датасета Zomato
    df = df.dropna(subset=['votes', 'approx_cost(for two people)'])

    # 2. Строгая валидация целевой переменной (никаких случайных заглушек!)
    target_column = 'rate'
    if target_column not in df.columns:
        raise ValueError(f"❌ Ошибка: Колонка '{target_column}' обязательна для обучения модели, но отсутствует в датасете.")

    # Очистка таргета от '/5' и лишних пробелов
    df['rate_clean'] = df[target_column].astype(str).str.replace('/5', '').str.strip()
    df['rate_clean'] = pd.to_numeric(df['rate_clean'], errors='coerce')
    df = df.dropna(subset=['rate_clean'])
    
    y = df['rate_clean'].values

    # 3. Выделяем числовые признаки
    df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(str).str.replace(',', '').astype(float)
    X = df[['votes', 'approx_cost(for two people)']].values

    if len(X) == 0:
        raise ValueError("❌ Ошибка: После очистки данных не осталось объектов для обучения.")

    # 4. Деление на train / test (отложенная выборка)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"🚀 Обучение модели на {len(X_train)} объектах (тест: {len(X_test)})...")
    
    # Инициализация и обучение модели
    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    # 5. Оценка качества на отложенной выборке (тесте)
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "r2_score": round(float(r2), 4)
    }

    print("📈 Метрики качества модели на тесте:")
    for k, v in metrics.items():
        print(f"   - {k.upper()}: {v}")

    # 6. Сохранение метрик в файл
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"💾 Метрики успешно сохранены в {METRICS_PATH}")

    # 7. Сохранение обученной модели
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"💾 Модель успешно сохранена в {MODEL_PATH}")

if __name__ == "__main__":
    main()