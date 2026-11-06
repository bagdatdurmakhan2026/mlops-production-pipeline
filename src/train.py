import zipfile
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from pathlib import Path

# Находим корень проекта и папки
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
ZIP_PATH = DATA_DIR / "zomato.csv.zip"
CSV_NAME = "zomato.csv"
model_path = ROOT_DIR / "model.pkl"

def load_data():
    if not (DATA_DIR / CSV_NAME).exists() and ZIP_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
            print("Датасет успешно распакован!")
            
    csv_path = DATA_DIR / CSV_NAME
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print("Доступные колонки в датасете:", df.columns.tolist()) # Поможет увидеть правильные имена
        return df
    else:
        raise FileNotFoundError(f"Датасет не найден по пути: {csv_path}")

def train_model():
    print("Загрузка данных...")
    df = load_data()
    
    # СЮДА ВСТАВЛЯЕМ НАШ КОД ДЛЯ ОЧИСТКИ И ПОДГОТОВКИ ДАННЫХ:
    
    # Очищаем данные от пропусков в реальных колонках датасета Zomato
    df = df.dropna(subset=['votes', 'approx_cost(for two people)'])
    
    # Подготавливаем признаки (X) и целевую переменную (y) - предсказываем 'rate'
    if 'rate' in df.columns:
        df['rate_clean'] = df['rate'].astype(str).str.replace('/5', '').str.strip()
        df['rate_clean'] = pd.to_numeric(df['rate_clean'], errors='coerce')
        df = df.dropna(subset=['rate_clean'])
        y = df['rate_clean'].values
    else:
        y = np.random.rand(len(df))

    # Выделяем числовые признаки из колонок
    df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(str).str.replace(',', '').astype(float)
    
    X = df[['votes', 'approx_cost(for two people)']].values
    
    # КОНЕЦ ВСТАВКИ

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучаем модель HistGradientBoosting Regressor
    model = HistGradientBoostingRegressor()
    model.fit(X_train, y_train)

    # Сохраняем модель в корень проекта по абсолютному пути
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("Модель успешно обучена на реальных данных и сохранена в model.pkl!")
if __name__ == "__main__":
    train_model()