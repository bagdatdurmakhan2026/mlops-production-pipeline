# 🚀 MLOps Production Pipeline: Restaurant Rating Prediction

Промышленный MLOps-пайплайн для предсказания рейтинга ресторанов на базе реального датасета Zomato. Проект спроектирован с соблюдением стандартов разработки: воспроизводимое обучение, строгая валидация данных, автоматическое тестирование и контейнеризация.

---

## 🛠 Технологии и стек
* **Язык**: Python 3.14 / 3.10
* **ML & Data Science**: Scikit-learn (`HistGradientBoostingRegressor`), Pandas, NumPy
* **API**: FastAPI, Uvicorn, Pydantic
* **Тестирование**: Pytest, FastAPI TestClient, HTTPX
* **CI/CD & DevOps**: Docker, GitHub Actions

---

## 📊 Метрики качества модели (на отложенной выборке)

Модель (`HistGradientBoostingRegressor`) обучена на честной разделенной выборке (80/20) и превосходит простой Baseline (среднее значение).

| Метрика | Основная модель | Baseline (Dummy) |
| :--- | :--- | :--- |
| **RMSE** | *Смотри в metrics.json* | *Смотри в metrics.json* |
| **MAE** | *Смотри в metrics.json* | *Смотри в metrics.json* |
| **$R^2$ Score** | *Смотри в metrics.json* | *Смотри в metrics.json* |

*(Полные актуальные данные автоматически сохраняются в файле `metrics.json` при каждом переобучении).*

---

## 📁 Структура проекта
```text
mlops-production-pipeline/
│
├── .github/
│   └── workflows/ci.yml     # Автоматический CI/CD пайплайн (GitHub Actions)
├── api/
│   └── app.py               # FastAPI приложение и эндпоинты инференса
├── data/
│   └── zomato.csv.zip       # Исходный датасет
├── src/
│   └── train.py             # Скрипт обучения модели, валидации и генерации метрик
├── tests/
│   └── test_api.py          # Интеграционные тесты
├── .dockerignore
├── Dockerfile
├── requirements.txt         # Зафиксированные зависимости
├── model.pkl                # Сериализованная обученная модель
└── metrics.json             # Артефакт с метриками качества

1. Клонирование репозитория и установка зависимостей
git clone [https://github.com/bagdatdurmakhan2026/mlops-production-pipeline.git](https://github.com/bagdatdurmakhan2026/mlops-production-pipeline.git)
cd mlops-production-pipeline
pip install -r requirements.txt
2. Обучение модели
Запусти скрипт обучения (он распакует данные, обучит модель, посчитает метрики и запишет model.pkl и metrics.json):
python src/train.py
3. Запуск тестов
python -m pytest
4. Запуск FastAPI сервера локально
uvicorn api.app:app --reload
После запуска документация будет доступна по адресу: http://127.0.0.1:8000/docs
🐳 Запуск через Docker
Сборка и запуск изолированного контейнера:
docker build -t mlops-app .
docker run --rm -p 8000:8000 mlops-app