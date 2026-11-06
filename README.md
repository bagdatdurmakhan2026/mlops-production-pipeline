# 🚀 MLOps Production Pipeline: Restaurant Rating Prediction

An industrial MLOps pipeline for predicting restaurant ratings based on the real-world Zomato dataset. The project is designed in compliance with software development standards: reproducible training, strict data validation, automated testing, and containerization.

---

## 🛠 Tech Stack
* **Language**: Python 3.14 / 3.10
* **ML & Data Science**: Scikit-learn (`HistGradientBoostingRegressor`), Pandas, NumPy
* **API**: FastAPI, Uvicorn, Pydantic
* **Testing**: Pytest, FastAPI TestClient, HTTPX
* **CI/CD & DevOps**: Docker, GitHub Actions

---

## 📊 Model Quality Metrics (on Holdout Set)

The model (`HistGradientBoostingRegressor`) is trained on a proper 80/20 data split and outperforms a simple Baseline (mean predictor).

| Metric | Main Model | Baseline (Dummy) |
| :--- | :--- | :--- |
| **RMSE** | *Check metrics.json* | *Check metrics.json* |
| **MAE** | *Check metrics.json* | *Check metrics.json* |
| **$R^2$ Score** | *Check metrics.json* | *Check metrics.json* |

*(Complete up-to-date metrics are automatically saved to `metrics.json` upon every training run).*

---

## 📁 Project Structure
```text
mlops-production-pipeline/
│
├── .github/
│   └── workflows/ci.yml     # Automated CI/CD pipeline (GitHub Actions)
├── api/
│   └── app.py               # FastAPI application and inference endpoints
├── data/
│   └── zomato.csv.zip       # Source dataset
├── src/
│   └── train.py             # Model training, validation, and metrics generation script
├── tests/
│   └── test_api.py          # Integration tests
├── .dockerignore
├── Dockerfile
├── requirements.txt         # Locked dependencies
├── model.pkl                # Serialized trained model
└── metrics.json             # Quality metrics artifact

1. Clone the Repository and Install Dependencies
git clone [https://github.com/bagdatdurmakhan2026/mlops-production-pipeline.git](https://github.com/bagdatdurmakhan2026/mlops-production-pipeline.git)
cd mlops-production-pipeline
pip install -r requirements.txt
2. Train the Model
Run the training script (it will unpack data, train the model, compute metrics, and output model.pkl and metrics.json):
python src/train.py
3. Run Tests
python -m pytest
4. Run FastAPI Server Locally
uvicorn api.app:app --reload
Once started, the interactive documentation will be available at: http://127.0.0.1:8000/docs
🐳 Running via Docker
Build and run the isolated container:
docker build -t mlops-app .
docker run --rm -p 8000:8000 mlops-app