import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

class CuisineClassifier:
    def __init__(self, data_path="data/zomato.csv.zip"):
        print("Загрузка данных...")
        try:
            self.df = pd.read_csv(data_path, compression='zip')
        except Exception:
            self.df = pd.read_csv(data_path, compression='zip', encoding='latin-1')
            
        self.le = LabelEncoder()
        self.model = HistGradientBoostingClassifier(random_state=42)

    def preprocess(self):
        print("Очистка фичей и подготовка таргета...")
        # 1. Восстанавливаем наши чистые фичи
        self.df['rate_clean'] = self.df['rate'].astype(str).str.split('/').str[0].str.strip()
        self.df['rate_clean'] = pd.to_numeric(self.df['rate_clean'], errors='coerce')
        
        self.df['cost_clean'] = self.df['approx_cost(for two people)'].astype(str).str.replace(',', '')
        self.df['cost_clean'] = pd.to_numeric(self.df['cost_clean'], errors='coerce')
        
        # 2. Вытаскиваем таргет (основную кухню)
        self.df = self.df.dropna(subset=['cuisines'])
        self.df['main_cuisine'] = self.df['cuisines'].astype(str).str.split(',').str[0].str.strip()
        
        # 3. Фильтруем Топ-10 кухонь (борьба с шумом)
        top_10 = self.df['main_cuisine'].value_counts().nlargest(10).index
        self.df = self.df[self.df['main_cuisine'].isin(top_10)]
        print(f"Выбраны кухни: {list(top_10)}")
        
        # 4. Формируем X и y
        features = ['cost_clean', 'rate_clean', 'votes']
        self.X = self.df[features].fillna(0)
        self.y = self.le.fit_transform(self.df['main_cuisine'])

    def train(self):
        print("\nРазбиваем на train/test (80/20)...")
        # stratify=self.y гарантирует, что в тесте и трейне будет одинаковая пропорция классов
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print("Считаем веса для балансировки классов...")
        # compute_sample_weight дает редким классам больший вес, чтобы бустинг их не игнорировал
        weights = compute_sample_weight(class_weight='balanced', y=y_train)
        
        print("Обучаем HistGradientBoosting...")
        self.model.fit(X_train, y_train, sample_weight=weights)
        
        print("\n=== ОТЧЕТ ПО КЛАССИФИКАЦИИ ===")
        y_pred = self.model.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
        print(classification_report(y_test, y_pred, target_names=self.le.classes_))

if __name__ == "__main__":
    classifier = CuisineClassifier()
    classifier.preprocess()
    classifier.train()