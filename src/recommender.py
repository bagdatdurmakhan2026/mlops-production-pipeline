import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

class RestaurantRecommender:
    def __init__(self, data_path="data/zomato.csv.zip"):
        print("Загрузка данных...")
        try:
            self.df = pd.read_csv(data_path, compression='zip')
        except Exception:
            self.df = pd.read_csv(data_path, compression='zip', encoding='latin-1')
            
        self.scaler = StandardScaler()
        self.model = NearestNeighbors(metric='cosine', algorithm='brute')
        
    def preprocess(self):
        print("Очистка данных и фиче-инжиниринг...")
        
        # 0. Убиваем дубликаты по имени и локации, чтобы не рекомендовать одно и то же.
        # И сбрасываем индексы, это критически важно для работы k-NN!
        self.df = self.df.drop_duplicates(subset=['name', 'location']).reset_index(drop=True)
        
        # 1. Чистим рейтинг: безопасно режем строку через векторизованный .str
        self.df['rate_clean'] = self.df['rate'].astype(str).str.split('/').str[0].str.strip()
        self.df['rate_clean'] = pd.to_numeric(self.df['rate_clean'], errors='coerce')
        
        # 2. Чистим цены: убираем запятые ('1,200' -> 1200)
        self.df['cost_clean'] = self.df['approx_cost(for two people)'].astype(str).str.replace(',', '')
        self.df['cost_clean'] = pd.to_numeric(self.df['cost_clean'], errors='coerce')
        
        # 3. Фичи для рекомендаций
        features = ['cost_clean', 'rate_clean', 'votes']
        
        self.model_data = self.df[features].fillna(0)
        self.scaled_data = self.scaler.fit_transform(self.model_data)
        
    def fit(self):
        print("Обучение модели ближайших соседей (Cosine)...")
        self.model.fit(self.scaled_data)
        
    def recommend(self, restaurant_idx, top_k=5):
        # Ищем соседей. Берем top_k + 1, так как самый первый результат — это сам исходный ресторан
        distances, indices = self.model.kneighbors([self.scaled_data[restaurant_idx]], n_neighbors=top_k+1)
        
        # Отрезаем первый элемент (сам ресторан)
        top_indices = indices[0][1:]
        
        # Выводим инфу по найденным рекомендациям
        return self.df.iloc[top_indices][['name', 'location', 'rate', 'approx_cost(for two people)']]

if __name__ == "__main__":
    recommender = RestaurantRecommender()
    recommender.preprocess()
    recommender.fit()
    
    # Возьмем ресторан под индексом 10 для теста
    target_idx = 10
    target_rest = recommender.df.iloc[target_idx]
    
    print(f"\n🎯 ИЩЕМ ПОХОЖИЕ НА: {target_rest['name']}")
    print(f"Локация: {target_rest['location']} | Рейтинг: {target_rest['rate']} | Чек: {target_rest['approx_cost(for two people)']}")
    
    recommendations = recommender.recommend(target_idx)
    print("\n✅ РЕКОМЕНДАЦИИ:")
    print(recommendations.to_string(index=False))