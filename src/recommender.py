import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pathlib import Path

# Находим корень проекта
ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "zomato.csv.zip"

class RestaurantRecommender:
    def __init__(self, data_path=None):
        if data_path is None:
            data_path = DEFAULT_DATA_PATH
        else:
            data_path = Path(data_path)
            
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
    def fit_kmeans(self, n_clusters=5):
        print(f"Разбиваем рестораны на {n_clusters} кластеров через K-Means...")
        # Инициализируем и обучаем K-Means
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.kmeans.fit(self.scaled_data)
        
        # Записываем номер кластера для каждого ресторана прямо в датафрейм
        self.df['cluster'] = self.kmeans.labels_
        
    def get_cluster_sample(self, cluster_id, sample_size=5):
        # Вытаскиваем случайные рестораны из нужного кластера, чтобы посмотреть, кто там обитает
        cluster_data = self.df[self.df['cluster'] == cluster_id]
        return cluster_data[['name', 'location', 'rate', 'approx_cost(for two people)']].head(sample_size)

if __name__ == "__main__":
    recommender = RestaurantRecommender()
    recommender.preprocess()
    
    # 1. Тестируем Cosine Similarity
    recommender.fit()
    target_idx = 10
    target_rest = recommender.df.iloc[target_idx]
    
    print(f"\n🎯 ИЩЕМ ПОХОЖИЕ НА: {target_rest['name']}")
    print(f"Чек: {target_rest['approx_cost(for two people)']} | Рейтинг: {target_rest['rate']}")
    print("\n✅ РЕКОМЕНДАЦИИ (Cosine):")
    print(recommender.recommend(target_idx).to_string(index=False))
    
    # 2. Тестируем K-Means
    print("\n" + "="*50 + "\n")
    recommender.fit_kmeans(n_clusters=5)
    
    # Узнаем, в какой кластер попал наш таргет
    target_cluster = recommender.df.iloc[target_idx]['cluster']
    print(f"\nНаш ресторан '{target_rest['name']}' попал в Кластер №{target_cluster}")
    
    print(f"\n✅ ДРУГИЕ РЕСТОРАНЫ ИЗ КЛАСТЕРА №{target_cluster}:")
    print(recommender.get_cluster_sample(target_cluster).to_string(index=False))