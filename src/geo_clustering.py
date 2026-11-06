import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import folium
import os
from pathlib import Path

# Находим корень проекта
ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "zomato.csv.zip"

class GeoAnalyzer:
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

    def preprocess_and_mock_geo(self):
        print("Очистка и генерация GPS-координат для локаций...")
        # Убираем пустые локации и дубликаты ресторанов
        self.df = self.df.dropna(subset=['location'])
        self.df = self.df.drop_duplicates(subset=['name', 'location']).reset_index(drop=True)
        
        # Центр Бангалора
        base_lat, base_lon = 12.9716, 77.5946 
        
        # Каждому уникальному району даем случайные координаты в радиусе ~10 км от центра
        unique_locations = self.df['location'].unique()
        np.random.seed(42)
        loc_coords = {
            loc: (
                base_lat + np.random.uniform(-0.1, 0.1),
                base_lon + np.random.uniform(-0.1, 0.1)
            ) for loc in unique_locations
        }
        
        # Присваиваем координаты ресторанам с небольшим шумом (чтобы они не стояли в одной точке)
        self.df['latitude'] = self.df['location'].map(lambda x: loc_coords[x][0] + np.random.uniform(-0.005, 0.005))
        self.df['longitude'] = self.df['location'].map(lambda x: loc_coords[x][1] + np.random.uniform(-0.005, 0.005))
        
        # Матрица координат для DBSCAN (переводим в радианы для вычисления дистанции на сфере Земли)
        self.coords = np.radians(self.df[['latitude', 'longitude']].values)

    def run_dbscan(self):
        print("Запускаем DBSCAN для поиска ресторанных хабов...")
        # eps = 0.5 км (в радианах: 0.5 / 6371 радиус Земли), минимум 15 ресторанов для создания "хаба"
        kms_per_radian = 6371.0088
        epsilon = 0.5 / kms_per_radian
        
        self.dbscan = DBSCAN(eps=epsilon, min_samples=15, algorithm='ball_tree', metric='haversine')
        self.df['hub_cluster'] = self.dbscan.fit_predict(self.coords)
        
        n_clusters = len(set(self.df['hub_cluster'])) - (1 if -1 in self.df['hub_cluster'] else 0)
        n_noise = list(self.df['hub_cluster']).count(-1)
        
        print(f"Найдено плотных хабов: {n_clusters}")
        print(f"Ресторанов вне хабов (шум/изолированные): {n_noise}")

    def generate_map(self, output_file="restaurant_hubs.html"):
        print("Генерация интерактивной карты...")
        # Создаем базовую карту по центру первого ресторана
        m = folium.Map(
    location=[self.df['latitude'].mean(), self.df['longitude'].mean()],
    zoom_start=12,
    max_zoom = 16,
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
    attr='Esri'
)
        # Цвета для разных хабов (кластер -1 это серый шум)
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']
        
        # Берем случайную выборку из 1000 точек, чтобы браузер не завис при рендере
        sample_df = self.df.sample(n=min(1000, len(self.df)), random_state=42)
        
        for _, row in sample_df.iterrows():
            cluster_id = row['hub_cluster']
            color = 'gray' if cluster_id == -1 else colors[cluster_id % len(colors)]
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=4,
                popup=f"{row['name']} ({row['location']})",
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7
            ).add_to(m)
            
        m.save(output_file)
        print(f"Готово! Интерактивная карта сохранена как {output_file}. Открой её в браузере.")

if __name__ == "__main__":
    geo = GeoAnalyzer()
    geo.preprocess_and_mock_geo()
    geo.run_dbscan()
    geo.generate_map()