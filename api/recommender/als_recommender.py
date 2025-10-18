from implicit.als import AlternatingLeastSquares
import scipy.sparse as sp
import pandas as pd
import numpy as np

class ALSRecommender:
    def __init__(self, csv_path="dataMatrix.csv"):
        # Загружаем CSV
        self.df = pd.read_csv(csv_path)
        
        # Уникальные пользователи и товары
        self.users = sorted(self.df["user_id"].unique())
        self.items = sorted(self.df["item_id"].unique())
        
        # Маппинги user/item -> индекс
        self.user_map = {u: i for i, u in enumerate(self.users)}
        self.reverse_user_map = {i: u for u, i in self.user_map.items()}
        self.item_map = {item: i for i, item in enumerate(self.items)}
        self.reverse_item_map = {i: item for item, i in self.item_map.items()}
        
        # Создаём разреженную матрицу user x item
        self._build_matrix()
        
        # Обучаем ALS модель
        self.fit_model()

    def _build_matrix(self):
        user_idx = self.df["user_id"].map(self.user_map).to_numpy()
        item_idx = self.df["item_id"].map(self.item_map).to_numpy()
        ratings = self.df["rating"].astype(float).to_numpy()
        
        self.matrix = sp.coo_matrix(
            (ratings, (user_idx, item_idx)),
            shape=(len(self.users), len(self.items))
        ).tocsr()  # user x item

    def fit_model(self):
        # ALS ожидает item x user
        self.model = AlternatingLeastSquares(factors=32, regularization=0.1, iterations=15)
        self.model.fit(self.matrix.T)  # item x user

    def recommend(self, user_id, N=5):
        if user_id not in self.user_map:
            return []

        u_idx = self.user_map[user_id]        # индекс пользователя
        user_items = self.matrix[u_idx]       # строка CSR для пользователя

        # Рекомендации
        recs, _ = self.model.recommend(
            u_idx,
            user_items,
            N=N,
            filter_already_liked_items=True
        )

        # Конвертируем обратно в оригинальные item_id
        return [self.reverse_item_map[i] for i in recs]
