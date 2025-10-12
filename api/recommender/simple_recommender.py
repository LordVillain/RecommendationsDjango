import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class SimpleRecommender:
    """Простейшая модель рекомендаций на фейковых числовых данных"""

    def __init__(self):
        # Теперь пользователи — числа, а не строки
        self.users = list(range(1, 11))  # user_id = 1..10
        self.items = [f"item_{i}" for i in range(1, 8)]
        self.df = self._generate_fake_data()
        self.matrix = self._build_matrix()

    def _generate_fake_data(self):
        data = []
        for u in self.users:
            for it in self.items:
                rating = 1 if random.random() > 0.5 else 0
                data.append({"user_id": u, "item_id": it, "rating": rating})
        return pd.DataFrame(data)

    def _build_matrix(self):
        """Строим user-item матрицу"""
        return self.df.pivot_table(
            index="user_id", columns="item_id", values="rating", fill_value=0
        )

    def recommend(self, user_id: int):
        if user_id not in self.matrix.index:
            return []

        similarity = cosine_similarity(self.matrix)
        sim_df = pd.DataFrame(similarity, index=self.matrix.index, columns=self.matrix.index)
        similar_users = sim_df[user_id].sort_values(ascending=False)[1:3].index

        user_items = set(self.df[self.df.user_id == user_id]["item_id"])
        rec_items = (
            self.df[self.df.user_id.isin(similar_users) & (self.df.rating == 1)]
            .item_id.value_counts()
            .index.tolist()
        )
        recommendations = [it for it in rec_items if it not in user_items][:3]

        # 💡 Если нет рекомендаций, выдаём случайные товары
        if not recommendations:
            recommendations = random.sample(self.items, 3)

        return recommendations

