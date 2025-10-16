import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class SimpleRecommender:
    def __init__(self):
        self.users = list(range(1, 11))
        self.item_ids = list(range(1, 10))
        self.df = self._generate_fake_data()
        self.matrix = self._build_matrix()

    def _generate_fake_data(self):
        data = []
        for u in self.users:
            for item_id in self.item_ids:
                rating = 1 if random.random() > 0.5 else 0
                data.append({"user_id": u, "item_id": item_id, "rating": rating})
        return pd.DataFrame(data)

    def _build_matrix(self):
        return self.df.pivot_table(
            index="user_id", columns="item_id", values="rating", fill_value=0
        )

    def recommend(self, user_id: int) -> list[int]:
        print(f"Received user_id: {user_id}")
        
        if user_id not in self.matrix.index:
            return []

        # Вычисляем косинусное сходство между всеми пользователями
        similarity = cosine_similarity(self.matrix)
        sim_df = pd.DataFrame(
            similarity,
            index=self.matrix.index,
            columns=self.matrix.index
        )

        # Находим 2 самых похожих пользователя (исключая самого себя)
        similar_users = sim_df[user_id].sort_values(ascending=False).index[1:3]

        # Товары, которые уже лайкал целевой пользователь
        user_seen_items = set(
            self.df[self.df["user_id"] == user_id]["item_id"].tolist()
        )
        # Товары, лайкнутые похожими пользователями
        liked_by_similar = self.df[
            (self.df["user_id"].isin(similar_users)) &
            (self.df["rating"] == 1)
        ]["item_id"]

        # Сортируем по популярности среди похожих
        rec_candidates = liked_by_similar.value_counts().index.tolist()

        # Фильтруем: только то, чего пользователь ещё не видел
        recommendations = [
            item_id for item_id in rec_candidates
            if item_id not in user_seen_items
        ][:3]

        # Если нечего рекомендовать — случайные товары
        if not recommendations:
            recommendations = random.sample(self.item_ids, 3)

        print("Recommendations:", recommendations)

        return recommendations