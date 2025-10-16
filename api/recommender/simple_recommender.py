import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class SimpleRecommender:
    def __init__(self):
        self.users = list(range(1, 101))
        self.item_ids = list(range(1, 20))
        self.df = self._generate_fake_data()
        self.matrix = self._build_matrix()

    def _generate_fake_data(self):
        data = []
        total_ones = 0
        for u in self.users:
            user_ones = 0
            for item_id in self.item_ids:
                rating = 1 if random.random() > 0.5 else 0
                if rating == 1:
                    user_ones += 1
                data.append({"user_id": u, "item_id": item_id, "rating": rating})
            total_ones += user_ones
        return pd.DataFrame(data)

    def _build_matrix(self):
        return self.df.pivot_table(
            index="user_id", columns="item_id", values="rating", fill_value=0
        )

    def recommend(self, user_id: int) -> list[int]:
        print(f"=== RECOMMEND FOR USER {user_id} ===")

        if user_id not in self.matrix.index:
            print("User not in matrix!")
            return []

        user_seen_items = set(
            self.df[(self.df["user_id"] == user_id) & (self.df["rating"] == 1)]["item_id"].tolist()
        )
        print(f"User {user_id} seen items: {sorted(user_seen_items)} (total: {len(user_seen_items)})")

        similarity = cosine_similarity(self.matrix)
        sim_df = pd.DataFrame(similarity, index=self.matrix.index, columns=self.matrix.index)
        similar_users = sim_df[user_id].sort_values(ascending=False).index[1:11]
        print(f"Top similar users: {list(similar_users)}")

        # Сколько лайков у похожих пользователей?
        similar_likes = self.df[
            (self.df["user_id"].isin(similar_users)) & (self.df["rating"] == 1)
        ]
        print(f"Total likes from similar users: {len(similar_likes)}")
        if similar_likes.empty:
            print("⚠️  No likes from similar users at all!")
            # fallback
        else:
            liked_by_similar = similar_likes["item_id"]
            rec_candidates = liked_by_similar.value_counts().index.tolist()
            print(f"Rec candidates (before filtering): {rec_candidates[:10]}")

            recommendations = [item for item in rec_candidates if item not in user_seen_items][:3]
            print(f"Recommendations after filtering: {recommendations}")

            if recommendations:
                return recommendations

        # Fallback
        print("→ Falling back to random items")
        unseen = [item for item in self.item_ids if item not in user_seen_items]
        if len(unseen) >= 3:
            recommendations = random.sample(unseen, 3)
        else:
            recommendations = unseen + random.sample(self.item_ids, 3 - len(unseen))[:3]

        print(f"Final fallback recommendations: {recommendations}")
        return recommendations