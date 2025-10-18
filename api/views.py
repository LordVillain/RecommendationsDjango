from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .recommender.als_recommender import ALSRecommender

model = ALSRecommender("dataMatrix.csv")

class PredictView(APIView):
    def get(self, request, user_id: int):
        recommendations = model.recommend(user_id)
        if not recommendations:
            return Response(
                {"error": f"No recommendations found for user {user_id}"},
                status=status.HTTP_404_NOT_FOUND
            )
        return Response({"user_id": user_id, "product_ids": recommendations})
