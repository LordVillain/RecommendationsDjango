from django.urls import path
from .views import PredictView

urlpatterns = [
    path("predict/<int:user_id>/", PredictView.as_view(), name="predict"),
]
