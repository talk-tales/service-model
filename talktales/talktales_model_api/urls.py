from django.urls import path
from .views import TalkTalesModelApi

urlpatterns = [
    path('predict/', TalkTalesModelApi.as_view(), name='predict'),
]