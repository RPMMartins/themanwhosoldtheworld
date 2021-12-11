from django.urls import path
from . import views


app_name = "ARIMA"
urlpatterns = [
     path("", views.index, name="index")
]
