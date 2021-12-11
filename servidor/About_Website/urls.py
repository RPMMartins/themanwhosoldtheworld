from django.urls import path
from . import views


app_name = "About_Website"
urlpatterns = [
     path("", views.index, name="index")
]
