from django.urls import path
from . import views

app_name= "CV"
urlpatterns = [
     path("", views.index, name="index")
]
