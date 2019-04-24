from django.contrib import admin
from django.urls import path, include
from . import views

app_name = 'tts'
urlpatterns = [
    path('', views.index, name='index'),
    path('synthesize/', views.synthesize, name='synthesize'),
]
