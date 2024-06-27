from django.urls import path
from . import views
from django.urls import reverse

from django.contrib import admin

urlpatterns = [
     path('delete_dataset/<int:dataset_id>/', views.delete_dataset, name='delete_dataset'),


    path('upload/', views.upload_dataset, name='upload_dataset'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('dataset/<int:dataset_id>/preprocess/', views.data_preprocessing, name='data_preprocessing'),
    path('dataset/<int:dataset_id>/', views.dataset_detail, name='dataset_detail'),
    path('dataset/<int:dataset_id>/graphs/', views.dataset_graph, name='dataset_graph'),

]
