from django.db import models
from django.utils import timezone

class Dataset(models.Model):
    name = models.CharField(max_length=100)
    file = models.FileField(upload_to='datasets/')

class MLModel(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
