from django.contrib import admin
from .models import ModelConfig, PredictResult
# Register your models here.

admin.site.register(ModelConfig)
admin.site.register(PredictResult)
