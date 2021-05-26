from django.db.models import fields
from rest_framework import serializers
from .models import ModelConfig, PredictResult

class ModelConfigSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = ModelConfig
        fields = ('model_name','weight_file_name','max_text_length','max_summary_length','x_tokenizer_name','y_tokenizer_name')


class ModelPredictResultSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = PredictResult
        fields = ('id','user_input','teacher_input','user_input_abstract','teacher_input_abstract','similarity_score')

class ModelPredictPostSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = PredictResult
        fields = ('user_input','teacher_input')        