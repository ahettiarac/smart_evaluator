from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ModelConfigSerializer, ModelPredictPostSerializer, ModelPredictResultSerializer
from .models import ModelConfig, PredictResult
from evaluator import serializers
from .context_predictor.AbstractPredictor import AbstractPredictor
from .context_predictor.PredictorRNN import PredictorRNN
from .context_predictor.similarity_predictor import SimilarityPredictor
from rest_framework.permissions import IsAuthenticated

# Create your views here.
class ModelConfigViewSet(viewsets.ModelViewSet):
    permission_classes = [IsAuthenticated]
    queryset = ModelConfig.objects.all().order_by('model_name')
    serializer_class = ModelConfigSerializer

class PredictResultConfigViewSet(viewsets.ModelViewSet):
    permission_classes = [IsAuthenticated]
    queryset = PredictResult.objects.all().order_by('id')
    serializer_class = ModelPredictResultSerializer
    http_method_names = ['get']

class SubmitResultConfigViewSet(APIView):
    permission_classes = [IsAuthenticated]

    def get(self,request,format=None):
        predictResults = PredictResult.objects.all().order_by('id')
        serializer = ModelPredictResultSerializer(predictResults,many=True)
        return Response(data=serializer.data)

    def post(self,request,format=serializers.ModelPredictPostSerializer):
       serializer = ModelPredictPostSerializer(data=request.data)
       if serializer.is_valid():
           input_serializer = ModelPredictResultSerializer(data=serializer.data)
           if input_serializer.is_valid():
            model_config = ModelConfig.objects.filter(**{'model_name':'review_abstract_model'})
            data_preprocessor = AbstractPredictor()
            data_preprocessor.max_text_length = model_config[0].max_text_length
            data_preprocessor.max_summary_length = model_config[0].max_summary_length
            user_test_seq,y_tokenizer,x_tokenizer = data_preprocessor.get_input_seq(serializer.data['user_input'],
            model_config[0].x_tokenizer_name,model_config[0].y_tokenizer_name)
            teacher_test_seq,y_tokenizer,x_tokenizer = data_preprocessor.get_input_seq(serializer.data['teacher_input'],
            model_config[0].x_tokenizer_name,model_config[0].y_tokenizer_name)
            model = PredictorRNN(data_preprocessor.max_text_length,data_preprocessor.max_summary_length,
            data_preprocessor.vocab_size,data_preprocessor.x_vocab_size)
            model.x_vocab_size = x_tokenizer.num_words+1
            model.vocab_size = y_tokenizer.num_words+1
            model.build_model()
            encoder_model,decoder_model = model.get_inference_model(data_preprocessor.max_text_length)
            user_abstract = model.decode_sequence(user_test_seq,encoder_model,decoder_model,
            y_tokenizer.word_index,data_preprocessor.max_summary_length,y_tokenizer.index_word)
            teacher_abstract = model.decode_sequence(teacher_test_seq,encoder_model,decoder_model,
            y_tokenizer.word_index,data_preprocessor.max_summary_length,y_tokenizer.index_word)
            similarity = SimilarityPredictor()
            input_serializer.validated_data['similarity_score'] = similarity.prepare_for_prediction(user_abstract,teacher_abstract)
            input_serializer.validated_data['user_input_abstract'] = user_abstract
            input_serializer.validated_data['teacher_input_abstract'] = teacher_abstract
            input_serializer.save()
            return Response(data=input_serializer.data,status=status.HTTP_201_CREATED)
           return Response(data=input_serializer.errors,status=status.HTTP_400_BAD_REQUEST) 
       return Response(data=serializer.errors,status=status.HTTP_400_BAD_REQUEST)    