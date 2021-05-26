from django.db import models

# Create your models here.
class ModelConfig(models.Model):
    id = models.AutoField(primary_key=True)
    model_name = models.CharField(max_length=50)
    weight_file_name = models.CharField(max_length=50)
    max_text_length = models.IntegerField()
    max_summary_length = models.IntegerField()
    x_tokenizer_name = models.CharField(max_length=50)
    y_tokenizer_name = models.CharField(max_length=50)

    def __str__(self):
        return self.model_name

class PredictResult(models.Model):
    id = models.AutoField(primary_key=True)
    user_input = models.TextField()
    teacher_input = models.TextField()
    user_input_abstract = models.TextField(null=True)
    teacher_input_abstract = models.TextField(null=True)
    similarity_score = models.FloatField(default=0)

    def __str__(self):
        return self.id