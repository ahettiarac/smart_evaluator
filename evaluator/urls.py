from django.urls import include,path
from rest_framework import routers, urlpatterns
from . import views

router = routers.DefaultRouter()
router.register(r'model_config',views.ModelConfigViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('predict/',views.SubmitResultConfigViewSet.as_view()),
    path('api-auth',include('rest_framework.urls',namespace='rest_framework'))
]