from django.urls import path
#from .views import predict
from .views import predict_live_frame as lp
urlpatterns = [
    # path('predict/', predict, name='predict'),
     path('predict_live/', lp, name='predict_digit'),
]