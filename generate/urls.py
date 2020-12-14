from django.urls import path,include
from . import views

app_name = 'generate'
urlpatterns = [
    path('',views.home,name='gen'),
    path('record',views.record,name='record'),
    path('trans',views.translation,name='translate'),
    path('textgen',views.text_gen,name='text_gen'),
    #path('home_1',views.home_1,name='home_1'),
    path('record_1',views.record_1,name='record_1'),
    path('final',views.final,name='final'),
    path('sentimental',views.sentimental,name='sentimental'),
    
]