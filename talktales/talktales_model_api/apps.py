from django.apps import AppConfig
import tensorflow as tf
import keras
import numpy as np
import os

class TalktalesModelApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'talktales_model_api'
    model = None

    # def ready(self):
    #   # PredictModel().init_model()
        
        
