from tensorflow.python.keras.applications.resnet50 import ResNet50, preprocess_input
from config import get_config

config = get_config()


def get_model(pooling='avg', input_shape=None):
    base_model = ResNet50(weights='imagenet', include_top=False, pooling=pooling, input_shape=input_shape)

    return base_model, preprocess_input