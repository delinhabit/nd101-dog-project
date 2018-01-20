import os

from keras.models import load_model
from keras.applications.resnet50 import ResNet50


# Load our pretrained breeds prediction model
breeds = load_model(os.environ['KERAS_MODEL_PATH'])

# Load the ResNet50 model without the fully connected layers
resnet50_cnn = ResNet50(weights='imagenet', include_top=False)
