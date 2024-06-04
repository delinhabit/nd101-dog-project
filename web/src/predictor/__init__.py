import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.applications import resnet50

from . import models
from .breeds import BREED_NAMES

graph = tf.get_default_graph()


def predict_breed(image_file):
    features = extract_features(get_tensor_from_image(image_file))
    with graph.as_default():
        prediction = np.argmax(models.breeds.predict(features))
    return BREED_NAMES[prediction]


def get_tensor_from_image(image_file):
    """
    Load an image file into a (1, 224, 224, 3) tensor.
    """
    img = image.load_img(image_file, target_size=(224, 224))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)


def extract_features(tensor):
    """
    The input features to our breeds model are the output features of the
    convolutional layers of the ResNet50 model.
    """
    input_features = resnet50.preprocess_input(tensor)
    with graph.as_default():
        return models.resnet50_cnn.predict(input_features)
