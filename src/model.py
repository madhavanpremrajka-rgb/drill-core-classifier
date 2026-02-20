"""
model.py
Defines the EfficientNet-B0 transfer learning model using TensorFlow Hub.
"""
import tensorflow_hub as hub
import tf_keras as keras
import datetime

EFFICIENTNET_URL = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"

def create_model(C, R, L, trainable=False, model_url=EFFICIENTNET_URL):
    """
    Builds a transfer learning model using the model_url.
    Uses EfficientNet-B0 as a frozen feature extractor with a
    Dense classification head on top by default.

    :param C: Number of output classes (7 or 35)
    :param R: Image resolution (32 or 128) — defines input shape
    :param trainable: Whether to fine-tune EfficientNet weights (default False)
    :param model_url: The url of the feature extractor model to use (default - efficientnetb0 feature vector)
    :return: Tuple of (uncompiled Keras model, model name string)
    """
    
    input_shape = (R, R, 3)

    feature_extractor = hub.KerasLayer(
        model_url,
        trainable=trainable,
        input_shape=input_shape
    )

    model = keras.Sequential([
        feature_extractor,
        keras.layers.Dense(C, activation='softmax')
    ])

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = f"EfficientNetB0_C-{C}_R-{R}_L-{L}_trainable-{trainable}_{timestamp}"

    return model, model_name