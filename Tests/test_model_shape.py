import numpy as np
import os
import tf_keras as keras
import tensorflow_hub as hub
import pytest

BEST_C = 35
BEST_R = 128
BEST_L = 0.2
MODELS_DIR = "models"

@pytest.fixture
def model():
    return keras.models.load_model(
                os.path.join(MODELS_DIR, f"EfficientNetB0_C-35_R-128_L-0.2_trainable-True_20260220-092325", 'model.keras'),
                custom_objects={'KerasLayer': hub.KerasLayer}
            )

def test_model_loads(model):
    assert model is not None

def test_model_output_shape(model):
    arr = np.random.rand(1, 128, 128, 3)
    preds = model.predict(arr)
    assert preds.shape == (1, 35)