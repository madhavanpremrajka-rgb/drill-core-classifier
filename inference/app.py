"""
app.py
FastAPI inference server for the drill-core classifier.

Endpoints:
    GET  /                      - health check
    GET  /model-info            - metadata about the loaded model
    POST /predict               - single image upload
    POST /predict/url           - predict from image URL
"""

import os
import io
import json
import base64
import httpx
import numpy as np
import tensorflow as tf
import tf_keras as keras
import tensorflow_hub as hub

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from schemas import (
    PredictionResult,
    URLPredictionRequest,
    URLPredictionResponse,
    ModelInfo,
)

#==============================================#
#==============================================#
MODELS_DIR   = os.environ.get("MODELS_DIR", "models")
METRICS_PATH = os.environ.get("METRICS_PATH", "Artifacts/metrics.json")
CLASS_NAMES_PATH = os.environ.get("CLASS_NAMES_PATH", "Artifacts/class_names.json")

BEST_C = 35
BEST_R = 128
BEST_L = 0.2

BATCH_SIZE = 32

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
#==============================================#
#==============================================#


#==============================================#
#==============================================#
def _load_class_names() -> list[str]:
    """
    Loads the class names for given dataset configuration
    """
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
    return class_names

def _get_model_name(C: int, R: int, L: float) -> str:
    """
    Gets the name of the model for given configuration from metrics.json
    """
    with open(METRICS_PATH, "r") as f:
        results = json.load(f)
    for entry in results:
        if entry["C"] == C and entry["R"] == R and entry["L"] == L:
            return entry["model_name"]
    raise ValueError(f"No model found in metrics.json for C={C}, R={R}, L={L}")

def _preprocess(image_bytes: bytes, target_size: int) -> tf.Tensor:
    """
    Decodes the raw image into a (1, R, R, 3) tensor for prediction
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((target_size, target_size))
    arr = np.array(img, dtype = np.float32)/255
    return tf.expand_dims(arr, axis=0)

def _predict_tensor(tensor: tf.Tensor) -> tuple[str, float]:
    """
    Runs a single processed image through the model
    """
    pred = model.predict(tensor, verbose = 0)[0]
    idx = int(np.argmax(pred))
    return class_names[idx], float(round(pred[idx], 4))

def _is_valid_image(file: UploadFile) -> bool:
    if file.content_type.startswith("image/"):
        return True
    ext = os.path.splitext(file.filename or "")[1].lower()
    return ext in ALLOWED_EXTENSIONS
#==============================================#
#==============================================#

#==============================================#
#==============================================#
model: keras.Model = None
class_names: list[str] = []
model_meta: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, class_names, model_meta

    model_name = _get_model_name(BEST_C, BEST_R, BEST_L)
    model_path = os.path.join(MODELS_DIR, model_name, "model.keras")
    class_names = _load_class_names()

    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(
        model_path,
        custom_objects={'KerasLayer': hub.KerasLayer}
    )
    
    model_meta = {
        "model_name"    :   model_name,
        "num_classes"   :   BEST_C,
        "resolution"    :   BEST_R,
        "rwda_level"    :   BEST_L,
        "class_names"   :   class_names
    }

    print(f"Model loaded!")
    yield
#==============================================#
#==============================================#

#==============================================#
#==============================================#
app = FastAPI(
    title = 'Drill Core Classifier',
    description = 'EfficientNetB0 lithology classifier for drill core images',
    version = '1.0.0',
    lifespan = lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
#==============================================#
#==============================================#
@app.get("/", tags = ["Health"])
def health_check():
    return {"status":"ok", "message":"Drill Core Classifier is running!"}

@app.get("/model-info", response_model = ModelInfo, tags = ["Meta"])
def get_model_info():
    return ModelInfo(**model_meta)

@app.post("/predict", response_model = PredictionResult, tags = ["Inference"])
async def predict_single(file: UploadFile = File(...)):
    if not _is_valid_image(file):
        raise HTTPException(status_code=415, detail="File must be an image.")

    image_bytes = await file.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    try:
        tensor = _preprocess(image_bytes=image_bytes, target_size=BEST_R)
    except Exception as e:
        raise HTTPException(status_code = 400, detail = f"Could not process image: {e}")
    
    predicted_class, confidence = _predict_tensor(tensor)

    return PredictionResult(
        predicted_class = predicted_class,
        confidence = confidence,
        filename = file.filename,
        image = image_b64
    )

@app.post("/predict/url", response_model=URLPredictionResponse, tags=["Inference"])
async def predict_from_url(request: URLPredictionRequest):
    url_str = str(request.url)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url_str)
            response.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image from URL: {e}")

    content_type = response.headers.get("content-type", "")
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="URL does not point to a valid image.")

    image_b64 = base64.b64encode(response.content).decode("utf-8")

    try:
        tensor = _preprocess(response.content, BEST_R)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")

    predicted_class, confidence = _predict_tensor(tensor)
    return URLPredictionResponse(
        url = url_str,
        predicted_class = predicted_class,
        confidence = confidence,
        image = image_b64
    )