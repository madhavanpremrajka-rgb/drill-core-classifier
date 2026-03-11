"""
schemas.py
Pydantic request and response models for the inference API.
"""

from pydantic import BaseModel, HttpUrl, field_validator
from typing import Optional

class PredictionResult(BaseModel):
    """Single image prediction result."""
    predicted_class: str
    confidence: float
    filename: Optional[str] = None
    image: Optional[str] = None


class URLPredictionRequest(BaseModel):
    """Request body for URL-based prediction."""
    url: HttpUrl

    @field_validator("url")
    @classmethod
    def must_be_image_url(cls, v):
        url_str = str(v).lower()
        if not any(url_str.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]):
            raise ValueError("URL must point to an image file (.jpg, .jpeg, .png, .bmp, .webp)")
        return v


class URLPredictionResponse(BaseModel):
    """Response for URL-based prediction."""
    url: str
    predicted_class: str
    confidence: float
    image: Optional[str] = None


class ModelInfo(BaseModel):
    """Information about the currently loaded model."""
    model_name: str
    num_classes: int
    resolution: int
    rwda_level: float
    class_names: list[str]