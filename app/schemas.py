from typing import List, Optional
from pydantic import BaseModel


class PredictionResult(BaseModel):
    label: str
    probability: float


class PredictionResponse(BaseModel):
    top_predictions: List[PredictionResult]
    inference_time_ms: float


# V2 API Schemas with plant detection
class BoundingBox(BaseModel):
    """Bounding box coordinates in pixels [x, y, width, height]."""
    x: int
    y: int
    width: int
    height: int


class DetectionInfo(BaseModel):
    """Plant detection information from YOLOv8."""
    plant_detected: bool
    plant_confidence: float
    plant_bbox: Optional[BoundingBox]
    detection_time_ms: float


class PredictionResultV2(BaseModel):
    """Single prediction result with label and probability."""
    label: str
    probability: float


class PredictionResponseV2(BaseModel):
    """Response for /predict/v2 endpoint with two-stage detection."""
    detection: DetectionInfo
    top_predictions: List[PredictionResultV2]
    classification_time_ms: float
    total_time_ms: float
