from typing import List
from pydantic import BaseModel


class PredictionResult(BaseModel):
    label: str
    probability: float


class PredictionResponse(BaseModel):
    top_predictions: List[PredictionResult]
    inference_time_ms: float
