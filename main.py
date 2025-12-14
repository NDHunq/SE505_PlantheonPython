from __future__ import annotations

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO

from app.inference import DiseasePredictor
from app.schemas import (
    PredictionResponse,
    PredictionResult,
    PredictionResponseV2,
    PredictionResultV2,
    DetectionInfo,
    BoundingBox,
)

app = FastAPI(title="Plant Disease Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

try:
    predictor = DiseasePredictor()
    predictor_error: Exception | None = None
except Exception as exc:  # pragma: no cover - initialization guard
    predictor = None
    predictor_error = exc


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(status_code=500, detail=f"Model not ready: {predictor_error}")
    if not file.content_type or "image" not in file.content_type:
        raise HTTPException(status_code=400, detail="Upload an image file (JPEG/PNG)")

    contents = await file.read()
    try:
        image = Image.open(BytesIO(contents))
    except Exception as exc:  # pragma: no cover - defensive branch
        raise HTTPException(status_code=400, detail=f"Failed to parse image: {exc}") from exc

    labels, probabilities, elapsed_ms = predictor.predict(image)
    results = [
        PredictionResult(label=label, probability=round(float(prob), 6))
        for label, prob in zip(labels, probabilities)
    ]
    return PredictionResponse(top_predictions=results, inference_time_ms=round(float(elapsed_ms), 3))


@app.post("/predict/v2", response_model=PredictionResponseV2)
async def predict_v2(file: UploadFile = File(...)):
    """
    Two-stage plant disease detection endpoint.
    
    Stage 1: Detect plant using YOLOv8
    Stage 2: Classify disease using EfficientNet-B4
    
    Returns HTTP 422 if no plant is detected in the image.
    """
    if predictor is None:
        raise HTTPException(status_code=500, detail=f"Model not ready: {predictor_error}")
    if not file.content_type or "image" not in file.content_type:
        raise HTTPException(status_code=400, detail="Upload an image file (JPEG/PNG)")

    contents = await file.read()
    try:
        image = Image.open(BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse image: {exc}") from exc

    try:
        result = predictor.predict_with_detection(image)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=f"Plant detection not available: {exc}") from exc
    except ValueError as exc:
        # No plant detected - return HTTP 422
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Convert bbox list to BoundingBox model if present
    bbox_model = None
    if result["plant_bbox"] is not None:
        x, y, w, h = result["plant_bbox"]
        bbox_model = BoundingBox(x=x, y=y, width=w, height=h)

    detection_info = DetectionInfo(
        plant_detected=result["plant_detected"],
        plant_confidence=round(result["plant_confidence"], 6),
        plant_bbox=bbox_model,
        detection_time_ms=round(result["detection_time_ms"], 3),
    )

    predictions = [
        PredictionResultV2(
            label=pred["label"],
            probability=round(pred["probability"], 6)
        )
        for pred in result["top_predictions"]
    ]

    total_time = result["detection_time_ms"] + result["classification_time_ms"]

    return PredictionResponseV2(
        detection=detection_info,
        top_predictions=predictions,
        classification_time_ms=round(result["classification_time_ms"], 3),
        total_time_ms=round(total_time, 3),
    )
