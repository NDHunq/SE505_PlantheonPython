from __future__ import annotations

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO

from app.inference import DiseasePredictor
from app.schemas import PredictionResponse, PredictionResult

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
