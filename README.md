# Plant Disease FastAPI Service

This service provides plant disease detection using a two-stage approach:
1. **Stage 1 (YOLOv8)**: Detect if the image contains a plant
2. **Stage 2 (EfficientNet-B4)**: Classify the disease on detected plants

## Models

- **YOLOv8** (`best.pt`): Plant detection model trained to identify plants in images
- **EfficientNet-B4** (`BEST_MODEL.pth`): Disease classification model for plant pathology

## 1. Prepare artifacts
1. Ensure `BEST_MODEL.pth`, `best.pt`, and `FINAL_REPORT.json` are in the project root.
2. To regenerate or update labels, run the script below to produce `class_names.json` from the existing `FINAL_REPORT.json`:
  ```powershell
  python scripts/export_class_names.py --report "c:/Users/PC/Desktop/EfficientNetPython/FINAL_REPORT.json"
  ```
  (If you have a new dataset, replace the flag with `--dataset "D:/path/to/Dataset"`.)

## 2. Create and activate a virtual environment
```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

## 3. Install dependencies
```powershell
pip install -r requirements.txt
```

## 4. Start the API server
```powershell
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 5. API Endpoints

### Health Check
- **GET** `http://localhost:8000/health`

### Classic Prediction (v1)
- **POST** `http://localhost:8000/predict`
- Direct classification without plant detection
- Backward compatible with existing clients

### Two-Stage Prediction (v2) - Recommended
- **POST** `http://localhost:8000/predict/v2`
- Detects plant first, then classifies disease
- Returns HTTP 422 if no plant is detected
- Provides bounding box and confidence scores

## 6. Example Requests

### Using /predict/v2 (Recommended)
```powershell
curl -X POST "http://localhost:8000/predict/v2" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@example.jpg;type=image/jpeg"
```

**Success Response (200)**:
```json
{
  "detection": {
    "plant_detected": true,
    "plant_confidence": 0.856234,
    "plant_bbox": {
      "x": 120,
      "y": 80,
      "width": 450,
      "height": 380
    },
    "detection_time_ms": 45.2
  },
  "top_predictions": [
    {"label": "Tomato___Late_blight", "probability": 0.923456},
    {"label": "Tomato___Early_blight", "probability": 0.045123}
  ],
  "classification_time_ms": 234.5,
  "total_time_ms": 279.7
}
```

**No Plant Detected (422)**:
```json
{
  "detail": "No plant detected in the image"
}
```

### Using /predict (Classic)
```powershell
curl -X POST "http://localhost:8000/predict" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@example.jpg;type=image/jpeg"
```

## 7. Configuration

### YOLOv8 Confidence Threshold
Default: 0.3 (30% confidence required to detect a plant)

To change, modify the `yolo_confidence` parameter in `main.py`:
```python
predictor = DiseasePredictor(yolo_confidence=0.5)  # 50% threshold
```

## 8. Explore Swagger UI / ReDoc
- Navigate to `http://localhost:8000/docs` for Swagger UI (interactive calls, quick test uploads, schema inspection).
- Visit `http://localhost:8000/redoc` for a static documentation view.

## Troubleshooting

### "No plant detected in the image"
- The image does not contain a recognizable plant
- Try using a clearer image with better lighting
- Ensure the plant is the main subject of the image
- Lower the confidence threshold if needed

### "Plant detection not available"
- The `best.pt` model file is missing
- Check that the YOLOv8 model is in the project root
