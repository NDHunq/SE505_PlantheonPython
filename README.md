# Plant Disease FastAPI Service

This guide shows how to turn the trained EfficientNet-B4 checkpoint into a FastAPI inference service.

## 1. Prepare artifacts
1. Ensure `BEST_MODEL.pth` and `FINAL_REPORT.json` are in the project root.
2. To regenerate or update labels, run the script below to produce `class_names.json` from the existing `FINAL_REPORT.json`:
  ```powershell
  python scripts/export_class_names.py --report "c:/Users/PC/Desktop/EfficientNetPython/FINAL_REPORT.json"
  ```
  (If you have a new dataset, replace the flag with `--dataset "D:/path/to/Dataset"`.)

## 2. Create and activate a virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. Install dependencies
```powershell
pip install -r requirements.txt
```

## 4. Start the API server
```powershell
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 5. Call the endpoints
- Health check: `GET http://localhost:8000/health`
- Prediction: `POST http://localhost:8000/predict` with `multipart/form-data` containing an `image` file field.

### Example request via `curl`
```powershell
curl -X POST "http://localhost:8000/predict" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@example.jpg;type=image/jpeg"
```

The response provides the top-k classes with probabilities and the measured inference time.

## 6. Explore Swagger UI / ReDoc
- Navigate to `http://localhost:8000/docs` for Swagger UI (interactive calls, quick test uploads, schema inspection).
- Visit `http://localhost:8000/redoc` for a static documentation view.
- Open `README_API.md` anytime for the full offline walkthrough.
