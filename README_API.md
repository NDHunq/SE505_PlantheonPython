# Plant Disease FastAPI Service

This guide shows how to turn the trained EfficientNet-B4 checkpoint into a FastAPI inference service.

## 1. Prepare artifacts
1. `BEST_MODEL.pth` và `FINAL_REPORT.json` đã có trong thư mục gốc dự án.
2. Nếu cần tái tạo hoặc cập nhật nhãn, chạy script sau để sinh `class_names.json` từ `FINAL_REPORT.json` sẵn có:
  ```powershell
  python scripts/export_class_names.py --report "c:/Users/PC/Desktop/EfficientNetPython/FINAL_REPORT.json"
  ```
  (Trường hợp bạn có dataset mới, thay bằng `--dataset "D:/path/to/Dataset"`.)

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
