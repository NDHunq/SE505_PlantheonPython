from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional
import time

import torch
from PIL import Image
from ultralytics import YOLO
import numpy as np


class PlantDetector:
    """Wrap YOLOv8 model for plant detection."""

    def __init__(
        self,
        model_path: Path | str = Path("best.pt"),
        confidence_threshold: float = 0.3,
        device: str | None = None,
    ) -> None:
        """
        Initialize PlantDetector with YOLOv8 model.

        Args:
            model_path: Path to YOLOv8 model weights (.pt file)
            confidence_threshold: Minimum confidence score to accept detection (0-1)
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLOv8 model not found at {self.model_path}")

        # Load YOLOv8 model
        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)

    def detect(
        self, image: Image.Image
    ) -> Tuple[bool, float, Optional[list], Optional[Image.Image], float]:
        """
        Detect plant in the image.

        Args:
            image: PIL Image to detect plant in

        Returns:
            Tuple of:
            - has_plant: Whether a plant was detected above confidence threshold
            - confidence: Confidence score of best detection (0 if no detection)
            - bbox: Bounding box [x, y, width, height] in pixels (None if no detection)
            - cropped_image: Cropped image of detected plant (None if no detection)
            - inference_time_ms: Inference time in milliseconds
        """
        # Convert PIL Image to format YOLO expects
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Run inference with timing
        start_time = time.perf_counter()
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            verbose=False,
            device=self.device,
            imgsz=640,  # Resize to 640x640 for faster inference
            half=False,  # Use FP16 if GPU available (set to True for GPU)
        )
        inference_time_ms = (time.perf_counter() - start_time) * 1000

        # Extract best detection
        has_plant, confidence, bbox, cropped_image = self._extract_best_detection(
            results[0], image
        )

        return has_plant, confidence, bbox, cropped_image, inference_time_ms

    def _extract_best_detection(
        self, result, original_image: Image.Image
    ) -> Tuple[bool, float, Optional[list], Optional[Image.Image]]:
        """
        Extract the best detection from YOLO results.

        Args:
            result: YOLO result object
            original_image: Original PIL Image

        Returns:
            Tuple of (has_plant, confidence, bbox, cropped_image)
        """
        # Check if any detections were made
        if result.boxes is None or len(result.boxes) == 0:
            return False, 0.0, None, None

        # Get detection with highest confidence
        boxes = result.boxes
        confidences = boxes.conf.cpu().numpy()
        best_idx = np.argmax(confidences)
        best_confidence = float(confidences[best_idx])

        # Get bounding box in xyxy format and convert to xywh
        xyxy = boxes.xyxy[best_idx].cpu().numpy()
        x1, y1, x2, y2 = map(int, xyxy)
        
        # Convert to [x, y, width, height] format (original bbox for response)
        bbox = [x1, y1, x2 - x1, y2 - y1]

        # Add padding to preserve context for classification
        # Expand bbox by 20% on each side to include surrounding area
        img_width, img_height = original_image.size
        width = x2 - x1
        height = y2 - y1
        
        padding_x = int(width * 0.2)
        padding_y = int(height * 0.2)
        
        # Calculate expanded bbox with bounds checking
        x1_padded = max(0, x1 - padding_x)
        y1_padded = max(0, y1 - padding_y)
        x2_padded = min(img_width, x2 + padding_x)
        y2_padded = min(img_height, y2 + padding_y)
        
        # Crop with padding
        cropped_image = original_image.crop((x1_padded, y1_padded, x2_padded, y2_padded))
        
        # Add additional padding to make it square (better for EfficientNet)
        # This helps maintain aspect ratio similar to training data
        crop_width, crop_height = cropped_image.size
        if crop_width != crop_height:
            # Make square by adding padding to the shorter side
            max_dim = max(crop_width, crop_height)
            # Create a new square image with white/gray background
            square_image = Image.new('RGB', (max_dim, max_dim), (240, 240, 240))
            # Paste cropped image in the center
            paste_x = (max_dim - crop_width) // 2
            paste_y = (max_dim - crop_height) // 2
            square_image.paste(cropped_image, (paste_x, paste_y))
            cropped_image = square_image

        return True, best_confidence, bbox, cropped_image
