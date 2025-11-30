from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import json
import time

import torch
from PIL import Image
from timm import create_model
from torchvision import transforms


class DiseasePredictor:
    """Wrap EfficientNet-B4 checkpoint loading and inference."""

    def __init__(
        self,
        model_path: Path | str = Path("BEST_MODEL.pth"),
        class_names_path: Path | str = Path("class_names.json"),
        input_size: int = 380,
        device: str | None = None,
        top_k: int = 5,
    ) -> None:
        self.model_path = Path(model_path)
        self.class_names_path = Path(class_names_path)
        self.input_size = input_size
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k = top_k

        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.class_names = self._load_class_names()
        self.top_k = min(self.top_k, len(self.class_names))
        self.model = self._load_model()
        self.model.eval()

    def _load_class_names(self) -> List[str]:
        if not self.class_names_path.exists():
            raise FileNotFoundError(
                f"Missing class names file at {self.class_names_path}. "
                "Create it with the ordered list of disease labels (one per class)."
            )
        with self.class_names_path.open("r", encoding="utf-8") as f:
            class_names = json.load(f)
        if not isinstance(class_names, list) or not class_names:
            raise ValueError("class_names.json must be a non-empty JSON list of label strings.")
        return class_names

    def _load_model(self) -> torch.nn.Module:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {self.model_path}")
        model = create_model("efficientnet_b4", pretrained=False, num_classes=len(self.class_names))
        state_dict = torch.load(self.model_path, map_location=self.device)

        # Một số checkpoint bị chèn thêm thống kê FLOPs/params (do THOP). Bỏ các khóa này để tránh lỗi.
        filtered_state = {k: v for k, v in state_dict.items() if "total_ops" not in k and "total_params" not in k}
        missing_keys = set(state_dict.keys()) - set(filtered_state.keys())
        if missing_keys:
            print(f"[DiseasePredictor] Ignored {len(missing_keys)} profiling keys when loading checkpoint.")

        model.load_state_dict(filtered_state, strict=False)
        model.to(self.device)
        return model

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> Tuple[List[str], List[float], float]:
        inputs = self._preprocess(image)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            outputs = self.model(inputs)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
        else:
            start_time = time.perf_counter()
            outputs = self.model(inputs)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

        probs = torch.softmax(outputs, dim=1)[0]
        top_probs, top_idxs = probs.topk(self.top_k)
        labels = [self.class_names[idx] for idx in top_idxs.tolist()]
        return labels, top_probs.tolist(), elapsed_ms
