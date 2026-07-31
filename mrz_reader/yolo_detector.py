from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np

from .env_config import env_value, read_env_file


@dataclass
class YoloMrzDetection:
    bbox_xyxy: list[float]
    bbox_percent: dict[str, float]
    confidence: float
    class_id: int
    class_name: str


class YoloMrzDetector:
    def __init__(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is not installed. Run: pip install ultralytics") from exc

        env = read_env_file()
        model_path = Path(env_value(env, "READMRZ_YOLO_MODEL_PATH", "models/mrz_yolo11n_best.pt")).expanduser()
        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parents[1] / model_path
        model_path = model_path.resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"READMRZ_YOLO_MODEL_PATH does not exist: {model_path}")

        self.model_path = model_path
        self.imgsz = int(env_value(env, "READMRZ_YOLO_IMGSZ", "640"))
        self.conf = float(env_value(env, "READMRZ_YOLO_CONF", "0.25"))
        self.device = env_value(env, "READMRZ_YOLO_DEVICE", "cpu")

        started = time.perf_counter()
        self.model = YOLO(str(model_path))
        self.load_ms = int((time.perf_counter() - started) * 1000)

    def detect(self, image: np.ndarray) -> dict[str, Any]:
        height, width = image.shape[:2]
        started = time.perf_counter()
        results = self.model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        detector_ms = int((time.perf_counter() - started) * 1000)

        detections: list[YoloMrzDetection] = []
        if results:
            boxes = getattr(results[0], "boxes", None)
            if boxes is not None:
                xyxy_values = boxes.xyxy.cpu().numpy().tolist()
                conf_values = boxes.conf.cpu().numpy().tolist()
                cls_values = boxes.cls.cpu().numpy().tolist()
                names = getattr(results[0], "names", {}) or {}
                for xyxy, confidence, class_id in zip(xyxy_values, conf_values, cls_values, strict=False):
                    bbox = [round(float(value), 2) for value in xyxy]
                    cls_int = int(class_id)
                    detections.append(
                        YoloMrzDetection(
                            bbox_xyxy=bbox,
                            bbox_percent=bbox_percent(bbox, width, height),
                            confidence=round(float(confidence), 6),
                            class_id=cls_int,
                            class_name=str(names.get(cls_int, "mrz")),
                        )
                    )

        detections.sort(key=lambda item: item.confidence, reverse=True)
        return {
            "found": bool(detections),
            "image_width": width,
            "image_height": height,
            "detector_ms": detector_ms,
            "model_load_ms": self.load_ms,
            "model_path": str(self.model_path),
            "imgsz": self.imgsz,
            "conf": self.conf,
            "device": self.device,
            "boxes": [detection.__dict__ for detection in detections],
            "best_box": detections[0].__dict__ if detections else None,
        }


def bbox_percent(bbox_xyxy: list[float], width: int, height: int) -> dict[str, float]:
    x_min, y_min, x_max, y_max = [float(value) for value in bbox_xyxy]
    return {
        "left": round((x_min / width) * 100, 4),
        "top": round((y_min / height) * 100, 4),
        "width": round(((x_max - x_min) / width) * 100, 4),
        "height": round(((y_max - y_min) / height) * 100, 4),
    }
