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
    rotation_angle: int


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
        self.rotation_fallback = env_value(env, "READMRZ_YOLO_ROTATION_FALLBACK", "true").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.rotation_fallback_min_conf = float(env_value(env, "READMRZ_YOLO_ROTATION_FALLBACK_MIN_CONF", "0.85"))

        started = time.perf_counter()
        self.model = YOLO(str(model_path))
        self.load_ms = int((time.perf_counter() - started) * 1000)

    def detect(self, image: np.ndarray) -> dict[str, Any]:
        height, width = image.shape[:2]
        started = time.perf_counter()
        detections, attempts = self._detect_with_optional_rotation_fallback(image, width, height)
        detector_ms = int((time.perf_counter() - started) * 1000)

        detections.sort(key=lambda item: item.confidence, reverse=True)
        best_detection = detections[0] if detections else None
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
            "rotation_fallback_enabled": self.rotation_fallback,
            "rotation_fallback_min_conf": self.rotation_fallback_min_conf,
            "fallback_used": bool(best_detection and best_detection.rotation_angle != 0),
            "selected_rotation_angle": best_detection.rotation_angle if best_detection else 0,
            "attempts": attempts,
            "boxes": [detection.__dict__ for detection in detections],
            "best_box": best_detection.__dict__ if best_detection else None,
        }

    def _detect_with_optional_rotation_fallback(
        self,
        image: np.ndarray,
        original_width: int,
        original_height: int,
    ) -> tuple[list[YoloMrzDetection], list[dict[str, Any]]]:
        detections, first_attempt = self._detect_once(
            image,
            rotation_angle=0,
            original_width=original_width,
            original_height=original_height,
        )
        attempts = [first_attempt]
        best_detections = detections
        best_confidence = detections[0].confidence if detections else 0.0
        should_try_rotations = (
            self.rotation_fallback
            and (not detections or best_confidence < self.rotation_fallback_min_conf)
        )
        if not should_try_rotations:
            return detections, attempts

        for rotation_angle, rotated_image in rotated_candidates(image):
            rotated_detections, attempt = self._detect_once(
                rotated_image,
                rotation_angle=rotation_angle,
                original_width=original_width,
                original_height=original_height,
            )
            attempts.append(attempt)
            rotated_confidence = rotated_detections[0].confidence if rotated_detections else 0.0
            if rotated_confidence > best_confidence:
                best_confidence = rotated_confidence
                best_detections = rotated_detections

        return best_detections, attempts

    def _detect_once(
        self,
        image: np.ndarray,
        *,
        rotation_angle: int,
        original_width: int,
        original_height: int,
    ) -> tuple[list[YoloMrzDetection], dict[str, Any]]:
        rotated_height, rotated_width = image.shape[:2]
        started = time.perf_counter()
        results = self.model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        detections: list[YoloMrzDetection] = []
        if results:
            boxes = getattr(results[0], "boxes", None)
            if boxes is not None:
                xyxy_values = boxes.xyxy.cpu().numpy().tolist()
                conf_values = boxes.conf.cpu().numpy().tolist()
                cls_values = boxes.cls.cpu().numpy().tolist()
                names = getattr(results[0], "names", {}) or {}
                for xyxy, confidence, class_id in zip(xyxy_values, conf_values, cls_values, strict=False):
                    bbox = map_rotated_bbox_to_original(
                        [float(value) for value in xyxy],
                        rotation_angle=rotation_angle,
                        original_width=original_width,
                        original_height=original_height,
                    )
                    cls_int = int(class_id)
                    detections.append(
                        YoloMrzDetection(
                            bbox_xyxy=bbox,
                            bbox_percent=bbox_percent(bbox, original_width, original_height),
                            confidence=round(float(confidence), 6),
                            class_id=cls_int,
                            class_name=str(names.get(cls_int, "mrz")),
                            rotation_angle=rotation_angle,
                        )
                    )

        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections, {
            "rotation_angle": rotation_angle,
            "detector_ms": elapsed_ms,
            "boxes": len(detections),
            "image_width": rotated_width,
            "image_height": rotated_height,
            "best_confidence": detections[0].confidence if detections else None,
        }


def bbox_percent(bbox_xyxy: list[float], width: int, height: int) -> dict[str, float]:
    x_min, y_min, x_max, y_max = [float(value) for value in bbox_xyxy]
    return {
        "left": round((x_min / width) * 100, 4),
        "top": round((y_min / height) * 100, 4),
        "width": round(((x_max - x_min) / width) * 100, 4),
        "height": round(((y_max - y_min) / height) * 100, 4),
    }


def rotated_candidates(image: np.ndarray) -> list[tuple[int, np.ndarray]]:
    return [
        (90, np.ascontiguousarray(np.rot90(image, 1))),
        (180, np.ascontiguousarray(np.rot90(image, 2))),
        (270, np.ascontiguousarray(np.rot90(image, 3))),
    ]


def map_rotated_bbox_to_original(
    bbox_xyxy: list[float],
    *,
    rotation_angle: int,
    original_width: int,
    original_height: int,
) -> list[float]:
    x_min, y_min, x_max, y_max = bbox_xyxy
    corners = [
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max),
    ]
    mapped = [
        map_rotated_point_to_original(
            x,
            y,
            rotation_angle=rotation_angle,
            original_width=original_width,
            original_height=original_height,
        )
        for x, y in corners
    ]
    xs = [point[0] for point in mapped]
    ys = [point[1] for point in mapped]
    return [
        round(clamp(min(xs), 0, original_width), 2),
        round(clamp(min(ys), 0, original_height), 2),
        round(clamp(max(xs), 0, original_width), 2),
        round(clamp(max(ys), 0, original_height), 2),
    ]


def map_rotated_point_to_original(
    x: float,
    y: float,
    *,
    rotation_angle: int,
    original_width: int,
    original_height: int,
) -> tuple[float, float]:
    if rotation_angle == 0:
        return x, y
    if rotation_angle == 90:
        return original_width - y, x
    if rotation_angle == 180:
        return original_width - x, original_height - y
    if rotation_angle == 270:
        return y, original_height - x
    raise ValueError(f"Unsupported rotation_angle: {rotation_angle}")


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
