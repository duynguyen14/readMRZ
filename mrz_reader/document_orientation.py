from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
import time
from typing import Any

import numpy as np

from .env_config import env_value, read_env_file


# Paddle 3.x on Windows can otherwise enter an unstable oneDNN/PIR path for
# the document-orientation classifier.
os.environ.setdefault("FLAGS_use_onednn", "0")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")


def env_bool(env: dict[str, str], key: str, default: bool) -> bool:
    raw_value = env_value(env, key, "true" if default else "false").strip().lower()
    return raw_value in {"1", "true", "yes", "on"}


class PaddleDocumentOrientation:
    def __init__(self) -> None:
        env = read_env_file()
        self.enabled = env_bool(env, "READMRZ_YOLO_PADDLE_ORIENTATION", True)
        self.min_confidence = float(
            env_value(env, "READMRZ_YOLO_PADDLE_ORIENTATION_MIN_CONF", "0.50")
        )
        self.device = env_value(env, "PADDLE_OCR_DEVICE", "cpu")
        self.model_path: Path | None = None
        self.model: Any | None = None
        self.load_ms = 0
        self._predict_lock = Lock()

        if not self.enabled:
            return

        self.model_path = self._resolve_model_path(env)

        try:
            from paddleocr import DocImgOrientationClassification
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is required for YOLO upload orientation. Install paddleocr/paddlepaddle "
                "or set READMRZ_YOLO_PADDLE_ORIENTATION=false."
            ) from exc

        kwargs: dict[str, Any] = {"device": self.device}
        if self.model_path is not None:
            kwargs["model_dir"] = str(self.model_path)

        started = time.perf_counter()
        self.model = DocImgOrientationClassification(**kwargs)
        self.load_ms = int((time.perf_counter() - started) * 1000)

    def warmup(self) -> int:
        if not self.enabled or self.model is None:
            return 0
        image = np.full((224, 224, 3), 255, dtype=np.uint8)
        _, payload = self.normalize(image)
        return int(payload["latency_ms"])

    def normalize(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        if not self.enabled or self.model is None:
            return image, {
                "enabled": False,
                "predicted_angle": 0,
                "applied_angle": 0,
                "confidence": None,
                "min_confidence": self.min_confidence,
                "latency_ms": 0,
                "model_load_ms": self.load_ms,
                "device": self.device,
            }

        started = time.perf_counter()
        with self._predict_lock:
            results = self.model.predict(image)
        latency_ms = int((time.perf_counter() - started) * 1000)

        predicted_angle = 0
        confidence = 0.0
        if results:
            result = results[0]
            labels = result.get("label_names", [])
            scores = np.asarray(result.get("scores", []), dtype=np.float32).reshape(-1)
            if labels:
                predicted_angle = normalize_right_angle(labels[0])
            if scores.size:
                confidence = float(scores[0])

        applied_angle = predicted_angle if confidence >= self.min_confidence else 0
        normalized_image = rotate_counter_clockwise(image, applied_angle)
        height, width = normalized_image.shape[:2]
        return normalized_image, {
            "enabled": True,
            "predicted_angle": predicted_angle,
            "applied_angle": applied_angle,
            "confidence": round(confidence, 6),
            "min_confidence": self.min_confidence,
            "latency_ms": latency_ms,
            "model_load_ms": self.load_ms,
            "device": self.device,
            "image_width": width,
            "image_height": height,
        }

    @staticmethod
    def _resolve_model_path(env: dict[str, str]) -> Path | None:
        raw_path = env_value(env, "PADDLE_DOC_ORIENTATION_MODEL_DIR").strip()
        if not raw_path:
            return None
        model_path = Path(raw_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"PADDLE_DOC_ORIENTATION_MODEL_DIR does not exist: {model_path}")
        return model_path


def normalize_right_angle(value: Any) -> int:
    try:
        angle = int(round(float(value))) % 360
    except (TypeError, ValueError):
        return 0
    return angle if angle in {0, 90, 180, 270} else 0


def rotate_counter_clockwise(image: np.ndarray, angle: int) -> np.ndarray:
    normalized_angle = normalize_right_angle(angle)
    if normalized_angle == 0:
        return image
    return np.ascontiguousarray(np.rot90(image, normalized_angle // 90))
