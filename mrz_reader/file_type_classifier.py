from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
import time
from typing import Any

import cv2
import numpy as np

from .env_config import env_value, read_env_file


DEFAULT_CLASS_NAMES = ["passport", "face", "EVISA_RESULT", "VOA_RESULT"]


class FileTypeClassifier:
    def __init__(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required. Run: python -m pip install onnxruntime") from exc

        env = read_env_file()
        self.model_path = resolve_model_path(env)
        self.metadata_path = resolve_metadata_path(env, self.model_path)
        self.metadata = read_metadata(self.metadata_path)
        self.class_names = read_class_names(self.metadata)
        self.img_size = int(self.metadata.get("img_size") or env_value(env, "READMRZ_FILE_DETECT_IMAGE_SIZE", "224"))
        self.min_confidence = float(env_value(env, "READMRZ_FILE_DETECT_MIN_CONF", "0.0"))
        self.cpu_threads = max(1, int(env_value(env, "READMRZ_FILE_DETECT_CPU_THREADS", "4")))
        self.mean = np.asarray(
            self.metadata.get("normalize_mean") or [0.485, 0.456, 0.406],
            dtype=np.float32,
        ).reshape(1, 1, 3)
        self.std = np.asarray(
            self.metadata.get("normalize_std") or [0.229, 0.224, 0.225],
            dtype=np.float32,
        ).reshape(1, 1, 3)

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = self.cpu_threads
        session_options.inter_op_num_threads = 1
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        started = time.perf_counter()
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.load_ms = int((time.perf_counter() - started) * 1000)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self._predict_lock = Lock()

    def warmup(self) -> int:
        image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        started = time.perf_counter()
        self.predict(image)
        return int((time.perf_counter() - started) * 1000)

    def predict(self, image: np.ndarray) -> dict[str, Any]:
        if image is None or image.size == 0:
            raise ValueError("image is empty")

        total_started = time.perf_counter()
        height, width = image.shape[:2]
        preprocess_started = time.perf_counter()
        batch = preprocess_image(image, self.img_size, self.mean, self.std)
        preprocess_ms = int((time.perf_counter() - preprocess_started) * 1000)

        inference_started = time.perf_counter()
        with self._predict_lock:
            outputs = self.session.run([self.output_name], {self.input_name: batch})
        inference_ms = int((time.perf_counter() - inference_started) * 1000)

        logits = np.asarray(outputs[0], dtype=np.float32)
        if logits.ndim == 2:
            logits = logits[0]
        probabilities = softmax(logits.reshape(-1))
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])
        label = self.class_names[best_index] if best_index < len(self.class_names) else str(best_index)
        ranked = [
            {
                "labelId": index,
                "label": self.class_names[index] if index < len(self.class_names) else str(index),
                "confidence": round(float(score), 6),
            }
            for index, score in enumerate(probabilities.tolist())
        ]
        ranked.sort(key=lambda item: item["confidence"], reverse=True)

        return {
            "ok": True,
            "found": True,
            "label": label,
            "labelId": best_index,
            "confidence": round(confidence, 6),
            "isConfidenceInformation": confidence >= self.min_confidence,
            "probabilities": ranked,
            "probabilitiesByLabel": {
                item["label"]: item["confidence"]
                for item in sorted(ranked, key=lambda value: int(value["labelId"]))
            },
            "image": {
                "width": int(width),
                "height": int(height),
            },
            "processing": {
                "preprocessMs": preprocess_ms,
                "inferenceMs": inference_ms,
                "totalMs": int((time.perf_counter() - total_started) * 1000),
                "modelLoadMs": self.load_ms,
            },
            "model": self.summary(),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "engine": "onnxruntime-cpu",
            "modelPath": str(self.model_path),
            "metadataPath": str(self.metadata_path) if self.metadata_path else None,
            "modelName": self.metadata.get("model_name") or "mobilenet_v3_small",
            "classNames": self.class_names,
            "imgSize": self.img_size,
            "minConfidence": self.min_confidence,
            "cpuThreads": self.cpu_threads,
            "inputName": self.input_name,
            "outputName": self.output_name,
        }


def resolve_model_path(env: dict[str, str]) -> Path:
    configured = env_value(env, "READMRZ_FILE_DETECT_MODEL_PATH", "").strip()
    if not configured:
        raise ValueError("READMRZ_FILE_DETECT_MODEL_PATH is required")

    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    path = path.resolve()
    if path.is_dir():
        path = path / "model.onnx"
    if not path.is_file():
        raise FileNotFoundError(f"READMRZ_FILE_DETECT_MODEL_PATH does not exist: {path}")
    return path


def resolve_metadata_path(env: dict[str, str], model_path: Path) -> Path | None:
    configured = env_value(env, "READMRZ_FILE_DETECT_METADATA_PATH", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"READMRZ_FILE_DETECT_METADATA_PATH does not exist: {path}")
        return path

    candidate = model_path.parent / "metadata.json"
    return candidate if candidate.is_file() else None


def read_metadata(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_class_names(metadata: dict[str, Any]) -> list[str]:
    values = metadata.get("class_names")
    if isinstance(values, list) and values:
        return [str(value) for value in values]
    labels = metadata.get("labels")
    if isinstance(labels, dict) and labels:
        ordered = sorted(labels.items(), key=lambda item: int(item[1]))
        return [str(name) for name, _ in ordered]
    return DEFAULT_CLASS_NAMES


def preprocess_image(
    image: np.ndarray,
    img_size: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - mean) / std
    chw = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(chw, axis=0).astype(np.float32)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / np.sum(exp)
