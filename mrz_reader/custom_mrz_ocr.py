from __future__ import annotations

from pathlib import Path
from threading import Lock
import time
from typing import Any

import cv2
import numpy as np
import yaml

from .document_orientation import env_bool
from .env_config import env_value, read_env_file
from .mrz import normalize_mrz_text


class CustomMrzCtcRecognizer:
    def __init__(self) -> None:
        env = read_env_file()
        self.enabled = env_bool(env, "READMRZ_CUSTOM_OCR_ENABLED", False)
        self.device = env_value(env, "READMRZ_CUSTOM_OCR_DEVICE", "cpu")
        self.batch_size = max(1, int(env_value(env, "READMRZ_CUSTOM_OCR_BATCH_SIZE", "2")))
        self.min_confidence = float(env_value(env, "READMRZ_CUSTOM_OCR_MIN_CONF", "0.0"))
        self.model_dir: Path | None = None
        self.dict_path: Path | None = None
        self.model_name: str | None = None
        self.model: Any | None = None
        self.input_name = ""
        self.output_name = ""
        self.image_shape = (3, 48, 640)
        self.characters: list[str] = []
        self.load_ms = 0
        self._predict_lock = Lock()

        if not self.enabled:
            return

        self.model_dir = resolve_required_path(env, "READMRZ_CUSTOM_OCR_MODEL_DIR", directory=True)
        configured_dict = env_value(
            env,
            "READMRZ_CUSTOM_OCR_DICT_PATH",
            str(self.model_dir / "mrz_dict.txt"),
        )
        self.dict_path = Path(configured_dict).expanduser().resolve()
        if not self.dict_path.is_file():
            raise FileNotFoundError(f"READMRZ_CUSTOM_OCR_DICT_PATH does not exist: {self.dict_path}")

        for file_name in ("inference.json", "inference.pdiparams", "inference.yml"):
            model_file = self.model_dir / file_name
            if not model_file.is_file():
                raise FileNotFoundError(f"Custom MRZ OCR model file does not exist: {model_file}")

        inference_config = yaml.safe_load(
            (self.model_dir / "inference.yml").read_text(encoding="utf-8")
        )
        self.model_name = str((inference_config or {}).get("Global", {}).get("model_name") or "").strip()
        self.image_shape = read_image_shape(inference_config)
        self.characters = read_character_dictionary(self.dict_path)

        try:
            import paddle.inference as paddle_infer
        except ImportError as exc:
            raise RuntimeError(
                "PaddlePaddle is required for custom MRZ recognition. Install paddlepaddle "
                "or set READMRZ_CUSTOM_OCR_ENABLED=false."
            ) from exc

        started = time.perf_counter()
        config = paddle_infer.Config(
            str(self.model_dir / "inference.json"),
            str(self.model_dir / "inference.pdiparams"),
        )
        configure_device(config, self.device, env)
        if env_bool(env, "READMRZ_CUSTOM_OCR_MEMORY_OPTIM", False):
            config.enable_memory_optim()
        config.disable_glog_info()
        self.model = paddle_infer.create_predictor(config)
        input_names = self.model.get_input_names()
        output_names = self.model.get_output_names()
        if len(input_names) != 1 or not output_names:
            raise RuntimeError(
                f"Unexpected custom MRZ OCR inputs/outputs: inputs={input_names}, outputs={output_names}"
            )
        self.input_name = input_names[0]
        self.output_name = output_names[0]
        self.load_ms = int((time.perf_counter() - started) * 1000)

    def recognize(self, images: list[np.ndarray]) -> tuple[list[dict[str, Any]], int]:
        if not images:
            return [], 0
        if not self.enabled or self.model is None:
            return [empty_result("Custom MRZ OCR is disabled") for _ in images], 0

        started = time.perf_counter()
        batch = np.stack([preprocess_line(image, self.image_shape) for image in images])
        with self._predict_lock:
            input_handle = self.model.get_input_handle(self.input_name)
            input_handle.reshape(batch.shape)
            input_handle.copy_from_cpu(batch)
            self.model.run()
            logits = self.model.get_output_handle(self.output_name).copy_to_cpu()
        latency_ms = int((time.perf_counter() - started) * 1000)
        per_line_ms = round(latency_ms / max(1, len(images)), 2)

        results: list[dict[str, Any]] = []
        for text, confidence in decode_ctc_batch(logits, self.characters):
            normalized_text = normalize_mrz_text(text)
            results.append(
                {
                    "ocr_text": text,
                    "ocr_normalized_text": normalized_text,
                    "ocr_confidence": round(confidence, 6),
                    "ocr_accepted": bool(normalized_text) and confidence >= self.min_confidence,
                    "ocr_latency_ms": per_line_ms,
                    "ocr_error": None,
                }
            )

        while len(results) < len(images):
            results.append(empty_result("OCR returned fewer results than input lines", per_line_ms))
        return results[: len(images)], latency_ms

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "engine": "paddle-text-recognition-ctc",
            "device": self.device,
            "batch_size": self.batch_size,
            "min_confidence": self.min_confidence,
            "model_dir": str(self.model_dir) if self.model_dir else None,
            "model_name": self.model_name,
            "dict_path": str(self.dict_path) if self.dict_path else None,
            "model_load_ms": self.load_ms,
            "image_shape": list(self.image_shape),
        }


def resolve_required_path(env: dict[str, str], key: str, *, directory: bool) -> Path:
    raw_path = env_value(env, key).strip()
    if not raw_path:
        raise ValueError(f"{key} is required when READMRZ_CUSTOM_OCR_ENABLED=true")
    path = Path(raw_path).expanduser().resolve()
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        raise FileNotFoundError(f"{key} does not exist: {path}")
    return path


def empty_result(error: str, latency_ms: float = 0.0) -> dict[str, Any]:
    return {
        "ocr_text": "",
        "ocr_normalized_text": "",
        "ocr_confidence": 0.0,
        "ocr_accepted": False,
        "ocr_latency_ms": latency_ms,
        "ocr_error": error,
    }


def read_image_shape(config: dict[str, Any]) -> tuple[int, int, int]:
    transforms = (config or {}).get("PreProcess", {}).get("transform_ops", [])
    for transform in transforms:
        if isinstance(transform, dict) and "RecResizeImg" in transform:
            values = transform["RecResizeImg"].get("image_shape", [3, 48, 640])
            if len(values) >= 3:
                return int(values[0]), int(values[1]), int(values[2])
    return 3, 48, 640


def read_character_dictionary(path: Path) -> list[str]:
    characters = [line.rstrip("\r\n") for line in path.read_text(encoding="utf-8").splitlines()]
    characters = [character for character in characters if character]
    if not characters:
        raise ValueError(f"Custom MRZ OCR dictionary is empty: {path}")
    return characters


def configure_device(config: Any, device: str, env: dict[str, str]) -> None:
    normalized = device.strip().lower()
    if normalized.startswith("gpu"):
        device_id = 0
        if ":" in normalized:
            try:
                device_id = int(normalized.split(":", 1)[1])
            except ValueError:
                device_id = 0
        memory_mb = int(env_value(env, "READMRZ_CUSTOM_OCR_GPU_MEMORY_MB", "500"))
        config.enable_use_gpu(memory_mb, device_id)
        return
    config.disable_gpu()
    config.set_cpu_math_library_num_threads(
        max(1, int(env_value(env, "READMRZ_CUSTOM_OCR_CPU_THREADS", "4")))
    )


def preprocess_line(image: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
    channels, target_height, target_width = image_shape
    if channels != 3:
        raise ValueError(f"Custom MRZ OCR expects 3 channels, got {channels}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    source_height, source_width = image.shape[:2]
    resized_width = min(
        target_width,
        max(1, int(round(target_height * source_width / max(1, source_height)))),
    )
    resized = cv2.resize(image, (resized_width, target_height), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - 0.5) / 0.5
    normalized = normalized.transpose(2, 0, 1)
    padded = np.zeros((channels, target_height, target_width), dtype=np.float32)
    padded[:, :, :resized_width] = normalized
    return padded


def decode_ctc_batch(logits: np.ndarray, characters: list[str]) -> list[tuple[str, float]]:
    predictions = np.asarray(logits)
    if predictions.ndim != 3:
        raise RuntimeError(f"Custom MRZ OCR output must be [batch,time,class], got {predictions.shape}")
    expected_classes = len(characters) + 1
    if predictions.shape[2] != expected_classes:
        raise RuntimeError(
            "Custom MRZ OCR output/dictionary mismatch: "
            f"model has {predictions.shape[2]} classes, dictionary expects {expected_classes}"
        )
    class_sums = predictions.sum(axis=2)
    already_probabilities = (
        float(predictions.min()) >= 0.0
        and float(predictions.max()) <= 1.0
        and float(np.mean(np.abs(class_sums - 1.0))) < 1e-3
    )
    probabilities = predictions if already_probabilities else softmax(predictions)
    indices = probabilities.argmax(axis=2)
    scores = probabilities.max(axis=2)
    decoded: list[tuple[str, float]] = []
    for row_indices, row_scores in zip(indices, scores, strict=False):
        text_parts: list[str] = []
        char_scores: list[float] = []
        previous = -1
        for class_index, score in zip(row_indices.tolist(), row_scores.tolist(), strict=False):
            if class_index != 0 and class_index != previous:
                dictionary_index = class_index - 1
                if 0 <= dictionary_index < len(characters):
                    text_parts.append(characters[dictionary_index])
                    char_scores.append(float(score))
            previous = class_index
        confidence = sum(char_scores) / len(char_scores) if char_scores else 0.0
        decoded.append(("".join(text_parts), confidence))
    return decoded


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=2, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.maximum(exp_values.sum(axis=2, keepdims=True), 1e-12)
