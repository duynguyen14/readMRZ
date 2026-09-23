from __future__ import annotations

from dataclasses import dataclass
import hashlib
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid
from typing import Any

import cv2
import numpy as np
import yaml

from .document_orientation import PaddleDocumentOrientation, env_bool
from .env_config import PROJECT_ROOT, env_value, read_env_file
from .image_payload import ImagePayload, crop_bbox, decode_base64_image_payload, resize_for_inference


DEFAULT_FIELDS = [
    "visa_number",
    "visa_code",
    "valid_from",
    "valid_until",
    "number_of_entries",
    "passport_number",
    "mrz_zone",
    "issue_date",
    "date_of_birth",
    "full_name",
    "nationality",
    "sex",
    "issue_place",
    "issuing_authority",
    "remarks",
]


@dataclass(frozen=True)
class VisaFieldDetection:
    field_name: str
    class_id: int
    bbox_xyxy: list[float]
    confidence: float


class VnVisaReadService:
    def __init__(self) -> None:
        env = read_env_file()
        self.model_path = resolve_model_path(
            env_value(
                env,
                "READMRZ_VN_VISA_READ_YOLO_MODEL_PATH",
                env_value(env, "READMRZ_VN_VISA_YOLO_MODEL_PATH", ""),
            )
        )
        self.conf = float(
            env_value(
                env,
                "READMRZ_VN_VISA_READ_YOLO_CONF",
                env_value(env, "READMRZ_VN_VISA_YOLO_PREDICT_CONF", "0.25"),
            )
        )
        self.iou = float(
            env_value(
                env,
                "READMRZ_VN_VISA_READ_YOLO_IOU",
                env_value(env, "READMRZ_VN_VISA_YOLO_PREDICT_IOU", "0.45"),
            )
        )
        self.imgsz = max(
            32,
            int(
                env_value(
                    env,
                    "READMRZ_VN_VISA_READ_YOLO_IMGSZ",
                    env_value(env, "READMRZ_VN_VISA_YOLO_PREDICT_IMGSZ", "960"),
                )
            ),
        )
        self.device = env_value(
            env,
            "READMRZ_VN_VISA_READ_DEVICE",
            env_value(env, "READMRZ_VN_VISA_YOLO_PREDICT_DEVICE", "cpu"),
        ).strip()
        self.max_det = max(1, int(env_value(env, "READMRZ_VN_VISA_READ_MAX_DET", "50")))
        self.max_side = max(
            0,
            int(env_value(env, "READMRZ_VN_VISA_READ_MAX_IMAGE_SIDE", env_value(env, "READMRZ_INFERENCE_MAX_IMAGE_SIDE", "1600"))),
        )
        self.padding_ratio = max(0.0, float(env_value(env, "READMRZ_VN_VISA_READ_CROP_PADDING_RATIO", "0.02")))
        self.ocr_batch_size = max(1, int(env_value(env, "READMRZ_VN_VISA_READ_OCR_BATCH_SIZE", "16")))
        self.ocr_model_name = env_value(env, "READMRZ_VN_VISA_READ_OCR_MODEL_NAME", "latin_PP-OCRv5_mobile_rec")
        self.ocr_model_dir = optional_path(env_value(env, "READMRZ_VN_VISA_READ_OCR_MODEL_DIR", ""))
        self.ocr_device = env_value(env, "READMRZ_VN_VISA_READ_OCR_DEVICE", "cpu").strip()
        self.skip_fields = set(
            csv_values(env_value(env, "READMRZ_VN_VISA_READ_SKIP_FIELDS", "mrz_zone,issuing_authority"))
        )
        self.multiline_fields = set(
            csv_values(env_value(env, "READMRZ_VN_VISA_READ_MULTILINE_FIELDS", "number_of_entries,issue_place"))
        )
        configured_fields = csv_values(
            env_value(
                env,
                "READMRZ_VN_VISA_READ_FIELDS",
                env_value(env, "READMRZ_VN_VISA_YOLO_PREDICT_FIELDS", ""),
            )
        )
        self.allowed_fields = set(configured_fields or DEFAULT_FIELDS) - self.skip_fields
        self.temp_dir = Path(
            env_value(env, "READMRZ_VN_VISA_READ_TEMP_DIR", str(PROJECT_ROOT / "tmp" / "vn_visa_read"))
        ).expanduser().resolve()
        self.keep_temp = env_bool(env, "READMRZ_VN_VISA_READ_KEEP_TEMP", False)

        from ultralytics import YOLO

        os.environ["PADDLE_PDX_MODEL_SOURCE"] = env_value(env, "PADDLE_PDX_MODEL_SOURCE", "BOS")

        yolo_started = time.perf_counter()
        self.yolo_model = YOLO(str(self.model_path))
        self.yolo_names = model_class_names(self.yolo_model)
        self.yolo_load_ms = int((time.perf_counter() - yolo_started) * 1000)

        ocr_started = time.perf_counter()
        self.ocr_model = create_ocr_recognizer(
            env=env,
            model_name=self.ocr_model_name,
            model_dir=self.ocr_model_dir,
            device=self.ocr_device,
        )
        self.ocr_load_ms = int((time.perf_counter() - ocr_started) * 1000)
        self._yolo_lock = threading.Lock()
        self._ocr_lock = threading.Lock()

    def runtime_info(self) -> dict[str, Any]:
        return {
            "detector": "YOLO",
            "detector_model_path": str(self.model_path),
            "detector_model_load_ms": self.yolo_load_ms,
            "detector_conf": self.conf,
            "detector_iou": self.iou,
            "detector_imgsz": self.imgsz,
            "device": self.device,
            "ocr": getattr(self.ocr_model, "engine_name", "PaddleOCR TextRecognition"),
            "ocr_model_name": self.ocr_model_name,
            "ocr_model_dir": str(self.ocr_model_dir) if self.ocr_model_dir else "",
            "ocr_device": self.ocr_device,
            "ocr_model_load_ms": self.ocr_load_ms,
            "skip_fields": sorted(self.skip_fields),
            "multiline_fields": sorted(self.multiline_fields),
        }

    def warmup(self) -> int:
        image = np.full((320, 480, 3), 255, dtype=np.uint8)
        temp_root = self.temp_dir / "warmup"
        temp_path = temp_root / "warmup.jpg"
        started = time.perf_counter()
        try:
            imwrite(temp_path, image)
            with self._yolo_lock:
                self.yolo_model.predict(
                    source=str(temp_path),
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    device=self.device or None,
                    max_det=self.max_det,
                    verbose=False,
                )
            self.predict_ocr([temp_path])
        finally:
            if not self.keep_temp:
                safe_unlink(temp_path)
        return int((time.perf_counter() - started) * 1000)

    def read_base64(
        self,
        *,
        image_base64: str,
        file_name: str,
        orientation: PaddleDocumentOrientation,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        payload = decode_base64_image_payload(image_base64, file_name)
        working = resize_for_inference(payload.image, self.max_side)
        oriented_image, orientation_meta = orientation.normalize(working.image)
        detect_started = time.perf_counter()
        detections, raw_detections = self.detect_fields(oriented_image)
        detect_ms = round((time.perf_counter() - detect_started) * 1000, 2)

        ocr_started = time.perf_counter()
        fields = self.ocr_fields(oriented_image, detections, payload.file_name)
        ocr_ms = round((time.perf_counter() - ocr_started) * 1000, 2)
        height, width = oriented_image.shape[:2]
        return {
            "file_name": payload.file_name,
            "request_hash": hashlib.sha256(payload.file_bytes).hexdigest(),
            "image": {
                "original_width": payload.width,
                "original_height": payload.height,
                "working_width": working.width,
                "working_height": working.height,
                "width": width,
                "height": height,
                "resized_for_inference": working.resized,
                "orientation": orientation_meta,
            },
            "fields": fields,
            "detections": [
                {
                    "field": detection.field_name,
                    "class_id": detection.class_id,
                    "confidence": detection.confidence,
                    "bbox": xyxy_to_ltrb(detection.bbox_xyxy),
                    "bbox_xyxy": detection.bbox_xyxy,
                }
                for detection in detections
            ],
            "engine": self.runtime_info(),
            "performance": {
                "orientation_ms": orientation_meta.get("latency_ms", 0),
                "detect_ms": detect_ms,
                "ocr_ms": ocr_ms,
                "total_ms": round((time.perf_counter() - started) * 1000, 2),
            },
            "raw_detections": raw_detections,
        }

    def detect_fields(self, image: np.ndarray) -> tuple[list[VisaFieldDetection], int]:
        temp_root = self.temp_dir / f"detect_{uuid.uuid4().hex}"
        temp_path = temp_root / "image.jpg"
        try:
            imwrite(temp_path, image)
            with self._yolo_lock:
                results = self.yolo_model.predict(
                    source=str(temp_path),
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    device=self.device or None,
                    max_det=self.max_det,
                    verbose=False,
                )
        finally:
            if not self.keep_temp:
                safe_unlink(temp_path)
                safe_rmdir(temp_root)
        result = results[0] if results else None
        return result_detections(result, self.yolo_names, self.allowed_fields)

    def ocr_fields(
        self,
        image: np.ndarray,
        detections: list[VisaFieldDetection],
        file_name: str,
    ) -> dict[str, Any]:
        request_dir = self.temp_dir / f"ocr_{Path(file_name).stem}_{uuid.uuid4().hex}"
        crop_jobs: list[dict[str, Any]] = []
        fields: dict[str, Any] = {}
        for detection in detections:
            bbox = xyxy_to_ltrb(detection.bbox_xyxy)
            crop = crop_bbox(image, bbox, padding_ratio=self.padding_ratio)
            if crop is None:
                fields[detection.field_name] = {
                    "text": "",
                    "confidence": 0.0,
                    "bbox": bbox,
                    "bbox_xyxy": detection.bbox_xyxy,
                    "detection_confidence": detection.confidence,
                    "error": "empty_crop",
                }
                continue

            line_items = split_text_lines(crop) if detection.field_name in self.multiline_fields else []
            line_items = [
                {"image": crop, "bbox_in_crop": [0.0, 0.0, float(crop.shape[1]), float(crop.shape[0])], "source": "full_crop"},
                *line_items,
            ]

            job_indexes: list[int] = []
            for line_index, line_item in enumerate(line_items, start=1):
                crop_path = request_dir / detection.field_name / f"{line_index:02d}.jpg"
                imwrite(crop_path, line_item["image"])
                job_indexes.append(len(crop_jobs))
                crop_jobs.append(
                    {
                        "path": crop_path,
                        "field_name": detection.field_name,
                        "bbox_in_crop": line_item["bbox_in_crop"],
                        "source": line_item.get("source") or "split_line",
                    }
                )

            fields[detection.field_name] = {
                "text": "",
                "confidence": 0.0,
                "bbox": bbox,
                "bbox_xyxy": detection.bbox_xyxy,
                "detection_confidence": detection.confidence,
                "lines": [],
                "_job_indexes": job_indexes,
            }

        try:
            predictions = self.predict_ocr([job["path"] for job in crop_jobs])
        finally:
            if not self.keep_temp:
                safe_rmtree(request_dir)

        for job, prediction in zip(crop_jobs, predictions):
            field_payload = fields.get(job["field_name"])
            if not isinstance(field_payload, dict):
                continue
            text = normalize_space(str(prediction.get("text") or ""))
            score = round(float(prediction.get("score") or 0.0), 6)
            field_payload["lines"].append(
                {
                    "text": text,
                    "confidence": score,
                    "bbox_in_crop": job["bbox_in_crop"],
                    "source": job.get("source") or "split_line",
                }
            )

        for field_name, field_payload in fields.items():
            if not isinstance(field_payload, dict):
                continue
            field_payload.pop("_job_indexes", None)
            lines = field_payload.get("lines") or []
            texts = [str(line.get("text") or "").strip() for line in lines if str(line.get("text") or "").strip()]
            scores = [float(line.get("confidence") or 0.0) for line in lines]
            raw_text = normalize_space(" ".join(texts))
            field_payload["text"] = raw_text
            field_payload["confidence"] = round(sum(scores) / len(scores), 6) if scores else 0.0
            post_process_field(field_name, field_payload)
            if len(lines) <= 1:
                field_payload.pop("lines", None)

        return fields

    def predict_ocr(self, paths: list[Path]) -> list[dict[str, Any]]:
        if not paths:
            return []
        with self._ocr_lock:
            if hasattr(self.ocr_model, "predict_paths"):
                return self.ocr_model.predict_paths(paths, batch_size=self.ocr_batch_size)
            output = self.ocr_model.predict(input=[str(path) for path in paths], batch_size=self.ocr_batch_size)
        return [parse_recognition_result(item) for item in output]


def create_ocr_recognizer(
    *,
    env: dict[str, str],
    model_name: str,
    model_dir: Path | None,
    device: str,
) -> Any:
    engine = env_value(env, "READMRZ_VN_VISA_READ_OCR_ENGINE", "auto").strip().lower()
    if engine in {"custom_ctc", "paddle_inference_ctc"} or (
        engine == "auto" and model_dir is not None and is_paddle_ctc_inference_dir(model_dir)
    ):
        return PaddleCtcTextRecognizer(model_dir=model_dir, device=device, env=env)

    from paddleocr import TextRecognition

    ocr_kwargs: dict[str, Any] = {"model_name": model_name, "device": device}
    if model_dir is not None:
        ocr_kwargs["model_dir"] = str(model_dir)
    model = TextRecognition(**ocr_kwargs)
    setattr(model, "engine_name", "PaddleOCR TextRecognition")
    return model


def is_paddle_ctc_inference_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "inference.pdiparams").is_file()
        and ((path / "inference.json").is_file() or (path / "inference.pdmodel").is_file())
        and ((path / "vn_visa_rec_dict.txt").is_file() or (path / "ppocr_keys_v1.txt").is_file())
    )


class PaddleCtcTextRecognizer:
    engine_name = "Paddle inference CRNN CTC"

    def __init__(self, *, model_dir: Path | None, device: str, env: dict[str, str]) -> None:
        if model_dir is None:
            raise ValueError("READMRZ_VN_VISA_READ_OCR_MODEL_DIR is required for custom CTC OCR")
        self.model_dir = model_dir
        self.device = device
        self.cpu_threads = max(1, int(env_value(env, "READMRZ_VN_VISA_READ_OCR_CPU_THREADS", "4")))
        self.model_file = first_existing(model_dir / "inference.json", model_dir / "inference.pdmodel")
        self.params_file = model_dir / "inference.pdiparams"
        if self.model_file is None:
            raise FileNotFoundError(f"Missing inference.json or inference.pdmodel in OCR model dir: {model_dir}")
        if not self.params_file.is_file():
            raise FileNotFoundError(f"Missing inference.pdiparams in OCR model dir: {model_dir}")

        configured_dict = env_value(env, "READMRZ_VN_VISA_READ_OCR_DICT_PATH", "").strip()
        self.dict_path = Path(configured_dict).expanduser().resolve() if configured_dict else first_existing(
            model_dir / "vn_visa_rec_dict.txt",
            model_dir / "ppocr_keys_v1.txt",
        )
        if self.dict_path is None or not self.dict_path.is_file():
            raise FileNotFoundError(
                "Missing OCR dictionary. Put vn_visa_rec_dict.txt in OCR model dir "
                "or set READMRZ_VN_VISA_READ_OCR_DICT_PATH."
            )

        self.inference_config = load_yaml(model_dir / "inference.yml")
        self.training_config = load_yaml(model_dir / "vn_visa_crnn_ctc_48x640.yml")
        self.image_shape = read_rec_image_shape(self.inference_config, self.training_config)
        self.characters = read_rec_dictionary(
            self.dict_path,
            use_space_char=read_use_space_char(self.inference_config, self.training_config),
        )

        try:
            import paddle.inference as paddle_infer
        except ImportError as exc:
            raise RuntimeError("PaddlePaddle is required for custom VN visa OCR inference.") from exc

        config = paddle_infer.Config(str(self.model_file), str(self.params_file))
        configure_paddle_inference_device(config, device, self.cpu_threads)
        if env_bool(env, "READMRZ_VN_VISA_READ_OCR_MEMORY_OPTIM", True):
            config.enable_memory_optim()
        config.disable_glog_info()
        self.predictor = paddle_infer.create_predictor(config)
        input_names = self.predictor.get_input_names()
        output_names = self.predictor.get_output_names()
        if len(input_names) != 1 or not output_names:
            raise RuntimeError(f"Unexpected VN visa OCR inputs/outputs: inputs={input_names}, outputs={output_names}")
        self.input_name = input_names[0]
        self.output_name = output_names[0]

    def predict_paths(self, paths: list[Path], *, batch_size: int) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for start in range(0, len(paths), max(1, batch_size)):
            batch_paths = paths[start : start + max(1, batch_size)]
            images = [read_image(path) for path in batch_paths]
            batch = np.stack([preprocess_rec_image(image, self.image_shape) for image in images])
            input_handle = self.predictor.get_input_handle(self.input_name)
            input_handle.reshape(batch.shape)
            input_handle.copy_from_cpu(batch)
            self.predictor.run()
            logits = self.predictor.get_output_handle(self.output_name).copy_to_cpu()
            for text, confidence in decode_ctc_batch(logits, self.characters):
                output.append({"text": normalize_space(text), "score": round(confidence, 6), "raw": {}})
        while len(output) < len(paths):
            output.append({"text": "", "score": 0.0, "raw": {"error": "OCR returned fewer results than inputs"}})
        return output[: len(paths)]


def resolve_model_path(raw_value: str) -> Path:
    if not raw_value:
        raise ValueError("Missing READMRZ_VN_VISA_READ_YOLO_MODEL_PATH or READMRZ_VN_VISA_YOLO_MODEL_PATH")
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if path.is_file():
        return path
    for candidate in (path / "best.pt", path / "weights" / "best.pt", path / "last.pt", path / "weights" / "last.pt"):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find best.pt or last.pt under YOLO model path: {path}")


def optional_path(raw_value: str) -> Path | None:
    if not raw_value.strip():
        return None
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"OCR model dir does not exist: {path}")
    return path


def first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.is_file():
            return path.resolve()
    return None


def csv_values(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def read_rec_image_shape(*configs: dict[str, Any]) -> tuple[int, int, int]:
    for config in configs:
        for transform in (config or {}).get("PreProcess", {}).get("transform_ops", []):
            shape = transform_image_shape(transform)
            if shape is not None:
                return shape
        for section in ("Eval", "Train"):
            transforms = (((config or {}).get(section) or {}).get("dataset") or {}).get("transforms", [])
            for transform in transforms:
                shape = transform_image_shape(transform)
                if shape is not None:
                    return shape
    return 3, 48, 640


def transform_image_shape(transform: Any) -> tuple[int, int, int] | None:
    if not isinstance(transform, dict) or "RecResizeImg" not in transform:
        return None
    values = (transform.get("RecResizeImg") or {}).get("image_shape", [])
    if not isinstance(values, list | tuple) or len(values) < 3:
        return None
    return int(values[0]), int(values[1]), int(values[2])


def read_use_space_char(*configs: dict[str, Any]) -> bool:
    for config in configs:
        global_config = (config or {}).get("Global") or {}
        if "use_space_char" in global_config:
            return bool(global_config.get("use_space_char"))
        post_process = (config or {}).get("PostProcess") or {}
        if "use_space_char" in post_process:
            return bool(post_process.get("use_space_char"))
    return True


def read_rec_dictionary(path: Path, *, use_space_char: bool) -> list[str]:
    characters = [line.rstrip("\r\n") for line in path.read_text(encoding="utf-8").splitlines()]
    characters = [character for character in characters if character]
    if use_space_char and " " not in characters:
        characters.append(" ")
    if not characters:
        raise ValueError(f"OCR dictionary is empty: {path}")
    return characters


def configure_paddle_inference_device(config: Any, device: str, cpu_threads: int) -> None:
    normalized = str(device or "cpu").strip().lower()
    if normalized.startswith("gpu") or normalized.startswith("cuda"):
        device_id = 0
        if ":" in normalized:
            try:
                device_id = int(normalized.split(":", 1)[1])
            except ValueError:
                device_id = 0
        config.enable_use_gpu(500, device_id)
        return
    config.disable_gpu()
    config.set_cpu_math_library_num_threads(max(1, cpu_threads))


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read OCR crop image: {path}")
    return image


def preprocess_rec_image(image: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
    channels, target_height, target_width = image_shape
    if channels != 3:
        raise ValueError(f"VN visa OCR expects 3 channels, got {channels}")
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
        raise RuntimeError(f"VN visa OCR output must be [batch,time,class], got {predictions.shape}")
    expected_classes = len(characters) + 1
    if predictions.shape[2] != expected_classes:
        raise RuntimeError(
            "VN visa OCR output/dictionary mismatch: "
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



def model_class_names(model: Any) -> dict[int, str]:
    names = getattr(model, "names", {})
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    return {index: field for index, field in enumerate(DEFAULT_FIELDS)}


def result_detections(
    result: Any,
    names: dict[int, str],
    allowed_fields: set[str],
) -> tuple[list[VisaFieldDetection], int]:
    if result is None:
        return [], 0
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return [], 0
    orig_shape = getattr(result, "orig_shape", None) or (0, 0)
    image_height = int(orig_shape[0] or 0)
    image_width = int(orig_shape[1] or 0)
    xyxy_values = getattr(boxes, "xyxy", None)
    conf_values = getattr(boxes, "conf", None)
    cls_values = getattr(boxes, "cls", None)
    if xyxy_values is None or conf_values is None or cls_values is None:
        return [], 0

    by_field: dict[str, VisaFieldDetection] = {}
    raw_count = 0
    for raw_box, raw_conf, raw_class_id in zip(
        xyxy_values.cpu().numpy().tolist(),
        conf_values.cpu().numpy().tolist(),
        cls_values.cpu().numpy().tolist(),
    ):
        raw_count += 1
        class_id = int(raw_class_id)
        field_name = names.get(class_id, str(class_id))
        if field_name not in allowed_fields:
            continue
        bbox = clamp_bbox([float(value) for value in raw_box], image_width, image_height)
        if bbox is None:
            continue
        detection = VisaFieldDetection(
            field_name=field_name,
            class_id=class_id,
            bbox_xyxy=bbox,
            confidence=round(float(raw_conf), 6),
        )
        previous = by_field.get(field_name)
        if previous is None or detection.confidence > previous.confidence:
            by_field[field_name] = detection

    return sorted(by_field.values(), key=lambda item: DEFAULT_FIELDS.index(item.field_name) if item.field_name in DEFAULT_FIELDS else 999), raw_count


def clamp_bbox(box: list[float], image_width: int, image_height: int) -> list[float] | None:
    if image_width <= 1 or image_height <= 1:
        return None
    x1, y1, x2, y2 = box
    left = max(0.0, min(x1, x2))
    top = max(0.0, min(y1, y2))
    right = min(float(image_width - 1), max(x1, x2))
    bottom = min(float(image_height - 1), max(y1, y2))
    if right - left < 2 or bottom - top < 2:
        return None
    return [round(left, 2), round(top, 2), round(right, 2), round(bottom, 2)]


def xyxy_to_ltrb(box: list[float]) -> dict[str, float]:
    left, top, right, bottom = [float(value) for value in box]
    return {
        "left": round(left, 2),
        "top": round(top, 2),
        "width": round(max(0.0, right - left), 2),
        "height": round(max(0.0, bottom - top), 2),
    }


def split_text_lines(crop: np.ndarray) -> list[dict[str, Any]]:
    if crop is None or crop.size == 0:
        return []
    height, width = crop.shape[:2]
    if height < 18 or width < 12:
        return []

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        max(15, (height // 3) * 2 + 1),
        7,
    )
    binary = cv2.bitwise_or(otsu, adaptive)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(2, width // 60), 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    projection = (binary > 0).sum(axis=1)
    threshold = max(1, int(width * 0.008))

    bands: list[list[int]] = []
    start: int | None = None
    for row_index, value in enumerate(projection):
        if int(value) >= threshold:
            if start is None:
                start = row_index
        elif start is not None:
            bands.append([start, row_index])
            start = None
    if start is not None:
        bands.append([start, height])

    if not bands:
        return []

    merged: list[list[int]] = []
    max_gap = max(2, int(height * 0.06))
    for band in bands:
        if not merged or band[0] - merged[-1][1] > max_gap:
            merged.append(band)
        else:
            merged[-1][1] = band[1]

    min_height = max(5, int(height * 0.12))
    padded: list[list[int]] = []
    for top, bottom in merged:
        if bottom - top < min_height:
            continue
        y1 = max(0, top - 2)
        y2 = min(height, bottom + 2)
        padded.append([y1, y2])

    if len(padded) <= 1:
        middle_split = split_crop_middle(crop)
        return middle_split

    return [
        {
            "image": crop[y1:y2, :].copy(),
            "bbox_in_crop": [0.0, float(y1), float(width), float(y2)],
            "source": "projection_line",
        }
        for y1, y2 in padded
        if y2 > y1
    ]


def split_crop_middle(crop: np.ndarray) -> list[dict[str, Any]]:
    height, width = crop.shape[:2]
    if height < 32:
        return []
    mid = height // 2
    gap = max(1, height // 24)
    parts = [(0, max(1, mid + gap)), (max(0, mid - gap), height)]
    output: list[dict[str, Any]] = []
    for top, bottom in parts:
        if bottom - top < max(8, int(height * 0.28)):
            continue
        output.append(
            {
                "image": crop[top:bottom, :].copy(),
                "bbox_in_crop": [0.0, float(top), float(width), float(bottom)],
                "source": "middle_split",
            }
        )
    return output if len(output) >= 2 else []


def post_process_field(field_name: str, payload: dict[str, Any]) -> None:
    if field_name == "number_of_entries":
        normalize_number_of_entries(payload)


def normalize_number_of_entries(payload: dict[str, Any]) -> None:
    lines = payload.get("lines") or []
    candidates = [str(payload.get("text") or "")]
    candidates.extend(str(line.get("text") or "") for line in lines)
    joined = " ".join(candidates)
    normalized = normalize_for_match(joined)
    single_score = max(
        fuzzy_score(normalized, target)
        for target in (
            "motlan",
            "motlansingleentry",
            "single",
            "singleentry",
            "oneentry",
            "01",
        )
    )
    multiple_score = max(
        fuzzy_score(normalized, target)
        for target in (
            "nhieulan",
            "nhieulanmultipleentries",
            "multiple",
            "multipleentries",
            "multientry",
        )
    )

    if normalized in {"1", "01"}:
        single_score += 0.55
    if normalized == "n":
        multiple_score += 0.55
    if "one" in normalized or "single" in normalized:
        single_score += 0.35
    if "multi" in normalized or "multiple" in normalized:
        multiple_score += 0.35
    if "entry" in normalized or "ntry" in normalized or "nry" in normalized:
        single_score += 0.15
    if "entries" in normalized or "tries" in normalized:
        multiple_score += 0.15

    best_text = ""
    best_score = 0.0
    best_kind = ""
    if single_score >= multiple_score and single_score >= 0.42:
        best_text = "Một lần/Single entry"
        best_score = single_score
        best_kind = "single_entry_dictionary"
    elif multiple_score > single_score and multiple_score >= 0.42:
        best_text = "Nhiều lần/Multiple entries"
        best_score = multiple_score
        best_kind = "multiple_entries_dictionary"

    if not best_text:
        return

    raw_text = normalize_space(str(payload.get("text") or ""))
    payload["raw_text"] = raw_text
    payload["text"] = best_text
    payload["normalized_by"] = best_kind
    clamped_score = min(1.0, round(best_score, 6))
    payload["normalization_score"] = clamped_score
    payload["confidence"] = max(float(payload.get("confidence") or 0.0), min(0.95, clamped_score))


def normalize_for_match(value: str) -> str:
    text = str(value or "").lower()
    replacements = {
        "ộ": "o",
        "ồ": "o",
        "ố": "o",
        "ỗ": "o",
        "ổ": "o",
        "ơ": "o",
        "ợ": "o",
        "ờ": "o",
        "ớ": "o",
        "ở": "o",
        "ỡ": "o",
        "ư": "u",
        "ừ": "u",
        "ứ": "u",
        "ử": "u",
        "ữ": "u",
        "ự": "u",
        "ầ": "a",
        "ấ": "a",
        "ậ": "a",
        "ẩ": "a",
        "ẫ": "a",
        "ă": "a",
        "ằ": "a",
        "ắ": "a",
        "ặ": "a",
        "ẳ": "a",
        "ẵ": "a",
        "á": "a",
        "à": "a",
        "ả": "a",
        "ã": "a",
        "ạ": "a",
        "é": "e",
        "è": "e",
        "ẻ": "e",
        "ẽ": "e",
        "ẹ": "e",
        "ê": "e",
        "ề": "e",
        "ế": "e",
        "ể": "e",
        "ễ": "e",
        "ệ": "e",
        "í": "i",
        "ì": "i",
        "ỉ": "i",
        "ĩ": "i",
        "ị": "i",
        "ó": "o",
        "ò": "o",
        "ỏ": "o",
        "õ": "o",
        "ọ": "o",
        "ú": "u",
        "ù": "u",
        "ủ": "u",
        "ũ": "u",
        "ụ": "u",
        "ý": "y",
        "ỳ": "y",
        "ỷ": "y",
        "ỹ": "y",
        "ỵ": "y",
        "đ": "d",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return re.sub(r"[^a-z0-9]+", "", text)


def fuzzy_score(value: str, target: str) -> float:
    if not value and not target:
        return 1.0
    if not value or not target:
        return 0.0
    if target in value:
        return 1.0
    if value in target and len(value) >= 2:
        return 0.75
    return SequenceMatcher(None, value, target).ratio()


def parse_recognition_result(item: Any) -> dict[str, Any]:
    payload = result_to_plain_data(item)
    text = payload.get("rec_text") or payload.get("text") or payload.get("label") or ""
    score = payload.get("rec_score") or payload.get("score") or payload.get("confidence") or 0.0
    try:
        score_float = float(score)
    except (TypeError, ValueError):
        score_float = 0.0
    return {
        "text": normalize_space(str(text or "")),
        "score": round(score_float, 6),
        "raw": payload,
    }


def result_to_plain_data(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return json_safe(item)
    if hasattr(item, "json"):
        try:
            raw_json = item.json() if callable(item.json) else item.json
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                return json_safe(parsed)
        except Exception:
            pass
    if hasattr(item, "to_dict"):
        try:
            parsed = item.to_dict()
            if isinstance(parsed, dict):
                return json_safe(parsed)
        except Exception:
            pass
    if hasattr(item, "__dict__"):
        return json_safe(vars(item))
    return {"repr": repr(item)}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if not success:
        raise RuntimeError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def safe_rmdir(path: Path) -> None:
    try:
        path.rmdir()
    except Exception:
        pass


def safe_rmtree(path: Path) -> None:
    try:
        if path.exists():
            for child in sorted(path.rglob("*"), reverse=True):
                if child.is_file():
                    safe_unlink(child)
                else:
                    safe_rmdir(child)
            safe_rmdir(path)
    except Exception:
        pass
