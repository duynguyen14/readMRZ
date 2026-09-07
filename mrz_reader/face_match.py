from __future__ import annotations

import base64
import hashlib
import tempfile
import threading
import urllib.request
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .document_orientation import env_bool
from .env_config import PROJECT_ROOT, env_value, read_env_file
from .image_payload import decode_base64_image_payload


YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
SFACE_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
    "face_recognition_sface_2021dec.onnx"
)
_YUNET_SCORE_THRESHOLD = 0.7
_YUNET_NMS_THRESHOLD = 0.3
_YUNET_TOP_K = 5000


class FaceMatchService:
    def __init__(self) -> None:
        env = read_env_file()
        self.model_dir = resolve_path(
            env_value(env, "FACE_MATCH_MODEL_DIR", str(PROJECT_ROOT / "models" / "opencv_face"))
        )
        self.detector_model_path = resolve_path(
            env_value(
                env,
                "FACE_MATCH_DETECTOR_MODEL_PATH",
                str(self.model_dir / "face_detection_yunet_2023mar.onnx"),
            )
        )
        self.recognizer_model_path = resolve_path(
            env_value(
                env,
                "FACE_MATCH_RECOGNIZER_MODEL_PATH",
                str(self.model_dir / "face_recognition_sface_2021dec.onnx"),
            )
        )
        self.auto_download = env_bool(env, "FACE_MATCH_AUTO_DOWNLOAD", False)
        self.prefer_cuda = env_bool(env, "FACE_MATCH_PREFER_CUDA", False)
        self.match_threshold = float(env_value(env, "FACE_MATCH_MATCH_THRESHOLD", "0.363"))
        self.review_threshold = float(env_value(env, "FACE_MATCH_REVIEW_THRESHOLD", "0.300"))
        self.input_width = max(160, int(env_value(env, "FACE_MATCH_INPUT_WIDTH", "640")))
        self.input_height = max(160, int(env_value(env, "FACE_MATCH_INPUT_HEIGHT", "640")))
        self.score_threshold = float(env_value(env, "FACE_MATCH_DETECT_SCORE_THRESHOLD", str(_YUNET_SCORE_THRESHOLD)))
        self.nms_threshold = float(env_value(env, "FACE_MATCH_DETECT_NMS_THRESHOLD", str(_YUNET_NMS_THRESHOLD)))
        self.top_k = max(1, int(env_value(env, "FACE_MATCH_DETECT_TOP_K", str(_YUNET_TOP_K))))
        self._lock = threading.Lock()

        self._ensure_models()
        backend_id, target_id, device, target = self._resolve_backend_target()
        started = perf_counter()
        self.detector = cv2.FaceDetectorYN_create(
            str(self.detector_model_path),
            "",
            (self.input_width, self.input_height),
            self.score_threshold,
            self.nms_threshold,
            self.top_k,
            backend_id,
            target_id,
        )
        self.recognizer = cv2.FaceRecognizerSF_create(
            str(self.recognizer_model_path),
            "",
            backend_id,
            target_id,
        )
        self.load_ms = int((perf_counter() - started) * 1000)
        self.device = device
        self.target = target

    def _ensure_models(self) -> None:
        if self.detector_model_path.is_file() and self.recognizer_model_path.is_file():
            return
        if not self.auto_download:
            missing = [
                str(path)
                for path in (self.detector_model_path, self.recognizer_model_path)
                if not path.is_file()
            ]
            raise FileNotFoundError(
                "Face match model file does not exist. Configure FACE_MATCH_* paths. "
                f"missing={missing}"
            )
        if not self.detector_model_path.is_file():
            download_model(YUNET_MODEL_URL, self.detector_model_path)
        if not self.recognizer_model_path.is_file():
            download_model(SFACE_MODEL_URL, self.recognizer_model_path)

    def _resolve_backend_target(self) -> tuple[int, int, str, str]:
        if self.prefer_cuda and hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            return (
                cv2.dnn.DNN_BACKEND_CUDA,
                cv2.dnn.DNN_TARGET_CUDA_FP16,
                "cuda",
                "opencv_cuda_fp16",
            )
        return cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU, "cpu", "opencv_cpu"

    def runtime_info(self) -> dict[str, Any]:
        return {
            "runtime": "opencv",
            "device": self.device,
            "target": self.target,
            "detector_model_path": str(self.detector_model_path),
            "recognizer_model_path": str(self.recognizer_model_path),
            "input_width": self.input_width,
            "input_height": self.input_height,
            "model_load_ms": self.load_ms,
        }

    def warmup(self) -> int:
        image = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
        started = perf_counter()
        self.detect_primary_face(image)
        return int((perf_counter() - started) * 1000)

    def detect_primary_face(self, image: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
        face_rows, meta = self.detect_faces(image)
        return select_primary_face(face_rows), meta

    def detect_faces(self, image: np.ndarray) -> tuple[list[np.ndarray], dict[str, Any]]:
        if image is None or image.size == 0:
            raise ValueError("image is empty")
        started = perf_counter()
        image_height, image_width = image.shape[:2]
        with self._lock:
            self.detector.setInputSize((int(image_width), int(image_height)))
            _, faces = self.detector.detect(image)
        duration_ms = round((perf_counter() - started) * 1000, 2)
        face_rows = [] if faces is None else [face for face in faces]
        return face_rows, {
            "count": len(face_rows),
            "duration_ms": duration_ms,
        }

    def extract_embedding(self, image: np.ndarray, face_row: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        started = perf_counter()
        with self._lock:
            aligned_face = self.recognizer.alignCrop(image, face_row.reshape(1, -1))
            feature = self.recognizer.feature(aligned_face)
        content_type, aligned_base64 = encode_image_to_base64(aligned_face)
        return feature, {
            "aligned_face_content_type": content_type,
            "aligned_face_base64": aligned_base64,
            "align_and_embed_duration_ms": round((perf_counter() - started) * 1000, 2),
            "embedding_norm": round(float(np.linalg.norm(feature.reshape(-1))), 6),
        }

    def match_embeddings(self, left_feature: np.ndarray, right_feature: np.ndarray) -> float:
        with self._lock:
            score = self.recognizer.match(
                left_feature,
                right_feature,
                cv2.FaceRecognizerSF_FR_COSINE,
            )
        return float(score)

    def decision(self, score: float) -> tuple[str, bool, bool]:
        if score >= self.match_threshold:
            return "match", True, False
        if score >= self.review_threshold:
            return "review", False, True
        return "mismatch", False, False

    def verify_base64_pair(
        self,
        *,
        passport_face_base64: str,
        passport_face_file_name: str,
        uploaded_face_base64: str,
        uploaded_face_file_name: str,
    ) -> dict[str, Any]:
        total_started = perf_counter()
        passport_payload = decode_base64_image_payload(passport_face_base64, passport_face_file_name)
        uploaded_payload = decode_base64_image_payload(uploaded_face_base64, uploaded_face_file_name)

        passport_face_row, passport_detect_meta = self.detect_primary_face(passport_payload.image)
        uploaded_face_row, uploaded_detect_meta = self.detect_primary_face(uploaded_payload.image)
        response: dict[str, Any] = {
            "matched": False,
            "review_required": False,
            "decision": "review",
            "score": 0.0,
            "message": "",
            "thresholds": {
                "match": self.match_threshold,
                "review": self.review_threshold,
                "metric": "cosine_similarity",
            },
            "engine": {
                "detector": "YuNet",
                "recognizer": "SFace",
                **self.runtime_info(),
            },
            "passport_face": {
                "file_name": passport_payload.file_name,
                "content_type": passport_payload.content_type,
                "base64": passport_payload.base64_value,
                "detected": passport_face_row is not None,
                "face_count": passport_detect_meta["count"],
                "face_bbox": face_bbox(passport_face_row) if passport_face_row is not None else None,
                "face_confidence": face_confidence(passport_face_row) if passport_face_row is not None else 0.0,
                "aligned_face_content_type": "",
                "aligned_face_base64": "",
            },
            "uploaded_face": {
                "file_name": uploaded_payload.file_name,
                "content_type": uploaded_payload.content_type,
                "base64": uploaded_payload.base64_value,
                "detected": uploaded_face_row is not None,
                "face_count": uploaded_detect_meta["count"],
                "face_bbox": face_bbox(uploaded_face_row) if uploaded_face_row is not None else None,
                "face_confidence": face_confidence(uploaded_face_row) if uploaded_face_row is not None else 0.0,
                "aligned_face_content_type": "",
                "aligned_face_base64": "",
            },
            "performance": {
                "passport_detect_duration_ms": passport_detect_meta["duration_ms"],
                "uploaded_detect_duration_ms": uploaded_detect_meta["duration_ms"],
                "passport_align_embed_duration_ms": 0.0,
                "uploaded_align_embed_duration_ms": 0.0,
                "match_duration_ms": 0.0,
                "total_duration_ms": 0.0,
            },
            "request_hash": hashlib.sha256(
                passport_payload.file_bytes + b"::" + uploaded_payload.file_bytes
            ).hexdigest(),
        }

        if passport_face_row is None and uploaded_face_row is None:
            response["message"] = "Khong detect duoc mat tren ca 2 anh."
        elif passport_face_row is None:
            response["message"] = "Khong detect duoc mat tren anh chan dung cat tu passport."
        elif uploaded_face_row is None:
            response["message"] = "Khong detect duoc mat tren anh mat vua upload."
        else:
            passport_feature, passport_meta = self.extract_embedding(passport_payload.image, passport_face_row)
            uploaded_feature, uploaded_meta = self.extract_embedding(uploaded_payload.image, uploaded_face_row)
            response["passport_face"]["aligned_face_content_type"] = passport_meta["aligned_face_content_type"]
            response["passport_face"]["aligned_face_base64"] = passport_meta["aligned_face_base64"]
            response["uploaded_face"]["aligned_face_content_type"] = uploaded_meta["aligned_face_content_type"]
            response["uploaded_face"]["aligned_face_base64"] = uploaded_meta["aligned_face_base64"]
            response["performance"]["passport_align_embed_duration_ms"] = passport_meta["align_and_embed_duration_ms"]
            response["performance"]["uploaded_align_embed_duration_ms"] = uploaded_meta["align_and_embed_duration_ms"]

            match_started = perf_counter()
            score = self.match_embeddings(passport_feature, uploaded_feature)
            response["performance"]["match_duration_ms"] = round((perf_counter() - match_started) * 1000, 2)
            response["score"] = round(score, 6)
            decision, matched, review_required = self.decision(score)
            response["decision"] = decision
            response["matched"] = matched
            response["review_required"] = review_required
            response["message"] = {
                "match": "Anh mat khop nhau.",
                "review": "Anh mat gan giong, nen review them truoc khi chap nhan.",
                "mismatch": "Anh mat khong khop nhau.",
            }[decision]

        response["performance"]["total_duration_ms"] = round((perf_counter() - total_started) * 1000, 2)
        return response


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def download_model(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        with tempfile.NamedTemporaryFile(delete=False, dir=str(output_path.parent), suffix=".tmp") as temp_file:
            temp_file.write(response.read())
            temp_path = Path(temp_file.name)
    temp_path.replace(output_path)


def select_primary_face(faces: list[np.ndarray]) -> np.ndarray | None:
    if not faces:
        return None

    def sort_key(face_row: np.ndarray) -> float:
        width = max(0.0, float(face_row[2]))
        height = max(0.0, float(face_row[3]))
        score = float(face_row[14]) if len(face_row) > 14 else 0.0
        return score * max(1.0, width * height)

    return max(faces, key=sort_key)


def face_bbox(face_row: np.ndarray) -> dict[str, float]:
    return {
        "left": round(float(face_row[0]), 2),
        "top": round(float(face_row[1]), 2),
        "width": round(float(face_row[2]), 2),
        "height": round(float(face_row[3]), 2),
    }


def face_confidence(face_row: np.ndarray) -> float:
    if len(face_row) <= 14:
        return 0.0
    return round(float(face_row[14]), 6)


def encode_image_to_base64(image: np.ndarray) -> tuple[str, str]:
    success, encoded_image = cv2.imencode(".jpg", image)
    if not success:
        return "", ""
    return "image/jpeg", base64.b64encode(encoded_image.tobytes()).decode("ascii")
