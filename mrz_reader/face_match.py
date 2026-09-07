from __future__ import annotations

import base64
import hashlib
import threading
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .document_orientation import env_bool
from .env_config import PROJECT_ROOT, env_value, read_env_file
from .image_payload import crop_bbox, decode_base64_image_payload


_DEFAULT_DET_SIZE = 640
_DEFAULT_SCORE_THRESHOLD = 0.5
_DEFAULT_MATCH_THRESHOLD = 0.45
_DEFAULT_REVIEW_THRESHOLD = 0.35


class FaceMatchService:
    def __init__(self) -> None:
        env = read_env_file()
        self.model_name = env_value(env, "INSIGHTFACE_MODEL_NAME", "buffalo_l")
        self.model_root = resolve_path(
            env_value(env, "INSIGHTFACE_MODEL_ROOT", str(PROJECT_ROOT / "models" / "insightface"))
        )
        self.providers = parse_providers(env_value(env, "INSIGHTFACE_PROVIDERS", "CPUExecutionProvider"))
        self.prefer_cuda = env_bool(env, "FACE_MATCH_PREFER_CUDA", False)
        if self.prefer_cuda and "CUDAExecutionProvider" not in self.providers:
            self.providers = ["CUDAExecutionProvider", *self.providers]
        self.device = "cuda" if "CUDAExecutionProvider" in self.providers else "cpu"
        self.ctx_id = int(env_value(env, "INSIGHTFACE_CTX_ID", "0" if self.device == "cuda" else "-1"))
        self.match_threshold = float(
            env_value(env, "INSIGHTFACE_MATCH_THRESHOLD", env_value(env, "FACE_MATCH_MATCH_THRESHOLD", str(_DEFAULT_MATCH_THRESHOLD)))
        )
        self.review_threshold = float(
            env_value(
                env,
                "INSIGHTFACE_REVIEW_THRESHOLD",
                env_value(env, "FACE_MATCH_REVIEW_THRESHOLD", str(_DEFAULT_REVIEW_THRESHOLD)),
            )
        )
        self.input_width = max(
            160,
            int(env_value(env, "INSIGHTFACE_DET_WIDTH", env_value(env, "FACE_MATCH_INPUT_WIDTH", str(_DEFAULT_DET_SIZE)))),
        )
        self.input_height = max(
            160,
            int(env_value(env, "INSIGHTFACE_DET_HEIGHT", env_value(env, "FACE_MATCH_INPUT_HEIGHT", str(_DEFAULT_DET_SIZE)))),
        )
        self.score_threshold = float(
            env_value(
                env,
                "INSIGHTFACE_DETECT_SCORE_THRESHOLD",
                env_value(env, "FACE_MATCH_DETECT_SCORE_THRESHOLD", str(_DEFAULT_SCORE_THRESHOLD)),
            )
        )
        self.max_detections = max(
            1,
            int(env_value(env, "READMRZ_FACE_MATCH_MAX_DETECTIONS_PER_IMAGE", "5")),
        )
        self._lock = threading.Lock()

        FaceAnalysis, _ = load_insightface()
        started = perf_counter()
        self.app = FaceAnalysis(
            name=self.model_name,
            root=str(self.model_root),
            providers=self.providers,
            allowed_modules=["detection", "recognition"],
        )
        self.app.prepare(
            ctx_id=self.ctx_id,
            det_thresh=self.score_threshold,
            det_size=(self.input_width, self.input_height),
        )
        self.load_ms = int((perf_counter() - started) * 1000)

    def runtime_info(self) -> dict[str, Any]:
        return {
            "runtime": "insightface",
            "device": self.device,
            "target": "onnxruntime",
            "model_name": self.model_name,
            "model_root": str(self.model_root),
            "providers": self.providers,
            "ctx_id": self.ctx_id,
            "input_width": self.input_width,
            "input_height": self.input_height,
            "detect_score_threshold": self.score_threshold,
            "max_detections_per_image": self.max_detections,
            "model_load_ms": self.load_ms,
        }

    def warmup(self) -> int:
        image = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
        started = perf_counter()
        self.detect_primary_face(image)
        return int((perf_counter() - started) * 1000)

    def detect_primary_face(self, image: np.ndarray) -> tuple[Any | None, dict[str, Any]]:
        face_rows, meta = self.detect_faces(image)
        return select_primary_face(face_rows), meta

    def detect_faces(self, image: np.ndarray) -> tuple[list[Any], dict[str, Any]]:
        if image is None or image.size == 0:
            raise ValueError("image is empty")
        started = perf_counter()
        with self._lock:
            faces = self.app.get(image)
        duration_ms = round((perf_counter() - started) * 1000, 2)
        face_rows = sorted(list(faces or []), key=face_confidence, reverse=True)
        attach_source_image(face_rows, image)
        return face_rows[: self.max_detections], {
            "count": len(face_rows),
            "duration_ms": duration_ms,
        }

    def extract_embedding(self, image: np.ndarray, face_row: Any) -> tuple[np.ndarray, dict[str, Any]]:
        started = perf_counter()
        feature = normalized_embedding(face_row)
        aligned_face = aligned_face_image(image, face_row)
        content_type, aligned_base64 = encode_image_to_base64(aligned_face)
        return feature, {
            "aligned_face_content_type": content_type,
            "aligned_face_base64": aligned_base64,
            "align_and_embed_duration_ms": round((perf_counter() - started) * 1000, 2),
            "embedding_norm": round(float(np.linalg.norm(feature.reshape(-1))), 6),
        }

    def match_embeddings(self, left_feature: np.ndarray, right_feature: np.ndarray) -> float:
        left = normalize_vector(left_feature)
        right = normalize_vector(right_feature)
        return float(np.dot(left, right))

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

        passport_faces, passport_detect_meta = self.detect_faces(passport_payload.image)
        uploaded_faces, uploaded_detect_meta = self.detect_faces(uploaded_payload.image)
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
                "detector": "SCRFD",
                "recognizer": "ArcFace",
                **self.runtime_info(),
            },
            "passport_face": {
                "file_name": passport_payload.file_name,
                "content_type": passport_payload.content_type,
                "base64": passport_payload.base64_value,
                "detected": bool(passport_faces),
                "face_count": passport_detect_meta["count"],
                "face_bbox": face_bbox(passport_faces[0]) if passport_faces else None,
                "face_confidence": face_confidence(passport_faces[0]) if passport_faces else 0.0,
                "aligned_face_content_type": "",
                "aligned_face_base64": "",
            },
            "uploaded_face": {
                "file_name": uploaded_payload.file_name,
                "content_type": uploaded_payload.content_type,
                "base64": uploaded_payload.base64_value,
                "detected": bool(uploaded_faces),
                "face_count": uploaded_detect_meta["count"],
                "face_bbox": face_bbox(uploaded_faces[0]) if uploaded_faces else None,
                "face_confidence": face_confidence(uploaded_faces[0]) if uploaded_faces else 0.0,
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

        if not passport_faces and not uploaded_faces:
            response["message"] = "Khong detect duoc mat tren ca 2 anh."
        elif not passport_faces:
            response["message"] = "Khong detect duoc mat tren anh chan dung cat tu passport."
        elif not uploaded_faces:
            response["message"] = "Khong detect duoc mat tren anh mat vua upload."
        else:
            best = self.best_face_pair(
                passport_faces,
                uploaded_faces,
                left_image=passport_payload.image,
                right_image=uploaded_payload.image,
            )
            if best is not None:
                score, passport_face_row, uploaded_face_row, passport_meta, uploaded_meta = best
                response["passport_face"]["face_bbox"] = face_bbox(passport_face_row)
                response["passport_face"]["face_confidence"] = face_confidence(passport_face_row)
                response["uploaded_face"]["face_bbox"] = face_bbox(uploaded_face_row)
                response["uploaded_face"]["face_confidence"] = face_confidence(uploaded_face_row)
                response["passport_face"]["aligned_face_content_type"] = passport_meta["aligned_face_content_type"]
                response["passport_face"]["aligned_face_base64"] = passport_meta["aligned_face_base64"]
                response["uploaded_face"]["aligned_face_content_type"] = uploaded_meta["aligned_face_content_type"]
                response["uploaded_face"]["aligned_face_base64"] = uploaded_meta["aligned_face_base64"]
                response["performance"]["passport_align_embed_duration_ms"] = passport_meta["align_and_embed_duration_ms"]
                response["performance"]["uploaded_align_embed_duration_ms"] = uploaded_meta["align_and_embed_duration_ms"]
                response["performance"]["match_duration_ms"] = passport_meta.get("match_duration_ms", 0.0)
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

    def best_face_pair(
        self,
        left_faces: list[Any],
        right_faces: list[Any],
        *,
        left_image: np.ndarray | None = None,
        right_image: np.ndarray | None = None,
    ) -> tuple[float, Any, Any, dict[str, Any], dict[str, Any]] | None:
        best: tuple[float, Any, Any, dict[str, Any], dict[str, Any]] | None = None
        for left_face in left_faces:
            left_feature, left_meta = self.extract_embedding(left_image, left_face)
            for right_face in right_faces:
                right_feature, right_meta = self.extract_embedding(right_image, right_face)
                match_started = perf_counter()
                score = self.match_embeddings(left_feature, right_feature)
                left_meta = {**left_meta, "match_duration_ms": round((perf_counter() - match_started) * 1000, 2)}
                if best is None or score > best[0]:
                    best = (score, left_face, right_face, left_meta, right_meta)
        return best


def load_insightface() -> tuple[Any, Any]:
    try:
        from insightface.app import FaceAnalysis
        from insightface.utils import face_align
    except ImportError as exc:
        raise ImportError(
            "InsightFace is required for face match. Run `pip install -r requirements.txt` "
            "and make sure insightface/onnxruntime are installed."
        ) from exc
    return FaceAnalysis, face_align


def parse_providers(value: str) -> list[str]:
    providers = [item.strip() for item in str(value or "").split(",") if item.strip()]
    return providers or ["CPUExecutionProvider"]


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def select_primary_face(faces: list[Any]) -> Any | None:
    if not faces:
        return None
    return max(faces, key=lambda face: face_confidence(face) * max(1.0, face_area(face)))


def face_bbox(face_row: Any) -> dict[str, float]:
    if hasattr(face_row, "bbox"):
        left, top, right, bottom = [float(value) for value in face_row.bbox[:4]]
        return {
            "left": round(left, 2),
            "top": round(top, 2),
            "width": round(max(0.0, right - left), 2),
            "height": round(max(0.0, bottom - top), 2),
        }
    return {
        "left": round(float(face_row[0]), 2),
        "top": round(float(face_row[1]), 2),
        "width": round(float(face_row[2]), 2),
        "height": round(float(face_row[3]), 2),
    }


def face_confidence(face_row: Any) -> float:
    if hasattr(face_row, "det_score"):
        return round(float(face_row.det_score), 6)
    if len(face_row) <= 14:
        return 0.0
    return round(float(face_row[14]), 6)


def face_area(face_row: Any) -> float:
    box = face_bbox(face_row)
    return max(0.0, float(box["width"])) * max(0.0, float(box["height"]))


def normalized_embedding(face_row: Any) -> np.ndarray:
    feature = getattr(face_row, "normed_embedding", None)
    if feature is None:
        feature = getattr(face_row, "embedding", None)
    if feature is None:
        raise ValueError("InsightFace did not return a face embedding.")
    return normalize_vector(np.asarray(feature, dtype=np.float32).reshape(-1))


def normalize_vector(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise ValueError("Face embedding norm is zero.")
    return vector / norm


def aligned_face_image(image: np.ndarray | None, face_row: Any) -> np.ndarray:
    source_image = image if image is not None else getattr(face_row, "_image", None)
    if source_image is None:
        return np.zeros((112, 112, 3), dtype=np.uint8)

    _, face_align = load_insightface()
    landmarks = getattr(face_row, "kps", None)
    if landmarks is not None:
        try:
            return face_align.norm_crop(source_image, landmark=np.asarray(landmarks), image_size=112)
        except Exception:
            pass

    crop = crop_bbox(source_image, face_bbox(face_row), padding_ratio=0.2)
    if crop is not None:
        return crop
    return np.zeros((112, 112, 3), dtype=np.uint8)


def attach_source_image(faces: list[Any], image: np.ndarray) -> list[Any]:
    for face in faces:
        try:
            setattr(face, "_image", image)
        except Exception:
            pass
    return faces


def encode_image_to_base64(image: np.ndarray) -> tuple[str, str]:
    success, encoded_image = cv2.imencode(".jpg", image)
    if not success:
        return "", ""
    return "image/jpeg", base64.b64encode(encoded_image.tobytes()).decode("ascii")
