from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SUPPORTED_UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SUPPORTED_IMAGE_DATA_URI_TYPES = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/webp": ".webp",
}


@dataclass
class ImagePayload:
    file_name: str
    content_type: str
    base64_value: str
    file_bytes: bytes
    image: np.ndarray
    width: int
    height: int


@dataclass
class WorkingImage:
    image: np.ndarray
    width: int
    height: int
    scale_x_to_original: float
    scale_y_to_original: float
    resized: bool


def decode_base64_image_payload(base64_payload: str, file_name: str | None = None) -> ImagePayload:
    normalized_payload = str(base64_payload or "").strip()
    if not normalized_payload:
        raise ValueError("Base64 image payload is empty.")

    resolved_file_name = Path((file_name or "").strip() or "image.jpg").name
    payload_to_decode = normalized_payload
    content_type = content_type_from_file_name(resolved_file_name)

    if normalized_payload.startswith("data:"):
        header, separator, encoded_payload = normalized_payload.partition(",")
        if not separator:
            raise ValueError("Invalid base64 data URI.")
        if ";base64" not in header.lower():
            raise ValueError("Base64 data URI must include ';base64'.")

        mime_type = header[5:].split(";", 1)[0].strip().lower()
        payload_to_decode = encoded_payload.strip()
        content_type = mime_type or content_type

        if Path(resolved_file_name).suffix.lower() not in SUPPORTED_UPLOAD_EXTENSIONS:
            resolved_extension = SUPPORTED_IMAGE_DATA_URI_TYPES.get(mime_type, ".jpg")
            resolved_file_name = f"{Path(resolved_file_name).stem}{resolved_extension}"

    try:
        file_bytes = base64.b64decode(payload_to_decode, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Invalid base64 image payload.") from exc

    if not file_bytes:
        raise ValueError("Decoded image payload is empty.")

    image = decode_image_bytes(file_bytes)
    height, width = image.shape[:2]
    if Path(resolved_file_name).suffix.lower() not in SUPPORTED_UPLOAD_EXTENSIONS:
        resolved_file_name = f"{Path(resolved_file_name).stem}{extension_from_image_bytes(file_bytes)}"
    if content_type == "application/octet-stream":
        content_type = content_type_from_file_name(resolved_file_name)

    return ImagePayload(
        file_name=Path(resolved_file_name).name,
        content_type=content_type,
        base64_value=base64.b64encode(file_bytes).decode("ascii"),
        file_bytes=file_bytes,
        image=image,
        width=int(width),
        height=int(height),
    )


def decode_image_bytes(file_bytes: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image.")
    return image


def content_type_from_file_name(file_name: str) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".bmp":
        return "image/bmp"
    if suffix in {".tif", ".tiff"}:
        return "image/tiff"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def extension_from_image_bytes(file_bytes: bytes) -> str:
    if file_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if file_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if file_bytes.startswith(b"BM"):
        return ".bmp"
    if file_bytes.startswith((b"II*\x00", b"MM\x00*")):
        return ".tiff"
    if len(file_bytes) >= 12 and file_bytes[:4] == b"RIFF" and file_bytes[8:12] == b"WEBP":
        return ".webp"
    return ".jpg"


def resize_for_inference(image: np.ndarray, max_side: int) -> WorkingImage:
    height, width = image.shape[:2]
    if max_side <= 0 or max(width, height) <= max_side:
        return WorkingImage(
            image=image,
            width=int(width),
            height=int(height),
            scale_x_to_original=1.0,
            scale_y_to_original=1.0,
            resized=False,
        )

    scale = max_side / float(max(width, height))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    return WorkingImage(
        image=resized,
        width=resized_width,
        height=resized_height,
        scale_x_to_original=width / float(resized_width),
        scale_y_to_original=height / float(resized_height),
        resized=True,
    )


def map_bbox_to_original(
    bbox: dict[str, Any],
    working: WorkingImage,
    original_width: int,
    original_height: int,
) -> dict[str, float]:
    left = float(bbox.get("left") or 0.0) * working.scale_x_to_original
    top = float(bbox.get("top") or 0.0) * working.scale_y_to_original
    width = float(bbox.get("width") or 0.0) * working.scale_x_to_original
    height = float(bbox.get("height") or 0.0) * working.scale_y_to_original
    left = clamp(left, 0.0, float(original_width))
    top = clamp(top, 0.0, float(original_height))
    width = clamp(width, 0.0, float(original_width) - left)
    height = clamp(height, 0.0, float(original_height) - top)
    return {
        "left": round(left, 2),
        "top": round(top, 2),
        "width": round(width, 2),
        "height": round(height, 2),
    }


def crop_bbox(image: np.ndarray, bbox: dict[str, Any], padding_ratio: float = 0.0) -> np.ndarray | None:
    height, width = image.shape[:2]
    left = float(bbox.get("left") or 0.0)
    top = float(bbox.get("top") or 0.0)
    box_width = float(bbox.get("width") or 0.0)
    box_height = float(bbox.get("height") or 0.0)
    if box_width <= 1 or box_height <= 1:
        return None

    pad_x = box_width * padding_ratio
    pad_y = box_height * padding_ratio
    x1 = int(round(clamp(left - pad_x, 0.0, float(width))))
    y1 = int(round(clamp(top - pad_y, 0.0, float(height))))
    x2 = int(round(clamp(left + box_width + pad_x, 0.0, float(width))))
    y2 = int(round(clamp(top + box_height + pad_y, 0.0, float(height))))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2].copy()
    return crop if crop.size else None


def encode_jpeg_base64(image: np.ndarray, quality: int = 94) -> tuple[str, str]:
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not success:
        raise ValueError("Cannot encode image.")
    return "image/jpeg", base64.b64encode(encoded.tobytes()).decode("ascii")


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
