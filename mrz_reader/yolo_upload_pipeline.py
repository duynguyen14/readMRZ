from __future__ import annotations

import base64
import time
from typing import Any

import cv2
import numpy as np

from tools.generate_mrz_ocr_line2_dataset import (
    clamp_box,
    crop_line_images,
    deskew_crop,
    projection_bands,
)

from .custom_mrz_ocr import CustomMrzCtcRecognizer
from .document_orientation import PaddleDocumentOrientation
from .env_config import env_value, read_env_file
from .mrz import parse_mrz, result_to_dict
from .yolo_detector import YoloMrzDetector


def process_yolo_upload(
    image: np.ndarray,
    detector: YoloMrzDetector,
    orientation: PaddleDocumentOrientation,
    recognizer: CustomMrzCtcRecognizer,
) -> dict[str, Any]:
    pipeline_started = time.perf_counter()
    env = read_env_file()
    normalized_image, orientation_payload = orientation.normalize(image)
    payload = detector.detect(normalized_image)
    payload["orientation"] = orientation_payload
    payload["oriented_image"] = encoded_image(normalized_image)
    payload["mrz_raw_crop"] = None
    payload["mrz_crop"] = None
    payload["line_crops"] = []
    payload["mrz_ocr"] = empty_ocr_payload(recognizer)
    payload["cropper"] = {
        "name": "opencv-deskew-projection-line2",
        "expected_lines": int(env_value(env, "READMRZ_OCR_LINE2_EXPECTED_LINES", "2")),
        "bands_detected": 0,
        "line_count": 0,
        "deskew_angle": 0.0,
        "error": None,
    }
    payload["processing"] = {
        "orientation_ms": int(orientation_payload.get("latency_ms") or 0),
        "detector_ms": int(payload.get("detector_ms") or 0),
        "crop_ms": 0,
        "ocr_ms": 0,
        "parse_ms": 0,
        "pipeline_ms": 0,
    }

    best_box = payload.get("best_box")
    if not best_box:
        finish_pipeline_timing(payload, pipeline_started)
        return payload

    try:
        crop_started = time.perf_counter()
        height, width = normalized_image.shape[:2]
        padding_ratio = float(env_value(env, "READMRZ_OCR_LINE2_MRZ_PADDING_RATIO", "0.08"))
        x1, y1, x2, y2 = clamp_box(best_box["bbox_xyxy"], width, height, padding_ratio)
        raw_crop = normalized_image[y1:y2, x1:x2].copy()
        if raw_crop.size == 0:
            raise ValueError("MRZ crop is empty")

        deskewed_crop, binary, deskew_angle = deskew_crop(raw_crop, env)
        bands = projection_bands(binary, env)
        lines = crop_line_images(deskewed_crop, bands, env)
        payload["processing"]["crop_ms"] = int((time.perf_counter() - crop_started) * 1000)

        payload["mrz_raw_crop"] = {
            **encoded_image(raw_crop),
            "bbox_xyxy": [x1, y1, x2, y2],
        }
        payload["mrz_crop"] = encoded_image(deskewed_crop)
        line_payloads = [
            {
                **encoded_image(line["image"]),
                "line_index": index,
                "bbox_crop": line["bbox_crop"],
                "projection_score": round(float(line["projection_score"]), 4),
            }
            for index, line in enumerate(lines, start=1)
        ]
        payload["line_crops"] = line_payloads
        payload["cropper"].update(
            {
                "bands_detected": len(bands),
                "line_count": len(lines),
                "deskew_angle": round(float(deskew_angle), 4),
            }
        )

        if lines and recognizer.enabled:
            try:
                ocr_results, ocr_latency_ms = recognizer.recognize(
                    [line["image"] for line in lines]
                )
                payload["processing"]["ocr_ms"] = ocr_latency_ms
                for line_payload, ocr_result in zip(line_payloads, ocr_results, strict=False):
                    line_payload.update(ocr_result)
                payload["mrz_ocr"] = build_ocr_parse_payload(
                    ocr_results,
                    recognizer,
                    detector_confidence=float(best_box.get("confidence") or 0.0),
                    ocr_latency_ms=ocr_latency_ms,
                )
                payload["processing"]["parse_ms"] = payload["mrz_ocr"]["parse_latency_ms"]
            except Exception as exc:
                payload["mrz_ocr"] = empty_ocr_payload(recognizer, str(exc))
        elif lines:
            for line_payload in line_payloads:
                line_payload.update(
                    {
                        "ocr_text": "",
                        "ocr_normalized_text": "",
                        "ocr_confidence": 0.0,
                        "ocr_accepted": False,
                        "ocr_latency_ms": 0.0,
                        "ocr_error": "Custom MRZ OCR is disabled",
                    }
                )
    except Exception as exc:
        payload["cropper"]["error"] = str(exc)

    finish_pipeline_timing(payload, pipeline_started)
    return payload


def encoded_image(image: np.ndarray) -> dict[str, Any]:
    success, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not success:
        raise ValueError("Cannot encode output image")
    height, width = image.shape[:2]
    return {
        "content_type": "image/jpeg",
        "image_base64": base64.b64encode(buffer.tobytes()).decode("ascii"),
        "width": width,
        "height": height,
    }


def build_ocr_parse_payload(
    ocr_results: list[dict[str, Any]],
    recognizer: CustomMrzCtcRecognizer,
    *,
    detector_confidence: float,
    ocr_latency_ms: int,
) -> dict[str, Any]:
    accepted_results = [result for result in ocr_results if result.get("ocr_accepted")]
    raw_lines = [str(result.get("ocr_normalized_text") or "") for result in accepted_results]
    scores = [float(result.get("ocr_confidence") or 0.0) for result in accepted_results]
    average_confidence = sum(scores) / len(scores) if scores else 0.0

    parse_started = time.perf_counter()
    parsed = parse_mrz(raw_lines)
    parse_latency_ms = int((time.perf_counter() - parse_started) * 1000)
    parsed_payload = result_to_dict(
        parsed,
        raw_lines=raw_lines,
        ocr_score=average_confidence,
        detector_score=detector_confidence,
        latency_ms=ocr_latency_ms + parse_latency_ms,
        detector_latency_ms=0,
        ocr_latency_ms=ocr_latency_ms,
        parse_latency_ms=parse_latency_ms,
        candidates_evaluated=len(raw_lines),
        ocr_passes=1,
    )
    return {
        **recognizer.summary(),
        **parsed_payload,
        "raw_lines": raw_lines,
        "average_confidence": round(average_confidence, 6),
        "parse_latency_ms": parse_latency_ms,
        "error": None if parsed is not None else "OCR lines could not be parsed as MRZ",
    }


def empty_ocr_payload(
    recognizer: CustomMrzCtcRecognizer,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        **recognizer.summary(),
        "found": False,
        "raw_lines": [],
        "mrz_raw": [],
        "document_type": None,
        "confidence": 0.0,
        "average_confidence": 0.0,
        "ocr_confidence": 0.0,
        "detector_confidence": 0.0,
        "checksum_pass": False,
        "latency_ms": 0,
        "ocr_latency_ms": 0,
        "parse_latency_ms": 0,
        "fields": {},
        "checks": [],
        "error": error,
    }


def finish_pipeline_timing(payload: dict[str, Any], started: float) -> None:
    payload["processing"]["pipeline_ms"] = int((time.perf_counter() - started) * 1000)
