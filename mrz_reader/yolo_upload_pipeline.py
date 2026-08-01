from __future__ import annotations

import base64
import time
from typing import Any

import cv2
import numpy as np

from tools.generate_mrz_ocr_line2_dataset import (
    binarize_for_text,
    clamp_box,
    crop_line_images,
    deskew_crop,
    projection_bands,
    rotate_image,
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
    *,
    include_images: bool = True,
) -> dict[str, Any]:
    pipeline_started = time.perf_counter()
    env = read_env_file()
    normalized_image, orientation_payload = orientation.normalize(image)
    payload = detector.detect(normalized_image)
    payload["orientation"] = orientation_payload
    if include_images:
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
    payload["ocr_fallback"] = {
        "enabled": env_bool_value(env, "READMRZ_OCR_FALLBACK_ENABLED", True),
        "attempted": False,
        "used": False,
        "trigger": None,
        "selected_variant": "fast-projection",
        "candidates_evaluated": 0,
        "text_corrections": 0,
        "latency_ms": 0,
        "error": None,
    }
    payload["processing"] = {
        "orientation_ms": int(orientation_payload.get("latency_ms") or 0),
        "detector_ms": int(payload.get("detector_ms") or 0),
        "crop_ms": 0,
        "ocr_ms": 0,
        "parse_ms": 0,
        "fallback_ms": 0,
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

        payload["cropper"].update(
            {
                "bands_detected": len(bands),
                "line_count": len(lines),
                "deskew_angle": round(float(deskew_angle), 4),
            }
        )

        if recognizer.enabled:
            selected_candidate, fallback_payload, total_ocr_ms, total_parse_ms = run_ocr_with_fallback(
                raw_crop=raw_crop,
                deskewed_crop=deskewed_crop,
                deskew_angle=deskew_angle,
                initial_lines=lines,
                recognizer=recognizer,
                detector_confidence=float(best_box.get("confidence") or 0.0),
                env=env,
            )
            payload["ocr_fallback"] = fallback_payload
            payload["processing"]["ocr_ms"] = total_ocr_ms
            payload["processing"]["parse_ms"] = total_parse_ms
            payload["processing"]["fallback_ms"] = fallback_payload["latency_ms"]

            if selected_candidate is not None:
                selected_crop = selected_candidate["crop"]
                selected_lines = selected_candidate["lines"]
                selected_results = selected_candidate["ocr_results"]
                payload["mrz_ocr"] = selected_candidate["ocr_payload"]
                payload["mrz_ocr"]["ocr_latency_ms"] = total_ocr_ms
                payload["mrz_ocr"]["latency_ms"] = total_ocr_ms + total_parse_ms
                payload["line_crops"] = build_line_payloads(
                    selected_lines,
                    selected_results,
                    include_images=include_images,
                )
                if include_images:
                    payload["mrz_crop"] = encoded_image(selected_crop)
        else:
            disabled_results = [disabled_ocr_result() for _ in lines]
            payload["line_crops"] = build_line_payloads(
                lines,
                disabled_results,
                include_images=include_images,
            )

        if include_images:
            payload["mrz_raw_crop"] = {
                **encoded_image(raw_crop),
                "bbox_xyxy": [x1, y1, x2, y2],
            }
            if payload.get("mrz_crop") is None:
                payload["mrz_crop"] = encoded_image(deskewed_crop)
    except Exception as exc:
        payload["cropper"]["error"] = str(exc)

    finish_pipeline_timing(payload, pipeline_started)
    return payload


def run_ocr_with_fallback(
    *,
    raw_crop: np.ndarray,
    deskewed_crop: np.ndarray,
    deskew_angle: float,
    initial_lines: list[dict[str, Any]],
    recognizer: CustomMrzCtcRecognizer,
    detector_confidence: float,
    env: dict[str, str],
) -> tuple[dict[str, Any] | None, dict[str, Any], int, int]:
    fallback_enabled = env_bool_value(env, "READMRZ_OCR_FALLBACK_ENABLED", True)
    selected: dict[str, Any] | None = None
    total_ocr_ms = 0
    total_parse_ms = 0
    candidates_evaluated = 0

    if initial_lines:
        initial_results, initial_ocr_ms = recognizer.recognize(
            [line["image"] for line in initial_lines]
        )
        total_ocr_ms += initial_ocr_ms
        selected = build_ocr_candidate(
            "fast-projection",
            deskewed_crop,
            initial_lines,
            initial_results,
            recognizer,
            detector_confidence=detector_confidence,
            ocr_latency_ms=initial_ocr_ms,
        )
        total_parse_ms += int(selected["ocr_payload"].get("parse_latency_ms") or 0)

    trigger = fallback_trigger(selected)
    fallback_payload = {
        "enabled": fallback_enabled,
        "attempted": False,
        "used": False,
        "trigger": trigger,
        "selected_variant": selected["name"] if selected else "none",
        "candidates_evaluated": 0,
        "text_corrections": 0,
        "latency_ms": 0,
        "error": None,
    }
    if not fallback_enabled or trigger is None:
        return selected, fallback_payload, total_ocr_ms, total_parse_ms

    fallback_started = time.perf_counter()
    fallback_payload["attempted"] = True

    if selected is not None and env_bool_value(
        env, "READMRZ_OCR_FALLBACK_TEXT_CORRECTION", True
    ):
        selected, correction_parse_ms, correction_evaluated = apply_checksum_text_correction(
            selected,
            recognizer,
            detector_confidence=detector_confidence,
            variant_name="checksum-text-correction",
        )
        total_parse_ms += correction_parse_ms
        candidates_evaluated += correction_evaluated

    if selected is None or not selected["ocr_payload"].get("checksum_pass"):
        specs = build_fallback_line_candidates(
            raw_crop,
            deskewed_crop,
            deskew_angle,
            env,
        )
        flat_images = [
            line["image"]
            for spec in specs
            for line in spec["lines"]
        ]
        if flat_images:
            try:
                flat_results, fallback_ocr_ms = recognizer.recognize(flat_images)
                total_ocr_ms += fallback_ocr_ms
                cursor = 0
                for spec in specs:
                    line_count = len(spec["lines"])
                    candidate_results = flat_results[cursor : cursor + line_count]
                    cursor += line_count
                    if len(candidate_results) != line_count:
                        continue
                    candidate = build_ocr_candidate(
                        spec["name"],
                        spec["crop"],
                        spec["lines"],
                        candidate_results,
                        recognizer,
                        detector_confidence=detector_confidence,
                        ocr_latency_ms=fallback_ocr_ms,
                    )
                    total_parse_ms += int(candidate["ocr_payload"].get("parse_latency_ms") or 0)
                    candidates_evaluated += 1
                    if env_bool_value(
                        env, "READMRZ_OCR_FALLBACK_TEXT_CORRECTION", True
                    ):
                        candidate, correction_parse_ms, correction_evaluated = (
                            apply_checksum_text_correction(
                                candidate,
                                recognizer,
                                detector_confidence=detector_confidence,
                                variant_name=f"{spec['name']}-text-correction",
                            )
                        )
                        total_parse_ms += correction_parse_ms
                        candidates_evaluated += correction_evaluated
                    if selected is None or candidate["score"] > selected["score"]:
                        selected = candidate
            except Exception as exc:
                fallback_payload["error"] = f"Candidate OCR failed: {exc}"

    if (
        selected is not None
        and not selected["ocr_payload"].get("checksum_pass")
        and env_bool_value(env, "READMRZ_OCR_FALLBACK_CLAHE", True)
        and selected["lines"]
    ):
        enhanced_lines = enhance_candidate_lines(selected["lines"])
        try:
            enhanced_results, enhanced_ocr_ms = recognizer.recognize(
                [line["image"] for line in enhanced_lines]
            )
            total_ocr_ms += enhanced_ocr_ms
            enhanced = build_ocr_candidate(
                f"{selected['name']}-clahe",
                selected["crop"],
                enhanced_lines,
                enhanced_results,
                recognizer,
                detector_confidence=detector_confidence,
                ocr_latency_ms=enhanced_ocr_ms,
            )
            total_parse_ms += int(enhanced["ocr_payload"].get("parse_latency_ms") or 0)
            candidates_evaluated += 1
            if env_bool_value(
                env, "READMRZ_OCR_FALLBACK_TEXT_CORRECTION", True
            ):
                enhanced, correction_parse_ms, correction_evaluated = (
                    apply_checksum_text_correction(
                        enhanced,
                        recognizer,
                        detector_confidence=detector_confidence,
                        variant_name=f"{selected['name']}-clahe-text-correction",
                    )
                )
                total_parse_ms += correction_parse_ms
                candidates_evaluated += correction_evaluated
            if enhanced["score"] > selected["score"]:
                selected = enhanced
        except Exception as exc:
            if fallback_payload["error"] is None:
                fallback_payload["error"] = f"CLAHE OCR failed: {exc}"

    fallback_payload.update(
        {
            "used": bool(selected and selected["name"] != "fast-projection"),
            "selected_variant": selected["name"] if selected else "none",
            "candidates_evaluated": candidates_evaluated,
            "text_corrections": int(selected.get("text_corrections") or 0) if selected else 0,
            "latency_ms": int((time.perf_counter() - fallback_started) * 1000),
        }
    )
    return selected, fallback_payload, total_ocr_ms, total_parse_ms


def build_fallback_line_candidates(
    raw_crop: np.ndarray,
    deskewed_crop: np.ndarray,
    deskew_angle: float,
    env: dict[str, str],
) -> list[dict[str, Any]]:
    max_candidates = max(
        1, int(env_value(env, "READMRZ_OCR_FALLBACK_MAX_CANDIDATES", "4"))
    )
    specs: list[dict[str, Any]] = []

    add_line_candidate(
        specs,
        "fixed-split",
        deskewed_crop,
        fixed_split_lines(deskewed_crop, env),
        max_candidates,
    )
    add_line_candidate(
        specs,
        "no-deskew",
        raw_crop,
        extract_projection_or_fixed_lines(raw_crop, env),
        max_candidates,
    )

    offsets = parse_float_list(
        env_value(env, "READMRZ_OCR_FALLBACK_DESKEW_OFFSETS", "-2,2")
    )
    for offset in offsets:
        if len(specs) >= max_candidates or abs(offset) < 0.01:
            continue
        adjusted = rotate_image(
            raw_crop,
            deskew_angle + offset,
            border_value=(255, 255, 255),
        )
        sign = "plus" if offset > 0 else "minus"
        add_line_candidate(
            specs,
            f"deskew-{sign}-{abs(offset):g}",
            adjusted,
            extract_projection_or_fixed_lines(adjusted, env),
            max_candidates,
        )
    return specs


def add_line_candidate(
    specs: list[dict[str, Any]],
    name: str,
    crop: np.ndarray,
    lines: list[dict[str, Any]],
    limit: int,
) -> None:
    if len(specs) >= limit or len(lines) < 2:
        return
    specs.append({"name": name, "crop": crop, "lines": lines[:2]})


def extract_projection_or_fixed_lines(
    crop: np.ndarray,
    env: dict[str, str],
) -> list[dict[str, Any]]:
    bands = projection_bands(binarize_for_text(crop), env)
    lines = crop_line_images(crop, bands, env)
    return lines if len(lines) >= 2 else fixed_split_lines(crop, env)


def fixed_split_lines(
    crop: np.ndarray,
    env: dict[str, str],
) -> list[dict[str, Any]]:
    height, width = crop.shape[:2]
    if height < 4 or width < 4:
        return []

    binary = binarize_for_text(crop)
    projection = binary.sum(axis=1).astype(np.float32)
    threshold = max(1.0, float(projection.max()) * 0.03) if projection.size else 1.0
    active = np.where(projection >= threshold)[0]
    top = int(active.min()) if active.size else 0
    bottom = int(active.max()) + 1 if active.size else height
    active_height = max(2, bottom - top)
    center = int(round((top + bottom) / 2.0))
    search_radius = max(2, int(round(active_height * 0.16)))
    search_top = max(top + 1, center - search_radius)
    search_bottom = min(bottom - 1, center + search_radius)
    separator = center
    if search_bottom > search_top:
        separator = search_top + int(np.argmin(projection[search_top:search_bottom]))

    overlap_ratio = float(
        env_value(env, "READMRZ_OCR_FALLBACK_LINE_OVERLAP_RATIO", "0.10")
    )
    overlap = max(2, int(round(active_height * overlap_ratio)))
    pad = max(
        2,
        int(
            round(
                height
                * float(
                    env_value(env, "READMRZ_OCR_FALLBACK_LINE_PADDING_RATIO", "0.05")
                )
            )
        ),
    )
    bounds = [
        (max(0, top - pad), min(height, separator + overlap)),
        (max(0, separator - overlap), min(height, bottom + pad)),
    ]
    lines: list[dict[str, Any]] = []
    for line_top, line_bottom in bounds:
        if line_bottom <= line_top:
            continue
        image = crop[line_top:line_bottom, :].copy()
        if image.size == 0:
            continue
        lines.append(
            {
                "image": image,
                "width": width,
                "height": line_bottom - line_top,
                "bbox_crop": [0.0, float(line_top), float(width), float(line_bottom)],
                "projection_score": float(projection[line_top:line_bottom].mean()),
            }
        )
    return lines


def enhance_candidate_lines(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced: list[dict[str, Any]] = []
    for line in lines:
        gray = cv2.cvtColor(line["image"], cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
        enhanced.append({**line, "image": image})
    return enhanced


def build_ocr_candidate(
    name: str,
    crop: np.ndarray,
    lines: list[dict[str, Any]],
    ocr_results: list[dict[str, Any]],
    recognizer: CustomMrzCtcRecognizer,
    *,
    detector_confidence: float,
    ocr_latency_ms: int,
) -> dict[str, Any]:
    ocr_payload = build_ocr_parse_payload(
        ocr_results,
        recognizer,
        detector_confidence=detector_confidence,
        ocr_latency_ms=ocr_latency_ms,
    )
    return {
        "name": name,
        "crop": crop,
        "lines": lines,
        "ocr_results": ocr_results,
        "ocr_payload": ocr_payload,
        "score": ocr_candidate_score(ocr_payload),
        "text_corrections": 0,
    }


def apply_checksum_text_correction(
    candidate: dict[str, Any],
    recognizer: CustomMrzCtcRecognizer,
    *,
    detector_confidence: float,
    variant_name: str,
) -> tuple[dict[str, Any], int, int]:
    corrected_results, correction_count = correct_numeric_ocr_results(
        candidate["ocr_results"]
    )
    if correction_count <= 0:
        return candidate, 0, 0

    corrected = build_ocr_candidate(
        variant_name,
        candidate["crop"],
        candidate["lines"],
        corrected_results,
        recognizer,
        detector_confidence=detector_confidence,
        ocr_latency_ms=0,
    )
    parse_ms = int(corrected["ocr_payload"].get("parse_latency_ms") or 0)
    if corrected["ocr_payload"].get("checksum_pass") and corrected["score"] > candidate["score"]:
        corrected["text_corrections"] = correction_count
        return corrected, parse_ms, 1
    return candidate, parse_ms, 1


def ocr_candidate_score(payload: dict[str, Any]) -> tuple[int, int, int, int, float]:
    checks = payload.get("checks") or []
    checks_passed = sum(1 for item in checks if item.get("passed"))
    lines = payload.get("raw_lines") or payload.get("mrz_raw") or []
    target_length = 36 if lines and max(len(line) for line in lines) <= 38 else 44
    length_error = sum(abs(len(line) - target_length) for line in lines[:2])
    if len(lines) < 2:
        length_error += target_length * (2 - len(lines))
    return (
        int(bool(payload.get("checksum_pass"))),
        checks_passed,
        int(bool(payload.get("found"))),
        -length_error,
        float(payload.get("average_confidence") or 0.0),
    )


def fallback_trigger(candidate: dict[str, Any] | None) -> str | None:
    if candidate is None or len(candidate.get("lines") or []) < 2:
        return "missing-lines"
    payload = candidate["ocr_payload"]
    if not payload.get("found"):
        return "parse-failed"
    if not payload.get("checksum_pass"):
        return "checksum-failed"
    return None


def correct_numeric_ocr_results(
    ocr_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    if len(ocr_results) < 2:
        return ocr_results, 0
    texts = [str(result.get("ocr_normalized_text") or "") for result in ocr_results]
    if len(texts[1]) < 28:
        return ocr_results, 0

    replacements = {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "G": "6",
        "B": "8",
    }
    numeric_positions = set(range(13, 20)) | set(range(21, 28)) | {9, 43}
    line_two = list(texts[1])
    corrections = 0
    for index in numeric_positions:
        if index >= len(line_two):
            continue
        replacement = replacements.get(line_two[index])
        if replacement is not None:
            line_two[index] = replacement
            corrections += 1
    if corrections == 0:
        return ocr_results, 0

    corrected = [dict(result) for result in ocr_results]
    corrected[1]["ocr_normalized_text"] = "".join(line_two)
    corrected[1]["ocr_corrected"] = True
    return corrected, corrections


def build_line_payloads(
    lines: list[dict[str, Any]],
    ocr_results: list[dict[str, Any]],
    *,
    include_images: bool,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        payload = {
            "line_index": index,
            "bbox_crop": line["bbox_crop"],
            "projection_score": round(float(line["projection_score"]), 4),
        }
        if include_images:
            payload.update(encoded_image(line["image"]))
        if index <= len(ocr_results):
            payload.update(ocr_results[index - 1])
        payloads.append(payload)
    return payloads


def disabled_ocr_result() -> dict[str, Any]:
    return {
        "ocr_text": "",
        "ocr_normalized_text": "",
        "ocr_confidence": 0.0,
        "ocr_accepted": False,
        "ocr_latency_ms": 0.0,
        "ocr_error": "Custom MRZ OCR is disabled",
    }


def parse_float_list(value: str) -> list[float]:
    values: list[float] = []
    for item in value.split(","):
        try:
            values.append(float(item.strip()))
        except ValueError:
            continue
    return values


def env_bool_value(env: dict[str, str], key: str, default: bool) -> bool:
    raw_value = env_value(env, key, "true" if default else "false").strip().lower()
    return raw_value in {"1", "true", "yes", "on"}


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


def compact_yolo_read_payload(payload: dict[str, Any]) -> dict[str, Any]:
    ocr_payload = payload.get("mrz_ocr") or {}
    cropper_payload = payload.get("cropper") or {}
    lines = [
        {
            "line_index": int(line.get("line_index") or index),
            "text": str(line.get("ocr_normalized_text") or line.get("ocr_text") or ""),
            "confidence": round(float(line.get("ocr_confidence") or 0.0), 6),
            "accepted": bool(line.get("ocr_accepted")),
        }
        for index, line in enumerate(payload.get("line_crops") or [], start=1)
    ]

    error = ocr_payload.get("error") or cropper_payload.get("error")
    if not error and not payload.get("found"):
        error = "MRZ region was not detected"

    return {
        "found": bool(ocr_payload.get("found")),
        "document_type": ocr_payload.get("document_type"),
        "checksum_pass": bool(ocr_payload.get("checksum_pass")),
        "confidence": round(float(ocr_payload.get("confidence") or 0.0), 6),
        "detector_confidence": round(float(ocr_payload.get("detector_confidence") or 0.0), 6),
        "average_ocr_confidence": round(float(ocr_payload.get("average_confidence") or 0.0), 6),
        "lines": lines,
        "fields": ocr_payload.get("fields") or {},
        "checks": ocr_payload.get("checks") or [],
        "processing": payload.get("processing") or {},
        "fallback": payload.get("ocr_fallback") or {},
        "error": error,
    }
