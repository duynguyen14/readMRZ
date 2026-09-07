from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np

from .custom_mrz_ocr import CustomMrzCtcRecognizer
from .document_orientation import PaddleDocumentOrientation
from .env_config import env_value, read_env_file
from .face_match import FaceMatchService, face_bbox, face_confidence
from .file_type_classifier import FileTypeClassifier
from .image_payload import (
    ImagePayload,
    WorkingImage,
    crop_bbox,
    decode_base64_image_payload,
    encode_jpeg_base64,
    map_bbox_to_original,
    resize_for_inference,
)
from .yolo_detector import YoloMrzDetector
from .yolo_upload_pipeline import compact_yolo_read_payload, process_yolo_upload


@dataclass
class InputItem:
    file_name: str
    base64_value: str


@dataclass
class DetectedFace:
    embedding: np.ndarray
    bbox: dict[str, float]
    confidence: float
    aligned_face_content_type: str
    aligned_face_base64: str


@dataclass
class FaceCandidate:
    payload: ImagePayload
    working: WorkingImage
    response: dict[str, Any]
    detections: list[DetectedFace]


@dataclass
class PassportCandidate:
    payload: ImagePayload
    working: WorkingImage
    response: dict[str, Any]
    detections: list[DetectedFace]


def process_batch(
    items: list[dict[str, Any]],
    *,
    classifier: FileTypeClassifier,
    face_matcher: FaceMatchService,
    yolo_detector: YoloMrzDetector,
    orientation: PaddleDocumentOrientation,
    recognizer: CustomMrzCtcRecognizer,
    logger: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    env = read_env_file()
    max_items = max(1, int(env_value(env, "READMRZ_BATCH_MAX_ITEMS", "20")))
    if len(items) > max_items:
        raise ValueError(f"Too many items. Max allowed is {max_items}.")

    max_side = max(0, int(env_value(env, "READMRZ_INFERENCE_MAX_IMAGE_SIDE", "1600")))
    log_batch(
        logger,
        "PASSPORT_FACE_MATCH_BATCH detail_start "
        f"items={len(items)} max_side={max_side} "
        f"match_threshold={face_matcher.match_threshold} "
        f"review_threshold={face_matcher.review_threshold}",
    )
    faces: list[FaceCandidate] = []
    passports: list[PassportCandidate] = []

    for item_index, raw_item in enumerate(items, start=1):
        input_item = normalize_input_item(raw_item)
        payload = decode_base64_image_payload(input_item.base64_value, input_item.file_name)
        working = resize_for_inference(payload.image, max_side)
        classification = classifier.predict(working.image)
        label = normalize_class_label(classification.get("label"))
        log_batch(
            logger,
            "PASSPORT_FACE_MATCH_BATCH item_classified "
            f"index={item_index} file={payload.file_name} "
            f"class={label or 'unknown'} "
            f"class_raw={classification.get('label') or ''} "
            f"class_conf={round(float(classification.get('confidence') or 0.0), 6)} "
            f"size={payload.width}x{payload.height} resized={working.resized}",
        )
        if label == "face":
            faces.append(build_face_candidate(payload, working, classification, face_matcher, logger))
        elif label == "passport":
            passports.append(
                build_passport_candidate(
                    payload,
                    working,
                    classification,
                    face_matcher,
                    yolo_detector,
                    orientation,
                    recognizer,
                    logger,
                )
            )
        else:
            log_batch(
                logger,
                "PASSPORT_FACE_MATCH_BATCH item_ignored "
                f"index={item_index} file={payload.file_name} class={label or 'unknown'}",
            )

    data = pair_candidates(faces, passports, face_matcher, logger)
    log_batch(
        logger,
        "PASSPORT_FACE_MATCH_BATCH detail_done "
        f"faces={len(faces)} passports={len(passports)} rows={len(data)}",
    )
    return {"data": data}


def normalize_class_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    if label in {"face", "portrait", "uploaded_face"}:
        return "face"
    if "face" in label:
        return "face"
    if label in {"passport", "passport_page"} or "passport" in label:
        return "passport"
    return label


def normalize_input_item(item: dict[str, Any]) -> InputItem:
    file_name = str(item.get("file_name") or item.get("filename") or item.get("name") or "image.jpg")
    base64_value = item.get("base64") or item.get("image_base64") or item.get("dataBase64")
    if not isinstance(base64_value, str) or not base64_value.strip():
        raise ValueError(f"base64 is required for item {file_name}")
    return InputItem(file_name=file_name, base64_value=base64_value)


def build_face_candidate(
    payload: ImagePayload,
    working: WorkingImage,
    classification: dict[str, Any],
    face_matcher: FaceMatchService,
    logger: Callable[[str], None] | None = None,
) -> FaceCandidate:
    detections, detect_meta = build_detected_faces(payload, working, face_matcher)
    primary = detections[0] if detections else None
    response = {
        "file_name": payload.file_name,
        "content_type": payload.content_type,
        "base64": payload.base64_value,
        "classification": build_classification_payload(classification),
        "image": {
            "width": payload.width,
            "height": payload.height,
            "resized_for_inference": working.resized,
        },
        "detected": primary is not None,
        "face_count": int(detect_meta["count"]),
        "face_bbox": primary.bbox if primary is not None else None,
        "face_confidence": primary.confidence if primary is not None else 0.0,
    }
    log_batch(
        logger,
        "PASSPORT_FACE_MATCH_BATCH face_ready "
        f"file={payload.file_name} detected={primary is not None} "
        f"face_count={int(detect_meta['count'])} "
        f"face_conf={response['face_confidence']} "
        f"embeddings={len(detections)} "
        f"detect_ms={detect_meta['duration_ms']}",
    )
    return FaceCandidate(payload=payload, working=working, response=response, detections=detections)


def build_passport_candidate(
    payload: ImagePayload,
    working: WorkingImage,
    classification: dict[str, Any],
    face_matcher: FaceMatchService,
    yolo_detector: YoloMrzDetector,
    orientation: PaddleDocumentOrientation,
    recognizer: CustomMrzCtcRecognizer,
    logger: Callable[[str], None] | None = None,
) -> PassportCandidate:
    started = perf_counter()
    mrz_payload = process_yolo_upload(
        working.image,
        yolo_detector,
        orientation,
        recognizer,
        include_images=False,
    )
    compact_mrz = compact_yolo_read_payload(mrz_payload)
    detections, detect_meta = build_detected_faces(payload, working, face_matcher, include_aligned=True)
    primary = detections[0] if detections else None

    face_base64 = ""
    face_content_type = ""
    if primary is not None:
        face_base64 = primary.aligned_face_base64
        face_content_type = primary.aligned_face_content_type

    response = {
        "file_name": payload.file_name,
        "content_type": payload.content_type,
        "base64": payload.base64_value,
        "classification": build_classification_payload(classification),
        "image": {
            "width": payload.width,
            "height": payload.height,
            "resized_for_inference": working.resized,
        },
        "face_base64": face_base64,
        "face_content_type": face_content_type,
        "face_bbox": primary.bbox if primary is not None else None,
        "face_confidence": primary.confidence if primary is not None else 0.0,
        "mrz": build_mrz_payload(compact_mrz),
        "parsed": build_parsed_payload(compact_mrz),
        "processing_ms": int((perf_counter() - started) * 1000),
    }
    log_batch(
        logger,
        "PASSPORT_FACE_MATCH_BATCH passport_ready "
        f"file={payload.file_name} face_detected={primary is not None} "
        f"face_count={int(detect_meta['count'])} "
        f"face_conf={response['face_confidence']} "
        f"embeddings={len(detections)} "
        f"mrz_found={response['mrz']['found']} "
        f"mrz_conf={response['mrz']['confidence']} "
        f"detect_ms={detect_meta['duration_ms']} "
        f"processing_ms={response['processing_ms']}",
    )
    return PassportCandidate(payload=payload, working=working, response=response, detections=detections)


def build_detected_faces(
    payload: ImagePayload,
    working: WorkingImage,
    face_matcher: FaceMatchService,
    *,
    include_aligned: bool = False,
) -> tuple[list[DetectedFace], dict[str, Any]]:
    env = read_env_file()
    max_detections = max(1, int(env_value(env, "READMRZ_FACE_MATCH_MAX_DETECTIONS_PER_IMAGE", "5")))
    face_rows, detect_meta = face_matcher.detect_faces(working.image)
    face_rows = sorted(face_rows, key=face_confidence, reverse=True)[:max_detections]

    detections: list[DetectedFace] = []
    for face_row in face_rows:
        embedding, face_meta = face_matcher.extract_embedding(working.image, face_row)
        aligned_base64 = str(face_meta.get("aligned_face_base64") or "")
        aligned_content_type = str(face_meta.get("aligned_face_content_type") or "")
        if include_aligned and not aligned_base64:
            crop = crop_bbox(working.image, face_bbox(face_row), padding_ratio=0.2)
            if crop is not None:
                aligned_content_type, aligned_base64 = encode_jpeg_base64(crop)

        detections.append(
            DetectedFace(
                embedding=embedding,
                bbox=map_bbox_to_original(face_bbox(face_row), working, payload.width, payload.height),
                confidence=face_confidence(face_row),
                aligned_face_content_type=aligned_content_type if include_aligned else "",
                aligned_face_base64=aligned_base64 if include_aligned else "",
            )
        )
    return detections, detect_meta


def build_classification_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": payload.get("label"),
        "confidence": round(float(payload.get("confidence") or 0.0), 6),
    }


def build_mrz_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "found": bool(payload.get("found")),
        "confidence": round(float(payload.get("confidence") or 0.0), 6),
        "lines": [
            {
                "text": str(line.get("text") or ""),
                "confidence": round(float(line.get("confidence") or 0.0), 6),
            }
            for line in payload.get("lines") or []
            if str(line.get("text") or "")
        ],
    }


def build_parsed_payload(payload: dict[str, Any]) -> dict[str, Any]:
    fields = payload.get("fields") or {}
    surname = clean_value(fields.get("surname"))
    given_names = clean_value(fields.get("given_names"))
    full_name = " ".join(part for part in (surname, given_names) if part) or None
    document_number = clean_value(fields.get("document_number"))
    return {
        "document_type": clean_value(payload.get("document_type")),
        "document_code": clean_value(fields.get("document_code")),
        "issuing_country": clean_value(fields.get("issuing_country")),
        "passport_no": document_number,
        "document_number": document_number,
        "surname": surname,
        "given_names": given_names,
        "full_name": full_name,
        "nationality": clean_value(fields.get("nationality")),
        "date_of_birth": clean_value(fields.get("birth_date")),
        "birth_date": clean_value(fields.get("birth_date")),
        "sex": clean_value(fields.get("sex")),
        "expiry_date": clean_value(fields.get("expiry_date")),
        "personal_number": clean_value(fields.get("optional_data")),
        "optional_data": clean_value(fields.get("optional_data")),
    }


def clean_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def pair_candidates(
    faces: list[FaceCandidate],
    passports: list[PassportCandidate],
    face_matcher: FaceMatchService,
    logger: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    pairs: list[tuple[float, int, int, int, int, dict[str, Any]]] = []
    for face_index, face in enumerate(faces):
        if not face.detections:
            log_batch(
                logger,
                "PASSPORT_FACE_MATCH_BATCH face_skip_no_embedding "
                f"file={face.payload.file_name}",
            )
            continue
        for face_detection_index, face_detection in enumerate(face.detections):
            for passport_index, passport in enumerate(passports):
                if not passport.detections:
                    log_batch(
                        logger,
                        "PASSPORT_FACE_MATCH_BATCH compare_skip_no_passport_embedding "
                        f"face={face.payload.file_name} passport={passport.payload.file_name}",
                    )
                    continue
                for passport_detection_index, passport_detection in enumerate(passport.detections):
                    score = face_matcher.match_embeddings(face_detection.embedding, passport_detection.embedding)
                    decision, matched, review_required = face_matcher.decision(score)
                    log_batch(
                        logger,
                        "PASSPORT_FACE_MATCH_BATCH compare "
                        f"face={face.payload.file_name} face_det={face_detection_index} "
                        f"face_conf={face_detection.confidence} "
                        f"passport={passport.payload.file_name} pass_det={passport_detection_index} "
                        f"pass_conf={passport_detection.confidence} "
                        f"score={round(score, 6)} decision={decision}",
                    )
                    if decision == "mismatch":
                        continue
                    pairs.append(
                        (
                            score,
                            face_index,
                            passport_index,
                            face_detection_index,
                            passport_detection_index,
                            {
                                "score": round(score, 6),
                                "decision": decision,
                                "matched": matched,
                                "review_required": review_required,
                            },
                        )
                    )

    pairs.sort(key=lambda item: item[0], reverse=True)
    used_faces: set[int] = set()
    used_passports: set[int] = set()
    data: list[dict[str, Any]] = []
    for _, face_index, passport_index, face_detection_index, passport_detection_index, match_payload in pairs:
        if face_index in used_faces or passport_index in used_passports:
            continue
        face = faces[face_index]
        passport = passports[passport_index]
        face_payload = response_with_face_detection(face.response, face.detections[face_detection_index])
        passport_payload = response_with_passport_detection(passport.response, passport.detections[passport_detection_index])
        passport_payload["match"] = match_payload
        data.append({"face": face_payload, "passport": passport_payload})
        log_batch(
            logger,
            "PASSPORT_FACE_MATCH_BATCH pair_selected "
            f"face={face.payload.file_name} face_det={face_detection_index} "
            f"passport={passport.payload.file_name} pass_det={passport_detection_index} "
            f"score={match_payload['score']} decision={match_payload['decision']}",
        )
        used_faces.add(face_index)
        used_passports.add(passport_index)

    for face_index, face in enumerate(faces):
        if face_index not in used_faces:
            data.append({"face": face.response, "passport": None})
            log_batch(
                logger,
                "PASSPORT_FACE_MATCH_BATCH unmatched_face "
                f"file={face.payload.file_name} detected={face.response.get('detected')} "
                f"face_conf={face.response.get('face_confidence')}",
            )

    for passport_index, passport in enumerate(passports):
        if passport_index not in used_passports:
            passport_payload = dict(passport.response)
            passport_payload["match"] = None
            data.append({"face": None, "passport": passport_payload})
            log_batch(
                logger,
                "PASSPORT_FACE_MATCH_BATCH unmatched_passport "
                f"file={passport.payload.file_name} "
                f"face_conf={passport.response.get('face_confidence')} "
                f"mrz_found={(passport.response.get('mrz') or {}).get('found')}",
            )

    return data


def response_with_face_detection(response: dict[str, Any], detection: DetectedFace) -> dict[str, Any]:
    payload = dict(response)
    payload["detected"] = True
    payload["face_bbox"] = detection.bbox
    payload["face_confidence"] = detection.confidence
    return payload


def response_with_passport_detection(response: dict[str, Any], detection: DetectedFace) -> dict[str, Any]:
    payload = dict(response)
    payload["face_bbox"] = detection.bbox
    payload["face_confidence"] = detection.confidence
    payload["face_content_type"] = detection.aligned_face_content_type
    payload["face_base64"] = detection.aligned_face_base64
    return payload


def log_batch(logger: Callable[[str], None] | None, message: str) -> None:
    if logger is not None:
        logger(message)
