from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

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
class FaceCandidate:
    payload: ImagePayload
    working: WorkingImage
    response: dict[str, Any]
    embedding: np.ndarray | None


@dataclass
class PassportCandidate:
    payload: ImagePayload
    working: WorkingImage
    response: dict[str, Any]
    embedding: np.ndarray | None


def process_batch(
    items: list[dict[str, Any]],
    *,
    classifier: FileTypeClassifier,
    face_matcher: FaceMatchService,
    yolo_detector: YoloMrzDetector,
    orientation: PaddleDocumentOrientation,
    recognizer: CustomMrzCtcRecognizer,
) -> dict[str, Any]:
    env = read_env_file()
    max_items = max(1, int(env_value(env, "READMRZ_BATCH_MAX_ITEMS", "20")))
    if len(items) > max_items:
        raise ValueError(f"Too many items. Max allowed is {max_items}.")

    max_side = max(0, int(env_value(env, "READMRZ_INFERENCE_MAX_IMAGE_SIDE", "1600")))
    faces: list[FaceCandidate] = []
    passports: list[PassportCandidate] = []

    for raw_item in items:
        input_item = normalize_input_item(raw_item)
        payload = decode_base64_image_payload(input_item.base64_value, input_item.file_name)
        working = resize_for_inference(payload.image, max_side)
        classification = classifier.predict(working.image)
        label = normalize_class_label(classification.get("label"))
        if label == "face":
            faces.append(build_face_candidate(payload, working, classification, face_matcher))
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
                )
            )

    return {"data": pair_candidates(faces, passports, face_matcher)}


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
) -> FaceCandidate:
    face_row, detect_meta = face_matcher.detect_primary_face(working.image)
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
        "detected": face_row is not None,
        "face_count": int(detect_meta["count"]),
        "face_bbox": (
            map_bbox_to_original(face_bbox(face_row), working, payload.width, payload.height)
            if face_row is not None
            else None
        ),
        "face_confidence": face_confidence(face_row) if face_row is not None else 0.0,
    }
    embedding = None
    if face_row is not None:
        embedding, _ = face_matcher.extract_embedding(working.image, face_row)
    return FaceCandidate(payload=payload, working=working, response=response, embedding=embedding)


def build_passport_candidate(
    payload: ImagePayload,
    working: WorkingImage,
    classification: dict[str, Any],
    face_matcher: FaceMatchService,
    yolo_detector: YoloMrzDetector,
    orientation: PaddleDocumentOrientation,
    recognizer: CustomMrzCtcRecognizer,
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
    face_row, detect_meta = face_matcher.detect_primary_face(working.image)

    face_base64 = ""
    face_content_type = ""
    embedding = None
    if face_row is not None:
        embedding, face_meta = face_matcher.extract_embedding(working.image, face_row)
        face_base64 = str(face_meta.get("aligned_face_base64") or "")
        face_content_type = str(face_meta.get("aligned_face_content_type") or "")
        if not face_base64:
            crop = crop_bbox(working.image, face_bbox(face_row), padding_ratio=0.2)
            if crop is not None:
                face_content_type, face_base64 = encode_jpeg_base64(crop)

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
        "face_bbox": (
            map_bbox_to_original(face_bbox(face_row), working, payload.width, payload.height)
            if face_row is not None
            else None
        ),
        "face_confidence": face_confidence(face_row) if face_row is not None else 0.0,
        "mrz": build_mrz_payload(compact_mrz),
        "parsed": build_parsed_payload(compact_mrz),
        "processing_ms": int((perf_counter() - started) * 1000),
    }
    return PassportCandidate(payload=payload, working=working, response=response, embedding=embedding)


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
) -> list[dict[str, Any]]:
    pairs: list[tuple[float, int, int, dict[str, Any]]] = []
    for face_index, face in enumerate(faces):
        if face.embedding is None:
            continue
        for passport_index, passport in enumerate(passports):
            if passport.embedding is None:
                continue
            score = face_matcher.match_embeddings(face.embedding, passport.embedding)
            decision, matched, review_required = face_matcher.decision(score)
            if decision == "mismatch":
                continue
            pairs.append(
                (
                    score,
                    face_index,
                    passport_index,
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
    for _, face_index, passport_index, match_payload in pairs:
        if face_index in used_faces or passport_index in used_passports:
            continue
        face = faces[face_index]
        passport = passports[passport_index]
        passport_payload = dict(passport.response)
        passport_payload["match"] = match_payload
        data.append({"face": face.response, "passport": passport_payload})
        used_faces.add(face_index)
        used_passports.add(passport_index)

    for face_index, face in enumerate(faces):
        if face_index not in used_faces:
            data.append({"face": face.response, "passport": None})

    for passport_index, passport in enumerate(passports):
        if passport_index not in used_passports:
            passport_payload = dict(passport.response)
            passport_payload["match"] = None
            data.append({"face": None, "passport": passport_payload})

    return data
