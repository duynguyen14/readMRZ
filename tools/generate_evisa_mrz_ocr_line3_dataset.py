from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

# Keep Paddle settings aligned with the line2 script if OCR fallback is enabled.
os.environ.setdefault("FLAGS_use_onednn", "0")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")

import cv2
import numpy as np
import pyodbc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from generate_mrz_yolo_dataset import (  # noqa: E402
    build_ocr,
    env_bool,
    env_float,
    env_int,
    env_value,
    line_mrz_likeness,
    read_env_file,
)
from generate_mrz_ocr_line2_dataset import (  # noqa: E402
    crop_line_images,
    deskew_crop,
    line2_ocr_env,
    maybe_reexec_with_paddle_python,
    paddle_ocr_line,
    projection_bands,
    relative_to_base,
    resolve_base_dir,
)
from mrz_reader.db import connect, execute_sql_file  # noqa: E402
from mrz_reader.mrz import normalize_mrz_text  # noqa: E402


DEFAULT_CSHARP_CONFIG = Path(r"D:\DocHochieu\ToolCheckDuLieu\Check\CheckduLieu\application.json")


def ensure_line3_schema() -> None:
    execute_sql_file(PROJECT_ROOT / "sql" / "create_mrz_ocr_line3_tables.sql")


def load_csharp_config(env: dict[str, str]) -> dict[str, Any]:
    configured_path = env_value(env, "READMRZ_EVISA_APPLICATION_JSON", str(DEFAULT_CSHARP_CONFIG))
    config_path = Path(configured_path).expanduser()
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8-sig"))


def parse_connection_string(value: str) -> dict[str, str]:
    parts: dict[str, str] = {}
    for raw_part in str(value or "").split(";"):
        if not raw_part.strip() or "=" not in raw_part:
            continue
        key, raw_value = raw_part.split("=", 1)
        normalized_key = key.strip().lower().replace(" ", "")
        parts[normalized_key] = raw_value.strip()
    return parts


def odbc_yes_no(value: str, default: str = "yes") -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"true", "yes", "1", "y"}:
        return "yes"
    if normalized in {"false", "no", "0", "n"}:
        return "no"
    return default


def build_source_connection_string(env: dict[str, str], csharp_config: dict[str, Any]) -> str:
    explicit = env_value(env, "READMRZ_EVISA_SOURCE_CONNECTION_STRING")
    if explicit:
        if "driver=" in explicit.lower():
            return explicit
        parts = parse_connection_string(explicit)
    else:
        parts = parse_connection_string(str(csharp_config.get("SourceConnectionString") or ""))

    driver = env_value(env, "READMRZ_EVISA_SOURCE_DB_DRIVER", env_value(env, "READMRZ_DB_DRIVER", "ODBC Driver 17 for SQL Server"))
    server = env_value(env, "READMRZ_EVISA_SOURCE_DB_SERVER", parts.get("server", ""))
    database = env_value(env, "READMRZ_EVISA_SOURCE_DB_DATABASE", parts.get("database", ""))
    username = env_value(env, "READMRZ_EVISA_SOURCE_DB_USERNAME", parts.get("userid", parts.get("uid", "")))
    password = env_value(env, "READMRZ_EVISA_SOURCE_DB_PASSWORD", parts.get("password", parts.get("pwd", "")))
    trusted = env_value(env, "READMRZ_EVISA_SOURCE_DB_TRUSTED_CONNECTION", parts.get("trustedconnection", "false"))
    trust_cert = env_value(env, "READMRZ_EVISA_SOURCE_DB_TRUST_SERVER_CERTIFICATE", parts.get("trustservercertificate", "true"))

    connection_parts = [
        f"DRIVER={{{driver}}}",
        f"SERVER={server}",
        f"DATABASE={database}",
        f"TrustServerCertificate={odbc_yes_no(trust_cert, 'yes')}",
    ]
    if odbc_yes_no(trusted, "no") == "yes":
        connection_parts.append("Trusted_Connection=yes")
    else:
        connection_parts.extend([f"UID={username}", f"PWD={password}"])
    return ";".join(connection_parts)


def source_root_dir(env: dict[str, str], csharp_config: dict[str, Any]) -> Path:
    configured = env_value(env, "READMRZ_EVISA_SOURCE_ROOT_DIR", str(csharp_config.get("SourceRootDirectory") or ""))
    if not configured:
        raise ValueError("Missing READMRZ_EVISA_SOURCE_ROOT_DIR and SourceRootDirectory in application.json")
    return Path(configured).expanduser().resolve()


def connect_source(env: dict[str, str], csharp_config: dict[str, Any]) -> pyodbc.Connection:
    return pyodbc.connect(build_source_connection_string(env, csharp_config), autocommit=False)


def row_to_dict(cursor: Any, row: Any) -> dict[str, Any]:
    columns = [column[0] for column in cursor.description]
    return dict(zip(columns, row))


def fetch_evisa_records(
    source_cursor: Any,
    *,
    limit: int,
    min_point: float,
    order_by: str,
) -> list[dict[str, Any]]:
    top_clause = f"TOP ({int(limit)})" if limit > 0 else ""
    if order_by == "newest":
        order_clause = "CreatedDate DESC, Id DESC"
    elif order_by == "random":
        order_clause = "NEWID()"
    else:
        order_clause = "CreatedDate ASC, Id ASC"

    source_cursor.execute(
        f"""
        SELECT {top_clause}
            Id,
            GUID,
            CreatedDate,
            PassportNo,
            FullPassportImage,
            MrzlineOne,
            MrzlineTwo,
            MrzlineOnePoint,
            MrzlineTwoPoint
        FROM [db_dichvu_visa].[dbo].[TransactionEVisa]
        WHERE ISNULL(LTRIM(RTRIM(MrzlineOne)), '') <> ''
          AND ISNULL(LTRIM(RTRIM(MrzlineTwo)), '') <> ''
          AND TRY_CONVERT(float, MrzlineOnePoint) > ?
          AND TRY_CONVERT(float, MrzlineTwoPoint) > ?
          AND ISNULL(LTRIM(RTRIM(FullPassportImage)), '') <> ''
        ORDER BY {order_clause}
        """,
        min_point,
        min_point,
    )
    return [row_to_dict(source_cursor, row) for row in source_cursor.fetchall()]


def resolve_source_image_path(full_passport_image: Any, root_dir: Path) -> Path:
    raw_value = str(full_passport_image or "").strip()
    if not raw_value:
        raise ValueError("FullPassportImage is empty")
    normalized = raw_value.replace("/", "\\").lstrip("~")
    path = Path(normalized)
    if path.is_absolute():
        return path.resolve()
    return (root_dir / normalized.lstrip("\\")).resolve()


def safe_source_key(record: dict[str, Any]) -> str:
    return f"TransactionEVisa:{int(record['Id'])}"


def yolo_label_from_bbox(bbox: list[float], image_width: int, image_height: int) -> str:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    center_x = ((x1 + x2) / 2.0) / max(1, image_width)
    center_y = ((y1 + y2) / 2.0) / max(1, image_height)
    width = abs(x2 - x1) / max(1, image_width)
    height = abs(y2 - y1) / max(1, image_height)
    return f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"


def upsert_label_item(
    cursor: Any,
    *,
    record: dict[str, Any],
    source_key: str,
    source_image_path: Path | None,
    split: str,
    image_file_name: str | None,
    bbox: list[float] | None,
    image_width: int | None,
    image_height: int | None,
    yolo_confidence: float | None,
    yolo_rotation_angle: int | None,
    status: str,
    error_message: str | None,
) -> int:
    mrz_lines = [
        normalize_mrz_text(str(record.get("MrzlineOne") or "")),
        normalize_mrz_text(str(record.get("MrzlineTwo") or "")),
    ]
    ocr_config = {
        "source": "TransactionEVisa",
        "id": int(record["Id"]),
        "guid": str(record.get("GUID") or ""),
        "passport_no": str(record.get("PassportNo") or ""),
        "mrzline_one_point": point_to_float(record.get("MrzlineOnePoint")),
        "mrzline_two_point": point_to_float(record.get("MrzlineTwoPoint")),
        "source_image": str(source_image_path or ""),
        "yolo_rotation_angle": yolo_rotation_angle,
    }
    bbox_values = bbox or [None, None, None, None]
    yolo_label = yolo_label_from_bbox(bbox, image_width or 1, image_height or 1) if bbox and image_width and image_height else None
    cursor.execute(
        """
        UPDATE dbo.readmrz_label_items
        SET
            source_file_name = ?,
            status = ?,
            review_status = 'approved',
            split = ?,
            image_file_name = ?,
            bbox_x1 = ?,
            bbox_y1 = ?,
            bbox_x2 = ?,
            bbox_y2 = ?,
            yolo_label = ?,
            mrz_lines_json = ?,
            mrz_score = ?,
            ocr_engine = ?,
            ocr_config_json = ?,
            error_message = ?,
            processed_at = SYSUTCDATETIME(),
            reviewed_at = SYSUTCDATETIME(),
            updated_at = SYSUTCDATETIME()
        WHERE source_key = ?
        """,
        Path(str(record.get("FullPassportImage") or "")).name or None,
        status,
        split,
        image_file_name,
        bbox_values[0],
        bbox_values[1],
        bbox_values[2],
        bbox_values[3],
        yolo_label,
        json.dumps(mrz_lines, ensure_ascii=False),
        yolo_confidence,
        "transaction_evisa_mrz+yolo",
        json.dumps(ocr_config, ensure_ascii=False),
        error_message,
        source_key,
    )
    if not cursor.rowcount:
        cursor.execute(
            """
            INSERT INTO dbo.readmrz_label_items (
                source_key,
                source_file_name,
                status,
                review_status,
                split,
                image_file_name,
                bbox_x1,
                bbox_y1,
                bbox_x2,
                bbox_y2,
                yolo_label,
                mrz_lines_json,
                mrz_score,
                ocr_engine,
                ocr_config_json,
                error_message,
                processed_at,
                reviewed_at
            )
            VALUES (?, ?, ?, 'approved', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, SYSUTCDATETIME(), SYSUTCDATETIME())
            """,
            source_key,
            Path(str(record.get("FullPassportImage") or "")).name or None,
            status,
            split,
            image_file_name,
            bbox_values[0],
            bbox_values[1],
            bbox_values[2],
            bbox_values[3],
            yolo_label,
            json.dumps(mrz_lines, ensure_ascii=False),
            yolo_confidence,
            "transaction_evisa_mrz+yolo",
            json.dumps(ocr_config, ensure_ascii=False),
            error_message,
        )

    cursor.execute("SELECT id FROM dbo.readmrz_label_items WHERE source_key = ?", source_key)
    row = cursor.fetchone()
    if not row:
        raise RuntimeError(f"Cannot resolve label_item_id for source_key={source_key}")
    return int(row[0])


def count_line3_rows(cursor: Any, label_item_id: int) -> int:
    cursor.execute("SELECT COUNT(*) FROM dbo.readmrz_ocr_line_items3 WHERE label_item_id = ?", label_item_id)
    row = cursor.fetchone()
    return int(row[0] or 0) if row else 0


def point_to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def point_to_score(value: Any) -> float:
    point = point_to_float(value)
    if point is None:
        return 0.0
    return point / 100.0 if point > 1.0 else point


def build_yolo_model(env: dict[str, str]) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is not installed. Run: pip install ultralytics") from exc

    model_path = Path(env_value(env, "READMRZ_YOLO_MODEL_PATH", "models/mrz_yolo11n_best.pt")).expanduser()
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    model_path = model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"READMRZ_YOLO_MODEL_PATH does not exist: {model_path}")
    print(f"Loading YOLO model: {model_path}")
    return YOLO(str(model_path))


def rotated_candidates(image: np.ndarray) -> list[tuple[int, np.ndarray]]:
    return [
        (90, np.ascontiguousarray(np.rot90(image, 1))),
        (180, np.ascontiguousarray(np.rot90(image, 2))),
        (270, np.ascontiguousarray(np.rot90(image, 3))),
    ]


def detect_once(model: Any, image: np.ndarray, env: dict[str, str], rotation_angle: int) -> dict[str, Any]:
    started = time.perf_counter()
    results = model.predict(
        source=image,
        imgsz=env_int(env, "READMRZ_YOLO_IMGSZ", 640),
        conf=env_float(env, "READMRZ_YOLO_CONF", 0.25),
        device=env_value(env, "READMRZ_YOLO_DEVICE", "cpu"),
        verbose=False,
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    detections: list[dict[str, Any]] = []
    if results:
        boxes = getattr(results[0], "boxes", None)
        names = getattr(results[0], "names", {}) or {}
        if boxes is not None:
            xyxy_values = boxes.xyxy.cpu().numpy().tolist()
            conf_values = boxes.conf.cpu().numpy().tolist()
            cls_values = boxes.cls.cpu().numpy().tolist()
            for bbox, confidence, class_id in zip(xyxy_values, conf_values, cls_values, strict=False):
                class_int = int(class_id)
                detections.append(
                    {
                        "bbox_xyxy": [float(value) for value in bbox],
                        "confidence": float(confidence),
                        "class_id": class_int,
                        "class_name": str(names.get(class_int, "mrz")),
                    }
                )
    detections.sort(key=lambda item: float(item["confidence"]), reverse=True)
    return {
        "rotation_angle": rotation_angle,
        "elapsed_ms": elapsed_ms,
        "detections": detections,
        "best": detections[0] if detections else None,
    }


def detect_best_mrz(model: Any, image: np.ndarray, env: dict[str, str]) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    first_attempt = detect_once(model, image, env, 0)
    attempts.append(first_attempt)
    best_attempt = first_attempt
    best_image = image
    best_confidence = float(first_attempt["best"]["confidence"]) if first_attempt["best"] else 0.0
    min_conf = env_float(env, "READMRZ_YOLO_ROTATION_FALLBACK_MIN_CONF", 0.85)
    should_try_rotations = (
        env_bool(env, "READMRZ_YOLO_ROTATION_FALLBACK", True)
        and (not first_attempt["best"] or best_confidence < min_conf)
    )
    if should_try_rotations:
        for rotation_angle, rotated_image in rotated_candidates(image):
            attempt = detect_once(model, rotated_image, env, rotation_angle)
            attempts.append(attempt)
            confidence = float(attempt["best"]["confidence"]) if attempt["best"] else 0.0
            if confidence > best_confidence:
                best_confidence = confidence
                best_attempt = attempt
                best_image = rotated_image

    best_detection = best_attempt["best"]
    return {
        "found": bool(best_detection),
        "rotation_angle": int(best_attempt["rotation_angle"]),
        "image": best_image,
        "bbox_xyxy": best_detection["bbox_xyxy"] if best_detection else None,
        "confidence": float(best_detection["confidence"]) if best_detection else 0.0,
        "attempts": [
            {
                "rotation_angle": attempt["rotation_angle"],
                "elapsed_ms": attempt["elapsed_ms"],
                "boxes": len(attempt["detections"]),
                "best_confidence": float(attempt["best"]["confidence"]) if attempt["best"] else None,
            }
            for attempt in attempts
        ],
    }


def clamp_box(box: list[Any], width: int, height: int, padding_ratio: float) -> list[int]:
    x1, y1, x2, y2 = [float(value) for value in box]
    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)
    pad_x = max(2.0, (right - left) * padding_ratio)
    pad_y = max(2.0, (bottom - top) * padding_ratio)
    return [
        max(0, int(round(left - pad_x))),
        max(0, int(round(top - pad_y))),
        min(width, int(round(right + pad_x))),
        min(height, int(round(bottom + pad_y))),
    ]


def rotate_vertical_crop_if_needed(crop: np.ndarray, env: dict[str, str]) -> tuple[np.ndarray, int]:
    height, width = crop.shape[:2]
    ratio = env_float(env, "READMRZ_EVISA_LINE3_VERTICAL_BOX_RATIO", 1.15)
    if height <= width * ratio:
        return crop, 0
    direction = env_value(env, "READMRZ_EVISA_LINE3_VERTICAL_ROTATE_DIRECTION", "cw").strip().lower()
    if direction == "ccw":
        return np.ascontiguousarray(np.rot90(crop, 1)), 270
    return np.ascontiguousarray(np.rot90(crop, 3)), 90


def write_line_row3(
    cursor: Any,
    *,
    label_item_id: int,
    source_key: str,
    split: str,
    line_index: int,
    crop_rel: str,
    line_rel: str,
    line: dict[str, Any],
    mrz_bbox: list[int],
    doc_orientation_angle: int,
    deskew_angle: float,
    text: str,
    ocr_score: float,
    error_message: str | None,
) -> None:
    normalized = normalize_mrz_text(text)
    line_bbox = line["bbox_crop"]
    cursor.execute(
        """
        INSERT INTO dbo.readmrz_ocr_line_items3 (
            label_item_id,
            source_key,
            split,
            line_index,
            mrz_crop_file_name,
            line_image_file_name,
            line_image_width,
            line_image_height,
            mrz_bbox_x1,
            mrz_bbox_y1,
            mrz_bbox_x2,
            mrz_bbox_y2,
            line_bbox_x1,
            line_bbox_y1,
            line_bbox_x2,
            line_bbox_y2,
            doc_orientation_angle,
            deskew_angle,
            projection_score,
            split_method,
            ocr_text,
            normalized_text,
            final_text,
            ocr_score,
            mrz_likeness,
            review_status,
            error_message,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'opencv_projection', ?, ?, ?, ?, ?, 'pending', ?, SYSUTCDATETIME(), SYSUTCDATETIME())
        """,
        label_item_id,
        source_key,
        split,
        line_index,
        crop_rel,
        line_rel,
        int(line["width"]),
        int(line["height"]),
        mrz_bbox[0],
        mrz_bbox[1],
        mrz_bbox[2],
        mrz_bbox[3],
        float(line_bbox[0]),
        float(line_bbox[1]),
        float(line_bbox[2]),
        float(line_bbox[3]),
        doc_orientation_angle,
        deskew_angle,
        float(line.get("projection_score") or 0.0),
        text,
        normalized,
        normalized or None,
        ocr_score,
        line_mrz_likeness(text) if text else 0.0,
        error_message,
    )


def process_record(
    cursor: Any,
    *,
    record: dict[str, Any],
    source_root: Path,
    yolo_model: Any,
    line_image_base_dir: Path,
    crop_base_dir: Path,
    raw_crop_base_dir: Path,
    split: str,
    env: dict[str, str],
    temp_dir: Path,
    ocr: Any | None,
    force: bool,
) -> dict[str, Any]:
    source_key = safe_source_key(record)
    source_image_path = resolve_source_image_path(record.get("FullPassportImage"), source_root)
    label_item_id = existing_label_item_id(cursor, source_key)
    if label_item_id and not force and count_line3_rows(cursor, label_item_id) > 0:
        return {"status": "skipped", "source_key": source_key, "label_item_id": label_item_id, "lines": 0}

    image = cv2.imread(str(source_image_path))
    if image is None:
        label_item_id = upsert_label_item(
            cursor,
            record=record,
            source_key=source_key,
            source_image_path=source_image_path,
            split=split,
            image_file_name=str(source_image_path),
            bbox=None,
            image_width=None,
            image_height=None,
            yolo_confidence=None,
            yolo_rotation_angle=None,
            status="error",
            error_message=f"Cannot read source image: {source_image_path}",
        )
        return {"status": "error", "source_key": source_key, "label_item_id": label_item_id, "lines": 0, "error": "cannot_read_image"}

    detection = detect_best_mrz(yolo_model, image, env)
    if not detection["found"]:
        label_item_id = upsert_label_item(
            cursor,
            record=record,
            source_key=source_key,
            source_image_path=source_image_path,
            split=split,
            image_file_name=str(source_image_path),
            bbox=None,
            image_width=image.shape[1],
            image_height=image.shape[0],
            yolo_confidence=None,
            yolo_rotation_angle=int(detection["rotation_angle"]),
            status="no_mrz",
            error_message="YOLO did not find MRZ",
        )
        return {"status": "no_mrz", "source_key": source_key, "label_item_id": label_item_id, "lines": 0}

    yolo_image = detection["image"]
    image_height, image_width = yolo_image.shape[:2]
    mrz_bbox = clamp_box(
        detection["bbox_xyxy"],
        image_width,
        image_height,
        env_float(env, "READMRZ_OCR_LINE2_MRZ_PADDING_RATIO", 0.08),
    )
    x1, y1, x2, y2 = mrz_bbox
    raw_crop = yolo_image[y1:y2, x1:x2].copy()
    if raw_crop.size == 0:
        raise ValueError(f"MRZ crop is empty for {source_key}")

    normalized_crop, vertical_rotation_angle = rotate_vertical_crop_if_needed(raw_crop, env)
    doc_orientation_angle = (int(detection["rotation_angle"]) + vertical_rotation_angle) % 360
    deskewed_crop, binary, deskew_angle = deskew_crop(normalized_crop, env)
    bands = projection_bands(binary, env)
    lines = crop_line_images(deskewed_crop, bands, env)

    label_item_id = upsert_label_item(
        cursor,
        record=record,
        source_key=source_key,
        source_image_path=source_image_path,
        split=split,
        image_file_name=str(source_image_path),
        bbox=mrz_bbox,
        image_width=image_width,
        image_height=image_height,
        yolo_confidence=float(detection["confidence"]),
        yolo_rotation_angle=doc_orientation_angle,
        status="labeled",
        error_message=None if lines else "No projection bands found",
    )

    cursor.execute("DELETE FROM dbo.readmrz_ocr_line_items3 WHERE label_item_id = ?", label_item_id)
    if not lines:
        return {
            "status": "no_lines",
            "source_key": source_key,
            "label_item_id": label_item_id,
            "lines": 0,
            "bands": len(bands),
            "confidence": float(detection["confidence"]),
            "rotation_angle": doc_orientation_angle,
        }

    line_output_dir = line_image_base_dir / split
    crop_output_dir = crop_base_dir / split
    raw_crop_output_dir = raw_crop_base_dir / split
    line_output_dir.mkdir(parents=True, exist_ok=True)
    crop_output_dir.mkdir(parents=True, exist_ok=True)
    raw_crop_output_dir.mkdir(parents=True, exist_ok=True)

    stem = source_key.replace(":", "_")
    raw_crop_path = raw_crop_output_dir / f"{stem}_mrz_raw_crop.jpg"
    cv2.imwrite(str(raw_crop_path), normalized_crop)
    crop_path = crop_output_dir / f"{stem}_mrz_crop.jpg"
    cv2.imwrite(str(crop_path), deskewed_crop)
    crop_rel = relative_to_base(crop_path, crop_base_dir)

    source_texts = [
        normalize_mrz_text(str(record.get("MrzlineOne") or "")),
        normalize_mrz_text(str(record.get("MrzlineTwo") or "")),
    ]
    source_scores = [
        point_to_score(record.get("MrzlineOnePoint")),
        point_to_score(record.get("MrzlineTwoPoint")),
    ]

    for line_index, line in enumerate(lines, start=1):
        line_path = line_output_dir / f"{stem}_line{line_index}.jpg"
        cv2.imwrite(str(line_path), line["image"])
        text = source_texts[line_index - 1] if line_index <= len(source_texts) else ""
        score = source_scores[line_index - 1] if line_index <= len(source_scores) else 0.0
        error_message = "text_from_transaction_evisa_mrz_lines"
        if not text and ocr is not None:
            try:
                text, score = paddle_ocr_line(ocr, line["image"], temp_dir, f"{stem}_line{line_index}")
                error_message = "text_from_paddle_line_fallback"
            except Exception as exc:
                error_message = str(exc)[:1900]
        write_line_row3(
            cursor,
            label_item_id=label_item_id,
            source_key=source_key,
            split=split,
            line_index=line_index,
            crop_rel=crop_rel,
            line_rel=relative_to_base(line_path, line_image_base_dir),
            line=line,
            mrz_bbox=mrz_bbox,
            doc_orientation_angle=doc_orientation_angle,
            deskew_angle=deskew_angle,
            text=text,
            ocr_score=score,
            error_message=error_message,
        )

    return {
        "status": "done",
        "source_key": source_key,
        "label_item_id": label_item_id,
        "lines": len(lines),
        "bands": len(bands),
        "confidence": float(detection["confidence"]),
        "rotation_angle": doc_orientation_angle,
    }


def existing_label_item_id(cursor: Any, source_key: str) -> int | None:
    cursor.execute("SELECT id FROM dbo.readmrz_label_items WHERE source_key = ?", source_key)
    row = cursor.fetchone()
    return int(row[0]) if row else None


def maybe_reexec_for_paddle_line(env: dict[str, str], text_source: str) -> int | None:
    if text_source != "paddle-line":
        return None
    engine = env_value(env, "READMRZ_OCR_ENGINE", "paddle").strip().lower()
    if engine != "paddle" or importlib.util.find_spec("paddleocr") is not None:
        return None
    return maybe_reexec_with_paddle_python(env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate readmrz_ocr_line_items3 from db_dichvu_visa.dbo.TransactionEVisa MRZ lines."
    )
    parser.add_argument("--env", default=str(PROJECT_ROOT / ".env"), help="Path to .env config file.")
    parser.add_argument("--limit", type=int, default=4000, help="Max TransactionEVisa rows to fetch. Use 0 for all.")
    parser.add_argument("--min-point", type=float, default=90.0, help="Minimum MrzlineOnePoint/MrzlineTwoPoint.")
    parser.add_argument("--order-by", choices=["oldest", "newest", "random"], default="newest", help="Source row order.")
    parser.add_argument("--split", choices=["train", "val"], default="train", help="Dataset split folder to write.")
    parser.add_argument("--force", action="store_true", help="Regenerate line3 records already present in DB.")
    parser.add_argument("--skip-schema", action="store_true", help="Do not run sql/create_mrz_ocr_line3_tables.sql.")
    parser.add_argument(
        "--text-source",
        choices=["transaction", "paddle-line"],
        default="transaction",
        help="Use TransactionEVisa MRZ text, or OCR each cropped line with Paddle fallback.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    env_path = Path(args.env)
    if not env_path.is_absolute():
        cwd_env_path = Path.cwd() / env_path
        project_env_path = PROJECT_ROOT / env_path
        env_path = cwd_env_path if cwd_env_path.exists() else project_env_path

    env = read_env_file(env_path)
    reexec_code = maybe_reexec_for_paddle_line(env, args.text_source)
    if reexec_code is not None:
        return reexec_code

    csharp_config = load_csharp_config(env)
    source_root = source_root_dir(env, csharp_config)
    line3_dataset_dir = resolve_base_dir(env, "READMRZ_EVISA_LINE3_DATASET_DIR", PROJECT_ROOT / "generated_datasets" / "mrz_ocr_lines3")
    line3_image_base_dir = resolve_base_dir(env, "READMRZ_EVISA_LINE3_IMAGE_BASE_DIR", line3_dataset_dir / "images")
    line3_crop_base_dir = resolve_base_dir(env, "READMRZ_EVISA_LINE3_CROP_BASE_DIR", line3_dataset_dir / "crops")
    line3_raw_crop_base_dir = resolve_base_dir(env, "READMRZ_EVISA_LINE3_RAW_CROP_BASE_DIR", line3_dataset_dir / "raw_crops")
    batch_size = max(1, env_int(env, "READMRZ_EVISA_LINE3_BATCH_SIZE", env_int(env, "READMRZ_OCR_LINE2_BATCH_SIZE", 25)))

    if not args.skip_schema:
        ensure_line3_schema()

    print(f"source_root: {source_root}")
    print(f"line3_dataset_dir: {line3_dataset_dir}")
    print(f"line3_image_base_dir: {line3_image_base_dir}")
    print(f"line3_crop_base_dir: {line3_crop_base_dir}")
    print(f"line3_raw_crop_base_dir: {line3_raw_crop_base_dir}")
    print(f"limit: {args.limit}")
    print(f"min_point: {args.min_point}")
    print(f"order_by: {args.order_by}")
    print(f"split: {args.split}")
    print(f"force: {args.force}")
    print(f"text_source: {args.text_source}")

    yolo_model = build_yolo_model(env)
    ocr = build_ocr(line2_ocr_env(env)) if args.text_source == "paddle-line" else None

    totals = {
        "done": 0,
        "skipped": 0,
        "no_mrz": 0,
        "no_lines": 0,
        "errors": 0,
        "fetched": 0,
    }

    with tempfile.TemporaryDirectory(prefix="readmrz_evisa_line3_") as temp_name:
        temp_dir = Path(temp_name)
        with connect_source(env, csharp_config) as source_connection, connect() as destination_connection:
            source_cursor = source_connection.cursor()
            destination_cursor = destination_connection.cursor()
            records = fetch_evisa_records(
                source_cursor,
                limit=args.limit,
                min_point=args.min_point,
                order_by=args.order_by,
            )
            totals["fetched"] = len(records)
            print(f"Fetched {len(records)} TransactionEVisa rows")

            for index, record in enumerate(records, start=1):
                started = time.perf_counter()
                result: dict[str, Any]
                try:
                    result = process_record(
                        destination_cursor,
                        record=record,
                        source_root=source_root,
                        yolo_model=yolo_model,
                        line_image_base_dir=line3_image_base_dir,
                        crop_base_dir=line3_crop_base_dir,
                        raw_crop_base_dir=line3_raw_crop_base_dir,
                        split=args.split,
                        env=env,
                        temp_dir=temp_dir,
                        ocr=ocr,
                        force=args.force,
                    )
                    status = str(result["status"])
                    if status in totals:
                        totals[status] += 1
                    elif status == "error":
                        totals["errors"] += 1
                except Exception as exc:
                    totals["errors"] += 1
                    result = {
                        "status": "error",
                        "source_key": safe_source_key(record),
                        "error": str(exc)[:1900],
                    }
                    try:
                        upsert_label_item(
                            destination_cursor,
                            record=record,
                            source_key=safe_source_key(record),
                            source_image_path=None,
                            split=args.split,
                            image_file_name=None,
                            bbox=None,
                            image_width=None,
                            image_height=None,
                            yolo_confidence=None,
                            yolo_rotation_angle=None,
                            status="error",
                            error_message=str(exc)[:1900],
                        )
                    except Exception:
                        pass

                if index % batch_size == 0:
                    destination_connection.commit()
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                print(
                    f"[{index}/{len(records)}] source={result.get('source_key')} "
                    f"status={result.get('status')} lines={result.get('lines', 0)} "
                    f"conf={float(result.get('confidence') or 0.0):.4f} "
                    f"rot={result.get('rotation_angle', 0)} elapsed_ms={elapsed_ms}"
                    + (f" error={result.get('error')}" if result.get("error") else "")
                )

            destination_connection.commit()

    summary = {
        **totals,
        "line3_dataset_dir": str(line3_dataset_dir),
        "line3_image_base_dir": str(line3_image_base_dir),
        "line3_crop_base_dir": str(line3_crop_base_dir),
        "line3_raw_crop_base_dir": str(line3_raw_crop_base_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
