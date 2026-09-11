from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

import cv2

from .db import connect
from .env_config import env_value, read_env_file


def crop_output_dir() -> Path:
    env = read_env_file()
    raw_output_dir = env_value(env, "READMRZ_VN_VISA_CROP_OCR_OUTPUT_DIR", "")
    if not raw_output_dir:
        raise ValueError("Missing READMRZ_VN_VISA_CROP_OCR_OUTPUT_DIR in .env")
    return Path(raw_output_dir).expanduser().resolve()


def visa_image_dir() -> Path:
    env = read_env_file()
    raw_output_dir = (
        env_value(env, "READMRZ_VN_VISA_CROP_OCR_SOURCE_IMAGE_DIR", "")
        or env_value(env, "READMRZ_VN_VISA_OUTPUT_DIR", "")
        or env_value(env, "READMRZ_VN_VISA_IMPORT_ORIENTED_IMAGE_DIR", "")
    )
    if not raw_output_dir:
        raise ValueError(
            "Missing READMRZ_VN_VISA_CROP_OCR_SOURCE_IMAGE_DIR, "
            "READMRZ_VN_VISA_OUTPUT_DIR, or READMRZ_VN_VISA_IMPORT_ORIENTED_IMAGE_DIR in .env"
        )
    return Path(raw_output_dir).expanduser().resolve()


def row_to_dict(cursor: Any, row: Any) -> dict[str, Any]:
    columns = [column[0] for column in cursor.description]
    return dict(zip(columns, row))


def fetch_one_dict(cursor: Any) -> dict[str, Any] | None:
    row = cursor.fetchone()
    return row_to_dict(cursor, row) if row else None


def image_to_base64(path: Path) -> tuple[str, str, int, int]:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    height, width = image.shape[:2]
    content_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return encoded, content_type, width, height


def review_stats(cursor: Any) -> dict[str, int]:
    cursor.execute(
        """
        SELECT
            COUNT(*) AS total_crops,
            SUM(CASE WHEN ReviewStatus = N'pending' THEN 1 ELSE 0 END) AS pending_crops,
            SUM(CASE WHEN ReviewStatus = N'approved' THEN 1 ELSE 0 END) AS approved_crops,
            SUM(CASE WHEN ReviewStatus = N'rejected' THEN 1 ELSE 0 END) AS rejected_crops,
            COUNT(DISTINCT VisaItemId) AS total_visas,
            COUNT(DISTINCT CASE WHEN ReviewStatus = N'pending' THEN VisaItemId END) AS pending_visas
        FROM dbo.readmrz_vn_visa_ocr_crops
        """
    )
    row = fetch_one_dict(cursor) or {}
    keys = ("total_crops", "pending_crops", "approved_crops", "rejected_crops", "total_visas", "pending_visas")
    return {key: int(row.get(key) or 0) for key in keys}


def get_next_vn_visa_ocr_crop_review(after_id: int = 0) -> dict[str, Any]:
    with connect() as connection:
        cursor = connection.cursor()
        stats = review_stats(cursor)
        row = fetch_review_visa(cursor, "next", after_id)
        if not row and after_id > 0:
            row = fetch_review_visa(cursor, "next", 0)
        if not row:
            return {"status": "empty", "current": None, "stats": stats}
        return build_response(cursor, row, stats)


def get_previous_vn_visa_ocr_crop_review(before_id: int = 0) -> dict[str, Any]:
    if before_id <= 0:
        with connect() as connection:
            cursor = connection.cursor()
            return {"status": "empty", "current": None, "stats": review_stats(cursor)}

    with connect() as connection:
        cursor = connection.cursor()
        stats = review_stats(cursor)
        row = fetch_review_visa(cursor, "previous", before_id)
        if not row:
            return {"status": "empty", "current": None, "stats": stats}
        return build_response(cursor, row, stats)


def get_next_vn_visa_ocr_crop_single_review(after_id: int = 0) -> dict[str, Any]:
    with connect() as connection:
        cursor = connection.cursor()
        stats = review_stats(cursor)
        row = fetch_review_crop(cursor, "next", after_id)
        if not row and after_id > 0:
            row = fetch_review_crop(cursor, "next", 0)
        if not row:
            return {"status": "empty", "current": None, "stats": stats}
        return {"status": "ok", "current": build_single_crop_response(row), "stats": stats}


def get_previous_vn_visa_ocr_crop_single_review(before_id: int = 0) -> dict[str, Any]:
    if before_id <= 0:
        with connect() as connection:
            cursor = connection.cursor()
            return {"status": "empty", "current": None, "stats": review_stats(cursor)}

    with connect() as connection:
        cursor = connection.cursor()
        stats = review_stats(cursor)
        row = fetch_review_crop(cursor, "previous", before_id)
        if not row:
            return {"status": "empty", "current": None, "stats": stats}
        return {"status": "ok", "current": build_single_crop_response(row), "stats": stats}


def fetch_review_crop(cursor: Any, direction: str, anchor_id: int) -> dict[str, Any] | None:
    if direction == "previous":
        comparator = "<"
        order = "DESC"
    else:
        comparator = ">"
        order = "ASC"

    if anchor_id > 0:
        cursor.execute(
            f"""
            SELECT TOP 1
                c.*,
                vi.OriginalFileName AS VisaOriginalFileName,
                vi.SourceKey AS VisaSourceKey
            FROM dbo.readmrz_vn_visa_ocr_crops c
            INNER JOIN dbo.readmrz_vn_visa_items vi ON vi.Id = c.VisaItemId
            WHERE c.Id {comparator} ?
              AND c.ReviewStatus = N'pending'
            ORDER BY c.Id {order}
            """,
            anchor_id,
        )
    else:
        cursor.execute(
            """
            SELECT TOP 1
                c.*,
                vi.OriginalFileName AS VisaOriginalFileName,
                vi.SourceKey AS VisaSourceKey
            FROM dbo.readmrz_vn_visa_ocr_crops c
            INNER JOIN dbo.readmrz_vn_visa_items vi ON vi.Id = c.VisaItemId
            WHERE c.ReviewStatus = N'pending'
            ORDER BY c.Id ASC
            """
        )
    return fetch_one_dict(cursor)


def build_single_crop_response(row: dict[str, Any]) -> dict[str, Any]:
    output_dir = crop_output_dir()
    crop_path = output_dir / str(row.get("CropRelativePath") or "")
    crop_base64 = ""
    crop_content_type = mimetypes.guess_type(str(crop_path))[0] or "image/jpeg"
    if crop_path.is_file():
        crop_base64 = base64.b64encode(crop_path.read_bytes()).decode("ascii")

    return {
        "id": int(row["Id"]),
        "visa_item_id": int(row["VisaItemId"]),
        "source_key": row.get("SourceKey") or row.get("VisaSourceKey") or "",
        "visa_image_name": row.get("VisaOriginalFileName") or "",
        "field_name": row.get("FieldName") or "",
        "crop_relative_path": row.get("CropRelativePath") or "",
        "crop_content_type": crop_content_type,
        "crop_base64": crop_base64,
        "crop_width": int(row.get("CropWidth") or 0),
        "crop_height": int(row.get("CropHeight") or 0),
        "bbox": parse_bbox(row.get("BboxJson")),
        "ocr_raw_text": row.get("OcrRawText") or "",
        "ocr_score": float(row.get("OcrScore") or 0.0),
        "review_status": row.get("ReviewStatus") or "pending",
        "reviewed_text": row.get("ReviewedText") or "",
    }


def fetch_review_visa(cursor: Any, direction: str, anchor_id: int) -> dict[str, Any] | None:
    if direction == "previous":
        comparator = "<"
        order = "DESC"
    else:
        comparator = ">"
        order = "ASC"

    if anchor_id > 0:
        cursor.execute(
            f"""
            SELECT TOP 1 vi.*
            FROM dbo.readmrz_vn_visa_items vi
            WHERE vi.Id {comparator} ?
              AND EXISTS (
                  SELECT 1
                  FROM dbo.readmrz_vn_visa_ocr_crops c
                  WHERE c.VisaItemId = vi.Id
                    AND c.ReviewStatus = N'pending'
              )
            ORDER BY vi.Id {order}
            """,
            anchor_id,
        )
    else:
        cursor.execute(
            """
            SELECT TOP 1 vi.*
            FROM dbo.readmrz_vn_visa_items vi
            WHERE EXISTS (
                SELECT 1
                FROM dbo.readmrz_vn_visa_ocr_crops c
                WHERE c.VisaItemId = vi.Id
                  AND c.ReviewStatus = N'pending'
            )
            ORDER BY vi.Id ASC
            """
        )
    return fetch_one_dict(cursor)


def build_response(cursor: Any, visa_row: dict[str, Any], stats: dict[str, int]) -> dict[str, Any]:
    visa_id = int(visa_row["Id"])
    image_path = resolve_visa_image_path(visa_row)
    image_base64, content_type, width, height = image_to_base64(image_path)
    crops = fetch_crops(cursor, visa_id)

    return {
        "status": "ok",
        "current": {
            "id": visa_id,
            "key": visa_row.get("SourceKey"),
            "image_name": visa_row.get("OriginalFileName") or image_path.name,
            "image_content_type": content_type,
            "image_base64": image_base64,
            "image_width": width,
            "image_height": height,
            "review_status": visa_row.get("ReviewStatus"),
            "crops": crops,
        },
        "stats": stats,
    }


def resolve_visa_image_path(row: dict[str, Any]) -> Path:
    root = visa_image_dir()
    for key in ("OrientedRelativeImagePath", "RelativeImagePath", "SourceKey"):
        relative_path = str(row.get(key) or "").strip()
        if not relative_path:
            continue
        direct = root / relative_path
        if direct.is_file():
            return direct

    file_name = str(row.get("OriginalFileName") or Path(str(row.get("SourceKey") or "")).name).strip()
    if file_name:
        direct = root / file_name
        if direct.is_file():
            return direct
        matches = list(root.rglob(file_name))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Cannot find visa image for item {row.get('Id')}")


def fetch_crops(cursor: Any, visa_item_id: int) -> list[dict[str, Any]]:
    cursor.execute(
        """
        SELECT *
        FROM dbo.readmrz_vn_visa_ocr_crops
        WHERE VisaItemId = ?
        ORDER BY Id ASC
        """,
        visa_item_id,
    )
    rows = [row_to_dict(cursor, row) for row in cursor.fetchall()]
    output_dir = crop_output_dir()
    items: list[dict[str, Any]] = []
    for row in rows:
        crop_path = output_dir / str(row.get("CropRelativePath") or "")
        crop_base64 = ""
        crop_content_type = mimetypes.guess_type(str(crop_path))[0] or "image/jpeg"
        if crop_path.is_file():
            crop_base64 = base64.b64encode(crop_path.read_bytes()).decode("ascii")

        items.append(
            {
                "id": int(row["Id"]),
                "visa_item_id": int(row["VisaItemId"]),
                "field_name": row.get("FieldName"),
                "crop_relative_path": row.get("CropRelativePath"),
                "crop_content_type": crop_content_type,
                "crop_base64": crop_base64,
                "crop_width": int(row.get("CropWidth") or 0),
                "crop_height": int(row.get("CropHeight") or 0),
                "bbox": parse_bbox(row.get("BboxJson")),
                "ocr_raw_text": row.get("OcrRawText") or "",
                "ocr_score": float(row.get("OcrScore") or 0.0),
                "review_status": row.get("ReviewStatus") or "pending",
                "reviewed_text": row.get("ReviewedText") or "",
            }
        )
    return items


def parse_bbox(raw_value: Any) -> list[float]:
    if not raw_value:
        return []
    try:
        payload = json.loads(str(raw_value))
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict) and isinstance(payload.get("xyxy"), list):
        return [float(value) for value in payload["xyxy"][:4]]
    if isinstance(payload, list):
        return [float(value) for value in payload[:4]]
    return []


def save_vn_visa_ocr_crop_text(crop_id: int, reviewed_text: str, decision: str = "approved") -> dict[str, Any]:
    if crop_id <= 0:
        raise ValueError("id is required")
    if decision not in {"pending", "approved", "rejected"}:
        raise ValueError("decision must be pending, approved, or rejected")

    with connect() as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT VisaItemId FROM dbo.readmrz_vn_visa_ocr_crops WHERE Id = ?", crop_id)
        row = cursor.fetchone()
        if not row:
            raise KeyError(f"Crop not found: {crop_id}")
        visa_item_id = int(row[0])

        reviewed_value = reviewed_text.strip()
        cursor.execute(
            """
            UPDATE dbo.readmrz_vn_visa_ocr_crops
            SET ReviewStatus = ?,
                ReviewedText = ?,
                ReviewedAt = CASE WHEN ? IN (N'approved', N'rejected') THEN SYSDATETIME() ELSE ReviewedAt END,
                UpdatedDate = SYSDATETIME()
            WHERE Id = ?
            """,
            decision,
            reviewed_value if reviewed_value else None,
            decision,
            crop_id,
        )
        connection.commit()

    return get_vn_visa_ocr_crop_review_by_id(visa_item_id)


def save_vn_visa_ocr_crop_single_text(crop_id: int, reviewed_text: str, decision: str = "approved") -> dict[str, Any]:
    if crop_id <= 0:
        raise ValueError("id is required")
    if decision not in {"pending", "approved", "rejected"}:
        raise ValueError("decision must be pending, approved, or rejected")

    with connect() as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT Id FROM dbo.readmrz_vn_visa_ocr_crops WHERE Id = ?", crop_id)
        row = cursor.fetchone()
        if not row:
            raise KeyError(f"Crop not found: {crop_id}")

        reviewed_value = reviewed_text.strip()
        cursor.execute(
            """
            UPDATE dbo.readmrz_vn_visa_ocr_crops
            SET ReviewStatus = ?,
                ReviewedText = ?,
                ReviewedAt = CASE WHEN ? IN (N'approved', N'rejected') THEN SYSDATETIME() ELSE ReviewedAt END,
                UpdatedDate = SYSDATETIME()
            WHERE Id = ?
            """,
            decision,
            reviewed_value if reviewed_value else None,
            decision,
            crop_id,
        )
        connection.commit()

    next_response = get_next_vn_visa_ocr_crop_single_review(crop_id)
    return {
        "status": "ok",
        "id": crop_id,
        "decision": decision,
        "next": next_response.get("current"),
        "stats": next_response.get("stats", {}),
    }


def approve_vn_visa_ocr_crop_visa(visa_item_id: int, crops: list[dict[str, Any]]) -> dict[str, Any]:
    if visa_item_id <= 0:
        raise ValueError("visa_item_id is required")

    crop_texts: dict[int, str] = {}
    for item in crops:
        try:
            crop_id = int(item.get("id") or 0)
        except (TypeError, ValueError):
            crop_id = 0
        if crop_id > 0:
            crop_texts[crop_id] = str(item.get("reviewed_text") or item.get("text") or "").strip()

    with connect() as connection:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT Id, OcrRawText FROM dbo.readmrz_vn_visa_ocr_crops WHERE VisaItemId = ?",
            visa_item_id,
        )
        rows = [(int(row[0]), str(row[1] or "")) for row in cursor.fetchall()]
        if not rows:
            raise KeyError(f"No crops found for visa_item_id={visa_item_id}")

        for crop_id, ocr_text in rows:
            reviewed_text = crop_texts.get(crop_id, ocr_text).strip()
            cursor.execute(
                """
                UPDATE dbo.readmrz_vn_visa_ocr_crops
                SET ReviewStatus = N'approved',
                    ReviewedText = ?,
                    ReviewedAt = SYSDATETIME(),
                    UpdatedDate = SYSDATETIME()
                WHERE Id = ?
                """,
                reviewed_text if reviewed_text else None,
                crop_id,
            )
        connection.commit()

    next_response = get_next_vn_visa_ocr_crop_review(visa_item_id)
    return {
        "status": "ok",
        "visa_item_id": visa_item_id,
        "next": next_response.get("current"),
        "stats": next_response.get("stats", {}),
    }


def get_vn_visa_ocr_crop_review_by_id(visa_item_id: int) -> dict[str, Any]:
    with connect() as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM dbo.readmrz_vn_visa_items WHERE Id = ?", visa_item_id)
        row = fetch_one_dict(cursor)
        if not row:
            raise KeyError(f"Visa item not found: {visa_item_id}")
        stats = review_stats(cursor)
        return build_response(cursor, row, stats)
