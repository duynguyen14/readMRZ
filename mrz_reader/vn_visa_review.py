from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

import cv2

from .db import connect
from .env_config import env_value, read_env_file


def vn_visa_output_dir() -> Path:
    env = read_env_file()
    raw_output_dir = env_value(env, "READMRZ_VN_VISA_OUTPUT_DIR", "")
    if not raw_output_dir:
        raise ValueError("Missing READMRZ_VN_VISA_OUTPUT_DIR in .env")
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
            SUM(CASE WHEN Status = 'mapped' THEN 1 ELSE 0 END) AS total,
            SUM(CASE WHEN Status = 'mapped' AND ReviewStatus IN ('pending', 'needs_review') THEN 1 ELSE 0 END) AS pending,
            SUM(CASE WHEN Status = 'mapped' AND ReviewStatus = 'approved' THEN 1 ELSE 0 END) AS approved,
            SUM(CASE WHEN Status = 'mapped' AND ReviewStatus = 'rejected' THEN 1 ELSE 0 END) AS rejected
        FROM dbo.readmrz_vn_visa_items
        """
    )
    row = fetch_one_dict(cursor) or {}
    return {key: int(row.get(key) or 0) for key in ("total", "pending", "approved", "rejected")}


def get_next_vn_visa_review_item(after_key: str = "") -> dict[str, Any]:
    output_dir = vn_visa_output_dir()
    with connect() as connection:
        cursor = connection.cursor()
        
        # Determine the starting key
        if not after_key:
            cursor.execute(
                """
                SELECT TOP 1 SourceKey 
                FROM dbo.readmrz_vn_visa_items 
                WHERE Status = 'mapped' AND ReviewStatus IN ('pending', 'needs_review')
                ORDER BY SourceKey ASC
                """
            )
            row = cursor.fetchone()
            after_key = row[0] if row else ""

        stats = review_stats(cursor)
        
        if not after_key:
            return {"status": "empty", "current": None, "stats": stats}
            
        cursor.execute(
            """
            SELECT TOP 1 *
            FROM dbo.readmrz_vn_visa_items
            WHERE SourceKey >= ?
              AND Status = 'mapped'
              AND ReviewStatus IN ('pending', 'needs_review')
            ORDER BY SourceKey ASC
            """,
            after_key,
        )
        row_data = fetch_one_dict(cursor)
        
        if not row_data:
            return {"status": "empty", "current": None, "stats": stats}
            
        return build_response(cursor, row_data, stats, output_dir)


def get_previous_vn_visa_review_item(before_key: str = "") -> dict[str, Any]:
    output_dir = vn_visa_output_dir()
    with connect() as connection:
        cursor = connection.cursor()
        
        if not before_key:
            return {"status": "empty", "current": None, "stats": review_stats(cursor)}
            
        cursor.execute(
            """
            SELECT TOP 1 *
            FROM dbo.readmrz_vn_visa_items
            WHERE SourceKey < ?
              AND Status = 'mapped'
            ORDER BY SourceKey DESC
            """,
            before_key,
        )
        row_data = fetch_one_dict(cursor)
        stats = review_stats(cursor)
        
        if not row_data:
            return {"status": "empty", "current": None, "stats": stats}
            
        return build_response(cursor, row_data, stats, output_dir)


def build_response(cursor: Any, row: dict[str, Any], stats: dict[str, int], output_dir: Path) -> dict[str, Any]:
    image_path = output_dir / row["RelativeImagePath"]
    if not image_path.exists():
        # Fallback to OriginalFileName if needed or just error
        pass
        
    image_base64, content_type, width, height = image_to_base64(image_path)
    
    data_json_str = row.get("DataJson")
    data_json = {}
    if data_json_str:
        try:
            data_json = json.loads(data_json_str)
        except json.JSONDecodeError:
            pass
            
    raw_ocr_json_str = row.get("RawOcrJson")
    raw_ocr_json = {}
    if raw_ocr_json_str:
        try:
            raw_ocr_json = json.loads(raw_ocr_json_str)
        except json.JSONDecodeError:
            pass

    return {
        "status": "ok",
        "current": {
            "key": row["SourceKey"],
            "image_name": row.get("OriginalFileName") or image_path.name,
            "image_content_type": content_type,
            "image_base64": image_base64,
            "image_width": width,
            "image_height": height,
            "data_json": data_json,
            "raw_ocr_json": raw_ocr_json,
            "review_status": row.get("ReviewStatus"),
        },
        "stats": stats,
    }


def submit_vn_visa_review_decision(key: str, decision: str) -> dict[str, Any]:
    if decision not in {"approved", "rejected"}:
        raise ValueError("decision must be approved or rejected")

    with connect() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            UPDATE dbo.readmrz_vn_visa_items
            SET ReviewStatus = ?,
                UpdatedDate = SYSDATETIME()
            WHERE SourceKey = ?
            """,
            decision,
            key,
        )
        connection.commit()

    return get_next_vn_visa_review_item(key + "\0")  # Get next item strictly greater than key


def correct_vn_visa_review_field(key: str, field_name: str, bbox_xyxy: list[float], normalized_value: str) -> dict[str, Any]:
    with connect() as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT DataJson FROM dbo.readmrz_vn_visa_items WHERE SourceKey = ?", key)
        row = cursor.fetchone()
        if not row:
            raise KeyError(f"Item not found: {key}")
            
        data_json_str = row[0]
        data_json = json.loads(data_json_str) if data_json_str else {"schema_version": 1, "fields": {}, "missing_fields": []}
        
        if "fields" not in data_json:
            data_json["fields"] = {}
            
        # Update or create the field
        field_data = data_json["fields"].get(field_name, {})
        field_data["bbox"] = bbox_xyxy
        field_data["normalized_value"] = normalized_value
        field_data["raw_text"] = field_data.get("raw_text", normalized_value)
        field_data["bbox_source"] = "manual_review"
        
        data_json["fields"][field_name] = field_data
        
        # Remove from missing_fields if present
        if "missing_fields" in data_json and field_name in data_json["missing_fields"]:
            data_json["missing_fields"].remove(field_name)
            
        updated_data_json_str = json.dumps(data_json, ensure_ascii=False)
        
        cursor.execute(
            """
            UPDATE dbo.readmrz_vn_visa_items
            SET DataJson = ?,
                UpdatedDate = SYSDATETIME()
            WHERE SourceKey = ?
            """,
            updated_data_json_str,
            key,
        )
        connection.commit()
        
    # Return the refreshed item
    output_dir = vn_visa_output_dir()
    with connect() as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM dbo.readmrz_vn_visa_items WHERE SourceKey = ?", key)
        row_data = fetch_one_dict(cursor)
        stats = review_stats(cursor)
        return build_response(cursor, row_data, stats, output_dir)
