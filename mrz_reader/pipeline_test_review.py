from __future__ import annotations

import base64
from datetime import date, datetime
from pathlib import Path
from typing import Any

import cv2

from .db import connect


ALLOWED_FILTERS = {"all", "match", "mismatch", "error", "fallback"}


def row_to_dict(cursor: Any, row: Any) -> dict[str, Any]:
    columns = [column[0] for column in cursor.description]
    return dict(zip(columns, row))


def scalar_int(cursor: Any) -> int:
    row = cursor.fetchone()
    return int(row[0] or 0) if row else 0


def json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    return value


def image_thumbnail_payload(path_value: Any) -> dict[str, Any] | None:
    path = Path(str(path_value or ""))
    if not path.is_file():
        return None

    image = cv2.imread(str(path))
    if image is None:
        return None

    height, width = image.shape[:2]
    max_side = 900
    scale = min(1.0, max_side / max(1, width, height))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        height, width = image.shape[:2]

    success, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not success:
        return None

    return {
        "content_type": "image/jpeg",
        "image_base64": base64.b64encode(buffer.tobytes()).decode("ascii"),
        "width": width,
        "height": height,
    }


def build_where_clause(filter_name: str) -> str:
    if filter_name == "match":
        return "WHERE IsFullMatch = 1"
    if filter_name == "mismatch":
        return "WHERE IsFullMatch = 0 AND ErrorMessage IS NULL"
    if filter_name == "error":
        return "WHERE ErrorMessage IS NOT NULL"
    if filter_name == "fallback":
        return "WHERE UsedFallback = 1"
    return ""


def review_stats(cursor: Any) -> dict[str, int]:
    cursor.execute("SELECT COUNT(*) FROM dbo.readmrz_pipeline_test_items")
    total = scalar_int(cursor)
    cursor.execute("SELECT COUNT(*) FROM dbo.readmrz_pipeline_test_items WHERE IsFullMatch = 1")
    matched = scalar_int(cursor)
    cursor.execute(
        "SELECT COUNT(*) FROM dbo.readmrz_pipeline_test_items WHERE IsFullMatch = 0 AND ErrorMessage IS NULL"
    )
    mismatched = scalar_int(cursor)
    cursor.execute("SELECT COUNT(*) FROM dbo.readmrz_pipeline_test_items WHERE ErrorMessage IS NOT NULL")
    errors = scalar_int(cursor)
    cursor.execute("SELECT COUNT(*) FROM dbo.readmrz_pipeline_test_items WHERE UsedFallback = 1")
    fallback = scalar_int(cursor)
    return {
        "total": total,
        "matched": matched,
        "mismatched": mismatched,
        "errors": errors,
        "fallback": fallback,
    }


def get_pipeline_test_review_items(
    *,
    filter_name: str = "all",
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    filter_name = filter_name if filter_name in ALLOWED_FILTERS else "all"
    limit = min(200, max(1, int(limit or 50)))
    offset = max(0, int(offset or 0))
    where_clause = build_where_clause(filter_name)

    with connect() as connection:
        cursor = connection.cursor()
        stats = review_stats(cursor)
        cursor.execute(
            f"""
            SELECT COUNT(*)
            FROM dbo.readmrz_pipeline_test_items
            {where_clause}
            """
        )
        total_filtered = scalar_int(cursor)
        cursor.execute(
            f"""
            SELECT *
            FROM dbo.readmrz_pipeline_test_items
            {where_clause}
            ORDER BY Id DESC
            OFFSET ? ROWS FETCH NEXT ? ROWS ONLY
            """,
            offset,
            limit,
        )
        rows = []
        for row in cursor.fetchall():
            item = {key: json_value(value) for key, value in row_to_dict(cursor, row).items()}
            item["OriginalImage"] = image_thumbnail_payload(item.get("ImagePath"))
            rows.append(item)

    return {
        "status": "ok",
        "filter": filter_name,
        "items": rows,
        "stats": stats,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "total": total_filtered,
        },
    }
