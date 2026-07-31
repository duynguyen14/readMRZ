from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

import cv2

from .db import connect
from .env_config import env_value, read_env_file
from .mrz import normalize_mrz_text


LINE2_REVIEW_TABLE = "dbo.readmrz_ocr_line_items2"


def line2_paths() -> dict[str, Path]:
    env = read_env_file()
    dataset_dir = Path(
        env_value(env, "READMRZ_OCR_LINE2_DATASET_DIR", str(Path.cwd() / "generated_datasets" / "mrz_ocr_lines2"))
    ).expanduser().resolve()
    image_base_dir = Path(env_value(env, "READMRZ_OCR_LINE2_IMAGE_BASE_DIR", str(dataset_dir / "images"))).expanduser().resolve()
    return {
        "dataset_dir": dataset_dir,
        "image_base_dir": image_base_dir,
    }


def row_to_dict(cursor: Any, row: Any) -> dict[str, Any]:
    columns = [column[0] for column in cursor.description]
    return dict(zip(columns, row))


def fetch_one_dict(cursor: Any) -> dict[str, Any] | None:
    row = cursor.fetchone()
    return row_to_dict(cursor, row) if row else None


def resolve_line_image(path_value: Any, image_base_dir: Path) -> Path:
    path = Path(str(path_value or ""))
    if path.is_absolute():
        return path
    return image_base_dir / path


def image_to_base64(path: Path) -> tuple[str, str, int, int]:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read OCR line2 image: {path}")
    height, width = image.shape[:2]
    content_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return encoded, content_type, width, height


def review_stats(cursor: Any) -> dict[str, int]:
    cursor.execute(
        """
        WITH grouped AS (
            SELECT
                label_item_id,
                SUM(CASE WHEN review_status = 'pending' THEN 1 ELSE 0 END) AS pending_count,
                SUM(CASE WHEN review_status = 'approved' THEN 1 ELSE 0 END) AS approved_count,
                SUM(CASE WHEN review_status = 'rejected' THEN 1 ELSE 0 END) AS rejected_count
            FROM dbo.readmrz_ocr_line_items2
            GROUP BY label_item_id
        )
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN pending_count > 0 THEN 1 ELSE 0 END) AS pending,
            SUM(CASE WHEN pending_count = 0 AND rejected_count = 0 AND approved_count > 0 THEN 1 ELSE 0 END) AS approved,
            SUM(CASE WHEN rejected_count > 0 THEN 1 ELSE 0 END) AS rejected
        FROM grouped
        """
    )
    row = fetch_one_dict(cursor) or {}
    return {key: int(row.get(key) or 0) for key in ("total", "pending", "approved", "rejected")}


def next_pending_label_id(cursor: Any, after_id: int) -> int | None:
    for min_id in (after_id, 0):
        cursor.execute(
            """
            SELECT TOP 1 label_item_id
            FROM dbo.readmrz_ocr_line_items2
            WHERE label_item_id > ?
              AND review_status = 'pending'
            GROUP BY label_item_id
            ORDER BY label_item_id ASC
            """,
            min_id,
        )
        row = cursor.fetchone()
        if row:
            return int(row[0])
    return None


def previous_label_id(cursor: Any, before_id: int) -> int | None:
    if before_id <= 0:
        cursor.execute("SELECT ISNULL(MAX(label_item_id) + 1, 0) FROM dbo.readmrz_ocr_line_items2")
        row = cursor.fetchone()
        before_id = int(row[0] or 0) if row else 0

    cursor.execute(
        """
        SELECT TOP 1 label_item_id
        FROM dbo.readmrz_ocr_line_items2
        WHERE label_item_id < ?
        GROUP BY label_item_id
        ORDER BY label_item_id DESC
        """,
        before_id,
    )
    row = cursor.fetchone()
    return int(row[0]) if row else None


def rows_for_label(cursor: Any, label_item_id: int) -> list[dict[str, Any]]:
    cursor.execute(
        """
        SELECT *
        FROM dbo.readmrz_ocr_line_items2
        WHERE label_item_id = ?
        ORDER BY line_index ASC, id ASC
        """,
        label_item_id,
    )
    return [row_to_dict(cursor, row) for row in cursor.fetchall()]


def build_line(row: dict[str, Any], image_base_dir: Path) -> dict[str, Any]:
    image_path = resolve_line_image(row.get("line_image_file_name"), image_base_dir)
    image_base64, content_type, width, height = image_to_base64(image_path)
    return {
        "id": int(row["id"]),
        "line_index": int(row.get("line_index") or 0),
        "line_image": str(image_path),
        "line_image_file_name": row.get("line_image_file_name") or "",
        "image_content_type": content_type,
        "image_base64": image_base64,
        "image_width": width,
        "image_height": height,
        "ocr_text": row.get("ocr_text") or "",
        "normalized_text": row.get("normalized_text") or "",
        "final_text": row.get("final_text") or row.get("normalized_text") or "",
        "ocr_score": row.get("ocr_score") or 0,
        "mrz_likeness": row.get("mrz_likeness") or 0,
        "review_status": row.get("review_status") or "pending",
    }


def build_review_item(rows: list[dict[str, Any]], image_base_dir: Path) -> dict[str, Any] | None:
    if not rows:
        return None
    first = rows[0]
    return {
        "label_item_id": int(first["label_item_id"]),
        "source_key": first.get("source_key") or "",
        "split": first.get("split") or "",
        "review_status": "pending" if any((row.get("review_status") or "") == "pending" for row in rows) else str(first.get("review_status") or ""),
        "lines": [build_line(row, image_base_dir) for row in rows],
    }


def get_next_ocr_line2_review_item(after_id: int = 0) -> dict[str, Any]:
    paths = line2_paths()
    with connect() as connection:
        cursor = connection.cursor()
        label_item_id = next_pending_label_id(cursor, after_id)
        stats = review_stats(cursor)
        if label_item_id is None:
            return {"status": "empty", "current": None, "stats": stats}
        return {
            "status": "ok",
            "current": build_review_item(rows_for_label(cursor, label_item_id), paths["image_base_dir"]),
            "stats": stats,
        }


def get_previous_ocr_line2_review_item(before_id: int = 0) -> dict[str, Any]:
    paths = line2_paths()
    with connect() as connection:
        cursor = connection.cursor()
        label_item_id = previous_label_id(cursor, before_id)
        stats = review_stats(cursor)
        if label_item_id is None:
            return {"status": "empty", "current": None, "stats": stats}
        return {
            "status": "ok",
            "current": build_review_item(rows_for_label(cursor, label_item_id), paths["image_base_dir"]),
            "stats": stats,
        }


def submit_ocr_line2_review_decision(label_item_id: int, decision: str, lines: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if decision not in {"approved", "rejected"}:
        raise ValueError("decision must be approved or rejected")

    final_text_by_id: dict[int, str] = {}
    for line in lines or []:
        line_id = int(line.get("id") or 0)
        if line_id > 0:
            final_text_by_id[line_id] = normalize_mrz_text(str(line.get("final_text") or ""))

    with connect() as connection:
        cursor = connection.cursor()
        rows = rows_for_label(cursor, label_item_id)
        if not rows:
            raise KeyError(f"OCR line2 group not found: {label_item_id}")

        for row in rows:
            line_id = int(row["id"])
            final_text = final_text_by_id.get(line_id)
            if decision == "approved" and not final_text:
                final_text = row.get("normalized_text") or normalize_mrz_text(str(row.get("ocr_text") or ""))
            cursor.execute(
                """
                UPDATE dbo.readmrz_ocr_line_items2
                SET review_status = ?,
                    final_text = COALESCE(?, final_text),
                    reviewed_at = SYSUTCDATETIME(),
                    updated_at = SYSUTCDATETIME()
                WHERE id = ?
                """,
                decision,
                final_text,
                line_id,
            )
        connection.commit()

    next_response = get_next_ocr_line2_review_item(label_item_id)
    return {
        "status": "ok",
        "label_item_id": label_item_id,
        "decision": decision,
        "next": next_response.get("current"),
        "stats": next_response.get("stats", {}),
    }
