from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

import cv2

from .db import connect
from .env_config import env_value, read_env_file
from .mrz import normalize_mrz_text


def line_paths() -> dict[str, Path]:
    env = read_env_file()
    dataset_dir = Path(
        env_value(env, "READMRZ_OCR_LINE_DATASET_DIR", str(Path.cwd() / "generated_datasets" / "mrz_ocr_lines"))
    ).expanduser().resolve()
    image_base_dir = Path(env_value(env, "READMRZ_OCR_LINE_IMAGE_BASE_DIR", str(dataset_dir / "images"))).expanduser().resolve()
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
        raise ValueError(f"Cannot read OCR line image: {path}")
    height, width = image.shape[:2]
    content_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return encoded, content_type, width, height


def review_stats(cursor: Any) -> dict[str, int]:
    cursor.execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN review_status = 'pending' THEN 1 ELSE 0 END) AS pending,
            SUM(CASE WHEN review_status = 'approved' THEN 1 ELSE 0 END) AS approved,
            SUM(CASE WHEN review_status = 'rejected' THEN 1 ELSE 0 END) AS rejected
        FROM dbo.readmrz_ocr_line_items
        """
    )
    row = fetch_one_dict(cursor) or {}
    return {key: int(row.get(key) or 0) for key in ("total", "pending", "approved", "rejected")}


def next_pending_row(cursor: Any, after_id: int) -> dict[str, Any] | None:
    for min_id in (after_id, 0):
        cursor.execute(
            """
            SELECT TOP 1 *
            FROM dbo.readmrz_ocr_line_items
            WHERE id > ?
              AND review_status = 'pending'
            ORDER BY id ASC
            """,
            min_id,
        )
        row = fetch_one_dict(cursor)
        if row:
            return row
    return None


def build_review_item(row: dict[str, Any], image_base_dir: Path) -> dict[str, Any]:
    image_path = resolve_line_image(row.get("line_image_file_name"), image_base_dir)
    image_base64, content_type, width, height = image_to_base64(image_path)
    return {
        "id": int(row["id"]),
        "label_item_id": int(row["label_item_id"]),
        "source_key": row.get("source_key") or "",
        "split": row.get("split") or "",
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


def get_next_ocr_line_review_item(after_id: int = 0) -> dict[str, Any]:
    paths = line_paths()
    with connect() as connection:
        cursor = connection.cursor()
        row = next_pending_row(cursor, after_id)
        stats = review_stats(cursor)
        if row is None:
            return {
                "status": "empty",
                "current": None,
                "stats": stats,
            }
        return {
            "status": "ok",
            "current": build_review_item(row, paths["image_base_dir"]),
            "stats": stats,
        }


def submit_ocr_line_review_decision(line_id: int, decision: str, final_text: str = "") -> dict[str, Any]:
    if decision not in {"approved", "rejected"}:
        raise ValueError("decision must be approved or rejected")

    normalized_final = normalize_mrz_text(final_text) if final_text else None
    with connect() as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM dbo.readmrz_ocr_line_items WHERE id = ?", line_id)
        row = fetch_one_dict(cursor)
        if row is None:
            raise KeyError(f"OCR line item not found: {line_id}")

        if decision == "approved" and not normalized_final:
            normalized_final = row.get("normalized_text") or normalize_mrz_text(str(row.get("ocr_text") or ""))

        cursor.execute(
            """
            UPDATE dbo.readmrz_ocr_line_items
            SET review_status = ?,
                final_text = COALESCE(?, final_text),
                reviewed_at = SYSUTCDATETIME(),
                updated_at = SYSUTCDATETIME()
            WHERE id = ?
            """,
            decision,
            normalized_final,
            line_id,
        )
        connection.commit()

    next_response = get_next_ocr_line_review_item(line_id)
    return {
        "status": "ok",
        "id": line_id,
        "decision": decision,
        "final_text": normalized_final or "",
        "next": next_response.get("current"),
        "stats": next_response.get("stats", {}),
    }
