from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
import shutil
from typing import Any

import cv2

from .db import connect
from .env_config import env_value, read_env_file, yolo_dataset_dir


def review_paths() -> dict[str, Path]:
    env = read_env_file()
    dataset_dir = yolo_dataset_dir(env)
    return {
        "source_base_dir": Path(env_value(env, "READMRZ_SOURCE_IMAGE_DIR", str(Path.cwd()))).expanduser().resolve(),
        "dataset_dir": dataset_dir,
        "image_base_dir": Path(env_value(env, "READMRZ_YOLO_IMAGE_BASE_DIR", str(dataset_dir / "images"))).expanduser().resolve(),
        "label_base_dir": Path(env_value(env, "READMRZ_YOLO_LABEL_BASE_DIR", str(dataset_dir / "labels"))).expanduser().resolve(),
    }


def row_to_dict(cursor: Any, row: Any) -> dict[str, Any]:
    columns = [column[0] for column in cursor.description]
    return dict(zip(columns, row))


def fetch_one_dict(cursor: Any) -> dict[str, Any] | None:
    row = cursor.fetchone()
    return row_to_dict(cursor, row) if row else None


def resolve_under_base(base_dir: Path, value: Any) -> Path:
    path_text = str(value or "").strip()
    if not path_text:
        return base_dir
    path = Path(path_text)
    if path.is_absolute():
        return path
    return base_dir / path


def relative_to_base(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return path.name


def image_to_base64(path: Path) -> tuple[str, str, int, int]:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read review image: {path}")
    height, width = image.shape[:2]
    content_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return encoded, content_type, width, height


def bbox_percent(bbox_xyxy: list[float], width: int, height: int) -> dict[str, float]:
    x_min, y_min, x_max, y_max = [float(value) for value in bbox_xyxy]
    return {
        "left": round((x_min / width) * 100, 4),
        "top": round((y_min / height) * 100, 4),
        "width": round(((x_max - x_min) / width) * 100, 4),
        "height": round(((y_max - y_min) / height) * 100, 4),
    }


def review_stats(cursor: Any) -> dict[str, int]:
    cursor.execute(
        """
        SELECT
            SUM(CASE WHEN status = 'labeled' THEN 1 ELSE 0 END) AS total,
            SUM(CASE WHEN status = 'labeled' AND review_status = 'pending' THEN 1 ELSE 0 END) AS pending,
            SUM(CASE WHEN status = 'labeled' AND review_status = 'approved' THEN 1 ELSE 0 END) AS approved,
            SUM(CASE WHEN status = 'labeled' AND review_status = 'rejected' THEN 1 ELSE 0 END) AS rejected,
            SUM(CASE WHEN status = 'no_mrz' THEN 1 ELSE 0 END) AS no_mrz
        FROM dbo.readmrz_label_items
        """
    )
    row = fetch_one_dict(cursor) or {}
    return {key: int(row.get(key) or 0) for key in ("total", "pending", "approved", "rejected", "no_mrz")}


def after_id(cursor: Any, after_key: str) -> int:
    if not after_key:
        return 0
    cursor.execute("SELECT id FROM dbo.readmrz_label_items WHERE source_key = ?", after_key)
    row = cursor.fetchone()
    return int(row[0]) if row else 0


def pending_candidates(cursor: Any, min_id: int) -> list[dict[str, Any]]:
    cursor.execute(
        """
        SELECT TOP 50 *
        FROM dbo.readmrz_label_items
        WHERE id > ?
          AND status = 'labeled'
          AND review_status = 'pending'
          AND image_file_name IS NOT NULL
          AND label_file_name IS NOT NULL
        ORDER BY id ASC
        """,
        min_id,
    )
    return [row_to_dict(cursor, row) for row in cursor.fetchall()]


def row_artifacts_exist(row: dict[str, Any], paths: dict[str, Path]) -> bool:
    image_path = resolve_under_base(paths["image_base_dir"], row.get("image_file_name"))
    label_path = resolve_under_base(paths["label_base_dir"], row.get("label_file_name"))
    return image_path.exists() and label_path.exists()


def next_pending_row(cursor: Any, after_key: str, paths: dict[str, Path]) -> dict[str, Any] | None:
    start_id = after_id(cursor, after_key)
    for min_id in (start_id, 0):
        for row in pending_candidates(cursor, min_id):
            if row_artifacts_exist(row, paths):
                return row
    return None


def pending_position(cursor: Any, row_id: int) -> int:
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM dbo.readmrz_label_items
        WHERE status = 'labeled'
          AND review_status = 'pending'
          AND id <= ?
        """,
        row_id,
    )
    row = cursor.fetchone()
    return int(row[0] or 0) if row else 0


def decode_mrz_lines(value: Any) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(line) for line in parsed]


def build_review_item(
    cursor: Any,
    row: dict[str, Any],
    stats: dict[str, int],
    paths: dict[str, Path],
) -> dict[str, Any]:
    image_path = resolve_under_base(paths["image_base_dir"], row.get("image_file_name"))
    label_path = resolve_under_base(paths["label_base_dir"], row.get("label_file_name"))
    image_base64, content_type, width, height = image_to_base64(image_path)
    bbox_xyxy = [row.get("bbox_x1"), row.get("bbox_y1"), row.get("bbox_x2"), row.get("bbox_y2")]
    if any(value is None for value in bbox_xyxy):
        raise ValueError(f"Missing bbox for review item: {row.get('source_key')}")
    bbox_values = [float(value) for value in bbox_xyxy]
    source_path = resolve_under_base(paths["source_base_dir"], row.get("source_key"))

    return {
        "key": row.get("source_key", ""),
        "source": str(source_path),
        "split": row.get("split") or "",
        "output_image": str(image_path),
        "output_label": str(label_path),
        "image_name": image_path.name,
        "image_content_type": content_type,
        "image_base64": image_base64,
        "image_width": width,
        "image_height": height,
        "bbox_xyxy": bbox_values,
        "bbox_percent": bbox_percent(bbox_values, width, height),
        "yolo_label": row.get("yolo_label") or "",
        "mrz_lines": decode_mrz_lines(row.get("mrz_lines_json")),
        "mrz_score": row.get("mrz_score") or 0,
        "ocr_ms": row.get("ocr_ms") or 0,
        "position": pending_position(cursor, int(row["id"])),
        "stats": stats,
    }


def get_next_review_item(after_key: str = "") -> dict[str, Any]:
    paths = review_paths()
    with connect() as connection:
        cursor = connection.cursor()
        row = next_pending_row(cursor, after_key, paths)
        stats = review_stats(cursor)
        if row is None:
            return {
                "status": "empty",
                "current": None,
                "stats": stats,
            }
        return {
            "status": "ok",
            "current": build_review_item(cursor, row, stats, paths),
            "stats": stats,
        }


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    suffix = path.suffix
    stem = path.stem
    parent = path.parent
    for index in range(1, 10000):
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot create unique destination for {path}")


def move_rejected_artifacts(row: dict[str, Any], paths: dict[str, Path]) -> dict[str, str]:
    image_path = resolve_under_base(paths["image_base_dir"], row.get("image_file_name"))
    label_path = resolve_under_base(paths["label_base_dir"], row.get("label_file_name"))
    split = str(row.get("split") or "unknown")
    rejected_image_dir = paths["dataset_dir"] / "review" / "rejected" / "images" / split
    rejected_label_dir = paths["dataset_dir"] / "review" / "rejected" / "labels" / split
    rejected_image_dir.mkdir(parents=True, exist_ok=True)
    rejected_label_dir.mkdir(parents=True, exist_ok=True)

    moved: dict[str, str] = {}
    if image_path.exists():
        target_image = unique_destination(rejected_image_dir / image_path.name)
        shutil.move(str(image_path), str(target_image))
        moved["rejected_image_file_name"] = relative_to_base(target_image, paths["dataset_dir"])
    if label_path.exists():
        target_label = unique_destination(rejected_label_dir / label_path.name)
        shutil.move(str(label_path), str(target_label))
        moved["rejected_label_file_name"] = relative_to_base(target_label, paths["dataset_dir"])
    return moved


def fetch_review_row(cursor: Any, key: str) -> dict[str, Any] | None:
    cursor.execute("SELECT * FROM dbo.readmrz_label_items WHERE source_key = ?", key)
    return fetch_one_dict(cursor)


def submit_review_decision(key: str, decision: str) -> dict[str, Any]:
    if decision not in {"approved", "rejected"}:
        raise ValueError("decision must be approved or rejected")

    paths = review_paths()
    with connect() as connection:
        cursor = connection.cursor()
        row = fetch_review_row(cursor, key)
        if row is None:
            raise KeyError(f"Review item not found: {key}")

        moved = move_rejected_artifacts(row, paths) if decision == "rejected" else {}
        cursor.execute(
            """
            UPDATE dbo.readmrz_label_items
            SET review_status = ?,
                rejected_image_file_name = COALESCE(?, rejected_image_file_name),
                rejected_label_file_name = COALESCE(?, rejected_label_file_name),
                reviewed_at = SYSUTCDATETIME(),
                updated_at = SYSUTCDATETIME()
            WHERE source_key = ?
            """,
            decision,
            moved.get("rejected_image_file_name"),
            moved.get("rejected_label_file_name"),
            key,
        )
        cursor.execute(
            """
            INSERT INTO dbo.readmrz_label_review_history (label_item_id, source_key, decision)
            VALUES (?, ?, ?)
            """,
            row["id"],
            key,
            decision,
        )
        connection.commit()

    next_response = get_next_review_item(key)
    return {
        "status": "ok",
        "decision": decision,
        "key": key,
        "moved": moved,
        "next": next_response.get("current"),
        "stats": next_response.get("stats", {}),
    }
