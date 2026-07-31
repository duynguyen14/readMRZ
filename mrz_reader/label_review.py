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
        "image_base_dir": (dataset_dir / "images").expanduser().resolve(),
        "label_base_dir": (dataset_dir / "labels").expanduser().resolve(),
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


def resolve_optional_under_base(base_dir: Path, value: Any) -> Path | None:
    path_text = str(value or "").strip()
    if not path_text:
        return None
    return resolve_under_base(base_dir, path_text)


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


def clamp_bbox_xyxy(bbox_xyxy: list[Any], width: int, height: int) -> list[float]:
    if len(bbox_xyxy) != 4:
        raise ValueError("bbox_xyxy must have 4 values")
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    left = max(0.0, min(x1, x2, float(width)))
    top = max(0.0, min(y1, y2, float(height)))
    right = max(0.0, min(max(x1, x2), float(width)))
    bottom = max(0.0, min(max(y1, y2), float(height)))
    if right - left < 4 or bottom - top < 4:
        raise ValueError("bbox is too small")
    return [round(left, 2), round(top, 2), round(right, 2), round(bottom, 2)]


def yolo_label_from_bbox(bbox_xyxy: list[float], width: int, height: int) -> str:
    x1, y1, x2, y2 = bbox_xyxy
    x_center = ((x1 + x2) / 2.0) / width
    y_center = ((y1 + y2) / 2.0) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


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


def pending_candidates(cursor: Any, min_id: int, limit: int = 500) -> list[dict[str, Any]]:
    cursor.execute(
        """
        SELECT TOP (?) *
        FROM dbo.readmrz_label_items
        WHERE id > ?
          AND status = 'labeled'
          AND review_status = 'pending'
          AND image_file_name IS NOT NULL
          AND label_file_name IS NOT NULL
        ORDER BY id ASC
        """,
        limit,
        min_id,
    )
    return [row_to_dict(cursor, row) for row in cursor.fetchall()]


def row_artifacts_exist(row: dict[str, Any], paths: dict[str, Path]) -> bool:
    image_path, label_path = artifact_paths(row, paths)
    return image_path.is_file() and label_path.is_file()


def next_pending_row(cursor: Any, after_key: str, paths: dict[str, Path]) -> dict[str, Any] | None:
    start_id = after_id(cursor, after_key)
    for min_id in (start_id, 0):
        cursor_id = min_id
        while True:
            rows = pending_candidates(cursor, cursor_id)
            if not rows:
                break
            for row in rows:
                cursor_id = int(row["id"])
                if row_artifacts_exist(row, paths):
                    return row
    return None


def previous_candidates(cursor: Any, max_id: int, limit: int = 500) -> list[dict[str, Any]]:
    cursor.execute(
        """
        SELECT TOP (?) *
        FROM dbo.readmrz_label_items
        WHERE id < ?
          AND status = 'labeled'
          AND image_file_name IS NOT NULL
          AND label_file_name IS NOT NULL
        ORDER BY id DESC
        """,
        limit,
        max_id,
    )
    return [row_to_dict(cursor, row) for row in cursor.fetchall()]


def previous_review_row(cursor: Any, before_key: str, paths: dict[str, Path]) -> dict[str, Any] | None:
    start_id = after_id(cursor, before_key)
    if start_id <= 0:
        cursor.execute("SELECT ISNULL(MAX(id) + 1, 0) FROM dbo.readmrz_label_items WHERE status = 'labeled'")
        row = cursor.fetchone()
        start_id = int(row[0] or 0) if row else 0
    cursor_id = start_id
    while True:
        rows = previous_candidates(cursor, cursor_id)
        if not rows:
            break
        for row in rows:
            cursor_id = int(row["id"])
            if row_artifacts_exist(row, paths):
                return row
    return None


def label_position(cursor: Any, row_id: int) -> int:
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM dbo.readmrz_label_items
        WHERE status = 'labeled'
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


def artifact_paths(row: dict[str, Any], paths: dict[str, Path]) -> tuple[Path, Path]:
    image_path = resolve_under_base(paths["image_base_dir"], row.get("image_file_name"))
    label_path = resolve_under_base(paths["label_base_dir"], row.get("label_file_name"))
    if image_path.is_file() and label_path.is_file():
        return image_path, label_path

    rejected_image = resolve_optional_under_base(paths["dataset_dir"], row.get("rejected_image_file_name"))
    rejected_label = resolve_optional_under_base(paths["dataset_dir"], row.get("rejected_label_file_name"))
    if rejected_image and rejected_label and rejected_image.is_file() and rejected_label.is_file():
        return rejected_image, rejected_label

    return image_path, label_path


def build_review_item(
    cursor: Any,
    row: dict[str, Any],
    stats: dict[str, int],
    paths: dict[str, Path],
) -> dict[str, Any]:
    image_path, label_path = artifact_paths(row, paths)
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
        "review_status": row.get("review_status") or "",
        "position": label_position(cursor, int(row["id"])),
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


def get_previous_review_item(before_key: str = "") -> dict[str, Any]:
    paths = review_paths()
    with connect() as connection:
        cursor = connection.cursor()
        row = previous_review_row(cursor, before_key, paths)
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
    if image_path.is_file():
        target_image = unique_destination(rejected_image_dir / image_path.name)
        shutil.move(str(image_path), str(target_image))
        moved["rejected_image_file_name"] = relative_to_base(target_image, paths["dataset_dir"])
    if label_path.is_file():
        target_label = unique_destination(rejected_label_dir / label_path.name)
        shutil.move(str(label_path), str(target_label))
        moved["rejected_label_file_name"] = relative_to_base(target_label, paths["dataset_dir"])
    return moved


def restore_rejected_artifacts(row: dict[str, Any], paths: dict[str, Path]) -> dict[str, str]:
    rejected_image = resolve_optional_under_base(paths["dataset_dir"], row.get("rejected_image_file_name"))
    rejected_label = resolve_optional_under_base(paths["dataset_dir"], row.get("rejected_label_file_name"))
    target_image = resolve_under_base(paths["image_base_dir"], row.get("image_file_name"))
    target_label = resolve_under_base(paths["label_base_dir"], row.get("label_file_name"))

    restored: dict[str, str] = {}
    if rejected_image and rejected_image.is_file():
        target_image.parent.mkdir(parents=True, exist_ok=True)
        final_image = unique_destination(target_image)
        shutil.move(str(rejected_image), str(final_image))
        restored["image_file_name"] = relative_to_base(final_image, paths["image_base_dir"])
    if rejected_label and rejected_label.is_file():
        target_label.parent.mkdir(parents=True, exist_ok=True)
        final_label = unique_destination(target_label)
        shutil.move(str(rejected_label), str(final_label))
        restored["label_file_name"] = relative_to_base(final_label, paths["label_base_dir"])
    return restored


def ensure_train_artifacts(row: dict[str, Any], paths: dict[str, Path]) -> dict[str, str]:
    if str(row.get("review_status") or "") == "rejected":
        return restore_rejected_artifacts(row, paths)
    return {}


def write_yolo_label(label_path: Path, yolo_label: str) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(yolo_label + "\n", encoding="utf-8")


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
        restored = restore_rejected_artifacts(row, paths) if decision == "approved" else {}
        cursor.execute(
            """
            UPDATE dbo.readmrz_label_items
            SET review_status = ?,
                image_file_name = COALESCE(?, image_file_name),
                label_file_name = COALESCE(?, label_file_name),
                rejected_image_file_name = ?,
                rejected_label_file_name = ?,
                reviewed_at = SYSUTCDATETIME(),
                updated_at = SYSUTCDATETIME()
            WHERE source_key = ?
            """,
            decision,
            restored.get("image_file_name"),
            restored.get("label_file_name"),
            None if decision == "approved" else moved.get("rejected_image_file_name") or row.get("rejected_image_file_name"),
            None if decision == "approved" else moved.get("rejected_label_file_name") or row.get("rejected_label_file_name"),
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
        "restored": restored,
        "next": next_response.get("current"),
        "stats": next_response.get("stats", {}),
    }


def correct_review_box(key: str, bbox_xyxy: list[Any]) -> dict[str, Any]:
    paths = review_paths()
    with connect() as connection:
        cursor = connection.cursor()
        row = fetch_review_row(cursor, key)
        if row is None:
            raise KeyError(f"Review item not found: {key}")

        restored = ensure_train_artifacts(row, paths)
        if restored:
            row = {**row, **restored, "review_status": "approved"}

        image_path = resolve_under_base(paths["image_base_dir"], row.get("image_file_name"))
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image for corrected box: {image_path}")
        height, width = image.shape[:2]
        corrected_bbox = clamp_bbox_xyxy(bbox_xyxy, width, height)
        yolo_label = yolo_label_from_bbox(corrected_bbox, width, height)
        label_path = resolve_under_base(paths["label_base_dir"], row.get("label_file_name"))
        write_yolo_label(label_path, yolo_label)

        cursor.execute(
            """
            UPDATE dbo.readmrz_label_items
            SET review_status = 'approved',
                image_file_name = COALESCE(?, image_file_name),
                label_file_name = COALESCE(?, label_file_name),
                rejected_image_file_name = NULL,
                rejected_label_file_name = NULL,
                bbox_x1 = ?,
                bbox_y1 = ?,
                bbox_x2 = ?,
                bbox_y2 = ?,
                yolo_label = ?,
                reviewed_at = SYSUTCDATETIME(),
                updated_at = SYSUTCDATETIME()
            WHERE source_key = ?
            """,
            restored.get("image_file_name"),
            restored.get("label_file_name"),
            corrected_bbox[0],
            corrected_bbox[1],
            corrected_bbox[2],
            corrected_bbox[3],
            yolo_label,
            key,
        )
        cursor.execute(
            """
            INSERT INTO dbo.readmrz_label_review_history (label_item_id, source_key, decision, note)
            VALUES (?, ?, 'approved', ?)
            """,
            row["id"],
            key,
            "corrected_box",
        )
        connection.commit()

        refreshed = fetch_review_row(cursor, key)
        stats = review_stats(cursor)
        if refreshed is None:
            raise KeyError(f"Review item not found after correction: {key}")
        return {
            "status": "ok",
            "key": key,
            "corrected_bbox_xyxy": corrected_bbox,
            "yolo_label": yolo_label,
            "restored": restored,
            "current": build_review_item(cursor, refreshed, stats, paths),
            "stats": stats,
        }
