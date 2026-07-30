from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mrz_reader.db import connect
from mrz_reader.env_config import env_value, read_env_file, yolo_dataset_dir


def parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def load_processed(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"processed.json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_base_dir(env: dict[str, str], key: str, default_path: Path) -> Path:
    return Path(env_value(env, key, str(default_path))).expanduser().resolve()


def relative_or_name(path_value: Any, base_dir: Path | None = None) -> str | None:
    if not path_value:
        return None
    path = Path(str(path_value)).expanduser()
    if base_dir is not None:
        try:
            return path.resolve().relative_to(base_dir.resolve()).as_posix()
        except ValueError:
            pass
    return path.name


def bbox_values(item: dict[str, Any]) -> tuple[float | None, float | None, float | None, float | None]:
    bbox = item.get("bbox_xyxy") or []
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None, None, None, None
    return tuple(float(value) for value in bbox)  # type: ignore[return-value]


def normalize_review_status(item: dict[str, Any]) -> str:
    review_status = str(item.get("review_status") or "").strip().lower()
    if review_status in {"approved", "rejected"}:
        return review_status
    return "pending"


def normalize_status(item: dict[str, Any]) -> str:
    status = str(item.get("status") or "error").strip().lower()
    if status in {"labeled", "no_mrz", "error"}:
        return status
    return "error"


def upsert_item(cursor: Any, row: dict[str, Any]) -> None:
    cursor.execute(
        """
        UPDATE dbo.readmrz_label_items
        SET
            source_file_name = ?,
            status = ?,
            review_status = ?,
            split = ?,
            image_file_name = ?,
            label_file_name = ?,
            rejected_image_file_name = ?,
            rejected_label_file_name = ?,
            bbox_x1 = ?,
            bbox_y1 = ?,
            bbox_x2 = ?,
            bbox_y2 = ?,
            yolo_label = ?,
            mrz_lines_json = ?,
            mrz_score = ?,
            ocr_ms = ?,
            elapsed_ms = ?,
            fingerprint_size = ?,
            fingerprint_mtime_ns = ?,
            ocr_engine = ?,
            ocr_config_json = ?,
            error_message = ?,
            processed_at = ?,
            reviewed_at = ?,
            updated_at = SYSUTCDATETIME()
        WHERE source_key = ?
        """,
        row["source_file_name"],
        row["status"],
        row["review_status"],
        row["split"],
        row["image_file_name"],
        row["label_file_name"],
        row["rejected_image_file_name"],
        row["rejected_label_file_name"],
        row["bbox_x1"],
        row["bbox_y1"],
        row["bbox_x2"],
        row["bbox_y2"],
        row["yolo_label"],
        row["mrz_lines_json"],
        row["mrz_score"],
        row["ocr_ms"],
        row["elapsed_ms"],
        row["fingerprint_size"],
        row["fingerprint_mtime_ns"],
        row["ocr_engine"],
        row["ocr_config_json"],
        row["error_message"],
        row["processed_at"],
        row["reviewed_at"],
        row["source_key"],
    )
    if cursor.rowcount:
        return

    cursor.execute(
        """
        INSERT INTO dbo.readmrz_label_items (
            source_key,
            source_file_name,
            status,
            review_status,
            split,
            image_file_name,
            label_file_name,
            rejected_image_file_name,
            rejected_label_file_name,
            bbox_x1,
            bbox_y1,
            bbox_x2,
            bbox_y2,
            yolo_label,
            mrz_lines_json,
            mrz_score,
            ocr_ms,
            elapsed_ms,
            fingerprint_size,
            fingerprint_mtime_ns,
            ocr_engine,
            ocr_config_json,
            error_message,
            processed_at,
            reviewed_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        row["source_key"],
        row["source_file_name"],
        row["status"],
        row["review_status"],
        row["split"],
        row["image_file_name"],
        row["label_file_name"],
        row["rejected_image_file_name"],
        row["rejected_label_file_name"],
        row["bbox_x1"],
        row["bbox_y1"],
        row["bbox_x2"],
        row["bbox_y2"],
        row["yolo_label"],
        row["mrz_lines_json"],
        row["mrz_score"],
        row["ocr_ms"],
        row["elapsed_ms"],
        row["fingerprint_size"],
        row["fingerprint_mtime_ns"],
        row["ocr_engine"],
        row["ocr_config_json"],
        row["error_message"],
        row["processed_at"],
        row["reviewed_at"],
    )


def build_row(
    *,
    source_key: str,
    item: dict[str, Any],
    processed: dict[str, Any],
    dataset_dir: Path,
    source_base_dir: Path,
    image_base_dir: Path,
    label_base_dir: Path,
) -> dict[str, Any]:
    fingerprint = item.get("fingerprint") if isinstance(item.get("fingerprint"), dict) else {}
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox_values(item)
    normalized_source_key = relative_or_name(item.get("source") or source_key, source_base_dir) or Path(source_key).name
    return {
        "source_key": normalized_source_key,
        "source_file_name": Path(normalized_source_key).name,
        "status": normalize_status(item),
        "review_status": normalize_review_status(item),
        "split": item.get("split"),
        "image_file_name": relative_or_name(item.get("output_image"), image_base_dir),
        "label_file_name": relative_or_name(item.get("output_label"), label_base_dir),
        "rejected_image_file_name": relative_or_name(item.get("rejected_image"), dataset_dir),
        "rejected_label_file_name": relative_or_name(item.get("rejected_label"), dataset_dir),
        "bbox_x1": bbox_x1,
        "bbox_y1": bbox_y1,
        "bbox_x2": bbox_x2,
        "bbox_y2": bbox_y2,
        "yolo_label": item.get("yolo_label"),
        "mrz_lines_json": json.dumps(item.get("mrz_lines") or [], ensure_ascii=False),
        "mrz_score": item.get("mrz_score"),
        "ocr_ms": item.get("ocr_ms"),
        "elapsed_ms": item.get("elapsed_ms"),
        "fingerprint_size": fingerprint.get("size"),
        "fingerprint_mtime_ns": fingerprint.get("mtime_ns"),
        "ocr_engine": processed.get("ocr_engine"),
        "ocr_config_json": json.dumps(processed.get("ocr_config") or {}, ensure_ascii=False),
        "error_message": item.get("error"),
        "processed_at": parse_datetime(item.get("processed_at")),
        "reviewed_at": parse_datetime(item.get("reviewed_at")),
    }


def main() -> int:
    env = read_env_file()
    dataset_dir = yolo_dataset_dir(env)
    processed_path = dataset_dir / "processed.json"
    source_base_dir = resolve_base_dir(env, "READMRZ_SOURCE_IMAGE_DIR", PROJECT_ROOT)
    image_base_dir = resolve_base_dir(env, "READMRZ_YOLO_IMAGE_BASE_DIR", dataset_dir / "images")
    label_base_dir = resolve_base_dir(env, "READMRZ_YOLO_LABEL_BASE_DIR", dataset_dir / "labels")
    batch_size = max(1, int(env_value(env, "READMRZ_DB_IMPORT_BATCH_SIZE", "500")))
    processed = load_processed(processed_path)
    items = processed.get("items", {})
    if not isinstance(items, dict):
        raise ValueError("processed.json items must be an object")

    imported = 0
    with connect() as connection:
        cursor = connection.cursor()
        cursor.fast_executemany = False
        for index, (source_key, item) in enumerate(items.items(), start=1):
            if not isinstance(item, dict):
                continue
            row = build_row(
                source_key=source_key,
                item=item,
                processed=processed,
                dataset_dir=dataset_dir,
                source_base_dir=source_base_dir,
                image_base_dir=image_base_dir,
                label_base_dir=label_base_dir,
            )
            upsert_item(cursor, row)
            imported += 1
            if imported % batch_size == 0:
                connection.commit()
                print(f"Imported {imported}/{len(items)} items")
        connection.commit()

    print(
        json.dumps(
            {
                "done": True,
                "processed_json": str(processed_path),
                "source_base_dir": str(source_base_dir),
                "image_base_dir": str(image_base_dir),
                "label_base_dir": str(label_base_dir),
                "imported": imported,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
