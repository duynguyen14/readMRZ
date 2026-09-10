from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mrz_reader.db import connect
from mrz_reader.env_config import env_value, read_env_file


DEFAULT_REVIEW_STATUSES = ["pending", "needs_review"]
DEFAULT_FIELDS = [
    "visa_number",
    "visa_code",
    "valid_from",
    "valid_until",
    "number_of_entries",
    "passport_number",
    "mrz_zone",
    "issue_date",
    "date_of_birth",
    "full_name",
    "nationality",
    "sex",
    "issue_place",
    "issuing_authority",
    "remarks",
]


@dataclass(frozen=True)
class Config:
    model_path: Path
    source_image_dir: Path
    statuses: list[str]
    review_statuses: list[str]
    fields: list[str]
    conf: float
    iou: float
    imgsz: int
    device: str
    batch_size: int
    limit: int
    max_det: int
    overwrite_auto_bbox: bool
    overwrite_manual_bbox: bool
    dry_run: bool


def env_int(env: dict[str, str], key: str, default: int) -> int:
    raw_value = env_value(env, key, "")
    return int(raw_value) if raw_value else default


def env_float(env: dict[str, str], key: str, default: float) -> float:
    raw_value = env_value(env, key, "")
    return float(raw_value) if raw_value else default


def env_bool(env: dict[str, str], key: str, default: bool) -> bool:
    raw_value = env_value(env, key, "")
    if not raw_value:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def csv_values(text: str) -> list[str]:
    return [value.strip() for value in text.split(",") if value.strip()]


def resolve_existing_path(raw_value: str, *, label: str) -> Path:
    path = Path(raw_value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path.resolve()


def resolve_model_path(raw_value: str) -> Path:
    path = resolve_existing_path(raw_value, label="YOLO model")
    if path.is_file():
        return path
    for candidate in (path / "best.pt", path / "weights" / "best.pt", path / "last.pt", path / "weights" / "last.pt"):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find best.pt or last.pt under YOLO model dir: {path}")


def load_config(env: dict[str, str], args: argparse.Namespace) -> Config:
    model_path_raw = args.model_path or env_value(env, "READMRZ_VN_VISA_YOLO_MODEL_PATH", "")
    if not model_path_raw:
        raise ValueError("Missing READMRZ_VN_VISA_YOLO_MODEL_PATH")

    source_image_dir_raw = (
        args.source_image_dir
        or env_value(env, "READMRZ_VN_VISA_YOLO_PREDICT_SOURCE_IMAGE_DIR", "")
        or env_value(env, "READMRZ_VN_VISA_IMPORT_ORIENTED_IMAGE_DIR", "")
        or env_value(env, "READMRZ_VN_VISA_OUTPUT_DIR", "")
    )
    if not source_image_dir_raw:
        raise ValueError(
            "Missing READMRZ_VN_VISA_YOLO_PREDICT_SOURCE_IMAGE_DIR, "
            "READMRZ_VN_VISA_IMPORT_ORIENTED_IMAGE_DIR, or READMRZ_VN_VISA_OUTPUT_DIR"
        )

    fields_raw = args.fields or env_value(env, "READMRZ_VN_VISA_YOLO_PREDICT_FIELDS", "")
    if not fields_raw:
        fields_raw = env_value(env, "READMRZ_VN_VISA_YOLO_FIELDS", "")
    fields = csv_values(fields_raw) if fields_raw else list(DEFAULT_FIELDS)

    statuses_raw = args.statuses or env_value(env, "READMRZ_VN_VISA_YOLO_PREDICT_STATUSES", "mapped")
    review_statuses_raw = args.review_statuses or env_value(
        env,
        "READMRZ_VN_VISA_YOLO_PREDICT_REVIEW_STATUSES",
        ",".join(DEFAULT_REVIEW_STATUSES),
    )

    return Config(
        model_path=resolve_model_path(model_path_raw),
        source_image_dir=resolve_existing_path(source_image_dir_raw, label="Source image dir"),
        statuses=csv_values(statuses_raw),
        review_statuses=csv_values(review_statuses_raw),
        fields=fields,
        conf=max(0.0, min(1.0, args.conf if args.conf is not None else env_float(env, "READMRZ_VN_VISA_YOLO_PREDICT_CONF", 0.25))),
        iou=max(0.0, min(1.0, args.iou if args.iou is not None else env_float(env, "READMRZ_VN_VISA_YOLO_PREDICT_IOU", 0.45))),
        imgsz=max(32, args.imgsz if args.imgsz is not None else env_int(env, "READMRZ_VN_VISA_YOLO_PREDICT_IMGSZ", 960)),
        device=args.device or env_value(env, "READMRZ_VN_VISA_YOLO_PREDICT_DEVICE", "cpu").strip(),
        batch_size=max(1, args.batch_size if args.batch_size is not None else env_int(env, "READMRZ_VN_VISA_YOLO_PREDICT_BATCH_SIZE", 8)),
        limit=max(0, args.limit if args.limit is not None else env_int(env, "READMRZ_VN_VISA_YOLO_PREDICT_LIMIT", 0)),
        max_det=max(1, args.max_det if args.max_det is not None else env_int(env, "READMRZ_VN_VISA_YOLO_PREDICT_MAX_DET", 50)),
        overwrite_auto_bbox=args.overwrite_auto_bbox
        or env_bool(env, "READMRZ_VN_VISA_YOLO_PREDICT_OVERWRITE_AUTO_BBOX", True),
        overwrite_manual_bbox=args.overwrite_manual_bbox
        or env_bool(env, "READMRZ_VN_VISA_YOLO_PREDICT_OVERWRITE_MANUAL_BBOX", False),
        dry_run=args.dry_run or env_bool(env, "READMRZ_VN_VISA_YOLO_PREDICT_DRY_RUN", False),
    )


def fetch_rows(config: Config) -> list[dict[str, Any]]:
    if not config.statuses:
        raise ValueError("At least one status is required")
    if not config.review_statuses:
        raise ValueError("At least one review status is required")

    status_placeholders = ",".join("?" for _ in config.statuses)
    review_status_placeholders = ",".join("?" for _ in config.review_statuses)
    query = f"""
        SELECT
            Id,
            SourceKey,
            RelativeImagePath,
            OrientedRelativeImagePath,
            OriginalFileName,
            ImageWidth,
            ImageHeight,
            Status,
            ReviewStatus,
            DataJson
        FROM dbo.readmrz_vn_visa_items
        WHERE Status IN ({status_placeholders})
          AND ReviewStatus IN ({review_status_placeholders})
        ORDER BY Id ASC
    """
    if config.limit > 0:
        query = query.replace("SELECT", f"SELECT TOP {int(config.limit)}", 1)

    params = [*config.statuses, *config.review_statuses]
    with connect() as connection:
        cursor = connection.cursor()
        cursor.execute(query, *params)
        columns = [column[0] for column in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]


def parse_data_json(row: dict[str, Any]) -> dict[str, Any]:
    raw_value = row.get("DataJson") or "{}"
    if isinstance(raw_value, dict):
        payload = raw_value
    else:
        try:
            payload = json.loads(str(raw_value))
        except json.JSONDecodeError:
            payload = {}
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("schema_version", 1)
    fields = payload.get("fields")
    if not isinstance(fields, dict):
        payload["fields"] = {}
    missing_fields = payload.get("missing_fields")
    if not isinstance(missing_fields, list):
        payload["missing_fields"] = []
    return payload


def resolve_image_path(source_dir: Path, row: dict[str, Any]) -> Path | None:
    for key in ("OrientedRelativeImagePath", "RelativeImagePath", "SourceKey"):
        relative_path = str(row.get(key) or "").strip()
        if not relative_path:
            continue
        direct = source_dir / relative_path
        if direct.is_file():
            return direct

    file_name = str(row.get("OriginalFileName") or Path(str(row.get("SourceKey") or "")).name).strip()
    if file_name:
        direct = source_dir / file_name
        if direct.is_file():
            return direct
        matches = list(source_dir.rglob(file_name))
        if matches:
            return matches[0]
    return None


def normalize_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def model_class_names(model: Any) -> dict[int, str]:
    names = normalize_names(getattr(model, "names", {}))
    if names:
        return names
    return {index: field for index, field in enumerate(DEFAULT_FIELDS)}


def clamp_bbox(box: list[float], width: int, height: int) -> list[float] | None:
    if width <= 1 or height <= 1:
        return None
    x1, y1, x2, y2 = box
    left = max(0.0, min(x1, x2))
    top = max(0.0, min(y1, y2))
    right = min(float(width - 1), max(x1, x2))
    bottom = min(float(height - 1), max(y1, y2))
    if right - left < 2 or bottom - top < 2:
        return None
    return [round(left, 2), round(top, 2), round(right, 2), round(bottom, 2)]


def result_detections(result: Any, names: dict[int, str], allowed_fields: set[str]) -> tuple[dict[str, dict[str, Any]], int]:
    detections: dict[str, dict[str, Any]] = {}
    raw_count = 0
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections, raw_count

    orig_shape = getattr(result, "orig_shape", None) or (0, 0)
    image_height = int(orig_shape[0] or 0)
    image_width = int(orig_shape[1] or 0)
    xyxy_values = getattr(boxes, "xyxy", None)
    conf_values = getattr(boxes, "conf", None)
    cls_values = getattr(boxes, "cls", None)
    if xyxy_values is None or conf_values is None or cls_values is None:
        return detections, raw_count

    xyxy_list = xyxy_values.cpu().numpy().tolist()
    conf_list = conf_values.cpu().numpy().tolist()
    cls_list = cls_values.cpu().numpy().tolist()

    for raw_box, raw_conf, raw_class_id in zip(xyxy_list, conf_list, cls_list):
        raw_count += 1
        class_id = int(raw_class_id)
        field_name = names.get(class_id, str(class_id))
        if field_name not in allowed_fields:
            continue
        box = clamp_bbox([float(value) for value in raw_box], image_width, image_height)
        if box is None:
            continue
        confidence = float(raw_conf)
        previous = detections.get(field_name)
        if previous is None or confidence > float(previous["confidence"]):
            detections[field_name] = {
                "field_name": field_name,
                "class_id": class_id,
                "bbox": box,
                "confidence": round(confidence, 6),
            }
    return detections, raw_count


def should_update_field(field_data: dict[str, Any], config: Config) -> bool:
    if str(field_data.get("bbox_source") or "") == "manual_review":
        return config.overwrite_manual_bbox
    if "bbox" in field_data:
        return config.overwrite_auto_bbox
    return True


def apply_detections(
    row: dict[str, Any],
    detections: dict[str, dict[str, Any]],
    config: Config,
    model_name: str,
) -> tuple[dict[str, Any], int, int]:
    data_json = parse_data_json(row)
    fields = data_json.setdefault("fields", {})
    updated_count = 0
    skipped_manual = 0
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    data_json["field_bbox_detector"] = {
        "engine": "ultralytics_yolo",
        "model": model_name,
        "conf": config.conf,
        "iou": config.iou,
        "imgsz": config.imgsz,
        "updated_at": timestamp,
    }

    for field_name, detection in detections.items():
        field_data = fields.get(field_name)
        if not isinstance(field_data, dict):
            field_data = {}
        if not should_update_field(field_data, config):
            skipped_manual += 1
            continue

        field_data["bbox"] = detection["bbox"]
        field_data["bbox_source"] = "yolo_field_detector"
        field_data["bbox_confidence"] = detection["confidence"]
        field_data["bbox_model"] = model_name
        field_data["bbox_updated_at"] = timestamp
        field_data.setdefault("field_name", field_name)
        fields[field_name] = field_data
        updated_count += 1

    return data_json, updated_count, skipped_manual


def update_row(cursor: Any, row_id: int, data_json: dict[str, Any]) -> None:
    cursor.execute(
        """
        UPDATE dbo.readmrz_vn_visa_items
        SET DataJson = CAST(? AS NVARCHAR(MAX)),
            UpdatedDate = SYSDATETIME()
        WHERE Id = ?
        """,
        json.dumps(data_json, ensure_ascii=False),
        row_id,
    )


def iter_batches(items: list[Any], batch_size: int) -> Any:
    for offset in range(0, len(items), batch_size):
        yield offset // batch_size + 1, items[offset : offset + batch_size]


def run(config: Config) -> dict[str, Any]:
    from ultralytics import YOLO

    rows = fetch_rows(config)
    print(f"Target DB rows: {len(rows)}")
    print(f"Source images: {config.source_image_dir}")
    print(f"YOLO model: {config.model_path}")
    print(f"Statuses: {','.join(config.statuses)} review_statuses={','.join(config.review_statuses)}")
    print(f"Predict conf={config.conf} iou={config.iou} imgsz={config.imgsz} device={config.device or 'auto'} batch={config.batch_size}")
    print(f"Dry run: {'yes' if config.dry_run else 'no'}")

    model = YOLO(str(config.model_path))
    names = model_class_names(model)
    allowed_fields = set(config.fields)
    model_name = config.model_path.name

    items: list[tuple[dict[str, Any], Path]] = []
    summary: dict[str, Any] = {
        "db_rows": len(rows),
        "images_found": 0,
        "missing_images": 0,
        "images_predicted": 0,
        "images_without_detections": 0,
        "raw_detections": 0,
        "fields_detected": 0,
        "fields_updated": 0,
        "skipped_manual_bboxes": 0,
        "rows_updated": 0,
        "errors": 0,
        "dry_run": config.dry_run,
        "model_path": str(config.model_path),
    }

    for row in rows:
        image_path = resolve_image_path(config.source_image_dir, row)
        if image_path is None:
            summary["missing_images"] += 1
            print(f"[missing-image] id={row.get('Id')} key={row.get('SourceKey')}")
            continue
        summary["images_found"] += 1
        items.append((row, image_path))

    with connect() as connection:
        cursor = connection.cursor()
        for batch_index, batch in iter_batches(items, config.batch_size):
            batch_paths = [str(image_path) for _, image_path in batch]
            try:
                results = model.predict(
                    source=batch_paths,
                    conf=config.conf,
                    iou=config.iou,
                    imgsz=config.imgsz,
                    device=config.device or None,
                    max_det=config.max_det,
                    verbose=False,
                )
            except Exception as exc:
                summary["errors"] += len(batch)
                print(f"[batch {batch_index}] error: {exc}")
                continue

            batch_row_updates = 0
            for (row, image_path), result in zip(batch, results):
                row_id = int(row["Id"])
                detections, raw_count = result_detections(result, names, allowed_fields)
                summary["images_predicted"] += 1
                summary["raw_detections"] += raw_count
                summary["fields_detected"] += len(detections)

                if not detections:
                    summary["images_without_detections"] += 1
                    print(f"[no-detection] id={row_id} image={image_path.name}")
                    continue

                data_json, updated_count, skipped_manual = apply_detections(row, detections, config, model_name)
                summary["skipped_manual_bboxes"] += skipped_manual
                summary["fields_updated"] += updated_count
                if updated_count <= 0:
                    print(f"[skip] id={row_id} image={image_path.name} detections={len(detections)} updated=0")
                    continue

                if not config.dry_run:
                    update_row(cursor, row_id, data_json)
                summary["rows_updated"] += 1
                batch_row_updates += 1
                field_names = ",".join(sorted(detections.keys()))
                print(f"[update] id={row_id} image={image_path.name} fields={updated_count} detected={field_names}")

            if not config.dry_run and batch_row_updates:
                connection.commit()
            print(
                f"[batch {batch_index}] rows_updated={summary['rows_updated']} "
                f"fields_updated={summary['fields_updated']} missing={summary['missing_images']} errors={summary['errors']}"
            )

        if not config.dry_run:
            connection.commit()

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Update unreviewed Vietnam visa field bboxes in DB using a trained YOLO field detector."
    )
    parser.add_argument("--env", default=str(PROJECT_ROOT / ".env"), help="Path to .env config file.")
    parser.add_argument("--model-path", default="", help="Override READMRZ_VN_VISA_YOLO_MODEL_PATH.")
    parser.add_argument("--source-image-dir", default="", help="Override oriented image source dir.")
    parser.add_argument("--statuses", default="", help="Comma-separated DB Status values. Default: mapped.")
    parser.add_argument("--review-statuses", default="", help="Comma-separated ReviewStatus values. Default: pending,needs_review.")
    parser.add_argument("--fields", default="", help="Comma-separated allowed YOLO class names.")
    parser.add_argument("--conf", type=float, default=None, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=None, help="YOLO NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=None, help="YOLO inference image size.")
    parser.add_argument("--device", default="", help="YOLO device, for example cpu, 0, cuda:0.")
    parser.add_argument("--batch-size", type=int, default=None, help="YOLO predict batch size.")
    parser.add_argument("--limit", type=int, default=None, help="Limit DB rows.")
    parser.add_argument("--max-det", type=int, default=None, help="Max detections per image.")
    parser.add_argument("--overwrite-auto-bbox", action="store_true", help="Overwrite non-manual existing bboxes.")
    parser.add_argument("--overwrite-manual-bbox", action="store_true", help="Also overwrite manual_review bboxes.")
    parser.add_argument("--dry-run", action="store_true", help="Predict and print changes without updating DB.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    env_path = Path(args.env)
    if not env_path.is_absolute():
        env_path = (Path.cwd() / env_path).resolve()
        if not env_path.exists():
            env_path = PROJECT_ROOT / args.env

    env = read_env_file(env_path)
    config = load_config(env, args)
    summary = run(config)
    print("DONE")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
