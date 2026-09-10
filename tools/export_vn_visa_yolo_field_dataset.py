from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import random
import re
import shutil
import sys
from typing import Any

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mrz_reader.db import connect
from mrz_reader.env_config import env_value, read_env_file


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

VALID_SPLITS = {"train", "val", "test"}


@dataclass(frozen=True)
class Config:
    source_image_dir: Path
    output_dir: Path
    fields: list[str]
    include_extra_fields: bool
    val_ratio: float
    test_ratio: float
    limit: int
    force: bool
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
    values = [value.strip() for value in text.split(",")]
    return [value for value in values if value]


def load_config(env: dict[str, str], args: argparse.Namespace) -> Config:
    source_dir_raw = (
        args.source_image_dir
        or env_value(env, "READMRZ_VN_VISA_YOLO_SOURCE_IMAGE_DIR", "")
        or env_value(env, "READMRZ_VN_VISA_OUTPUT_DIR", "")
        or env_value(env, "READMRZ_VN_VISA_IMPORT_ORIENTED_IMAGE_DIR", "")
    )
    if not source_dir_raw:
        raise ValueError(
            "Missing READMRZ_VN_VISA_YOLO_SOURCE_IMAGE_DIR, "
            "READMRZ_VN_VISA_OUTPUT_DIR, or READMRZ_VN_VISA_IMPORT_ORIENTED_IMAGE_DIR"
        )

    output_dir_raw = (
        args.output_dir
        or env_value(env, "READMRZ_VN_VISA_YOLO_DATASET_DIR", "")
        or str(PROJECT_ROOT / "generated_datasets" / "vn_visa_yolo_fields")
    )

    fields_raw = args.fields or env_value(env, "READMRZ_VN_VISA_YOLO_FIELDS", "")
    fields = csv_values(fields_raw) if fields_raw else list(DEFAULT_FIELDS)
    if not fields:
        raise ValueError("READMRZ_VN_VISA_YOLO_FIELDS is empty")

    return Config(
        source_image_dir=Path(source_dir_raw).expanduser().resolve(),
        output_dir=Path(output_dir_raw).expanduser().resolve(),
        fields=fields,
        include_extra_fields=args.include_extra_fields
        or env_bool(env, "READMRZ_VN_VISA_YOLO_INCLUDE_EXTRA_FIELDS", False),
        val_ratio=max(0.0, min(1.0, args.val_ratio if args.val_ratio is not None else env_float(env, "READMRZ_VN_VISA_YOLO_VAL_RATIO", 0.1))),
        test_ratio=max(0.0, min(1.0, args.test_ratio if args.test_ratio is not None else env_float(env, "READMRZ_VN_VISA_YOLO_TEST_RATIO", 0.0))),
        limit=max(0, args.limit if args.limit is not None else env_int(env, "READMRZ_VN_VISA_YOLO_LIMIT", 0)),
        force=args.force or env_bool(env, "READMRZ_VN_VISA_YOLO_FORCE", False),
        dry_run=args.dry_run,
    )


def fetch_approved_rows(limit: int) -> list[dict[str, Any]]:
    query = """
        SELECT
            Id,
            SourceKey,
            DocumentType,
            RelativeImagePath,
            OrientedRelativeImagePath,
            OriginalFileName,
            ImageWidth,
            ImageHeight,
            Split,
            DataJson
        FROM dbo.readmrz_vn_visa_items
        WHERE Status = N'mapped'
          AND ReviewStatus = N'approved'
        ORDER BY Id ASC
    """
    if limit > 0:
        query = query.replace("SELECT", f"SELECT TOP {int(limit)}", 1)

    with connect() as connection:
        cursor = connection.cursor()
        cursor.execute(query)
        columns = [column[0] for column in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]


def parse_data_json(row: dict[str, Any]) -> dict[str, Any]:
    raw_value = row.get("DataJson") or "{}"
    if isinstance(raw_value, dict):
        return raw_value
    try:
        payload = json.loads(str(raw_value))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def resolve_image_path(source_dir: Path, row: dict[str, Any]) -> Path | None:
    for key in ("RelativeImagePath", "OrientedRelativeImagePath", "SourceKey"):
        relative_path = str(row.get(key) or "").strip()
        if not relative_path:
            continue
        direct = source_dir / relative_path
        if direct.is_file():
            return direct

    file_name = str(row.get("OriginalFileName") or Path(str(row.get("SourceKey") or "")).name).strip()
    if file_name:
        matches = list(source_dir.rglob(file_name))
        if matches:
            return matches[0]
    return None


def choose_split(row: dict[str, Any], val_ratio: float, test_ratio: float) -> str:
    db_split = str(row.get("Split") or "").strip().lower()
    if db_split in VALID_SPLITS:
        return db_split

    source_key = str(row.get("SourceKey") or row.get("Id") or "")
    rng = random.Random(hashlib.sha1(source_key.encode("utf-8")).hexdigest())
    value = rng.random()
    if test_ratio > 0 and value < test_ratio:
        return "test"
    if value < test_ratio + val_ratio:
        return "val"
    return "train"


def safe_output_name(row: dict[str, Any], image_path: Path) -> str:
    source_name = str(row.get("OriginalFileName") or image_path.name)
    suffix = image_path.suffix.lower() or Path(source_name).suffix.lower() or ".jpg"
    stem = Path(source_name).stem or image_path.stem or "image"
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "image"
    return f"{int(row['Id']):08d}_{safe_stem}{suffix}"


def image_size(image_path: Path, row: dict[str, Any]) -> tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is not None:
        height, width = image.shape[:2]
        return int(width), int(height)

    width = int(row.get("ImageWidth") or 0)
    height = int(row.get("ImageHeight") or 0)
    if width <= 0 or height <= 0:
        raise ValueError(f"Cannot read image size: {image_path}")
    return width, height


def bbox_xyxy(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return [x1, y1, x2, y2]


def clamp_bbox(box: list[float], image_width: int, image_height: int) -> list[float] | None:
    x1, y1, x2, y2 = box
    left = max(0.0, min(x1, x2))
    top = max(0.0, min(y1, y2))
    right = min(float(image_width - 1), max(x1, x2))
    bottom = min(float(image_height - 1), max(y1, y2))
    if right - left < 2 or bottom - top < 2:
        return None
    return [left, top, right, bottom]


def yolo_line(class_id: int, box: list[float], image_width: int, image_height: int) -> str:
    x1, y1, x2, y2 = box
    x_center = ((x1 + x2) / 2.0) / image_width
    y_center = ((y1 + y2) / 2.0) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def collect_field_names(rows: list[dict[str, Any]], configured_fields: list[str], include_extra: bool) -> list[str]:
    field_names = list(dict.fromkeys(configured_fields))
    if not include_extra:
        return field_names

    known = set(field_names)
    extras: set[str] = set()
    for row in rows:
        fields = parse_data_json(row).get("fields") or {}
        if isinstance(fields, dict):
            extras.update(str(name) for name in fields if str(name) not in known)

    return field_names + sorted(extras)


def collect_annotations(
    row: dict[str, Any],
    *,
    class_ids: dict[str, int],
    image_width: int,
    image_height: int,
) -> tuple[list[str], list[dict[str, Any]], int]:
    data_json = parse_data_json(row)
    fields = data_json.get("fields") or {}
    if not isinstance(fields, dict):
        return [], [], 0

    lines: list[str] = []
    annotations: list[dict[str, Any]] = []
    skipped = 0
    for field_name, field_payload in fields.items():
        field_name = str(field_name)
        if field_name not in class_ids or not isinstance(field_payload, dict):
            continue
        raw_box = bbox_xyxy(field_payload.get("bbox"))
        if raw_box is None:
            skipped += 1
            continue
        box = clamp_bbox(raw_box, image_width, image_height)
        if box is None:
            skipped += 1
            continue

        class_id = class_ids[field_name]
        lines.append(yolo_line(class_id, box, image_width, image_height))
        annotations.append(
            {
                "field_name": field_name,
                "class_id": class_id,
                "bbox_xyxy": [round(value, 2) for value in box],
                "raw_text": field_payload.get("raw_text") or "",
                "normalized_value": field_payload.get("normalized_value") or "",
                "bbox_source": field_payload.get("bbox_source") or "",
                "confidence": field_payload.get("confidence"),
            }
        )

    return lines, annotations, skipped


def write_data_yaml(output_dir: Path, field_names: list[str]) -> None:
    lines = [
        f"path: {output_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(field_names))
    lines.append("")
    (output_dir / "data.yaml").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "classes.txt").write_text("\n".join(field_names) + "\n", encoding="utf-8")


def clean_output_dir(output_dir: Path) -> None:
    for child_name in ("images", "labels"):
        child_path = output_dir / child_name
        if child_path.exists():
            shutil.rmtree(child_path)
    for file_name in ("data.yaml", "classes.txt", "manifest.jsonl", "summary.json"):
        file_path = output_dir / file_name
        if file_path.exists():
            file_path.unlink()


def export_dataset(config: Config) -> dict[str, Any]:
    if not config.source_image_dir.exists():
        raise FileNotFoundError(f"Source image dir does not exist: {config.source_image_dir}")

    rows = fetch_approved_rows(config.limit)
    field_names = collect_field_names(rows, config.fields, config.include_extra_fields)
    class_ids = {field_name: index for index, field_name in enumerate(field_names)}

    counts: dict[str, int] = {
        "db_rows": len(rows),
        "exported_images": 0,
        "exported_boxes": 0,
        "missing_images": 0,
        "images_without_boxes": 0,
        "skipped_boxes": 0,
    }
    split_counts: dict[str, int] = {split: 0 for split in ("train", "val", "test")}
    field_counts: dict[str, int] = {field_name: 0 for field_name in field_names}
    manifest_items: list[dict[str, Any]] = []

    if not config.dry_run:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        if config.force:
            clean_output_dir(config.output_dir)
        for split in ("train", "val", "test"):
            (config.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (config.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        write_data_yaml(config.output_dir, field_names)

    for row in rows:
        image_path = resolve_image_path(config.source_image_dir, row)
        if image_path is None:
            counts["missing_images"] += 1
            print(f"[missing-image] id={row.get('Id')} key={row.get('SourceKey')}")
            continue

        try:
            width, height = image_size(image_path, row)
        except ValueError as exc:
            counts["missing_images"] += 1
            print(f"[bad-image] id={row.get('Id')} {exc}")
            continue

        label_lines, annotations, skipped_boxes = collect_annotations(
            row,
            class_ids=class_ids,
            image_width=width,
            image_height=height,
        )
        counts["skipped_boxes"] += skipped_boxes
        if not label_lines:
            counts["images_without_boxes"] += 1
            print(f"[no-box] id={row.get('Id')} image={image_path.name}")
            continue

        split = choose_split(row, config.val_ratio, config.test_ratio)
        output_name = safe_output_name(row, image_path)
        image_rel = f"images/{split}/{output_name}"
        label_rel = f"labels/{split}/{Path(output_name).stem}.txt"

        if not config.dry_run:
            shutil.copy2(image_path, config.output_dir / image_rel)
            (config.output_dir / label_rel).write_text("\n".join(label_lines) + "\n", encoding="utf-8")

        for annotation in annotations:
            field_counts[annotation["field_name"]] += 1

        counts["exported_images"] += 1
        counts["exported_boxes"] += len(label_lines)
        split_counts[split] += 1
        manifest_items.append(
            {
                "id": int(row["Id"]),
                "source_key": row.get("SourceKey"),
                "document_type": row.get("DocumentType"),
                "source_image": str(image_path),
                "output_image": image_rel,
                "output_label": label_rel,
                "split": split,
                "image_width": width,
                "image_height": height,
                "annotations": annotations,
            }
        )
        print(f"[exported] id={row.get('Id')} split={split} boxes={len(label_lines)} image={output_name}")

    summary = {
        **counts,
        "splits": split_counts,
        "fields": field_counts,
        "class_names": field_names,
        "source_image_dir": str(config.source_image_dir),
        "output_dir": str(config.output_dir),
        "dry_run": config.dry_run,
    }

    if not config.dry_run:
        manifest_path = config.output_dir / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as file_obj:
            for item in manifest_items:
                file_obj.write(json.dumps(item, ensure_ascii=False) + "\n")
        (config.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export approved Vietnam visa review bboxes into a YOLO field-detection dataset."
    )
    parser.add_argument("--env", default=str(PROJECT_ROOT / ".env"), help="Path to .env config file.")
    parser.add_argument("--source-image-dir", default="", help="Override READMRZ_VN_VISA_OUTPUT_DIR.")
    parser.add_argument("--output-dir", default="", help="Override READMRZ_VN_VISA_YOLO_DATASET_DIR.")
    parser.add_argument("--fields", default="", help="Comma-separated YOLO class names. Defaults to VN visa fields.")
    parser.add_argument("--include-extra-fields", action="store_true", help="Append unknown DB field names to classes.")
    parser.add_argument("--val-ratio", type=float, default=None, help="Validation split ratio for rows without DB Split.")
    parser.add_argument("--test-ratio", type=float, default=None, help="Test split ratio for rows without DB Split.")
    parser.add_argument("--limit", type=int, default=None, help="Limit approved DB rows.")
    parser.add_argument("--force", action="store_true", help="Clear exported images/labels before writing.")
    parser.add_argument("--dry-run", action="store_true", help="Read DB and validate boxes without writing files.")
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
    print(f"Using env file: {env_path}")
    print(f"Source images: {config.source_image_dir}")
    print(f"Output dataset: {config.output_dir}")
    print(f"Classes: {', '.join(config.fields)}")
    print(f"Dry run: {'yes' if config.dry_run else 'no'}")

    summary = export_dataset(config)
    print("DONE")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
