from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mrz_reader.db import connect
from mrz_reader.env_config import env_value, read_env_file, yolo_dataset_dir


@dataclass(frozen=True)
class LabelItem:
    id: int
    source_key: str
    source_file_name: str | None
    split: str
    image_file_name: str
    label_file_name: str
    bbox_xyxy: list[float]
    yolo_label: str
    mrz_lines_json: str | None
    mrz_score: float | None
    ocr_ms: int | None
    elapsed_ms: int | None
    fingerprint_size: int | None
    fingerprint_mtime_ns: int | None
    ocr_engine: str | None
    ocr_config_json: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate rotated YOLO MRZ samples from approved readmrz_label_items records."
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print planned augmentations, do not write files or DB.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing augmented files and reset DB row to pending.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Defaults to READMRZ_YOLO_AUG_SEED.")
    return parser.parse_args()


def int_env(env: dict[str, str], key: str, default: int) -> int:
    raw = env_value(env, key, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        return default


def bool_env(env: dict[str, str], key: str, default: bool) -> bool:
    raw = env_value(env, key, "true" if default else "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def row_to_item(cursor: Any, row: Any) -> LabelItem:
    values = dict(zip([column[0] for column in cursor.description], row))
    return LabelItem(
        id=int(values["id"]),
        source_key=str(values["source_key"]),
        source_file_name=values.get("source_file_name"),
        split=str(values.get("split") or "train"),
        image_file_name=str(values["image_file_name"]),
        label_file_name=str(values["label_file_name"]),
        bbox_xyxy=[
            float(values["bbox_x1"]),
            float(values["bbox_y1"]),
            float(values["bbox_x2"]),
            float(values["bbox_y2"]),
        ],
        yolo_label=str(values["yolo_label"]),
        mrz_lines_json=values.get("mrz_lines_json"),
        mrz_score=values.get("mrz_score"),
        ocr_ms=values.get("ocr_ms"),
        elapsed_ms=values.get("elapsed_ms"),
        fingerprint_size=values.get("fingerprint_size"),
        fingerprint_mtime_ns=values.get("fingerprint_mtime_ns"),
        ocr_engine=values.get("ocr_engine"),
        ocr_config_json=values.get("ocr_config_json"),
    )


def fetch_approved_items(cursor: Any, include_augmented: bool) -> list[LabelItem]:
    extra_filter = "" if include_augmented else "AND source_key NOT LIKE '%__rot90' AND source_key NOT LIKE '%__rot180' AND source_key NOT LIKE '%__rot270'"
    cursor.execute(
        f"""
        SELECT *
        FROM dbo.readmrz_label_items
        WHERE status = 'labeled'
          AND review_status = 'approved'
          AND image_file_name IS NOT NULL
          AND label_file_name IS NOT NULL
          AND yolo_label IS NOT NULL
          AND bbox_x1 IS NOT NULL
          AND bbox_y1 IS NOT NULL
          AND bbox_x2 IS NOT NULL
          AND bbox_y2 IS NOT NULL
          {extra_filter}
        ORDER BY id ASC
        """
    )
    return [row_to_item(cursor, row) for row in cursor.fetchall()]


def filter_existing_items(items: list[LabelItem], image_base_dir: Path) -> tuple[list[LabelItem], int]:
    existing: list[LabelItem] = []
    missing = 0
    for item in items:
        image_path = resolve_under_base(image_base_dir, item.image_file_name)
        if image_path.is_file():
            existing.append(item)
        else:
            missing += 1
    return existing, missing


def resolve_under_base(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def split_relative_path(split: str, file_name: str, suffix: str) -> str:
    path = Path(file_name)
    stem = path.stem
    extension = path.suffix or suffix
    return f"{split}/{stem}{extension}".replace("\\", "/")


def augmented_relative_path(file_name: str, rotation_angle: int, suffix: str) -> str:
    path = Path(file_name)
    parent = path.parent.as_posix()
    stem = path.stem
    extension = path.suffix or suffix
    name = f"{stem}__rot{rotation_angle}{extension}"
    return f"{parent}/{name}".replace("\\", "/") if parent and parent != "." else name


def parse_yolo_label(label_text: str) -> tuple[int, float, float, float, float]:
    parts = label_text.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Invalid YOLO label: {label_text!r}")
    return int(float(parts[0])), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])


def format_yolo_label(values: tuple[int, float, float, float, float]) -> str:
    class_id, x_center, y_center, width, height = values
    return f"{class_id} {clamp01(x_center):.6f} {clamp01(y_center):.6f} {clamp01(width):.6f} {clamp01(height):.6f}"


def rotate_yolo_label(label_text: str, rotation_angle: int) -> str:
    class_id, x_center, y_center, width, height = parse_yolo_label(label_text)
    if rotation_angle == 90:
        return format_yolo_label((class_id, 1.0 - y_center, x_center, height, width))
    if rotation_angle == 180:
        return format_yolo_label((class_id, 1.0 - x_center, 1.0 - y_center, width, height))
    if rotation_angle == 270:
        return format_yolo_label((class_id, y_center, 1.0 - x_center, height, width))
    raise ValueError(f"Unsupported rotation angle: {rotation_angle}")


def bbox_from_yolo_label(label_text: str, image_width: int, image_height: int) -> list[float]:
    _, x_center, y_center, width, height = parse_yolo_label(label_text)
    box_width = width * image_width
    box_height = height * image_height
    center_x = x_center * image_width
    center_y = y_center * image_height
    return [
        round(clamp(center_x - box_width / 2.0, 0, image_width), 2),
        round(clamp(center_y - box_height / 2.0, 0, image_height), 2),
        round(clamp(center_x + box_width / 2.0, 0, image_width), 2),
        round(clamp(center_y + box_height / 2.0, 0, image_height), 2),
    ]


def rotate_image(image: Any, rotation_angle: int) -> Any:
    if rotation_angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation_angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if rotation_angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    raise ValueError(f"Unsupported rotation angle: {rotation_angle}")


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def clamp01(value: float) -> float:
    return clamp(value, 0.0, 1.0)


def source_key_for(item: LabelItem, rotation_angle: int) -> str:
    return f"{item.source_key}__rot{rotation_angle}"


def source_file_name_for(item: LabelItem, rotation_angle: int) -> str:
    base = Path(item.source_file_name or item.image_file_name)
    return f"{base.stem}__rot{rotation_angle}{base.suffix or '.jpg'}"


def record_exists(cursor: Any, source_key: str) -> bool:
    cursor.execute("SELECT 1 FROM dbo.readmrz_label_items WHERE source_key = ?", source_key)
    return cursor.fetchone() is not None


def insert_or_update_augmented_item(
    cursor: Any,
    *,
    item: LabelItem,
    source_key: str,
    source_file_name: str,
    image_file_name: str,
    label_file_name: str,
    bbox_xyxy: list[float],
    yolo_label: str,
    rotation_angle: int,
    force: bool,
) -> bool:
    ocr_config_json = item.ocr_config_json or "{}"
    note_prefix = f"augmented_rotation={rotation_angle}; parent_id={item.id}; "
    if record_exists(cursor, source_key):
        if not force:
            return False
        cursor.execute(
            """
            UPDATE dbo.readmrz_label_items
            SET source_file_name = ?,
                status = 'labeled',
                review_status = 'pending',
                split = ?,
                image_file_name = ?,
                label_file_name = ?,
                rejected_image_file_name = NULL,
                rejected_label_file_name = NULL,
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
                reviewed_at = NULL,
                updated_at = SYSUTCDATETIME()
            WHERE source_key = ?
            """,
            source_file_name,
            item.split,
            image_file_name,
            label_file_name,
            bbox_xyxy[0],
            bbox_xyxy[1],
            bbox_xyxy[2],
            bbox_xyxy[3],
            yolo_label,
            item.mrz_lines_json,
            item.mrz_score,
            item.ocr_ms,
            item.elapsed_ms,
            item.fingerprint_size,
            item.fingerprint_mtime_ns,
            item.ocr_engine,
            ocr_config_json,
            f"{note_prefix}generated from approved label",
            source_key,
        )
        return True

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
            processed_at
        )
        VALUES (?, ?, 'labeled', 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, SYSUTCDATETIME())
        """,
        source_key,
        source_file_name,
        item.split,
        image_file_name,
        label_file_name,
        bbox_xyxy[0],
        bbox_xyxy[1],
        bbox_xyxy[2],
        bbox_xyxy[3],
        yolo_label,
        item.mrz_lines_json,
        item.mrz_score,
        item.ocr_ms,
        item.elapsed_ms,
        item.fingerprint_size,
        item.fingerprint_mtime_ns,
        item.ocr_engine,
        ocr_config_json,
        f"{note_prefix}generated from approved label",
    )
    return True


def planned_train_items(items: list[LabelItem], env: dict[str, str], rng: random.Random) -> list[tuple[LabelItem, int]]:
    train_items = [item for item in items if item.split.lower() == "train"]
    plan: list[tuple[LabelItem, int]] = []
    for rotation_angle, default_limit in ((90, 3000), (270, 3000), (180, 1800)):
        limit = int_env(env, f"READMRZ_YOLO_AUG_ROT{rotation_angle}_TRAIN_LIMIT", default_limit)
        if limit <= 0:
            continue
        selected = train_items[:]
        rng.shuffle(selected)
        plan.extend((item, rotation_angle) for item in selected[: min(limit, len(selected))])
    return plan


def planned_val_items(items: list[LabelItem], env: dict[str, str], rng: random.Random) -> list[tuple[LabelItem, int]]:
    val_items = [item for item in items if item.split.lower() == "val"]
    rotations_text = env_value(env, "READMRZ_YOLO_AUG_VAL_ROTATIONS", "90,180,270")
    rotations = [int(part.strip()) for part in rotations_text.split(",") if part.strip()]
    plan: list[tuple[LabelItem, int]] = []
    for rotation_angle in rotations:
        limit = int_env(env, f"READMRZ_YOLO_AUG_ROT{rotation_angle}_VAL_LIMIT", len(val_items))
        if limit <= 0:
            continue
        selected = val_items[:]
        rng.shuffle(selected)
        plan.extend((item, rotation_angle) for item in selected[: min(limit, len(selected))])
    return plan


def make_augmented_sample(
    *,
    item: LabelItem,
    rotation_angle: int,
    image_base_dir: Path,
    label_base_dir: Path,
    dry_run: bool,
    force: bool,
) -> tuple[str, str, str, list[float], str, bool]:
    source_key = source_key_for(item, rotation_angle)
    target_image_rel = augmented_relative_path(item.image_file_name, rotation_angle, ".jpg")
    target_label_rel = augmented_relative_path(item.label_file_name, rotation_angle, ".txt")
    source_image_path = resolve_under_base(image_base_dir, item.image_file_name)
    target_image_path = resolve_under_base(image_base_dir, target_image_rel)
    target_label_path = resolve_under_base(label_base_dir, target_label_rel)

    if not source_image_path.is_file():
        raise FileNotFoundError(f"Missing source image: {source_image_path}")

    image = cv2.imread(str(source_image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {source_image_path}")

    rotated = rotate_image(image, rotation_angle)
    rotated_height, rotated_width = rotated.shape[:2]
    new_yolo_label = rotate_yolo_label(item.yolo_label, rotation_angle)
    new_bbox = bbox_from_yolo_label(new_yolo_label, rotated_width, rotated_height)

    if dry_run:
        return source_key, target_image_rel, target_label_rel, new_bbox, new_yolo_label, False

    target_image_path.parent.mkdir(parents=True, exist_ok=True)
    target_label_path.parent.mkdir(parents=True, exist_ok=True)
    if target_image_path.exists() and not force:
        return source_key, target_image_rel, target_label_rel, new_bbox, new_yolo_label, False

    ok = cv2.imwrite(str(target_image_path), rotated)
    if not ok:
        raise ValueError(f"Cannot write image: {target_image_path}")
    target_label_path.write_text(new_yolo_label + "\n", encoding="utf-8")
    return source_key, target_image_rel, target_label_rel, new_bbox, new_yolo_label, True


def main() -> int:
    args = parse_args()
    env = read_env_file()
    dataset_dir = yolo_dataset_dir(env)
    image_base_dir = Path(env_value(env, "READMRZ_YOLO_IMAGE_BASE_DIR", str(dataset_dir / "images"))).expanduser().resolve()
    label_base_dir = Path(env_value(env, "READMRZ_YOLO_LABEL_BASE_DIR", str(dataset_dir / "labels"))).expanduser().resolve()
    seed = args.seed if args.seed is not None else int_env(env, "READMRZ_YOLO_AUG_SEED", 20260731)
    include_augmented = bool_env(env, "READMRZ_YOLO_AUG_INCLUDE_AUGMENTED", False)
    batch_size = max(1, int_env(env, "READMRZ_YOLO_AUG_DB_BATCH_SIZE", 250))
    rng = random.Random(seed)

    print(f"dataset_dir: {dataset_dir}")
    print(f"image_base_dir: {image_base_dir}")
    print(f"label_base_dir: {label_base_dir}")
    print(f"seed: {seed}")
    print(f"dry_run: {args.dry_run}")
    print(f"force: {args.force}")

    with connect() as connection:
        cursor = connection.cursor()
        approved_items = fetch_approved_items(cursor, include_augmented)
        existing_items, missing_items = filter_existing_items(approved_items, image_base_dir)
        plan = planned_train_items(existing_items, env, rng) + planned_val_items(existing_items, env, rng)
        print(f"approved DB source items: {len(approved_items)}")
        print(f"approved source items with local files: {len(existing_items)}")
        print(f"approved source items missing local image: {missing_items}")
        print(f"planned augmentations: {len(plan)}")

        planned_by_split_rotation: dict[str, int] = {}
        for item, rotation_angle in plan:
            key = f"{item.split}_rot{rotation_angle}"
            planned_by_split_rotation[key] = planned_by_split_rotation.get(key, 0) + 1
        for key in sorted(planned_by_split_rotation):
            print(f"{key}: {planned_by_split_rotation[key]}")

        if args.dry_run:
            return 0

        inserted_or_updated = 0
        skipped = 0
        failed = 0
        for index, (item, rotation_angle) in enumerate(plan, start=1):
            try:
                source_key, image_rel, label_rel, bbox_xyxy, yolo_label, wrote_files = make_augmented_sample(
                    item=item,
                    rotation_angle=rotation_angle,
                    image_base_dir=image_base_dir,
                    label_base_dir=label_base_dir,
                    dry_run=False,
                    force=args.force,
                )
                source_file_name = source_file_name_for(item, rotation_angle)
                changed_db = insert_or_update_augmented_item(
                    cursor,
                    item=item,
                    source_key=source_key,
                    source_file_name=source_file_name,
                    image_file_name=image_rel,
                    label_file_name=label_rel,
                    bbox_xyxy=bbox_xyxy,
                    yolo_label=yolo_label,
                    rotation_angle=rotation_angle,
                    force=args.force,
                )
                if wrote_files or changed_db:
                    inserted_or_updated += 1
                else:
                    skipped += 1
            except Exception as exc:
                failed += 1
                print(f"ERROR item={item.source_key} rot={rotation_angle}: {exc}")

            if index % batch_size == 0:
                connection.commit()
                print(f"processed {index}/{len(plan)} inserted_or_updated={inserted_or_updated} skipped={skipped} failed={failed}")

        connection.commit()

    print(
        "done "
        f"planned={len(plan)} inserted_or_updated={inserted_or_updated} "
        f"skipped={skipped} failed={failed}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
