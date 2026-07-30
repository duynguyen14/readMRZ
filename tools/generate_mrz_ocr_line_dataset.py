from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from generate_mrz_yolo_dataset import (  # noqa: E402
    build_ocr,
    env_bool,
    env_float,
    env_int,
    env_value,
    find_mrz_group,
    line_mrz_likeness,
    read_env_file,
)
from mrz_reader.db import connect  # noqa: E402
from mrz_reader.env_config import yolo_dataset_dir  # noqa: E402
from mrz_reader.mrz import normalize_mrz_text  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_base_dir(env: dict[str, str], key: str, default_path: Path) -> Path:
    return Path(env_value(env, key, str(default_path))).expanduser().resolve()


def maybe_reexec_with_paddle_python(env: dict[str, str]) -> int | None:
    engine = env_value(env, "READMRZ_OCR_ENGINE", "paddle").strip().lower()
    if engine != "paddle" or importlib.util.find_spec("paddleocr") is not None:
        return None

    configured_python = env_value(env, "READMRZ_PADDLE_PYTHON")
    if not configured_python:
        return None

    python_path = Path(configured_python).expanduser().resolve()
    if not python_path.exists():
        raise FileNotFoundError(f"READMRZ_PADDLE_PYTHON does not exist: {python_path}")

    current_python = Path(sys.executable).resolve()
    if current_python == python_path:
        return None

    command = [str(python_path), str(Path(__file__).resolve()), *sys.argv[1:]]
    print(f"PaddleOCR is not installed in {current_python}")
    print(f"Re-running OCR line script with Paddle Python: {python_path}")
    return subprocess.run(command, cwd=str(PROJECT_ROOT), check=False).returncode


def relative_to_base(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return path.name


def row_to_dict(cursor: Any, row: Any) -> dict[str, Any]:
    columns = [column[0] for column in cursor.description]
    return dict(zip(columns, row))


def fetch_approved_label_items(cursor: Any, *, limit: int, force: bool) -> list[dict[str, Any]]:
    top_clause = f"TOP {limit}" if limit > 0 else ""
    force_filter = "" if force else "AND ISNULL(mrz_line_extract_status, '') <> 'done'"
    cursor.execute(
        f"""
        SELECT {top_clause}
            id,
            source_key,
            split,
            image_file_name,
            bbox_x1,
            bbox_y1,
            bbox_x2,
            bbox_y2
        FROM dbo.readmrz_label_items
        WHERE status = 'labeled'
          AND review_status = 'approved'
          AND image_file_name IS NOT NULL
          AND bbox_x1 IS NOT NULL
          AND bbox_y1 IS NOT NULL
          AND bbox_x2 IS NOT NULL
          AND bbox_y2 IS NOT NULL
          {force_filter}
        ORDER BY id ASC
        """
    )
    return [row_to_dict(cursor, row) for row in cursor.fetchall()]


def clamp_box(box: list[float], width: int, height: int, padding_ratio: float) -> list[int]:
    x1, y1, x2, y2 = [float(value) for value in box]
    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)
    pad_x = max(2.0, (right - left) * padding_ratio)
    pad_y = max(2.0, (bottom - top) * padding_ratio)
    return [
        max(0, int(round(left - pad_x))),
        max(0, int(round(top - pad_y))),
        min(width, int(round(right + pad_x))),
        min(height, int(round(bottom + pad_y))),
    ]


def rotate_to_readable(image: np.ndarray, angle: int) -> np.ndarray:
    normalized = int(angle or 0) % 360
    if normalized == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if normalized == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if normalized == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def line_crop_from_row(
    crop: np.ndarray,
    row: dict[str, Any],
    *,
    crop_origin_x: int,
    crop_origin_y: int,
    padding_ratio: float,
    normalize_orientation: bool,
) -> dict[str, Any]:
    crop_height, crop_width = crop.shape[:2]
    line_box = clamp_box(row["bbox_xyxy"], crop_width, crop_height, padding_ratio)
    x1, y1, x2, y2 = line_box
    line_image = crop[y1:y2, x1:x2].copy()
    angle = int(row.get("doc_orientation_angle") or 0)
    if normalize_orientation:
        line_image = rotate_to_readable(line_image, angle)

    height, width = line_image.shape[:2]
    return {
        "image": line_image,
        "width": width,
        "height": height,
        "bbox_full": [
            round(crop_origin_x + x1, 2),
            round(crop_origin_y + y1, 2),
            round(crop_origin_x + x2, 2),
            round(crop_origin_y + y2, 2),
        ],
        "doc_orientation_angle": angle,
    }


def fallback_line_split(crop: np.ndarray, *, expected_lines: int = 2) -> list[dict[str, Any]]:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    projection = closed.sum(axis=1)
    threshold = max(1, projection.max() * 0.12)

    bands: list[tuple[int, int]] = []
    in_band = False
    start = 0
    for index, value in enumerate(projection):
        if value >= threshold and not in_band:
            in_band = True
            start = index
        elif value < threshold and in_band:
            in_band = False
            if index - start >= 5:
                bands.append((start, index))
    if in_band and len(projection) - start >= 5:
        bands.append((start, len(projection) - 1))

    height, width = crop.shape[:2]
    bands = sorted(bands, key=lambda item: item[1] - item[0], reverse=True)[:expected_lines]
    bands = sorted(bands, key=lambda item: item[0])

    rows: list[dict[str, Any]] = []
    for y1, y2 in bands:
        rows.append(
            {
                "text": "",
                "normalized": "",
                "ocr_score": 0.0,
                "likeness": 0.0,
                "bbox_xyxy": [0.0, float(y1), float(width - 1), float(y2)],
                "x_center": width / 2,
                "y_center": (y1 + y2) / 2,
                "width": width,
                "height": max(1, y2 - y1),
                "bottom_bias": ((y1 + y2) / 2) / max(1, height),
                "width_ratio": 1.0,
                "doc_orientation_angle": 0,
            }
        )
    return rows


def ocr_rows_for_crop(ocr: Any, crop: np.ndarray, temp_dir: Path, label_item_id: int) -> list[dict[str, Any]]:
    temp_path = temp_dir / f"mrz_crop_{label_item_id}.jpg"
    cv2.imwrite(str(temp_path), crop)
    try:
        return ocr.predict_rows(temp_path, crop)
    finally:
        temp_path.unlink(missing_ok=True)


def select_mrz_line_rows(rows: list[dict[str, Any]], env: dict[str, str], crop: np.ndarray) -> list[dict[str, Any]]:
    height, width = crop.shape[:2]
    group, _ = find_mrz_group(
        rows,
        min_likeness=env_float(env, "READMRZ_OCR_LINE_MIN_MRZ_LIKENESS", 0.35),
        min_line_length=env_int(env, "READMRZ_OCR_LINE_MIN_TEXT_LENGTH", 20),
        image_height=height,
        image_width=width,
    )
    if group:
        return sorted(group, key=lambda row: row["y_center"])

    candidates = [
        row
        for row in rows
        if row.get("width_ratio", 0) >= 0.25
        and len(str(row.get("normalized") or "")) >= env_int(env, "READMRZ_OCR_LINE_MIN_TEXT_LENGTH", 20)
        and float(row.get("likeness") or 0) >= env_float(env, "READMRZ_OCR_LINE_MIN_MRZ_LIKENESS", 0.35)
    ]
    if candidates:
        return sorted(candidates, key=lambda row: row["y_center"])[:3]

    return fallback_line_split(crop, expected_lines=env_int(env, "READMRZ_OCR_LINE_FALLBACK_LINES", 2))


def write_line_row(
    cursor: Any,
    *,
    item: dict[str, Any],
    line_index: int,
    line_image_rel: str,
    line_image_width: int,
    line_image_height: int,
    line_bbox_full: list[float],
    doc_orientation_angle: int,
    ocr_text: str,
    ocr_score: float,
) -> None:
    normalized = normalize_mrz_text(ocr_text)
    likeness = line_mrz_likeness(ocr_text) if ocr_text else 0.0
    cursor.execute(
        """
        INSERT INTO dbo.readmrz_ocr_line_items (
            label_item_id,
            source_key,
            split,
            line_index,
            line_image_file_name,
            line_image_width,
            line_image_height,
            line_bbox_x1,
            line_bbox_y1,
            line_bbox_x2,
            line_bbox_y2,
            doc_orientation_angle,
            ocr_text,
            normalized_text,
            final_text,
            ocr_score,
            mrz_likeness,
            review_status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
        """,
        item["id"],
        item["source_key"],
        item.get("split"),
        line_index,
        line_image_rel,
        line_image_width,
        line_image_height,
        line_bbox_full[0],
        line_bbox_full[1],
        line_bbox_full[2],
        line_bbox_full[3],
        doc_orientation_angle,
        ocr_text,
        normalized,
        normalized or None,
        ocr_score,
        likeness,
    )


def mark_parent(cursor: Any, item_id: int, *, status: str, count: int = 0, error: str | None = None) -> None:
    cursor.execute(
        """
        UPDATE dbo.readmrz_label_items
        SET mrz_line_extract_status = ?,
            mrz_line_extract_count = ?,
            mrz_line_extract_error = ?,
            mrz_line_extracted_at = SYSUTCDATETIME(),
            updated_at = SYSUTCDATETIME()
        WHERE id = ?
        """,
        status,
        count,
        error,
        item_id,
    )


def process_item(
    cursor: Any,
    *,
    item: dict[str, Any],
    image_base_dir: Path,
    line_image_base_dir: Path,
    ocr: Any,
    env: dict[str, str],
    temp_dir: Path,
) -> int:
    image_path = image_base_dir / str(item["image_file_name"])
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read YOLO image: {image_path}")

    height, width = image.shape[:2]
    crop_box = clamp_box(
        [item["bbox_x1"], item["bbox_y1"], item["bbox_x2"], item["bbox_y2"]],
        width,
        height,
        env_float(env, "READMRZ_OCR_LINE_MRZ_PADDING_RATIO", 0.06),
    )
    x1, y1, x2, y2 = crop_box
    crop = image[y1:y2, x1:x2].copy()
    if crop.size == 0:
        raise ValueError("MRZ crop is empty")

    rows = ocr_rows_for_crop(ocr, crop, temp_dir, int(item["id"]))
    selected_rows = select_mrz_line_rows(rows, env, crop)
    if not selected_rows:
        return 0

    cursor.execute("DELETE FROM dbo.readmrz_ocr_line_items WHERE label_item_id = ?", item["id"])
    split = str(item.get("split") or "train")
    output_dir = line_image_base_dir / split
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(str(item["image_file_name"])).stem
    normalize_orientation = env_bool(env, "READMRZ_OCR_LINE_NORMALIZE_ORIENTATION", True)

    line_count = 0
    for line_index, row in enumerate(selected_rows, start=1):
        line = line_crop_from_row(
            crop,
            row,
            crop_origin_x=x1,
            crop_origin_y=y1,
            padding_ratio=env_float(env, "READMRZ_OCR_LINE_PADDING_RATIO", 0.08),
            normalize_orientation=normalize_orientation,
        )
        if line["image"].size == 0:
            continue

        output_path = output_dir / f"{stem}_line{line_index}.jpg"
        cv2.imwrite(str(output_path), line["image"])
        write_line_row(
            cursor,
            item=item,
            line_index=line_index,
            line_image_rel=relative_to_base(output_path, line_image_base_dir),
            line_image_width=int(line["width"]),
            line_image_height=int(line["height"]),
            line_bbox_full=line["bbox_full"],
            doc_orientation_angle=int(line["doc_orientation_angle"]),
            ocr_text=str(row.get("normalized") or row.get("text") or ""),
            ocr_score=float(row.get("ocr_score") or 0.0),
        )
        line_count += 1

    return line_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate MRZ OCR line dataset from approved YOLO MRZ boxes.")
    parser.add_argument("--env", default=str(PROJECT_ROOT / ".env"), help="Path to .env config file.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max approved boxes to process.")
    parser.add_argument("--force", action="store_true", help="Regenerate line records even when parent is already done.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    env_path = Path(args.env)
    if not env_path.is_absolute():
        cwd_env_path = Path.cwd() / env_path
        project_env_path = PROJECT_ROOT / env_path
        env_path = cwd_env_path if cwd_env_path.exists() else project_env_path

    env = read_env_file(env_path)
    reexec_code = maybe_reexec_with_paddle_python(env)
    if reexec_code is not None:
        return reexec_code

    dataset_dir = yolo_dataset_dir(env)
    image_base_dir = resolve_base_dir(env, "READMRZ_YOLO_IMAGE_BASE_DIR", dataset_dir / "images")
    line_dataset_dir = resolve_base_dir(env, "READMRZ_OCR_LINE_DATASET_DIR", PROJECT_ROOT / "generated_datasets" / "mrz_ocr_lines")
    line_image_base_dir = resolve_base_dir(env, "READMRZ_OCR_LINE_IMAGE_BASE_DIR", line_dataset_dir / "images")
    batch_size = max(1, env_int(env, "READMRZ_OCR_LINE_BATCH_SIZE", 25))

    ocr = build_ocr(env)
    total = 0
    done = 0
    no_lines = 0
    errors = 0

    with tempfile.TemporaryDirectory(prefix="readmrz_line_ocr_") as temp_name:
        temp_dir = Path(temp_name)
        with connect() as connection:
            cursor = connection.cursor()
            items = fetch_approved_label_items(cursor, limit=args.limit, force=args.force)
            total = len(items)
            print(f"Found {total} approved MRZ boxes to extract")
            for index, item in enumerate(items, start=1):
                started = time.perf_counter()
                status = "error"
                line_count = 0
                try:
                    line_count = process_item(
                        cursor,
                        item=item,
                        image_base_dir=image_base_dir,
                        line_image_base_dir=line_image_base_dir,
                        ocr=ocr,
                        env=env,
                        temp_dir=temp_dir,
                    )
                    if line_count:
                        status = "done"
                        mark_parent(cursor, int(item["id"]), status="done", count=line_count)
                        done += 1
                    else:
                        status = "no_lines"
                        mark_parent(cursor, int(item["id"]), status="no_lines", count=0, error="No MRZ line rows found")
                        no_lines += 1
                except Exception as exc:
                    status = "error"
                    mark_parent(cursor, int(item["id"]), status="error", count=0, error=str(exc))
                    errors += 1

                if index % batch_size == 0:
                    connection.commit()
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                print(
                    f"[{index}/{total}] source={item['source_key']} "
                    f"status={status} lines={line_count} elapsed_ms={elapsed_ms}"
                )
            connection.commit()

    summary = {
        "done": done,
        "no_lines": no_lines,
        "errors": errors,
        "total": total,
        "line_dataset_dir": str(line_dataset_dir),
        "line_image_base_dir": str(line_image_base_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
