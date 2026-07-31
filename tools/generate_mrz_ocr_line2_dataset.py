from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
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
    env_float,
    env_int,
    env_value,
    line_mrz_likeness,
    read_env_file,
)
from mrz_reader.db import connect, execute_sql_file  # noqa: E402
from mrz_reader.env_config import yolo_dataset_dir  # noqa: E402
from mrz_reader.mrz import normalize_mrz_text  # noqa: E402


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
    print(f"Re-running OCR line2 script with Paddle Python: {python_path}")
    return subprocess.run(command, cwd=str(PROJECT_ROOT), check=False).returncode


def resolve_base_dir(env: dict[str, str], key: str, default_path: Path) -> Path:
    return Path(env_value(env, key, str(default_path))).expanduser().resolve()


def relative_to_base(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return path.name


def row_to_dict(cursor: Any, row: Any) -> dict[str, Any]:
    columns = [column[0] for column in cursor.description]
    return dict(zip(columns, row))


def ensure_line2_schema() -> None:
    execute_sql_file(PROJECT_ROOT / "sql" / "create_mrz_ocr_line2_tables.sql")


def fetch_approved_label_items(
    cursor: Any,
    *,
    limit: int,
    force: bool,
    include_augmented: bool,
    split: str,
) -> list[dict[str, Any]]:
    top_clause = f"TOP {limit}" if limit > 0 else ""
    force_filter = "" if force else "AND ISNULL(mrz_line2_extract_status, '') <> 'done'"
    augmented_filter = "" if include_augmented else "AND source_key NOT LIKE '%__rot90' AND source_key NOT LIKE '%__rot180' AND source_key NOT LIKE '%__rot270'"
    split_filter = "AND split = ?" if split else ""
    params: list[Any] = []
    if split:
        params.append(split)
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
          {augmented_filter}
          {split_filter}
        ORDER BY id ASC
        """,
        *params,
    )
    return [row_to_dict(cursor, row) for row in cursor.fetchall()]


def clamp_box(box: list[Any], width: int, height: int, padding_ratio: float) -> list[int]:
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


def binarize_for_text(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def normalize_min_area_angle(rect: tuple[Any, Any, float]) -> float:
    (_, _), (width, height), raw_angle = rect
    angle = float(raw_angle)
    if width < height:
        angle += 90.0
    if angle > 45:
        angle -= 90.0
    if angle < -45:
        angle += 90.0
    return angle


def estimate_skew_angle(binary: np.ndarray, env: dict[str, str]) -> float:
    kernel_width = max(3, env_int(env, "READMRZ_OCR_LINE2_DILATE_KERNEL_WIDTH", 35))
    kernel_height = max(1, env_int(env, "READMRZ_OCR_LINE2_DILATE_KERNEL_HEIGHT", 3))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    image_height, image_width = binary.shape[:2]
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < image_width * 0.25 or h < 3:
            continue
        score = float(w * h) * (w / max(1, h))
        candidates.append((score, contour))
    if not candidates:
        return 0.0

    candidates.sort(key=lambda pair: pair[0], reverse=True)
    contour = candidates[0][1]
    angle = normalize_min_area_angle(cv2.minAreaRect(contour))
    max_angle = env_float(env, "READMRZ_OCR_LINE2_MAX_DESKEW_ANGLE", 15.0)
    if abs(angle) > max_angle:
        return 0.0
    return angle


def rotate_image(image: np.ndarray, angle: float, border_value: tuple[int, int, int] | int) -> np.ndarray:
    if abs(angle) < 0.1:
        return image
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
        borderValue=border_value,
    )


def deskew_crop(crop: np.ndarray, env: dict[str, str]) -> tuple[np.ndarray, np.ndarray, float]:
    binary = binarize_for_text(crop)
    angle = estimate_skew_angle(binary, env)
    deskewed = rotate_image(crop, angle, border_value=(255, 255, 255))
    deskewed_binary = binarize_for_text(deskewed)
    return deskewed, deskewed_binary, angle


def projection_bands(binary: np.ndarray, env: dict[str, str]) -> list[tuple[int, int, float]]:
    height, _ = binary.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 2))
    clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    projection = clean.sum(axis=1).astype(np.float32)
    if projection.max() <= 0:
        return []

    threshold = projection.max() * env_float(env, "READMRZ_OCR_LINE2_PROJECTION_THRESHOLD_RATIO", 0.12)
    min_band_height = max(3, env_int(env, "READMRZ_OCR_LINE2_MIN_BAND_HEIGHT", 6))
    bands: list[tuple[int, int, float]] = []
    in_band = False
    start = 0
    for index, value in enumerate(projection):
        if value >= threshold and not in_band:
            in_band = True
            start = index
        elif value < threshold and in_band:
            in_band = False
            if index - start >= min_band_height:
                bands.append((start, index, float(projection[start:index].mean())))
    if in_band and height - start >= min_band_height:
        bands.append((start, height - 1, float(projection[start:].mean())))

    expected = max(1, env_int(env, "READMRZ_OCR_LINE2_EXPECTED_LINES", 2))
    bands = split_wide_bands_by_valleys(projection, bands, expected, min_band_height)
    bands = force_expected_bands_from_projection(projection, bands, expected, min_band_height)
    bands = sorted(bands, key=lambda item: item[2] * (item[1] - item[0]), reverse=True)[:expected]
    return sorted(bands, key=lambda item: item[0])


def split_wide_bands_by_valleys(
    projection: np.ndarray,
    bands: list[tuple[int, int, float]],
    expected: int,
    min_band_height: int,
) -> list[tuple[int, int, float]]:
    if expected <= 1 or len(bands) >= expected or not bands:
        return bands

    result = list(bands)
    while len(result) < expected:
        result.sort(key=lambda item: item[1] - item[0], reverse=True)
        y1, y2, score = result.pop(0)
        if y2 - y1 < min_band_height * 3:
            result.append((y1, y2, score))
            break

        split_y = find_best_valley(projection, y1, y2, min_band_height)
        if split_y is None:
            result.append((y1, y2, score))
            break

        left = (y1, split_y, float(projection[y1:split_y].mean()))
        right = (split_y, y2, float(projection[split_y:y2].mean()))
        result.extend([left, right])

    return result


def find_best_valley(projection: np.ndarray, y1: int, y2: int, min_band_height: int) -> int | None:
    start = y1 + min_band_height
    end = y2 - min_band_height
    if end <= start:
        return None

    band_height = y2 - y1
    center_start = y1 + int(band_height * 0.30)
    center_end = y1 + int(band_height * 0.70)
    start = max(start, center_start)
    end = min(end, center_end)
    if end <= start:
        return None

    window = projection[start:end]
    if window.size == 0:
        return None
    return int(start + int(np.argmin(window)))


def force_expected_bands_from_projection(
    projection: np.ndarray,
    bands: list[tuple[int, int, float]],
    expected: int,
    min_band_height: int,
) -> list[tuple[int, int, float]]:
    if expected <= 1 or len(bands) >= expected:
        return bands

    active = np.where(projection > max(1.0, projection.max() * 0.03))[0]
    if active.size == 0:
        return bands

    top = int(active.min())
    bottom = int(active.max()) + 1
    if bottom - top < min_band_height * expected:
        return bands

    band_height = max(min_band_height, int(round((bottom - top) / expected)))
    forced: list[tuple[int, int, float]] = []
    for index in range(expected):
        y1 = top + index * band_height
        y2 = bottom if index == expected - 1 else min(bottom, y1 + band_height)
        if y2 - y1 >= min_band_height:
            forced.append((y1, y2, float(projection[y1:y2].mean())))
    return forced if len(forced) >= expected else bands


def crop_line_images(
    deskewed_crop: np.ndarray,
    bands: list[tuple[int, int, float]],
    env: dict[str, str],
) -> list[dict[str, Any]]:
    height, width = deskewed_crop.shape[:2]
    pad_y = max(2, int(round(height * env_float(env, "READMRZ_OCR_LINE2_LINE_PADDING_RATIO", 0.06))))
    lines: list[dict[str, Any]] = []
    for y1, y2, score in bands:
        top = max(0, y1 - pad_y)
        bottom = min(height, y2 + pad_y)
        image = deskewed_crop[top:bottom, :].copy()
        if image.size == 0:
            continue
        line_height, line_width = image.shape[:2]
        lines.append(
            {
                "image": image,
                "width": line_width,
                "height": line_height,
                "bbox_crop": [0.0, float(top), float(width - 1), float(bottom)],
                "projection_score": score,
            }
        )
    return lines


def paddle_ocr_line(ocr: Any, line_image: np.ndarray, temp_dir: Path, stem: str) -> tuple[str, float]:
    temp_path = temp_dir / f"{stem}.jpg"
    cv2.imwrite(str(temp_path), line_image)
    try:
        rows = ocr.predict_rows(temp_path, line_image)
    finally:
        temp_path.unlink(missing_ok=True)

    candidates = [
        row
        for row in rows
        if normalize_mrz_text(str(row.get("normalized") or row.get("text") or ""))
    ]
    if not candidates:
        return "", 0.0

    candidates.sort(
        key=lambda row: (
            line_mrz_likeness(str(row.get("normalized") or row.get("text") or "")),
            len(str(row.get("normalized") or "")),
            float(row.get("ocr_score") or 0),
        ),
        reverse=True,
    )
    best = candidates[0]
    text = str(best.get("normalized") or best.get("text") or "")
    return normalize_mrz_text(text), float(best.get("ocr_score") or 0.0)


def write_line_row2(
    cursor: Any,
    *,
    item: dict[str, Any],
    line_index: int,
    crop_rel: str,
    line_rel: str,
    line: dict[str, Any],
    mrz_bbox: list[int],
    deskew_angle: float,
    ocr_text: str,
    ocr_score: float,
) -> None:
    normalized = normalize_mrz_text(ocr_text)
    likeness = line_mrz_likeness(ocr_text) if ocr_text else 0.0
    line_bbox = line["bbox_crop"]
    cursor.execute(
        """
        INSERT INTO dbo.readmrz_ocr_line_items2 (
            label_item_id,
            source_key,
            split,
            line_index,
            mrz_crop_file_name,
            line_image_file_name,
            line_image_width,
            line_image_height,
            mrz_bbox_x1,
            mrz_bbox_y1,
            mrz_bbox_x2,
            mrz_bbox_y2,
            line_bbox_x1,
            line_bbox_y1,
            line_bbox_x2,
            line_bbox_y2,
            deskew_angle,
            projection_score,
            split_method,
            ocr_text,
            normalized_text,
            final_text,
            ocr_score,
            mrz_likeness,
            review_status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'opencv_projection', ?, ?, ?, ?, ?, 'pending')
        """,
        item["id"],
        item["source_key"],
        item.get("split"),
        line_index,
        crop_rel,
        line_rel,
        line["width"],
        line["height"],
        mrz_bbox[0],
        mrz_bbox[1],
        mrz_bbox[2],
        mrz_bbox[3],
        line_bbox[0],
        line_bbox[1],
        line_bbox[2],
        line_bbox[3],
        deskew_angle,
        float(line.get("projection_score") or 0.0),
        ocr_text,
        normalized,
        normalized or None,
        ocr_score,
        likeness,
    )


def mark_parent2(cursor: Any, item_id: int, *, status: str, count: int = 0, error: str | None = None) -> None:
    cursor.execute(
        """
        UPDATE dbo.readmrz_label_items
        SET mrz_line2_extract_status = ?,
            mrz_line2_extract_count = ?,
            mrz_line2_extract_error = ?,
            mrz_line2_extracted_at = SYSUTCDATETIME(),
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
    crop_base_dir: Path,
    ocr: Any,
    env: dict[str, str],
    temp_dir: Path,
) -> tuple[int, int, float]:
    image_path = image_base_dir / str(item["image_file_name"])
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read YOLO image: {image_path}")

    height, width = image.shape[:2]
    mrz_bbox = clamp_box(
        [item["bbox_x1"], item["bbox_y1"], item["bbox_x2"], item["bbox_y2"]],
        width,
        height,
        env_float(env, "READMRZ_OCR_LINE2_MRZ_PADDING_RATIO", 0.08),
    )
    x1, y1, x2, y2 = mrz_bbox
    crop = image[y1:y2, x1:x2].copy()
    if crop.size == 0:
        raise ValueError("MRZ crop is empty")

    deskewed_crop, binary, deskew_angle = deskew_crop(crop, env)
    bands = projection_bands(binary, env)
    lines = crop_line_images(deskewed_crop, bands, env)
    if not lines:
        return 0, len(bands), deskew_angle

    cursor.execute("DELETE FROM dbo.readmrz_ocr_line_items2 WHERE label_item_id = ?", item["id"])
    split = str(item.get("split") or "train")
    line_output_dir = line_image_base_dir / split
    crop_output_dir = crop_base_dir / split
    line_output_dir.mkdir(parents=True, exist_ok=True)
    crop_output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(str(item["image_file_name"])).stem
    crop_path = crop_output_dir / f"{stem}_mrz_crop.jpg"
    cv2.imwrite(str(crop_path), deskewed_crop)
    crop_rel = relative_to_base(crop_path, crop_base_dir)

    line_count = 0
    for line_index, line in enumerate(lines, start=1):
        line_path = line_output_dir / f"{stem}_line{line_index}.jpg"
        cv2.imwrite(str(line_path), line["image"])
        ocr_text, ocr_score = paddle_ocr_line(ocr, line["image"], temp_dir, f"{stem}_line{line_index}")
        write_line_row2(
            cursor,
            item=item,
            line_index=line_index,
            crop_rel=crop_rel,
            line_rel=relative_to_base(line_path, line_image_base_dir),
            line=line,
            mrz_bbox=mrz_bbox,
            deskew_angle=deskew_angle,
            ocr_text=ocr_text,
            ocr_score=ocr_score,
        )
        line_count += 1

    return line_count, len(bands), deskew_angle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate MRZ OCR line2 dataset using OpenCV deskew/projection + PaddleOCR text.")
    parser.add_argument("--env", default=str(PROJECT_ROOT / ".env"), help="Path to .env config file.")
    parser.add_argument("--limit", type=int, default=1000, help="Max approved boxes to process. Use 0 to process all.")
    parser.add_argument("--force", action="store_true", help="Regenerate line2 records even when parent is already done.")
    parser.add_argument("--include-augmented", action="store_true", help="Also process records whose source_key ends with __rot90/__rot180/__rot270.")
    parser.add_argument("--split", choices=["train", "val"], default="", help="Optional YOLO split filter.")
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

    ensure_line2_schema()

    dataset_dir = yolo_dataset_dir(env)
    image_base_dir = (dataset_dir / "images").expanduser().resolve()
    line2_dataset_dir = resolve_base_dir(env, "READMRZ_OCR_LINE2_DATASET_DIR", PROJECT_ROOT / "generated_datasets" / "mrz_ocr_lines2")
    line2_image_base_dir = resolve_base_dir(env, "READMRZ_OCR_LINE2_IMAGE_BASE_DIR", line2_dataset_dir / "images")
    line2_crop_base_dir = resolve_base_dir(env, "READMRZ_OCR_LINE2_CROP_BASE_DIR", line2_dataset_dir / "crops")
    batch_size = max(1, env_int(env, "READMRZ_OCR_LINE2_BATCH_SIZE", 25))

    print(f"dataset_dir: {dataset_dir}")
    print(f"image_base_dir: {image_base_dir}")
    print(f"line2_dataset_dir: {line2_dataset_dir}")
    print(f"line2_image_base_dir: {line2_image_base_dir}")
    print(f"line2_crop_base_dir: {line2_crop_base_dir}")
    print(f"limit: {args.limit}")
    print(f"force: {args.force}")
    print(f"include_augmented: {args.include_augmented}")
    print(f"split: {args.split or 'all'}")

    ocr = build_ocr(env)
    total = 0
    done = 0
    no_lines = 0
    errors = 0

    with tempfile.TemporaryDirectory(prefix="readmrz_line2_ocr_") as temp_name:
        temp_dir = Path(temp_name)
        with connect() as connection:
            cursor = connection.cursor()
            items = fetch_approved_label_items(
                cursor,
                limit=args.limit,
                force=args.force,
                include_augmented=args.include_augmented,
                split=args.split,
            )
            total = len(items)
            print(f"Found {total} approved MRZ boxes to extract with OpenCV line2")
            for index, item in enumerate(items, start=1):
                started = time.perf_counter()
                status = "error"
                line_count = 0
                band_count = 0
                deskew_angle = 0.0
                try:
                    line_count, band_count, deskew_angle = process_item(
                        cursor,
                        item=item,
                        image_base_dir=image_base_dir,
                        line_image_base_dir=line2_image_base_dir,
                        crop_base_dir=line2_crop_base_dir,
                        ocr=ocr,
                        env=env,
                        temp_dir=temp_dir,
                    )
                    if line_count:
                        status = "done"
                        mark_parent2(cursor, int(item["id"]), status="done", count=line_count)
                        done += 1
                    else:
                        status = "no_lines"
                        mark_parent2(cursor, int(item["id"]), status="no_lines", count=0, error="No projection bands found")
                        no_lines += 1
                except Exception as exc:
                    status = "error"
                    error_text = str(exc)
                    try:
                        mark_parent2(cursor, int(item["id"]), status="error", count=0, error=error_text)
                    except Exception as mark_exc:
                        error_text = f"{error_text}; mark_parent_failed={mark_exc}"
                    errors += 1

                if index % batch_size == 0:
                    connection.commit()
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                print(
                    f"[{index}/{total}] source={item['source_key']} "
                    f"status={status} lines={line_count} bands={band_count} "
                    f"angle={deskew_angle:.2f} elapsed_ms={elapsed_ms}"
                    + (f" error={error_text}" if status == "error" else "")
                )
            connection.commit()

    summary = {
        "done": done,
        "no_lines": no_lines,
        "errors": errors,
        "total": total,
        "line2_dataset_dir": str(line2_dataset_dir),
        "line2_image_base_dir": str(line2_image_base_dir),
        "line2_crop_base_dir": str(line2_crop_base_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
