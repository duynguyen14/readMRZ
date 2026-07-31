from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

# Paddle 3.x on Windows CPU can enter a PIR/oneDNN path that fails for
# PaddleOCR orientation models. Set these before importing Paddle/PaddleOCR.
os.environ.setdefault("FLAGS_use_onednn", "0")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")

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
from generate_mrz_ocr_line_dataset import (  # noqa: E402
    ocr_rows_for_crop,
    select_mrz_line_rows,
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


def line2_ocr_env(env: dict[str, str]) -> dict[str, str]:
    ocr_env = dict(env)
    # The image passed to Paddle here is already a single deskewed MRZ line.
    # Running document/textline orientation again is slower and can trigger
    # Paddle PIR/oneDNN runtime errors on Windows CPU.
    ocr_env["READMRZ_PADDLE_FAST_MODE"] = "true"
    ocr_env["PADDLE_USE_DOC_ORIENTATION_CLASSIFY"] = "false"
    ocr_env["PADDLE_USE_TEXTLINE_ORIENTATION"] = "false"
    return ocr_env


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
            mrz_lines_json,
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
    max_angle = env_float(env, "READMRZ_OCR_LINE2_MAX_DESKEW_ANGLE", 15.0)
    kernel_width = max(3, env_int(env, "READMRZ_OCR_LINE2_DILATE_KERNEL_WIDTH", 35))
    kernel_height = max(1, env_int(env, "READMRZ_OCR_LINE2_DILATE_KERNEL_HEIGHT", 3))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_height, image_width = binary.shape[:2]
    candidates: list[tuple[float, float]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < image_width * 0.25 or h < 3:
            continue
        if w > image_width * 0.95 and h > image_height * 0.70:
            continue
        score = float(w * h) * (w / max(1, h))
        angle = normalize_min_area_angle(cv2.minAreaRect(contour))
        if abs(angle) <= max_angle:
            candidates.append((score, angle))

    hough_angle = estimate_skew_angle_hough(binary, max_angle)
    if hough_angle is not None:
        return hough_angle

    if candidates:
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        return candidates[0][1]

    component_angle = estimate_skew_angle_components(binary, max_angle)
    return component_angle if component_angle is not None else 0.0


def estimate_skew_angle_hough(binary: np.ndarray, max_angle: float) -> float | None:
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=30,
        minLineLength=max(60, int(binary.shape[1] * 0.12)),
        maxLineGap=20,
    )
    if lines is None:
        return None

    angles: list[float] = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        if abs(int(x2) - int(x1)) < 30:
            continue
        angle = float(np.degrees(np.arctan2(int(y2) - int(y1), int(x2) - int(x1))))
        if abs(angle) <= max_angle:
            angles.append(angle)

    if len(angles) < 3:
        return None
    return float(np.median(angles))


def estimate_skew_angle_components(binary: np.ndarray, max_angle: float) -> float | None:
    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    points: list[tuple[float, float]] = []
    for index in range(1, count):
        _, _, width, height, area = stats[index]
        if 3 <= width <= 45 and 6 <= height <= 55 and area >= 10:
            points.append((float(centroids[index][0]), float(centroids[index][1])))

    if len(points) < 8:
        return None

    xs = np.array([point[0] for point in points], dtype=np.float32)
    ys = np.array([point[1] for point in points], dtype=np.float32)
    slope, _ = np.polyfit(xs, ys, 1)
    angle = float(np.degrees(np.arctan(float(slope))))
    return angle if abs(angle) <= max_angle else None


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
    expected = max(1, env_int(env, "READMRZ_OCR_LINE2_EXPECTED_LINES", 2))
    component_bands = connected_component_line_bands(binary, expected)
    if len(component_bands) >= expected:
        return component_bands

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

    bands = split_wide_bands_by_valleys(projection, bands, expected, min_band_height)
    bands = force_expected_bands_from_projection(projection, bands, expected, min_band_height)
    bands = sorted(bands, key=lambda item: item[2] * (item[1] - item[0]), reverse=True)[:expected]
    return sorted(bands, key=lambda item: item[0])


def connected_component_line_bands(binary: np.ndarray, expected: int) -> list[tuple[int, int, float]]:
    if expected <= 0:
        return []

    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    components: list[dict[str, float]] = []
    image_height, image_width = binary.shape[:2]
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if not (2 <= width <= min(70, image_width) and 6 <= height <= min(70, image_height) and area >= 8):
            continue
        components.append(
            {
                "x1": float(x),
                "y1": float(y),
                "x2": float(x + width),
                "y2": float(y + height),
                "height": float(height),
                "area": float(area),
                "cy": float(centroids[index][1]),
            }
        )

    if len(components) < expected * 5:
        return []

    median_height = float(np.median([component["height"] for component in components]))
    y_threshold = max(10.0, median_height * 1.4)
    groups: list[dict[str, Any]] = []
    for component in sorted(components, key=lambda item: item["cy"]):
        if not groups or abs(component["cy"] - float(groups[-1]["cy_mean"])) > y_threshold:
            groups.append({"items": [component], "cy_mean": component["cy"]})
            continue
        group = groups[-1]
        group["items"].append(component)
        group["cy_mean"] = float(np.mean([item["cy"] for item in group["items"]]))

    candidates: list[tuple[float, int, int, float]] = []
    for group in groups:
        items = group["items"]
        if len(items) < 3:
            continue
        x1 = min(item["x1"] for item in items)
        x2 = max(item["x2"] for item in items)
        y1 = int(max(0, round(min(item["y1"] for item in items))))
        y2 = int(min(image_height - 1, round(max(item["y2"] for item in items))))
        span = x2 - x1
        if span < image_width * 0.20:
            continue
        area = sum(item["area"] for item in items)
        score = float(span * len(items) + area)
        candidates.append((score, y1, y2, score / max(1, y2 - y1)))

    if len(candidates) < expected:
        return []

    selected = sorted(candidates, key=lambda item: item[0], reverse=True)[:expected]
    return sorted([(y1, y2, score) for _, y1, y2, score in selected], key=lambda item: item[0])


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
    sorted_bands = sorted(bands, key=lambda item: item[0])
    binary = binarize_for_text(deskewed_crop)
    projection = binary.sum(axis=1).astype(np.float32)
    separators = line_separators_from_projection(projection, sorted_bands)
    for index, (y1, y2, score) in enumerate(sorted_bands):
        min_top = separators[index - 1] if index > 0 else 0
        max_bottom = separators[index] if index < len(separators) else height
        top, bottom = active_bounds_in_segment(
            projection,
            min_top,
            max_bottom,
            pad_y,
            env_float(env, "READMRZ_OCR_LINE2_LINE_ACTIVE_THRESHOLD_RATIO", 0.05),
        )
        if bottom <= top:
            top = max(0, y1)
            bottom = min(height, max(y2, y1 + 1))
        image = deskewed_crop[top:bottom, :].copy()
        if image.size == 0:
            continue
        left = 0
        right = width
        if env_value(env, "READMRZ_OCR_LINE2_TRIM_X", "false").strip().lower() in {"1", "true", "yes", "on"}:
            left, right = trim_line_x_bounds(image, width, env)
            image = image[:, left:right].copy()
        line_height, line_width = image.shape[:2]
        lines.append(
            {
                "image": image,
                "width": line_width,
                "height": line_height,
                "bbox_crop": [float(left), float(top), float(right), float(bottom)],
                "projection_score": score,
            }
        )
    return lines


def line_separators_from_projection(
    projection: np.ndarray,
    bands: list[tuple[int, int, float]],
) -> list[int]:
    separators: list[int] = []
    if len(bands) < 2:
        return separators

    height = len(projection)
    for left_band, right_band in zip(bands, bands[1:], strict=False):
        left_center = int(round((left_band[0] + left_band[1]) / 2.0))
        right_center = int(round((right_band[0] + right_band[1]) / 2.0))
        start = max(0, min(left_center, right_center))
        end = min(height, max(left_center, right_center) + 1)
        if end - start <= 1:
            separators.append(max(0, min(height, int(round((left_band[1] + right_band[0]) / 2.0)))))
            continue
        window = projection[start:end]
        separators.append(start + int(np.argmin(window)))
    return separators


def active_bounds_in_segment(
    projection: np.ndarray,
    segment_top: int,
    segment_bottom: int,
    pad_y: int,
    threshold_ratio: float,
) -> tuple[int, int]:
    height = len(projection)
    segment_top = max(0, min(height, int(segment_top)))
    segment_bottom = max(segment_top + 1, min(height, int(segment_bottom)))
    segment = projection[segment_top:segment_bottom]
    if segment.size == 0 or segment.max() <= 0:
        return segment_top, segment_bottom

    threshold = max(1.0, float(segment.max()) * threshold_ratio)
    active = np.where(segment >= threshold)[0]
    if active.size == 0:
        return segment_top, segment_bottom

    top = max(segment_top, segment_top + int(active.min()) - pad_y)
    bottom = min(segment_bottom, segment_top + int(active.max()) + pad_y + 1)
    if bottom <= top:
        return segment_top, segment_bottom
    return top, bottom


def trim_line_x_bounds(line_image: np.ndarray, fallback_width: int, env: dict[str, str]) -> tuple[int, int]:
    binary = binarize_for_text(line_image)
    component_bounds = trim_line_x_bounds_from_components(binary, fallback_width, env)
    if component_bounds is not None:
        return component_bounds

    projection = binary.sum(axis=0).astype(np.float32)
    if projection.max() <= 0:
        return 0, fallback_width

    threshold = max(255.0, projection.max() * env_float(env, "READMRZ_OCR_LINE2_TRIM_X_THRESHOLD_RATIO", 0.08))
    active = np.where(projection >= threshold)[0]
    if active.size == 0:
        return 0, fallback_width

    pad = max(6, int(round(fallback_width * env_float(env, "READMRZ_OCR_LINE2_TRIM_X_PADDING_RATIO", 0.02))))
    left = max(0, int(active.min()) - pad)
    right = min(fallback_width, int(active.max()) + pad + 1)
    if right - left < fallback_width * 0.25:
        return 0, fallback_width
    return left, right


def trim_line_x_bounds_from_components(
    binary: np.ndarray,
    fallback_width: int,
    env: dict[str, str],
) -> tuple[int, int] | None:
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    boxes: list[tuple[int, int, int]] = []
    image_height, image_width = binary.shape[:2]
    for index in range(1, count):
        x, _, width, height, area = stats[index]
        if not (2 <= width <= min(70, image_width) and 6 <= height <= min(70, image_height) and area >= 8):
            continue
        boxes.append((int(x), int(x + width), int(area)))

    if len(boxes) < 8:
        return None

    left = min(box[0] for box in boxes)
    right = max(box[1] for box in boxes)
    pad = max(6, int(round(fallback_width * env_float(env, "READMRZ_OCR_LINE2_TRIM_X_PADDING_RATIO", 0.02))))
    left = max(0, left - pad)
    right = min(fallback_width, right + pad)
    if right - left < fallback_width * 0.25:
        return None
    return left, right


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
    error_message: str | None = None,
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
            doc_orientation_angle,
            deskew_angle,
            projection_score,
            split_method,
            ocr_text,
            normalized_text,
            final_text,
            ocr_score,
            mrz_likeness,
            review_status,
            error_message,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'opencv_projection', ?, ?, ?, ?, ?, 'pending', ?, SYSUTCDATETIME(), SYSUTCDATETIME())
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
        0,
        deskew_angle,
        float(line.get("projection_score") or 0.0),
        ocr_text,
        normalized,
        normalized or None,
        ocr_score,
        likeness,
        error_message,
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


def count_line2_rows(cursor: Any, label_item_id: int) -> int:
    cursor.execute("SELECT COUNT(*) FROM dbo.readmrz_ocr_line_items2 WHERE label_item_id = ?", label_item_id)
    row = cursor.fetchone()
    return int(row[0] or 0) if row else 0


def parent_mrz_lines(item: dict[str, Any]) -> list[str]:
    raw = item.get("mrz_lines_json")
    if not raw:
        return []
    if isinstance(raw, list):
        values = raw
    else:
        try:
            values = json.loads(str(raw))
        except json.JSONDecodeError:
            return []
    lines: list[str] = []
    for value in values:
        text = normalize_mrz_text(str(value or ""))
        if text:
            lines.append(text)
    return lines


def process_item(
    cursor: Any,
    *,
    item: dict[str, Any],
    image_base_dir: Path,
    line_image_base_dir: Path,
    crop_base_dir: Path,
    raw_crop_base_dir: Path,
    ocr: Any | None,
    text_source: str,
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
    raw_crop_output_dir = raw_crop_base_dir / split
    line_output_dir.mkdir(parents=True, exist_ok=True)
    crop_output_dir.mkdir(parents=True, exist_ok=True)
    raw_crop_output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(str(item["image_file_name"])).stem
    raw_crop_path = raw_crop_output_dir / f"{stem}_mrz_raw_crop.jpg"
    cv2.imwrite(str(raw_crop_path), crop)
    crop_path = crop_output_dir / f"{stem}_mrz_crop.jpg"
    cv2.imwrite(str(crop_path), deskewed_crop)
    crop_rel = relative_to_base(crop_path, crop_base_dir)

    line_count = 0
    source_lines = parent_mrz_lines(item) if text_source == "parent" else []
    crop_ocr_error: str | None = None
    crop_ocr_rows: list[dict[str, Any]] = []
    if text_source == "paddle-crop" and ocr is not None:
        try:
            crop_rows = ocr_rows_for_crop(ocr, deskewed_crop, temp_dir, int(item["id"]))
            crop_ocr_rows = select_mrz_line_rows(crop_rows, env, deskewed_crop)
        except Exception as exc:
            crop_ocr_error = str(exc)[:1900]

    for line_index, line in enumerate(lines, start=1):
        line_path = line_output_dir / f"{stem}_line{line_index}.jpg"
        cv2.imwrite(str(line_path), line["image"])
        ocr_text = ""
        ocr_score = 0.0
        ocr_error: str | None = None
        if text_source == "paddle-crop":
            if line_index <= len(crop_ocr_rows):
                row = crop_ocr_rows[line_index - 1]
                ocr_text = str(row.get("normalized") or row.get("text") or "")
                ocr_score = float(row.get("ocr_score") or 0.0)
                ocr_error = "text_from_paddle_mrz_crop"
            else:
                ocr_error = crop_ocr_error or "paddle_crop_text_missing"
        elif text_source == "parent":
            if line_index <= len(source_lines):
                ocr_text = source_lines[line_index - 1]
                ocr_error = "text_from_parent_mrz_lines_json"
            else:
                ocr_error = "parent_text_missing"
        elif text_source == "none":
            ocr_error = "ocr_skipped"
        elif text_source == "paddle-line" and ocr is not None:
            try:
                ocr_text, ocr_score = paddle_ocr_line(ocr, line["image"], temp_dir, f"{stem}_line{line_index}")
            except Exception as exc:
                ocr_error = str(exc)[:1900]
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
            error_message=ocr_error,
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
    parser.add_argument("--skip-ocr", action="store_true", help="Only crop/split MRZ lines and insert DB rows with blank OCR text.")
    parser.add_argument(
        "--text-source",
        choices=["paddle", "paddle-crop", "paddle-line", "parent", "none"],
        default="",
        help="Text source for line labels. Default READMRZ_OCR_LINE2_TEXT_SOURCE or paddle-crop.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    env_path = Path(args.env)
    if not env_path.is_absolute():
        cwd_env_path = Path.cwd() / env_path
        project_env_path = PROJECT_ROOT / env_path
        env_path = cwd_env_path if cwd_env_path.exists() else project_env_path

    env = read_env_file(env_path)
    text_source = "none" if args.skip_ocr else (args.text_source or env_value(env, "READMRZ_OCR_LINE2_TEXT_SOURCE", "paddle-crop")).strip().lower()
    if text_source == "paddle":
        text_source = "paddle-crop"
    if text_source not in {"paddle-crop", "paddle-line", "parent", "none"}:
        raise ValueError(f"Invalid text source: {text_source}")

    if text_source in {"paddle-crop", "paddle-line"}:
        reexec_code = maybe_reexec_with_paddle_python(env)
        if reexec_code is not None:
            return reexec_code

    ensure_line2_schema()

    dataset_dir = yolo_dataset_dir(env)
    image_base_dir = (dataset_dir / "images").expanduser().resolve()
    line2_dataset_dir = resolve_base_dir(env, "READMRZ_OCR_LINE2_DATASET_DIR", PROJECT_ROOT / "generated_datasets" / "mrz_ocr_lines2")
    line2_image_base_dir = resolve_base_dir(env, "READMRZ_OCR_LINE2_IMAGE_BASE_DIR", line2_dataset_dir / "images")
    line2_crop_base_dir = resolve_base_dir(env, "READMRZ_OCR_LINE2_CROP_BASE_DIR", line2_dataset_dir / "crops")
    line2_raw_crop_base_dir = resolve_base_dir(env, "READMRZ_OCR_LINE2_RAW_CROP_BASE_DIR", line2_dataset_dir / "raw_crops")
    batch_size = max(1, env_int(env, "READMRZ_OCR_LINE2_BATCH_SIZE", 25))

    print(f"dataset_dir: {dataset_dir}")
    print(f"image_base_dir: {image_base_dir}")
    print(f"line2_dataset_dir: {line2_dataset_dir}")
    print(f"line2_image_base_dir: {line2_image_base_dir}")
    print(f"line2_crop_base_dir: {line2_crop_base_dir}")
    print(f"line2_raw_crop_base_dir: {line2_raw_crop_base_dir}")
    print(f"limit: {args.limit}")
    print(f"force: {args.force}")
    print(f"include_augmented: {args.include_augmented}")
    print(f"split: {args.split or 'all'}")
    print(f"skip_ocr: {args.skip_ocr}")
    print(f"text_source: {text_source}")

    if text_source == "paddle-crop":
        ocr = build_ocr(env)
    elif text_source == "paddle-line":
        ocr = build_ocr(line2_ocr_env(env))
    else:
        ocr = None
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
                db_rows = 0
                try:
                    line_count, band_count, deskew_angle = process_item(
                        cursor,
                        item=item,
                        image_base_dir=image_base_dir,
                        line_image_base_dir=line2_image_base_dir,
                        crop_base_dir=line2_crop_base_dir,
                        raw_crop_base_dir=line2_raw_crop_base_dir,
                        ocr=ocr,
                        text_source=text_source,
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

                try:
                    db_rows = count_line2_rows(cursor, int(item["id"]))
                except Exception:
                    db_rows = -1
                if index % batch_size == 0:
                    connection.commit()
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                print(
                    f"[{index}/{total}] source={item['source_key']} "
                    f"status={status} lines={line_count} db_rows={db_rows} bands={band_count} "
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
        "line2_raw_crop_base_dir": str(line2_raw_crop_base_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
