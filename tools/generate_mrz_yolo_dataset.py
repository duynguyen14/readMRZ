from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mrz_reader.mrz import normalize_mrz_text


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def line_mrz_likeness(text: str) -> float:
    normalized = normalize_mrz_text(text)
    if not normalized:
        return 0.0
    charset_ratio = len(normalized) / max(1, len(text.replace(" ", "")))
    length_score = min(1.0, len(normalized) / 30)
    filler_score = min(1.0, normalized.count("<") / max(1, len(normalized)) * 4)
    digit_score = 1.0 if any(ch.isdigit() for ch in normalized) else 0.65
    return 0.45 * charset_ratio + 0.30 * length_score + 0.15 * filler_score + 0.10 * digit_score


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        values[key.strip()] = value
    return values


def env_value(env: dict[str, str], key: str, default: str = "") -> str:
    return os.environ.get(key) or env.get(key) or default


def env_float(env: dict[str, str], key: str, default: float) -> float:
    value = env_value(env, key)
    if not value:
        return default
    return float(value)


def env_int(env: dict[str, str], key: str, default: int) -> int:
    value = env_value(env, key)
    if not value:
        return default
    return int(value)


def env_bool(env: dict[str, str], key: str, default: bool) -> bool:
    value = env_value(env, key)
    if not value:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def read_local_model_name(model_dir: Path) -> str | None:
    inference_config_path = model_dir / "inference.yml"
    if not inference_config_path.exists():
        return None

    try:
        config_text = inference_config_path.read_text(encoding="utf-8")
    except OSError:
        return None

    for raw_line in config_text.splitlines():
        line = raw_line.strip()
        if line.startswith("model_name:"):
            return line.split(":", 1)[1].strip().split()[0]
    return None


class RapidOcrAdapter:
    name = "rapidocr_onnxruntime"

    def __init__(self, env: dict[str, str]) -> None:
        from rapidocr_onnxruntime import RapidOCR

        kwargs: dict[str, Any] = {
            "text_score": env_float(env, "READMRZ_MIN_OCR_SCORE", 0.35),
            "use_angle_cls": False,
            "print_verbose": False,
        }

        model_env_map = {
            "READMRZ_DET_MODEL_PATH": "det_model_path",
            "READMRZ_REC_MODEL_PATH": "rec_model_path",
            "READMRZ_CLS_MODEL_PATH": "cls_model_path",
        }
        for env_key, kwarg_key in model_env_map.items():
            model_path = env_value(env, env_key)
            if model_path:
                kwargs[kwarg_key] = model_path

        self.kwargs = kwargs
        self.pipeline = RapidOCR(**kwargs)

    def config_summary(self) -> dict[str, Any]:
        return dict(self.kwargs)

    def predict_rows(self, image_path: Path, image: np.ndarray) -> list[dict[str, Any]]:
        height, width = image.shape[:2]
        result, _ = self.pipeline(image)
        rows: list[dict[str, Any]] = []
        for raw_row in result or []:
            row = row_from_rapidocr(raw_row, height, width)
            if row is not None:
                rows.append(row)
        return rows


class PaddleOcrAdapter:
    name = "paddleocr"

    def __init__(self, env: dict[str, str]) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is not installed in this Python env. "
                "Run this script with D:/DocHochieu/Backend/.venv/Scripts/python.exe "
                "or install paddleocr/paddlepaddle in the current env."
            ) from exc

        os.environ["PADDLE_PDX_MODEL_SOURCE"] = env_value(env, "PADDLE_PDX_MODEL_SOURCE", "BOS")
        kwargs = self.build_kwargs(env)
        self.kwargs = kwargs
        print("Loading PaddleOCR with:")
        for key, value in kwargs.items():
            if "dir" in key or "name" in key or key in {"lang", "ocr_version", "device"}:
                print(f"  {key}={value}")
        self.pipeline = PaddleOCR(**kwargs)

    def config_summary(self) -> dict[str, Any]:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in self.kwargs.items()
        }

    def build_kwargs(self, env: dict[str, str]) -> dict[str, Any]:
        fast_mode = env_bool(env, "READMRZ_PADDLE_FAST_MODE", False)
        kwargs: dict[str, Any] = {
            "lang": env_value(env, "OCR_LANGUAGE", "en"),
            "ocr_version": env_value(env, "PADDLE_OCR_VERSION", "PP-OCRv5"),
            "device": env_value(env, "PADDLE_OCR_DEVICE", "cpu"),
            "use_doc_orientation_classify": False
            if fast_mode
            else env_bool(env, "PADDLE_USE_DOC_ORIENTATION_CLASSIFY", True),
            "use_doc_unwarping": False,
            "use_textline_orientation": False
            if fast_mode
            else env_bool(env, "PADDLE_USE_TEXTLINE_ORIENTATION", True),
            "return_word_box": True,
        }

        model_dir_map = {
            "PADDLE_DOC_ORIENTATION_MODEL_DIR": "doc_orientation_classify_model",
            "PADDLE_TEXT_DETECTION_MODEL_DIR": "text_detection_model",
            "PADDLE_TEXT_RECOGNITION_MODEL_DIR": "text_recognition_model",
            "PADDLE_TEXTLINE_ORIENTATION_MODEL_DIR": "textline_orientation_model",
        }
        for env_key, kwarg_prefix in model_dir_map.items():
            if fast_mode and env_key in {
                "PADDLE_DOC_ORIENTATION_MODEL_DIR",
                "PADDLE_TEXTLINE_ORIENTATION_MODEL_DIR",
            }:
                continue
            raw_model_dir = env_value(env, env_key)
            if not raw_model_dir:
                continue

            model_dir = Path(raw_model_dir).expanduser().resolve()
            model_name = read_local_model_name(model_dir)
            if model_name:
                kwargs[f"{kwarg_prefix}_name"] = model_name
            kwargs[f"{kwarg_prefix}_dir"] = str(model_dir)

        return kwargs

    def predict_rows(self, image_path: Path, image: np.ndarray) -> list[dict[str, Any]]:
        height, width = image.shape[:2]
        result = self.pipeline.predict(str(image_path))
        if not result:
            return []

        item = result[0]
        doc_orientation_angle = extract_doc_orientation_angle(item)
        rows: list[dict[str, Any]] = []
        for text, score, raw_polygon, raw_box in zip(
            item.get("rec_texts", []),
            item.get("rec_scores", []),
            item.get("rec_polys", []),
            item.get("rec_boxes", []),
            strict=False,
        ):
            row = row_from_paddleocr(
                text,
                score,
                raw_polygon,
                raw_box,
                height,
                width,
                doc_orientation_angle=doc_orientation_angle,
            )
            if row is not None:
                rows.append(row)
        return rows


def build_ocr(env: dict[str, str]) -> Any:
    engine = env_value(env, "READMRZ_OCR_ENGINE", "paddle").strip().lower()
    if engine == "rapidocr":
        return RapidOcrAdapter(env)
    if engine == "paddle":
        return PaddleOcrAdapter(env)
    raise ValueError("READMRZ_OCR_ENGINE must be 'paddle' or 'rapidocr'")


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
    print(f"Re-running with Paddle Python: {python_path}")
    return subprocess.run(command, cwd=str(PROJECT_ROOT), check=False).returncode


def build_rapid_ocr(env: dict[str, str]) -> Any:
    # Kept for backward compatibility with older imports.
    return RapidOcrAdapter(env)


def _legacy_build_rapidocr(env: dict[str, str]) -> Any:
    from rapidocr_onnxruntime import RapidOCR

    kwargs: dict[str, Any] = {
        "text_score": env_float(env, "READMRZ_MIN_OCR_SCORE", 0.35),
        "use_angle_cls": False,
        "print_verbose": False,
    }

    model_env_map = {
        "READMRZ_DET_MODEL_PATH": "det_model_path",
        "READMRZ_REC_MODEL_PATH": "rec_model_path",
        "READMRZ_CLS_MODEL_PATH": "cls_model_path",
    }
    for env_key, kwarg_key in model_env_map.items():
        model_path = env_value(env, env_key)
        if model_path:
            kwargs[kwarg_key] = model_path

    return RapidOCR(**kwargs)


def list_images(source_dir: Path, *, output_dir: Path) -> list[Path]:
    ignored_dir_names = {".git", ".idea", ".venv", "__pycache__", "build", "dist", "generated_datasets", "datasets"}
    resolved_output_dir = output_dir.resolve()
    images: list[Path] = []
    for path in source_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        path_parts = set(path.relative_to(source_dir).parts)
        if path_parts & ignored_dir_names:
            continue

        try:
            path.resolve().relative_to(resolved_output_dir)
            continue
        except ValueError:
            pass

        images.append(path)

    return sorted(images)


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def load_processed(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "items": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def load_processed_index(path: Path) -> dict[str, dict[str, Any]]:
    processed = load_processed(path)
    items = processed.get("items", {})
    if not isinstance(items, dict):
        return {}
    return {
        key: {
            "fingerprint": item.get("fingerprint"),
            "status": item.get("status"),
        }
        for key, item in items.items()
        if isinstance(item, dict)
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, items: dict[str, dict[str, Any]]) -> None:
    if not items:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output_file:
        for key, item in items.items():
            output_file.write(json.dumps({"key": key, **item}, ensure_ascii=False) + "\n")


def flush_processed_batch(
    *,
    processed_path: Path,
    run_items_path: Path,
    batch_items: dict[str, dict[str, Any]],
    ocr: Any,
) -> int:
    if not batch_items:
        return 0

    processed = load_processed(processed_path)
    processed_items = processed.setdefault("items", {})
    if not isinstance(processed_items, dict):
        processed_items = {}
        processed["items"] = processed_items

    processed["ocr_engine"] = getattr(ocr, "name", "unknown")
    if hasattr(ocr, "config_summary"):
        processed["ocr_config"] = ocr.config_summary()
    processed["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    processed_items.update(batch_items)
    save_json(processed_path, processed)
    append_jsonl(run_items_path, batch_items)
    flushed = len(batch_items)
    batch_items.clear()
    return flushed


def normalize_points(box: Any) -> np.ndarray:
    points = np.asarray(box, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Invalid OCR box")
    return points


def normalize_angle(value: Any) -> int:
    try:
        angle = int(round(float(value))) % 360
    except (TypeError, ValueError):
        return 0
    return angle if angle in {0, 90, 180, 270} else 0


def extract_doc_orientation_angle(item: dict[str, Any]) -> int:
    candidates: list[Any] = []
    doc_preprocessor = item.get("doc_preprocessor_res")
    if isinstance(doc_preprocessor, dict):
        for key in ("angle", "doc_orientation_angle", "doc_ori_angle", "orientation_angle", "rotate_angle"):
            if key in doc_preprocessor:
                candidates.append(doc_preprocessor.get(key))

    for key in ("doc_orientation_angle", "doc_ori_angle", "orientation_angle", "rotate_angle"):
        if key in item:
            candidates.append(item.get(key))

    for value in candidates:
        angle = normalize_angle(value)
        if angle:
            return angle
    return 0


def map_corrected_points_to_original(points: np.ndarray, angle: int, image_width: int, image_height: int) -> np.ndarray:
    normalized_angle = normalize_angle(angle)
    if normalized_angle == 0:
        return points

    mapped = points.astype(np.float32, copy=True)
    x = mapped[:, 0].copy()
    y = mapped[:, 1].copy()

    if normalized_angle == 180:
        mapped[:, 0] = image_width - 1 - x
        mapped[:, 1] = image_height - 1 - y
    elif normalized_angle == 90:
        mapped[:, 0] = image_width - 1 - y
        mapped[:, 1] = x
    elif normalized_angle == 270:
        mapped[:, 0] = y
        mapped[:, 1] = image_height - 1 - x

    mapped[:, 0] = np.clip(mapped[:, 0], 0, max(0, image_width - 1))
    mapped[:, 1] = np.clip(mapped[:, 1], 0, max(0, image_height - 1))
    return mapped


def build_row_from_points(
    *,
    text: str,
    score: float,
    points: np.ndarray,
    image_height: int,
    image_width: int,
) -> dict[str, Any] | None:
    normalized = normalize_mrz_text(text)
    if not normalized:
        return None

    if points.size == 0:
        return None

    x_min = float(np.min(points[:, 0]))
    y_min = float(np.min(points[:, 1]))
    x_max = float(np.max(points[:, 0]))
    y_max = float(np.max(points[:, 1]))
    height = max(1.0, y_max - y_min)
    width = max(1.0, x_max - x_min)
    y_center = (y_min + y_max) / 2
    x_center = (x_min + x_max) / 2

    return {
        "text": text,
        "normalized": normalized,
        "ocr_score": float(score),
        "likeness": line_mrz_likeness(text),
        "points": points,
        "bbox_xyxy": [x_min, y_min, x_max, y_max],
        "x_center": x_center,
        "y_center": y_center,
        "width": width,
        "height": height,
        "bottom_bias": y_center / max(1, image_height),
        "width_ratio": width / max(1, image_width),
    }


def row_from_rapidocr(raw_row: Any, image_height: int, image_width: int) -> dict[str, Any] | None:
    if len(raw_row) < 3:
        return None
    text = str(raw_row[1])

    try:
        points = normalize_points(raw_row[0])
    except Exception:
        return None

    return build_row_from_points(
        text=text,
        score=float(raw_row[2]),
        points=points,
        image_height=image_height,
        image_width=image_width,
    )


def row_from_paddleocr(
    text: Any,
    score: Any,
    raw_polygon: Any,
    raw_box: Any,
    image_height: int,
    image_width: int,
    doc_orientation_angle: int = 0,
) -> dict[str, Any] | None:
    try:
        points = normalize_points(raw_polygon)
    except Exception:
        points = np.empty((0, 2), dtype=np.float32)

    if points.size == 0 and raw_box is not None:
        try:
            box = [float(value) for value in raw_box[:4]]
            x_min, y_min, x_max, y_max = box
            points = np.asarray(
                [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
                dtype=np.float32,
            )
        except Exception:
            return None

    points = map_corrected_points_to_original(points, doc_orientation_angle, image_width, image_height)

    try:
        score_value = float(score)
    except (TypeError, ValueError):
        score_value = 0.0

    row = build_row_from_points(
        text=str(text),
        score=score_value,
        points=points,
        image_height=image_height,
        image_width=image_width,
    )
    if row is not None:
        row["doc_orientation_angle"] = doc_orientation_angle
    return row


def group_score(group: list[dict[str, Any]], image_height: int, image_width: int) -> float:
    points = np.vstack([row["points"] for row in group])
    x_min = float(np.min(points[:, 0]))
    y_min = float(np.min(points[:, 1]))
    x_max = float(np.max(points[:, 0]))
    y_max = float(np.max(points[:, 1]))
    union_width_ratio = (x_max - x_min) / max(1, image_width)
    union_center_y = ((y_min + y_max) / 2) / max(1, image_height)
    avg_height = sum(row["height"] for row in group) / len(group)
    vertical_span = y_max - y_min
    vertical_penalty = max(0.0, (vertical_span / max(1.0, avg_height) - (len(group) * 2.6))) * 0.08
    avg_quality = sum(
        row["ocr_score"] * 0.30
        + row["likeness"] * 0.45
        + min(1.0, len(row["normalized"]) / 36) * 0.25
        for row in group
    ) / len(group)
    return avg_quality + union_width_ratio * 0.20 + union_center_y * 0.18 - vertical_penalty


def find_mrz_group(
    rows: list[dict[str, Any]],
    *,
    min_likeness: float,
    min_line_length: int,
    image_height: int,
    image_width: int,
) -> tuple[list[dict[str, Any]], float]:
    candidates = [
        row
        for row in rows
        if len(row["normalized"]) >= min_line_length
        and row["likeness"] >= min_likeness
        and row["width_ratio"] >= 0.25
    ]
    candidates.sort(key=lambda row: row["y_center"])
    if len(candidates) < 2:
        return [], 0.0

    best_group: list[dict[str, Any]] = []
    best_score = -999.0
    for group_size in (2, 3):
        if len(candidates) < group_size:
            continue
        for start_idx in range(0, len(candidates) - group_size + 1):
            group = candidates[start_idx : start_idx + group_size]
            score = group_score(group, image_height, image_width)
            if score > best_score:
                best_group = group
                best_score = score

    return best_group, best_score


def padded_union_box(
    group: list[dict[str, Any]],
    *,
    image_height: int,
    image_width: int,
    padding_ratio: float = 0.012,
) -> list[float]:
    points = np.vstack([row["points"] for row in group])
    x_min = float(np.min(points[:, 0]))
    y_min = float(np.min(points[:, 1]))
    x_max = float(np.max(points[:, 0]))
    y_max = float(np.max(points[:, 1]))
    pad_x = image_width * padding_ratio
    pad_y = image_height * padding_ratio
    return [
        max(0.0, x_min - pad_x),
        max(0.0, y_min - pad_y),
        min(float(image_width - 1), x_max + pad_x),
        min(float(image_height - 1), y_max + pad_y),
    ]


def yolo_line_from_xyxy(box: list[float], image_width: int, image_height: int) -> str:
    x_min, y_min, x_max, y_max = box
    x_center = ((x_min + x_max) / 2) / image_width
    y_center = ((y_min + y_max) / 2) / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def stable_output_name(image_path: Path) -> str:
    digest = hashlib.sha1(str(image_path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{image_path.stem}_{digest}{image_path.suffix.lower()}"


def choose_split(rel_key: str, val_ratio: float, test_ratio: float) -> str:
    rng = random.Random(hashlib.sha1(rel_key.encode("utf-8")).hexdigest())
    value = rng.random()
    if test_ratio > 0 and value < test_ratio:
        return "test"
    if value < test_ratio + val_ratio:
        return "val"
    return "train"


def process_image(
    image_path: Path,
    *,
    source_dir: Path,
    output_dir: Path,
    ocr: Any,
    env: dict[str, str],
) -> dict[str, Any]:
    started = time.perf_counter()
    image = cv2.imread(str(image_path))
    if image is None:
        return {
            "status": "error",
            "error": "Cannot read image",
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }

    height, width = image.shape[:2]
    ocr_started = time.perf_counter()
    rows = ocr.predict_rows(image_path, image)
    ocr_ms = int((time.perf_counter() - ocr_started) * 1000)

    group, score = find_mrz_group(
        rows,
        min_likeness=env_float(env, "READMRZ_MIN_MRZ_LIKENESS", 0.48),
        min_line_length=env_int(env, "READMRZ_MIN_MRZ_LINE_LENGTH", 24),
        image_height=height,
        image_width=width,
    )

    rel_key = str(image_path.relative_to(source_dir)).replace("\\", "/")
    output_name = stable_output_name(image_path)
    if not group:
        review_dir = output_dir / "review" / "no_mrz"
        review_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, review_dir / output_name)
        return {
            "status": "no_mrz",
            "source": str(image_path),
            "review_image": str(review_dir / output_name),
            "ocr_ms": ocr_ms,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "ocr_rows": [
                {
                    "text": row["text"],
                    "normalized": row["normalized"],
                    "ocr_score": row["ocr_score"],
                    "likeness": row["likeness"],
                    "bbox_xyxy": row["bbox_xyxy"],
                }
                for row in rows
            ],
        }

    bbox = padded_union_box(group, image_height=height, image_width=width)
    split = choose_split(
        rel_key,
        env_float(env, "READMRZ_VAL_RATIO", 0.1),
        env_float(env, "READMRZ_TEST_RATIO", 0.0),
    )
    image_out_dir = output_dir / "images" / split
    label_out_dir = output_dir / "labels" / split
    image_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    output_image = image_out_dir / output_name
    output_label = label_out_dir / f"{Path(output_name).stem}.txt"
    shutil.copy2(image_path, output_image)
    output_label.write_text(yolo_line_from_xyxy(bbox, width, height) + "\n", encoding="utf-8")

    return {
        "status": "labeled",
        "source": str(image_path),
        "split": split,
        "output_image": str(output_image),
        "output_label": str(output_label),
        "bbox_xyxy": [round(value, 2) for value in bbox],
        "yolo_label": output_label.read_text(encoding="utf-8").strip(),
        "mrz_lines": [row["normalized"] for row in group],
        "mrz_score": round(score, 4),
        "doc_orientation_angle": int(group[0].get("doc_orientation_angle", 0)) if group else 0,
        "ocr_ms": ocr_ms,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


def write_data_yaml(output_dir: Path) -> None:
    data_yaml = "\n".join(
        [
            f"path: {output_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "",
            "names:",
            "  0: mrz",
            "",
        ]
    )
    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate YOLO MRZ pseudo-labels using RapidOCR/PaddleOCR text boxes."
    )
    parser.add_argument("--env", default=str(PROJECT_ROOT / ".env"), help="Path to .env config file.")
    parser.add_argument("--force", action="store_true", help="Reprocess images even if processed.json says they are done.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of images to process.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    env_path = Path(args.env)
    if not env_path.is_absolute():
        cwd_env_path = Path.cwd() / env_path
        project_env_path = PROJECT_ROOT / env_path
        env_path = cwd_env_path if cwd_env_path.exists() else project_env_path

    env = read_env_file(env_path)
    print(f"Using env file: {env_path}")
    reexec_code = maybe_reexec_with_paddle_python(env)
    if reexec_code is not None:
        return reexec_code

    source_dir = Path(env_value(env, "READMRZ_SOURCE_IMAGE_DIR")).expanduser()
    output_dir = Path(env_value(env, "READMRZ_YOLO_DATASET_DIR")).expanduser()
    if not source_dir.exists():
        raise FileNotFoundError(f"READMRZ_SOURCE_IMAGE_DIR does not exist: {source_dir}")
    if not str(output_dir):
        raise ValueError("READMRZ_YOLO_DATASET_DIR is required")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    processed_path = output_dir / "processed.json"
    run_items_path = output_dir / "last_run_annotations.jsonl"
    run_summary_path = output_dir / "last_run_summary.json"
    processed_index = {} if args.force else load_processed_index(processed_path)
    batch_size = max(1, env_int(env, "READMRZ_PROCESSED_BATCH_SIZE", 25))
    batch_items: dict[str, dict[str, Any]] = {}
    if run_items_path.exists():
        run_items_path.unlink()
    ocr = build_ocr(env)
    images = list_images(source_dir, output_dir=output_dir)
    if args.limit > 0:
        images = images[: args.limit]

    counts = {"labeled": 0, "no_mrz": 0, "error": 0, "skipped": 0}
    flushed_total = 0
    try:
        for index, image_path in enumerate(images, start=1):
            rel_key = str(image_path.relative_to(source_dir)).replace("\\", "/")
            fingerprint = file_fingerprint(image_path)
            previous = processed_index.get(rel_key)
            if (
                previous
                and not args.force
                and previous.get("fingerprint") == fingerprint
                and previous.get("status") in {"labeled", "no_mrz", "error"}
            ):
                counts["skipped"] += 1
                continue

            item = process_image(
                image_path,
                source_dir=source_dir,
                output_dir=output_dir,
                ocr=ocr,
                env=env,
            )
            item["fingerprint"] = fingerprint
            item["processed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            batch_items[rel_key] = item
            processed_index[rel_key] = {
                "fingerprint": fingerprint,
                "status": item.get("status", "error"),
            }
            status = item.get("status", "error")
            counts[status] = counts.get(status, 0) + 1
            print(
                f"[{index}/{len(images)}] {status} {rel_key} "
                f"ocr_ms={item.get('ocr_ms', '-')} elapsed_ms={item.get('elapsed_ms', '-')}"
            )

            if len(batch_items) >= batch_size:
                flushed = flush_processed_batch(
                    processed_path=processed_path,
                    run_items_path=run_items_path,
                    batch_items=batch_items,
                    ocr=ocr,
                )
                flushed_total += flushed
                print(f"Flushed {flushed} processed items to {processed_path}")
    finally:
        flushed = flush_processed_batch(
            processed_path=processed_path,
            run_items_path=run_items_path,
            batch_items=batch_items,
            ocr=ocr,
        )
        if flushed:
            flushed_total += flushed
            print(f"Flushed {flushed} processed items to {processed_path}")

    write_data_yaml(output_dir)
    summary = {
        "done": True,
        "counts": counts,
        "output_dir": str(output_dir),
        "processed_batch_size": batch_size,
        "flushed_items": flushed_total,
        "last_run_items_jsonl": str(run_items_path),
    }
    save_json(run_summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
