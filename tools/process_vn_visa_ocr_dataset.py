from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterable

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mrz_reader.db import connect
from mrz_reader.document_orientation import PaddleDocumentOrientation
from mrz_reader.env_config import env_value, read_env_file


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_REGEX_VERSION = "vn_visa_regex_v1"
DEFAULT_MAPPING_VERSION = "ocr_bbox_auto_v1"

VISA_CODE_PATTERN = re.compile(
    r"\b("
    r"DL|EV|DN1|DN2|LD1|LD2|DT1|DT2|DT3|DT4|D[TĐ]1|D[TĐ]2|D[TĐ]3|D[TĐ]4|"
    r"TT|VR|NG1|NG2|NG3|NG4|LV1|LV2|LS|NN1|NN2|NN3|DH|HN|PV1|PV2|SQ"
    r")\b",
    re.IGNORECASE,
)
DATE_PATTERN = re.compile(r"\b(\d{1,2})[./\-\s](\d{1,2})[./\-\s](\d{2,4})\b")
PASSPORT_PATTERN = re.compile(r"\b([A-Z][A-Z0-9]{5,12}|\d{7,12})\b", re.IGNORECASE)
VISA_NUMBER_PATTERN = re.compile(r"\b([A-Z]{1,4}\d{5,12}|\d{6,12})\b", re.IGNORECASE)


@dataclass(frozen=True)
class Config:
    input_dir: Path
    output_dir: Path
    batch_size: int
    document_type: str
    limit: int
    overwrite: bool
    reprocess_errors: bool
    min_ocr_score: float
    regex_version: str
    mapping_version: str
    extensions: set[str]


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


def load_config(env: dict[str, str]) -> Config:
    raw_input_dir = env_value(env, "READMRZ_VN_VISA_INPUT_DIR", "").strip()
    raw_output_dir = env_value(env, "READMRZ_VN_VISA_OUTPUT_DIR", "").strip()
    if not raw_input_dir:
        raise ValueError("Missing READMRZ_VN_VISA_INPUT_DIR in .env")
    if not raw_output_dir:
        raise ValueError("Missing READMRZ_VN_VISA_OUTPUT_DIR in .env")
    input_dir = Path(raw_input_dir).expanduser()
    output_dir = Path(raw_output_dir).expanduser()

    configured_extensions = env_value(env, "READMRZ_VN_VISA_EXTENSIONS", "")
    extensions = {
        item.strip().lower() if item.strip().startswith(".") else f".{item.strip().lower()}"
        for item in configured_extensions.split(",")
        if item.strip()
    } or IMAGE_EXTENSIONS

    document_type = env_value(env, "READMRZ_VN_VISA_DOCUMENT_TYPE", "visa_sticker_vn").strip()
    if document_type not in {"visa_sticker_vn", "loose_visa_vn"}:
        raise ValueError("READMRZ_VN_VISA_DOCUMENT_TYPE must be visa_sticker_vn or loose_visa_vn")

    return Config(
        input_dir=input_dir.resolve(),
        output_dir=output_dir.resolve(),
        batch_size=max(1, env_int(env, "READMRZ_VN_VISA_BATCH_SIZE", 1)),
        document_type=document_type,
        limit=max(0, env_int(env, "READMRZ_VN_VISA_LIMIT", 0)),
        overwrite=env_bool(env, "READMRZ_VN_VISA_OVERWRITE", False),
        reprocess_errors=env_bool(env, "READMRZ_VN_VISA_REPROCESS_ERRORS", False),
        min_ocr_score=env_float(env, "READMRZ_VN_VISA_MIN_OCR_SCORE", 0.20),
        regex_version=env_value(env, "READMRZ_VN_VISA_REGEX_VERSION", DEFAULT_REGEX_VERSION),
        mapping_version=env_value(env, "READMRZ_VN_VISA_AUTO_MAPPING_VERSION", DEFAULT_MAPPING_VERSION),
        extensions=extensions,
    )


def maybe_reexec_with_paddle_python(env: dict[str, str]) -> int | None:
    if importlib.util.find_spec("paddleocr") is not None:
        return None

    configured_python = env_value(env, "READMRZ_PADDLE_PYTHON", "")
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


class PaddleVisaOcr:
    name = "paddleocr"

    def __init__(self, env: dict[str, str]) -> None:
        from paddleocr import PaddleOCR

        os.environ["PADDLE_PDX_MODEL_SOURCE"] = env_value(env, "PADDLE_PDX_MODEL_SOURCE", "BOS")
        self.kwargs = self._build_kwargs(env)
        print("Loading PaddleOCR for Vietnam visa OCR:")
        for key, value in self.kwargs.items():
            if "dir" in key or "name" in key or key in {"lang", "ocr_version", "device"}:
                print(f"  {key}={value}")
        self.pipeline = PaddleOCR(**self.kwargs)

    def _build_kwargs(self, env: dict[str, str]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "lang": env_value(env, "OCR_LANGUAGE", "en"),
            "ocr_version": env_value(env, "PADDLE_OCR_VERSION", "PP-OCRv5"),
            "device": env_value(env, "PADDLE_OCR_DEVICE", "cpu"),
            "use_doc_orientation_classify": env_bool(
                env,
                "READMRZ_VN_VISA_PADDLE_DOC_ORIENTATION_IN_OCR",
                False,
            ),
            "use_doc_unwarping": env_bool(env, "READMRZ_VN_VISA_PADDLE_DOC_UNWARPING", False),
            "use_textline_orientation": env_bool(env, "PADDLE_USE_TEXTLINE_ORIENTATION", True),
            "return_word_box": True,
            "det_limit_side_len": env_int(env, "PADDLE_DET_LIMIT_SIDE_LEN", 1280),
        }

        model_dir_map = {
            "PADDLE_DOC_ORIENTATION_MODEL_DIR": "doc_orientation_classify_model",
            "PADDLE_TEXT_DETECTION_MODEL_DIR": "text_detection_model",
            "PADDLE_TEXT_RECOGNITION_MODEL_DIR": "text_recognition_model",
            "PADDLE_TEXTLINE_ORIENTATION_MODEL_DIR": "textline_orientation_model",
        }
        for env_key, kwarg_prefix in model_dir_map.items():
            raw_model_dir = env_value(env, env_key, "")
            if not raw_model_dir:
                continue
            model_dir = Path(raw_model_dir).expanduser().resolve()
            model_name = read_local_model_name(model_dir)
            if model_name:
                kwargs[f"{kwarg_prefix}_name"] = model_name
            kwargs[f"{kwarg_prefix}_dir"] = str(model_dir)

        return kwargs

    def config_summary(self) -> dict[str, Any]:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in self.kwargs.items()
        }

    def predict_rows(self, image_path: Path, image: np.ndarray, *, min_score: float) -> list[dict[str, Any]]:
        height, width = image.shape[:2]
        result = self.pipeline.predict(str(image_path))
        if not result:
            return []

        item = result[0]
        rows: list[dict[str, Any]] = []
        for text, score, raw_polygon, raw_box in zip(
            item.get("rec_texts", []),
            item.get("rec_scores", []),
            item.get("rec_polys", []),
            item.get("rec_boxes", []),
            strict=False,
        ):
            row = build_ocr_row(text, score, raw_polygon, raw_box, width, height)
            if row is not None and float(row["score"]) >= min_score:
                rows.append(row)

        rows.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
        return rows


def imread(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".jpg"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def list_images(input_dir: Path, extensions: set[str]) -> list[Path]:
    ignored_dir_names = {".git", ".idea", ".venv", "__pycache__", "build", "dist"}
    images: list[Path] = []
    for path in input_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if set(path.relative_to(input_dir).parts) & ignored_dir_names:
            continue
        images.append(path)
    return sorted(images)


def chunks(items: list[Path], batch_size: int) -> Iterable[list[Path]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def normalize_points(value: Any) -> np.ndarray | None:
    try:
        points = np.asarray(value, dtype=np.float32)
    except Exception:
        return None
    if points.ndim != 2 or points.shape[1] != 2 or points.size == 0:
        return None
    return points


def build_ocr_row(
    text: Any,
    score: Any,
    raw_polygon: Any,
    raw_box: Any,
    image_width: int,
    image_height: int,
) -> dict[str, Any] | None:
    normalized_text = normalize_space(str(text or ""))
    if not normalized_text:
        return None

    points = normalize_points(raw_polygon)
    if points is None and raw_box is not None:
        box = np.asarray(raw_box, dtype=np.float32).reshape(-1)
        if box.size >= 4:
            x1, y1, x2, y2 = box[:4]
            points = np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    if points is None:
        return None

    points[:, 0] = np.clip(points[:, 0], 0, max(0, image_width - 1))
    points[:, 1] = np.clip(points[:, 1], 0, max(0, image_height - 1))
    x1 = float(np.min(points[:, 0]))
    y1 = float(np.min(points[:, 1]))
    x2 = float(np.max(points[:, 0]))
    y2 = float(np.max(points[:, 1]))

    return {
        "text": normalized_text,
        "score": round(float(score or 0.0), 6),
        "bbox": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
        "polygon": [[round(float(x), 3), round(float(y), 3)] for x, y in points.tolist()],
    }


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_ocr_token(value: str) -> str:
    value = normalize_space(value).upper()
    value = value.replace("Đ", "D")
    return value


def normalize_date(value: str) -> str:
    match = DATE_PATTERN.search(value)
    if not match:
        return normalize_space(value)

    day = int(match.group(1))
    month = int(match.group(2))
    year = int(match.group(3))
    if year < 100:
        year += 2000 if year < 50 else 1900
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return normalize_space(value)
    return f"{year:04d}-{month:02d}-{day:02d}"


def union_bbox(rows: list[dict[str, Any]]) -> list[float] | None:
    if not rows:
        return None
    boxes = [row["bbox"] for row in rows if row.get("bbox")]
    if not boxes:
        return None
    return [
        round(min(box[0] for box in boxes), 3),
        round(min(box[1] for box in boxes), 3),
        round(max(box[2] for box in boxes), 3),
        round(max(box[3] for box in boxes), 3),
    ]


def make_field(
    *,
    raw_text: str,
    normalized_value: str,
    rows: list[dict[str, Any]],
    method: str,
) -> dict[str, Any]:
    confidence_values = [float(row.get("score") or 0.0) for row in rows]
    confidence = sum(confidence_values) / max(1, len(confidence_values))
    return {
        "raw_text": normalize_space(raw_text),
        "normalized_value": normalize_space(normalized_value),
        "bbox": union_bbox(rows),
        "bbox_source": "auto",
        "confidence": round(confidence, 6),
        "method": method,
    }


def row_contains_any(row: dict[str, Any], keywords: tuple[str, ...]) -> bool:
    text = normalize_ocr_token(row.get("text", ""))
    return any(keyword in text for keyword in keywords)


def right_or_next_value_rows(
    rows: list[dict[str, Any]],
    label_index: int,
    *,
    max_rows: int = 4,
) -> list[dict[str, Any]]:
    label_row = rows[label_index]
    label_box = label_row["bbox"]
    candidates: list[tuple[float, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        if index == label_index:
            continue
        box = row["bbox"]
        same_line = abs(((box[1] + box[3]) / 2) - ((label_box[1] + label_box[3]) / 2)) <= max(
            12,
            (label_box[3] - label_box[1]) * 1.5,
        )
        right_side = box[0] >= label_box[0] - 5
        near_below = 0 <= box[1] - label_box[1] <= max(80, (label_box[3] - label_box[1]) * 4)
        if same_line and right_side:
            candidates.append((box[0] - label_box[0], row))
        elif near_below:
            candidates.append((1000 + box[1] - label_box[1] + abs(box[0] - label_box[0]) / 10, row))

    candidates.sort(key=lambda item: item[0])
    return [row for _, row in candidates[:max_rows]]


def extract_after_label(text: str) -> str:
    if ":" in text:
        return normalize_space(text.split(":", 1)[1])
    parts = re.split(r"\b(?:NO|NUMBER|SỐ|SO|DATE|NGAY|NGÀY)\b", text, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) > 1:
        return normalize_space(parts[-1])
    return ""


def find_pattern_near_keywords(
    rows: list[dict[str, Any]],
    keywords: tuple[str, ...],
    pattern: re.Pattern[str],
    *,
    normalizer: Any = normalize_space,
    method: str,
) -> dict[str, Any] | None:
    for index, row in enumerate(rows):
        if not row_contains_any(row, keywords):
            continue

        candidate_rows = [row, *right_or_next_value_rows(rows, index)]
        for candidate in candidate_rows:
            match = pattern.search(candidate["text"])
            if match:
                raw_value = match.group(0)
                return make_field(
                    raw_text=raw_value,
                    normalized_value=normalizer(raw_value),
                    rows=[candidate],
                    method=method,
                )

        value_after_label = extract_after_label(row["text"])
        if value_after_label:
            match = pattern.search(value_after_label)
            if match:
                raw_value = match.group(0)
                return make_field(
                    raw_text=raw_value,
                    normalized_value=normalizer(raw_value),
                    rows=[row],
                    method=method,
                )
    return None


def find_text_near_keywords(
    rows: list[dict[str, Any]],
    keywords: tuple[str, ...],
    *,
    method: str,
) -> dict[str, Any] | None:
    for index, row in enumerate(rows):
        if not row_contains_any(row, keywords):
            continue

        value_after_label = extract_after_label(row["text"])
        if value_after_label and len(value_after_label) >= 2:
            return make_field(
                raw_text=value_after_label,
                normalized_value=value_after_label,
                rows=[row],
                method=method,
            )

        for candidate in right_or_next_value_rows(rows, index):
            text = candidate["text"]
            if len(text) >= 2 and not row_contains_any(candidate, keywords):
                return make_field(
                    raw_text=text,
                    normalized_value=text,
                    rows=[candidate],
                    method=method,
                )
    return None


def find_date_near_keywords(
    rows: list[dict[str, Any]],
    keywords: tuple[str, ...],
    *,
    method: str,
) -> dict[str, Any] | None:
    return find_pattern_near_keywords(
        rows,
        keywords,
        DATE_PATTERN,
        normalizer=normalize_date,
        method=method,
    )


def extract_mrz(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    mrz_rows = [
        row
        for row in rows
        if "<" in row.get("text", "")
        and len(re.sub(r"[^A-Z0-9<]", "", row.get("text", "").upper())) >= 20
    ]
    if not mrz_rows:
        return None

    mrz_rows.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    lines = [
        re.sub(r"[^A-Z0-9<]", "", row["text"].upper().replace(" ", ""))
        for row in mrz_rows
    ]
    value = "\n".join(line for line in lines if line)
    return make_field(
        raw_text="\n".join(row["text"] for row in mrz_rows),
        normalized_value=value,
        rows=mrz_rows,
        method="mrz_line_group",
    )


def extract_entries(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        token = normalize_ocr_token(row["text"])
        if any(value in token for value in ("MULTIPLE", "NHIEU LAN", "NHIỀU LẦN")):
            return make_field(
                raw_text=row["text"],
                normalized_value="multiple",
                rows=[row],
                method="entries_keyword",
            )
        if any(value in token for value in ("SINGLE", "MOT LAN", "MỘT LẦN")):
            return make_field(
                raw_text=row["text"],
                normalized_value="single",
                rows=[row],
                method="entries_keyword",
            )
    return None


def extract_dates_without_labels(rows: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    dates: list[tuple[str, dict[str, Any]]] = []
    for row in rows:
        for match in DATE_PATTERN.finditer(row["text"]):
            dates.append((match.group(0), row))
    dates.sort(key=lambda item: (item[1]["bbox"][1], item[1]["bbox"][0]))
    return dates


def map_visa_fields(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields: dict[str, Any] = {}

    keyword_specs = {
        "visa_number": (("VISA NO", "VISA NUMBER", "SO THI THUC", "SỐ THỊ THỰC", "NO."), VISA_NUMBER_PATTERN),
        "passport_number": (("PASSPORT", "HO CHIEU", "HỘ CHIẾU"), PASSPORT_PATTERN),
        "visa_code": (("CATEGORY", "LOAI", "LOẠI", "KY HIEU", "KÝ HIỆU"), VISA_CODE_PATTERN),
    }
    for field_name, (keywords, pattern) in keyword_specs.items():
        field = find_pattern_near_keywords(
            rows,
            keywords,
            pattern,
            normalizer=lambda value: normalize_ocr_token(value).replace("Đ", "D"),
            method=f"{field_name}_near_label",
        )
        if field:
            fields[field_name] = field

    if "visa_code" not in fields:
        for row in rows:
            match = VISA_CODE_PATTERN.search(normalize_ocr_token(row["text"]))
            if match:
                fields["visa_code"] = make_field(
                    raw_text=match.group(0),
                    normalized_value=match.group(0).upper().replace("Đ", "D"),
                    rows=[row],
                    method="visa_code_standalone",
                )
                break

    date_specs = {
        "valid_from": ("VALID FROM", "FROM", "GIA TRI TU", "GIÁ TRỊ TỪ", "TU NGAY", "TỪ NGÀY"),
        "valid_until": ("VALID UNTIL", "UNTIL", "EXPIRY", "EXPIRE", "GIA TRI DEN", "GIÁ TRỊ ĐẾN", "DEN NGAY", "ĐẾN NGÀY"),
        "issue_date": ("ISSUED", "ISSUE DATE", "NGAY CAP", "NGÀY CẤP"),
        "date_of_birth": ("DATE OF BIRTH", "DOB", "BIRTH", "NGAY SINH", "NGÀY SINH"),
    }
    for field_name, keywords in date_specs.items():
        field = find_date_near_keywords(rows, keywords, method=f"{field_name}_near_label")
        if field:
            fields[field_name] = field

    if "valid_from" not in fields or "valid_until" not in fields:
        dates = extract_dates_without_labels(rows)
        if "valid_from" not in fields and len(dates) >= 1:
            raw_value, row = dates[0]
            fields["valid_from"] = make_field(
                raw_text=raw_value,
                normalized_value=normalize_date(raw_value),
                rows=[row],
                method="date_order_fallback",
            )
        if "valid_until" not in fields and len(dates) >= 2:
            raw_value, row = dates[1]
            fields["valid_until"] = make_field(
                raw_text=raw_value,
                normalized_value=normalize_date(raw_value),
                rows=[row],
                method="date_order_fallback",
            )

    optional_text_specs = {
        "full_name": ("FULL NAME", "NAME", "HO TEN", "HỌ TÊN"),
        "nationality": ("NATIONALITY", "QUOC TICH", "QUỐC TỊCH"),
        "sex": ("SEX", "GENDER", "GIOI TINH", "GIỚI TÍNH"),
        "issue_place": ("PLACE OF ISSUE", "ISSUE PLACE", "NOI CAP", "NƠI CẤP"),
        "issuing_authority": ("ISSUING AUTHORITY", "AUTHORITY", "CO QUAN CAP", "CƠ QUAN CẤP"),
        "remarks": ("REMARK", "REMARKS", "GHI CHU", "GHI CHÚ"),
    }
    for field_name, keywords in optional_text_specs.items():
        field = find_text_near_keywords(rows, keywords, method=f"{field_name}_near_label")
        if field:
            fields[field_name] = field

    entries = extract_entries(rows)
    if entries:
        fields["number_of_entries"] = entries

    mrz_zone = extract_mrz(rows)
    if mrz_zone:
        fields["mrz_zone"] = mrz_zone

    return {
        "schema_version": 1,
        "fields": fields,
        "missing_fields": [
            field_name
            for field_name in (
                "visa_number",
                "visa_code",
                "valid_from",
                "valid_until",
                "number_of_entries",
                "passport_number",
                "mrz_zone",
            )
            if field_name not in fields
        ],
    }


def fetch_existing(cursor: Any, source_keys: list[str]) -> dict[str, str]:
    if not source_keys:
        return {}

    existing: dict[str, str] = {}
    for offset in range(0, len(source_keys), 500):
        batch = source_keys[offset : offset + 500]
        placeholders = ",".join("?" for _ in batch)
        cursor.execute(
            f"""
            SELECT SourceKey, Status
            FROM dbo.readmrz_vn_visa_items
            WHERE SourceKey IN ({placeholders})
            """,
            batch,
        )
        for source_key, status in cursor.fetchall():
            existing[str(source_key)] = str(status)
    return existing


def insert_records(cursor: Any, records: list[dict[str, Any]]) -> None:
    if not records:
        return

    sql = """
        INSERT INTO dbo.readmrz_vn_visa_items (
            SourceTable,
            SourceId,
            TransactionEVisaId,
            TransactionGuid,
            SourceKey,
            DocumentType,
            RelativeImagePath,
            OrientedRelativeImagePath,
            OriginalFileName,
            ImageWidth,
            ImageHeight,
            FileSizeBytes,
            Sha256,
            Split,
            Status,
            ReviewStatus,
            OcrEngine,
            OcrConfigJson,
            OrientationJson,
            RawOcrJson,
            RegexVersion,
            AutoMappingVersion,
            DataJson,
            FinalDataJson,
            ProcessStartedAt,
            ProcessedAt,
            ErrorMessage
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            CAST(? AS NVARCHAR(MAX)),
            CAST(? AS NVARCHAR(MAX)),
            CAST(? AS NVARCHAR(MAX)),
            ?, ?,
            CAST(? AS NVARCHAR(MAX)),
            CAST(? AS NVARCHAR(MAX)),
            ?, ?, ?
        )
    """
    for record in records:
        cursor.execute(
            sql,
            record["SourceTable"],
            record["SourceId"],
            record["TransactionEVisaId"],
            record["TransactionGuid"],
            record["SourceKey"],
            record["DocumentType"],
            record["RelativeImagePath"],
            record["OrientedRelativeImagePath"],
            record["OriginalFileName"],
            record["ImageWidth"],
            record["ImageHeight"],
            record["FileSizeBytes"],
            record["Sha256"],
            record["Split"],
            record["Status"],
            record["ReviewStatus"],
            record["OcrEngine"],
            record["OcrConfigJson"],
            record["OrientationJson"],
            record["RawOcrJson"],
            record["RegexVersion"],
            record["AutoMappingVersion"],
            record["DataJson"],
            record["FinalDataJson"],
            record["ProcessStartedAt"],
            record["ProcessedAt"],
            record["ErrorMessage"],
        )


def update_records(cursor: Any, records: list[dict[str, Any]]) -> None:
    if not records:
        return

    sql = """
        UPDATE dbo.readmrz_vn_visa_items
        SET
            DocumentType = ?,
            RelativeImagePath = ?,
            OrientedRelativeImagePath = ?,
            OriginalFileName = ?,
            ImageWidth = ?,
            ImageHeight = ?,
            FileSizeBytes = ?,
            Sha256 = ?,
            Status = ?,
            ReviewStatus = ?,
            OcrEngine = ?,
            OcrConfigJson = CAST(? AS NVARCHAR(MAX)),
            OrientationJson = CAST(? AS NVARCHAR(MAX)),
            RawOcrJson = CAST(? AS NVARCHAR(MAX)),
            RegexVersion = ?,
            AutoMappingVersion = ?,
            DataJson = CAST(? AS NVARCHAR(MAX)),
            FinalDataJson = CAST(? AS NVARCHAR(MAX)),
            ProcessStartedAt = ?,
            ProcessedAt = ?,
            ErrorMessage = ?,
            UpdatedDate = SYSDATETIME()
        WHERE SourceKey = ?
    """
    for record in records:
        cursor.execute(
            sql,
            record["DocumentType"],
            record["RelativeImagePath"],
            record["OrientedRelativeImagePath"],
            record["OriginalFileName"],
            record["ImageWidth"],
            record["ImageHeight"],
            record["FileSizeBytes"],
            record["Sha256"],
            record["Status"],
            record["ReviewStatus"],
            record["OcrEngine"],
            record["OcrConfigJson"],
            record["OrientationJson"],
            record["RawOcrJson"],
            record["RegexVersion"],
            record["AutoMappingVersion"],
            record["DataJson"],
            record["FinalDataJson"],
            record["ProcessStartedAt"],
            record["ProcessedAt"],
            record["ErrorMessage"],
            record["SourceKey"],
        )


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def process_one(
    *,
    image_path: Path,
    relative_image_path: str,
    config: Config,
    orientation: PaddleDocumentOrientation,
    ocr: PaddleVisaOcr,
) -> dict[str, Any]:
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    processed_at: str | None = None
    error_message: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    oriented_relative_path: str | None = None
    orientation_payload: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    data_json: dict[str, Any] | None = None
    status = "error"
    review_status = "needs_review"

    try:
        image = imread(image_path)
        if image is None:
            raise RuntimeError("Cannot read image")

        normalized_image, orientation_payload = orientation.normalize(image)
        image_height, image_width = normalized_image.shape[:2]

        output_path = config.output_dir / Path(relative_image_path)
        imwrite(output_path, normalized_image)
        oriented_relative_path = relative_posix(output_path, config.output_dir)

        rows = ocr.predict_rows(output_path, normalized_image, min_score=config.min_ocr_score)
        data_json = map_visa_fields(rows)
        data_json["source"] = {
            "relative_image_path": relative_image_path,
            "oriented_relative_image_path": oriented_relative_path,
        }
        data_json["ocr_row_count"] = len(rows)

        status = "mapped"
        review_status = "needs_review" if data_json.get("missing_fields") else "pending"
        processed_at = time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as exc:
        error_message = str(exc)[:2000]
        processed_at = time.strftime("%Y-%m-%d %H:%M:%S")

    stat = image_path.stat()
    return {
        "SourceTable": None,
        "SourceId": None,
        "TransactionEVisaId": None,
        "TransactionGuid": None,
        "SourceKey": relative_image_path,
        "DocumentType": config.document_type,
        "RelativeImagePath": relative_image_path,
        "OrientedRelativeImagePath": oriented_relative_path,
        "OriginalFileName": image_path.name,
        "ImageWidth": image_width,
        "ImageHeight": image_height,
        "FileSizeBytes": stat.st_size,
        "Sha256": file_sha256(image_path),
        "Split": None,
        "Status": status,
        "ReviewStatus": review_status,
        "OcrEngine": ocr.name,
        "OcrConfigJson": json_dumps(ocr.config_summary()),
        "OrientationJson": json_dumps(orientation_payload) if orientation_payload is not None else None,
        "RawOcrJson": json_dumps({"rows": rows}),
        "RegexVersion": config.regex_version,
        "AutoMappingVersion": config.mapping_version,
        "DataJson": json_dumps(data_json) if data_json is not None else None,
        "FinalDataJson": None,
        "ProcessStartedAt": started_at,
        "ProcessedAt": processed_at,
        "ErrorMessage": error_message,
    }


def run(config: Config, env: dict[str, str]) -> int:
    image_paths = list_images(config.input_dir, config.extensions)
    if config.limit:
        image_paths = image_paths[: config.limit]
    print(f"Found {len(image_paths)} images under {config.input_dir}")
    print(f"Batch size: {config.batch_size}")
    print(
        "Target DB: "
        f"{env_value(env, 'READMRZ_DB_SERVER', '')}/"
        f"{env_value(env, 'READMRZ_DB_DATABASE', '')}"
    )

    orientation = PaddleDocumentOrientation()
    if orientation.enabled:
        print("Loaded Paddle document orientation classifier")
    else:
        print("Paddle document orientation disabled")

    ocr = PaddleVisaOcr(env)

    processed_count = 0
    skipped_count = 0
    error_count = 0

    with connect() as connection:
        cursor = connection.cursor()
        for batch_index, path_batch in enumerate(chunks(image_paths, config.batch_size), start=1):
            source_keys = [relative_posix(path, config.input_dir) for path in path_batch]
            existing = fetch_existing(cursor, source_keys)

            insert_batch: list[dict[str, Any]] = []
            update_batch: list[dict[str, Any]] = []
            for image_path, source_key in zip(path_batch, source_keys, strict=False):
                existing_status = existing.get(source_key)
                if existing_status and not config.overwrite:
                    if not (config.reprocess_errors and existing_status == "error"):
                        skipped_count += 1
                        continue

                print(f"[batch {batch_index}] OCR {source_key}")
                record = process_one(
                    image_path=image_path,
                    relative_image_path=source_key,
                    config=config,
                    orientation=orientation,
                    ocr=ocr,
                )
                if record["Status"] == "error":
                    error_count += 1
                    print(f"  error: {record['ErrorMessage']}")

                if existing_status:
                    update_batch.append(record)
                else:
                    insert_batch.append(record)

            insert_records(cursor, insert_batch)
            update_records(cursor, update_batch)
            connection.commit()

            flushed = len(insert_batch) + len(update_batch)
            processed_count += flushed
            print(
                f"[batch {batch_index}] committed={flushed} "
                f"processed={processed_count} skipped={skipped_count} errors={error_count}"
            )

    print(f"Done. processed={processed_count} skipped={skipped_count} errors={error_count}")
    return 0 if error_count == 0 else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OCR Vietnam visa images, regex-map fields with OCR bboxes, and save JSON records to SQL Server."
    )
    parser.add_argument("--limit", type=int, default=None, help="Override READMRZ_VN_VISA_LIMIT.")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess existing SourceKey records.")
    parser.add_argument("--reprocess-errors", action="store_true", help="Reprocess existing records whose Status is error.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env = read_env_file()
    reexec_result = maybe_reexec_with_paddle_python(env)
    if reexec_result is not None:
        return reexec_result

    config = load_config(env)
    if args.limit is not None:
        config = Config(**{**config.__dict__, "limit": max(0, args.limit)})
    if args.overwrite:
        config = Config(**{**config.__dict__, "overwrite": True})
    if args.reprocess_errors:
        config = Config(**{**config.__dict__, "reprocess_errors": True})

    return run(config, env)


if __name__ == "__main__":
    raise SystemExit(main())
