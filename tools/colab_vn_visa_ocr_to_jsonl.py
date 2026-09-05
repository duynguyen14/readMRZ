from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any
import zipfile

import cv2
import numpy as np


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


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_ocr_token(value: str) -> str:
    return normalize_space(value).upper().replace("Đ", "D")


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


def resize_to_max_side(image: np.ndarray, max_side: int) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = image.shape[:2]
    if max_side <= 0 or max(height, width) <= max_side:
        return image, {
            "enabled": False,
            "original_width": width,
            "original_height": height,
            "output_width": width,
            "output_height": height,
            "scale": 1.0,
        }
    scale = max_side / float(max(height, width))
    output_width = max(1, int(round(width * scale)))
    output_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (output_width, output_height), interpolation=cv2.INTER_AREA)
    return resized, {
        "enabled": True,
        "original_width": width,
        "original_height": height,
        "output_width": output_width,
        "output_height": output_height,
        "scale": round(scale, 8),
    }


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


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


def union_bbox(rows: list[dict[str, Any]]) -> list[float] | None:
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
    scores = [float(row.get("score") or 0.0) for row in rows]
    confidence = sum(scores) / max(1, len(scores))
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


def find_text_near_keywords(rows: list[dict[str, Any]], keywords: tuple[str, ...], *, method: str) -> dict[str, Any] | None:
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
                return make_field(raw_text=text, normalized_value=text, rows=[candidate], method=method)
    return None


def find_date_near_keywords(rows: list[dict[str, Any]], keywords: tuple[str, ...], *, method: str) -> dict[str, Any] | None:
    return find_pattern_near_keywords(rows, keywords, DATE_PATTERN, normalizer=normalize_date, method=method)


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
    lines = [re.sub(r"[^A-Z0-9<]", "", row["text"].upper().replace(" ", "")) for row in mrz_rows]
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
            return make_field(raw_text=row["text"], normalized_value="multiple", rows=[row], method="entries_keyword")
        if any(value in token for value in ("SINGLE", "MOT LAN", "MỘT LẦN")):
            return make_field(raw_text=row["text"], normalized_value="single", rows=[row], method="entries_keyword")
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


def normalize_angle(value: Any) -> int:
    try:
        angle = int(round(float(value))) % 360
    except (TypeError, ValueError):
        return 0
    return angle if angle in {0, 90, 180, 270} else 0


def rotate_counter_clockwise(image: np.ndarray, angle: int) -> np.ndarray:
    angle = normalize_angle(angle)
    if angle == 0:
        return image
    return np.ascontiguousarray(np.rot90(image, angle // 90))


class PaddleDocumentOrientation:
    def __init__(self, *, device: str, model_dir: str | None, min_confidence: float) -> None:
        from paddleocr import DocImgOrientationClassification

        kwargs: dict[str, Any] = {"device": device}
        if model_dir:
            kwargs["model_dir"] = model_dir
        self.model = DocImgOrientationClassification(**kwargs)
        self.min_confidence = min_confidence
        self.device = device
        self.model_dir = model_dir

    def normalize(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        started = time.perf_counter()
        results = self.model.predict(image)
        latency_ms = int((time.perf_counter() - started) * 1000)

        predicted_angle = 0
        confidence = 0.0
        if results:
            result = results[0]
            labels = result.get("label_names", [])
            scores = np.asarray(result.get("scores", []), dtype=np.float32).reshape(-1)
            if labels:
                predicted_angle = normalize_angle(labels[0])
            if scores.size:
                confidence = float(scores[0])

        applied_angle = predicted_angle if confidence >= self.min_confidence else 0
        normalized = rotate_counter_clockwise(image, applied_angle)
        height, width = normalized.shape[:2]
        return normalized, {
            "enabled": True,
            "predicted_angle": predicted_angle,
            "applied_angle": applied_angle,
            "confidence": round(confidence, 6),
            "min_confidence": self.min_confidence,
            "latency_ms": latency_ms,
            "device": self.device,
            "model_dir": self.model_dir,
            "image_width": width,
            "image_height": height,
        }


class PaddleVisaOcr:
    name = "paddleocr"

    def __init__(
        self,
        *,
        device: str,
        text_det_model_dir: str | None,
        text_rec_model_dir: str | None,
        textline_orientation_model_dir: str | None,
        text_det_limit_side_len: int,
        use_textline_orientation: bool,
    ) -> None:
        from paddleocr import PaddleOCR

        kwargs: dict[str, Any] = {
            "lang": "en",
            "ocr_version": "PP-OCRv5",
            "device": device,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": use_textline_orientation,
            "return_word_box": True,
            "text_det_limit_side_len": text_det_limit_side_len,
        }
        if text_det_model_dir:
            kwargs["text_detection_model_dir"] = text_det_model_dir
        if text_rec_model_dir:
            kwargs["text_recognition_model_dir"] = text_rec_model_dir
        if textline_orientation_model_dir:
            kwargs["textline_orientation_model_dir"] = textline_orientation_model_dir

        self.kwargs = kwargs
        print("Loading PaddleOCR:")
        for key, value in kwargs.items():
            if "dir" in key or key in {"device", "text_det_limit_side_len"}:
                print(f"  {key}={value}")
        self.pipeline = PaddleOCR(**kwargs)

    def config_summary(self) -> dict[str, Any]:
        return dict(self.kwargs)

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


def infer_document_type(path: Path, fallback: str) -> str:
    text = normalize_ocr_token(" ".join(path.parts))
    if "DAN" in text or "DÁN" in text:
        return "visa_sticker_vn"
    if "ROI" in text or "RỜI" in text:
        return "loose_visa_vn"
    return fallback


def safe_extract_zip(zip_path: Path, extract_root: Path) -> Path:
    target_dir = extract_root / zip_path.stem
    marker_path = target_dir / ".extracted.ok"
    if marker_path.exists():
        print(f"Skip extracted zip: {zip_path.name}")
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} -> {target_dir}")
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"Unsafe zip member path: {member.filename}")
            archive.extract(member, target_dir)
    marker_path.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    return target_dir


def list_images(input_dir: Path) -> list[Path]:
    ignored_dir_names = {"__MACOSX", ".ipynb_checkpoints", ".git", "__pycache__"}
    images: list[Path] = []
    for path in input_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if set(path.relative_to(input_dir).parts) & ignored_dir_names:
            continue
        images.append(path)
    return sorted(images)


def relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def build_flat_name_map(images: list[Path], input_root: Path) -> dict[Path, str]:
    name_counts: dict[str, int] = {}
    for image_path in images:
        key = image_path.name.lower()
        name_counts[key] = name_counts.get(key, 0) + 1

    flat_names: dict[Path, str] = {}
    for image_path in images:
        if name_counts[image_path.name.lower()] == 1:
            flat_names[image_path] = image_path.name
            continue

        relative_key = relative_posix(image_path, input_root)
        digest = hashlib.sha1(relative_key.encode("utf-8")).hexdigest()[:8]
        flat_names[image_path] = f"{image_path.stem}__{digest}{image_path.suffix.lower()}"
    return flat_names


def load_progress(progress_path: Path) -> set[str]:
    if not progress_path.exists():
        return set()
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    processed = payload.get("processed", [])
    return set(str(item) for item in processed)


def save_progress(progress_path: Path, processed: set[str], *, total: int) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": total,
        "processed_count": len(processed),
        "processed": sorted(processed),
    }
    progress_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def part_file_for_index(results_dir: Path, processed_index: int, chunk_size: int) -> Path:
    part_index = processed_index // chunk_size + 1
    return results_dir / f"vn_visa_ocr_part_{part_index:05d}.txt"


def append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json_dumps(record) + "\n")
        file_obj.flush()
        os.fsync(file_obj.fileno())


def process_one(
    *,
    image_path: Path,
    input_root: Path,
    oriented_root: Path,
    flat_file_name: str,
    fallback_document_type: str,
    max_image_side: int,
    min_ocr_score: float,
    regex_version: str,
    mapping_version: str,
    orientation: PaddleDocumentOrientation,
    ocr: PaddleVisaOcr,
) -> dict[str, Any]:
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    image_started = time.perf_counter()
    relative_image_path = flat_file_name
    output_path = oriented_root / flat_file_name

    image_width: int | None = None
    image_height: int | None = None
    orientation_payload: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    data: dict[str, Any] | None = None
    status = "error"
    review_status = "needs_review"
    error_message: str | None = None

    try:
        image = imread(image_path)
        if image is None:
            raise RuntimeError("Cannot read image")

        image, resize_payload = resize_to_max_side(image, max_image_side)
        normalized_image, orientation_payload = orientation.normalize(image)
        orientation_payload["pre_ocr_resize"] = resize_payload
        image_height, image_width = normalized_image.shape[:2]

        imwrite(output_path, normalized_image)
        rows = ocr.predict_rows(output_path, normalized_image, min_score=min_ocr_score)
        data = map_visa_fields(rows)
        data["source"] = {
            "relative_image_path": relative_image_path,
            "oriented_relative_image_path": flat_file_name,
        }
        data["ocr_row_count"] = len(rows)
        status = "mapped"
        review_status = "needs_review" if data.get("missing_fields") else "pending"
    except Exception as exc:
        error_message = str(exc)[:2000] or "Unknown exception"

    elapsed_ms = int((time.perf_counter() - image_started) * 1000)
    if data is not None:
        data["elapsed_ms"] = elapsed_ms
    if orientation_payload is not None:
        orientation_payload["total_image_elapsed_ms"] = elapsed_ms

    stat = image_path.stat()
    return {
        "source_key": relative_image_path,
        "source_table": None,
        "source_id": None,
        "transaction_evisa_id": None,
        "transaction_guid": None,
        "document_type": infer_document_type(image_path.relative_to(input_root), fallback_document_type),
        "relative_image_path": relative_image_path,
        "oriented_relative_image_path": flat_file_name if output_path.exists() else None,
        "original_file_name": image_path.name,
        "image_width": image_width,
        "image_height": image_height,
        "file_size_bytes": stat.st_size,
        "sha256": file_sha256(image_path),
        "split": None,
        "status": status,
        "review_status": review_status,
        "ocr_engine": ocr.name,
        "ocr_config": ocr.config_summary(),
        "orientation": orientation_payload,
        "raw_ocr": {"rows": rows},
        "regex_version": regex_version,
        "auto_mapping_version": mapping_version,
        "data": data,
        "final_data": None,
        "process_started_at": started_at,
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "error_message": error_message,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Vietnam visa OCR on Colab and write chunked JSONL text files.")
    parser.add_argument("--drive-root", default="/content/drive/MyDrive/thi_thuc_roi")
    parser.add_argument("--zip", dest="zip_paths", action="append", default=[])
    parser.add_argument("--extract-dir", default="extracted")
    parser.add_argument("--results-dir", default="ocr_results")
    parser.add_argument("--oriented-dir", default="oriented_images")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--document-type", default="visa_sticker_vn", choices=["visa_sticker_vn", "loose_visa_vn"])
    parser.add_argument("--max-image-side", type=int, default=1600)
    parser.add_argument("--text-det-limit-side-len", type=int, default=1280)
    parser.add_argument("--min-ocr-score", type=float, default=0.20)
    parser.add_argument("--orientation-min-confidence", type=float, default=0.50)
    parser.add_argument("--doc-orientation-model-dir", default="")
    parser.add_argument("--text-det-model-dir", default="")
    parser.add_argument("--text-rec-model-dir", default="")
    parser.add_argument("--textline-orientation-model-dir", default="")
    parser.add_argument("--no-textline-orientation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-unzip", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    drive_root = Path(args.drive_root)
    extract_root = drive_root / args.extract_dir
    results_dir = drive_root / args.results_dir
    oriented_root = drive_root / args.oriented_dir
    progress_path = results_dir / "progress.json"
    errors_path = results_dir / "errors.txt"

    zip_paths = [Path(path) if Path(path).is_absolute() else drive_root / path for path in args.zip_paths]
    if not zip_paths:
        zip_paths = [
            drive_root / "Thi_Thuc_Roi.zip",
            drive_root / "Thi_thuc_dan.zip",
        ]

    if not args.skip_unzip:
        for zip_path in zip_paths:
            if zip_path.exists():
                safe_extract_zip(zip_path, extract_root)
            else:
                print(f"Missing zip, skip: {zip_path}")

    images = list_images(extract_root)
    if args.limit:
        images = images[: args.limit]
    flat_names = build_flat_name_map(images, extract_root)
    print(f"Found {len(images)} images under {extract_root}")

    processed = set() if args.overwrite else load_progress(progress_path)
    print(f"Already processed: {len(processed)}")

    orientation = PaddleDocumentOrientation(
        device=args.device,
        model_dir=args.doc_orientation_model_dir or None,
        min_confidence=args.orientation_min_confidence,
    )
    ocr = PaddleVisaOcr(
        device=args.device,
        text_det_model_dir=args.text_det_model_dir or None,
        text_rec_model_dir=args.text_rec_model_dir or None,
        textline_orientation_model_dir=args.textline_orientation_model_dir or None,
        text_det_limit_side_len=args.text_det_limit_side_len,
        use_textline_orientation=not args.no_textline_orientation,
    )

    processed_this_run = 0
    errors = 0
    for image_path in images:
        source_key = flat_names[image_path]
        if source_key in processed:
            continue

        output_index = len(processed)
        part_path = part_file_for_index(results_dir, output_index, max(1, args.chunk_size))
        print(f"[{output_index + 1}/{len(images)}] OCR {relative_posix(image_path, extract_root)} -> {source_key}")
        record = process_one(
            image_path=image_path,
            input_root=extract_root,
            oriented_root=oriented_root,
            flat_file_name=source_key,
            fallback_document_type=args.document_type,
            max_image_side=args.max_image_side,
            min_ocr_score=args.min_ocr_score,
            regex_version=DEFAULT_REGEX_VERSION,
            mapping_version=DEFAULT_MAPPING_VERSION,
            orientation=orientation,
            ocr=ocr,
        )
        append_record(part_path, record)
        if record["status"] == "error":
            errors += 1
            append_record(errors_path, record)
            print(f"  error: {record['error_message']}")

        processed.add(source_key)
        processed_this_run += 1
        save_progress(progress_path, processed, total=len(images))

    print(f"Done. processed_this_run={processed_this_run} total_processed={len(processed)} errors={errors}")
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
