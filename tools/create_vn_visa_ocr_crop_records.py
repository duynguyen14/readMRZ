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
from typing import Any, Iterable

import cv2
import numpy as np


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


@dataclass(frozen=True)
class Config:
    source_image_dir: Path
    crop_output_dir: Path
    fields: list[str]
    model_name: str
    model_dir: Path | None
    device: str
    batch_size: int
    limit: int
    padding: int
    image_extension: str
    overwrite: bool
    overwrite_reviewed: bool
    dry_run: bool


@dataclass(frozen=True)
class CropRecord:
    visa_item_id: int
    source_key: str
    field_name: str
    original_image_relative_path: str
    original_file_name: str
    image_width: int
    image_height: int
    bbox: list[float]
    crop_relative_path: str
    crop_width: int
    crop_height: int
    crop_path: Path


def env_int(env: dict[str, str], key: str, default: int) -> int:
    raw_value = env_value(env, key, "")
    return int(raw_value) if raw_value else default


def env_bool(env: dict[str, str], key: str, default: bool) -> bool:
    raw_value = env_value(env, key, "")
    if not raw_value:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def csv_values(text: str) -> list[str]:
    values = [value.strip() for value in text.split(",")]
    return [value for value in values if value]


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


def load_config(env: dict[str, str], args: argparse.Namespace) -> Config:
    source_dir_raw = (
        args.source_image_dir
        or env_value(env, "READMRZ_VN_VISA_CROP_OCR_SOURCE_IMAGE_DIR", "")
        or env_value(env, "READMRZ_VN_VISA_IMPORT_ORIENTED_IMAGE_DIR", "")
        or env_value(env, "READMRZ_VN_VISA_OUTPUT_DIR", "")
    )
    if not source_dir_raw:
        raise ValueError(
            "Missing READMRZ_VN_VISA_CROP_OCR_SOURCE_IMAGE_DIR, "
            "READMRZ_VN_VISA_IMPORT_ORIENTED_IMAGE_DIR, or READMRZ_VN_VISA_OUTPUT_DIR"
        )
    source_image_dir = Path(source_dir_raw).expanduser().resolve()
    if not source_image_dir.exists():
        raise FileNotFoundError(f"Source image dir does not exist: {source_image_dir}")

    crop_output_dir_raw = (
        args.crop_output_dir
        or env_value(env, "READMRZ_VN_VISA_CROP_OCR_OUTPUT_DIR", "")
        or str(PROJECT_ROOT / "generated_datasets" / "vn_visa_ocr_crops")
    )

    model_dir_raw = args.model_dir or env_value(env, "PADDLE_VN_VISA_CROP_REC_MODEL_DIR", "")
    model_dir = Path(model_dir_raw).expanduser().resolve() if model_dir_raw else None
    if model_dir is not None and not model_dir.exists():
        raise FileNotFoundError(f"PADDLE_VN_VISA_CROP_REC_MODEL_DIR does not exist: {model_dir}")

    model_name = (
        args.model_name
        or env_value(env, "PADDLE_VN_VISA_CROP_REC_MODEL_NAME", "")
        or (read_local_model_name(model_dir) if model_dir else "")
        or "latin_PP-OCRv5_mobile_rec"
    )

    fields_raw = (
        args.fields
        or env_value(env, "READMRZ_VN_VISA_CROP_OCR_FIELDS", "")
        or env_value(env, "READMRZ_VN_VISA_YOLO_FIELDS", "")
    )
    fields = csv_values(fields_raw) if fields_raw else list(DEFAULT_FIELDS)

    image_extension = (
        args.image_extension
        or env_value(env, "READMRZ_VN_VISA_CROP_OCR_IMAGE_EXTENSION", ".jpg")
    ).strip().lower()
    if not image_extension.startswith("."):
        image_extension = f".{image_extension}"

    return Config(
        source_image_dir=source_image_dir,
        crop_output_dir=Path(crop_output_dir_raw).expanduser().resolve(),
        fields=fields,
        model_name=model_name,
        model_dir=model_dir,
        device=args.device or env_value(env, "PADDLE_VN_VISA_CROP_OCR_DEVICE", "cpu").strip(),
        batch_size=max(1, args.batch_size if args.batch_size is not None else env_int(env, "READMRZ_VN_VISA_CROP_OCR_BATCH_SIZE", 16)),
        limit=max(0, args.limit if args.limit is not None else env_int(env, "READMRZ_VN_VISA_CROP_OCR_LIMIT", 0)),
        padding=max(0, args.padding if args.padding is not None else env_int(env, "READMRZ_VN_VISA_CROP_OCR_PADDING", 2)),
        image_extension=image_extension,
        overwrite=args.overwrite or env_bool(env, "READMRZ_VN_VISA_CROP_OCR_OVERWRITE", False),
        overwrite_reviewed=args.overwrite_reviewed or env_bool(env, "READMRZ_VN_VISA_CROP_OCR_OVERWRITE_REVIEWED", False),
        dry_run=args.dry_run or env_bool(env, "READMRZ_VN_VISA_CROP_OCR_DRY_RUN", False),
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


class PaddleCropRecognizer:
    def __init__(self, config: Config, env: dict[str, str]) -> None:
        from paddleocr import TextRecognition

        os.environ["PADDLE_PDX_MODEL_SOURCE"] = env_value(env, "PADDLE_PDX_MODEL_SOURCE", "BOS")
        kwargs: dict[str, Any] = {
            "model_name": config.model_name,
            "device": config.device,
        }
        if config.model_dir is not None:
            kwargs["model_dir"] = str(config.model_dir)

        print("Loading Paddle text recognition model:")
        for key, value in kwargs.items():
            print(f"  {key}={value}")
        self.model = TextRecognition(**kwargs)
        self.engine = "paddleocr_text_recognition"
        self.model_name = config.model_name
        self.model_dir = str(config.model_dir) if config.model_dir else None

    def predict(self, crop_paths: list[Path], batch_size: int) -> list[dict[str, Any]]:
        if not crop_paths:
            return []
        output = self.model.predict(input=[str(path) for path in crop_paths], batch_size=batch_size)
        return [parse_recognition_result(item) for item in output]


def parse_recognition_result(item: Any) -> dict[str, Any]:
    payload = result_to_plain_data(item)
    text = (
        payload.get("rec_text")
        or payload.get("text")
        or payload.get("label")
        or ""
    )
    score = (
        payload.get("rec_score")
        or payload.get("score")
        or payload.get("confidence")
        or 0.0
    )
    try:
        score_float = float(score)
    except (TypeError, ValueError):
        score_float = 0.0
    return {
        "text": normalize_space(str(text or "")),
        "score": round(score_float, 6),
        "raw": payload,
    }


def result_to_plain_data(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return json_safe(item)
    if hasattr(item, "json"):
        try:
            raw_json = item.json() if callable(item.json) else item.json
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                return json_safe(parsed)
        except Exception:
            pass
    if hasattr(item, "to_dict"):
        try:
            parsed = item.to_dict()
            if isinstance(parsed, dict):
                return json_safe(parsed)
        except Exception:
            pass
    if hasattr(item, "__dict__"):
        return json_safe(vars(item))
    return {"repr": repr(item)}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def fetch_approved_rows(limit: int) -> list[dict[str, Any]]:
    query = """
        SELECT
            Id,
            SourceKey,
            RelativeImagePath,
            OrientedRelativeImagePath,
            OriginalFileName,
            ImageWidth,
            ImageHeight,
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
        payload = raw_value
    else:
        try:
            payload = json.loads(str(raw_value))
        except json.JSONDecodeError:
            payload = {}
    return payload if isinstance(payload, dict) else {}


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


def source_relative_path(row: dict[str, Any], image_path: Path, source_dir: Path) -> str:
    for key in ("OrientedRelativeImagePath", "RelativeImagePath", "SourceKey"):
        relative_path = str(row.get(key) or "").strip()
        if relative_path and (source_dir / relative_path).is_file():
            return safe_relative_path(relative_path)
    try:
        return image_path.resolve().relative_to(source_dir.resolve()).as_posix()
    except ValueError:
        return image_path.name


def bbox_xyxy(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return [x1, y1, x2, y2]


def clamp_bbox(box: list[float], image_width: int, image_height: int, padding: int) -> list[int] | None:
    x1, y1, x2, y2 = box
    left = max(0, int(np.floor(min(x1, x2))) - padding)
    top = max(0, int(np.floor(min(y1, y2))) - padding)
    right = min(image_width, int(np.ceil(max(x1, x2))) + padding)
    bottom = min(image_height, int(np.ceil(max(y1, y2))) + padding)
    if right - left < 2 or bottom - top < 2:
        return None
    return [left, top, right, bottom]


def safe_relative_path(value: Any) -> str:
    text = normalize_space(str(value or "").replace("\\", "/"))
    if not text:
        return ""
    path = Path(text)
    first_part = path.parts[0] if path.parts else ""
    if path.is_absolute() or text.startswith("/") or text.startswith("\\") or ":" in first_part:
        return path.name
    clean_parts = [part for part in path.parts if part not in {"", ".", ".."}]
    return "/".join(clean_parts)


def safe_stem(value: str) -> str:
    stem = Path(value).stem or "image"
    safe_value = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return safe_value[:96] or "image"


def crop_relative_name(row: dict[str, Any], field_name: str, config: Config) -> str:
    source_name = str(row.get("OriginalFileName") or Path(str(row.get("SourceKey") or "")).name or row["Id"])
    source_hash = hashlib.sha1(str(row.get("SourceKey") or row["Id"]).encode("utf-8")).hexdigest()[:10]
    file_name = f"{int(row['Id']):08d}_{safe_stem(source_name)}_{source_hash}{config.image_extension}"
    return f"{field_name}/{file_name}"


def build_crop_records(config: Config) -> tuple[list[CropRecord], dict[str, int]]:
    rows = fetch_approved_rows(config.limit)
    allowed_fields = set(config.fields)
    counts = {
        "db_rows": len(rows),
        "missing_images": 0,
        "bad_images": 0,
        "missing_bboxes": 0,
        "invalid_bboxes": 0,
        "crop_records": 0,
    }
    records: list[CropRecord] = []

    for row in rows:
        image_path = resolve_image_path(config.source_image_dir, row)
        if image_path is None:
            counts["missing_images"] += 1
            print(f"[missing-image] id={row.get('Id')} key={row.get('SourceKey')}")
            continue
        image = imread(image_path)
        if image is None:
            counts["bad_images"] += 1
            print(f"[bad-image] id={row.get('Id')} image={image_path}")
            continue

        image_height, image_width = image.shape[:2]
        data_json = parse_data_json(row)
        fields = data_json.get("fields") or {}
        if not isinstance(fields, dict):
            continue
        original_relative_path = source_relative_path(row, image_path, config.source_image_dir)

        for field_name, field_payload in fields.items():
            field_name = str(field_name)
            if field_name not in allowed_fields or not isinstance(field_payload, dict):
                continue
            raw_bbox = bbox_xyxy(field_payload.get("bbox"))
            if raw_bbox is None:
                counts["missing_bboxes"] += 1
                continue
            bbox = clamp_bbox(raw_bbox, image_width, image_height, config.padding)
            if bbox is None:
                counts["invalid_bboxes"] += 1
                continue

            left, top, right, bottom = bbox
            crop = image[top:bottom, left:right]
            if crop.size == 0:
                counts["invalid_bboxes"] += 1
                continue

            crop_relative_path = crop_relative_name(row, field_name, config)
            crop_path = config.crop_output_dir / crop_relative_path
            imwrite(crop_path, crop)

            records.append(
                CropRecord(
                    visa_item_id=int(row["Id"]),
                    source_key=str(row.get("SourceKey") or ""),
                    field_name=field_name,
                    original_image_relative_path=original_relative_path,
                    original_file_name=str(row.get("OriginalFileName") or image_path.name),
                    image_width=image_width,
                    image_height=image_height,
                    bbox=[float(left), float(top), float(right), float(bottom)],
                    crop_relative_path=crop_relative_path,
                    crop_width=int(right - left),
                    crop_height=int(bottom - top),
                    crop_path=crop_path,
                )
            )
            counts["crop_records"] += 1

    return records, counts


def existing_crop(cursor: Any, record: CropRecord) -> tuple[int, str] | None:
    cursor.execute(
        """
        SELECT Id, ReviewStatus
        FROM dbo.readmrz_vn_visa_ocr_crops
        WHERE VisaItemId = ? AND FieldName = ?
        """,
        record.visa_item_id,
        record.field_name,
    )
    row = cursor.fetchone()
    if not row:
        return None
    return int(row[0]), str(row[1])


def insert_crop(cursor: Any, record: CropRecord, recognition: dict[str, Any], recognizer: PaddleCropRecognizer) -> None:
    cursor.execute(
        """
        INSERT INTO dbo.readmrz_vn_visa_ocr_crops (
            VisaItemId,
            SourceKey,
            FieldName,
            OriginalImageRelativePath,
            CropRelativePath,
            OriginalFileName,
            ImageWidth,
            ImageHeight,
            CropWidth,
            CropHeight,
            BboxJson,
            OcrEngine,
            OcrModel,
            OcrRawText,
            OcrScore,
            OcrRawJson,
            ReviewStatus,
            ErrorMessage
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            CAST(? AS NVARCHAR(MAX)),
            ?, ?, ?, ?,
            CAST(? AS NVARCHAR(MAX)),
            N'pending',
            ?
        )
        """,
        record.visa_item_id,
        record.source_key,
        record.field_name,
        record.original_image_relative_path,
        record.crop_relative_path,
        record.original_file_name,
        record.image_width,
        record.image_height,
        record.crop_width,
        record.crop_height,
        json_dumps({"xyxy": record.bbox}),
        recognizer.engine,
        recognizer.model_name,
        recognition.get("text") or None,
        recognition.get("score"),
        json_dumps(recognition.get("raw") or {}),
        None,
    )


def update_crop(cursor: Any, crop_id: int, record: CropRecord, recognition: dict[str, Any], recognizer: PaddleCropRecognizer) -> None:
    cursor.execute(
        """
        UPDATE dbo.readmrz_vn_visa_ocr_crops
        SET
            SourceKey = ?,
            OriginalImageRelativePath = ?,
            CropRelativePath = ?,
            OriginalFileName = ?,
            ImageWidth = ?,
            ImageHeight = ?,
            CropWidth = ?,
            CropHeight = ?,
            BboxJson = CAST(? AS NVARCHAR(MAX)),
            OcrEngine = ?,
            OcrModel = ?,
            OcrRawText = ?,
            OcrScore = ?,
            OcrRawJson = CAST(? AS NVARCHAR(MAX)),
            ErrorMessage = ?,
            UpdatedDate = SYSDATETIME()
        WHERE Id = ?
        """,
        record.source_key,
        record.original_image_relative_path,
        record.crop_relative_path,
        record.original_file_name,
        record.image_width,
        record.image_height,
        record.crop_width,
        record.crop_height,
        json_dumps({"xyxy": record.bbox}),
        recognizer.engine,
        recognizer.model_name,
        recognition.get("text") or None,
        recognition.get("score"),
        json_dumps(recognition.get("raw") or {}),
        None,
        crop_id,
    )


def chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def run(config: Config, env: dict[str, str]) -> dict[str, Any]:
    print(f"Source images: {config.source_image_dir}")
    print(f"Crop output: {config.crop_output_dir}")
    print(f"Fields: {', '.join(config.fields)}")
    print(f"Model: {config.model_name} dir={config.model_dir or '(auto-download/cache)'} device={config.device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Dry run: {'yes' if config.dry_run else 'no'}")

    records, crop_counts = build_crop_records(config)
    summary: dict[str, Any] = {
        **crop_counts,
        "inserted": 0,
        "updated": 0,
        "skipped_existing": 0,
        "skipped_reviewed": 0,
        "ocr_errors": 0,
        "dry_run": config.dry_run,
    }
    if not records:
        return summary

    recognizer = PaddleCropRecognizer(config, env)

    with connect() as connection:
        cursor = connection.cursor()
        for batch_index, batch in enumerate(chunks(records, config.batch_size), start=1):
            try:
                predictions = recognizer.predict([record.crop_path for record in batch], config.batch_size)
            except Exception as exc:
                summary["ocr_errors"] += len(batch)
                print(f"[batch {batch_index}] OCR error: {exc}")
                continue

            for record, recognition in zip(batch, predictions):
                existing = existing_crop(cursor, record)
                if existing is None:
                    if not config.dry_run:
                        insert_crop(cursor, record, recognition, recognizer)
                    summary["inserted"] += 1
                    action = "insert"
                else:
                    crop_id, review_status = existing
                    if not config.overwrite:
                        summary["skipped_existing"] += 1
                        continue
                    if review_status != "pending" and not config.overwrite_reviewed:
                        summary["skipped_reviewed"] += 1
                        continue
                    if not config.dry_run:
                        update_crop(cursor, crop_id, record, recognition, recognizer)
                    summary["updated"] += 1
                    action = "update"

                print(
                    f"[{action}] item={record.visa_item_id} field={record.field_name} "
                    f"score={recognition.get('score')} text={recognition.get('text')!r}"
                )

            if not config.dry_run:
                connection.commit()
            print(
                f"[batch {batch_index}] inserted={summary['inserted']} updated={summary['updated']} "
                f"skip_existing={summary['skipped_existing']} skip_reviewed={summary['skipped_reviewed']}"
            )

        if not config.dry_run:
            connection.commit()

    return summary


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crop approved VN visa field bboxes, run Paddle Latin text recognition, and save OCR crop rows to DB."
    )
    parser.add_argument("--env", default=str(PROJECT_ROOT / ".env"), help="Path to .env config file.")
    parser.add_argument("--source-image-dir", default="", help="Override oriented source image directory.")
    parser.add_argument("--crop-output-dir", default="", help="Override crop output directory.")
    parser.add_argument("--fields", default="", help="Comma-separated field names to crop.")
    parser.add_argument("--model-name", default="", help="Paddle TextRecognition model name.")
    parser.add_argument("--model-dir", default="", help="Local Paddle TextRecognition inference model directory.")
    parser.add_argument("--device", default="", help="Paddle device, for example cpu or gpu:0.")
    parser.add_argument("--batch-size", type=int, default=None, help="Recognition batch size.")
    parser.add_argument("--limit", type=int, default=None, help="Limit approved visa rows.")
    parser.add_argument("--padding", type=int, default=None, help="Crop padding in pixels.")
    parser.add_argument("--image-extension", default="", help="Crop image extension, default .jpg.")
    parser.add_argument("--overwrite", action="store_true", help="Update existing crop rows.")
    parser.add_argument("--overwrite-reviewed", action="store_true", help="Also update approved/rejected crop rows.")
    parser.add_argument("--dry-run", action="store_true", help="Create crop files and run OCR, but do not write DB.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    env_path = Path(args.env)
    if not env_path.is_absolute():
        env_path = (Path.cwd() / env_path).resolve()
        if not env_path.exists():
            env_path = PROJECT_ROOT / args.env

    env = read_env_file(env_path)
    reexec_code = maybe_reexec_with_paddle_python(env)
    if reexec_code is not None:
        return reexec_code

    config = load_config(env, args)
    summary = run(config, env)
    print("DONE")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
