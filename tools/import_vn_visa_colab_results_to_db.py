from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mrz_reader.db import connect
from mrz_reader.env_config import env_value, read_env_file


RESULT_EXTENSIONS = {".txt", ".jsonl"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ALLOWED_DOCUMENT_TYPES = {"visa_sticker_vn", "loose_visa_vn"}
ALLOWED_STATUSES = {"pending", "ocr_done", "mapped", "exported", "skipped", "error"}
ALLOWED_REVIEW_STATUSES = {"pending", "needs_review", "approved", "rejected"}


@dataclass(frozen=True)
class Config:
    results_dir: Path
    oriented_image_dir: Path
    batch_size: int
    limit: int
    overwrite: bool
    reprocess_errors: bool
    document_type: str
    regex_version: str
    mapping_version: str


def env_int(env: dict[str, str], key: str, default: int) -> int:
    raw_value = env_value(env, key, "")
    return int(raw_value) if raw_value else default


def env_bool(env: dict[str, str], key: str, default: bool) -> bool:
    raw_value = env_value(env, key, "")
    if not raw_value:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_config(env: dict[str, str]) -> Config:
    import_root_raw = env_value(env, "READMRZ_VN_VISA_IMPORT_ROOT", "").strip()
    import_root = Path(import_root_raw).expanduser() if import_root_raw else None

    results_dir_raw = env_value(env, "READMRZ_VN_VISA_IMPORT_OCR_RESULTS_DIR", "").strip()
    if results_dir_raw:
        results_dir = Path(results_dir_raw).expanduser()
    elif import_root is not None:
        results_dir = import_root / "ocr_results"
    else:
        raise ValueError("Missing READMRZ_VN_VISA_IMPORT_ROOT or READMRZ_VN_VISA_IMPORT_OCR_RESULTS_DIR in .env")

    oriented_dir_raw = env_value(env, "READMRZ_VN_VISA_IMPORT_ORIENTED_IMAGE_DIR", "").strip()
    if oriented_dir_raw:
        oriented_image_dir = Path(oriented_dir_raw).expanduser()
    elif import_root is not None:
        oriented_image_dir = import_root / "oriented_images"
    else:
        oriented_image_dir = Path(env_value(env, "READMRZ_VN_VISA_OUTPUT_DIR", "")).expanduser()

    if not oriented_image_dir:
        raise ValueError("Missing READMRZ_VN_VISA_IMPORT_ORIENTED_IMAGE_DIR in .env")

    document_type = env_value(env, "READMRZ_VN_VISA_DOCUMENT_TYPE", "loose_visa_vn").strip()
    if document_type not in ALLOWED_DOCUMENT_TYPES:
        raise ValueError("READMRZ_VN_VISA_DOCUMENT_TYPE must be visa_sticker_vn or loose_visa_vn")

    return Config(
        results_dir=results_dir.resolve(),
        oriented_image_dir=oriented_image_dir.resolve(),
        batch_size=max(1, env_int(env, "READMRZ_VN_VISA_IMPORT_BATCH_SIZE", 200)),
        limit=max(0, env_int(env, "READMRZ_VN_VISA_IMPORT_LIMIT", 0)),
        overwrite=env_bool(env, "READMRZ_VN_VISA_IMPORT_OVERWRITE", False),
        reprocess_errors=env_bool(env, "READMRZ_VN_VISA_IMPORT_REPROCESS_ERRORS", False),
        document_type=document_type,
        regex_version=env_value(env, "READMRZ_VN_VISA_REGEX_VERSION", "vn_visa_regex_v1").strip(),
        mapping_version=env_value(
            env,
            "READMRZ_VN_VISA_MAPPING_VERSION",
            env_value(env, "READMRZ_VN_VISA_AUTO_MAPPING_VERSION", "ocr_bbox_auto_v1"),
        ).strip(),
    )


def list_result_files(results_dir: Path) -> list[Path]:
    if not results_dir.exists():
        raise FileNotFoundError(f"OCR results dir does not exist: {results_dir}")
    return sorted(
        path
        for path in results_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in RESULT_EXTENSIONS
        and path.name.lower() not in {"errors.txt", "progress.txt"}
    )


def iter_json_records(result_files: Iterable[Path]) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    for result_file in result_files:
        with result_file.open("r", encoding="utf-8-sig") as file_obj:
            for line_number, raw_line in enumerate(file_obj, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("{"):
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON at {result_file}:{line_number}: {exc}") from exc
                    if not isinstance(payload, dict):
                        raise ValueError(f"JSON record must be an object at {result_file}:{line_number}")
                    yield result_file, line_number, payload


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
        cursor.execute(sql, *record_values(record))


def update_records(cursor: Any, records: list[dict[str, Any]]) -> None:
    if not records:
        return

    sql = """
        UPDATE dbo.readmrz_vn_visa_items
        SET
            SourceTable = ?,
            SourceId = ?,
            TransactionEVisaId = ?,
            TransactionGuid = ?,
            DocumentType = ?,
            RelativeImagePath = ?,
            OrientedRelativeImagePath = ?,
            OriginalFileName = ?,
            ImageWidth = ?,
            ImageHeight = ?,
            FileSizeBytes = ?,
            Sha256 = ?,
            Split = ?,
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
            record["SourceTable"],
            record["SourceId"],
            record["TransactionEVisaId"],
            record["TransactionGuid"],
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
            record["SourceKey"],
        )


def record_values(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
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


def build_record(payload: dict[str, Any], config: Config) -> dict[str, Any]:
    source_key = clean_text(payload.get("source_key")) or clean_text(payload.get("relative_image_path"))
    if not source_key:
        raise ValueError("Missing source_key")

    oriented_relative_path = safe_relative_path(
        payload.get("oriented_relative_image_path") or payload.get("relative_image_path") or source_key
    )
    relative_image_path = safe_relative_path(
        payload.get("relative_image_path") or oriented_relative_path or source_key
    )
    # The review API currently loads READMRZ_VN_VISA_OUTPUT_DIR / RelativeImagePath.
    # Use the oriented image path here so review opens the normalized image from Colab.
    if oriented_relative_path:
        relative_image_path = oriented_relative_path

    image_path = resolve_oriented_image(config.oriented_image_dir, oriented_relative_path or relative_image_path)
    image_width = int_or_none(payload.get("image_width"))
    image_height = int_or_none(payload.get("image_height"))
    file_size_bytes = int_or_none(payload.get("file_size_bytes"))
    sha256 = clean_text(payload.get("sha256"))
    if image_path is not None:
        stat = image_path.stat()
        file_size_bytes = file_size_bytes or stat.st_size
        sha256 = sha256 or file_sha256(image_path)

    document_type = clean_text(payload.get("document_type")) or config.document_type
    if document_type not in ALLOWED_DOCUMENT_TYPES:
        document_type = config.document_type

    status = clean_text(payload.get("status")) or "mapped"
    if status not in ALLOWED_STATUSES:
        status = "mapped"

    review_status = clean_text(payload.get("review_status")) or "pending"
    if review_status not in ALLOWED_REVIEW_STATUSES:
        review_status = "needs_review"

    data = payload.get("data")
    if isinstance(data, dict) and data.get("missing_fields"):
        review_status = "needs_review" if review_status == "pending" else review_status

    return {
        "SourceTable": clean_text(payload.get("source_table")),
        "SourceId": int_or_none(payload.get("source_id")),
        "TransactionEVisaId": int_or_none(payload.get("transaction_evisa_id")),
        "TransactionGuid": clean_text(payload.get("transaction_guid")),
        "SourceKey": safe_relative_path(source_key),
        "DocumentType": document_type,
        "RelativeImagePath": relative_image_path,
        "OrientedRelativeImagePath": oriented_relative_path,
        "OriginalFileName": clean_text(payload.get("original_file_name")) or Path(source_key).name,
        "ImageWidth": image_width,
        "ImageHeight": image_height,
        "FileSizeBytes": file_size_bytes,
        "Sha256": sha256,
        "Split": clean_text(payload.get("split")),
        "Status": status,
        "ReviewStatus": review_status,
        "OcrEngine": clean_text(payload.get("ocr_engine")) or "paddleocr",
        "OcrConfigJson": json_dumps(payload.get("ocr_config") or {}),
        "OrientationJson": json_dumps(payload.get("orientation") or {}),
        "RawOcrJson": json_dumps(payload.get("raw_ocr") or {"rows": []}),
        "RegexVersion": clean_text(payload.get("regex_version")) or config.regex_version,
        "AutoMappingVersion": clean_text(payload.get("auto_mapping_version")) or config.mapping_version,
        "DataJson": json_dumps(data) if data is not None else None,
        "FinalDataJson": json_dumps(payload.get("final_data")) if payload.get("final_data") is not None else None,
        "ProcessStartedAt": clean_text(payload.get("process_started_at")),
        "ProcessedAt": clean_text(payload.get("processed_at")),
        "ErrorMessage": truncate(clean_text(payload.get("error_message")), 2000),
    }


def resolve_oriented_image(root: Path, relative_path: str | None) -> Path | None:
    if not relative_path:
        return None
    direct = root / relative_path
    if direct.is_file():
        return direct
    matches = list(root.rglob(Path(relative_path).name))
    return matches[0] if matches else None


def safe_relative_path(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    path = Path(text.replace("\\", "/"))
    first_part = path.parts[0] if path.parts else ""
    if path.is_absolute() or text.startswith("/") or text.startswith("\\") or ":" in first_part:
        return path.name
    clean_parts = [part for part in path.parts if part not in {"", ".", ".."}]
    return "/".join(clean_parts)


def int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def truncate(value: str | None, max_length: int) -> str | None:
    if value is None:
        return None
    return value[:max_length]


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def run(config: Config, *, dry_run: bool = False) -> int:
    result_files = list_result_files(config.results_dir)
    records_with_location = list(iter_json_records(result_files))
    if config.limit:
        records_with_location = records_with_location[: config.limit]

    print(f"Result files: {len(result_files)} under {config.results_dir}")
    print(f"JSON records: {len(records_with_location)}")
    print(f"Oriented image dir: {config.oriented_image_dir}")
    if dry_run:
        print("Dry run: no DB writes")

    prepared: list[dict[str, Any]] = []
    errors = 0
    for result_file, line_number, payload in records_with_location:
        try:
            prepared.append(build_record(payload, config))
        except Exception as exc:
            errors += 1
            print(f"[prepare error] {result_file}:{line_number} {exc}")

    inserted = 0
    updated = 0
    skipped = 0
    if dry_run:
        print(f"Prepared={len(prepared)} errors={errors}")
        return 0 if errors == 0 else 2

    with connect() as connection:
        cursor = connection.cursor()
        for batch_index, batch in enumerate(chunks(prepared, config.batch_size), start=1):
            source_keys = [record["SourceKey"] for record in batch]
            existing = fetch_existing(cursor, source_keys)
            insert_batch: list[dict[str, Any]] = []
            update_batch: list[dict[str, Any]] = []

            for record in batch:
                existing_status = existing.get(record["SourceKey"])
                if existing_status:
                    if config.overwrite or (config.reprocess_errors and existing_status == "error"):
                        update_batch.append(record)
                    else:
                        skipped += 1
                else:
                    insert_batch.append(record)

            insert_records(cursor, insert_batch)
            update_records(cursor, update_batch)
            connection.commit()

            inserted += len(insert_batch)
            updated += len(update_batch)
            print(
                f"[batch {batch_index}] inserted={len(insert_batch)} "
                f"updated={len(update_batch)} skipped_total={skipped}"
            )

    print(f"Done. inserted={inserted} updated={updated} skipped={skipped} errors={errors}")
    return 0 if errors == 0 else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import Colab Vietnam visa OCR JSONL results to SQL Server.")
    parser.add_argument("--limit", type=int, default=None, help="Override READMRZ_VN_VISA_IMPORT_LIMIT.")
    parser.add_argument("--overwrite", action="store_true", help="Update existing rows by SourceKey.")
    parser.add_argument("--reprocess-errors", action="store_true", help="Update existing rows only when current Status is error.")
    parser.add_argument("--dry-run", action="store_true", help="Parse files and prepare records without writing DB.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env = read_env_file()
    config = load_config(env)
    if args.limit is not None:
        config = Config(**{**config.__dict__, "limit": max(0, args.limit)})
    if args.overwrite:
        config = Config(**{**config.__dict__, "overwrite": True})
    if args.reprocess_errors:
        config = Config(**{**config.__dict__, "reprocess_errors": True})
    return run(config, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
