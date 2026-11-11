from __future__ import annotations

import argparse
from datetime import date, datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mrz_reader.custom_mrz_ocr import CustomMrzCtcRecognizer  # noqa: E402
from mrz_reader.db import connect, execute_sql_file  # noqa: E402
from mrz_reader.document_orientation import PaddleDocumentOrientation  # noqa: E402
from mrz_reader.env_config import env_value, read_env_file  # noqa: E402
from mrz_reader.passport_classification import (  # noqa: E402
    VALIDITY_UNKNOWN,
    classify_passport_dates,
    country_match,
)
from mrz_reader.passport_evaluation import (  # noqa: E402
    EvaluationCandidate,
    discover_passport_images,
    infer_export_reference_date,
    sha256_file,
)
from mrz_reader.yolo_detector import YoloMrzDetector  # noqa: E402
from mrz_reader.yolo_upload_pipeline import (  # noqa: E402
    compact_yolo_read_payload,
    process_yolo_upload,
)


SCHEMA_PATH = PROJECT_ROOT / "sql" / "create_passport_evaluation_tables.sql"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate exported passport folders with the production MRZ pipeline and store "
            "per-file results in SQL Server."
        )
    )
    parser.add_argument(
        "source_root",
        nargs="?",
        help=(
            "Export root containing COUNTRY/GROUP/passport. If omitted, uses "
            "READMRZ_PASSPORT_EVALUATION_ROOT."
        ),
    )
    parser.add_argument("--env", default=str(PROJECT_ROOT / ".env"), help="Path to .env file.")
    parser.add_argument("--run-name", default="", help="Optional report run name.")
    parser.add_argument(
        "--reference-date",
        default="",
        help=(
            "Classification date in YYYY-MM-DD. Default: infer from quick_YYYYMMDD_HHMMSS, "
            "otherwise today."
        ),
    )
    parser.add_argument(
        "--country",
        action="append",
        default=[],
        help="Only process this country code. May be repeated or comma-separated.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Maximum files. 0 means all files.")
    parser.add_argument("--resume-run-id", type=int, default=0, help="Resume queued items in an existing run.")
    parser.add_argument("--retry-failed", action="store_true", help="Queue failed items again when resuming.")
    parser.add_argument("--skip-schema", action="store_true", help="Do not create/check SQL schema.")
    parser.add_argument("--dry-run", action="store_true", help="Scan only. Do not load models or write SQL.")
    parser.add_argument("--no-warmup", action="store_true", help="Skip model warmup.")
    parser.add_argument("--progress-every", type=int, default=10, help="Print progress every N files.")
    return parser


def parse_reference_date(value: str, source_root: Path) -> tuple[date, str]:
    if not value.strip():
        inferred = infer_export_reference_date(source_root)
        if inferred is not None:
            return inferred, "export_folder"
        return date.today(), "current_date"
    try:
        return date.fromisoformat(value.strip()), "command_line"
    except ValueError as exc:
        raise ValueError("--reference-date must use YYYY-MM-DD") from exc


def country_filters(values: list[str]) -> list[str]:
    return sorted(
        {
            item.strip().upper()
            for value in values
            for item in value.split(",")
            if item.strip()
        }
    )


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def row_to_dict(cursor: Any, row: Any) -> dict[str, Any]:
    return dict(zip((str(column[0]) for column in cursor.description), row))


def create_run(
    cursor: Any,
    *,
    run_name: str,
    source_root: Path,
    reference_date: date,
    candidates: list[EvaluationCandidate],
    config: dict[str, Any],
) -> tuple[int, str]:
    cursor.execute(
        """
        INSERT INTO dbo.readmrz_passport_evaluation_runs (
            RunName, SourceRoot, ReferenceDate, Status, TotalItems, ConfigJson,
            StartedAt, CreatedDate, UpdatedDate
        )
        OUTPUT INSERTED.Id, INSERTED.RunKey
        VALUES (?, ?, ?, N'created', ?, ?, NULL, SYSUTCDATETIME(), SYSUTCDATETIME());
        """,
        run_name or None,
        str(source_root),
        reference_date,
        len(candidates),
        json_text(config),
    )
    run_id, run_key = cursor.fetchone()

    rows = [
        (
            int(run_id),
            candidate.sequence_no,
            candidate.country_code,
            candidate.source_group,
            candidate.expected_validity_status,
            candidate.relative_image_path,
            candidate.relative_path_hash,
            candidate.image_path.name,
            candidate.file_size_bytes,
        )
        for candidate in candidates
    ]
    if rows:
        cursor.fast_executemany = True
        cursor.executemany(
            """
            INSERT INTO dbo.readmrz_passport_evaluation_items (
                RunId, SequenceNo, CountryCode, SourceGroup, ExpectedValidityStatus,
                RelativeImagePath, RelativePathHash, OriginalFileName, FileSizeBytes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            rows,
        )
        cursor.fast_executemany = False
    return int(run_id), str(run_key)


def load_run(cursor: Any, run_id: int) -> dict[str, Any]:
    cursor.execute(
        """
        SELECT Id, RunKey, RunName, SourceRoot, ReferenceDate, Status, TotalItems
        FROM dbo.readmrz_passport_evaluation_runs
        WHERE Id = ?;
        """,
        run_id,
    )
    row = cursor.fetchone()
    if row is None:
        raise LookupError(f"Evaluation run not found: {run_id}")
    return row_to_dict(cursor, row)


def queue_failed_items(cursor: Any, run_id: int) -> int:
    cursor.execute(
        """
        UPDATE dbo.readmrz_passport_evaluation_items
        SET Status = N'queued', ReadSuccess = NULL, ErrorCode = NULL, ErrorMessage = NULL,
            ProcessedAt = NULL, UpdatedDate = SYSUTCDATETIME()
        WHERE RunId = ? AND Status = N'failed';
        """,
        run_id,
    )
    return max(0, int(cursor.rowcount))


def queue_interrupted_items(cursor: Any, run_id: int) -> int:
    cursor.execute(
        """
        UPDATE dbo.readmrz_passport_evaluation_items
        SET Status = N'queued', UpdatedDate = SYSUTCDATETIME()
        WHERE RunId = ? AND Status = N'processing';
        """,
        run_id,
    )
    return max(0, int(cursor.rowcount))


def pending_items(cursor: Any, run_id: int) -> list[dict[str, Any]]:
    cursor.execute(
        """
        SELECT Id, SequenceNo, CountryCode, SourceGroup, ExpectedValidityStatus,
               RelativeImagePath, OriginalFileName
        FROM dbo.readmrz_passport_evaluation_items
        WHERE RunId = ? AND Status = N'queued'
        ORDER BY SequenceNo;
        """,
        run_id,
    )
    return [row_to_dict(cursor, row) for row in cursor.fetchall()]


def mark_run_running(cursor: Any, run_id: int) -> None:
    cursor.execute(
        """
        UPDATE dbo.readmrz_passport_evaluation_runs
        SET Status = N'running', StartedAt = COALESCE(StartedAt, SYSUTCDATETIME()),
            CompletedAt = NULL, ErrorMessage = NULL, UpdatedDate = SYSUTCDATETIME()
        WHERE Id = ?;
        """,
        run_id,
    )


def mark_item_processing(cursor: Any, item_id: int) -> None:
    cursor.execute(
        """
        UPDATE dbo.readmrz_passport_evaluation_items
        SET Status = N'processing', UpdatedDate = SYSUTCDATETIME()
        WHERE Id = ? AND Status = N'queued';
        """,
        item_id,
    )
    if cursor.rowcount != 1:
        raise RuntimeError(f"Evaluation item {item_id} is no longer queued")


def mark_item_failed(cursor: Any, item_id: int, exc: Exception, elapsed_ms: int) -> None:
    error_code = type(exc).__name__[:100]
    error_message = (str(exc) or error_code)[:4000]
    cursor.execute(
        """
        UPDATE dbo.readmrz_passport_evaluation_items
        SET Status = N'failed', ReadSuccess = 0, ProcessTimeMs = ?,
            ErrorCode = ?, ErrorMessage = ?, ProcessedAt = SYSUTCDATETIME(),
            UpdatedDate = SYSUTCDATETIME()
        WHERE Id = ?;
        """,
        max(0, elapsed_ms),
        error_code,
        error_message,
        item_id,
    )


def mark_item_succeeded(
    cursor: Any,
    *,
    item: dict[str, Any],
    image_path: Path,
    image_width: int,
    image_height: int,
    result: dict[str, Any],
    debug: dict[str, Any],
    reference_date: date,
    elapsed_ms: int,
) -> dict[str, Any]:
    fields = result.get("fields") if isinstance(result.get("fields"), dict) else {}
    lines = result.get("lines") if isinstance(result.get("lines"), list) else []
    fallback = result.get("fallback") if isinstance(result.get("fallback"), dict) else {}
    orientation = debug.get("orientation") if isinstance(debug.get("orientation"), dict) else {}
    classification = classify_passport_dates(
        fields.get("birth_date"),
        fields.get("expiry_date"),
        reference_date=reference_date,
    )
    actual_validity = str(classification["validity_status"])
    validity_match = (
        actual_validity != VALIDITY_UNKNOWN
        and actual_validity == str(item["ExpectedValidityStatus"])
    )
    matched_country = country_match(
        str(item["CountryCode"]),
        fields.get("nationality"),
        fields.get("issuing_country"),
    )
    read_success = bool(result.get("found"))
    line_values = [str(line.get("text") or "") for line in lines if isinstance(line, dict)]
    surname = clean_text(fields.get("surname"))
    given_names = clean_text(fields.get("given_names"))
    full_name = " ".join(value for value in (surname, given_names) if value) or None
    orientation_degree = (
        right_angle(orientation.get("applied_angle"))
        + right_angle(debug.get("selected_rotation_angle"))
    ) % 360
    error_message = clean_text(result.get("error"), 4000)
    error_code = None if read_success else "PIPELINE_NOT_FOUND"

    cursor.execute(
        """
        UPDATE dbo.readmrz_passport_evaluation_items
        SET Status = N'succeeded', FileSha256 = ?, ReadSuccess = ?, PipelineFound = ?,
            ChecksumValid = ?, IsValidityMatch = ?, IsCountryMatch = ?,
            ImageWidth = ?, ImageHeight = ?, OrientationDegree = ?,
            YoloConfidence = ?, OcrConfidence = ?, UsedFallback = ?, ProcessTimeMs = ?,
            MrzLineOne = ?, MrzLineTwo = ?, PassportType = ?, IssuingCountry = ?,
            Surname = ?, GivenNames = ?, FullName = ?, PassportNumber = ?, Nationality = ?,
            DateOfBirth = ?, Sex = ?, DateOfExpiry = ?, PersonalNumber = ?,
            AgeYears = ?, AgeGroup = ?, ActualValidityStatus = ?, DaysToExpiry = ?,
            ResultJson = ?, ErrorCode = ?, ErrorMessage = ?,
            ProcessedAt = SYSUTCDATETIME(), UpdatedDate = SYSUTCDATETIME()
        WHERE Id = ?;
        """,
        sha256_file(image_path),
        1 if read_success else 0,
        1 if result.get("found") else 0,
        1 if result.get("checksum_pass") else 0,
        1 if validity_match else 0,
        None if matched_country is None else (1 if matched_country else 0),
        image_width,
        image_height,
        orientation_degree,
        float(result.get("detector_confidence") or 0.0),
        float(result.get("average_ocr_confidence") or 0.0),
        1 if fallback.get("used") else 0,
        max(0, elapsed_ms),
        clean_text(line_values[0] if len(line_values) > 0 else None, 128),
        clean_text(line_values[1] if len(line_values) > 1 else None, 128),
        clean_text(result.get("document_type"), 20),
        clean_text(fields.get("issuing_country"), 20),
        clean_text(surname, 255),
        clean_text(given_names, 255),
        clean_text(full_name, 512),
        clean_text(fields.get("document_number"), 50),
        clean_text(fields.get("nationality"), 20),
        classification["birth_date"],
        clean_text(fields.get("sex"), 20),
        classification["expiry_date"],
        clean_text(fields.get("optional_data"), 100),
        classification["age_years"],
        classification["age_group"],
        actual_validity,
        classification["days_to_expiry"],
        json_text(result),
        error_code,
        error_message,
        int(item["Id"]),
    )
    return {
        "read_success": read_success,
        "validity_match": validity_match,
        "country_match": matched_country,
        "age_group": classification["age_group"],
        "validity_status": actual_validity,
        "checksum_valid": bool(result.get("checksum_pass")),
    }


def finish_run(cursor: Any, run_id: int) -> str:
    cursor.execute(
        """
        SELECT
            SUM(CASE WHEN Status IN (N'queued', N'processing') THEN 1 ELSE 0 END),
            SUM(CASE WHEN Status = N'failed' OR ReadSuccess = 0 THEN 1 ELSE 0 END)
        FROM dbo.readmrz_passport_evaluation_items
        WHERE RunId = ?;
        """,
        run_id,
    )
    pending_count, error_count = cursor.fetchone()
    if int(pending_count or 0) > 0:
        status = "running"
        completed_at = False
    else:
        status = "completed_with_errors" if int(error_count or 0) > 0 else "completed"
        completed_at = True
    cursor.execute(
        f"""
        UPDATE dbo.readmrz_passport_evaluation_runs
        SET Status = ?, CompletedAt = {'SYSUTCDATETIME()' if completed_at else 'NULL'},
            UpdatedDate = SYSUTCDATETIME()
        WHERE Id = ?;
        """,
        status,
        run_id,
    )
    return status


def fail_run(cursor: Any, run_id: int, exc: Exception) -> None:
    cursor.execute(
        """
        UPDATE dbo.readmrz_passport_evaluation_runs
        SET Status = N'failed', ErrorMessage = ?, CompletedAt = SYSUTCDATETIME(),
            UpdatedDate = SYSUTCDATETIME()
        WHERE Id = ?;
        """,
        (str(exc) or type(exc).__name__)[:4000],
        run_id,
    )


def run_report(cursor: Any, run_id: int) -> dict[str, Any]:
    cursor.execute(
        "SELECT * FROM dbo.vw_readmrz_passport_evaluation_summary WHERE RunId = ?;",
        run_id,
    )
    summary_row = cursor.fetchone()
    if summary_row is None:
        raise LookupError(f"Evaluation summary not found: {run_id}")
    summary = row_to_dict(cursor, summary_row)

    cursor.execute(
        """
        SELECT CountryCode,
               COUNT(*) AS TotalItems,
               SUM(CASE WHEN ReadSuccess = 1 THEN 1 ELSE 0 END) AS ReadSuccessItems,
               SUM(CASE WHEN IsValidityMatch = 1 THEN 1 ELSE 0 END) AS ValidityCorrectItems,
               SUM(CASE WHEN IsValidityMatch = 0 THEN 1 ELSE 0 END) AS ValidityIncorrectItems,
               SUM(CASE WHEN AgeGroup = N'child' THEN 1 ELSE 0 END) AS ChildPassportItems,
               SUM(CASE WHEN ActualValidityStatus = N'expired' THEN 1 ELSE 0 END) AS ExpiredItems,
               SUM(CASE WHEN ActualValidityStatus = N'expiring_within_6_months' THEN 1 ELSE 0 END) AS UnderSixMonthsItems,
               SUM(CASE WHEN ActualValidityStatus = N'valid_6_months_or_more' THEN 1 ELSE 0 END) AS SixMonthsOrMoreItems
        FROM dbo.readmrz_passport_evaluation_items
        WHERE RunId = ?
        GROUP BY CountryCode
        ORDER BY CountryCode;
        """,
        run_id,
    )
    countries = [row_to_dict(cursor, row) for row in cursor.fetchall()]
    return {"summary": summary, "countries": countries}


def clean_text(value: Any, limit: int = 4000) -> str | None:
    text = str(value or "").strip()
    return text[:limit] if text else None


def right_angle(value: Any) -> int:
    try:
        normalized = int(round(float(value))) % 360
    except (TypeError, ValueError):
        return 0
    return normalized if normalized in {0, 90, 180, 270} else 0


def apply_env_file(path: Path) -> dict[str, str]:
    env = read_env_file(path)
    for key, value in env.items():
        os.environ.setdefault(key, value)
    return env


def print_discovery(result: Any, source_root: Path, reference_date: date) -> None:
    expected_counts: dict[str, int] = {}
    country_counts: dict[str, int] = {}
    for candidate in result.candidates:
        expected_counts[candidate.expected_validity_status] = (
            expected_counts.get(candidate.expected_validity_status, 0) + 1
        )
        country_counts[candidate.country_code] = country_counts.get(candidate.country_code, 0) + 1
    print(
        json.dumps(
            {
                "source_root": str(source_root),
                "reference_date": reference_date.isoformat(),
                "passport_directories": result.passport_directories,
                "ignored_passport_directories": list(result.ignored_passport_directories),
                "files": len(result.candidates),
                "expected_validity": expected_counts,
                "countries": country_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> int:
    args = build_parser().parse_args()
    env_path = Path(args.env).expanduser().resolve()
    env = apply_env_file(env_path)
    selected_countries = country_filters(args.country)

    if args.resume_run_id:
        if args.dry_run:
            raise ValueError("--dry-run cannot be combined with --resume-run-id")
        if not args.skip_schema:
            execute_sql_file(SCHEMA_PATH)
        with connect() as connection:
            cursor = connection.cursor()
            run = load_run(cursor, args.resume_run_id)
            source_root = Path(str(run["SourceRoot"])).expanduser().resolve()
            reference_date = run["ReferenceDate"]
            if isinstance(reference_date, datetime):
                reference_date = reference_date.date()
            interrupted = queue_interrupted_items(cursor, args.resume_run_id)
            if interrupted:
                print(f"Recovered interrupted items: {interrupted}")
            if args.retry_failed:
                retried = queue_failed_items(cursor, args.resume_run_id)
                print(f"Queued failed items again: {retried}")
            mark_run_running(cursor, args.resume_run_id)
            connection.commit()
            run_id = args.resume_run_id
            run_key = str(run["RunKey"])
    else:
        configured_root = env_value(env, "READMRZ_PASSPORT_EVALUATION_ROOT", "")
        root_value = args.source_root or configured_root
        if not root_value:
            raise ValueError(
                "source_root is required, or configure READMRZ_PASSPORT_EVALUATION_ROOT"
            )
        source_root = Path(root_value).expanduser().resolve()
        reference_date, reference_date_source = parse_reference_date(
            args.reference_date,
            source_root,
        )
        discovery = discover_passport_images(
            source_root,
            country_codes=selected_countries,
            limit=max(0, args.limit),
        )
        print_discovery(discovery, source_root, reference_date)
        if not discovery.candidates:
            raise ValueError("No passport images were found in recognized export folders")
        if args.dry_run:
            print("DRY RUN: no models loaded and no database rows written.")
            return 0
        if not args.skip_schema:
            execute_sql_file(SCHEMA_PATH)
        config = {
            "countries": selected_countries,
            "limit": max(0, args.limit),
            "reference_date": reference_date.isoformat(),
            "reference_date_source": reference_date_source,
            "schema": SCHEMA_PATH.name,
        }
        with connect() as connection:
            cursor = connection.cursor()
            run_id, run_key = create_run(
                cursor,
                run_name=args.run_name or source_root.name,
                source_root=source_root,
                reference_date=reference_date,
                candidates=discovery.candidates,
                config=config,
            )
            mark_run_running(cursor, run_id)
            connection.commit()

    print(f"Evaluation run: id={run_id} key={run_key}")
    print("Loading MRZ pipeline models...")
    try:
        orientation = PaddleDocumentOrientation()
        detector = YoloMrzDetector()
        recognizer = CustomMrzCtcRecognizer()
        print(
            "Loaded models "
            f"orientation={orientation.load_ms}ms yolo={detector.load_ms}ms "
            f"ocr={recognizer.load_ms}ms"
        )
        if not args.no_warmup:
            print(
                "Warmup models "
                f"orientation={orientation.warmup()}ms yolo={detector.warmup()}ms "
                f"ocr={recognizer.warmup()}ms"
            )
    except Exception as exc:
        with connect() as connection:
            fail_run(connection.cursor(), run_id, exc)
            connection.commit()
        raise

    progress_every = max(1, int(args.progress_every))
    with connect() as connection:
        cursor = connection.cursor()
        items = pending_items(cursor, run_id)
        print(f"Queued items to process: {len(items)}")
        for index, item in enumerate(items, start=1):
            started = time.perf_counter()
            image_path = (source_root / str(item["RelativeImagePath"])).resolve()
            try:
                image_path.relative_to(source_root)
                mark_item_processing(cursor, int(item["Id"]))
                connection.commit()
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Cannot decode image: {image_path}")
                height, width = image.shape[:2]
                debug = process_yolo_upload(
                    image,
                    detector,
                    orientation,
                    recognizer,
                    include_images=False,
                )
                result = compact_yolo_read_payload(debug)
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                outcome = mark_item_succeeded(
                    cursor,
                    item=item,
                    image_path=image_path,
                    image_width=int(width),
                    image_height=int(height),
                    result=result,
                    debug=debug,
                    reference_date=reference_date,
                    elapsed_ms=elapsed_ms,
                )
                connection.commit()
                message = (
                    f"read={outcome['read_success']} validity={outcome['validity_status']} "
                    f"match={outcome['validity_match']} child={outcome['age_group'] == 'child'}"
                )
            except Exception as exc:
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                mark_item_failed(cursor, int(item["Id"]), exc, elapsed_ms)
                connection.commit()
                message = f"error={exc}"

            if index == 1 or index == len(items) or index % progress_every == 0:
                print(
                    f"[{index}/{len(items)}] {item['CountryCode']}/"
                    f"{item['SourceGroup']}/{item['OriginalFileName']} {message} "
                    f"elapsed_ms={elapsed_ms}"
                )

        final_status = finish_run(cursor, run_id)
        connection.commit()
        report = run_report(cursor, run_id)

    print(f"Run status: {final_status}")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted. Run again with --resume-run-id to continue.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
