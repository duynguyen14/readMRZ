from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import pyodbc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mrz_reader.custom_mrz_ocr import CustomMrzCtcRecognizer  # noqa: E402
from mrz_reader.db import connect, execute_sql_file  # noqa: E402
from mrz_reader.document_orientation import PaddleDocumentOrientation  # noqa: E402
from mrz_reader.env_config import env_value, read_env_file  # noqa: E402
from mrz_reader.mrz import normalize_mrz_text  # noqa: E402
from mrz_reader.yolo_detector import YoloMrzDetector  # noqa: E402
from mrz_reader.yolo_upload_pipeline import process_yolo_upload  # noqa: E402
from tools.generate_evisa_mrz_ocr_line3_dataset import (  # noqa: E402
    connect_source,
    load_csharp_config,
    point_to_float,
    resolve_source_image_path,
    source_root_dir,
)


def ensure_schema() -> None:
    execute_sql_file(PROJECT_ROOT / "sql" / "create_readmrz_pipeline_test_tables.sql")


def row_to_dict(cursor: Any, row: Any) -> dict[str, Any]:
    columns = [column[0] for column in cursor.description]
    return dict(zip(columns, row))


def fetch_source_records(
    cursor: pyodbc.Cursor,
    *,
    limit: int,
    min_point: float,
    order_by: str,
) -> list[dict[str, Any]]:
    if order_by == "random":
        order_clause = "NEWID()"
    elif order_by == "newest":
        order_clause = "CreatedDate DESC, Id DESC"
    else:
        order_clause = "CreatedDate ASC, Id ASC"

    cursor.execute(
        f"""
        SELECT TOP ({int(limit)})
            Id,
            GUID,
            PassportNo,
            FullPassportImage,
            MrzlineOne,
            MrzlineTwo,
            MrzlineOnePoint,
            MrzlineTwoPoint
        FROM [db_dichvu_visa].[dbo].[TransactionEVisa]
        WHERE ISNULL(LTRIM(RTRIM(MrzlineOne)), '') <> ''
          AND ISNULL(LTRIM(RTRIM(MrzlineTwo)), '') <> ''
          AND TRY_CONVERT(float, MrzlineOnePoint) > ?
          AND TRY_CONVERT(float, MrzlineTwoPoint) > ?
          AND ISNULL(LTRIM(RTRIM(FullPassportImage)), '') <> ''
        ORDER BY {order_clause}
        """,
        min_point,
        min_point,
    )
    return [row_to_dict(cursor, row) for row in cursor.fetchall()]


def normalize_for_match(value: Any) -> str:
    return normalize_mrz_text(str(value or ""))


def image_payload_to_file(payload: dict[str, Any] | None, path: Path) -> str | None:
    if not payload or not payload.get("image_base64"):
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(str(payload["image_base64"])))
    return str(path)


def parsed_full_name(fields: dict[str, Any]) -> str | None:
    names = [
        str(fields.get("surname") or "").strip(),
        str(fields.get("given_names") or "").strip(),
    ]
    value = " ".join(item for item in names if item)
    return value or None


def first_line_value(lines: list[dict[str, Any]], index: int, key: str, default: Any = None) -> Any:
    if index < len(lines):
        return lines[index].get(key, default)
    return default


def insert_result(
    cursor: pyodbc.Cursor,
    *,
    record: dict[str, Any],
    image_path: Path | None,
    payload: dict[str, Any] | None,
    saved_paths: dict[str, str | None],
    elapsed_ms: int,
    error_message: str | None,
) -> None:
    source_line1 = normalize_for_match(record.get("MrzlineOne"))
    source_line2 = normalize_for_match(record.get("MrzlineTwo"))
    line_crops = payload.get("line_crops") if payload else []
    if not isinstance(line_crops, list):
        line_crops = []

    predicted_line1 = normalize_for_match(first_line_value(line_crops, 0, "ocr_normalized_text", ""))
    predicted_line2 = normalize_for_match(first_line_value(line_crops, 1, "ocr_normalized_text", ""))
    is_line1_match = source_line1 == predicted_line1 and bool(source_line1)
    is_line2_match = source_line2 == predicted_line2 and bool(source_line2)
    ocr_payload = payload.get("mrz_ocr") if payload else {}
    fields = ocr_payload.get("fields") if isinstance(ocr_payload, dict) else {}
    fields = fields if isinstance(fields, dict) else {}
    fallback = payload.get("ocr_fallback") if payload else {}
    fallback = fallback if isinstance(fallback, dict) else {}

    cursor.execute(
        """
        INSERT INTO dbo.readmrz_pipeline_test_items (
            TransactionEVisaId,
            TransactionGuid,
            PassportNo,
            SourceMrzlineOne,
            SourceMrzlineTwo,
            SourceMrzlineOnePoint,
            SourceMrzlineTwoPoint,
            PredictedMrzlineOne,
            PredictedMrzlineTwo,
            LineOneConfidence,
            LineTwoConfidence,
            IsLineOneMatch,
            IsLineTwoMatch,
            IsFullMatch,
            ParseChecksumOk,
            ParsedPassportType,
            ParsedPassportNo,
            ParsedFullName,
            ParsedDob,
            ParsedGender,
            ParsedNationality,
            ParsedExpireDate,
            ParsedIssuerCountry,
            ImagePath,
            RawMrzCropPath,
            DeskewedMrzCropPath,
            LineOneCropPath,
            LineTwoCropPath,
            YoloConfidence,
            OrientationDegree,
            UsedFallback,
            FallbackReason,
            ProcessTimeMs,
            ErrorMessage
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        int(record["Id"]),
        str(record.get("GUID")) if record.get("GUID") else None,
        str(record.get("PassportNo") or "") or None,
        source_line1,
        source_line2,
        point_to_float(record.get("MrzlineOnePoint")),
        point_to_float(record.get("MrzlineTwoPoint")),
        predicted_line1 or None,
        predicted_line2 or None,
        float(first_line_value(line_crops, 0, "ocr_confidence", 0.0) or 0.0),
        float(first_line_value(line_crops, 1, "ocr_confidence", 0.0) or 0.0),
        1 if is_line1_match else 0,
        1 if is_line2_match else 0,
        1 if is_line1_match and is_line2_match else 0,
        1 if bool(ocr_payload.get("checksum_pass")) else 0,
        fields.get("document_code"),
        fields.get("document_number"),
        parsed_full_name(fields),
        fields.get("birth_date"),
        fields.get("sex"),
        fields.get("nationality"),
        fields.get("expiry_date"),
        fields.get("issuing_country"),
        str(image_path) if image_path else None,
        saved_paths.get("raw_crop"),
        saved_paths.get("deskewed_crop"),
        saved_paths.get("line1"),
        saved_paths.get("line2"),
        float(ocr_payload.get("detector_confidence") or 0.0),
        int((payload.get("orientation") or {}).get("applied_angle") or 0) if payload else None,
        1 if bool(fallback.get("used")) else 0,
        fallback.get("trigger") or fallback.get("selected_variant"),
        elapsed_ms,
        error_message,
    )


def save_pipeline_images(
    payload: dict[str, Any],
    *,
    output_dir: Path,
    source_id: int,
) -> dict[str, str | None]:
    stem = f"TransactionEVisa_{source_id}"
    line_crops = payload.get("line_crops") or []
    return {
        "raw_crop": image_payload_to_file(
            payload.get("mrz_raw_crop"),
            output_dir / "raw_crops" / f"{stem}_raw_mrz.jpg",
        ),
        "deskewed_crop": image_payload_to_file(
            payload.get("mrz_crop"),
            output_dir / "crops" / f"{stem}_mrz.jpg",
        ),
        "line1": image_payload_to_file(
            line_crops[0] if len(line_crops) > 0 else None,
            output_dir / "lines" / f"{stem}_line1.jpg",
        ),
        "line2": image_payload_to_file(
            line_crops[1] if len(line_crops) > 1 else None,
            output_dir / "lines" / f"{stem}_line2.jpg",
        ),
    }


def process_record(
    *,
    record: dict[str, Any],
    source_root: Path,
    output_dir: Path,
    detector: YoloMrzDetector,
    orientation: PaddleDocumentOrientation,
    recognizer: CustomMrzCtcRecognizer,
) -> tuple[dict[str, Any] | None, dict[str, str | None], Path | None, int, str | None]:
    started = time.perf_counter()
    image_path: Path | None = None
    try:
        image_path = resolve_source_image_path(record.get("FullPassportImage"), source_root)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read source image: {image_path}")
        payload = process_yolo_upload(
            image,
            detector,
            orientation,
            recognizer,
            include_images=True,
        )
        saved_paths = save_pipeline_images(
            payload,
            output_dir=output_dir,
            source_id=int(record["Id"]),
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        error_message = payload.get("error") or (payload.get("cropper") or {}).get("error")
        if not error_message and not payload.get("found"):
            error_message = "YOLO did not find MRZ"
        return payload, saved_paths, image_path, elapsed_ms, error_message
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return None, {}, image_path, elapsed_ms, str(exc)[:4000]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run 100 random TransactionEVisa images through the production MRZ pipeline and store comparison results."
    )
    parser.add_argument("--env", default=str(PROJECT_ROOT / ".env"), help="Path to .env config.")
    parser.add_argument("--limit", type=int, default=100, help="Number of TransactionEVisa rows to test.")
    parser.add_argument("--min-point", type=float, default=90.0, help="Minimum source MRZ point for both lines.")
    parser.add_argument("--order-by", choices=["random", "newest", "oldest"], default="random", help="Source row order.")
    parser.add_argument("--skip-schema", action="store_true", help="Do not create/update destination schema.")
    parser.add_argument("--clear", action="store_true", help="Delete existing pipeline test rows before inserting this run.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    env_path = Path(args.env)
    if not env_path.is_absolute():
        env_path = (Path.cwd() / env_path).resolve()
    env = read_env_file(env_path)
    csharp_config = load_csharp_config(env)
    source_root = source_root_dir(env, csharp_config)
    output_dir = Path(
        env_value(
            env,
            "READMRZ_PIPELINE_TEST_OUTPUT_DIR",
            str(PROJECT_ROOT / "generated_datasets" / "pipeline_tests"),
        )
    ).expanduser().resolve()

    if not args.skip_schema:
        ensure_schema()

    print(f"source_root: {source_root}")
    print(f"output_dir: {output_dir}")
    print(f"limit: {args.limit}")
    print(f"min_point: {args.min_point}")
    print(f"order_by: {args.order_by}")

    print("Loading pipeline models...")
    orientation = PaddleDocumentOrientation()
    detector = YoloMrzDetector()
    recognizer = CustomMrzCtcRecognizer()
    print(
        "Loaded models "
        f"orientation={orientation.load_ms}ms yolo={detector.load_ms}ms ocr={recognizer.load_ms}ms"
    )

    totals = {
        "fetched": 0,
        "inserted": 0,
        "matched": 0,
        "mismatched": 0,
        "errors": 0,
        "fallback": 0,
    }

    with connect_source(env, csharp_config) as source_connection, connect() as destination_connection:
        source_cursor = source_connection.cursor()
        destination_cursor = destination_connection.cursor()
        if args.clear:
            destination_cursor.execute("DELETE FROM dbo.readmrz_pipeline_test_items")
            destination_connection.commit()
            print("Cleared dbo.readmrz_pipeline_test_items")

        records = fetch_source_records(
            source_cursor,
            limit=args.limit,
            min_point=args.min_point,
            order_by=args.order_by,
        )
        totals["fetched"] = len(records)
        print(f"Fetched {len(records)} TransactionEVisa rows")

        for index, record in enumerate(records, start=1):
            payload, saved_paths, image_path, elapsed_ms, error_message = process_record(
                record=record,
                source_root=source_root,
                output_dir=output_dir,
                detector=detector,
                orientation=orientation,
                recognizer=recognizer,
            )
            insert_result(
                destination_cursor,
                record=record,
                image_path=image_path,
                payload=payload,
                saved_paths=saved_paths,
                elapsed_ms=elapsed_ms,
                error_message=error_message,
            )
            destination_connection.commit()

            line1 = normalize_for_match((payload.get("line_crops") or [{}])[0].get("ocr_normalized_text")) if payload and payload.get("line_crops") else ""
            line2 = normalize_for_match((payload.get("line_crops") or [{}, {}])[1].get("ocr_normalized_text")) if payload and len(payload.get("line_crops") or []) > 1 else ""
            matched = (
                line1 == normalize_for_match(record.get("MrzlineOne"))
                and line2 == normalize_for_match(record.get("MrzlineTwo"))
            )
            used_fallback = bool((payload.get("ocr_fallback") or {}).get("used")) if payload else False
            totals["inserted"] += 1
            totals["matched" if matched else "mismatched"] += 1
            totals["fallback"] += 1 if used_fallback else 0
            totals["errors"] += 1 if error_message else 0
            print(
                f"[{index}/{len(records)}] TransactionEVisa:{record['Id']} "
                f"match={matched} fallback={used_fallback} elapsed_ms={elapsed_ms} "
                f"error={error_message or ''}"
            )

    print(json.dumps(totals, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
