from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, datetime
import json
import os
from pathlib import Path
import sys
from typing import Any

import pyodbc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mrz_reader.db import connect, execute_sql_file  # noqa: E402
from mrz_reader.env_config import read_env_file as read_project_env  # noqa: E402
from mrz_reader.passport_evaluation import sha256_file  # noqa: E402
from mrz_reader.passport_source_comparison import (  # noqa: E402
    FIELD_COMPARATORS,
    SOURCE_MATCH_AMBIGUOUS,
    SOURCE_MATCH_ERROR,
    SOURCE_MATCH_MATCHED,
    SOURCE_MATCH_NOT_FOUND,
    SourceCandidate,
    compare_passport_fields,
    evaluation_observed_fields,
    exported_file_aliases,
    parse_source_record,
    resolve_source_image_path,
    select_source_candidate,
    serializable,
)
from tools.generate_evisa_mrz_ocr_line3_dataset import (  # noqa: E402
    build_source_connection_string,
    env_value,
    load_csharp_config,
    read_env_file,
)


SCHEMA_PATH = PROJECT_ROOT / "sql" / "extend_passport_evaluation_source_comparison.sql"
REVIEW_SCHEMA_PATH = PROJECT_ROOT / "sql" / "add_passport_evaluation_review.sql"

MATCH_COLUMN_BY_FIELD = {
    "mrz_line_one": "MrzLineOneMatch",
    "mrz_line_two": "MrzLineTwoMatch",
    "passport_type": "PassportTypeMatch",
    "issuing_country": "IssuingCountryMatch",
    "surname": "SurnameMatch",
    "given_names": "GivenNamesMatch",
    "full_name": "FullNameMatch",
    "passport_number": "PassportNumberMatch",
    "nationality": "NationalityMatch",
    "date_of_birth": "DateOfBirthMatch",
    "sex": "SexMatch",
    "date_of_expiry": "DateOfExpiryMatch",
    "personal_number": "PersonalNumberMatch",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Match passport evaluation items to db_dichvu_visa.dbo.TransactionEVisa, "
            "store a source snapshot and compare MRZ/parsed passport fields."
        )
    )
    parser.add_argument("--run-id", type=int, required=True, help="Passport evaluation run ID on the destination DB.")
    parser.add_argument("--env", default=str(PROJECT_ROOT / ".env"), help="READMRZ .env path.")
    parser.add_argument("--source-application-json", default="", help="Optional C# application.json containing source DB/root settings.")
    parser.add_argument("--source-root", default="", help="Override source image root directory.")
    parser.add_argument("--country", action="append", default=[], help="Only these country codes; repeat or comma-separate.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum destination items. 0 means all.")
    parser.add_argument("--force", action="store_true", help="Recompare items already enriched.")
    parser.add_argument("--no-hash", action="store_true", help="Do not read source files for SHA-256 disambiguation.")
    parser.add_argument("--skip-schema", action="store_true", help="Do not apply the idempotent comparison schema migration.")
    parser.add_argument("--dry-run", action="store_true", help="Match and compare without writing destination DB.")
    parser.add_argument("--commit-every", type=int, default=100, help="Commit every N destination updates.")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N items.")
    return parser


def apply_env(path: Path) -> dict[str, str]:
    env = read_project_env(path)
    for key, value in env.items():
        os.environ.setdefault(key, value)
    return env


def country_filters(values: list[str]) -> list[str]:
    return sorted(
        {
            item.strip().upper()
            for value in values
            for item in value.split(",")
            if item.strip()
        }
    )


def row_to_dict(cursor: Any, row: Any) -> dict[str, Any]:
    return dict(zip((str(column[0]) for column in cursor.description), row))


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=serializable)


def comparison_schema_exists(cursor: Any) -> bool:
    cursor.execute(
        "SELECT CASE WHEN COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourceMatchStatus') IS NULL THEN 0 ELSE 1 END;"
    )
    return bool(cursor.fetchone()[0])


def review_schema_exists(cursor: Any) -> bool:
    cursor.execute(
        "SELECT CASE WHEN COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'ReviewStatus') IS NULL THEN 0 ELSE 1 END;"
    )
    return bool(cursor.fetchone()[0])


def refresh_review_statuses(cursor: Any, run_id: int) -> None:
    cursor.execute(
        """
        UPDATE item
        SET ReviewStatus = CASE
                WHEN item.ChecksumValid = 1
                 AND item.AllComparedFieldsMatch = 1
                 AND JSON_VALUE(item.SourceDataJson, '$.normalized_passport_fields.source_mrz_checksum_valid') = N'true'
                    THEN N'auto_approved'
                ELSE N'pending'
            END,
            ReviewBasis = CASE
                WHEN item.ChecksumValid = 1
                 AND item.AllComparedFieldsMatch = 1
                 AND JSON_VALUE(item.SourceDataJson, '$.normalized_passport_fields.source_mrz_checksum_valid') = N'true'
                    THEN N'strict_checksum_and_source_match'
                ELSE N'requires_human_review'
            END,
            ReviewUpdatedAt = SYSUTCDATETIME()
        FROM dbo.readmrz_passport_evaluation_items item
        WHERE item.RunId = ?
          AND item.SourceComparedAt IS NOT NULL
          AND (item.ReviewStatus IS NULL OR item.ReviewStatus IN (N'auto_approved', N'pending'));
        """,
        run_id,
    )


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


def load_items(
    cursor: Any,
    *,
    run_id: int,
    countries: list[str],
    limit: int,
    force: bool,
    has_comparison_schema: bool,
) -> list[dict[str, Any]]:
    top_clause = f"TOP ({int(limit)})" if limit > 0 else ""
    conditions = ["RunId = ?"]
    params: list[Any] = [run_id]
    if countries:
        placeholders = ",".join("?" for _ in countries)
        conditions.append(f"CountryCode IN ({placeholders})")
        params.extend(countries)
    if has_comparison_schema and not force:
        conditions.append("SourceComparedAt IS NULL")
    cursor.execute(
        f"""
        SELECT {top_clause}
            Id, RunId, SequenceNo, CountryCode, SourceGroup, ExpectedValidityStatus,
            RelativeImagePath, OriginalFileName, FileSha256,
            MrzLineOne, MrzLineTwo, PassportType, IssuingCountry, Surname, GivenNames,
            FullName, PassportNumber, Nationality, DateOfBirth, Sex, DateOfExpiry,
            PersonalNumber
        FROM dbo.readmrz_passport_evaluation_items
        WHERE {' AND '.join(conditions)}
        ORDER BY SequenceNo;
        """,
        params,
    )
    return [row_to_dict(cursor, row) for row in cursor.fetchall()]


def create_requested_file_table(cursor: Any, items: list[dict[str, Any]]) -> None:
    cursor.execute(
        """
        IF OBJECT_ID('tempdb..#RequestedPassportFiles') IS NOT NULL DROP TABLE #RequestedPassportFiles;
        CREATE TABLE #RequestedPassportFiles (
            ItemId BIGINT NOT NULL,
            FileNameLower NVARCHAR(512) NOT NULL,
            MatchPriority INT NOT NULL
        );
        CREATE INDEX IX_RequestedPassportFiles_Name ON #RequestedPassportFiles (FileNameLower);
        """
    )
    aliases = [
        (int(item["Id"]), alias, priority)
        for item in items
        for alias, priority in exported_file_aliases(str(item.get("OriginalFileName") or ""))
    ]
    if aliases:
        cursor.executemany(
            "INSERT INTO #RequestedPassportFiles (ItemId, FileNameLower, MatchPriority) VALUES (?, ?, ?);",
            aliases,
        )


def fetch_source_candidates(cursor: Any, items: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    create_requested_file_table(cursor, items)
    cursor.execute(
        """
        SELECT
            requested.ItemId,
            requested.MatchPriority,
            source.Id,
            source.GUID,
            source.CreatedDate,
            source.ModifiedDate,
            source.PassportNo,
            source.IssuePassportDate,
            source.IssuePassportPlace,
            source.Fullname,
            source.Dob,
            source.DobStr,
            source.Gender,
            source.NationalityCode,
            source.NationalityOriginalCode,
            source.PassportType,
            source.FullPassportImage,
            source.FaceImage,
            source.MrzlineOne,
            source.MrzlineTwo,
            source.MrzlineOnePoint,
            source.MrzlineTwoPoint,
            source.PassportExpireDate,
            source.PassportImageNote,
            source.FaceImageNote
        FROM dbo.TransactionEVisa source
        CROSS APPLY (
            SELECT REPLACE(LTRIM(RTRIM(source.FullPassportImage)), '/', '\\') AS NormalizedPath
        ) normalized
        CROSS APPLY (
            SELECT RIGHT(
                normalized.NormalizedPath,
                CHARINDEX('\\', REVERSE(normalized.NormalizedPath) + '\\') - 1
            ) AS FileName
        ) file_name
        INNER JOIN #RequestedPassportFiles requested
            ON LOWER(file_name.FileName) COLLATE DATABASE_DEFAULT
             = requested.FileNameLower COLLATE DATABASE_DEFAULT
        WHERE ISNULL(LTRIM(RTRIM(source.FullPassportImage)), '') <> ''
        ORDER BY requested.ItemId, requested.MatchPriority, source.Id DESC;
        """
    )
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in cursor.fetchall():
        record = row_to_dict(cursor, row)
        item_id = int(record.pop("ItemId"))
        grouped[item_id].append(record)
    return grouped


def source_candidate_objects(
    item_id: int,
    records: list[dict[str, Any]],
    *,
    source_root: Path,
    exported_sha256: str | None,
    verify_hash: bool,
    hash_cache: dict[str, str],
) -> list[SourceCandidate]:
    candidates: list[SourceCandidate] = []
    should_hash = verify_hash and bool(exported_sha256) and len(records) > 1
    for record in records:
        source_path = resolve_source_image_path(record.get("FullPassportImage"), source_root)
        source_hash = None
        if should_hash and source_path is not None and source_path.is_file():
            key = str(source_path).lower()
            source_hash = hash_cache.get(key)
            if source_hash is None:
                source_hash = sha256_file(source_path)
                hash_cache[key] = source_hash
        candidates.append(
            SourceCandidate(
                item_id=item_id,
                match_priority=int(record.get("MatchPriority") or 0),
                record=record,
                source_path=source_path,
                file_sha256=source_hash,
            )
        )
    return candidates


def clear_source_comparison(cursor: Any, item_id: int, status: str, reason: str, candidate_count: int) -> None:
    cursor.execute(
        """
        UPDATE dbo.readmrz_passport_evaluation_items
        SET SourceMatchStatus = ?, SourceMatchReason = ?, SourceCandidateCount = ?,
            SourceTransactionEVisaId = NULL, SourceTransactionEVisaGuid = NULL,
            SourceCreatedDate = NULL, SourceModifiedDate = NULL,
            SourceFullPassportImage = NULL, SourceFaceImage = NULL, SourceFileSha256 = NULL,
            SourceMrzLineOne = NULL, SourceMrzLineTwo = NULL,
            SourceMrzLineOnePoint = NULL, SourceMrzLineTwoPoint = NULL,
            SourcePassportType = NULL, SourceDatabasePassportType = NULL,
            SourceIssuingCountry = NULL,
            SourceSurname = NULL, SourceGivenNames = NULL, SourceFullName = NULL,
            SourceDatabaseFullName = NULL, SourcePassportNumber = NULL,
            SourceNationality = NULL, SourceNationalityOriginal = NULL,
            SourceDateOfBirth = NULL, SourceSex = NULL, SourceDateOfExpiry = NULL,
            SourceIssuePassportDate = NULL, SourceIssuePassportPlace = NULL,
            SourcePersonalNumber = NULL, SourcePassportImageNote = NULL,
            SourceFaceImageNote = NULL, SourceDataJson = NULL,
            MrzLineOneMatch = NULL, MrzLineTwoMatch = NULL, PassportTypeMatch = NULL,
            IssuingCountryMatch = NULL, SurnameMatch = NULL, GivenNamesMatch = NULL,
            FullNameMatch = NULL, PassportNumberMatch = NULL, NationalityMatch = NULL,
            DateOfBirthMatch = NULL, SexMatch = NULL, DateOfExpiryMatch = NULL,
            PersonalNumberMatch = NULL, ComparedFieldCount = NULL,
            MatchedFieldCount = NULL, AllComparedFieldsMatch = NULL,
            ComparisonJson = NULL, SourceComparedAt = SYSUTCDATETIME(),
            UpdatedDate = SYSUTCDATETIME()
        WHERE Id = ?;
        """,
        status,
        reason[:500],
        candidate_count,
        item_id,
    )


def update_matched_item(
    cursor: Any,
    *,
    item_id: int,
    selection_reason: str,
    candidate_count: int,
    source: dict[str, Any],
    source_snapshot: dict[str, Any],
    source_sha256: str | None,
    comparison: dict[str, Any],
) -> None:
    matches = comparison["fields"]
    values = {
        MATCH_COLUMN_BY_FIELD[field]: details["match"]
        for field, details in matches.items()
    }
    cursor.execute(
        """
        UPDATE dbo.readmrz_passport_evaluation_items
        SET SourceMatchStatus = N'matched', SourceMatchReason = ?, SourceCandidateCount = ?,
            SourceTransactionEVisaId = ?, SourceTransactionEVisaGuid = ?,
            SourceCreatedDate = ?, SourceModifiedDate = ?,
            SourceFullPassportImage = ?, SourceFaceImage = ?, SourceFileSha256 = ?,
            SourceMrzLineOne = ?, SourceMrzLineTwo = ?,
            SourceMrzLineOnePoint = ?, SourceMrzLineTwoPoint = ?,
            SourcePassportType = ?, SourceDatabasePassportType = ?, SourceIssuingCountry = ?,
            SourceSurname = ?, SourceGivenNames = ?, SourceFullName = ?,
            SourceDatabaseFullName = ?, SourcePassportNumber = ?,
            SourceNationality = ?, SourceNationalityOriginal = ?, SourceDateOfBirth = ?,
            SourceSex = ?, SourceDateOfExpiry = ?, SourcePersonalNumber = ?,
            SourceIssuePassportDate = ?, SourceIssuePassportPlace = ?,
            SourcePassportImageNote = ?, SourceFaceImageNote = ?,
            SourceDataJson = ?,
            MrzLineOneMatch = ?, MrzLineTwoMatch = ?, PassportTypeMatch = ?,
            IssuingCountryMatch = ?, SurnameMatch = ?, GivenNamesMatch = ?,
            FullNameMatch = ?, PassportNumberMatch = ?, NationalityMatch = ?,
            DateOfBirthMatch = ?, SexMatch = ?, DateOfExpiryMatch = ?,
            PersonalNumberMatch = ?, ComparedFieldCount = ?, MatchedFieldCount = ?,
            AllComparedFieldsMatch = ?, ComparisonJson = ?,
            SourceComparedAt = SYSUTCDATETIME(), UpdatedDate = SYSUTCDATETIME()
        WHERE Id = ?;
        """,
        selection_reason[:500],
        candidate_count,
        source["transaction_evisa_id"],
        source["transaction_evisa_guid"],
        source["source_created_date"],
        source["source_modified_date"],
        source["full_passport_image"],
        source["face_image"],
        source_sha256,
        source["mrz_line_one"],
        source["mrz_line_two"],
        source["mrz_line_one_point"],
        source["mrz_line_two_point"],
        source["passport_type"],
        source["database_passport_type"],
        source["issuing_country"],
        source["surname"],
        source["given_names"],
        source["full_name"],
        source["database_full_name"],
        source["passport_number"],
        source["nationality"],
        source["nationality_original"],
        source["date_of_birth"],
        source["sex"],
        source["date_of_expiry"],
        source["personal_number"],
        source["issue_passport_date"],
        source["issue_passport_place"],
        source["passport_image_note"],
        source["face_image_note"],
        json_text(source_snapshot),
        bit_or_none(values["MrzLineOneMatch"]),
        bit_or_none(values["MrzLineTwoMatch"]),
        bit_or_none(values["PassportTypeMatch"]),
        bit_or_none(values["IssuingCountryMatch"]),
        bit_or_none(values["SurnameMatch"]),
        bit_or_none(values["GivenNamesMatch"]),
        bit_or_none(values["FullNameMatch"]),
        bit_or_none(values["PassportNumberMatch"]),
        bit_or_none(values["NationalityMatch"]),
        bit_or_none(values["DateOfBirthMatch"]),
        bit_or_none(values["SexMatch"]),
        bit_or_none(values["DateOfExpiryMatch"]),
        bit_or_none(values["PersonalNumberMatch"]),
        comparison["compared_field_count"],
        comparison["matched_field_count"],
        bit_or_none(comparison["all_compared_fields_match"]),
        json_text(comparison),
        item_id,
    )


def bit_or_none(value: Any) -> int | None:
    return None if value is None else int(bool(value))


def source_snapshot(record: dict[str, Any], parsed: dict[str, Any], source_path: Path | None) -> dict[str, Any]:
    source_record = {
        key: serializable(value)
        for key, value in record.items()
        if key != "MatchPriority"
    }
    return {
        "source_table": "db_dichvu_visa.dbo.TransactionEVisa",
        "resolved_source_path": str(source_path) if source_path is not None else None,
        "record": source_record,
        "normalized_passport_fields": {
            key: serializable(value)
            for key, value in parsed.items()
        },
    }


def empty_summary(run_id: int, item_count: int, source_root: Path, dry_run: bool) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "dry_run": dry_run,
        "source_root": str(source_root),
        "items": item_count,
        "matched": 0,
        "not_found": 0,
        "ambiguous": 0,
        "errors": 0,
        "all_fields_match": 0,
        "any_field_mismatch": 0,
        "no_comparable_fields": 0,
        "fields": {
            field: {"compared": 0, "matched": 0, "mismatched": 0}
            for field in FIELD_COMPARATORS
        },
        "countries": {},
    }


def update_summary(summary: dict[str, Any], country: str, status: str, comparison: dict[str, Any] | None) -> None:
    country_summary = summary["countries"].setdefault(
        country,
        {"items": 0, "matched": 0, "not_found": 0, "ambiguous": 0, "errors": 0, "all_fields_match": 0, "any_field_mismatch": 0},
    )
    country_summary["items"] += 1
    summary[status] += 1
    country_summary[status] += 1
    if comparison is None:
        return
    all_match = comparison["all_compared_fields_match"]
    if all_match is True:
        summary["all_fields_match"] += 1
        country_summary["all_fields_match"] += 1
    elif all_match is False:
        summary["any_field_mismatch"] += 1
        country_summary["any_field_mismatch"] += 1
    else:
        summary["no_comparable_fields"] += 1
    for field, details in comparison["fields"].items():
        if not details["comparable"]:
            continue
        summary["fields"][field]["compared"] += 1
        if details["match"]:
            summary["fields"][field]["matched"] += 1
        else:
            summary["fields"][field]["mismatched"] += 1


def main() -> int:
    args = build_parser().parse_args()
    if args.run_id <= 0:
        raise ValueError("--run-id must be positive")
    if args.limit < 0:
        raise ValueError("--limit cannot be negative")
    if args.commit_every <= 0 or args.progress_every <= 0:
        raise ValueError("--commit-every and --progress-every must be positive")

    env_path = Path(args.env).expanduser().resolve()
    apply_env(env_path)
    generator_env = read_env_file(env_path)
    if args.source_application_json:
        configured_application_json = str(Path(args.source_application_json).expanduser().resolve())
        generator_env["READMRZ_EVISA_APPLICATION_JSON"] = configured_application_json
        os.environ["READMRZ_EVISA_APPLICATION_JSON"] = configured_application_json
    csharp_config = load_csharp_config(generator_env)
    if args.source_root:
        source_root = Path(args.source_root).expanduser().resolve()
    else:
        configured_root = env_value(
            generator_env,
            "READMRZ_EVISA_SOURCE_ROOT_DIR",
            str(csharp_config.get("SourceRootDirectory") or ""),
        )
        if not configured_root:
            raise ValueError("Missing source root. Set --source-root or READMRZ_EVISA_SOURCE_ROOT_DIR")
        source_root = Path(configured_root).expanduser().resolve()

    source_connection_string = build_source_connection_string(generator_env, csharp_config)
    filters = country_filters(args.country)

    if not args.dry_run and not args.skip_schema:
        execute_sql_file(SCHEMA_PATH)
        execute_sql_file(REVIEW_SCHEMA_PATH)

    with connect() as destination_connection:
        destination_cursor = destination_connection.cursor()
        has_schema = comparison_schema_exists(destination_cursor)
        if not args.dry_run and not has_schema:
            raise RuntimeError(f"Comparison schema is missing. Run {SCHEMA_PATH}")
        run = load_run(destination_cursor, args.run_id)
        items = load_items(
            destination_cursor,
            run_id=args.run_id,
            countries=filters,
            limit=args.limit,
            force=args.force,
            has_comparison_schema=has_schema,
        )

        with pyodbc.connect(source_connection_string, autocommit=False) as source_connection:
            source_cursor = source_connection.cursor()
            grouped_records = fetch_source_candidates(source_cursor, items)

        reference_date = run["ReferenceDate"]
        if isinstance(reference_date, datetime):
            reference_date = reference_date.date()
        elif not isinstance(reference_date, date):
            reference_date = date.fromisoformat(str(reference_date)[:10])

        summary = empty_summary(args.run_id, len(items), source_root, args.dry_run)
        hash_cache: dict[str, str] = {}
        pending_writes = 0
        for index, item in enumerate(items, start=1):
            item_id = int(item["Id"])
            country = str(item.get("CountryCode") or "UNKNOWN")
            try:
                candidates = source_candidate_objects(
                    item_id,
                    grouped_records.get(item_id, []),
                    source_root=source_root,
                    exported_sha256=item.get("FileSha256"),
                    verify_hash=not args.no_hash,
                    hash_cache=hash_cache,
                )
                selection = select_source_candidate(
                    candidates,
                    expected_country=country,
                    expected_validity=item.get("ExpectedValidityStatus"),
                    reference_date=reference_date,
                    exported_sha256=item.get("FileSha256"),
                )
                if selection.status != SOURCE_MATCH_MATCHED or selection.candidate is None:
                    status_key = {
                        SOURCE_MATCH_NOT_FOUND: "not_found",
                        SOURCE_MATCH_AMBIGUOUS: "ambiguous",
                    }.get(selection.status, "errors")
                    update_summary(summary, country, status_key, None)
                    if not args.dry_run:
                        clear_source_comparison(
                            destination_cursor,
                            item_id,
                            selection.status,
                            selection.reason,
                            selection.candidate_count,
                        )
                        pending_writes += 1
                else:
                    candidate = selection.candidate
                    parsed_source = parse_source_record(candidate.record)
                    observed = evaluation_observed_fields(item)
                    comparison = compare_passport_fields(parsed_source, observed)
                    source_hash = candidate.file_sha256
                    if (
                        not args.no_hash
                        and source_hash is None
                        and candidate.source_path is not None
                        and candidate.source_path.is_file()
                    ):
                        key = str(candidate.source_path).lower()
                        source_hash = hash_cache.get(key)
                        if source_hash is None:
                            source_hash = sha256_file(candidate.source_path)
                            hash_cache[key] = source_hash
                    snapshot = source_snapshot(candidate.record, parsed_source, candidate.source_path)
                    update_summary(summary, country, "matched", comparison)
                    if not args.dry_run:
                        update_matched_item(
                            destination_cursor,
                            item_id=item_id,
                            selection_reason=selection.reason,
                            candidate_count=selection.candidate_count,
                            source=parsed_source,
                            source_snapshot=snapshot,
                            source_sha256=source_hash,
                            comparison=comparison,
                        )
                        pending_writes += 1
            except Exception as exc:
                update_summary(summary, country, "errors", None)
                if not args.dry_run:
                    clear_source_comparison(
                        destination_cursor,
                        item_id,
                        SOURCE_MATCH_ERROR,
                        f"{type(exc).__name__}: {exc}",
                        len(grouped_records.get(item_id, [])),
                    )
                    pending_writes += 1

            if not args.dry_run and pending_writes >= args.commit_every:
                destination_connection.commit()
                pending_writes = 0
            if index == len(items) or index % args.progress_every == 0:
                print(
                    f"[{index}/{len(items)}] matched={summary['matched']} "
                    f"not_found={summary['not_found']} ambiguous={summary['ambiguous']} "
                    f"errors={summary['errors']}"
                )

        if not args.dry_run:
            if review_schema_exists(destination_cursor):
                refresh_review_statuses(destination_cursor, args.run_id)
            destination_connection.commit()

    summary["countries"] = dict(sorted(summary["countries"].items()))
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=serializable))
    if args.dry_run:
        print("DRY RUN: destination schema/data were not changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
