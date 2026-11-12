from __future__ import annotations

from datetime import date, datetime
import json
import mimetypes
from pathlib import Path
from typing import Any
from uuid import UUID

from .db import connect


REVIEW_STATUSES = {"auto_approved", "pending", "approved", "rejected"}
HUMAN_REVIEW_STATUSES = {"approved", "rejected"}


class EvaluationRunNotFoundError(LookupError):
    pass


class EvaluationItemNotFoundError(LookupError):
    pass


def strict_auto_approval(
    checksum_valid: bool | None,
    all_compared_fields_match: bool | None,
    source_checksum_valid: bool | None,
) -> bool:
    return checksum_valid is True and all_compared_fields_match is True and source_checksum_valid is True


class PassportEvaluationReviewRepository:
    def list_runs(self, *, page: int = 1, page_size: int = 20) -> dict[str, Any]:
        page, page_size, offset = _pagination(page, page_size, max_page_size=100)
        with connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM dbo.readmrz_passport_evaluation_runs run
                WHERE EXISTS (
                    SELECT 1 FROM dbo.readmrz_passport_evaluation_items item
                    WHERE item.RunId = run.Id AND item.SourceComparedAt IS NOT NULL
                );
                """
            )
            total = int(cursor.fetchone()[0] or 0)
            cursor.execute(
                """
                SELECT
                    run.Id, run.RunKey, run.RunName, run.SourceRoot, run.ReferenceDate,
                    run.Status, run.CreatedDate, run.CompletedAt,
                    COUNT_BIG(item.Id) AS TotalItems,
                    SUM(CASE WHEN item.ReviewStatus = N'auto_approved' THEN 1 ELSE 0 END) AS AutoApprovedItems,
                    SUM(CASE WHEN item.ReviewStatus = N'pending' THEN 1 ELSE 0 END) AS PendingItems,
                    SUM(CASE WHEN item.ReviewStatus = N'approved' THEN 1 ELSE 0 END) AS ApprovedItems,
                    SUM(CASE WHEN item.ReviewStatus = N'rejected' THEN 1 ELSE 0 END) AS RejectedItems
                FROM dbo.readmrz_passport_evaluation_runs run
                INNER JOIN dbo.readmrz_passport_evaluation_items item
                    ON item.RunId = run.Id AND item.SourceComparedAt IS NOT NULL
                GROUP BY run.Id, run.RunKey, run.RunName, run.SourceRoot, run.ReferenceDate,
                         run.Status, run.CreatedDate, run.CompletedAt
                ORDER BY run.Id DESC
                OFFSET ? ROWS FETCH NEXT ? ROWS ONLY;
                """,
                offset,
                page_size,
            )
            rows = [_row_dict(cursor, row) for row in cursor.fetchall()]
        return {
            "items": [_run_payload(row) for row in rows],
            "pagination": _pagination_payload(page, page_size, total),
        }

    def list_items(
        self,
        run_id: int,
        *,
        review_status: str | None = "pending",
        country: str | None = None,
        page: int = 1,
        page_size: int = 1,
    ) -> dict[str, Any]:
        run_id = _positive_int(run_id, "run_id")
        page, page_size, offset = _pagination(page, page_size, max_page_size=100)
        normalized_status = str(review_status or "").strip().lower()
        if normalized_status == "all":
            normalized_status = ""
        if normalized_status and normalized_status not in REVIEW_STATUSES:
            raise ValueError(f"review_status must be one of: {', '.join(sorted(REVIEW_STATUSES))}")
        normalized_country = str(country or "").strip().upper()[:20]

        with connect() as connection:
            cursor = connection.cursor()
            run = self._get_run(cursor, run_id)
            summary = self._summary(cursor, run_id)
            filters = ["item.RunId = ?", "item.SourceComparedAt IS NOT NULL"]
            parameters: list[Any] = [run_id]
            if normalized_status:
                filters.append("item.ReviewStatus = ?")
                parameters.append(normalized_status)
            if normalized_country:
                filters.append("item.CountryCode = ?")
                parameters.append(normalized_country)
            where_sql = " AND ".join(filters)

            cursor.execute(
                f"SELECT COUNT(*) FROM dbo.readmrz_passport_evaluation_items item WHERE {where_sql};",
                *parameters,
            )
            total = int(cursor.fetchone()[0] or 0)
            cursor.execute(
                f"""
                SELECT
                    item.*,
                    JSON_VALUE(item.SourceDataJson, '$.normalized_passport_fields.source_mrz_checksum_valid')
                        AS SourceChecksumValid
                FROM dbo.readmrz_passport_evaluation_items item
                WHERE {where_sql}
                ORDER BY item.SequenceNo, item.Id
                OFFSET ? ROWS FETCH NEXT ? ROWS ONLY;
                """,
                *parameters,
                offset,
                page_size,
            )
            rows = [_row_dict(cursor, row) for row in cursor.fetchall()]

        return {
            "run": _run_identity_payload(run),
            "summary": summary,
            "items": [_item_payload(row) for row in rows],
            "pagination": _pagination_payload(page, page_size, total),
            "filters": {"review_status": normalized_status, "country": normalized_country},
        }

    def get_item_image(self, item_key: str) -> dict[str, Any]:
        item_uuid = _uuid(item_key, "item_key")
        with connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT run.SourceRoot, item.RelativeImagePath, item.OriginalFileName
                FROM dbo.readmrz_passport_evaluation_items item
                INNER JOIN dbo.readmrz_passport_evaluation_runs run ON run.Id = item.RunId
                WHERE item.ItemKey = ?;
                """,
                str(item_uuid),
            )
            row = cursor.fetchone()
        if row is None:
            raise EvaluationItemNotFoundError(f"Evaluation item not found: {item_key}")
        root = Path(str(row[0])).expanduser().resolve()
        relative = Path(str(row[1]).replace("\\", "/"))
        image_path = (root / relative).resolve()
        try:
            image_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("Stored image path escapes the evaluation source root") from exc
        if not image_path.is_file():
            raise FileNotFoundError(f"Evaluation image not found: {image_path}")
        return {
            "path": image_path,
            "file_name": str(row[2] or image_path.name),
            "content_type": mimetypes.guess_type(image_path.name)[0] or "application/octet-stream",
        }

    def update_review(
        self,
        item_key: str,
        review_status: str,
        *,
        reviewed_by: str | None = None,
        review_note: str | None = None,
        is_poor_quality_image: bool = False,
    ) -> dict[str, Any]:
        item_uuid = _uuid(item_key, "item_key")
        normalized_status = str(review_status or "").strip().lower()
        if normalized_status not in HUMAN_REVIEW_STATUSES:
            raise ValueError("review_status must be approved or rejected")
        reviewer = _optional_text(reviewed_by, 255)
        note = _optional_text(review_note, 2000)

        with connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                UPDATE dbo.readmrz_passport_evaluation_items
                SET ReviewStatus = ?,
                    ReviewBasis = ?,
                    ReviewedBy = ?,
                    ReviewNote = ?,
                    IsPoorQualityImage = ?,
                    ReviewedAt = SYSUTCDATETIME(),
                    ReviewUpdatedAt = SYSUTCDATETIME(),
                    UpdatedDate = SYSUTCDATETIME()
                OUTPUT inserted.RunId
                WHERE ItemKey = ? AND SourceComparedAt IS NOT NULL;
                """,
                normalized_status,
                "human_approved" if normalized_status == "approved" else "human_rejected",
                reviewer,
                note,
                1 if is_poor_quality_image else 0,
                str(item_uuid),
            )
            row = cursor.fetchone()
            if row is None:
                raise EvaluationItemNotFoundError(f"Evaluation item not found: {item_key}")
            run_id = int(row[0])
            connection.commit()

            cursor.execute(
                """
                SELECT item.*,
                       JSON_VALUE(item.SourceDataJson, '$.normalized_passport_fields.source_mrz_checksum_valid')
                           AS SourceChecksumValid
                FROM dbo.readmrz_passport_evaluation_items item
                WHERE item.ItemKey = ?;
                """,
                str(item_uuid),
            )
            item = _item_payload(_row_dict(cursor, cursor.fetchone()))
            summary = self._summary(cursor, run_id)
        return {"status": "ok", "item": item, "summary": summary}

    @staticmethod
    def _get_run(cursor: Any, run_id: int) -> dict[str, Any]:
        cursor.execute(
            """
            SELECT Id, RunKey, RunName, SourceRoot, ReferenceDate, Status, CreatedDate, CompletedAt
            FROM dbo.readmrz_passport_evaluation_runs WHERE Id = ?;
            """,
            run_id,
        )
        row = cursor.fetchone()
        if row is None:
            raise EvaluationRunNotFoundError(f"Evaluation run not found: {run_id}")
        return _row_dict(cursor, row)

    @staticmethod
    def _summary(cursor: Any, run_id: int) -> dict[str, Any]:
        cursor.execute(
            """
            SELECT
                COUNT_BIG(*) AS TotalItems,
                SUM(CASE WHEN ReviewStatus = N'auto_approved' THEN 1 ELSE 0 END) AS AutoApprovedItems,
                SUM(CASE WHEN ReviewStatus = N'pending' THEN 1 ELSE 0 END) AS PendingItems,
                SUM(CASE WHEN ReviewStatus = N'approved' THEN 1 ELSE 0 END) AS ApprovedItems,
                SUM(CASE WHEN ReviewStatus = N'rejected' THEN 1 ELSE 0 END) AS RejectedItems
            FROM dbo.readmrz_passport_evaluation_items
            WHERE RunId = ? AND SourceComparedAt IS NOT NULL;
            """,
            run_id,
        )
        return _summary_payload(_row_dict(cursor, cursor.fetchone()))


FIELD_DEFINITIONS = (
    ("mrz_line_one", "MRZ dòng 1", "MrzLineOne", "SourceMrzLineOne", "MrzLineOneMatch"),
    ("mrz_line_two", "MRZ dòng 2", "MrzLineTwo", "SourceMrzLineTwo", "MrzLineTwoMatch"),
    ("passport_type", "Loại hộ chiếu", "PassportType", "SourcePassportType", "PassportTypeMatch"),
    ("issuing_country", "Quốc gia cấp", "IssuingCountry", "SourceIssuingCountry", "IssuingCountryMatch"),
    ("surname", "Họ", "Surname", "SourceSurname", "SurnameMatch"),
    ("given_names", "Tên", "GivenNames", "SourceGivenNames", "GivenNamesMatch"),
    ("full_name", "Họ và tên", "FullName", "SourceFullName", "FullNameMatch"),
    ("passport_number", "Số hộ chiếu", "PassportNumber", "SourcePassportNumber", "PassportNumberMatch"),
    ("nationality", "Quốc tịch", "Nationality", "SourceNationality", "NationalityMatch"),
    ("date_of_birth", "Ngày sinh", "DateOfBirth", "SourceDateOfBirth", "DateOfBirthMatch"),
    ("sex", "Giới tính", "Sex", "SourceSex", "SexMatch"),
    ("date_of_expiry", "Ngày hết hạn", "DateOfExpiry", "SourceDateOfExpiry", "DateOfExpiryMatch"),
    ("personal_number", "Số cá nhân", "PersonalNumber", "SourcePersonalNumber", "PersonalNumberMatch"),
)


def _item_payload(row: dict[str, Any]) -> dict[str, Any]:
    comparisons = [
        {
            "key": key,
            "label": label,
            "read_value": _json_value(row.get(read_column)),
            "source_value": _json_value(row.get(source_column)),
            "match": _optional_bool(row.get(match_column)),
        }
        for key, label, read_column, source_column, match_column in FIELD_DEFINITIONS
    ]
    source_checksum = str(row.get("SourceChecksumValid") or "").lower() == "true"
    return {
        "item_key": str(row.get("ItemKey")),
        "run_id": int(row.get("RunId") or 0),
        "sequence_no": int(row.get("SequenceNo") or 0),
        "country_code": str(row.get("CountryCode") or ""),
        "source_group": str(row.get("SourceGroup") or ""),
        "file_name": str(row.get("OriginalFileName") or ""),
        "relative_image_path": str(row.get("RelativeImagePath") or ""),
        "image_url": f"/api/passport-evaluation-reviews/items/{row.get('ItemKey')}/image",
        "review_status": str(row.get("ReviewStatus") or "pending"),
        "review_basis": row.get("ReviewBasis"),
        "review_note": row.get("ReviewNote"),
        "reviewed_by": row.get("ReviewedBy"),
        "reviewed_at": _json_value(row.get("ReviewedAt")),
        "is_poor_quality_image": bool(row.get("IsPoorQualityImage") or False),
        "checksum": {
            "read_valid": _optional_bool(row.get("ChecksumValid")),
            "source_valid": source_checksum,
        },
        "comparison": {
            "all_fields_match": _optional_bool(row.get("AllComparedFieldsMatch")),
            "compared_count": int(row.get("ComparedFieldCount") or 0),
            "matched_count": int(row.get("MatchedFieldCount") or 0),
            "fields": comparisons,
        },
        "processing": {
            "status": row.get("Status"),
            "read_success": _optional_bool(row.get("ReadSuccess")),
            "detector_confidence": row.get("YoloConfidence"),
            "ocr_confidence": row.get("OcrConfidence"),
            "process_time_ms": row.get("ProcessTimeMs"),
            "error": row.get("ErrorMessage"),
        },
        "source": {
            "transaction_id": row.get("SourceTransactionEVisaId"),
            "image_path": row.get("SourceFullPassportImage"),
            "match_status": row.get("SourceMatchStatus"),
            "match_reason": row.get("SourceMatchReason"),
        },
    }


def _run_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {**_run_identity_payload(row), "summary": _summary_payload(row)}


def _run_identity_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": int(row.get("Id") or 0),
        "run_key": str(row.get("RunKey") or ""),
        "run_name": row.get("RunName"),
        "source_root": row.get("SourceRoot"),
        "reference_date": _json_value(row.get("ReferenceDate")),
        "status": row.get("Status"),
        "created_at": _json_value(row.get("CreatedDate")),
        "completed_at": _json_value(row.get("CompletedAt")),
    }


def _summary_payload(row: dict[str, Any]) -> dict[str, Any]:
    total = int(row.get("TotalItems") or 0)
    auto = int(row.get("AutoApprovedItems") or 0)
    pending = int(row.get("PendingItems") or 0)
    approved = int(row.get("ApprovedItems") or 0)
    rejected = int(row.get("RejectedItems") or 0)
    accepted = auto + approved
    reviewed = approved + rejected
    return {
        "total": total,
        "auto_approved": auto,
        "pending": pending,
        "approved": approved,
        "rejected": rejected,
        "accepted": accepted,
        "human_reviewed": reviewed,
        "review_progress_percent": round((reviewed * 100 / (pending + reviewed)), 2) if pending + reviewed else 100.0,
    }


def _row_dict(cursor: Any, row: Any) -> dict[str, Any]:
    columns = [str(column[0]) for column in cursor.description]
    return dict(zip(columns, row))


def _pagination(page: int, page_size: int, *, max_page_size: int) -> tuple[int, int, int]:
    page = _positive_int(page, "page")
    page_size = _positive_int(page_size, "page_size")
    page_size = min(page_size, max_page_size)
    return page, page_size, (page - 1) * page_size


def _pagination_payload(page: int, page_size: int, total: int) -> dict[str, int]:
    return {
        "page": page,
        "page_size": page_size,
        "total_items": total,
        "total_pages": (total + page_size - 1) // page_size if total else 0,
    }


def _positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed


def _uuid(value: Any, name: str) -> UUID:
    try:
        return UUID(str(value or ""))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{name} must be a UUID") from exc


def _optional_text(value: Any, max_length: int) -> str | None:
    text = str(value or "").strip()
    return text[:max_length] if text else None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _json_value(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    return value
