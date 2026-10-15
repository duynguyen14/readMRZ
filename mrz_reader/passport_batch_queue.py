from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from email import policy
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import socket
import threading
import time
from typing import Any, Callable
from uuid import UUID, uuid4

import cv2

from .db import connect
from .env_config import env_value, read_env_file


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ITEM_STATUSES = {"queued", "processing", "retry_wait", "succeeded", "failed", "cancelled"}
REVIEW_STATUSES = {"not_ready", "pending", "approved", "corrected", "rejected"}
BATCH_STATUSES = {
    "receiving",
    "queued",
    "processing",
    "completed",
    "completed_with_errors",
    "failed",
    "cancelled",
}


class BatchNotFoundError(LookupError):
    pass


class ItemNotFoundError(LookupError):
    pass


class ReviewConflictError(RuntimeError):
    pass


@dataclass(frozen=True)
class UploadedFile:
    file_name: str
    content_type: str
    data: bytes


@dataclass(frozen=True)
class ClaimedItem:
    id: int
    item_key: str
    batch_id: int
    relative_image_path: str
    attempt_count: int
    max_attempts: int


def parse_multipart_form(content_type: str, body: bytes) -> tuple[list[UploadedFile], dict[str, str]]:
    if "multipart/form-data" not in content_type.lower():
        raise ValueError("Content-Type must be multipart/form-data")

    message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: " + content_type.encode("latin-1") + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
    )
    if not message.is_multipart():
        raise ValueError("Invalid multipart/form-data body")

    files: list[UploadedFile] = []
    fields: dict[str, str] = {}
    for part in message.iter_parts():
        field_name = str(part.get_param("name", header="content-disposition") or "").strip()
        file_name = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if file_name is not None:
            safe_name = Path(str(file_name).replace("\\", "/")).name.strip()
            if not safe_name:
                raise ValueError("Every uploaded image must have a file name")
            if not payload:
                raise ValueError(f"Uploaded image is empty: {safe_name}")
            files.append(
                UploadedFile(
                    file_name=safe_name,
                    content_type=str(part.get_content_type() or "application/octet-stream"),
                    data=payload,
                )
            )
        elif field_name:
            charset = part.get_content_charset() or "utf-8"
            fields[field_name] = payload.decode(charset, errors="replace").strip()

    if not files:
        raise ValueError("At least one image file is required")
    return files, fields


class PassportBatchRepository:
    def __init__(self, storage_root: Path, *, max_attempts: int = 3) -> None:
        self.storage_root = storage_root.expanduser().resolve()
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.max_attempts = max(1, int(max_attempts))

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "PassportBatchRepository":
        values = env or read_env_file()
        default_root = Path(__file__).resolve().parents[1] / "tmp" / "passport_batches"
        storage_root = Path(env_value(values, "READMRZ_PASSPORT_BATCH_STORAGE_DIR", str(default_root)))
        max_attempts = int(env_value(values, "READMRZ_PASSPORT_BATCH_MAX_ATTEMPTS", "3"))
        return cls(storage_root, max_attempts=max_attempts)

    def create_batch(
        self,
        files: list[UploadedFile],
        *,
        client_request_id: str | None = None,
        source_system: str | None = None,
        requested_by: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not files:
            raise ValueError("At least one image file is required")

        batch_key = uuid4()
        batch_directory = self.storage_root / str(batch_key)
        batch_directory.mkdir(parents=True, exist_ok=False)
        saved_items: list[dict[str, Any]] = []

        try:
            for sequence_no, uploaded in enumerate(files, start=1):
                suffix = _safe_image_suffix(uploaded.file_name, uploaded.content_type)
                item_key = uuid4()
                stored_name = f"{sequence_no:06d}_{item_key}{suffix}"
                target_path = batch_directory / stored_name
                temporary_path = target_path.with_suffix(target_path.suffix + ".part")
                temporary_path.write_bytes(uploaded.data)
                temporary_path.replace(target_path)
                saved_items.append(
                    {
                        "item_key": item_key,
                        "sequence_no": sequence_no,
                        "original_file_name": uploaded.file_name,
                        "relative_path": f"{batch_key}/{stored_name}",
                        "content_type": uploaded.content_type,
                        "file_size": len(uploaded.data),
                        "sha256": hashlib.sha256(uploaded.data).hexdigest(),
                    }
                )

            with connect() as connection:
                cursor = connection.cursor()
                cursor.execute(
                    """
                    INSERT INTO dbo.readmrz_passport_batches (
                        BatchKey, ClientRequestId, SourceSystem, RequestedBy,
                        Status, UploadCompleted, TotalItems, QueuedItems,
                        StorageDirectory, MetadataJson, UploadCompletedAt
                    )
                    OUTPUT inserted.Id
                    VALUES (?, ?, ?, ?, N'queued', 1, ?, ?, ?, ?, SYSUTCDATETIME())
                    """,
                    str(batch_key),
                    _optional_text(client_request_id, 128),
                    _optional_text(source_system, 128),
                    _optional_text(requested_by, 128),
                    len(saved_items),
                    len(saved_items),
                    str(batch_key),
                    _json_dump(metadata) if metadata else None,
                )
                batch_id = int(cursor.fetchone()[0])
                cursor.fast_executemany = True
                cursor.executemany(
                    """
                    INSERT INTO dbo.readmrz_passport_items (
                        ItemKey, BatchId, SequenceNo, OriginalFileName,
                        RelativeImagePath, ContentType, FileSizeBytes, Sha256,
                        Status, MaxAttempts
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, N'queued', ?)
                    """,
                    [
                        (
                            str(item["item_key"]),
                            batch_id,
                            item["sequence_no"],
                            item["original_file_name"],
                            item["relative_path"],
                            item["content_type"],
                            item["file_size"],
                            item["sha256"],
                            self.max_attempts,
                        )
                        for item in saved_items
                    ],
                )
                connection.commit()
        except Exception:
            _remove_batch_directory(batch_directory)
            raise

        return {
            "batch_id": str(batch_key),
            "received": len(saved_items),
            "status": "queued",
        }

    def list_batches(
        self,
        *,
        status: str | None = None,
        keyword: str | None = None,
        page: int = 1,
        page_size: int = 20,
        sort: str = "created_desc",
    ) -> dict[str, Any]:
        normalized_status = _validated_filter(status, BATCH_STATUSES, "status")
        page, page_size, offset = _pagination(page, page_size)
        sort_sql = {
            "created_desc": "batch.CreatedDate DESC, batch.Id DESC",
            "created_asc": "batch.CreatedDate ASC, batch.Id ASC",
            "updated_desc": "batch.UpdatedDate DESC, batch.Id DESC",
        }.get(sort)
        if sort_sql is None:
            raise ValueError("sort must be created_desc, created_asc or updated_desc")

        filters: list[str] = []
        parameters: list[Any] = []
        if normalized_status:
            filters.append("batch.Status = ?")
            parameters.append(normalized_status)
        normalized_keyword = str(keyword or "").strip()
        if normalized_keyword:
            filters.append(
                "(CONVERT(NVARCHAR(36), batch.BatchKey) LIKE ? "
                "OR batch.ClientRequestId LIKE ? OR batch.SourceSystem LIKE ? OR batch.RequestedBy LIKE ?)"
            )
            pattern = f"%{normalized_keyword}%"
            parameters.extend([pattern, pattern, pattern, pattern])
        where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""

        with connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"SELECT COUNT(*) FROM dbo.readmrz_passport_batches batch {where_sql}",
                *parameters,
            )
            total = int(cursor.fetchone()[0])
            cursor.execute(
                f"""
                SELECT
                    batch.Id,
                    batch.BatchKey,
                    batch.ClientRequestId,
                    batch.SourceSystem,
                    batch.RequestedBy,
                    batch.Status,
                    batch.UploadCompleted,
                    batch.CreatedDate,
                    batch.UpdatedDate,
                    batch.UploadCompletedAt,
                    batch.ProcessStartedAt,
                    batch.CompletedAt,
                    batch.ErrorMessage,
                    COUNT(item.Id) AS TotalItems,
                    SUM(CASE WHEN item.Status = N'queued' THEN 1 ELSE 0 END) AS QueuedItems,
                    SUM(CASE WHEN item.Status = N'processing' THEN 1 ELSE 0 END) AS ProcessingItems,
                    SUM(CASE WHEN item.Status = N'retry_wait' THEN 1 ELSE 0 END) AS RetryWaitItems,
                    SUM(CASE WHEN item.Status = N'succeeded' THEN 1 ELSE 0 END) AS SucceededItems,
                    SUM(CASE WHEN item.Status = N'failed' THEN 1 ELSE 0 END) AS FailedItems,
                    SUM(CASE WHEN item.Status = N'cancelled' THEN 1 ELSE 0 END) AS CancelledItems,
                    SUM(CASE WHEN item.ReviewStatus = N'not_ready' THEN 1 ELSE 0 END) AS NotReadyItems,
                    SUM(CASE WHEN item.ReviewStatus = N'pending' THEN 1 ELSE 0 END) AS PendingReviewItems,
                    SUM(CASE WHEN item.ReviewStatus = N'approved' THEN 1 ELSE 0 END) AS ApprovedItems,
                    SUM(CASE WHEN item.ReviewStatus = N'corrected' THEN 1 ELSE 0 END) AS CorrectedItems,
                    SUM(CASE WHEN item.ReviewStatus = N'rejected' THEN 1 ELSE 0 END) AS RejectedItems
                FROM dbo.readmrz_passport_batches batch
                LEFT JOIN dbo.readmrz_passport_items item ON item.BatchId = batch.Id
                {where_sql}
                GROUP BY
                    batch.Id, batch.BatchKey, batch.ClientRequestId, batch.SourceSystem,
                    batch.RequestedBy, batch.Status, batch.UploadCompleted,
                    batch.CreatedDate, batch.UpdatedDate, batch.UploadCompletedAt,
                    batch.ProcessStartedAt, batch.CompletedAt, batch.ErrorMessage
                ORDER BY {sort_sql}
                OFFSET ? ROWS FETCH NEXT ? ROWS ONLY
                """,
                *parameters,
                offset,
                page_size,
            )
            rows = [_row_dict(cursor, row) for row in cursor.fetchall()]

        return {
            "items": [_batch_payload(row) for row in rows],
            "pagination": _pagination_payload(page, page_size, total),
            "total_batches": total,
        }

    def list_items(
        self,
        batch_key: str,
        *,
        status: str | None = None,
        review_status: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        batch_uuid = _normalize_uuid(batch_key, "batch_id")
        normalized_status = _validated_filter(status, ITEM_STATUSES, "status")
        normalized_review_status = _validated_filter(
            review_status, REVIEW_STATUSES, "review_status"
        )
        page, page_size, offset = _pagination(page, page_size)

        with connect() as connection:
            cursor = connection.cursor()
            batch_row = self._get_batch_row(cursor, batch_uuid)
            batch_id = int(batch_row["Id"])
            statistics = self._get_statistics(cursor, batch_id)

            filters = ["item.BatchId = ?"]
            parameters: list[Any] = [batch_id]
            if normalized_status:
                filters.append("item.Status = ?")
                parameters.append(normalized_status)
            if normalized_review_status:
                filters.append("item.ReviewStatus = ?")
                parameters.append(normalized_review_status)
            where_sql = " AND ".join(filters)

            cursor.execute(
                f"SELECT COUNT(*) FROM dbo.readmrz_passport_items item WHERE {where_sql}",
                *parameters,
            )
            total = int(cursor.fetchone()[0])
            cursor.execute(
                f"""
                SELECT {_item_select_columns()}
                FROM dbo.readmrz_passport_items item
                WHERE {where_sql}
                ORDER BY item.SequenceNo ASC, item.Id ASC
                OFFSET ? ROWS FETCH NEXT ? ROWS ONLY
                """,
                *parameters,
                offset,
                page_size,
            )
            items = [_item_payload(_row_dict(cursor, row), include_result=True) for row in cursor.fetchall()]

        return {
            "batch": _batch_identity_payload(batch_row),
            "statistics": statistics,
            "items": items,
            "pagination": _pagination_payload(page, page_size, total),
        }

    def get_review_item(
        self,
        batch_key: str,
        *,
        direction: str = "next",
        current_item_key: str | None = None,
    ) -> dict[str, Any]:
        if direction not in {"next", "previous"}:
            raise ValueError("direction must be next or previous")
        batch_uuid = _normalize_uuid(batch_key, "batch_id")

        with connect() as connection:
            cursor = connection.cursor()
            batch_row = self._get_batch_row(cursor, batch_uuid)
            batch_id = int(batch_row["Id"])
            statistics = self._get_statistics(cursor, batch_id)

            cursor_values: tuple[datetime, int] | None = None
            if current_item_key:
                item_uuid = _normalize_uuid(current_item_key, "item_key")
                cursor.execute(
                    """
                    SELECT ProcessedAt, Id
                    FROM dbo.readmrz_passport_items
                    WHERE BatchId = ? AND ItemKey = ?
                    """,
                    batch_id,
                    str(item_uuid),
                )
                current_row = cursor.fetchone()
                if current_row is None:
                    raise ItemNotFoundError(f"Item not found in batch: {item_uuid}")
                processed_at = current_row[0] or datetime.min
                cursor_values = (processed_at, int(current_row[1]))

            comparison_sql = ""
            parameters: list[Any] = [batch_id]
            order_sql = "item.ProcessedAt DESC, item.Id DESC"
            if cursor_values is not None:
                processed_at, item_id = cursor_values
                if direction == "next":
                    comparison_sql = (
                        "AND (item.ProcessedAt < ? OR (item.ProcessedAt = ? AND item.Id < ?))"
                    )
                    parameters.extend([processed_at, processed_at, item_id])
                else:
                    comparison_sql = (
                        "AND (item.ProcessedAt > ? OR (item.ProcessedAt = ? AND item.Id > ?))"
                    )
                    parameters.extend([processed_at, processed_at, item_id])
                    order_sql = "item.ProcessedAt ASC, item.Id ASC"
            elif direction == "previous":
                order_sql = "item.ProcessedAt ASC, item.Id ASC"

            cursor.execute(
                f"""
                SELECT TOP (1) {_item_select_columns()}
                FROM dbo.readmrz_passport_items item
                WHERE item.BatchId = ?
                  AND item.Status = N'succeeded'
                  AND item.ReviewStatus = N'pending'
                  {comparison_sql}
                ORDER BY {order_sql}
                """,
                *parameters,
            )
            row = cursor.fetchone()
            item = _item_payload(_row_dict(cursor, row), include_result=True) if row else None

        return {
            "batch": _batch_identity_payload(batch_row),
            "statistics": statistics,
            "direction": direction,
            "item": item,
        }

    def get_item_image(self, item_key: str) -> dict[str, Any]:
        item_uuid = _normalize_uuid(item_key, "item_key")
        with connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT ItemKey, OriginalFileName, RelativeImagePath, ContentType
                FROM dbo.readmrz_passport_items
                WHERE ItemKey = ?
                """,
                str(item_uuid),
            )
            row = cursor.fetchone()
            if row is None:
                raise ItemNotFoundError(f"Passport item not found: {item_uuid}")
            values = _row_dict(cursor, row)

        image_path = self.absolute_image_path(str(values["RelativeImagePath"]))
        if not image_path.is_file():
            raise FileNotFoundError(f"Stored passport image does not exist: {image_path}")
        return {
            "path": image_path,
            "file_name": str(values.get("OriginalFileName") or image_path.name),
            "content_type": str(values.get("ContentType") or "application/octet-stream"),
        }

    def update_review_status(
        self,
        item_identifier: str | int,
        review_status: str,
        *,
        reviewed_by: str | None = None,
        review_note: str | None = None,
    ) -> dict[str, Any]:
        normalized_status = str(review_status or "").strip().lower()
        allowed_decisions = {"approved", "corrected", "rejected"}
        if normalized_status not in allowed_decisions:
            raise ValueError("review_status must be approved, corrected or rejected")

        lookup_sql, lookup_value = _item_identifier_lookup(item_identifier)
        with connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"""
                SELECT item.Id, item.ItemKey, item.BatchId, item.Status
                FROM dbo.readmrz_passport_items item WITH (UPDLOCK, ROWLOCK)
                WHERE {lookup_sql}
                """,
                lookup_value,
            )
            identity_row = cursor.fetchone()
            if identity_row is None:
                raise ItemNotFoundError(f"Passport item not found: {item_identifier}")
            item_id = int(identity_row[0])
            batch_id = int(identity_row[2])
            processing_status = str(identity_row[3] or "")
            if processing_status != "succeeded":
                raise ReviewConflictError(
                    "Only items with processing status succeeded can be reviewed"
                )

            cursor.execute(
                """
                UPDATE dbo.readmrz_passport_items
                SET ReviewStatus = ?,
                    ReviewedBy = COALESCE(?, ReviewedBy),
                    ReviewNote = COALESCE(?, ReviewNote),
                    ReviewedAt = SYSUTCDATETIME(),
                    ReviewUpdatedDate = SYSUTCDATETIME(),
                    UpdatedDate = SYSUTCDATETIME()
                WHERE Id = ? AND Status = N'succeeded';
                """,
                normalized_status,
                _optional_text(reviewed_by, 128),
                _optional_text(review_note, 2000),
                item_id,
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Review status update failed for item: {item_identifier}")

            cursor.execute(
                """
                UPDATE dbo.readmrz_passport_batches
                SET UpdatedDate = SYSUTCDATETIME()
                WHERE Id = ?;
                """,
                batch_id,
            )

            cursor.execute(
                f"""
                SELECT {_item_select_columns()}
                FROM dbo.readmrz_passport_items item
                WHERE item.Id = ?
                """,
                item_id,
            )
            updated_item = _item_payload(_row_dict(cursor, cursor.fetchone()), include_result=True)
            cursor.execute(
                """
                SELECT Id, BatchKey, ClientRequestId, SourceSystem, RequestedBy, Status,
                       UploadCompleted, CreatedDate, UpdatedDate, UploadCompletedAt,
                       ProcessStartedAt, CompletedAt, ErrorMessage
                FROM dbo.readmrz_passport_batches
                WHERE Id = ?
                """,
                batch_id,
            )
            batch_row = _row_dict(cursor, cursor.fetchone())
            statistics = self._get_statistics(cursor, batch_id)
            connection.commit()

        return {
            "batch": _batch_identity_payload(batch_row),
            "statistics": statistics,
            "item": updated_item,
        }

    @staticmethod
    def _get_batch_row(cursor: Any, batch_key: UUID) -> dict[str, Any]:
        cursor.execute(
            """
            SELECT Id, BatchKey, ClientRequestId, SourceSystem, RequestedBy, Status,
                   UploadCompleted, CreatedDate, UpdatedDate, UploadCompletedAt,
                   ProcessStartedAt, CompletedAt, ErrorMessage
            FROM dbo.readmrz_passport_batches
            WHERE BatchKey = ?
            """,
            str(batch_key),
        )
        row = cursor.fetchone()
        if row is None:
            raise BatchNotFoundError(f"Passport batch not found: {batch_key}")
        return _row_dict(cursor, row)

    @staticmethod
    def _get_statistics(cursor: Any, batch_id: int) -> dict[str, Any]:
        cursor.execute(
            """
            SELECT
                COUNT(*) AS TotalItems,
                SUM(CASE WHEN Status = N'queued' THEN 1 ELSE 0 END) AS QueuedItems,
                SUM(CASE WHEN Status = N'processing' THEN 1 ELSE 0 END) AS ProcessingItems,
                SUM(CASE WHEN Status = N'retry_wait' THEN 1 ELSE 0 END) AS RetryWaitItems,
                SUM(CASE WHEN Status = N'succeeded' THEN 1 ELSE 0 END) AS SucceededItems,
                SUM(CASE WHEN Status = N'failed' THEN 1 ELSE 0 END) AS FailedItems,
                SUM(CASE WHEN Status = N'cancelled' THEN 1 ELSE 0 END) AS CancelledItems,
                SUM(CASE WHEN ReviewStatus = N'not_ready' THEN 1 ELSE 0 END) AS NotReadyItems,
                SUM(CASE WHEN ReviewStatus = N'pending' THEN 1 ELSE 0 END) AS PendingReviewItems,
                SUM(CASE WHEN ReviewStatus = N'approved' THEN 1 ELSE 0 END) AS ApprovedItems,
                SUM(CASE WHEN ReviewStatus = N'corrected' THEN 1 ELSE 0 END) AS CorrectedItems,
                SUM(CASE WHEN ReviewStatus = N'rejected' THEN 1 ELSE 0 END) AS RejectedItems
            FROM dbo.readmrz_passport_items
            WHERE BatchId = ?
            """,
            batch_id,
        )
        return _statistics_payload(_row_dict(cursor, cursor.fetchone()))

    def claim_next(self, worker_id: str) -> ClaimedItem | None:
        with connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                ;WITH next_item AS (
                    SELECT TOP (1) *
                    FROM dbo.readmrz_passport_items WITH (UPDLOCK, READPAST, ROWLOCK)
                    WHERE Status IN (N'queued', N'retry_wait')
                      AND (NextAttemptAt IS NULL OR NextAttemptAt <= SYSUTCDATETIME())
                      AND AttemptCount < MaxAttempts
                    ORDER BY CASE WHEN Status = N'queued' THEN 0 ELSE 1 END, Id
                )
                UPDATE next_item
                SET Status = N'processing',
                    AttemptCount = AttemptCount + 1,
                    NextAttemptAt = NULL,
                    LockedBy = ?,
                    LockedAt = SYSUTCDATETIME(),
                    ProcessStartedAt = COALESCE(ProcessStartedAt, SYSUTCDATETIME()),
                    UpdatedDate = SYSUTCDATETIME()
                OUTPUT inserted.Id, inserted.ItemKey, inserted.BatchId,
                       inserted.RelativeImagePath, inserted.AttemptCount, inserted.MaxAttempts;
                """,
                worker_id,
            )
            row = cursor.fetchone()
            if row is None:
                connection.commit()
                return None
            claimed = ClaimedItem(
                id=int(row[0]),
                item_key=str(row[1]),
                batch_id=int(row[2]),
                relative_image_path=str(row[3]),
                attempt_count=int(row[4]),
                max_attempts=int(row[5]),
            )
            self._refresh_batch(cursor, claimed.batch_id)
            connection.commit()
            return claimed

    def recover_stale_items(self, lock_timeout_seconds: int) -> int:
        cutoff_seconds = max(60, int(lock_timeout_seconds))
        with connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                UPDATE dbo.readmrz_passport_items
                SET Status = CASE WHEN AttemptCount < MaxAttempts THEN N'retry_wait' ELSE N'failed' END,
                    NextAttemptAt = CASE
                        WHEN AttemptCount < MaxAttempts THEN SYSUTCDATETIME()
                        ELSE NULL
                    END,
                    LockedBy = NULL,
                    LockedAt = NULL,
                    ErrorCode = N'WORKER_LOCK_TIMEOUT',
                    ErrorMessage = N'Worker lock expired before processing completed.',
                    ProcessedAt = CASE
                        WHEN AttemptCount >= MaxAttempts THEN SYSUTCDATETIME()
                        ELSE ProcessedAt
                    END,
                    UpdatedDate = SYSUTCDATETIME()
                WHERE Status = N'processing'
                  AND LockedAt < DATEADD(SECOND, ?, SYSUTCDATETIME());
                """,
                -cutoff_seconds,
            )
            recovered = int(cursor.rowcount if cursor.rowcount >= 0 else 0)
            if recovered:
                cursor.execute(
                    "SELECT DISTINCT BatchId FROM dbo.readmrz_passport_items WHERE ErrorCode = N'WORKER_LOCK_TIMEOUT'"
                )
                for row in cursor.fetchall():
                    self._refresh_batch(cursor, int(row[0]))
            connection.commit()
            return recovered

    def mark_succeeded(
        self,
        item: ClaimedItem,
        *,
        result: dict[str, Any],
        debug: dict[str, Any],
        image_width: int,
        image_height: int,
        process_time_ms: int,
    ) -> None:
        fields = result.get("fields") or {}
        lines = result.get("lines") or []
        line_values = [str(line.get("text") or "") for line in lines if isinstance(line, dict)]
        if not line_values:
            line_values = [str(value or "") for value in (debug.get("mrz_ocr") or {}).get("mrz_raw") or []]
        surname = _optional_text(fields.get("surname"), 255)
        given_names = _optional_text(fields.get("given_names"), 255)
        full_name = " ".join(value for value in (surname, given_names) if value) or None
        orientation = debug.get("orientation") or {}
        fallback = result.get("fallback") or {}
        orientation_degree = (
            _right_angle(orientation.get("applied_angle"))
            + _right_angle(debug.get("selected_rotation_angle"))
        ) % 360
        detector_confidence = result.get("detector_confidence")
        if not detector_confidence:
            detector_confidence = (debug.get("best_box") or {}).get("confidence")

        with connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                UPDATE dbo.readmrz_passport_items
                SET Status = N'succeeded',
                    ReviewStatus = CASE
                        WHEN ReviewStatus = N'not_ready' THEN N'pending'
                        ELSE ReviewStatus
                    END,
                    ReviewUpdatedDate = CASE
                        WHEN ReviewStatus = N'not_ready' THEN SYSUTCDATETIME()
                        ELSE ReviewUpdatedDate
                    END,
                    ImageWidth = ?, ImageHeight = ?,
                    OrientationDegree = ?, YoloConfidence = ?, OcrConfidence = ?,
                    UsedFallback = ?, ProcessTimeMs = ?,
                    MrzLineOne = ?, MrzLineTwo = ?, PassportType = ?,
                    IssuingCountry = ?, Surname = ?, GivenNames = ?, FullName = ?,
                    PassportNumber = ?, Nationality = ?, DateOfBirth = ?, Sex = ?,
                    DateOfExpiry = ?, PersonalNumber = ?, ChecksumValid = ?,
                    ResultJson = ?, DebugJson = ?, ErrorCode = NULL,
                    ErrorMessage = ?, NextAttemptAt = NULL, LockedBy = NULL, LockedAt = NULL,
                    ProcessedAt = SYSUTCDATETIME(), UpdatedDate = SYSUTCDATETIME()
                WHERE Id = ? AND Status = N'processing';
                """,
                image_width,
                image_height,
                orientation_degree,
                _optional_float(detector_confidence),
                _optional_float(result.get("average_ocr_confidence")),
                1 if fallback.get("used") else 0,
                max(0, int(process_time_ms)),
                _optional_text(line_values[0] if len(line_values) > 0 else None, 128),
                _optional_text(line_values[1] if len(line_values) > 1 else None, 128),
                _optional_text(result.get("document_type"), 20),
                _optional_text(fields.get("issuing_country"), 20),
                surname,
                given_names,
                _optional_text(full_name, 512),
                _optional_text(fields.get("document_number"), 50),
                _optional_text(fields.get("nationality"), 20),
                _parse_iso_date(fields.get("birth_date")),
                _optional_text(fields.get("sex"), 20),
                _parse_iso_date(fields.get("expiry_date")),
                _optional_text(fields.get("optional_data"), 100),
                1 if result.get("checksum_pass") else 0,
                _json_dump(result),
                _json_dump(debug),
                _optional_text(result.get("error"), 2000),
                item.id,
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Passport item {item.id} is no longer locked for processing")
            self._refresh_batch(cursor, item.batch_id)
            connection.commit()

    def mark_failed(self, item: ClaimedItem, exc: Exception, retry_delay_seconds: int) -> str:
        will_retry = item.attempt_count < item.max_attempts
        status = "retry_wait" if will_retry else "failed"
        error_code = type(exc).__name__[:100]
        error_message = str(exc)[:2000] or error_code
        with connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                UPDATE dbo.readmrz_passport_items
                SET Status = ?,
                    NextAttemptAt = CASE
                        WHEN ? = 1 THEN DATEADD(SECOND, ?, SYSUTCDATETIME())
                        ELSE NULL
                    END,
                    LockedBy = NULL, LockedAt = NULL,
                    ErrorCode = ?, ErrorMessage = ?,
                    ProcessedAt = CASE WHEN ? = 1 THEN ProcessedAt ELSE SYSUTCDATETIME() END,
                    UpdatedDate = SYSUTCDATETIME()
                WHERE Id = ? AND Status = N'processing';
                """,
                status,
                1 if will_retry else 0,
                max(1, int(retry_delay_seconds)),
                error_code,
                error_message,
                1 if will_retry else 0,
                item.id,
            )
            self._refresh_batch(cursor, item.batch_id)
            connection.commit()
        return status

    def absolute_image_path(self, relative_path: str) -> Path:
        normalized = Path(str(relative_path).replace("\\", "/"))
        candidate = (self.storage_root / normalized).resolve()
        try:
            candidate.relative_to(self.storage_root)
        except ValueError as exc:
            raise ValueError(f"Stored image path escapes the configured root: {relative_path}") from exc
        return candidate

    @staticmethod
    def _refresh_batch(cursor: Any, batch_id: int) -> None:
        cursor.execute(
            """
            ;WITH counts AS (
                SELECT
                    COUNT(*) AS TotalItems,
                    SUM(CASE WHEN Status IN (N'queued', N'retry_wait') THEN 1 ELSE 0 END) AS QueuedItems,
                    SUM(CASE WHEN Status = N'processing' THEN 1 ELSE 0 END) AS ProcessingItems,
                    SUM(CASE WHEN Status = N'succeeded' THEN 1 ELSE 0 END) AS SucceededItems,
                    SUM(CASE WHEN Status = N'failed' THEN 1 ELSE 0 END) AS FailedItems,
                    SUM(CASE WHEN Status = N'cancelled' THEN 1 ELSE 0 END) AS CancelledItems
                FROM dbo.readmrz_passport_items
                WHERE BatchId = ?
            )
            UPDATE batch
            SET TotalItems = counts.TotalItems,
                QueuedItems = counts.QueuedItems,
                ProcessingItems = counts.ProcessingItems,
                SucceededItems = counts.SucceededItems,
                FailedItems = counts.FailedItems,
                CancelledItems = counts.CancelledItems,
                Status = CASE
                    WHEN batch.UploadCompleted = 0 THEN N'receiving'
                    WHEN counts.TotalItems = 0 THEN N'failed'
                    WHEN counts.QueuedItems + counts.ProcessingItems > 0
                        THEN CASE
                            WHEN counts.ProcessingItems > 0 OR counts.SucceededItems + counts.FailedItems > 0
                                THEN N'processing'
                            ELSE N'queued'
                        END
                    WHEN counts.FailedItems = 0 AND counts.CancelledItems = 0 THEN N'completed'
                    WHEN counts.SucceededItems = 0 THEN N'failed'
                    ELSE N'completed_with_errors'
                END,
                ProcessStartedAt = CASE
                    WHEN counts.ProcessingItems + counts.SucceededItems + counts.FailedItems > 0
                        THEN COALESCE(batch.ProcessStartedAt, SYSUTCDATETIME())
                    ELSE batch.ProcessStartedAt
                END,
                CompletedAt = CASE
                    WHEN counts.TotalItems > 0
                     AND counts.QueuedItems + counts.ProcessingItems = 0
                        THEN COALESCE(batch.CompletedAt, SYSUTCDATETIME())
                    ELSE NULL
                END,
                UpdatedDate = SYSUTCDATETIME()
            FROM dbo.readmrz_passport_batches batch
            CROSS JOIN counts
            WHERE batch.Id = ?;
            """,
            batch_id,
            batch_id,
        )


class PassportBatchWorker:
    def __init__(
        self,
        repository: PassportBatchRepository,
        processor: Callable[[Any], tuple[dict[str, Any], dict[str, Any]]],
        *,
        logger: Callable[[str], None],
        poll_seconds: float = 1.0,
        retry_delay_seconds: int = 30,
        lock_timeout_seconds: int = 900,
    ) -> None:
        self.repository = repository
        self.processor = processor
        self.logger = logger
        self.poll_seconds = max(0.1, float(poll_seconds))
        self.retry_delay_seconds = max(1, int(retry_delay_seconds))
        self.lock_timeout_seconds = max(60, int(lock_timeout_seconds))
        self.worker_id = f"{socket.gethostname()}-{uuid4()}"
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_recovery_at = 0.0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="readmrz-passport-batch-worker",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop_event.set()
        self._wake_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.0, timeout))

    def wake(self) -> None:
        self._wake_event.set()

    def _run(self) -> None:
        self.logger(f"Passport batch worker started worker_id={self.worker_id}")
        self._recover_stale_items_if_due()

        while not self._stop_event.is_set():
            self._recover_stale_items_if_due()
            try:
                item = self.repository.claim_next(self.worker_id)
            except Exception as exc:
                self.logger(f"Passport batch worker claim error {exc}")
                self._wait()
                continue

            if item is None:
                self._wait()
                continue

            started = time.perf_counter()
            try:
                image_path = self.repository.absolute_image_path(item.relative_image_path)
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Cannot decode stored image: {image_path}")
                height, width = image.shape[:2]
                result, debug = self.processor(image)
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                self.repository.mark_succeeded(
                    item,
                    result=result,
                    debug=debug,
                    image_width=int(width),
                    image_height=int(height),
                    process_time_ms=elapsed_ms,
                )
                self.logger(
                    "Passport batch item done "
                    f"item_id={item.id} batch_id={item.batch_id} found={result.get('found')} "
                    f"checksum={result.get('checksum_pass')} latency_ms={elapsed_ms}"
                )
            except Exception as exc:
                try:
                    status = self.repository.mark_failed(item, exc, self.retry_delay_seconds)
                except Exception as update_exc:
                    self.logger(
                        f"Passport batch item state update error item_id={item.id} "
                        f"processing_error={exc} update_error={update_exc}"
                    )
                else:
                    self.logger(
                        f"Passport batch item error item_id={item.id} status={status} "
                        f"attempt={item.attempt_count}/{item.max_attempts} error={exc}"
                    )

        self.logger(f"Passport batch worker stopped worker_id={self.worker_id}")

    def _wait(self) -> None:
        self._wake_event.wait(self.poll_seconds)
        self._wake_event.clear()

    def _recover_stale_items_if_due(self) -> None:
        now = time.monotonic()
        recovery_interval = min(60.0, max(10.0, self.lock_timeout_seconds / 3.0))
        if now - self._last_recovery_at < recovery_interval:
            return
        self._last_recovery_at = now
        try:
            recovered = self.repository.recover_stale_items(self.lock_timeout_seconds)
            if recovered:
                self.logger(f"Passport batch worker recovered stale_items={recovered}")
        except Exception as exc:
            self.logger(f"Passport batch worker stale recovery error {exc}")


def _safe_image_suffix(file_name: str, content_type: str) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix in SUPPORTED_IMAGE_EXTENSIONS:
        return ".jpg" if suffix == ".jpeg" else suffix
    mime_suffix = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
        "image/webp": ".webp",
    }.get(content_type.lower())
    if mime_suffix:
        return mime_suffix
    raise ValueError(f"Unsupported image type: {file_name}")


def _item_select_columns() -> str:
    return """
        item.Id,
        item.ItemKey,
        item.BatchId,
        item.SequenceNo,
        item.OriginalFileName,
        item.ContentType,
        item.FileSizeBytes,
        item.ImageWidth,
        item.ImageHeight,
        item.Status,
        item.AttemptCount,
        item.MaxAttempts,
        item.OrientationDegree,
        item.YoloConfidence,
        item.OcrConfidence,
        item.UsedFallback,
        item.ProcessTimeMs,
        item.MrzLineOne,
        item.MrzLineTwo,
        item.PassportType,
        item.IssuingCountry,
        item.Surname,
        item.GivenNames,
        item.FullName,
        item.PassportNumber,
        item.Nationality,
        item.DateOfBirth,
        item.Sex,
        item.DateOfExpiry,
        item.PersonalNumber,
        item.ChecksumValid,
        item.ResultJson,
        item.ErrorCode,
        item.ErrorMessage,
        item.ReviewStatus,
        item.ReviewedResultJson,
        item.ReviewNote,
        item.ReviewedBy,
        item.ReviewedAt,
        item.ReviewUpdatedDate,
        item.ProcessStartedAt,
        item.ProcessedAt,
        item.CreatedDate,
        item.UpdatedDate
    """


def _row_dict(cursor: Any, row: Any) -> dict[str, Any]:
    columns = [str(column[0]) for column in cursor.description]
    return dict(zip(columns, row))


def _batch_identity_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "batch_id": str(row.get("BatchKey") or ""),
        "client_request_id": row.get("ClientRequestId"),
        "source_system": row.get("SourceSystem"),
        "requested_by": row.get("RequestedBy"),
        "status": row.get("Status"),
        "upload_completed": bool(row.get("UploadCompleted")),
        "created_at": _iso_value(row.get("CreatedDate")),
        "updated_at": _iso_value(row.get("UpdatedDate")),
        "upload_completed_at": _iso_value(row.get("UploadCompletedAt")),
        "process_started_at": _iso_value(row.get("ProcessStartedAt")),
        "completed_at": _iso_value(row.get("CompletedAt")),
        "error": row.get("ErrorMessage"),
    }


def _batch_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = _batch_identity_payload(row)
    payload["statistics"] = _statistics_payload(row)
    return payload


def _statistics_payload(row: dict[str, Any]) -> dict[str, Any]:
    total = _int_value(row.get("TotalItems"))
    queued = _int_value(row.get("QueuedItems"))
    processing = _int_value(row.get("ProcessingItems"))
    retry_wait = _int_value(row.get("RetryWaitItems"))
    succeeded = _int_value(row.get("SucceededItems"))
    failed = _int_value(row.get("FailedItems"))
    cancelled = _int_value(row.get("CancelledItems"))
    not_ready = _int_value(row.get("NotReadyItems"))
    pending = _int_value(row.get("PendingReviewItems"))
    approved = _int_value(row.get("ApprovedItems"))
    corrected = _int_value(row.get("CorrectedItems"))
    rejected = _int_value(row.get("RejectedItems"))
    reviewed = approved + corrected + rejected
    terminal = succeeded + failed + cancelled
    review_ready = reviewed + pending
    return {
        "total": total,
        "queued": queued,
        "processing": processing,
        "retry_wait": retry_wait,
        "succeeded": succeeded,
        "failed": failed,
        "cancelled": cancelled,
        "reviewed": reviewed,
        "unreviewed": pending,
        "not_ready": not_ready,
        "approved": approved,
        "corrected": corrected,
        "rejected": rejected,
        "processing_progress_percent": _percent(terminal, total),
        "review_progress_percent": _percent(reviewed, review_ready),
    }


def _item_payload(row: dict[str, Any], *, include_result: bool) -> dict[str, Any]:
    item_key = str(row.get("ItemKey") or "")
    payload = {
        "item_key": item_key,
        "sequence_no": _int_value(row.get("SequenceNo")),
        "file_name": row.get("OriginalFileName"),
        "content_type": row.get("ContentType"),
        "file_size_bytes": _int_value(row.get("FileSizeBytes")),
        "image_width": row.get("ImageWidth"),
        "image_height": row.get("ImageHeight"),
        "status": row.get("Status"),
        "attempt_count": _int_value(row.get("AttemptCount")),
        "max_attempts": _int_value(row.get("MaxAttempts")),
        "review_status": row.get("ReviewStatus"),
        "review_note": row.get("ReviewNote"),
        "reviewed_by": row.get("ReviewedBy"),
        "reviewed_at": _iso_value(row.get("ReviewedAt")),
        "review_updated_at": _iso_value(row.get("ReviewUpdatedDate")),
        "image_url": f"/api/passport-batches/items/{item_key}/image",
        "confidence": {
            "detector": row.get("YoloConfidence"),
            "ocr": row.get("OcrConfidence"),
            "checksum_valid": bool(row.get("ChecksumValid")) if row.get("ChecksumValid") is not None else None,
        },
        "processing": {
            "orientation_degree": row.get("OrientationDegree"),
            "used_fallback": bool(row.get("UsedFallback")),
            "process_time_ms": row.get("ProcessTimeMs"),
            "started_at": _iso_value(row.get("ProcessStartedAt")),
            "processed_at": _iso_value(row.get("ProcessedAt")),
            "error_code": row.get("ErrorCode"),
            "error": row.get("ErrorMessage"),
        },
        "fields": {
            "mrz_line_one": row.get("MrzLineOne"),
            "mrz_line_two": row.get("MrzLineTwo"),
            "passport_type": row.get("PassportType"),
            "issuing_country": row.get("IssuingCountry"),
            "surname": row.get("Surname"),
            "given_names": row.get("GivenNames"),
            "full_name": row.get("FullName"),
            "passport_number": row.get("PassportNumber"),
            "nationality": row.get("Nationality"),
            "date_of_birth": _iso_value(row.get("DateOfBirth")),
            "sex": row.get("Sex"),
            "date_of_expiry": _iso_value(row.get("DateOfExpiry")),
            "personal_number": row.get("PersonalNumber"),
        },
        "created_at": _iso_value(row.get("CreatedDate")),
        "updated_at": _iso_value(row.get("UpdatedDate")),
    }
    if include_result:
        payload["ai_result"] = _json_load(row.get("ResultJson"))
        payload["reviewed_result"] = _json_load(row.get("ReviewedResultJson"))
    return payload


def _normalize_uuid(value: str, field_name: str) -> UUID:
    try:
        return UUID(str(value or "").strip())
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{field_name} must be a valid UUID") from exc


def _item_identifier_lookup(value: str | int) -> tuple[str, Any]:
    raw_value = str(value or "").strip()
    if not raw_value:
        raise ValueError("id is required")
    if raw_value.isdigit():
        numeric_id = int(raw_value)
        if numeric_id <= 0:
            raise ValueError("id must be a positive item ID or UUID")
        return "item.Id = ?", numeric_id
    return "item.ItemKey = ?", str(_normalize_uuid(raw_value, "id"))


def _validated_filter(value: str | None, allowed: set[str], field_name: str) -> str | None:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return None
    if normalized not in allowed:
        raise ValueError(f"Invalid {field_name}: {normalized}")
    return normalized


def _pagination(page: int, page_size: int) -> tuple[int, int, int]:
    normalized_page = max(1, int(page))
    normalized_page_size = min(200, max(1, int(page_size)))
    return normalized_page, normalized_page_size, (normalized_page - 1) * normalized_page_size


def _pagination_payload(page: int, page_size: int, total: int) -> dict[str, int]:
    total_pages = (total + page_size - 1) // page_size if total else 0
    return {
        "page": page,
        "page_size": page_size,
        "total_items": total,
        "total_pages": total_pages,
    }


def _percent(value: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(100.0 * value / total, 2)


def _int_value(value: Any) -> int:
    return int(value or 0)


def _iso_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _json_load(value: Any) -> Any:
    if value is None or value == "":
        return None
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _remove_batch_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_file():
            child.unlink(missing_ok=True)
    path.rmdir()


def _optional_text(value: Any, max_length: int) -> str | None:
    text = str(value or "").strip()
    return text[:max_length] if text else None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _right_angle(value: Any) -> int:
    try:
        angle = int(value or 0) % 360
    except (TypeError, ValueError):
        return 0
    return angle if angle in {0, 90, 180, 270} else 0


def _parse_iso_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime, UUID)):
        return value.isoformat()
    return str(value)
