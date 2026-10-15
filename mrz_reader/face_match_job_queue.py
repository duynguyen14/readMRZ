from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import socket
import threading
import time
from typing import Any, Callable
from uuid import UUID, uuid4

import numpy as np

from .db import connect
from .env_config import env_value, read_env_file
from .passport_batch_queue import UploadedFile


JOB_STATUSES = {
    "receiving",
    "queued",
    "extracting",
    "matching",
    "completed",
    "completed_with_errors",
    "failed",
    "cancelled",
}
ITEM_STATUSES = {"queued", "processing", "retry_wait", "succeeded", "failed", "cancelled"}
CLASS_LABELS = {"face", "passport", "unknown"}
PAIR_DECISIONS = {"match", "review", "unmatched_face", "unmatched_passport"}
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class FaceMatchJobNotFoundError(LookupError):
    pass


class FaceMatchItemNotFoundError(LookupError):
    pass


@dataclass(frozen=True)
class ClaimedFaceMatchItem:
    id: int
    item_key: str
    job_id: int
    job_key: str
    original_file_name: str
    relative_image_path: str
    content_type: str
    attempt_count: int
    max_attempts: int


@dataclass(frozen=True)
class ClaimedMatchJob:
    id: int
    job_key: str
    match_threshold: float
    review_threshold: float


class FaceMatchJobRepository:
    def __init__(self, storage_root: Path, *, max_attempts: int = 3) -> None:
        self.storage_root = storage_root.expanduser().resolve()
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.max_attempts = max(1, int(max_attempts))

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "FaceMatchJobRepository":
        values = env or read_env_file()
        default_root = Path(__file__).resolve().parents[1] / "tmp" / "face_match_jobs"
        return cls(
            Path(env_value(values, "READMRZ_FACE_MATCH_JOB_STORAGE_DIR", str(default_root))),
            max_attempts=int(env_value(values, "READMRZ_FACE_MATCH_JOB_MAX_ATTEMPTS", "3")),
        )

    def create_job(
        self,
        files: list[UploadedFile],
        *,
        match_threshold: float,
        review_threshold: float,
        client_request_id: str | None = None,
        source_system: str | None = None,
        requested_by: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not files:
            raise ValueError("At least one image file is required")
        if not -1.0 <= review_threshold <= match_threshold <= 1.0:
            raise ValueError("Thresholds must satisfy -1 <= review_threshold <= match_threshold <= 1")

        job_key = uuid4()
        job_directory = self.storage_root / str(job_key)
        images_directory = job_directory / "images"
        embeddings_directory = job_directory / "embeddings"
        images_directory.mkdir(parents=True, exist_ok=False)
        embeddings_directory.mkdir(parents=True, exist_ok=True)
        saved_items: list[dict[str, Any]] = []

        try:
            for sequence_no, uploaded in enumerate(files, start=1):
                suffix = _safe_image_suffix(uploaded.file_name, uploaded.content_type)
                item_key = uuid4()
                stored_name = f"{sequence_no:06d}_{item_key}{suffix}"
                target_path = images_directory / stored_name
                temporary_path = target_path.with_suffix(target_path.suffix + ".part")
                temporary_path.write_bytes(uploaded.data)
                temporary_path.replace(target_path)
                saved_items.append(
                    {
                        "item_key": item_key,
                        "sequence_no": sequence_no,
                        "original_file_name": uploaded.file_name,
                        "relative_path": f"{job_key}/images/{stored_name}",
                        "content_type": uploaded.content_type,
                        "file_size": len(uploaded.data),
                        "sha256": hashlib.sha256(uploaded.data).hexdigest(),
                    }
                )

            with connect() as connection:
                cursor = connection.cursor()
                cursor.execute(
                    """
                    INSERT INTO dbo.readmrz_face_match_jobs (
                        JobKey, ClientRequestId, SourceSystem, RequestedBy, Status,
                        UploadCompleted, TotalItems, QueuedItems, MatchThreshold,
                        ReviewThreshold, StorageDirectory, MetadataJson, UploadCompletedAt
                    )
                    OUTPUT inserted.Id
                    VALUES (?, ?, ?, ?, N'queued', 1, ?, ?, ?, ?, ?, ?, SYSUTCDATETIME())
                    """,
                    str(job_key),
                    _optional_text(client_request_id, 128),
                    _optional_text(source_system, 128),
                    _optional_text(requested_by, 128),
                    len(saved_items),
                    len(saved_items),
                    float(match_threshold),
                    float(review_threshold),
                    str(job_key),
                    _json_dump(metadata) if metadata else None,
                )
                job_id = int(cursor.fetchone()[0])
                cursor.fast_executemany = True
                cursor.executemany(
                    """
                    INSERT INTO dbo.readmrz_face_match_items (
                        ItemKey, JobId, SequenceNo, OriginalFileName, RelativeImagePath,
                        ContentType, FileSizeBytes, Sha256, Status, MaxAttempts
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, N'queued', ?)
                    """,
                    [
                        (
                            str(item["item_key"]),
                            job_id,
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
            shutil.rmtree(job_directory, ignore_errors=True)
            raise

        return {
            "job_id": str(job_key),
            "status": "queued",
            "received": len(saved_items),
        }

    def list_jobs(
        self,
        *,
        status: str | None = None,
        keyword: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        normalized_status = _validated_filter(status, JOB_STATUSES, "status")
        page, page_size, offset = _pagination(page, page_size)
        filters: list[str] = []
        parameters: list[Any] = []
        if normalized_status:
            filters.append("job.Status = ?")
            parameters.append(normalized_status)
        normalized_keyword = str(keyword or "").strip()
        if normalized_keyword:
            pattern = f"%{normalized_keyword}%"
            filters.append(
                "(CONVERT(NVARCHAR(36), job.JobKey) LIKE ? OR job.ClientRequestId LIKE ? "
                "OR job.SourceSystem LIKE ? OR job.RequestedBy LIKE ?)"
            )
            parameters.extend([pattern, pattern, pattern, pattern])
        where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""

        with connect() as connection:
            cursor = connection.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM dbo.readmrz_face_match_jobs job {where_sql}", *parameters)
            total = int(cursor.fetchone()[0])
            cursor.execute(
                f"""
                SELECT {_job_select_columns()}
                FROM dbo.readmrz_face_match_jobs job
                {where_sql}
                ORDER BY job.CreatedDate DESC, job.Id DESC
                OFFSET ? ROWS FETCH NEXT ? ROWS ONLY
                """,
                *parameters,
                offset,
                page_size,
            )
            rows = [_job_payload(_row_dict(cursor, row)) for row in cursor.fetchall()]
        return {"items": rows, "pagination": _pagination_payload(page, page_size, total), "total_jobs": total}

    def get_job(self, job_key: str) -> dict[str, Any]:
        job_uuid = _normalize_uuid(job_key, "job_id")
        with connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"SELECT {_job_select_columns()} FROM dbo.readmrz_face_match_jobs job WHERE job.JobKey = ?",
                str(job_uuid),
            )
            row = cursor.fetchone()
            if row is None:
                raise FaceMatchJobNotFoundError(f"Face match job not found: {job_uuid}")
            return _job_payload(_row_dict(cursor, row))

    def list_items(
        self,
        job_key: str,
        *,
        status: str | None = None,
        classification: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        job_uuid = _normalize_uuid(job_key, "job_id")
        normalized_status = _validated_filter(status, ITEM_STATUSES, "status")
        normalized_class = _validated_filter(classification, CLASS_LABELS, "classification")
        page, page_size, offset = _pagination(page, page_size)
        with connect() as connection:
            cursor = connection.cursor()
            job = self._get_job_row(cursor, job_uuid)
            filters = ["item.JobId = ?"]
            parameters: list[Any] = [int(job["Id"])]
            if normalized_status:
                filters.append("item.Status = ?")
                parameters.append(normalized_status)
            if normalized_class:
                filters.append("item.ClassificationLabel = ?")
                parameters.append(normalized_class)
            where_sql = " AND ".join(filters)
            cursor.execute(f"SELECT COUNT(*) FROM dbo.readmrz_face_match_items item WHERE {where_sql}", *parameters)
            total = int(cursor.fetchone()[0])
            cursor.execute(
                f"""
                SELECT {_item_select_columns()}
                FROM dbo.readmrz_face_match_items item
                WHERE {where_sql}
                ORDER BY item.SequenceNo, item.Id
                OFFSET ? ROWS FETCH NEXT ? ROWS ONLY
                """,
                *parameters,
                offset,
                page_size,
            )
            items = [_item_payload(_row_dict(cursor, row), include_result=True) for row in cursor.fetchall()]
        return {
            "job": _job_payload(job),
            "items": items,
            "pagination": _pagination_payload(page, page_size, total),
        }

    def list_pairs(
        self,
        job_key: str,
        *,
        decision: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        job_uuid = _normalize_uuid(job_key, "job_id")
        normalized_decision = _validated_filter(decision, PAIR_DECISIONS, "decision")
        page, page_size, offset = _pagination(page, page_size)
        with connect() as connection:
            cursor = connection.cursor()
            job = self._get_job_row(cursor, job_uuid)
            filters = ["pair.JobId = ?"]
            parameters: list[Any] = [int(job["Id"])]
            if normalized_decision:
                filters.append("pair.Decision = ?")
                parameters.append(normalized_decision)
            where_sql = " AND ".join(filters)
            cursor.execute(f"SELECT COUNT(*) FROM dbo.readmrz_face_match_pairs pair WHERE {where_sql}", *parameters)
            total = int(cursor.fetchone()[0])
            cursor.execute(
                f"""
                SELECT
                    pair.PairKey, pair.Score, pair.Decision, pair.ReviewRequired,
                    pair.FaceDetectionIndex, pair.PassportDetectionIndex, pair.ResultJson,
                    face.ItemKey AS FaceItemKey, face.SequenceNo AS FaceSequenceNo,
                    face.OriginalFileName AS FaceFileName,
                    passport.ItemKey AS PassportItemKey, passport.SequenceNo AS PassportSequenceNo,
                    passport.OriginalFileName AS PassportFileName
                FROM dbo.readmrz_face_match_pairs pair
                LEFT JOIN dbo.readmrz_face_match_items face ON face.Id = pair.FaceItemId
                LEFT JOIN dbo.readmrz_face_match_items passport ON passport.Id = pair.PassportItemId
                WHERE {where_sql}
                ORDER BY CASE pair.Decision WHEN N'match' THEN 0 WHEN N'review' THEN 1 ELSE 2 END,
                         pair.Score DESC, pair.Id
                OFFSET ? ROWS FETCH NEXT ? ROWS ONLY
                """,
                *parameters,
                offset,
                page_size,
            )
            pairs = [_pair_payload(_row_dict(cursor, row)) for row in cursor.fetchall()]
        return {
            "job": _job_payload(job),
            "pairs": pairs,
            "pagination": _pagination_payload(page, page_size, total),
        }

    def get_item_image(self, item_key: str) -> dict[str, Any]:
        item_uuid = _normalize_uuid(item_key, "item_key")
        with connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT OriginalFileName, RelativeImagePath, ContentType
                FROM dbo.readmrz_face_match_items WHERE ItemKey = ?
                """,
                str(item_uuid),
            )
            row = cursor.fetchone()
            if row is None:
                raise FaceMatchItemNotFoundError(f"Face match item not found: {item_uuid}")
        image_path = self.absolute_path(str(row[1]))
        if not image_path.is_file():
            raise FileNotFoundError(f"Stored face match image does not exist: {image_path}")
        return {
            "path": image_path,
            "file_name": str(row[0] or image_path.name),
            "content_type": str(row[2] or "application/octet-stream"),
        }

    def claim_next_item(self, worker_id: str) -> ClaimedFaceMatchItem | None:
        with connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                ;WITH next_item AS (
                    SELECT TOP (1) item.*
                    FROM dbo.readmrz_face_match_items item WITH (UPDLOCK, READPAST, ROWLOCK)
                    INNER JOIN dbo.readmrz_face_match_jobs job ON job.Id = item.JobId
                    WHERE item.Status IN (N'queued', N'retry_wait')
                      AND job.UploadCompleted = 1
                      AND job.Status IN (N'queued', N'extracting')
                      AND (item.NextAttemptAt IS NULL OR item.NextAttemptAt <= SYSUTCDATETIME())
                      AND item.AttemptCount < item.MaxAttempts
                    ORDER BY CASE WHEN item.Status = N'queued' THEN 0 ELSE 1 END, item.Id
                )
                UPDATE next_item
                SET Status = N'processing', AttemptCount = AttemptCount + 1,
                    NextAttemptAt = NULL, LockedBy = ?, LockedAt = SYSUTCDATETIME(),
                    ProcessStartedAt = COALESCE(ProcessStartedAt, SYSUTCDATETIME()),
                    UpdatedDate = SYSUTCDATETIME()
                OUTPUT inserted.Id, inserted.ItemKey, inserted.JobId, inserted.OriginalFileName,
                       inserted.RelativeImagePath, inserted.ContentType,
                       inserted.AttemptCount, inserted.MaxAttempts;
                """,
                worker_id,
            )
            row = cursor.fetchone()
            if row is None:
                connection.commit()
                return None
            cursor.execute("SELECT JobKey FROM dbo.readmrz_face_match_jobs WHERE Id = ?", int(row[2]))
            job_key = str(cursor.fetchone()[0])
            item = ClaimedFaceMatchItem(
                id=int(row[0]),
                item_key=str(row[1]),
                job_id=int(row[2]),
                job_key=job_key,
                original_file_name=str(row[3]),
                relative_image_path=str(row[4]),
                content_type=str(row[5] or "application/octet-stream"),
                attempt_count=int(row[6]),
                max_attempts=int(row[7]),
            )
            self._refresh_job(cursor, item.job_id)
            connection.commit()
            return item

    def mark_item_succeeded(
        self,
        item: ClaimedFaceMatchItem,
        *,
        classification: str,
        classification_confidence: float,
        result: dict[str, Any],
        embeddings: np.ndarray,
        image_width: int,
        image_height: int,
        process_time_ms: int,
    ) -> None:
        normalized_class = classification if classification in CLASS_LABELS else "unknown"
        embedding_path = self.save_embeddings(item, embeddings) if embeddings.size else None
        face_count = int(result.get("face_count") or len(result.get("embedding_candidates") or []))
        primary_confidence = result.get("face_confidence")
        embedding_count = int(embeddings.shape[0]) if embeddings.ndim == 2 else 0
        embedding_dimension = int(embeddings.shape[1]) if embeddings.ndim == 2 and embedding_count else None
        with connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                UPDATE dbo.readmrz_face_match_items
                SET Status = N'succeeded', ClassificationLabel = ?, ClassificationConfidence = ?,
                    RelativeEmbeddingPath = ?, FaceCount = ?, EmbeddingCount = ?,
                    EmbeddingDimension = ?, PrimaryFaceConfidence = ?, ResultJson = ?,
                    ImageWidth = ?, ImageHeight = ?, ProcessTimeMs = ?, ErrorCode = NULL,
                    ErrorMessage = NULL, NextAttemptAt = NULL, LockedBy = NULL, LockedAt = NULL,
                    ProcessedAt = SYSUTCDATETIME(), UpdatedDate = SYSUTCDATETIME()
                WHERE Id = ? AND Status = N'processing'
                """,
                normalized_class,
                float(classification_confidence),
                embedding_path,
                max(0, face_count),
                embedding_count,
                embedding_dimension,
                _optional_float(primary_confidence),
                _json_dump(result),
                int(image_width),
                int(image_height),
                max(0, int(process_time_ms)),
                item.id,
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Face match item {item.id} is no longer locked")
            self._refresh_job(cursor, item.job_id)
            connection.commit()

    def mark_item_failed(self, item: ClaimedFaceMatchItem, exc: Exception, retry_delay_seconds: int) -> str:
        will_retry = item.attempt_count < item.max_attempts
        status = "retry_wait" if will_retry else "failed"
        error_code = type(exc).__name__[:100]
        error_message = str(exc)[:2000] or error_code
        with connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                UPDATE dbo.readmrz_face_match_items
                SET Status = ?, NextAttemptAt = CASE WHEN ? = 1 THEN DATEADD(SECOND, ?, SYSUTCDATETIME()) ELSE NULL END,
                    LockedBy = NULL, LockedAt = NULL, ErrorCode = ?, ErrorMessage = ?,
                    ProcessedAt = CASE WHEN ? = 1 THEN ProcessedAt ELSE SYSUTCDATETIME() END,
                    UpdatedDate = SYSUTCDATETIME()
                WHERE Id = ? AND Status = N'processing'
                """,
                status,
                1 if will_retry else 0,
                max(1, int(retry_delay_seconds)),
                error_code,
                error_message,
                1 if will_retry else 0,
                item.id,
            )
            self._refresh_job(cursor, item.job_id)
            connection.commit()
        return status

    def claim_ready_match_job(self, worker_id: str) -> ClaimedMatchJob | None:
        with connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                ;WITH ready_job AS (
                    SELECT TOP (1) job.*
                    FROM dbo.readmrz_face_match_jobs job WITH (UPDLOCK, READPAST, ROWLOCK)
                    WHERE job.UploadCompleted = 1
                      AND job.Status IN (N'queued', N'extracting')
                      AND NOT EXISTS (
                          SELECT 1 FROM dbo.readmrz_face_match_items item
                          WHERE item.JobId = job.Id
                            AND item.Status IN (N'queued', N'processing', N'retry_wait')
                      )
                    ORDER BY job.Id
                )
                UPDATE ready_job
                SET Status = N'matching', MatchLockedBy = ?, MatchLockedAt = SYSUTCDATETIME(),
                    MatchingStartedAt = COALESCE(MatchingStartedAt, SYSUTCDATETIME()),
                    UpdatedDate = SYSUTCDATETIME()
                OUTPUT inserted.Id, inserted.JobKey, inserted.MatchThreshold, inserted.ReviewThreshold;
                """,
                worker_id,
            )
            row = cursor.fetchone()
            connection.commit()
            if row is None:
                return None
            return ClaimedMatchJob(int(row[0]), str(row[1]), float(row[2]), float(row[3]))

    def load_match_candidates(self, job: ClaimedMatchJob) -> list[dict[str, Any]]:
        with connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT Id, ItemKey, SequenceNo, OriginalFileName, ClassificationLabel,
                       RelativeEmbeddingPath, ResultJson
                FROM dbo.readmrz_face_match_items
                WHERE JobId = ? AND Status = N'succeeded'
                  AND ClassificationLabel IN (N'face', N'passport')
                ORDER BY SequenceNo, Id
                """,
                job.id,
            )
            rows = [_row_dict(cursor, row) for row in cursor.fetchall()]
        candidates: list[dict[str, Any]] = []
        for row in rows:
            embedding_path = row.get("RelativeEmbeddingPath")
            embeddings = self.load_embeddings(str(embedding_path)) if embedding_path else np.empty((0, 0), dtype=np.float32)
            candidates.append(
                {
                    "id": int(row["Id"]),
                    "item_key": str(row["ItemKey"]),
                    "sequence_no": int(row["SequenceNo"]),
                    "file_name": str(row["OriginalFileName"]),
                    "classification": str(row["ClassificationLabel"]),
                    "embeddings": embeddings,
                    "result": _json_load(row.get("ResultJson")) or {},
                }
            )
        return candidates

    def complete_matching(self, job: ClaimedMatchJob, pairs: list[dict[str, Any]]) -> None:
        matched = sum(1 for pair in pairs if pair["decision"] == "match")
        review = sum(1 for pair in pairs if pair["decision"] == "review")
        unmatched_faces = sum(1 for pair in pairs if pair["decision"] == "unmatched_face")
        unmatched_passports = sum(1 for pair in pairs if pair["decision"] == "unmatched_passport")
        with connect() as connection:
            cursor = connection.cursor()
            cursor.execute("DELETE FROM dbo.readmrz_face_match_pairs WHERE JobId = ?", job.id)
            if pairs:
                cursor.fast_executemany = True
                cursor.executemany(
                    """
                    INSERT INTO dbo.readmrz_face_match_pairs (
                        PairKey, JobId, FaceItemId, PassportItemId, FaceDetectionIndex,
                        PassportDetectionIndex, Score, Decision, ReviewRequired, ResultJson
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            str(uuid4()),
                            job.id,
                            pair.get("face_item_id"),
                            pair.get("passport_item_id"),
                            pair.get("face_detection_index"),
                            pair.get("passport_detection_index"),
                            pair.get("score"),
                            pair["decision"],
                            1 if pair["decision"] == "review" else 0,
                            _json_dump(pair.get("result") or {}),
                        )
                        for pair in pairs
                    ],
                )
            cursor.execute(
                """
                UPDATE dbo.readmrz_face_match_jobs
                SET Status = CASE
                        WHEN SucceededItems = 0 THEN N'failed'
                        WHEN FailedItems > 0 OR FaceItems = 0 OR PassportItems = 0 THEN N'completed_with_errors'
                        ELSE N'completed'
                    END,
                    MatchedPairs = ?, ReviewPairs = ?, UnmatchedFaces = ?, UnmatchedPassports = ?,
                    MatchLockedBy = NULL, MatchLockedAt = NULL, CompletedAt = SYSUTCDATETIME(),
                    ErrorMessage = CASE
                        WHEN SucceededItems = 0 THEN N'No item was extracted successfully.'
                        WHEN FaceItems = 0 THEN N'No valid face images were classified.'
                        WHEN PassportItems = 0 THEN N'No valid passport images were classified.'
                        ELSE NULL
                    END,
                    UpdatedDate = SYSUTCDATETIME()
                WHERE Id = ? AND Status = N'matching'
                """,
                matched,
                review,
                unmatched_faces,
                unmatched_passports,
                job.id,
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Face match job {job.id} is no longer locked for matching")
            connection.commit()

    def fail_matching(self, job: ClaimedMatchJob, exc: Exception) -> None:
        with connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                UPDATE dbo.readmrz_face_match_jobs
                SET Status = N'failed', ErrorMessage = ?, MatchLockedBy = NULL, MatchLockedAt = NULL,
                    CompletedAt = SYSUTCDATETIME(), UpdatedDate = SYSUTCDATETIME()
                WHERE Id = ? AND Status = N'matching'
                """,
                _optional_text(str(exc) or type(exc).__name__, 2000),
                job.id,
            )
            connection.commit()

    def recover_stale(self, lock_timeout_seconds: int) -> tuple[int, int]:
        cutoff_seconds = max(60, int(lock_timeout_seconds))
        with connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                UPDATE dbo.readmrz_face_match_items
                SET Status = CASE WHEN AttemptCount < MaxAttempts THEN N'retry_wait' ELSE N'failed' END,
                    NextAttemptAt = CASE WHEN AttemptCount < MaxAttempts THEN SYSUTCDATETIME() ELSE NULL END,
                    LockedBy = NULL, LockedAt = NULL, ErrorCode = N'WORKER_LOCK_TIMEOUT',
                    ErrorMessage = N'Worker lock expired before extraction completed.',
                    ProcessedAt = CASE WHEN AttemptCount >= MaxAttempts THEN SYSUTCDATETIME() ELSE ProcessedAt END,
                    UpdatedDate = SYSUTCDATETIME()
                WHERE Status = N'processing' AND LockedAt < DATEADD(SECOND, ?, SYSUTCDATETIME())
                """,
                -cutoff_seconds,
            )
            recovered_items = max(0, int(cursor.rowcount))
            cursor.execute(
                """
                UPDATE dbo.readmrz_face_match_jobs
                SET Status = N'extracting', MatchLockedBy = NULL, MatchLockedAt = NULL,
                    ErrorMessage = N'Matching lock expired; job returned to queue.', UpdatedDate = SYSUTCDATETIME()
                WHERE Status = N'matching' AND MatchLockedAt < DATEADD(SECOND, ?, SYSUTCDATETIME())
                """,
                -cutoff_seconds,
            )
            recovered_jobs = max(0, int(cursor.rowcount))
            if recovered_items:
                cursor.execute(
                    "SELECT DISTINCT JobId FROM dbo.readmrz_face_match_items WHERE ErrorCode = N'WORKER_LOCK_TIMEOUT'"
                )
                for row in cursor.fetchall():
                    self._refresh_job(cursor, int(row[0]))
            connection.commit()
        return recovered_items, recovered_jobs

    def save_embeddings(self, item: ClaimedFaceMatchItem, embeddings: np.ndarray) -> str:
        normalized = np.asarray(embeddings, dtype=np.float32)
        if normalized.ndim != 2:
            raise ValueError("Embeddings must be a two-dimensional array")
        relative_path = f"{item.job_key}/embeddings/{item.item_key}.npy"
        target = self.absolute_path(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".npy.part")
        with temporary.open("wb") as stream:
            np.save(stream, normalized, allow_pickle=False)
        temporary.replace(target)
        return relative_path

    def load_embeddings(self, relative_path: str) -> np.ndarray:
        path = self.absolute_path(relative_path)
        with path.open("rb") as stream:
            values = np.load(stream, allow_pickle=False)
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"Invalid embedding array: {path}")
        return values

    def absolute_path(self, relative_path: str) -> Path:
        normalized = Path(str(relative_path).replace("\\", "/"))
        candidate = (self.storage_root / normalized).resolve()
        try:
            candidate.relative_to(self.storage_root)
        except ValueError as exc:
            raise ValueError(f"Stored path escapes the face match root: {relative_path}") from exc
        return candidate

    @staticmethod
    def _get_job_row(cursor: Any, job_key: UUID) -> dict[str, Any]:
        cursor.execute(
            f"SELECT {_job_select_columns()} FROM dbo.readmrz_face_match_jobs job WHERE job.JobKey = ?",
            str(job_key),
        )
        row = cursor.fetchone()
        if row is None:
            raise FaceMatchJobNotFoundError(f"Face match job not found: {job_key}")
        return _row_dict(cursor, row)

    @staticmethod
    def _refresh_job(cursor: Any, job_id: int) -> None:
        cursor.execute(
            """
            ;WITH counts AS (
                SELECT COUNT(*) AS TotalItems,
                    SUM(CASE WHEN Status IN (N'queued', N'retry_wait') THEN 1 ELSE 0 END) AS QueuedItems,
                    SUM(CASE WHEN Status = N'processing' THEN 1 ELSE 0 END) AS ProcessingItems,
                    SUM(CASE WHEN Status = N'succeeded' THEN 1 ELSE 0 END) AS SucceededItems,
                    SUM(CASE WHEN Status = N'failed' THEN 1 ELSE 0 END) AS FailedItems,
                    SUM(CASE WHEN Status = N'succeeded' AND ClassificationLabel = N'face' THEN 1 ELSE 0 END) AS FaceItems,
                    SUM(CASE WHEN Status = N'succeeded' AND ClassificationLabel = N'passport' THEN 1 ELSE 0 END) AS PassportItems,
                    SUM(CASE WHEN Status = N'succeeded' AND ClassificationLabel = N'unknown' THEN 1 ELSE 0 END) AS UnknownItems
                FROM dbo.readmrz_face_match_items WHERE JobId = ?
            )
            UPDATE job
            SET TotalItems = counts.TotalItems, QueuedItems = counts.QueuedItems,
                ProcessingItems = counts.ProcessingItems, SucceededItems = counts.SucceededItems,
                FailedItems = counts.FailedItems, FaceItems = counts.FaceItems,
                PassportItems = counts.PassportItems, UnknownItems = counts.UnknownItems,
                Status = CASE
                    WHEN job.Status IN (N'matching', N'completed', N'completed_with_errors', N'failed', N'cancelled')
                        THEN job.Status
                    WHEN counts.ProcessingItems + counts.SucceededItems + counts.FailedItems > 0 THEN N'extracting'
                    ELSE N'queued'
                END,
                ProcessStartedAt = CASE
                    WHEN counts.ProcessingItems + counts.SucceededItems + counts.FailedItems > 0
                        THEN COALESCE(job.ProcessStartedAt, SYSUTCDATETIME())
                    ELSE job.ProcessStartedAt
                END,
                UpdatedDate = SYSUTCDATETIME()
            FROM dbo.readmrz_face_match_jobs job CROSS JOIN counts
            WHERE job.Id = ?
            """,
            job_id,
            job_id,
        )


class FaceMatchJobWorker:
    def __init__(
        self,
        repository: FaceMatchJobRepository,
        processor: Callable[[bytes, str, str], tuple[dict[str, Any], np.ndarray, int, int]],
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
            name="readmrz-face-match-job-worker",
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
        self.logger(f"Face match job worker started worker_id={self.worker_id}")
        self._recover_if_due()
        while not self._stop_event.is_set():
            self._recover_if_due()
            if self._try_matching():
                continue
            try:
                item = self.repository.claim_next_item(self.worker_id)
            except Exception as exc:
                self.logger(f"Face match job worker claim error {exc}")
                self._wait()
                continue
            if item is None:
                self._wait()
                continue
            started = time.perf_counter()
            try:
                image_path = self.repository.absolute_path(item.relative_image_path)
                result, embeddings, width, height = self.processor(
                    image_path.read_bytes(), item.original_file_name, item.content_type
                )
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                classification = result.get("classification") or {}
                self.repository.mark_item_succeeded(
                    item,
                    classification=str(classification.get("label") or "unknown"),
                    classification_confidence=float(classification.get("confidence") or 0.0),
                    result=result,
                    embeddings=embeddings,
                    image_width=width,
                    image_height=height,
                    process_time_ms=elapsed_ms,
                )
                self.logger(
                    "Face match job item done "
                    f"item_id={item.id} job_id={item.job_id} class={classification.get('label')} "
                    f"embeddings={len(embeddings)} latency_ms={elapsed_ms}"
                )
            except Exception as exc:
                try:
                    status = self.repository.mark_item_failed(item, exc, self.retry_delay_seconds)
                except Exception as update_exc:
                    self.logger(
                        f"Face match item state update error item_id={item.id} "
                        f"processing_error={exc} update_error={update_exc}"
                    )
                else:
                    self.logger(
                        f"Face match job item error item_id={item.id} status={status} "
                        f"attempt={item.attempt_count}/{item.max_attempts} error={exc}"
                    )
        self.logger(f"Face match job worker stopped worker_id={self.worker_id}")

    def _try_matching(self) -> bool:
        try:
            job = self.repository.claim_ready_match_job(self.worker_id)
        except Exception as exc:
            self.logger(f"Face match job matching claim error {exc}")
            return False
        if job is None:
            return False
        started = time.perf_counter()
        try:
            candidates = self.repository.load_match_candidates(job)
            pairs = build_optimal_pairs(
                candidates,
                match_threshold=job.match_threshold,
                review_threshold=job.review_threshold,
            )
            self.repository.complete_matching(job, pairs)
            self.logger(
                "Face match job matching done "
                f"job_id={job.id} candidates={len(candidates)} pairs={len(pairs)} "
                f"latency_ms={int((time.perf_counter() - started) * 1000)}"
            )
        except Exception as exc:
            try:
                self.repository.fail_matching(job, exc)
            finally:
                self.logger(f"Face match job matching error job_id={job.id} error={exc}")
        return True

    def _wait(self) -> None:
        self._wake_event.wait(self.poll_seconds)
        self._wake_event.clear()

    def _recover_if_due(self) -> None:
        now = time.monotonic()
        interval = min(60.0, max(10.0, self.lock_timeout_seconds / 3.0))
        if now - self._last_recovery_at < interval:
            return
        self._last_recovery_at = now
        try:
            items, jobs = self.repository.recover_stale(self.lock_timeout_seconds)
            if items or jobs:
                self.logger(f"Face match job worker recovered items={items} matching_jobs={jobs}")
        except Exception as exc:
            self.logger(f"Face match job worker recovery error {exc}")


def build_optimal_pairs(
    candidates: list[dict[str, Any]],
    *,
    match_threshold: float,
    review_threshold: float,
) -> list[dict[str, Any]]:
    faces = [candidate for candidate in candidates if candidate.get("classification") == "face"]
    passports = [candidate for candidate in candidates if candidate.get("classification") == "passport"]
    score_matrix = np.zeros((len(faces), len(passports)), dtype=np.float64)
    detection_indexes: dict[tuple[int, int], tuple[int, int]] = {}
    valid_pairs: set[tuple[int, int]] = set()

    for face_index, face in enumerate(faces):
        face_embeddings = _normalized_embeddings(face.get("embeddings"))
        for passport_index, passport in enumerate(passports):
            passport_embeddings = _normalized_embeddings(passport.get("embeddings"))
            if not face_embeddings.size or not passport_embeddings.size:
                continue
            scores = face_embeddings @ passport_embeddings.T
            flat_index = int(np.argmax(scores))
            face_detection_index, passport_detection_index = np.unravel_index(flat_index, scores.shape)
            score_matrix[face_index, passport_index] = float(scores[face_detection_index, passport_detection_index])
            detection_indexes[(face_index, passport_index)] = (
                int(face_detection_index), int(passport_detection_index)
            )
            valid_pairs.add((face_index, passport_index))

    valid_mask = np.zeros_like(score_matrix, dtype=bool)
    for face_index, passport_index in valid_pairs:
        valid_mask[face_index, passport_index] = True
    benefits = np.where(
        valid_mask & (score_matrix >= review_threshold),
        score_matrix - review_threshold + 1e-9,
        0.0,
    )
    assignments = hungarian_maximize(benefits.tolist()) if faces and passports else []
    used_faces: set[int] = set()
    used_passports: set[int] = set()
    pairs: list[dict[str, Any]] = []

    for face_index, passport_index in assignments:
        score = float(score_matrix[face_index, passport_index])
        if (face_index, passport_index) not in valid_pairs or score < review_threshold:
            continue
        face = faces[face_index]
        passport = passports[passport_index]
        face_detection_index, passport_detection_index = detection_indexes[(face_index, passport_index)]
        decision = "match" if score >= match_threshold else "review"
        pairs.append(
            {
                "face_item_id": face["id"],
                "passport_item_id": passport["id"],
                "face_detection_index": face_detection_index,
                "passport_detection_index": passport_detection_index,
                "score": round(score, 6),
                "decision": decision,
                "result": _pair_result(face, passport, score, decision),
            }
        )
        used_faces.add(face_index)
        used_passports.add(passport_index)

    for index, face in enumerate(faces):
        if index not in used_faces:
            pairs.append(
                {
                    "face_item_id": face["id"],
                    "passport_item_id": None,
                    "score": None,
                    "decision": "unmatched_face",
                    "result": {"face": _candidate_identity(face), "passport": None},
                }
            )
    for index, passport in enumerate(passports):
        if index not in used_passports:
            pairs.append(
                {
                    "face_item_id": None,
                    "passport_item_id": passport["id"],
                    "score": None,
                    "decision": "unmatched_passport",
                    "result": {"face": None, "passport": _candidate_identity(passport)},
                }
            )
    return pairs


def hungarian_maximize(weights: list[list[float]]) -> list[tuple[int, int]]:
    if not weights or not weights[0]:
        return []
    row_count = len(weights)
    column_count = len(weights[0])
    if any(len(row) != column_count for row in weights):
        raise ValueError("Weight matrix must be rectangular")
    transposed = row_count > column_count
    matrix = [list(row) for row in weights]
    if transposed:
        matrix = [list(row) for row in zip(*matrix)]
    n = len(matrix)
    m = len(matrix[0])
    maximum = max(max(row) for row in matrix)
    costs = [[maximum - float(value) for value in row] for row in matrix]
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        min_values = [float("inf")] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                current = costs[i0 - 1][j - 1] - u[i0] - v[j]
                if current < min_values[j]:
                    min_values[j] = current
                    way[j] = j0
                if min_values[j] < delta:
                    delta = min_values[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    min_values[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignments: list[tuple[int, int]] = []
    for column in range(1, m + 1):
        if p[column] == 0:
            continue
        row_index = p[column] - 1
        column_index = column - 1
        assignments.append(
            (column_index, row_index) if transposed else (row_index, column_index)
        )
    return assignments


def _normalized_embeddings(value: Any) -> np.ndarray:
    embeddings = np.asarray(value if value is not None else [], dtype=np.float32)
    if embeddings.ndim != 2 or not embeddings.size:
        return np.empty((0, 0), dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return embeddings / norms


def _pair_result(face: dict[str, Any], passport: dict[str, Any], score: float, decision: str) -> dict[str, Any]:
    passport_result = passport.get("result") or {}
    return {
        "face": _candidate_identity(face),
        "passport": {
            **_candidate_identity(passport),
            "mrz": passport_result.get("mrz"),
            "parsed": passport_result.get("parsed"),
        },
        "match": {
            "score": round(float(score), 6),
            "decision": decision,
            "matched": decision == "match",
            "review_required": decision == "review",
        },
    }


def _candidate_identity(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "item_key": candidate.get("item_key"),
        "sequence_no": candidate.get("sequence_no"),
        "file_name": candidate.get("file_name"),
    }


def _job_select_columns() -> str:
    return """
        job.Id, job.JobKey, job.ClientRequestId, job.SourceSystem, job.RequestedBy,
        job.Status, job.UploadCompleted, job.TotalItems, job.QueuedItems,
        job.ProcessingItems, job.SucceededItems, job.FailedItems, job.FaceItems,
        job.PassportItems, job.UnknownItems, job.MatchedPairs, job.ReviewPairs,
        job.UnmatchedFaces, job.UnmatchedPassports, job.MatchThreshold,
        job.ReviewThreshold, job.ErrorMessage, job.UploadCompletedAt,
        job.ProcessStartedAt, job.MatchingStartedAt, job.CompletedAt,
        job.MetadataJson, job.CreatedDate, job.UpdatedDate
    """


def _item_select_columns() -> str:
    return """
        item.ItemKey, item.SequenceNo, item.OriginalFileName, item.ContentType,
        item.FileSizeBytes, item.ImageWidth, item.ImageHeight, item.Status,
        item.AttemptCount, item.MaxAttempts, item.ClassificationLabel,
        item.ClassificationConfidence, item.FaceCount, item.EmbeddingCount,
        item.EmbeddingDimension, item.PrimaryFaceConfidence, item.ResultJson,
        item.ErrorCode, item.ErrorMessage, item.ProcessTimeMs,
        item.ProcessStartedAt, item.ProcessedAt, item.CreatedDate, item.UpdatedDate
    """


def _job_payload(row: dict[str, Any]) -> dict[str, Any]:
    total = _int_value(row.get("TotalItems"))
    succeeded = _int_value(row.get("SucceededItems"))
    failed = _int_value(row.get("FailedItems"))
    terminal = succeeded + failed
    return {
        "job_id": str(row.get("JobKey") or ""),
        "client_request_id": row.get("ClientRequestId"),
        "source_system": row.get("SourceSystem"),
        "requested_by": row.get("RequestedBy"),
        "status": row.get("Status"),
        "upload_completed": bool(row.get("UploadCompleted")),
        "statistics": {
            "total_items": total,
            "queued_items": _int_value(row.get("QueuedItems")),
            "processing_items": _int_value(row.get("ProcessingItems")),
            "succeeded_items": succeeded,
            "failed_items": failed,
            "face_items": _int_value(row.get("FaceItems")),
            "passport_items": _int_value(row.get("PassportItems")),
            "unknown_items": _int_value(row.get("UnknownItems")),
            "matched_pairs": _int_value(row.get("MatchedPairs")),
            "review_required_pairs": _int_value(row.get("ReviewPairs")),
            "unmatched_faces": _int_value(row.get("UnmatchedFaces")),
            "unmatched_passports": _int_value(row.get("UnmatchedPassports")),
            "extraction_progress_percent": _percent(terminal, total),
        },
        "thresholds": {
            "match": row.get("MatchThreshold"),
            "review": row.get("ReviewThreshold"),
        },
        "metadata": _json_load(row.get("MetadataJson")),
        "error": row.get("ErrorMessage"),
        "upload_completed_at": _iso_value(row.get("UploadCompletedAt")),
        "process_started_at": _iso_value(row.get("ProcessStartedAt")),
        "matching_started_at": _iso_value(row.get("MatchingStartedAt")),
        "completed_at": _iso_value(row.get("CompletedAt")),
        "created_at": _iso_value(row.get("CreatedDate")),
        "updated_at": _iso_value(row.get("UpdatedDate")),
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
        "classification": {
            "label": row.get("ClassificationLabel"),
            "confidence": row.get("ClassificationConfidence"),
        },
        "face_count": _int_value(row.get("FaceCount")),
        "embedding_count": _int_value(row.get("EmbeddingCount")),
        "embedding_dimension": row.get("EmbeddingDimension"),
        "primary_face_confidence": row.get("PrimaryFaceConfidence"),
        "process_time_ms": row.get("ProcessTimeMs"),
        "error_code": row.get("ErrorCode"),
        "error": row.get("ErrorMessage"),
        "image_url": f"/api/passport-face-match/jobs/items/{item_key}/image",
        "processed_at": _iso_value(row.get("ProcessedAt")),
        "created_at": _iso_value(row.get("CreatedDate")),
        "updated_at": _iso_value(row.get("UpdatedDate")),
    }
    if include_result:
        payload["result"] = _json_load(row.get("ResultJson"))
    return payload


def _pair_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair_id": str(row.get("PairKey") or ""),
        "score": row.get("Score"),
        "decision": row.get("Decision"),
        "review_required": bool(row.get("ReviewRequired")),
        "face_detection_index": row.get("FaceDetectionIndex"),
        "passport_detection_index": row.get("PassportDetectionIndex"),
        "face": None if row.get("FaceItemKey") is None else {
            "item_key": str(row.get("FaceItemKey")),
            "sequence_no": _int_value(row.get("FaceSequenceNo")),
            "file_name": row.get("FaceFileName"),
            "image_url": f"/api/passport-face-match/jobs/items/{row.get('FaceItemKey')}/image",
        },
        "passport": None if row.get("PassportItemKey") is None else {
            "item_key": str(row.get("PassportItemKey")),
            "sequence_no": _int_value(row.get("PassportSequenceNo")),
            "file_name": row.get("PassportFileName"),
            "image_url": f"/api/passport-face-match/jobs/items/{row.get('PassportItemKey')}/image",
        },
        "result": _json_load(row.get("ResultJson")),
    }


def _safe_image_suffix(file_name: str, content_type: str) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix in SUPPORTED_IMAGE_EXTENSIONS:
        return ".jpg" if suffix == ".jpeg" else suffix
    mapped = {
        "image/jpeg": ".jpg", "image/png": ".png", "image/bmp": ".bmp",
        "image/tiff": ".tiff", "image/webp": ".webp",
    }.get(str(content_type or "").lower())
    if mapped:
        return mapped
    raise ValueError(f"Unsupported image type: {file_name}")


def _row_dict(cursor: Any, row: Any) -> dict[str, Any]:
    return dict(zip([str(column[0]) for column in cursor.description], row))


def _normalize_uuid(value: str, field_name: str) -> UUID:
    try:
        return UUID(str(value or "").strip())
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{field_name} must be a valid UUID") from exc


def _validated_filter(value: str | None, allowed: set[str], field_name: str) -> str | None:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return None
    if normalized not in allowed:
        raise ValueError(f"Invalid {field_name}: {normalized}")
    return normalized


def _pagination(page: int, page_size: int) -> tuple[int, int, int]:
    normalized_page = max(1, int(page))
    normalized_size = min(200, max(1, int(page_size)))
    return normalized_page, normalized_size, (normalized_page - 1) * normalized_size


def _pagination_payload(page: int, page_size: int, total: int) -> dict[str, int]:
    return {
        "page": page,
        "page_size": page_size,
        "total_items": total,
        "total_pages": (total + page_size - 1) // page_size if total else 0,
    }


def _optional_text(value: Any, limit: int) -> str | None:
    text = str(value or "").strip()
    return text[:limit] if text else None


def _optional_float(value: Any) -> float | None:
    return None if value is None or value == "" else float(value)


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _json_load(value: Any) -> Any:
    if value is None or value == "":
        return None
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _int_value(value: Any) -> int:
    return int(value or 0)


def _iso_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _percent(value: int, total: int) -> float:
    return round(100.0 * value / total, 2) if total > 0 else 0.0
