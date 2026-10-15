from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mrz_reader.passport_batch_queue import (
    PassportBatchRepository,
    _item_identifier_lookup,
    _pagination,
    _statistics_payload,
    _validated_filter,
    parse_multipart_form,
)


class MultipartFormTests(unittest.TestCase):
    def test_parses_multiple_files_and_fields(self) -> None:
        boundary = "readmrz-test-boundary"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="client_request_id"\r\n\r\n'
            "request-001\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="files"; filename="passport-1.jpg"\r\n'
            "Content-Type: image/jpeg\r\n\r\n"
        ).encode("ascii") + b"jpeg-one" + (
            f"\r\n--{boundary}\r\n"
            'Content-Disposition: form-data; name="files"; filename="folder/passport-2.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode("ascii") + b"png-two" + f"\r\n--{boundary}--\r\n".encode("ascii")

        files, fields = parse_multipart_form(
            f"multipart/form-data; boundary={boundary}",
            body,
        )

        self.assertEqual(fields["client_request_id"], "request-001")
        self.assertEqual([item.file_name for item in files], ["passport-1.jpg", "passport-2.png"])
        self.assertEqual([item.data for item in files], [b"jpeg-one", b"png-two"])

    def test_rejects_request_without_files(self) -> None:
        boundary = "readmrz-empty-boundary"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="source_system"\r\n\r\n'
            "test\r\n"
            f"--{boundary}--\r\n"
        ).encode("ascii")

        with self.assertRaisesRegex(ValueError, "At least one image file"):
            parse_multipart_form(f"multipart/form-data; boundary={boundary}", body)

    def test_stored_path_cannot_escape_storage_root(self) -> None:
        with TemporaryDirectory() as directory:
            repository = PassportBatchRepository(Path(directory))
            with self.assertRaisesRegex(ValueError, "escapes the configured root"):
                repository.absolute_image_path("../outside.jpg")

    def test_statistics_keep_processing_and_review_counts_separate(self) -> None:
        statistics = _statistics_payload(
            {
                "TotalItems": 10,
                "QueuedItems": 1,
                "ProcessingItems": 1,
                "RetryWaitItems": 1,
                "SucceededItems": 6,
                "FailedItems": 1,
                "CancelledItems": 0,
                "NotReadyItems": 4,
                "PendingReviewItems": 2,
                "ApprovedItems": 2,
                "CorrectedItems": 1,
                "RejectedItems": 1,
            }
        )

        self.assertEqual(statistics["reviewed"], 4)
        self.assertEqual(statistics["unreviewed"], 2)
        self.assertEqual(statistics["processing_progress_percent"], 70.0)
        self.assertEqual(statistics["review_progress_percent"], 66.67)

    def test_filters_and_pagination_are_bounded(self) -> None:
        self.assertEqual(_pagination(0, 999), (1, 200, 0))
        with self.assertRaisesRegex(ValueError, "Invalid status"):
            _validated_filter("unknown", {"queued"}, "status")

    def test_review_identifier_supports_numeric_id_and_uuid(self) -> None:
        self.assertEqual(_item_identifier_lookup(12), ("item.Id = ?", 12))
        lookup, value = _item_identifier_lookup("00000000-0000-0000-0000-000000000012")
        self.assertEqual(lookup, "item.ItemKey = ?")
        self.assertEqual(value, "00000000-0000-0000-0000-000000000012")

    def test_review_rejects_unknown_decision_before_database_access(self) -> None:
        with TemporaryDirectory() as directory:
            repository = PassportBatchRepository(Path(directory))
            with self.assertRaisesRegex(ValueError, "approved, corrected or rejected"):
                repository.update_review_status(1, "pending")


if __name__ == "__main__":
    unittest.main()
