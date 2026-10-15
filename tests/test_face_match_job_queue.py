from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from mrz_reader.face_match_job_queue import (
    ClaimedFaceMatchItem,
    FaceMatchJobRepository,
    build_optimal_pairs,
    hungarian_maximize,
)


def candidate(
    item_id: int,
    classification: str,
    embeddings: list[list[float]],
) -> dict:
    return {
        "id": item_id,
        "item_key": f"item-{item_id}",
        "sequence_no": item_id,
        "file_name": f"image-{item_id}.jpg",
        "classification": classification,
        "embeddings": np.asarray(embeddings, dtype=np.float32),
        "result": {"parsed": {"passport_no": f"P{item_id}"}},
    }


class HungarianMatchingTests(unittest.TestCase):
    def test_hungarian_uses_global_optimum_instead_of_greedy_pairing(self) -> None:
        weights = [[0.90, 0.80], [0.85, 0.10]]

        assignments = set(hungarian_maximize(weights))

        self.assertEqual(assignments, {(0, 1), (1, 0)})

    def test_hungarian_supports_more_rows_than_columns(self) -> None:
        weights = [[0.10, 0.80], [0.90, 0.20], [0.30, 0.40]]

        assignments = set(hungarian_maximize(weights))

        self.assertEqual(assignments, {(0, 1), (1, 0)})

    def test_optimal_pairs_returns_matches_and_unmatched_items(self) -> None:
        candidates = [
            candidate(1, "face", [[1.0, 0.0]]),
            candidate(2, "face", [[0.0, 1.0]]),
            candidate(3, "passport", [[1.0, 0.0]]),
        ]

        pairs = build_optimal_pairs(candidates, match_threshold=0.80, review_threshold=0.50)

        self.assertEqual(sum(pair["decision"] == "match" for pair in pairs), 1)
        self.assertEqual(sum(pair["decision"] == "unmatched_face" for pair in pairs), 1)
        match = next(pair for pair in pairs if pair["decision"] == "match")
        self.assertEqual((match["face_item_id"], match["passport_item_id"]), (1, 3))
        self.assertAlmostEqual(match["score"], 1.0)

    def test_score_between_thresholds_requires_review(self) -> None:
        candidates = [
            candidate(1, "face", [[1.0, 0.0]]),
            candidate(2, "passport", [[0.4, np.sqrt(0.84)]]),
        ]

        pairs = build_optimal_pairs(candidates, match_threshold=0.50, review_threshold=0.35)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["decision"], "review")
        self.assertTrue(pairs[0]["result"]["match"]["review_required"])

    def test_missing_embeddings_are_never_matched(self) -> None:
        candidates = [
            candidate(1, "face", []),
            candidate(2, "passport", [[1.0, 0.0]]),
        ]

        pairs = build_optimal_pairs(candidates, match_threshold=0.0, review_threshold=-1.0)

        self.assertEqual(
            {pair["decision"] for pair in pairs},
            {"unmatched_face", "unmatched_passport"},
        )


class EmbeddingStorageTests(unittest.TestCase):
    def test_embedding_round_trip_and_path_guard(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            repository = FaceMatchJobRepository(Path(temporary_directory))
            item = ClaimedFaceMatchItem(
                id=1,
                item_key="7db9c5a1-30cd-4be0-b1fc-89c8b3341af9",
                job_id=1,
                job_key="e21f6e91-407b-4ea5-acff-2a4b5de36ba7",
                original_file_name="face.jpg",
                relative_image_path="unused.jpg",
                content_type="image/jpeg",
                attempt_count=1,
                max_attempts=3,
            )
            expected = np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32)

            relative_path = repository.save_embeddings(item, expected)
            actual = repository.load_embeddings(relative_path)

            np.testing.assert_allclose(actual, expected)
            with self.assertRaises(ValueError):
                repository.absolute_path("../outside.npy")


if __name__ == "__main__":
    unittest.main()
