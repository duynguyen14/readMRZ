from __future__ import annotations

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from mrz_reader.passport_classification import (
    VALIDITY_EXPIRED,
    VALIDITY_SIX_MONTHS_OR_MORE,
    VALIDITY_UNDER_SIX_MONTHS,
    classify_passport_dates,
    country_match,
)
from mrz_reader.passport_evaluation import (
    discover_passport_images,
    infer_export_reference_date,
)


class PassportEvaluationTests(unittest.TestCase):
    def test_discovers_only_passport_images_in_known_export_groups(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            expected = [
                ("CHN", "hochieuhethan", "expired.jpg", VALIDITY_EXPIRED),
                ("CHN", "conhanduoi6thang", "under.png", VALIDITY_UNDER_SIX_MONTHS),
                ("USA", "conhantren6thang", "valid.jpeg", VALIDITY_SIX_MONTHS_OR_MORE),
            ]
            for country, group, file_name, _ in expected:
                passport_dir = root / country / group / "passport"
                passport_dir.mkdir(parents=True, exist_ok=True)
                (passport_dir / file_name).write_bytes(b"image")
                face_dir = root / country / group / "face"
                face_dir.mkdir(parents=True, exist_ok=True)
                (face_dir / file_name).write_bytes(b"face")

            unknown_dir = root / "CHN" / "unknown_group" / "passport"
            unknown_dir.mkdir(parents=True)
            (unknown_dir / "ignored.jpg").write_bytes(b"ignored")
            (root / "CHN" / "hochieuhethan" / "passport" / "note.txt").write_text("x")

            result = discover_passport_images(root)

            self.assertEqual(len(result.candidates), 3)
            self.assertEqual(result.passport_directories, 4)
            self.assertEqual(len(result.ignored_passport_directories), 1)
            self.assertEqual(
                {
                    (
                        candidate.country_code,
                        candidate.source_group,
                        candidate.image_path.name,
                        candidate.expected_validity_status,
                    )
                    for candidate in result.candidates
                },
                set(expected),
            )

    def test_country_filter_and_limit_are_applied_after_stable_sort(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for country in ("USA", "CHN"):
                passport_dir = root / country / "hochieuhethan" / "passport"
                passport_dir.mkdir(parents=True)
                for name in ("b.jpg", "a.jpg"):
                    (passport_dir / name).write_bytes(name.encode("ascii"))

            result = discover_passport_images(root, country_codes=["chn"], limit=1)

            self.assertEqual(len(result.candidates), 1)
            self.assertEqual(result.candidates[0].country_code, "CHN")
            self.assertEqual(result.candidates[0].image_path.name, "a.jpg")

    def test_classification_uses_exact_age_and_calendar_month_boundaries(self) -> None:
        reference = date(2026, 11, 11)
        child = classify_passport_dates("2012-11-12", "2027-05-10", reference_date=reference)
        adult = classify_passport_dates("2012-11-11", "2027-05-11", reference_date=reference)

        self.assertEqual(child["age_group"], "child")
        self.assertEqual(child["validity_status"], VALIDITY_UNDER_SIX_MONTHS)
        self.assertEqual(adult["age_group"], "adult")
        self.assertEqual(adult["validity_status"], VALIDITY_SIX_MONTHS_OR_MORE)

    def test_country_match_checks_nationality_or_issuer(self) -> None:
        self.assertTrue(country_match("CHN", "chn", "UTO"))
        self.assertTrue(country_match("CHN", "USA", "CHN"))
        self.assertFalse(country_match("CHN", "USA", "UTO"))
        self.assertIsNone(country_match("CHN", None, None))

    def test_reference_date_is_inferred_from_quick_export_folder(self) -> None:
        path = Path("C:/exports/quick_20261016_094736/quick_20261016_094736")
        self.assertEqual(infer_export_reference_date(path), date(2026, 10, 16))
        self.assertIsNone(infer_export_reference_date(Path("C:/exports/manual")))


if __name__ == "__main__":
    unittest.main()
