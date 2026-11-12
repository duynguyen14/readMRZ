from __future__ import annotations

from datetime import date
from pathlib import Path
import unittest

from mrz_reader.passport_source_comparison import (
    SOURCE_MATCH_AMBIGUOUS,
    SOURCE_MATCH_MATCHED,
    SourceCandidate,
    compare_passport_fields,
    exported_file_aliases,
    parse_source_record,
    select_source_candidate,
)


class PassportSourceComparisonTests(unittest.TestCase):
    def test_exported_file_aliases_keep_exact_name_before_collision_suffix_fallback(self) -> None:
        self.assertEqual(
            exported_file_aliases("ABC_1.JPG"),
            [("abc_1.jpg", 0), ("abc.jpg", 1)],
        )
        self.assertEqual(exported_file_aliases("ABC.JPG"), [("abc.jpg", 0)])

    def test_unique_country_and_validity_candidate_is_selected(self) -> None:
        candidates = [
            SourceCandidate(
                item_id=1,
                match_priority=0,
                record={
                    "Id": 10,
                    "NationalityCode": "CHN",
                    "PassportExpireDate": date(2027, 1, 1),
                    "Dob": date(1990, 1, 1),
                    "FullPassportImage": "a/x.jpg",
                },
                source_path=Path("a/x.jpg"),
            ),
            SourceCandidate(
                item_id=1,
                match_priority=0,
                record={
                    "Id": 20,
                    "NationalityCode": "USA",
                    "PassportExpireDate": date(2027, 1, 1),
                    "Dob": date(1990, 1, 1),
                    "FullPassportImage": "b/x.jpg",
                },
                source_path=Path("b/x.jpg"),
            ),
        ]
        selection = select_source_candidate(
            candidates,
            expected_country="CHN",
            expected_validity="expiring_within_6_months",
            reference_date=date(2026, 10, 16),
            exported_sha256=None,
        )
        self.assertEqual(selection.status, SOURCE_MATCH_MATCHED)
        self.assertEqual(selection.candidate.record["Id"], 10)

    def test_tied_candidates_are_not_selected(self) -> None:
        candidates = [
            SourceCandidate(
                item_id=1,
                match_priority=0,
                record={
                    "Id": value,
                    "NationalityCode": "CHN",
                    "PassportExpireDate": date(2030, 1, 1),
                    "Dob": date(1990, 1, 1),
                    "FullPassportImage": f"folder{value}/x.jpg",
                },
                source_path=Path(f"folder{value}/x.jpg"),
            )
            for value in (10, 20)
        ]
        selection = select_source_candidate(
            candidates,
            expected_country="CHN",
            expected_validity="valid_6_months_or_more",
            reference_date=date(2026, 10, 16),
            exported_sha256=None,
        )
        self.assertEqual(selection.status, SOURCE_MATCH_AMBIGUOUS)
        self.assertIsNone(selection.candidate)

    def test_same_source_path_uses_newest_transaction_row(self) -> None:
        candidates = [
            SourceCandidate(
                item_id=1,
                match_priority=0,
                record={
                    "Id": value,
                    "NationalityCode": "CHN",
                    "PassportExpireDate": date(2030, 1, 1),
                    "Dob": date(1990, 1, 1),
                    "FullPassportImage": "same/path/x.jpg",
                },
                source_path=Path("same/path/x.jpg"),
            )
            for value in (10, 20)
        ]
        selection = select_source_candidate(
            candidates,
            expected_country="CHN",
            expected_validity="valid_6_months_or_more",
            reference_date=date(2026, 10, 16),
            exported_sha256=None,
        )
        self.assertEqual(selection.status, SOURCE_MATCH_MATCHED)
        self.assertEqual(selection.candidate.record["Id"], 20)

    def test_comparison_normalizes_mrz_names_identifiers_and_dates(self) -> None:
        source = {
            "mrz_line_one": "P<CHNDOE<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
            "mrz_line_two": "E1234567<0CHN9001011M3001019<<<<<<<<<<<<<<04",
            "passport_type": "P<",
            "issuing_country": "CHN",
            "surname": "Đỗ",
            "given_names": "John",
            "full_name": "Đỗ John",
            "passport_number": "E1234567<",
            "nationality": "CHN",
            "date_of_birth": date(1990, 1, 1),
            "sex": "M",
            "date_of_expiry": date(2030, 1, 1),
            "personal_number": "ABC<123",
        }
        observed = {
            **source,
            "passport_type": "P",
            "surname": "DO",
            "full_name": "DO JOHN",
            "passport_number": "E1234567",
            "date_of_birth": "1990-01-01",
            "date_of_expiry": "2030-01-01",
            "personal_number": "ABC123",
        }
        comparison = compare_passport_fields(source, observed)
        self.assertTrue(comparison["all_compared_fields_match"])
        self.assertEqual(comparison["matched_field_count"], comparison["compared_field_count"])

    def test_source_record_uses_database_fields_and_parsed_mrz(self) -> None:
        source = parse_source_record(
            {
                "Id": 42,
                "GUID": "source-guid",
                "FullPassportImage": "folder/passport.jpg",
                "MrzlineOne": "P<CHNDOE<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
                "MrzlineTwo": "E1234567<0CHN9001011M3001019<<<<<<<<<<<<<<04",
                "PassportNo": "E1234567",
                "PassportType": "P",
                "Fullname": "DOE JOHN",
                "Dob": date(1990, 1, 1),
                "Gender": "M",
                "NationalityCode": "CHN",
                "PassportExpireDate": date(2030, 1, 1),
            }
        )
        self.assertEqual(source["transaction_evisa_id"], 42)
        self.assertEqual(source["passport_number"], "E1234567")
        self.assertEqual(source["nationality"], "CHN")
        self.assertEqual(source["date_of_expiry"], date(2030, 1, 1))


if __name__ == "__main__":
    unittest.main()
