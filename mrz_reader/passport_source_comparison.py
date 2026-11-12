from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

from .mrz import normalize_mrz_text, parse_mrz
from .passport_classification import classify_passport_dates, coerce_date


SOURCE_MATCH_MATCHED = "matched"
SOURCE_MATCH_NOT_FOUND = "not_found"
SOURCE_MATCH_AMBIGUOUS = "ambiguous"
SOURCE_MATCH_ERROR = "error"


@dataclass(frozen=True)
class SourceCandidate:
    item_id: int
    match_priority: int
    record: dict[str, Any]
    source_path: Path | None
    file_sha256: str | None = None


@dataclass(frozen=True)
class SourceSelection:
    status: str
    candidate: SourceCandidate | None
    candidate_count: int
    reason: str


def exported_file_aliases(file_name: str) -> list[tuple[str, int]]:
    normalized = Path(str(file_name or "").strip()).name.lower()
    if not normalized:
        return []
    aliases = [(normalized, 0)]
    stem = Path(normalized).stem
    suffix = Path(normalized).suffix
    stripped = re.sub(r"_\d+$", "", stem)
    if stripped and stripped != stem:
        aliases.append((stripped + suffix, 1))
    return aliases


def source_basename(value: Any) -> str:
    normalized = str(value or "").strip().replace("/", "\\").rstrip("\\")
    return normalized.rsplit("\\", 1)[-1].lower() if normalized else ""


def resolve_source_image_path(stored_path: Any, source_root: Path) -> Path | None:
    raw_value = str(stored_path or "").strip()
    if not raw_value:
        return None
    normalized = raw_value.replace("/", "\\").lstrip("~")
    path = Path(normalized)
    if path.is_absolute():
        return path.resolve()
    return (source_root / normalized.lstrip("\\")).resolve()


def deduplicate_source_candidates(candidates: Iterable[SourceCandidate]) -> list[SourceCandidate]:
    by_path: dict[tuple[int, str], SourceCandidate] = {}
    without_path: list[SourceCandidate] = []
    for candidate in candidates:
        raw_path = str(candidate.record.get("FullPassportImage") or "").strip().replace("/", "\\").lower()
        if not raw_path:
            without_path.append(candidate)
            continue
        key = (candidate.item_id, raw_path)
        previous = by_path.get(key)
        if previous is None or int(candidate.record.get("Id") or 0) > int(previous.record.get("Id") or 0):
            by_path[key] = candidate
    return list(by_path.values()) + without_path


def select_source_candidate(
    candidates: Iterable[SourceCandidate],
    *,
    expected_country: str | None,
    expected_validity: str | None,
    reference_date: date,
    exported_sha256: str | None,
) -> SourceSelection:
    unique = deduplicate_source_candidates(candidates)
    if not unique:
        return SourceSelection(SOURCE_MATCH_NOT_FOUND, None, 0, "No source row matched the exported file name")

    expected_hash = str(exported_sha256 or "").strip().lower()
    hash_matches = [
        candidate
        for candidate in unique
        if expected_hash and str(candidate.file_sha256 or "").strip().lower() == expected_hash
    ]
    if len(hash_matches) == 1:
        return SourceSelection(SOURCE_MATCH_MATCHED, hash_matches[0], len(unique), "Unique SHA-256 match")
    if len(hash_matches) > 1:
        newest = max(hash_matches, key=lambda item: int(item.record.get("Id") or 0))
        return SourceSelection(SOURCE_MATCH_MATCHED, newest, len(unique), "Same source file hash; newest source row selected")

    scored: list[tuple[int, SourceCandidate]] = []
    normalized_country = normalize_code(expected_country)
    for candidate in unique:
        record = candidate.record
        score = 100 - (20 * max(0, int(candidate.match_priority)))
        source_country = normalize_code(record.get("NationalityCode"))
        if normalized_country and source_country == normalized_country:
            score += 20
        source_classification = classify_passport_dates(
            record.get("Dob"),
            record.get("PassportExpireDate"),
            reference_date=reference_date,
        )
        if expected_validity and source_classification["validity_status"] == expected_validity:
            score += 15
        scored.append((score, candidate))

    best_score = max(score for score, _ in scored)
    best = [candidate for score, candidate in scored if score == best_score]
    if len(best) == 1:
        return SourceSelection(SOURCE_MATCH_MATCHED, best[0], len(unique), f"Unique metadata score {best_score}")
    return SourceSelection(
        SOURCE_MATCH_AMBIGUOUS,
        None,
        len(unique),
        f"{len(best)} source rows tied at metadata score {best_score}",
    )


def parse_source_record(record: dict[str, Any]) -> dict[str, Any]:
    line_one = text_or_none(record.get("MrzlineOne"))
    line_two = text_or_none(record.get("MrzlineTwo"))
    parsed = parse_mrz([line_one or "", line_two or ""])
    parsed_fields = parsed.fields if parsed is not None else {}
    surname = text_or_none(parsed_fields.get("surname"))
    given_names = text_or_none(parsed_fields.get("given_names"))
    parsed_full_name = " ".join(value for value in (surname, given_names) if value) or None
    return {
        "transaction_evisa_id": int(record["Id"]),
        "transaction_evisa_guid": text_or_none(record.get("GUID")),
        "source_created_date": record.get("CreatedDate"),
        "source_modified_date": record.get("ModifiedDate"),
        "full_passport_image": text_or_none(record.get("FullPassportImage")),
        "face_image": text_or_none(record.get("FaceImage")),
        "mrz_line_one": line_one,
        "mrz_line_two": line_two,
        "mrz_line_one_point": float_or_none(record.get("MrzlineOnePoint")),
        "mrz_line_two_point": float_or_none(record.get("MrzlineTwoPoint")),
        # The evaluation pipeline stores ICAO document type (for example TD3), while
        # TransactionEVisa.PassportType usually stores the document code (for example P).
        "passport_type": first_nonempty(None if parsed is None else parsed.document_type, record.get("PassportType")),
        "database_passport_type": text_or_none(record.get("PassportType")),
        "issuing_country": text_or_none(parsed_fields.get("issuing_country")),
        "surname": surname,
        "given_names": given_names,
        # Compare in MRZ order (surname, given names); retain the raw database name too.
        "full_name": first_nonempty(parsed_full_name, record.get("Fullname")),
        "database_full_name": text_or_none(record.get("Fullname")),
        "passport_number": first_nonempty(record.get("PassportNo"), parsed_fields.get("document_number")),
        "nationality": first_nonempty(record.get("NationalityCode"), parsed_fields.get("nationality")),
        "nationality_original": text_or_none(record.get("NationalityOriginalCode")),
        "date_of_birth": coerce_date(record.get("Dob")) or coerce_date(parsed_fields.get("birth_date")),
        "sex": first_nonempty(record.get("Gender"), parsed_fields.get("sex")),
        "date_of_expiry": coerce_date(record.get("PassportExpireDate")) or coerce_date(parsed_fields.get("expiry_date")),
        "issue_passport_date": coerce_date(record.get("IssuePassportDate")),
        "issue_passport_place": text_or_none(record.get("IssuePassportPlace")),
        "personal_number": text_or_none(parsed_fields.get("optional_data")),
        "passport_image_note": text_or_none(record.get("PassportImageNote")),
        "face_image_note": text_or_none(record.get("FaceImageNote")),
        "source_mrz_checksum_valid": None if parsed is None else bool(parsed.checksum_pass),
        "source_mrz_document_type": None if parsed is None else parsed.document_type,
    }


def compare_passport_fields(source: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, dict[str, Any]] = {}
    compared_count = 0
    matched_count = 0
    for field_name, normalizer in FIELD_COMPARATORS.items():
        source_value = source.get(field_name)
        observed_value = observed.get(field_name)
        normalized_source = normalizer(source_value)
        normalized_observed = normalizer(observed_value)
        comparable = normalized_source is not None and normalized_observed is not None
        matched = None if not comparable else normalized_source == normalized_observed
        if comparable:
            compared_count += 1
            matched_count += int(bool(matched))
        fields[field_name] = {
            "source": serializable(source_value),
            "observed": serializable(observed_value),
            "normalized_source": serializable(normalized_source),
            "normalized_observed": serializable(normalized_observed),
            "comparable": comparable,
            "match": matched,
        }
    return {
        "compared_field_count": compared_count,
        "matched_field_count": matched_count,
        "all_compared_fields_match": None if compared_count == 0 else matched_count == compared_count,
        "fields": fields,
    }


def evaluation_observed_fields(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "mrz_line_one": item.get("MrzLineOne"),
        "mrz_line_two": item.get("MrzLineTwo"),
        "passport_type": item.get("PassportType"),
        "issuing_country": item.get("IssuingCountry"),
        "surname": item.get("Surname"),
        "given_names": item.get("GivenNames"),
        "full_name": item.get("FullName"),
        "passport_number": item.get("PassportNumber"),
        "nationality": item.get("Nationality"),
        "date_of_birth": item.get("DateOfBirth"),
        "sex": item.get("Sex"),
        "date_of_expiry": item.get("DateOfExpiry"),
        "personal_number": item.get("PersonalNumber"),
    }


def normalize_code(value: Any) -> str | None:
    normalized = re.sub(r"[^A-Z0-9]", "", str(value or "").upper().replace("<", ""))
    return normalized or None


def normalize_identifier(value: Any) -> str | None:
    normalized = re.sub(r"[^A-Z0-9]", "", str(value or "").upper().replace("<", ""))
    return normalized or None


def normalize_name(value: Any) -> str | None:
    text = str(value or "").strip().upper().replace("<", " ")
    if not text:
        return None
    # Vietnamese D with stroke does not decompose under NFKD.
    text = text.replace("Đ", "D")
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(character for character in decomposed if not unicodedata.combining(character))
    normalized = re.sub(r"[^A-Z0-9]+", " ", ascii_text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized or None


def normalize_date(value: Any) -> str | None:
    normalized = coerce_date(value)
    return normalized.isoformat() if normalized is not None else None


FIELD_COMPARATORS = {
    "mrz_line_one": lambda value: normalize_mrz_text(str(value or "")) or None,
    "mrz_line_two": lambda value: normalize_mrz_text(str(value or "")) or None,
    "passport_type": normalize_code,
    "issuing_country": normalize_code,
    "surname": normalize_name,
    "given_names": normalize_name,
    "full_name": normalize_name,
    "passport_number": normalize_identifier,
    "nationality": normalize_code,
    "date_of_birth": normalize_date,
    "sex": normalize_code,
    "date_of_expiry": normalize_date,
    "personal_number": normalize_identifier,
}


def text_or_none(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def first_nonempty(*values: Any) -> str | None:
    for value in values:
        normalized = text_or_none(value)
        if normalized is not None:
            return normalized
    return None


def float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def serializable(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value
