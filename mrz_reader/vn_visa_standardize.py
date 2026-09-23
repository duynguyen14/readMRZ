from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from .document_orientation import env_bool
from .env_config import PROJECT_ROOT, env_value, read_env_file


DATA_DIR = Path(__file__).resolve().parent / "data"
STANDARD_FIELDS = [
    "visa_number",
    "visa_code",
    "valid_from",
    "valid_until",
    "number_of_entries",
    "passport_number",
    "date_of_birth",
    "full_name",
    "nationality",
    "issue_place",
    "issue_date",
]


@dataclass(frozen=True)
class CatalogCandidate:
    code: str
    label: str
    alias: str
    normalized_alias: str
    source: str


@dataclass(frozen=True)
class MatchResult:
    code: str | None
    score: float
    alias: str
    label: str
    source: str


class VnVisaStandardizer:
    def __init__(self, env: dict[str, str] | None = None) -> None:
        self.env = env or read_env_file()
        self.include_debug = env_bool(self.env, "READMRZ_VN_VISA_STANDARD_DEBUG", False)
        self.nationality_threshold = float(env_value(self.env, "READMRZ_VN_VISA_STANDARD_NATIONALITY_THRESHOLD", "0.72"))
        self.issue_place_threshold = float(env_value(self.env, "READMRZ_VN_VISA_STANDARD_ISSUE_PLACE_THRESHOLD", "0.68"))
        self.entries_threshold = float(env_value(self.env, "READMRZ_VN_VISA_STANDARD_ENTRIES_THRESHOLD", "0.42"))

        aliases_path = resolve_data_path(
            env_value(self.env, "READMRZ_VN_VISA_STANDARD_ALIASES_PATH", ""),
            DATA_DIR / "vn_visa_aliases.json",
        )
        nationality_path = resolve_data_path(
            env_value(self.env, "READMRZ_VN_VISA_STANDARD_NATIONALITY_CATALOG_PATH", ""),
            DATA_DIR / "vn_visa_nationalities.json",
        )
        issue_place_path = resolve_data_path(
            env_value(self.env, "READMRZ_VN_VISA_STANDARD_ISSUE_PLACE_CATALOG_PATH", ""),
            DATA_DIR / "vn_visa_issue_places.json",
        )

        self.aliases = load_json_object(aliases_path)
        self.nationality_candidates = build_catalog_candidates(
            nationality_path,
            custom_aliases=(self.aliases.get("nationality") or {}),
            source_name="nationality",
        )
        self.issue_place_candidates = build_catalog_candidates(
            issue_place_path,
            custom_aliases=(self.aliases.get("issue_place") or {}),
            source_name="issue_place",
        )
        self.entries_candidates = build_simple_candidates(self.aliases.get("number_of_entries") or {}, source_name="entries")

    def standardize(self, read_payload: dict[str, Any], *, include_debug: bool | None = None) -> dict[str, Any]:
        fields = read_payload.get("fields") if isinstance(read_payload, dict) else {}
        fields = fields if isinstance(fields, dict) else {}
        raw = {field: extract_text(fields.get(field)) for field in STANDARD_FIELDS}

        entry_match = best_match(raw.get("number_of_entries", ""), self.entries_candidates)
        nationality_match = best_match(raw.get("nationality", ""), self.nationality_candidates)
        issue_place_match = best_match(raw.get("issue_place", ""), self.issue_place_candidates)

        output: dict[str, Any] = {
            "visa_number": clean_code(raw.get("visa_number", "")),
            "visa_code": clean_code(raw.get("visa_code", "")),
            "valid_from": parse_date(raw.get("valid_from", "")),
            "valid_until": parse_date(raw.get("valid_until", "")),
            "number_of_entries": entry_match.code if entry_match.score >= self.entries_threshold else None,
            "passport_number": clean_code(raw.get("passport_number", "")),
            "date_of_birth": parse_date(raw.get("date_of_birth", "")),
            "full_name": clean_name(raw.get("full_name", "")),
            "nationality": nationality_match.code if nationality_match.score >= self.nationality_threshold else None,
            "issue_place": issue_place_match.code if issue_place_match.score >= self.issue_place_threshold else None,
            "issue_date": parse_date(raw.get("issue_date", "")),
        }

        debug_enabled = self.include_debug if include_debug is None else include_debug
        if debug_enabled:
            output["_debug"] = {
                "raw_fields": raw,
                "matches": {
                    "number_of_entries": match_to_dict(entry_match),
                    "nationality": match_to_dict(nationality_match),
                    "issue_place": match_to_dict(issue_place_match),
                },
                "thresholds": {
                    "number_of_entries": self.entries_threshold,
                    "nationality": self.nationality_threshold,
                    "issue_place": self.issue_place_threshold,
                },
                "read_payload": read_payload,
            }
        return output


_STANDARDIZER: VnVisaStandardizer | None = None


def get_vn_visa_standardizer() -> VnVisaStandardizer:
    global _STANDARDIZER
    if _STANDARDIZER is None:
        _STANDARDIZER = VnVisaStandardizer()
    return _STANDARDIZER


def standardize_vn_visa_read_payload(
    read_payload: dict[str, Any],
    *,
    include_debug: bool | None = None,
) -> dict[str, Any]:
    return get_vn_visa_standardizer().standardize(read_payload, include_debug=include_debug)


def resolve_data_path(raw_value: str, default_path: Path) -> Path:
    raw_value = str(raw_value or "").strip()
    if not raw_value:
        return default_path.resolve()
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def load_catalog_rows(path: Path) -> list[dict[str, Any]]:
    payload = load_json_object(path)
    rows = payload.get("data")
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def build_catalog_candidates(
    path: Path,
    *,
    custom_aliases: dict[str, Any],
    source_name: str,
) -> list[CatalogCandidate]:
    candidates = build_simple_candidates(custom_aliases, source_name=f"{source_name}_alias")
    seen = {(candidate.code, candidate.normalized_alias) for candidate in candidates}
    for row in load_catalog_rows(path):
        code = clean_code(row.get("code"))
        if not code:
            continue
        label = clean_text(row.get("name")) or clean_text(row.get("title")) or code
        aliases = [code, row.get("code"), row.get("name"), row.get("title")]
        for key in ("codeValueDes1", "codeValueDes2", "codeValueDes3"):
            aliases.append(row.get(key))
        for alias_value in aliases:
            alias = clean_text(alias_value)
            normalized = normalize_for_match(alias)
            if not normalized:
                continue
            key = (code, normalized)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                CatalogCandidate(
                    code=code,
                    label=label,
                    alias=alias,
                    normalized_alias=normalized,
                    source=source_name,
                )
            )
    return candidates


def build_simple_candidates(values_by_code: dict[str, Any], *, source_name: str) -> list[CatalogCandidate]:
    candidates: list[CatalogCandidate] = []
    seen: set[tuple[str, str]] = set()
    for raw_code, raw_aliases in values_by_code.items():
        code = clean_code(raw_code)
        aliases = raw_aliases if isinstance(raw_aliases, list) else [raw_aliases]
        aliases = [raw_code, *aliases]
        for alias_value in aliases:
            alias = clean_text(alias_value)
            normalized = normalize_for_match(alias)
            if not code or not normalized:
                continue
            key = (code, normalized)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                CatalogCandidate(
                    code=code,
                    label=code,
                    alias=alias,
                    normalized_alias=normalized,
                    source=source_name,
                )
            )
    return candidates


def best_match(value: str, candidates: list[CatalogCandidate]) -> MatchResult:
    normalized = normalize_for_match(value)
    if not normalized or not candidates:
        return MatchResult(None, 0.0, "", "", "")
    best_candidate: CatalogCandidate | None = None
    best_score = 0.0
    for candidate in candidates:
        score = fuzzy_score(normalized, candidate.normalized_alias)
        if score > best_score:
            best_score = score
            best_candidate = candidate
    if best_candidate is None:
        return MatchResult(None, 0.0, "", "", "")
    return MatchResult(
        code=best_candidate.code,
        score=round(min(1.0, best_score), 6),
        alias=best_candidate.alias,
        label=best_candidate.label,
        source=best_candidate.source,
    )


def fuzzy_score(value: str, target: str) -> float:
    if not value and not target:
        return 1.0
    if not value or not target:
        return 0.0
    if value == target:
        return 1.0
    if target in value and len(target) >= 2:
        return 0.96
    if value in target and len(value) >= 2:
        return 0.88
    token_score = SequenceMatcher(None, token_sort(value), token_sort(target)).ratio()
    direct_score = SequenceMatcher(None, value, target).ratio()
    return max(token_score, direct_score)


def token_sort(value: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", value)
    return "".join(sorted(tokens)) or value


def extract_text(field_payload: Any) -> str:
    if isinstance(field_payload, dict):
        return clean_text(field_payload.get("raw_text") or field_payload.get("text") or "")
    return ""


def parse_date(value: str) -> str | None:
    text = normalize_date_text(value)
    patterns = [
        (r"(?<!\d)(\d{1,2})[\/\.\-](\d{1,2})[\/\.\-](\d{4})(?!\d)", "%d-%m-%Y"),
        (r"(?<!\d)(\d{4})[\/\.\-](\d{1,2})[\/\.\-](\d{1,2})(?!\d)", "%Y-%m-%d"),
        (r"(?<!\d)(\d{2})(\d{2})(\d{4})(?!\d)", "%d-%m-%Y"),
        (r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)", "%Y-%m-%d"),
    ]
    for pattern, fmt in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        parts = match.groups()
        candidate = "-".join(parts)
        try:
            return datetime.strptime(candidate, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def normalize_date_text(value: str) -> str:
    text = clean_text(value)
    text = text.replace("O", "0").replace("o", "0")
    text = text.replace("I", "1").replace("l", "1").replace("|", "1")
    return text


def clean_code(value: Any) -> str | None:
    text = clean_text(value).upper()
    text = re.sub(r"[^A-Z0-9]+", "", text)
    return text or None


def clean_name(value: Any) -> str | None:
    text = clean_text(value).upper()
    text = re.sub(r"[^A-ZÀ-ỸĐ\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_for_match(value: str) -> str:
    text = unicodedata.normalize("NFD", str(value or "").lower())
    text = "".join(character for character in text if unicodedata.category(character) != "Mn")
    text = text.replace("đ", "d")
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def match_to_dict(match: MatchResult) -> dict[str, Any]:
    return {
        "code": match.code,
        "score": match.score,
        "alias": match.alias,
        "label": match.label,
        "source": match.source,
    }
