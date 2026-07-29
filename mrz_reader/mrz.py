from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Any


MRZ_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
WEIGHTS = (7, 3, 1)


@dataclass
class CheckResult:
    name: str
    expected: str
    actual: str | None
    passed: bool


@dataclass
class MrzParseResult:
    document_type: str
    lines: list[str]
    fields: dict[str, str | None]
    checks: list[CheckResult]
    checksum_pass: bool
    format_score: float


def normalize_mrz_text(text: str) -> str:
    text = text.upper()
    replacements = {
        "«": "<",
        "‹": "<",
        ">": "<",
        " ": "",
        "\t": "",
        "\r": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return "".join(ch for ch in text if ch in MRZ_CHARS)


def normalize_lines(raw_lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    for raw in raw_lines:
        line = normalize_mrz_text(raw)
        if len(line) >= 20:
            cleaned.append(line)
    return cleaned


def mrz_char_value(ch: str) -> int:
    if ch == "<":
        return 0
    if "0" <= ch <= "9":
        return ord(ch) - ord("0")
    if "A" <= ch <= "Z":
        return ord(ch) - ord("A") + 10
    return 0


def check_digit(data: str) -> str:
    total = 0
    for idx, ch in enumerate(data):
        total += mrz_char_value(ch) * WEIGHTS[idx % 3]
    return str(total % 10)


def compact_date(value: str | None) -> str | None:
    if not value or not re.fullmatch(r"\d{6}", value):
        return None
    yy = int(value[:2])
    mm = int(value[2:4])
    dd = int(value[4:6])
    current_yy = int(datetime.utcnow().strftime("%y"))
    century = 2000 if yy <= current_yy + 10 else 1900
    try:
        return datetime(century + yy, mm, dd).strftime("%Y-%m-%d")
    except ValueError:
        return None


def clean_field(value: str) -> str | None:
    cleaned = value.replace("<", " ").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or None


def split_names(value: str) -> tuple[str | None, str | None]:
    parts = value.split("<<", 1)
    surname = clean_field(parts[0]) if parts else None
    given = clean_field(parts[1]) if len(parts) > 1 else None
    return surname, given


def check(name: str, payload: str, actual: str | None) -> CheckResult:
    expected = check_digit(payload)
    return CheckResult(name=name, expected=expected, actual=actual, passed=actual == expected)


def parse_td3(lines: list[str]) -> MrzParseResult:
    l1 = lines[0].ljust(44, "<")[:44]
    l2 = lines[1].ljust(44, "<")[:44]
    surname, given_names = split_names(l1[5:44])
    checks = [
        check("document_number", l2[0:9], l2[9:10]),
        check("birth_date", l2[13:19], l2[19:20]),
        check("expiry_date", l2[21:27], l2[27:28]),
        check("composite", l2[0:10] + l2[13:20] + l2[21:43], l2[43:44]),
    ]
    fields: dict[str, str | None] = {
        "issuing_country": l1[2:5],
        "surname": surname,
        "given_names": given_names,
        "document_number": clean_field(l2[0:9]),
        "nationality": l2[10:13],
        "birth_date": compact_date(l2[13:19]),
        "sex": clean_field(l2[20:21]),
        "expiry_date": compact_date(l2[21:27]),
        "optional_data": clean_field(l2[28:43]),
    }
    return build_result("TD3", [l1, l2], fields, checks)


def parse_mrv_a(lines: list[str]) -> MrzParseResult:
    l1 = lines[0].ljust(44, "<")[:44]
    l2 = lines[1].ljust(44, "<")[:44]
    surname, given_names = split_names(l1[5:44])
    checks = [
        check("document_number", l2[0:9], l2[9:10]),
        check("birth_date", l2[13:19], l2[19:20]),
        check("expiry_date", l2[21:27], l2[27:28]),
    ]
    fields: dict[str, str | None] = {
        "issuing_country": l1[2:5],
        "surname": surname,
        "given_names": given_names,
        "document_number": clean_field(l2[0:9]),
        "nationality": l2[10:13],
        "birth_date": compact_date(l2[13:19]),
        "sex": clean_field(l2[20:21]),
        "expiry_date": compact_date(l2[21:27]),
        "optional_data": clean_field(l2[28:44]),
    }
    return build_result("MRV-A", [l1, l2], fields, checks)


def parse_mrv_b(lines: list[str]) -> MrzParseResult:
    l1 = lines[0].ljust(36, "<")[:36]
    l2 = lines[1].ljust(36, "<")[:36]
    surname, given_names = split_names(l1[5:36])
    checks = [
        check("document_number", l2[0:9], l2[9:10]),
        check("birth_date", l2[13:19], l2[19:20]),
        check("expiry_date", l2[21:27], l2[27:28]),
    ]
    fields: dict[str, str | None] = {
        "issuing_country": l1[2:5],
        "surname": surname,
        "given_names": given_names,
        "document_number": clean_field(l2[0:9]),
        "nationality": l2[10:13],
        "birth_date": compact_date(l2[13:19]),
        "sex": clean_field(l2[20:21]),
        "expiry_date": compact_date(l2[21:27]),
        "optional_data": clean_field(l2[28:36]),
    }
    return build_result("MRV-B", [l1, l2], fields, checks)


def parse_td1(lines: list[str]) -> MrzParseResult:
    l1 = lines[0].ljust(30, "<")[:30]
    l2 = lines[1].ljust(30, "<")[:30]
    l3 = lines[2].ljust(30, "<")[:30]
    surname, given_names = split_names(l3)
    checks = [
        check("document_number", l1[5:14], l1[14:15]),
        check("birth_date", l2[0:6], l2[6:7]),
        check("expiry_date", l2[8:14], l2[14:15]),
        check("composite", l1[5:30] + l2[0:7] + l2[8:15] + l2[18:29], l2[29:30]),
    ]
    fields: dict[str, str | None] = {
        "issuing_country": l1[2:5],
        "surname": surname,
        "given_names": given_names,
        "document_number": clean_field(l1[5:14]),
        "nationality": l2[15:18],
        "birth_date": compact_date(l2[0:6]),
        "sex": clean_field(l2[7:8]),
        "expiry_date": compact_date(l2[8:14]),
        "optional_data": clean_field(l1[15:30] + l2[18:29]),
    }
    return build_result("TD1", [l1, l2, l3], fields, checks)


def build_result(
    document_type: str,
    lines: list[str],
    fields: dict[str, str | None],
    checks: list[CheckResult],
) -> MrzParseResult:
    passed = sum(1 for item in checks if item.passed)
    charset_hits = sum(ch in MRZ_CHARS for line in lines for ch in line)
    total_chars = max(1, sum(len(line) for line in lines))
    checksum_score = passed / max(1, len(checks))
    format_score = (0.55 * checksum_score) + (0.45 * (charset_hits / total_chars))
    return MrzParseResult(
        document_type=document_type,
        lines=lines,
        fields=fields,
        checks=checks,
        checksum_pass=passed == len(checks),
        format_score=round(format_score, 4),
    )


def parse_mrz(lines: list[str]) -> MrzParseResult | None:
    normalized = normalize_lines(lines)
    if len(normalized) >= 3:
        td1 = [line for line in normalized if 27 <= len(line) <= 33]
        if len(td1) >= 3:
            return parse_td1(td1[:3])

    two_line = [line for line in normalized if len(line) >= 30]
    if len(two_line) < 2:
        return None

    selected = two_line[:2]
    lengths = [len(line) for line in selected]
    first = selected[0]
    if first.startswith("V"):
        if max(lengths) <= 38:
            return parse_mrv_b(selected)
        return parse_mrv_a(selected)
    if max(lengths) <= 38:
        return parse_mrv_b(selected)
    return parse_td3(selected)


def result_to_dict(
    parsed: MrzParseResult | None,
    *,
    raw_lines: list[str],
    ocr_score: float,
    detector_score: float,
    latency_ms: int,
    detector_latency_ms: int = 0,
    ocr_latency_ms: int = 0,
    parse_latency_ms: int = 0,
    candidates_evaluated: int = 0,
    ocr_passes: int = 0,
) -> dict[str, Any]:
    if parsed is None:
        final_confidence = round(0.45 * ocr_score + 0.2 * detector_score, 4)
        return {
            "found": False,
            "mrz_raw": raw_lines,
            "document_type": None,
            "confidence": final_confidence,
            "ocr_confidence": round(ocr_score, 4),
            "detector_confidence": round(detector_score, 4),
            "checksum_pass": False,
            "latency_ms": latency_ms,
            "detector_latency_ms": detector_latency_ms,
            "ocr_latency_ms": ocr_latency_ms,
            "parse_latency_ms": parse_latency_ms,
            "candidates_evaluated": candidates_evaluated,
            "ocr_passes": ocr_passes,
            "fields": {},
            "checks": [],
        }

    final_confidence = (
        0.45 * ocr_score + 0.20 * detector_score + 0.35 * parsed.format_score
    )
    return {
        "found": True,
        "mrz_raw": parsed.lines,
        "document_type": parsed.document_type,
        "confidence": round(min(0.999, final_confidence), 4),
        "ocr_confidence": round(ocr_score, 4),
        "detector_confidence": round(detector_score, 4),
        "checksum_pass": parsed.checksum_pass,
        "latency_ms": latency_ms,
        "detector_latency_ms": detector_latency_ms,
        "ocr_latency_ms": ocr_latency_ms,
        "parse_latency_ms": parse_latency_ms,
        "candidates_evaluated": candidates_evaluated,
        "ocr_passes": ocr_passes,
        "fields": parsed.fields,
        "checks": [check_item.__dict__ for check_item in parsed.checks],
    }
