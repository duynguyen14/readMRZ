from __future__ import annotations

import calendar
from datetime import date, datetime
from typing import Any


AGE_GROUP_CHILD = "child"
AGE_GROUP_ADULT = "adult"
AGE_GROUP_UNKNOWN = "unknown"

VALIDITY_EXPIRED = "expired"
VALIDITY_UNDER_SIX_MONTHS = "expiring_within_6_months"
VALIDITY_SIX_MONTHS_OR_MORE = "valid_6_months_or_more"
VALIDITY_UNKNOWN = "unknown"


def classify_passport_dates(
    birth_date: Any,
    expiry_date: Any,
    *,
    reference_date: date | None = None,
) -> dict[str, Any]:
    current_date = reference_date or date.today()
    normalized_birth_date = coerce_date(birth_date)
    normalized_expiry_date = coerce_date(expiry_date)

    age_years: int | None = None
    age_group = AGE_GROUP_UNKNOWN
    if normalized_birth_date is not None and normalized_birth_date <= current_date:
        age_years = current_date.year - normalized_birth_date.year
        if current_date < add_years(normalized_birth_date, age_years):
            age_years -= 1
        age_group = (
            AGE_GROUP_CHILD
            if current_date < add_years(normalized_birth_date, 14)
            else AGE_GROUP_ADULT
        )

    validity_status = VALIDITY_UNKNOWN
    days_to_expiry: int | None = None
    if normalized_expiry_date is not None:
        days_to_expiry = (normalized_expiry_date - current_date).days
        if normalized_expiry_date < current_date:
            validity_status = VALIDITY_EXPIRED
        elif normalized_expiry_date < add_months(current_date, 6):
            validity_status = VALIDITY_UNDER_SIX_MONTHS
        else:
            validity_status = VALIDITY_SIX_MONTHS_OR_MORE

    return {
        "as_of_date": current_date.isoformat(),
        "birth_date": normalized_birth_date,
        "expiry_date": normalized_expiry_date,
        "age_years": age_years,
        "age_group": age_group,
        "validity_status": validity_status,
        "days_to_expiry": days_to_expiry,
    }


def coerce_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip()[:10])
        except (TypeError, ValueError):
            return None
    return None


def add_years(value: date, years: int) -> date:
    target_year = value.year + years
    target_day = min(value.day, calendar.monthrange(target_year, value.month)[1])
    return date(target_year, value.month, target_day)


def add_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    target_year = value.year + month_index // 12
    target_month = month_index % 12 + 1
    target_day = min(value.day, calendar.monthrange(target_year, target_month)[1])
    return date(target_year, target_month, target_day)


def country_match(
    expected_country: str | None,
    nationality: str | None,
    issuing_country: str | None,
) -> bool | None:
    expected = normalize_country_code(expected_country)
    actual_codes = {
        code
        for code in (
            normalize_country_code(nationality),
            normalize_country_code(issuing_country),
        )
        if code
    }
    if not expected or not actual_codes:
        return None
    return expected in actual_codes


def normalize_country_code(value: Any) -> str:
    return str(value or "").strip().upper()
