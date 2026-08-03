from __future__ import annotations

from datetime import date, datetime
from typing import Any

from .db import connect


ALLOWED_FILTERS = {"all", "match", "mismatch", "error", "fallback"}


def row_to_dict(cursor: Any, row: Any) -> dict[str, Any]:
    columns = [column[0] for column in cursor.description]
    return dict(zip(columns, row))


def scalar_int(cursor: Any) -> int:
    row = cursor.fetchone()
    return int(row[0] or 0) if row else 0


def json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    return value


def build_where_clause(filter_name: str) -> str:
    if filter_name == "match":
        return "WHERE IsFullMatch = 1"
    if filter_name == "mismatch":
        return "WHERE IsFullMatch = 0 AND ErrorMessage IS NULL"
    if filter_name == "error":
        return "WHERE ErrorMessage IS NOT NULL"
    if filter_name == "fallback":
        return "WHERE UsedFallback = 1"
    return ""


def review_stats(cursor: Any) -> dict[str, int]:
    cursor.execute("SELECT COUNT(*) FROM dbo.readmrz_pipeline_test_items")
    total = scalar_int(cursor)
    cursor.execute("SELECT COUNT(*) FROM dbo.readmrz_pipeline_test_items WHERE IsFullMatch = 1")
    matched = scalar_int(cursor)
    cursor.execute(
        "SELECT COUNT(*) FROM dbo.readmrz_pipeline_test_items WHERE IsFullMatch = 0 AND ErrorMessage IS NULL"
    )
    mismatched = scalar_int(cursor)
    cursor.execute("SELECT COUNT(*) FROM dbo.readmrz_pipeline_test_items WHERE ErrorMessage IS NOT NULL")
    errors = scalar_int(cursor)
    cursor.execute("SELECT COUNT(*) FROM dbo.readmrz_pipeline_test_items WHERE UsedFallback = 1")
    fallback = scalar_int(cursor)
    return {
        "total": total,
        "matched": matched,
        "mismatched": mismatched,
        "errors": errors,
        "fallback": fallback,
    }


def get_pipeline_test_review_items(
    *,
    filter_name: str = "all",
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    filter_name = filter_name if filter_name in ALLOWED_FILTERS else "all"
    limit = min(200, max(1, int(limit or 50)))
    offset = max(0, int(offset or 0))
    where_clause = build_where_clause(filter_name)

    with connect() as connection:
        cursor = connection.cursor()
        stats = review_stats(cursor)
        cursor.execute(
            f"""
            SELECT COUNT(*)
            FROM dbo.readmrz_pipeline_test_items
            {where_clause}
            """
        )
        total_filtered = scalar_int(cursor)
        cursor.execute(
            f"""
            SELECT *
            FROM dbo.readmrz_pipeline_test_items
            {where_clause}
            ORDER BY Id DESC
            OFFSET ? ROWS FETCH NEXT ? ROWS ONLY
            """,
            offset,
            limit,
        )
        rows = [
            {key: json_value(value) for key, value in row_to_dict(cursor, row).items()}
            for row in cursor.fetchall()
        ]

    return {
        "status": "ok",
        "filter": filter_name,
        "items": rows,
        "stats": stats,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "total": total_filtered,
        },
    }
