from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mrz_reader.db import execute_sql_file


def main() -> int:
    sql_path = PROJECT_ROOT / "sql" / "create_mrz_ocr_line2_tables.sql"
    execute_sql_file(sql_path)
    print(f"Created MRZ OCR line2 tables using: {sql_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
