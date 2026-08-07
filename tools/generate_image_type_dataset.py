from __future__ import annotations

import argparse
import base64
import csv
import hashlib
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any

import cv2
import numpy as np
import pyodbc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mrz_reader.db import connect, execute_sql_file  # noqa: E402
from mrz_reader.env_config import env_value, read_env_file  # noqa: E402
from tools.generate_evisa_mrz_ocr_line3_dataset import (  # noqa: E402
    connect_source,
    load_csharp_config,
    source_root_dir,
)


LABELS = {
    "FullPassportImage": ("passport", 0),
    "FaceImage": ("face", 1),
}


def ensure_schema() -> None:
    execute_sql_file(PROJECT_ROOT / "sql" / "create_image_type_dataset_tables.sql")


def row_to_dict(cursor: Any, row: Any) -> dict[str, Any]:
    columns = [column[0] for column in cursor.description]
    return dict(zip(columns, row))


def fetch_records(
    cursor: pyodbc.Cursor,
    *,
    limit: int,
    order_by: str,
    require_both: bool,
) -> list[dict[str, Any]]:
    top_clause = f"TOP ({int(limit)})" if limit > 0 else ""
    if order_by == "random":
        order_clause = "NEWID()"
    elif order_by == "newest":
        order_clause = "CreatedDate DESC, Id DESC"
    else:
        order_clause = "CreatedDate ASC, Id ASC"
    condition = "AND" if require_both else "OR"

    cursor.execute(
        f"""
        SELECT {top_clause}
            Id,
            GUID,
            FullPassportImage,
            FaceImage
        FROM [db_dichvu_visa].[dbo].[TransactionEVisa]
        WHERE ISNULL(LTRIM(RTRIM(FullPassportImage)), '') <> ''
          {condition} ISNULL(LTRIM(RTRIM(FaceImage)), '') <> ''
        ORDER BY {order_clause}
        """
    )
    return [row_to_dict(cursor, row) for row in cursor.fetchall()]


def resolve_source_path(value: Any, root_dir: Path) -> Path:
    raw_value = str(value or "").strip()
    if not raw_value:
        raise ValueError("source image value is empty")
    normalized = raw_value.replace("/", "\\").lstrip("~")
    path = Path(normalized)
    if path.is_absolute():
        return path.resolve()
    return (root_dir / normalized.lstrip("\\")).resolve()


def maybe_decode_base64_image(value: Any) -> np.ndarray | None:
    if isinstance(value, (bytes, bytearray)):
        data = bytes(value)
    else:
        text = str(value or "").strip()
        if "," in text and text.lower().startswith("data:"):
            text = text.split(",", 1)[1]
        if len(text) < 300:
            return None
        try:
            data = base64.b64decode(text, validate=True)
        except Exception:
            return None

    array = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def read_source_image(value: Any, root_dir: Path) -> tuple[np.ndarray, str]:
    decoded = maybe_decode_base64_image(value)
    if decoded is not None:
        return decoded, "base64"

    path = resolve_source_path(value, root_dir)
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read source image: {path}")
    return image, str(path)


def relative_path(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_for_id(source_id: int, *, val_ratio: float, test_ratio: float, seed: int) -> str:
    rng = random.Random(f"{seed}:{source_id}")
    value = rng.random()
    if test_ratio > 0 and value < test_ratio:
        return "test"
    if val_ratio > 0 and value < test_ratio + val_ratio:
        return "val"
    return "train"


def upsert_item(
    cursor: pyodbc.Cursor,
    *,
    record: dict[str, Any],
    source_field: str,
    source_table: str = "TransactionEVisa",
    source_value: Any,
    label: str,
    label_id: int,
    relative_image_path: str,
    split: str,
    status: str,
    image_width: int | None,
    image_height: int | None,
    file_size: int | None,
    sha256: str | None,
    error_message: str | None,
) -> None:
    cursor.execute(
        """
        UPDATE dbo.readmrz_image_type_dataset_items
        SET
            SourceTable = ?,
            TransactionGuid = ?,
            SourceImageValue = ?,
            Label = ?,
            LabelId = ?,
            RelativeImagePath = ?,
            ImageWidth = ?,
            ImageHeight = ?,
            FileSizeBytes = ?,
            Sha256 = ?,
            Split = ?,
            Status = ?,
            ErrorMessage = ?,
            UpdatedDate = SYSDATETIME()
        WHERE TransactionEVisaId = ?
          AND SourceField = ?
        """,
        source_table,
        str(record.get("GUID")) if record.get("GUID") else None,
        str(source_value or "")[:1000] or None,
        label,
        label_id,
        relative_image_path,
        image_width,
        image_height,
        file_size,
        sha256,
        split,
        status,
        error_message,
        int(record["Id"]),
        source_field,
    )
    if cursor.rowcount:
        return

    cursor.execute(
        """
        INSERT INTO dbo.readmrz_image_type_dataset_items (
            SourceTable,
            TransactionEVisaId,
            TransactionGuid,
            SourceField,
            SourceImageValue,
            Label,
            LabelId,
            RelativeImagePath,
            ImageWidth,
            ImageHeight,
            FileSizeBytes,
            Sha256,
            Split,
            Status,
            ErrorMessage
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        source_table,
        int(record["Id"]),
        str(record.get("GUID")) if record.get("GUID") else None,
        source_field,
        str(source_value or "")[:1000] or None,
        label,
        label_id,
        relative_image_path,
        image_width,
        image_height,
        file_size,
        sha256,
        split,
        status,
        error_message,
    )


def process_image_item(
    cursor: pyodbc.Cursor,
    *,
    record: dict[str, Any],
    source_field: str,
    source_root: Path,
    dataset_dir: Path,
    image_base_dir: Path,
    split: str,
    jpeg_quality: int,
) -> dict[str, Any]:
    label, label_id = LABELS[source_field]
    source_id = int(record["Id"])
    output_path = image_base_dir / label / f"TransactionEVisa_{source_id}_{label}.jpg"
    relative_image_path = relative_path(output_path, dataset_dir)
    source_value = record.get(source_field)

    try:
        image, source_description = read_source_image(source_value, source_root)
        if image.size == 0:
            raise ValueError("decoded source image is empty")
        height, width = image.shape[:2]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(
            str(output_path),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, jpeg_quality))],
        )
        if not ok:
            raise ValueError(f"Cannot write output image: {output_path}")
        file_size = output_path.stat().st_size
        sha256 = sha256_file(output_path)
        upsert_item(
            cursor,
            record=record,
            source_field=source_field,
            source_value=source_value,
            label=label,
            label_id=label_id,
            relative_image_path=relative_image_path,
            split=split,
            status="copied",
            image_width=width,
            image_height=height,
            file_size=file_size,
            sha256=sha256,
            error_message=None,
        )
        return {
            "status": "copied",
            "label": label,
            "path": relative_image_path,
            "source": source_description,
        }
    except Exception as exc:
        upsert_item(
            cursor,
            record=record,
            source_field=source_field,
            source_value=source_value,
            label=label,
            label_id=label_id,
            relative_image_path=relative_image_path,
            split=split,
            status="error",
            image_width=None,
            image_height=None,
            file_size=None,
            sha256=None,
            error_message=str(exc)[:1900],
        )
        return {
            "status": "error",
            "label": label,
            "path": relative_image_path,
            "error": str(exc),
        }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "relative_path",
                "label",
                "label_id",
                "split",
                "source_table",
                "source_id",
                "source_field",
                "sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def remove_tree_or_rename(path: Path) -> None:
    if not path.exists():
        return

    last_error: Exception | None = None
    for _ in range(3):
        try:
            shutil.rmtree(path)
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.35)

    suffix = time.strftime("%Y%m%d_%H%M%S")
    renamed = path.with_name(f"{path.name}_old_{suffix}")
    try:
        path.rename(renamed)
        print(
            f"Warning: cannot delete {path}; renamed old folder to {renamed}. "
            "Close Explorer/viewers and delete it later."
        )
        return
    except OSError:
        if last_error is not None:
            raise last_error
        raise


def clear_managed_outputs(dataset_dir: Path, image_base_dir: Path) -> None:
    # On Windows, removing the dataset root can fail if Explorer or antivirus keeps
    # a transient handle. Only clean outputs owned by this script.
    remove_tree_or_rename(image_base_dir)
    for file_name in ("labels.csv", "train.csv", "val.csv", "test.csv"):
        path = dataset_dir / file_name
        if not path.exists():
            continue
        for _ in range(3):
            try:
                path.unlink()
                break
            except OSError:
                time.sleep(0.2)


def export_csvs(connection: pyodbc.Connection, dataset_dir: Path) -> dict[str, int]:
    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT
            RelativeImagePath,
            Label,
            LabelId,
            Split,
            SourceTable,
            TransactionEVisaId,
            SourceField,
            Sha256
        FROM dbo.readmrz_image_type_dataset_items
        WHERE Status = 'copied'
        ORDER BY Split, LabelId, TransactionEVisaId, SourceField
        """
    )
    rows = [
        {
            "relative_path": row.RelativeImagePath,
            "label": row.Label,
            "label_id": int(row.LabelId),
            "split": row.Split,
            "source_table": row.SourceTable,
            "source_id": int(row.TransactionEVisaId),
            "source_field": row.SourceField,
            "sha256": row.Sha256 or "",
        }
        for row in cursor.fetchall()
    ]

    write_csv(dataset_dir / "labels.csv", rows)
    counts = {"labels": len(rows), "train": 0, "val": 0, "test": 0}
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        if split_rows or split != "test":
            write_csv(dataset_dir / f"{split}.csv", split_rows)
        counts[split] = len(split_rows)
    return counts


def env_float(env: dict[str, str], key: str, default: float) -> float:
    try:
        return float(env_value(env, key, str(default)))
    except ValueError:
        return default


def env_int(env: dict[str, str], key: str, default: int) -> int:
    try:
        return int(env_value(env, key, str(default)))
    except ValueError:
        return default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate passport-vs-face image classification dataset from TransactionEVisa."
    )
    parser.add_argument("--env", default=str(PROJECT_ROOT / ".env"), help="Path to .env config file.")
    parser.add_argument("--limit", type=int, default=0, help="Max TransactionEVisa rows to fetch. Use 0 for all.")
    parser.add_argument("--order-by", choices=["oldest", "newest", "random"], default="newest")
    parser.add_argument("--allow-partial", action="store_true", help="Accept rows with only one of FullPassportImage/FaceImage.")
    parser.add_argument("--skip-schema", action="store_true", help="Do not run SQL schema creation.")
    parser.add_argument("--clear", action="store_true", help="Delete destination table rows and output dataset folder before running.")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality for normalized output images.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    env_path = Path(args.env)
    if not env_path.is_absolute():
        cwd_env_path = Path.cwd() / env_path
        project_env_path = PROJECT_ROOT / env_path
        env_path = cwd_env_path if cwd_env_path.exists() else project_env_path

    env = read_env_file(env_path)
    csharp_config = load_csharp_config(env)
    source_root = source_root_dir(env, csharp_config)
    dataset_dir = Path(
        env_value(
            env,
            "READMRZ_IMAGE_TYPE_DATASET_DIR",
            str(PROJECT_ROOT / "generated_datasets" / "image_type_classifier"),
        )
    ).expanduser().resolve()
    image_base_dir = Path(
        env_value(env, "READMRZ_IMAGE_TYPE_IMAGE_BASE_DIR", str(dataset_dir / "images"))
    ).expanduser().resolve()
    val_ratio = max(0.0, min(1.0, env_float(env, "READMRZ_IMAGE_TYPE_VAL_RATIO", 0.1)))
    test_ratio = max(0.0, min(1.0 - val_ratio, env_float(env, "READMRZ_IMAGE_TYPE_TEST_RATIO", 0.0)))
    seed = env_int(env, "READMRZ_IMAGE_TYPE_SPLIT_SEED", 20260805)
    batch_size = max(1, env_int(env, "READMRZ_IMAGE_TYPE_BATCH_SIZE", 200))

    if not args.skip_schema:
        ensure_schema()

    print(f"source_root: {source_root}")
    print(f"dataset_dir: {dataset_dir}")
    print(f"image_base_dir: {image_base_dir}")
    print(f"limit: {args.limit}")
    print(f"order_by: {args.order_by}")
    print(f"require_both: {not args.allow_partial}")
    print(f"val_ratio: {val_ratio}")
    print(f"test_ratio: {test_ratio}")
    print(f"seed: {seed}")

    totals = {
        "fetched": 0,
        "copied": 0,
        "errors": 0,
        "passport": 0,
        "face": 0,
    }

    with connect_source(env, csharp_config) as source_connection, connect() as destination_connection:
        destination_cursor = destination_connection.cursor()
        if args.clear:
            destination_cursor.execute("DELETE FROM dbo.readmrz_image_type_dataset_items")
            destination_connection.commit()
            clear_managed_outputs(dataset_dir, image_base_dir)
            print("Cleared table dbo.readmrz_image_type_dataset_items and managed dataset outputs")

        source_cursor = source_connection.cursor()
        records = fetch_records(
            source_cursor,
            limit=args.limit,
            order_by=args.order_by,
            require_both=not args.allow_partial,
        )
        totals["fetched"] = len(records)
        print(f"Fetched {len(records)} TransactionEVisa rows")

        for index, record in enumerate(records, start=1):
            split = split_for_id(
                int(record["Id"]),
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
            )
            for source_field in LABELS:
                if not str(record.get(source_field) or "").strip():
                    continue
                result = process_image_item(
                    destination_cursor,
                    record=record,
                    source_field=source_field,
                    source_root=source_root,
                    dataset_dir=dataset_dir,
                    image_base_dir=image_base_dir,
                    split=split,
                    jpeg_quality=args.jpeg_quality,
                )
                if result["status"] == "copied":
                    totals["copied"] += 1
                    totals[str(result["label"])] += 1
                else:
                    totals["errors"] += 1

            if index % batch_size == 0:
                destination_connection.commit()
                print(f"[{index}/{len(records)}] committed copied={totals['copied']} errors={totals['errors']}")

        destination_connection.commit()
        csv_counts = export_csvs(destination_connection, dataset_dir)

    print("Done")
    print({**totals, **csv_counts})
    print(f"CSV: {dataset_dir / 'labels.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
