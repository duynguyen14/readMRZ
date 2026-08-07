from __future__ import annotations

import argparse
import base64
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import pyodbc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mrz_reader.db import connect  # noqa: E402
from mrz_reader.env_config import env_value, read_env_file  # noqa: E402
from tools.generate_evisa_mrz_ocr_line3_dataset import (  # noqa: E402
    connect_source,
    load_csharp_config,
    source_root_dir,
)
from tools.generate_image_type_dataset import (  # noqa: E402
    ensure_schema,
    env_float,
    env_int,
    export_csvs,
    relative_path,
    remove_tree_or_rename,
    resolve_source_path,
    sha256_file,
    split_for_id,
    upsert_item,
)


PDF_CLASSES: dict[str, dict[str, Any]] = {
    "EVISA_RESULT": {
        "source_table": "TransactionEVisa",
        "source_field": "FileEVisa",
        "label": "EVISA_RESULT",
        "label_id": 2,
        "output_folder": "evisa_result",
        "id_prefix": "TransactionEVisa",
    },
    "VOA_RESULT": {
        "source_table": "Dispatchs",
        "source_field": "FilePath",
        "label": "VOA_RESULT",
        "label_id": 3,
        "output_folder": "voa_result",
        "id_prefix": "Dispatchs",
    },
}


def row_to_dict(cursor: Any, row: Any) -> dict[str, Any]:
    columns = [column[0] for column in cursor.description]
    return dict(zip(columns, row))


def order_clause_for(order_by: str) -> str:
    if order_by == "random":
        return "NEWID()"
    if order_by == "oldest":
        return "CreatedDate ASC, Id ASC"
    return "CreatedDate DESC, Id DESC"


def fetch_pdf_records(
    cursor: pyodbc.Cursor,
    *,
    class_key: str,
    limit: int,
    order_by: str,
) -> list[dict[str, Any]]:
    config = PDF_CLASSES[class_key]
    top_clause = f"TOP ({int(limit)})" if limit > 0 else ""
    order_clause = order_clause_for(order_by)

    if class_key == "EVISA_RESULT":
        cursor.execute(
            f"""
            SELECT {top_clause}
                Id,
                GUID,
                FileEVisa
            FROM [db_dichvu_visa].[dbo].[TransactionEVisa]
            WHERE ISNULL(LTRIM(RTRIM(FileEVisa)), '') <> ''
              AND LOWER(FileEVisa) LIKE '%.pdf%'
            ORDER BY {order_clause}
            """
        )
    elif class_key == "VOA_RESULT":
        cursor.execute(
            f"""
            SELECT {top_clause}
                Id,
                CAST(NULL AS uniqueidentifier) AS GUID,
                FilePath
            FROM [db_dichvu_visa].[dbo].[Dispatchs]
            WHERE ISNULL(LTRIM(RTRIM(FilePath)), '') <> ''
              AND LOWER(FilePath) LIKE '%.pdf%'
            ORDER BY {order_clause}
            """
        )
    else:
        raise ValueError(f"Unsupported class: {class_key}")

    return [row_to_dict(cursor, row) for row in cursor.fetchall()]


def decode_base64_pdf(text: str) -> bytes | None:
    value = text.strip()
    if "," in value and value.lower().startswith("data:"):
        value = value.split(",", 1)[1]
    if len(value) < 500:
        return None
    try:
        data = base64.b64decode(value, validate=True)
    except Exception:
        return None
    if not data.startswith(b"%PDF"):
        return None
    return data


def read_pdf_bytes(value: Any, source_root: Path) -> tuple[bytes, str]:
    if isinstance(value, (bytes, bytearray)):
        data = bytes(value)
        if not data.startswith(b"%PDF"):
            raise ValueError("binary source is not a PDF")
        return data, "binary"

    text = str(value or "").strip()
    if not text:
        raise ValueError("source PDF value is empty")

    if text.lower().startswith("data:"):
        decoded = decode_base64_pdf(text)
        if decoded is not None:
            return decoded, "base64"

    try:
        path = resolve_source_path(text, source_root)
    except OSError as exc:
        decoded = decode_base64_pdf(text)
        if decoded is not None:
            return decoded, "base64"
        raise ValueError(f"Cannot resolve source PDF path: {exc}") from exc

    if path.exists():
        return path.read_bytes(), str(path)

    decoded = decode_base64_pdf(text)
    if decoded is not None:
        return decoded, "base64"

    raise FileNotFoundError(f"Cannot find source PDF: {path}")


def render_pdf_first_page(pdf_bytes: bytes, *, zoom: float) -> np.ndarray:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("Missing PyMuPDF. Install it with: python -m pip install pymupdf") from exc

    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        if document.page_count < 1:
            raise ValueError("PDF has no pages")
        page = document.load_page(0)
        matrix = fitz.Matrix(max(0.5, zoom), max(0.5, zoom))
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height,
            pixmap.width,
            pixmap.n,
        )
        if pixmap.n == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def process_pdf_item(
    cursor: pyodbc.Cursor,
    *,
    record: dict[str, Any],
    class_key: str,
    source_root: Path,
    dataset_dir: Path,
    image_base_dir: Path,
    split: str,
    jpeg_quality: int,
    zoom: float,
) -> dict[str, Any]:
    config = PDF_CLASSES[class_key]
    source_id = int(record["Id"])
    source_field = str(config["source_field"])
    source_value = record.get(source_field)
    label = str(config["label"])
    label_id = int(config["label_id"])
    output_path = (
        image_base_dir
        / str(config["output_folder"])
        / f"{config['id_prefix']}_{source_id}_{config['output_folder']}.jpg"
    )
    relative_image_path = relative_path(output_path, dataset_dir)

    try:
        pdf_bytes, source_description = read_pdf_bytes(source_value, source_root)
        image = render_pdf_first_page(pdf_bytes, zoom=zoom)
        if image.size == 0:
            raise ValueError("rendered PDF image is empty")

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
            source_table=str(config["source_table"]),
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
            source_table=str(config["source_table"]),
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


def parse_classes(value: str) -> list[str]:
    classes = [part.strip().upper() for part in value.split(",") if part.strip()]
    if not classes:
        raise ValueError("At least one class is required")
    unknown = [class_key for class_key in classes if class_key not in PDF_CLASSES]
    if unknown:
        raise ValueError(f"Unknown classes: {', '.join(unknown)}")
    return classes


def clear_pdf_outputs(
    cursor: pyodbc.Cursor,
    *,
    class_keys: list[str],
    image_base_dir: Path,
) -> None:
    source_fields = [str(PDF_CLASSES[class_key]["source_field"]) for class_key in class_keys]
    placeholders = ",".join("?" for _ in source_fields)
    cursor.execute(
        f"DELETE FROM dbo.readmrz_image_type_dataset_items WHERE SourceField IN ({placeholders})",
        *source_fields,
    )
    for class_key in class_keys:
        remove_tree_or_rename(image_base_dir / str(PDF_CLASSES[class_key]["output_folder"]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Append EVisa/VOA PDF result classes to the image type classifier dataset."
    )
    parser.add_argument("--env", default=str(PROJECT_ROOT / ".env"), help="Path to .env config file.")
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=2000,
        help="Max valid PDF rows to fetch for each class. Use 0 for all.",
    )
    parser.add_argument("--order-by", choices=["oldest", "newest", "random"], default="random")
    parser.add_argument(
        "--classes",
        default="EVISA_RESULT,VOA_RESULT",
        help="Comma-separated classes: EVISA_RESULT,VOA_RESULT.",
    )
    parser.add_argument("--skip-schema", action="store_true", help="Do not run SQL schema creation/migration.")
    parser.add_argument(
        "--clear-pdf-classes",
        action="store_true",
        help="Delete only PDF-class rows/folders before generating them again.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality for rendered PDF pages.")
    parser.add_argument("--zoom", type=float, default=None, help="PyMuPDF render zoom. Default from env or 2.0.")
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
    zoom = args.zoom if args.zoom is not None else env_float(env, "READMRZ_IMAGE_TYPE_PDF_RENDER_ZOOM", 2.0)
    class_keys = parse_classes(args.classes)

    if not args.skip_schema:
        ensure_schema()

    print(f"source_root: {source_root}")
    print(f"dataset_dir: {dataset_dir}")
    print(f"image_base_dir: {image_base_dir}")
    print(f"classes: {', '.join(class_keys)}")
    print(f"limit_per_class: {args.limit_per_class}")
    print(f"order_by: {args.order_by}")
    print(f"val_ratio: {val_ratio}")
    print(f"test_ratio: {test_ratio}")
    print(f"seed: {seed}")
    print(f"pdf_render_zoom: {zoom}")

    totals: dict[str, int] = {
        "fetched": 0,
        "copied": 0,
        "errors": 0,
        "EVISA_RESULT": 0,
        "VOA_RESULT": 0,
    }

    with connect_source(env, csharp_config) as source_connection, connect() as destination_connection:
        destination_cursor = destination_connection.cursor()
        if args.clear_pdf_classes:
            clear_pdf_outputs(destination_cursor, class_keys=class_keys, image_base_dir=image_base_dir)
            destination_connection.commit()
            print("Cleared PDF class rows/folders only")

        source_cursor = source_connection.cursor()
        for class_key in class_keys:
            config = PDF_CLASSES[class_key]
            records = fetch_pdf_records(
                source_cursor,
                class_key=class_key,
                limit=args.limit_per_class,
                order_by=args.order_by,
            )
            totals["fetched"] += len(records)
            print(f"Fetched {len(records)} {config['source_table']}.{config['source_field']} rows")

            class_seed = seed + int(config["label_id"]) * 1000
            for index, record in enumerate(records, start=1):
                split = split_for_id(
                    int(record["Id"]),
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                    seed=class_seed,
                )
                result = process_pdf_item(
                    destination_cursor,
                    record=record,
                    class_key=class_key,
                    source_root=source_root,
                    dataset_dir=dataset_dir,
                    image_base_dir=image_base_dir,
                    split=split,
                    jpeg_quality=args.jpeg_quality,
                    zoom=zoom,
                )
                if result["status"] == "copied":
                    totals["copied"] += 1
                    totals[str(result["label"])] += 1
                else:
                    totals["errors"] += 1

                if index % batch_size == 0:
                    destination_connection.commit()
                    print(
                        f"[{class_key} {index}/{len(records)}] "
                        f"committed copied={totals['copied']} errors={totals['errors']}"
                    )

            destination_connection.commit()

        csv_counts = export_csvs(destination_connection, dataset_dir)

    print("Done")
    print({**totals, **csv_counts})
    print(f"CSV: {dataset_dir / 'labels.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
