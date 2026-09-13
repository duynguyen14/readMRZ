from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import random
import re
import shutil
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mrz_reader.db import connect
from mrz_reader.env_config import env_value, read_env_file


DEFAULT_FIELDS = [
    "visa_number",
    "visa_code",
    "valid_from",
    "valid_until",
    "number_of_entries",
    "passport_number",
    "mrz_zone",
    "issue_date",
    "date_of_birth",
    "full_name",
    "nationality",
    "sex",
    "issue_place",
    "issuing_authority",
    "remarks",
]

VALID_SPLITS = {"train", "val", "test"}


@dataclass(frozen=True)
class Config:
    crop_image_dir: Path
    output_dir: Path
    fields: list[str]
    review_statuses: list[str]
    val_ratio: float
    test_ratio: float
    limit: int
    force: bool
    dry_run: bool
    zip_output: bool


def env_int(env: dict[str, str], key: str, default: int) -> int:
    raw_value = env_value(env, key, "")
    return int(raw_value) if raw_value else default


def env_float(env: dict[str, str], key: str, default: float) -> float:
    raw_value = env_value(env, key, "")
    return float(raw_value) if raw_value else default


def env_bool(env: dict[str, str], key: str, default: bool) -> bool:
    raw_value = env_value(env, key, "")
    if not raw_value:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def csv_values(text: str) -> list[str]:
    values = [value.strip() for value in text.split(",")]
    return [value for value in values if value]


def load_config(env: dict[str, str], args: argparse.Namespace) -> Config:
    crop_dir_raw = (
        args.crop_image_dir
        or env_value(env, "READMRZ_VN_VISA_OCR_REC_CROP_IMAGE_DIR", "")
        or env_value(env, "READMRZ_VN_VISA_CROP_OCR_OUTPUT_DIR", "")
    )
    if not crop_dir_raw:
        raise ValueError(
            "Missing READMRZ_VN_VISA_OCR_REC_CROP_IMAGE_DIR or READMRZ_VN_VISA_CROP_OCR_OUTPUT_DIR"
        )

    output_dir_raw = (
        args.output_dir
        or env_value(env, "READMRZ_VN_VISA_OCR_REC_DATASET_DIR", "")
        or str(PROJECT_ROOT / "generated_datasets" / "vn_visa_paddle_ocr_rec")
    )

    fields_raw = (
        args.fields
        or env_value(env, "READMRZ_VN_VISA_OCR_REC_FIELDS", "")
        or env_value(env, "READMRZ_VN_VISA_CROP_OCR_FIELDS", "")
    )
    fields = csv_values(fields_raw) if fields_raw else list(DEFAULT_FIELDS)
    if not fields:
        raise ValueError("READMRZ_VN_VISA_OCR_REC_FIELDS is empty")

    review_statuses_raw = (
        args.review_statuses
        or env_value(env, "READMRZ_VN_VISA_OCR_REC_REVIEW_STATUSES", "")
        or "approved"
    )
    review_statuses = csv_values(review_statuses_raw)
    if not review_statuses:
        raise ValueError("READMRZ_VN_VISA_OCR_REC_REVIEW_STATUSES is empty")

    return Config(
        crop_image_dir=Path(crop_dir_raw).expanduser().resolve(),
        output_dir=Path(output_dir_raw).expanduser().resolve(),
        fields=fields,
        review_statuses=review_statuses,
        val_ratio=max(
            0.0,
            min(
                1.0,
                args.val_ratio
                if args.val_ratio is not None
                else env_float(env, "READMRZ_VN_VISA_OCR_REC_VAL_RATIO", 0.1),
            ),
        ),
        test_ratio=max(
            0.0,
            min(
                1.0,
                args.test_ratio
                if args.test_ratio is not None
                else env_float(env, "READMRZ_VN_VISA_OCR_REC_TEST_RATIO", 0.0),
            ),
        ),
        limit=max(
            0,
            args.limit
            if args.limit is not None
            else env_int(env, "READMRZ_VN_VISA_OCR_REC_LIMIT", 0),
        ),
        force=args.force or env_bool(env, "READMRZ_VN_VISA_OCR_REC_FORCE", False),
        dry_run=args.dry_run or env_bool(env, "READMRZ_VN_VISA_OCR_REC_DRY_RUN", False),
        zip_output=args.zip or env_bool(env, "READMRZ_VN_VISA_OCR_REC_ZIP", False),
    )


def sql_placeholders(values: list[str]) -> str:
    return ", ".join("?" for _ in values)


def fetch_rows(config: Config) -> list[dict[str, Any]]:
    query = f"""
        SELECT
            c.Id,
            c.VisaItemId,
            c.SourceKey,
            c.FieldName,
            c.CropRelativePath,
            c.CropWidth,
            c.CropHeight,
            c.OcrRawText,
            c.OcrScore,
            c.ReviewStatus,
            c.ReviewedText,
            vi.OriginalFileName AS VisaOriginalFileName,
            vi.Split AS VisaSplit
        FROM dbo.readmrz_vn_visa_ocr_crops c
        LEFT JOIN dbo.readmrz_vn_visa_items vi ON vi.Id = c.VisaItemId
        WHERE c.ReviewStatus IN ({sql_placeholders(config.review_statuses)})
          AND c.FieldName IN ({sql_placeholders(config.fields)})
          AND NULLIF(LTRIM(RTRIM(c.ReviewedText)), N'') IS NOT NULL
        ORDER BY c.Id ASC
    """
    if config.limit > 0:
        query = query.replace("SELECT", f"SELECT TOP {int(config.limit)}", 1)

    params: list[Any] = [*config.review_statuses, *config.fields]
    with connect() as connection:
        cursor = connection.cursor()
        cursor.execute(query, *params)
        columns = [column[0] for column in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]


def normalize_label(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[\t\r\n]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text


def resolve_crop_path(crop_image_dir: Path, relative_path: str) -> Path | None:
    raw_path = Path(str(relative_path or "").strip())
    if not raw_path:
        return None
    if raw_path.is_absolute() and raw_path.is_file():
        return raw_path
    direct = crop_image_dir / raw_path
    if direct.is_file():
        return direct
    file_name = raw_path.name
    if file_name:
        matches = list(crop_image_dir.rglob(file_name))
        if matches:
            return matches[0]
    return None


def choose_split(row: dict[str, Any], val_ratio: float, test_ratio: float) -> str:
    db_split = str(row.get("VisaSplit") or "").strip().lower()
    if db_split in VALID_SPLITS:
        return db_split

    # Split by visa id so crops from the same visa do not leak across train/val.
    key = str(row.get("VisaItemId") or row.get("SourceKey") or row.get("Id") or "")
    rng = random.Random(hashlib.sha1(key.encode("utf-8")).hexdigest())
    value = rng.random()
    if test_ratio > 0 and value < test_ratio:
        return "test"
    if value < test_ratio + val_ratio:
        return "val"
    return "train"


def safe_output_name(row: dict[str, Any], crop_path: Path) -> str:
    field_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row.get("FieldName") or "field")).strip("._")
    source_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", crop_path.stem).strip("._") or "crop"
    suffix = crop_path.suffix.lower() or ".jpg"
    return f"{field_name}/{int(row['Id']):08d}_{source_stem}{suffix}"


def clean_output_dir(output_dir: Path) -> None:
    for child_name in ("images",):
        child_path = output_dir / child_name
        if child_path.exists():
            shutil.rmtree(child_path)
    for file_name in (
        "train.txt",
        "val.txt",
        "test.txt",
        "rec_gt_train.txt",
        "rec_gt_val.txt",
        "rec_gt_test.txt",
        "charset.txt",
        "manifest.jsonl",
        "summary.json",
    ):
        file_path = output_dir / file_name
        if file_path.exists():
            file_path.unlink()


def export_dataset(config: Config) -> dict[str, Any]:
    if not config.crop_image_dir.exists():
        raise FileNotFoundError(f"Crop image dir does not exist: {config.crop_image_dir}")

    rows = fetch_rows(config)
    counts: dict[str, int] = {
        "db_rows": len(rows),
        "exported_crops": 0,
        "missing_crops": 0,
        "empty_labels": 0,
    }
    split_counts: dict[str, int] = {split: 0 for split in ("train", "val", "test")}
    field_counts: dict[str, int] = {field_name: 0 for field_name in config.fields}
    split_lines: dict[str, list[str]] = {split: [] for split in ("train", "val", "test")}
    manifest_items: list[dict[str, Any]] = []
    charset: set[str] = set()

    if not config.dry_run:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        if config.force:
            clean_output_dir(config.output_dir)
        for split in ("train", "val", "test"):
            (config.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)

    for row in rows:
        label = normalize_label(row.get("ReviewedText"))
        if not label:
            counts["empty_labels"] += 1
            continue

        crop_path = resolve_crop_path(config.crop_image_dir, str(row.get("CropRelativePath") or ""))
        if crop_path is None:
            counts["missing_crops"] += 1
            print(f"[missing-crop] id={row.get('Id')} path={row.get('CropRelativePath')}")
            continue

        split = choose_split(row, config.val_ratio, config.test_ratio)
        output_name = safe_output_name(row, crop_path)
        image_rel = f"images/{split}/{output_name}".replace("\\", "/")
        output_path = config.output_dir / image_rel

        if not config.dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(crop_path, output_path)

        line = f"{image_rel}\t{label}"
        split_lines[split].append(line)
        charset.update(label)
        field_name = str(row.get("FieldName") or "")
        field_counts[field_name] = field_counts.get(field_name, 0) + 1
        split_counts[split] += 1
        counts["exported_crops"] += 1

        manifest_items.append(
            {
                "id": int(row["Id"]),
                "visa_item_id": int(row["VisaItemId"]),
                "field_name": field_name,
                "source_key": row.get("SourceKey") or "",
                "visa_image_name": row.get("VisaOriginalFileName") or "",
                "crop_source": str(crop_path),
                "image": image_rel,
                "label": label,
                "ocr_raw_text": row.get("OcrRawText") or "",
                "ocr_score": float(row.get("OcrScore") or 0.0),
                "split": split,
                "crop_width": int(row.get("CropWidth") or 0),
                "crop_height": int(row.get("CropHeight") or 0),
            }
        )
        print(f"[exported] id={row.get('Id')} split={split} field={field_name} text={label!r}")

    summary = {
        **counts,
        "splits": split_counts,
        "fields": field_counts,
        "charset_size": len(charset),
        "crop_image_dir": str(config.crop_image_dir),
        "output_dir": str(config.output_dir),
        "review_statuses": config.review_statuses,
        "dry_run": config.dry_run,
    }

    if not config.dry_run:
        for split in ("train", "val", "test"):
            text = "\n".join(split_lines[split])
            if text:
                text += "\n"
            (config.output_dir / f"{split}.txt").write_text(text, encoding="utf-8")
            (config.output_dir / f"rec_gt_{split}.txt").write_text(text, encoding="utf-8")

        (config.output_dir / "charset.txt").write_text(
            "".join(sorted(charset)) + "\n",
            encoding="utf-8",
        )
        with (config.output_dir / "manifest.jsonl").open("w", encoding="utf-8") as file_obj:
            for item in manifest_items:
                file_obj.write(json.dumps(item, ensure_ascii=False) + "\n")
        (config.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if config.zip_output:
            zip_base = str(config.output_dir)
            shutil.make_archive(zip_base, "zip", config.output_dir)
            summary["zip_path"] = f"{zip_base}.zip"

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export reviewed VN visa OCR crop labels into a PaddleOCR recognition dataset."
    )
    parser.add_argument("--env", default=str(PROJECT_ROOT / ".env"), help="Path to .env config file.")
    parser.add_argument("--crop-image-dir", default="", help="Override crop image root directory.")
    parser.add_argument("--output-dir", default="", help="Override output dataset directory.")
    parser.add_argument("--fields", default="", help="Comma-separated field names to export.")
    parser.add_argument("--review-statuses", default="", help="Comma-separated review statuses. Defaults to approved.")
    parser.add_argument("--val-ratio", type=float, default=None, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=None, help="Test split ratio.")
    parser.add_argument("--limit", type=int, default=None, help="Limit exported DB rows.")
    parser.add_argument("--force", action="store_true", help="Clear existing output before writing.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize without writing files.")
    parser.add_argument("--zip", action="store_true", help="Also create a .zip beside the dataset folder.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    env_path = Path(args.env)
    if not env_path.is_absolute():
        env_path = (Path.cwd() / env_path).resolve()
        if not env_path.exists():
            env_path = PROJECT_ROOT / args.env

    env = read_env_file(env_path)
    config = load_config(env, args)
    print(f"Using env file: {env_path}")
    print(f"Crop images: {config.crop_image_dir}")
    print(f"Output dataset: {config.output_dir}")
    print(f"Fields: {', '.join(config.fields)}")
    print(f"Review statuses: {', '.join(config.review_statuses)}")
    print(f"Dry run: {'yes' if config.dry_run else 'no'}")

    summary = export_dataset(config)
    print("DONE")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
