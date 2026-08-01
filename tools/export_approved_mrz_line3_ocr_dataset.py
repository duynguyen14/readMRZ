from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generate_mrz_yolo_dataset import env_bool, env_int, env_value, read_env_file  # noqa: E402
from mrz_reader.db import connect  # noqa: E402
from mrz_reader.mrz import normalize_mrz_text  # noqa: E402


def resolve_base_dir(env: dict[str, str], key: str, default_path: Path) -> Path:
    return Path(env_value(env, key, str(default_path))).expanduser().resolve()


def row_to_dict(cursor: Any, row: Any) -> dict[str, Any]:
    columns = [column[0] for column in cursor.description]
    return dict(zip(columns, row))


def fetch_approved_rows(cursor: Any, *, limit: int, split: str) -> list[dict[str, Any]]:
    top_clause = f"TOP ({int(limit)})" if limit > 0 else ""
    split_filter = "AND ISNULL(split, '') = ?" if split else ""
    params: list[Any] = []
    if split:
        params.append(split)
    cursor.execute(
        f"""
        SELECT {top_clause}
            id,
            label_item_id,
            source_key,
            split,
            line_index,
            line_image_file_name,
            line_image_width,
            line_image_height,
            final_text,
            normalized_text,
            ocr_text,
            ocr_score,
            mrz_likeness,
            reviewed_at
        FROM dbo.readmrz_ocr_line_items3
        WHERE review_status = 'approved'
          AND ISNULL(LTRIM(RTRIM(COALESCE(final_text, normalized_text, ocr_text, ''))), '') <> ''
          AND ISNULL(LTRIM(RTRIM(line_image_file_name)), '') <> ''
          {split_filter}
        ORDER BY label_item_id ASC, line_index ASC, id ASC
        """,
        *params,
    )
    return [row_to_dict(cursor, row) for row in cursor.fetchall()]


def resolve_source_image(path_value: Any, image_base_dir: Path) -> Path:
    path = Path(str(path_value or ""))
    if path.is_absolute():
        return path
    return (image_base_dir / path).resolve()


def safe_stem(row: dict[str, Any]) -> str:
    label_item_id = int(row.get("label_item_id") or 0)
    line_index = int(row.get("line_index") or 0)
    row_id = int(row.get("id") or 0)
    return f"line3_{label_item_id:08d}_line{line_index}_{row_id:08d}"


def copy_with_unique_name(source_path: Path, target_dir: Path, stem: str) -> Path:
    extension = source_path.suffix.lower() or ".jpg"
    candidate = target_dir / f"{stem}{extension}"
    counter = 1
    while candidate.exists():
        candidate = target_dir / f"{stem}_{counter}{extension}"
        counter += 1
    shutil.copy2(source_path, candidate)
    return candidate


def write_text_label(path: Path, text: str) -> None:
    path.write_text(text + "\n", encoding="utf-8", newline="\n")


def export_rows(
    rows: list[dict[str, Any]],
    *,
    image_base_dir: Path,
    output_dir: Path,
    preserve_split: bool,
    dry_run: bool,
) -> dict[str, Any]:
    images_root = output_dir / "images"
    labels_root = output_dir / "labels"
    exported_jsonl_path = output_dir / "labels.jsonl"
    exported_csv_path = output_dir / "labels.csv"
    missing: list[dict[str, Any]] = []
    exported: list[dict[str, Any]] = []

    if not dry_run:
        images_root.mkdir(parents=True, exist_ok=True)
        labels_root.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        source_image = resolve_source_image(row.get("line_image_file_name"), image_base_dir)
        text = normalize_mrz_text(str(row.get("final_text") or row.get("normalized_text") or row.get("ocr_text") or ""))
        if not source_image.exists():
            missing.append(
                {
                    "id": int(row.get("id") or 0),
                    "label_item_id": int(row.get("label_item_id") or 0),
                    "line_index": int(row.get("line_index") or 0),
                    "source_image": str(source_image),
                }
            )
            continue
        if not text:
            continue

        split = str(row.get("split") or "train") if preserve_split else ""
        image_dir = images_root / split if split else images_root
        label_dir = labels_root / split if split else labels_root
        stem = safe_stem(row)

        if dry_run:
            image_rel = str((Path("images") / split / f"{stem}{source_image.suffix.lower() or '.jpg'}") if split else (Path("images") / f"{stem}{source_image.suffix.lower() or '.jpg'}")).replace("\\", "/")
            label_rel = str((Path("labels") / split / f"{stem}.txt") if split else (Path("labels") / f"{stem}.txt")).replace("\\", "/")
        else:
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            target_image = copy_with_unique_name(source_image, image_dir, stem)
            target_label = label_dir / f"{target_image.stem}.txt"
            write_text_label(target_label, text)
            image_rel = target_image.relative_to(output_dir).as_posix()
            label_rel = target_label.relative_to(output_dir).as_posix()

        exported.append(
            {
                "id": int(row.get("id") or 0),
                "label_item_id": int(row.get("label_item_id") or 0),
                "source_key": str(row.get("source_key") or ""),
                "split": str(row.get("split") or ""),
                "line_index": int(row.get("line_index") or 0),
                "image": image_rel,
                "label": label_rel,
                "text": text,
                "ocr_score": float(row.get("ocr_score") or 0.0),
                "mrz_likeness": float(row.get("mrz_likeness") or 0.0),
            }
        )

    if not dry_run:
        with exported_jsonl_path.open("w", encoding="utf-8", newline="\n") as jsonl_file:
            for item in exported:
                jsonl_file.write(json.dumps(item, ensure_ascii=False) + "\n")

        with exported_csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "id",
                    "label_item_id",
                    "source_key",
                    "split",
                    "line_index",
                    "image",
                    "label",
                    "text",
                    "ocr_score",
                    "mrz_likeness",
                ],
            )
            writer.writeheader()
            writer.writerows(exported)

        (output_dir / "charset.txt").write_text("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<\n", encoding="utf-8", newline="\n")
        if missing:
            (output_dir / "missing_images.json").write_text(
                json.dumps(missing, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    return {
        "exported": len(exported),
        "missing_images": len(missing),
        "output_dir": str(output_dir),
        "images_dir": str(images_root),
        "labels_dir": str(labels_root),
        "labels_jsonl": str(exported_jsonl_path),
        "labels_csv": str(exported_csv_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export approved readmrz_ocr_line_items3 rows into a final OCR training dataset.")
    parser.add_argument("--env", default=str(PROJECT_ROOT / ".env"), help="Path to .env config file.")
    parser.add_argument("--output-dir", default="", help="Output final dataset dir. Default READMRZ_EVISA_LINE3_FINAL_DATASET_DIR.")
    parser.add_argument("--limit", type=int, default=0, help="Max approved line rows to export. Use 0 for all.")
    parser.add_argument("--split", choices=["", "train", "val"], default="", help="Optional source split filter.")
    parser.add_argument("--flat", action="store_true", help="Do not preserve train/val subfolders.")
    parser.add_argument("--dry-run", action="store_true", help="Count and validate without copying files.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    env_path = Path(args.env)
    if not env_path.is_absolute():
        cwd_env_path = Path.cwd() / env_path
        project_env_path = PROJECT_ROOT / env_path
        env_path = cwd_env_path if cwd_env_path.exists() else project_env_path

    env = read_env_file(env_path)
    line3_dataset_dir = resolve_base_dir(env, "READMRZ_EVISA_LINE3_DATASET_DIR", PROJECT_ROOT / "generated_datasets" / "mrz_ocr_lines3")
    image_base_dir = resolve_base_dir(env, "READMRZ_EVISA_LINE3_IMAGE_BASE_DIR", line3_dataset_dir / "images")
    output_dir = Path(
        args.output_dir
        or env_value(env, "READMRZ_EVISA_LINE3_FINAL_DATASET_DIR", str(PROJECT_ROOT / "generated_datasets" / "mrz_ocr_lines3_final"))
    ).expanduser().resolve()
    preserve_split = not args.flat and env_bool(env, "READMRZ_EVISA_LINE3_FINAL_PRESERVE_SPLIT", True)
    batch_size = max(1, env_int(env, "READMRZ_EVISA_LINE3_FINAL_EXPORT_BATCH_SIZE", 500))

    print(f"image_base_dir: {image_base_dir}")
    print(f"output_dir: {output_dir}")
    print(f"limit: {args.limit}")
    print(f"split: {args.split or 'all'}")
    print(f"preserve_split: {preserve_split}")
    print(f"dry_run: {args.dry_run}")

    with connect() as connection:
        cursor = connection.cursor()
        rows = fetch_approved_rows(cursor, limit=args.limit, split=args.split)

    print(f"approved rows found: {len(rows)}")
    summary = export_rows(
        rows,
        image_base_dir=image_base_dir,
        output_dir=output_dir,
        preserve_split=preserve_split,
        dry_run=args.dry_run,
    )
    summary["batch_size"] = batch_size
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
