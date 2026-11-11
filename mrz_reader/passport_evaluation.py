from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
from pathlib import Path
import re
from typing import Iterable

from .passport_classification import (
    VALIDITY_EXPIRED,
    VALIDITY_SIX_MONTHS_OR_MORE,
    VALIDITY_UNDER_SIX_MONTHS,
)


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

EXPECTED_VALIDITY_BY_FOLDER = {
    "hochieuhethan": VALIDITY_EXPIRED,
    "conhanduoi6thang": VALIDITY_UNDER_SIX_MONTHS,
    "conhantren6thang": VALIDITY_SIX_MONTHS_OR_MORE,
}


@dataclass(frozen=True)
class EvaluationCandidate:
    sequence_no: int
    image_path: Path
    relative_image_path: str
    relative_path_hash: str
    country_code: str
    source_group: str
    expected_validity_status: str
    file_size_bytes: int


@dataclass(frozen=True)
class DiscoveryResult:
    candidates: list[EvaluationCandidate]
    passport_directories: int
    ignored_passport_directories: tuple[str, ...]


def discover_passport_images(
    source_root: Path,
    *,
    country_codes: Iterable[str] | None = None,
    limit: int = 0,
) -> DiscoveryResult:
    root = source_root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Source root does not exist or is not a directory: {root}")

    country_filter = {
        str(value).strip().upper()
        for value in (country_codes or [])
        if str(value).strip()
    }
    discovered: list[tuple[Path, str, str, str]] = []
    ignored_directories: list[str] = []
    passport_directory_count = 0

    passport_directories = sorted(
        (path for path in root.rglob("*") if path.is_dir() and path.name.lower() == "passport"),
        key=lambda path: str(path).lower(),
    )
    for passport_directory in passport_directories:
        passport_directory_count += 1
        if len(passport_directory.parents) < 2:
            ignored_directories.append(str(passport_directory))
            continue
        source_group = passport_directory.parent.name.strip().lower()
        expected_status = EXPECTED_VALIDITY_BY_FOLDER.get(source_group)
        country_code = passport_directory.parent.parent.name.strip().upper()
        if expected_status is None or not country_code:
            ignored_directories.append(str(passport_directory.relative_to(root)))
            continue
        if country_filter and country_code not in country_filter:
            continue

        for image_path in sorted(passport_directory.rglob("*"), key=lambda path: str(path).lower()):
            if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            discovered.append((image_path.resolve(), country_code, source_group, expected_status))

    if limit > 0:
        discovered = discovered[:limit]

    candidates: list[EvaluationCandidate] = []
    for sequence_no, (image_path, country_code, source_group, expected_status) in enumerate(
        discovered,
        start=1,
    ):
        relative_path = image_path.relative_to(root).as_posix()
        candidates.append(
            EvaluationCandidate(
                sequence_no=sequence_no,
                image_path=image_path,
                relative_image_path=relative_path,
                relative_path_hash=hashlib.sha256(relative_path.lower().encode("utf-8")).hexdigest(),
                country_code=country_code,
                source_group=source_group,
                expected_validity_status=expected_status,
                file_size_bytes=image_path.stat().st_size,
            )
        )

    return DiscoveryResult(
        candidates=candidates,
        passport_directories=passport_directory_count,
        ignored_passport_directories=tuple(ignored_directories),
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def infer_export_reference_date(source_root: Path) -> date | None:
    pattern = re.compile(r"^quick_(\d{8})(?:_\d{6})?$", flags=re.IGNORECASE)
    for part in reversed(source_root.expanduser().resolve().parts):
        match = pattern.match(part)
        if match is None:
            continue
        try:
            return date.fromisoformat(
                f"{match.group(1)[0:4]}-{match.group(1)[4:6]}-{match.group(1)[6:8]}"
            )
        except ValueError:
            return None
    return None
