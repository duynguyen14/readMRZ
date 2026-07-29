from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_env_file(path: Path | None = None) -> dict[str, str]:
    env_path = path or PROJECT_ROOT / ".env"
    values: dict[str, str] = {}
    if not env_path.exists():
        return values

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def env_value(env: dict[str, str], key: str, default: str = "") -> str:
    return os.environ.get(key) or env.get(key) or default


def yolo_dataset_dir(env: dict[str, str] | None = None) -> Path:
    loaded_env = env if env is not None else read_env_file()
    configured = env_value(
        loaded_env,
        "READMRZ_YOLO_DATASET_DIR",
        str(PROJECT_ROOT / "generated_datasets" / "mrz_yolo"),
    )
    return Path(configured).expanduser().resolve()
