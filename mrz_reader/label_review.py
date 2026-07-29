from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
import shutil
import time
from typing import Any

import cv2

from .env_config import read_env_file, yolo_dataset_dir


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def dataset_paths() -> dict[str, Path]:
    dataset_dir = yolo_dataset_dir(read_env_file())
    return {
        "dataset_dir": dataset_dir,
        "processed": dataset_dir / "processed.json",
        "review_state": dataset_dir / "review_state.json",
    }


def load_processed() -> dict[str, Any]:
    paths = dataset_paths()
    return load_json(paths["processed"], {"version": 1, "items": {}})


def save_processed(processed: dict[str, Any]) -> None:
    save_json(dataset_paths()["processed"], processed)


def load_state() -> dict[str, Any]:
    return load_json(dataset_paths()["review_state"], {"last_key": "", "history": []})


def save_state(state: dict[str, Any]) -> None:
    save_json(dataset_paths()["review_state"], state)


def item_is_pending(item: dict[str, Any]) -> bool:
    if item.get("status") != "labeled":
        return False
    if item.get("review_status") in {"approved", "rejected"}:
        return False
    image_path = Path(str(item.get("output_image", "")))
    label_path = Path(str(item.get("output_label", "")))
    return image_path.exists() and label_path.exists()


def sorted_items(processed: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    items = processed.get("items", {})
    if not isinstance(items, dict):
        return []
    return sorted(items.items(), key=lambda pair: pair[0])


def review_stats(processed: dict[str, Any]) -> dict[str, int]:
    total = 0
    pending = 0
    approved = 0
    rejected = 0
    no_mrz = 0
    for _, item in sorted_items(processed):
        status = item.get("status")
        if status == "no_mrz":
            no_mrz += 1
            continue
        if status != "labeled":
            continue
        total += 1
        review_status = item.get("review_status")
        if review_status == "approved":
            approved += 1
        elif review_status == "rejected":
            rejected += 1
        elif item_is_pending(item):
            pending += 1
    return {
        "total": total,
        "pending": pending,
        "approved": approved,
        "rejected": rejected,
        "no_mrz": no_mrz,
    }


def next_pending_key(after_key: str = "") -> str | None:
    processed = load_processed()
    items = sorted_items(processed)
    if not items:
        return None

    keys = [key for key, item in items if item_is_pending(item)]
    if not keys:
        return None
    if not after_key or after_key not in keys:
        return keys[0]

    next_index = keys.index(after_key) + 1
    if next_index < len(keys):
        return keys[next_index]
    return keys[0]


def image_to_base64(path: Path) -> tuple[str, str, int, int]:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read review image: {path}")
    height, width = image.shape[:2]
    content_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return encoded, content_type, width, height


def bbox_percent(bbox_xyxy: list[float], width: int, height: int) -> dict[str, float]:
    x_min, y_min, x_max, y_max = [float(value) for value in bbox_xyxy]
    return {
        "left": round((x_min / width) * 100, 4),
        "top": round((y_min / height) * 100, 4),
        "width": round(((x_max - x_min) / width) * 100, 4),
        "height": round(((y_max - y_min) / height) * 100, 4),
    }


def build_review_item(key: str, item: dict[str, Any], processed: dict[str, Any]) -> dict[str, Any]:
    image_path = Path(str(item["output_image"]))
    label_path = Path(str(item["output_label"]))
    image_base64, content_type, width, height = image_to_base64(image_path)
    bbox_xyxy = item.get("bbox_xyxy") or []
    if len(bbox_xyxy) != 4:
        raise ValueError(f"Missing bbox_xyxy for review item: {key}")

    stats = review_stats(processed)
    pending_keys = [pending_key for pending_key, pending_item in sorted_items(processed) if item_is_pending(pending_item)]
    position = pending_keys.index(key) + 1 if key in pending_keys else 0
    return {
        "key": key,
        "source": item.get("source", ""),
        "split": item.get("split", ""),
        "output_image": str(image_path),
        "output_label": str(label_path),
        "image_name": image_path.name,
        "image_content_type": content_type,
        "image_base64": image_base64,
        "image_width": width,
        "image_height": height,
        "bbox_xyxy": bbox_xyxy,
        "bbox_percent": bbox_percent(bbox_xyxy, width, height),
        "yolo_label": item.get("yolo_label", ""),
        "mrz_lines": item.get("mrz_lines", []),
        "mrz_score": item.get("mrz_score", 0),
        "ocr_ms": item.get("ocr_ms", 0),
        "position": position,
        "stats": stats,
    }


def get_next_review_item(after_key: str = "") -> dict[str, Any]:
    processed = load_processed()
    key = next_pending_key(after_key)
    if key is None:
        return {
            "status": "empty",
            "current": None,
            "stats": review_stats(processed),
        }
    item = processed["items"][key]
    return {
        "status": "ok",
        "current": build_review_item(key, item, processed),
        "stats": review_stats(processed),
    }


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    suffix = path.suffix
    stem = path.stem
    parent = path.parent
    for index in range(1, 10000):
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot create unique destination for {path}")


def move_rejected_artifacts(item: dict[str, Any]) -> dict[str, str]:
    image_path = Path(str(item.get("output_image", "")))
    label_path = Path(str(item.get("output_label", "")))
    split = str(item.get("split") or "unknown")
    dataset_dir = dataset_paths()["dataset_dir"]
    rejected_image_dir = dataset_dir / "review" / "rejected" / "images" / split
    rejected_label_dir = dataset_dir / "review" / "rejected" / "labels" / split
    rejected_image_dir.mkdir(parents=True, exist_ok=True)
    rejected_label_dir.mkdir(parents=True, exist_ok=True)

    moved: dict[str, str] = {}
    if image_path.exists():
        target_image = unique_destination(rejected_image_dir / image_path.name)
        shutil.move(str(image_path), str(target_image))
        moved["rejected_image"] = str(target_image)
    if label_path.exists():
        target_label = unique_destination(rejected_label_dir / label_path.name)
        shutil.move(str(label_path), str(target_label))
        moved["rejected_label"] = str(target_label)
    return moved


def submit_review_decision(key: str, decision: str) -> dict[str, Any]:
    if decision not in {"approved", "rejected"}:
        raise ValueError("decision must be approved or rejected")

    processed = load_processed()
    items = processed.get("items", {})
    if key not in items:
        raise KeyError(f"Review item not found: {key}")

    item = items[key]
    moved = move_rejected_artifacts(item) if decision == "rejected" else {}
    item["review_status"] = decision
    item["reviewed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    item.update(moved)
    save_processed(processed)

    state = load_state()
    state["last_key"] = key
    history = state.setdefault("history", [])
    if isinstance(history, list):
        history.append({"key": key, "decision": decision, "reviewed_at": item["reviewed_at"]})
        del history[:-100]
    save_state(state)

    return {
        "status": "ok",
        "decision": decision,
        "key": key,
        "moved": moved,
        "next": get_next_review_item(key).get("current"),
        "stats": review_stats(processed),
    }
