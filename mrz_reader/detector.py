from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Candidate:
    image: np.ndarray
    box: tuple[int, int, int, int]
    score: float
    source: str


def resize_for_speed(image: np.ndarray, max_side: int = 1600) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    largest = max(height, width)
    if largest <= max_side:
        return image, 1.0
    scale = max_side / largest
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return resized, scale


def crop_box(image: np.ndarray, x: int, y: int, w: int, h: int, padding: int = 12) -> np.ndarray:
    height, width = image.shape[:2]
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(width, x + w + padding)
    y1 = min(height, y + h + padding)
    return image[y0:y1, x0:x1]


def bottom_band_candidates(image: np.ndarray) -> list[Candidate]:
    height, width = image.shape[:2]
    candidates: list[Candidate] = []
    for idx, fraction in enumerate((0.34, 0.46)):
        y = int(height * (1.0 - fraction))
        crop = image[y:height, :]
        candidates.append(Candidate(crop, (0, y, width, height - y), 0.66 + idx * 0.03, "bottom_band"))
    return candidates


def morphology_candidates(image: np.ndarray) -> list[Candidate]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)
    grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad = np.absolute(grad)
    min_val, max_val = float(np.min(grad)), float(np.max(grad))
    if max_val - min_val > 0:
        grad = ((grad - min_val) / (max_val - min_val) * 255).astype("uint8")
    else:
        grad = np.zeros_like(gray)

    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rect_kernel)
    _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (45, 9)),
        iterations=2,
    )
    thresh = cv2.erode(thresh, None, iterations=1)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[Candidate] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < width * height * 0.008:
            continue
        aspect = w / max(1, h)
        if aspect < 3.0:
            continue
        if w < width * 0.35:
            continue
        bottom_bias = y / max(1, height)
        score = min(0.95, 0.45 + 0.25 * min(aspect / 10, 1) + 0.30 * bottom_bias)
        roi = crop_box(image, x, y, w, h, padding=18)
        candidates.append(Candidate(roi, (x, y, w, h), score, "morphology"))

    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[:3]


def preprocess_candidate(image: np.ndarray) -> list[np.ndarray]:
    variants = [image]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    variants.append(cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR))
    return variants


def generate_candidates(image: np.ndarray) -> list[Candidate]:
    resized, scale = resize_for_speed(image)
    candidates = morphology_candidates(resized) + bottom_band_candidates(resized)
    if scale != 1.0:
        # Candidate images are already cropped from the resized source, which is intentional for speed.
        pass
    unique: list[Candidate] = []
    seen: set[tuple[int, int, int, int, str]] = set()
    for candidate in candidates:
        key = (*candidate.box, candidate.source)
        if key not in seen and candidate.image.size:
            seen.add(key)
            unique.append(candidate)
    unique.sort(key=lambda item: item.score, reverse=True)
    return unique[:4]
