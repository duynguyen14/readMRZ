from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Iterable

import numpy as np
from rapidocr_onnxruntime import RapidOCR

from .detector import Candidate, generate_candidates, preprocess_candidate
from .mrz import normalize_mrz_text


@dataclass
class OcrAttempt:
    lines: list[str]
    ocr_score: float
    detector_score: float
    source: str
    detector_latency_ms: int = 0
    ocr_latency_ms: int = 0
    candidates_evaluated: int = 0
    ocr_passes: int = 0


def line_mrz_likeness(text: str) -> float:
    normalized = normalize_mrz_text(text)
    if not normalized:
        return 0.0
    charset_ratio = len(normalized) / max(1, len(text.replace(" ", "")))
    length_score = min(1.0, len(normalized) / 30)
    filler_score = min(1.0, normalized.count("<") / max(1, len(normalized)) * 4)
    digit_score = 1.0 if re.search(r"\d", normalized) else 0.65
    return 0.45 * charset_ratio + 0.30 * length_score + 0.15 * filler_score + 0.10 * digit_score


def choose_mrz_lines(items: Iterable[tuple[str, float, float]]) -> tuple[list[str], float]:
    scored: list[tuple[str, float, float, float]] = []
    for text, score, y_pos in items:
        line = normalize_mrz_text(text)
        if len(line) < 12:
            continue
        likeness = line_mrz_likeness(text)
        if likeness < 0.48:
            continue
        scored.append((line, float(score), likeness, y_pos))

    if not scored:
        return [], 0.0

    scored.sort(key=lambda item: (len(item[0]), item[1] + item[2]), reverse=True)
    target_lengths = (44, 36, 30)
    best_group: list[tuple[str, float, float, float]] = []
    best_score = -1.0
    for target in target_lengths:
        group = [item for item in scored if abs(len(item[0]) - target) <= 10]
        if len(group) >= 2:
            candidate = group[:3]
            score = sum(item[1] + item[2] for item in candidate) / len(candidate)
            if score > best_score:
                best_group = candidate
                best_score = score

    if not best_group:
        best_group = scored[:3]

    best_group.sort(key=lambda item: item[3])
    lines = [item[0] for item in best_group[:3]]
    avg_ocr = sum(item[1] for item in best_group[: max(1, len(lines))]) / max(1, len(lines))
    return lines, max(0.0, min(1.0, avg_ocr))


class MrzOcrEngine:
    def __init__(self, *, text_score: float = 0.35) -> None:
        self.ocr = RapidOCR(text_score=text_score, use_angle_cls=False, print_verbose=False)

    def run_candidate(self, candidate: Candidate) -> OcrAttempt:
        best = OcrAttempt([], 0.0, candidate.score, candidate.source)
        ocr_started = time.perf_counter()
        ocr_passes = 0
        for image in preprocess_candidate(candidate.image):
            ocr_passes += 1
            result, _ = self.ocr(np.asarray(image))
            if not result:
                continue
            items = []
            for row in result:
                if len(row) < 3:
                    continue
                y_pos = 0.0
                try:
                    y_pos = float(sum(point[1] for point in row[0]) / len(row[0]))
                except Exception:
                    y_pos = 0.0
                items.append((str(row[1]), float(row[2]), y_pos))
            lines, score = choose_mrz_lines(items)
            if len(lines) >= len(best.lines) and score >= best.ocr_score:
                best = OcrAttempt(lines, score, candidate.score, candidate.source)
        best.ocr_latency_ms = int((time.perf_counter() - ocr_started) * 1000)
        best.ocr_passes = ocr_passes
        return best

    def read(self, image: np.ndarray) -> OcrAttempt:
        attempts: list[OcrAttempt] = []
        detector_started = time.perf_counter()
        candidates = generate_candidates(image)
        detector_latency_ms = int((time.perf_counter() - detector_started) * 1000)
        total_ocr_latency_ms = 0
        total_ocr_passes = 0
        for candidate in candidates:
            attempt = self.run_candidate(candidate)
            attempts.append(attempt)
            total_ocr_latency_ms += attempt.ocr_latency_ms
            total_ocr_passes += attempt.ocr_passes
            if len(attempt.lines) >= 2 and attempt.ocr_score >= 0.82 and attempt.detector_score >= 0.65:
                break
        if not attempts:
            return OcrAttempt(
                [],
                0.0,
                0.0,
                "none",
                detector_latency_ms=detector_latency_ms,
                candidates_evaluated=0,
            )
        attempts.sort(
            key=lambda item: (len(item.lines), item.ocr_score + item.detector_score),
            reverse=True,
        )
        best_attempt = attempts[0]
        best_attempt.detector_latency_ms = detector_latency_ms
        best_attempt.ocr_latency_ms = total_ocr_latency_ms
        best_attempt.candidates_evaluated = len(attempts)
        best_attempt.ocr_passes = total_ocr_passes
        return best_attempt
