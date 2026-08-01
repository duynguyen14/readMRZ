from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import time
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from .label_review import (
    approve_pending_rotation,
    correct_review_box,
    get_next_review_item,
    get_previous_review_item,
    submit_review_decision,
)
from .mrz import parse_mrz, result_to_dict
from .ocr import MrzOcrEngine
from .ocr_line_review import (
    get_next_ocr_line_review_item,
    get_previous_ocr_line_review_item,
    submit_ocr_line_review_decision,
)
from .ocr_line2_review import (
    get_next_ocr_line2_review_item,
    get_previous_ocr_line2_review_item,
    submit_ocr_line2_review_decision,
)
from .ocr_line3_review import (
    get_next_ocr_line3_review_item,
    get_previous_ocr_line3_review_item,
    submit_ocr_line3_review_decision,
)
from .document_orientation import PaddleDocumentOrientation, env_bool
from .custom_mrz_ocr import CustomMrzCtcRecognizer
from .env_config import env_value, read_env_file
from .yolo_detector import YoloMrzDetector
from .yolo_upload_pipeline import compact_yolo_read_payload, process_yolo_upload


LOG_PATH = Path(__file__).resolve().parents[1] / "readmrz-api.log"


def log_api(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    try:
        with LOG_PATH.open("a", encoding="utf-8") as log_file:
            log_file.write(line + "\n")
    except Exception:
        pass


def read_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def read_array(image, engine: MrzOcrEngine, *, input_name: str) -> dict:
    started = time.perf_counter()
    attempt = engine.read(image)
    parse_started = time.perf_counter()
    parsed = parse_mrz(attempt.lines)
    parse_latency_ms = int((time.perf_counter() - parse_started) * 1000)
    latency_ms = int((time.perf_counter() - started) * 1000)
    payload = result_to_dict(
        parsed,
        raw_lines=attempt.lines,
        ocr_score=attempt.ocr_score,
        detector_score=attempt.detector_score,
        latency_ms=latency_ms,
        detector_latency_ms=attempt.detector_latency_ms,
        ocr_latency_ms=attempt.ocr_latency_ms,
        parse_latency_ms=parse_latency_ms,
        candidates_evaluated=attempt.candidates_evaluated,
        ocr_passes=attempt.ocr_passes,
    )
    payload["input"] = input_name
    payload["engine"] = {
        "detector": "opencv-mrz-heuristic",
        "ocr": "rapidocr-onnxruntime-cpu",
        "portable": True,
    }
    return payload


def decode_image_bytes(data: bytes):
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode request body as PNG/JPG image")
    return image


def decode_base64_image(value: str):
    if "," in value and value.lower().startswith("data:"):
        value = value.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(value, validate=True)
    except Exception as exc:
        raise ValueError("Invalid image_base64 value") from exc
    return decode_image_bytes(image_bytes)


def decode_request_image(data: bytes, content_type: str):
    if "application/json" in content_type.lower():
        try:
            payload = json.loads(data.decode("utf-8"))
        except Exception as exc:
            raise ValueError("Request body must be valid JSON") from exc
        image_base64 = payload.get("image_base64") or payload.get("base64")
        if not isinstance(image_base64, str) or not image_base64.strip():
            raise ValueError("JSON body must include image_base64")
        return decode_base64_image(image_base64), str(payload.get("filename") or "base64_request")
    return decode_image_bytes(data), "raw_image_request"


def run_server(port: int, *, host: str = "127.0.0.1") -> int:
    server_env = read_env_file()
    opencv_cpu_threads = max(
        1, int(env_value(server_env, "READMRZ_OPENCV_CPU_THREADS", "2"))
    )
    cv2.setNumThreads(opencv_cpu_threads)
    yolo_cpu_threads = max(
        1, int(env_value(server_env, "READMRZ_YOLO_CPU_THREADS", "8"))
    )
    orientation_cpu_threads = max(
        1, int(env_value(server_env, "READMRZ_ORIENTATION_CPU_THREADS", "2"))
    )
    ocr_cpu_threads = max(
        1, int(env_value(server_env, "READMRZ_CUSTOM_OCR_CPU_THREADS", "4"))
    )
    log_api(
        "CPU thread limits "
        f"yolo={yolo_cpu_threads} orientation={orientation_cpu_threads} "
        f"ocr={ocr_cpu_threads} opencv={cv2.getNumThreads()}"
    )
    engine: MrzOcrEngine | None = None
    yolo_detector: YoloMrzDetector | None = None
    document_orientation: PaddleDocumentOrientation | None = None
    custom_mrz_ocr: CustomMrzCtcRecognizer | None = None

    def get_engine() -> MrzOcrEngine:
        nonlocal engine
        if engine is None:
            log_api("Loading MRZ OCR engine")
            engine = MrzOcrEngine()
        return engine

    def get_yolo_detector() -> YoloMrzDetector:
        nonlocal yolo_detector
        if yolo_detector is None:
            log_api("Loading MRZ YOLO detector")
            yolo_detector = YoloMrzDetector()
            log_api(f"Loaded MRZ YOLO detector model_load_ms={yolo_detector.load_ms}")
        return yolo_detector

    def get_document_orientation() -> PaddleDocumentOrientation:
        nonlocal document_orientation
        if document_orientation is None:
            log_api("Loading Paddle document orientation model")
            document_orientation = PaddleDocumentOrientation()
            log_api(
                "Loaded Paddle document orientation model "
                f"enabled={document_orientation.enabled} model_load_ms={document_orientation.load_ms}"
            )
        return document_orientation

    def get_custom_mrz_ocr() -> CustomMrzCtcRecognizer:
        nonlocal custom_mrz_ocr
        if custom_mrz_ocr is None:
            log_api("Loading custom MRZ CTC recognizer")
            custom_mrz_ocr = CustomMrzCtcRecognizer()
            log_api(
                "Loaded custom MRZ CTC recognizer "
                f"enabled={custom_mrz_ocr.enabled} model_load_ms={custom_mrz_ocr.load_ms}"
            )
        return custom_mrz_ocr

    if env_bool(server_env, "READMRZ_API_PRELOAD_MODELS", True):
        preload_started = time.perf_counter()
        log_api("Preloading upload pipeline models")
        orientation_model = get_document_orientation()
        detector_model = get_yolo_detector()
        recognizer_model = get_custom_mrz_ocr()
        log_api(
            "Preloaded upload pipeline models "
            f"total_ms={int((time.perf_counter() - preload_started) * 1000)}"
        )
        if env_bool(server_env, "READMRZ_API_WARMUP_MODELS", True):
            warmup_started = time.perf_counter()
            orientation_ms = orientation_model.warmup()
            yolo_ms = detector_model.warmup()
            ocr_ms = recognizer_model.warmup()
            log_api(
                "Warmed upload pipeline models "
                f"orientation_ms={orientation_ms} yolo_ms={yolo_ms} ocr_ms={ocr_ms} "
                f"total_ms={int((time.perf_counter() - warmup_started) * 1000)}"
            )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            return

        def send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:
            log_api(f"OPTIONS {self.path}")
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Max-Age", "86400")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:
            log_api(f"GET {self.path}")
            parsed_url = urlparse(self.path)
            if parsed_url.path == "/health":
                self.send_json(200, {"ok": True, "engine": "readmrz"})
                return
            if parsed_url.path == "/label-review/next":
                params = parse_qs(parsed_url.query)
                after_key = params.get("after_key", [""])[0]
                self.send_json(200, get_next_review_item(after_key))
                return
            if parsed_url.path == "/label-review/previous":
                params = parse_qs(parsed_url.query)
                before_key = params.get("before_key", [""])[0]
                self.send_json(200, get_previous_review_item(before_key))
                return
            if parsed_url.path == "/ocr-line-review/next":
                params = parse_qs(parsed_url.query)
                after_id = int(params.get("after_id", ["0"])[0] or "0")
                self.send_json(200, get_next_ocr_line_review_item(after_id))
                return
            if parsed_url.path == "/ocr-line-review/previous":
                params = parse_qs(parsed_url.query)
                before_id = int(params.get("before_id", ["0"])[0] or "0")
                self.send_json(200, get_previous_ocr_line_review_item(before_id))
                return
            if parsed_url.path == "/ocr-line2-review/next":
                params = parse_qs(parsed_url.query)
                after_id = int(params.get("after_id", ["0"])[0] or "0")
                self.send_json(200, get_next_ocr_line2_review_item(after_id))
                return
            if parsed_url.path == "/ocr-line2-review/previous":
                params = parse_qs(parsed_url.query)
                before_id = int(params.get("before_id", ["0"])[0] or "0")
                self.send_json(200, get_previous_ocr_line2_review_item(before_id))
                return
            if parsed_url.path == "/ocr-line3-review/next":
                params = parse_qs(parsed_url.query)
                after_id = int(params.get("after_id", ["0"])[0] or "0")
                self.send_json(200, get_next_ocr_line3_review_item(after_id))
                return
            if parsed_url.path == "/ocr-line3-review/previous":
                params = parse_qs(parsed_url.query)
                before_id = int(params.get("before_id", ["0"])[0] or "0")
                self.send_json(200, get_previous_ocr_line3_review_item(before_id))
                return
            self.send_json(404, {"error": "Unknown route"})

        def do_POST(self) -> None:
            log_api(f"POST {self.path}")
            parsed_url = urlparse(self.path)
            if parsed_url.path == "/label-review/decision":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    data = self.rfile.read(length)
                    payload = json.loads(data.decode("utf-8")) if data else {}
                    key = str(payload.get("key") or "")
                    decision = str(payload.get("decision") or "")
                    if not key:
                        raise ValueError("key is required")
                    result = submit_review_decision(key, decision)
                    self.send_json(200, result)
                except Exception as exc:
                    log_api(f"LABEL_REVIEW error {exc}")
                    self.send_json(400, {"status": "error", "error": str(exc)})
                return

            if parsed_url.path == "/label-review/correct-box":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    data = self.rfile.read(length)
                    payload = json.loads(data.decode("utf-8")) if data else {}
                    key = str(payload.get("key") or "")
                    bbox_xyxy = payload.get("bbox_xyxy") or []
                    if not key:
                        raise ValueError("key is required")
                    result = correct_review_box(key, bbox_xyxy)
                    self.send_json(200, result)
                except Exception as exc:
                    log_api(f"LABEL_REVIEW_CORRECT_BOX error {exc}")
                    self.send_json(400, {"status": "error", "error": str(exc)})
                return

            if parsed_url.path == "/label-review/approve-rotation":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    data = self.rfile.read(length)
                    payload = json.loads(data.decode("utf-8")) if data else {}
                    rotation_angle = int(payload.get("rotation_angle") or 0)
                    result = approve_pending_rotation(rotation_angle)
                    log_api(
                        "LABEL_REVIEW_APPROVE_ROTATION done "
                        f"rotation={rotation_angle} updated={result.get('updated')}"
                    )
                    self.send_json(200, result)
                except Exception as exc:
                    log_api(f"LABEL_REVIEW_APPROVE_ROTATION error {exc}")
                    self.send_json(400, {"status": "error", "error": str(exc)})
                return

            if parsed_url.path == "/ocr-line-review/decision":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    data = self.rfile.read(length)
                    payload = json.loads(data.decode("utf-8")) if data else {}
                    line_id = int(payload.get("id") or 0)
                    decision = str(payload.get("decision") or "")
                    final_text = str(payload.get("final_text") or "")
                    if line_id <= 0:
                        raise ValueError("id is required")
                    result = submit_ocr_line_review_decision(line_id, decision, final_text)
                    self.send_json(200, result)
                except Exception as exc:
                    log_api(f"OCR_LINE_REVIEW error {exc}")
                    self.send_json(400, {"status": "error", "error": str(exc)})
                return

            if parsed_url.path == "/ocr-line2-review/decision":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    data = self.rfile.read(length)
                    payload = json.loads(data.decode("utf-8")) if data else {}
                    label_item_id = int(payload.get("label_item_id") or 0)
                    decision = str(payload.get("decision") or "")
                    lines = payload.get("lines") or []
                    if label_item_id <= 0:
                        raise ValueError("label_item_id is required")
                    result = submit_ocr_line2_review_decision(label_item_id, decision, lines)
                    self.send_json(200, result)
                except Exception as exc:
                    log_api(f"OCR_LINE2_REVIEW error {exc}")
                    self.send_json(400, {"status": "error", "error": str(exc)})
                return

            if parsed_url.path == "/ocr-line3-review/decision":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    data = self.rfile.read(length)
                    payload = json.loads(data.decode("utf-8")) if data else {}
                    label_item_id = int(payload.get("label_item_id") or 0)
                    decision = str(payload.get("decision") or "")
                    lines = payload.get("lines") or []
                    if label_item_id <= 0:
                        raise ValueError("label_item_id is required")
                    result = submit_ocr_line3_review_decision(label_item_id, decision, lines)
                    self.send_json(200, result)
                except Exception as exc:
                    log_api(f"OCR_LINE3_REVIEW error {exc}")
                    self.send_json(400, {"status": "error", "error": str(exc)})
                return

            if parsed_url.path == "/yolo-mrz-read-base64":
                try:
                    request_started = time.perf_counter()
                    content_type = self.headers.get("Content-Type", "")
                    if "application/json" not in content_type.lower():
                        raise ValueError("Content-Type must be application/json")
                    length = int(self.headers.get("Content-Length", "0"))
                    data = self.rfile.read(length)
                    image, input_name = decode_request_image(data, content_type)
                    pipeline_payload = process_yolo_upload(
                        image,
                        get_yolo_detector(),
                        get_document_orientation(),
                        get_custom_mrz_ocr(),
                        include_images=False,
                    )
                    payload = compact_yolo_read_payload(pipeline_payload)
                    payload["input"] = input_name
                    payload["latency_ms"] = int(
                        (time.perf_counter() - request_started) * 1000
                    )
                    log_api(
                        "YOLO_MRZ_READ_BASE64 done "
                        f"input={input_name} found={payload['found']} "
                        f"lines={len(payload['lines'])} "
                        f"detector_ms={payload['processing'].get('detector_ms')} "
                        f"ocr_ms={payload['processing'].get('ocr_ms')} "
                        f"latency_ms={payload['latency_ms']}"
                    )
                    self.send_json(200, payload)
                except Exception as exc:
                    log_api(f"YOLO_MRZ_READ_BASE64 error {exc}")
                    self.send_json(400, {"found": False, "error": str(exc)})
                return

            if parsed_url.path == "/yolo-mrz-detect":
                try:
                    request_started = time.perf_counter()
                    length = int(self.headers.get("Content-Length", "0"))
                    data = self.rfile.read(length)
                    content_type = self.headers.get("Content-Type", "")
                    image, input_name = decode_request_image(data, content_type)
                    payload = process_yolo_upload(
                        image,
                        get_yolo_detector(),
                        get_document_orientation(),
                        get_custom_mrz_ocr(),
                    )
                    payload["input"] = input_name
                    payload["latency_ms"] = int((time.perf_counter() - request_started) * 1000)
                    log_api(
                        "YOLO_MRZ_DETECT done "
                        f"input={input_name} found={payload['found']} "
                        f"boxes={len(payload['boxes'])} detector_ms={payload['detector_ms']} "
                        f"fallback_used={payload.get('fallback_used')} "
                        f"rotation={payload.get('selected_rotation_angle')} "
                        f"paddle_rotation={payload.get('orientation', {}).get('applied_angle')} "
                        f"lines={len(payload.get('line_crops', []))} "
                        f"ocr_found={payload.get('mrz_ocr', {}).get('found')} "
                        f"ocr_ms={payload.get('processing', {}).get('ocr_ms')} "
                        f"latency_ms={payload['latency_ms']}"
                    )
                    self.send_json(200, payload)
                except Exception as exc:
                    log_api(f"YOLO_MRZ_DETECT error {exc}")
                    self.send_json(400, {"found": False, "boxes": [], "best_box": None, "error": str(exc)})
                return

            if parsed_url.path != "/read":
                self.send_json(404, {"error": "Unknown route"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                data = self.rfile.read(length)
                content_type = self.headers.get("Content-Type", "")
                image, input_name = decode_request_image(data, content_type)
                log_api(f"READ start input={input_name} content_type={content_type} bytes={length}")
                payload = read_array(image, get_engine(), input_name=input_name)
                log_api(
                    "READ done "
                    f"input={input_name} found={payload['found']} "
                    f"confidence={payload['confidence']} "
                    f"detector_ms={payload['detector_latency_ms']} "
                    f"ocr_ms={payload['ocr_latency_ms']} "
                    f"parse_ms={payload['parse_latency_ms']} "
                    f"total_ms={payload['latency_ms']}"
                )
                self.send_json(200 if payload["found"] else 422, payload)
            except Exception as exc:
                log_api(f"READ error {exc}")
                self.send_json(400, {"found": False, "confidence": 0.0, "error": str(exc)})

    server = ThreadingHTTPServer((host, port), Handler)
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    log_api(f"READMRZ server listening on http://{display_host}:{port} bind={host}:{port}")
    log_api("POST JSON {\"image_base64\":\"...\"} to /read")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read passport/visa MRZ from a PNG or JPG image.")
    parser.add_argument("image", nargs="?", help="Input PNG/JPG image path")
    parser.add_argument(
        "-o",
        "--output",
        help="Optional JSON output path. If omitted, JSON is printed to stdout.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    parser.add_argument(
        "--server",
        type=int,
        metavar="PORT",
        help="Run a local HTTP server and keep the OCR model warm in memory.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host for --server. Use 0.0.0.0 for LAN access.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.server:
        return run_server(args.server, host=args.host)
    if not args.image:
        print("Missing image path. Use --help for usage.", file=sys.stderr)
        return 1
    started = time.perf_counter()
    image_path = Path(args.image)
    try:
        image = read_image(image_path)
        engine = MrzOcrEngine()
        payload = read_array(image, engine, input_name=str(image_path))
        payload["latency_ms"] = int((time.perf_counter() - started) * 1000)
        json_text = json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None)
        if args.output:
            Path(args.output).write_text(json_text, encoding="utf-8")
        else:
            print(json_text)
        return 0 if payload["found"] else 2
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        error_payload = {
            "found": False,
            "input": str(image_path),
            "confidence": 0.0,
            "latency_ms": latency_ms,
            "error": str(exc),
        }
        print(json.dumps(error_payload, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
