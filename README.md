# READMRZ

MRZ and Vietnamese visa reader for passport and visa images.

## Run

Run local API server from source:

```powershell
python main.py
```

The API bind host and port are configured in `.env`:

```text
READMRZ_API_HOST=0.0.0.0
READMRZ_API_PORT=8080
```

On the same machine, call `http://127.0.0.1:8080`. From LAN, call `http://<server-lan-ip>:8080`.

Portable build target:

```powershell
.\build_portable.ps1
.\dist\readmrz\readmrz.exe
```

No-setup portable usage after build:

```powershell
.\run_server.bat
```

Server mode keeps the OCR model loaded in RAM:

```powershell
.\dist\readmrz\readmrz.exe --port 8080
```

Then send a base64 image:

```powershell
$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("path\to\passport.jpg"))
$body = @{ image_base64 = $b64; filename = "passport.jpg" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/yolo-mrz-read-base64 -Body $body -ContentType "application/json"
```

## Async passport batch upload

Create the SQL tables once with `sql/create_passport_batch_tables.sql`, then configure the
storage and worker settings from `.env`. Upload one or many image parts named `files`:

```powershell
curl.exe -X POST "http://127.0.0.1:8080/api/passport-batches" `
  -H "X-API-Key: YOUR_KEY" `
  -F "files=@C:\images\passport-001.jpg" `
  -F "files=@C:\images\passport-002.jpg" `
  -F "client_request_id=import-20261015-001" `
  -F "source_system=passport-import"
```

The API persists the files and queue rows, then responds without waiting for OCR:

```json
{
  "batch_id": "9e884bd5-ef8f-4339-8244-94e9fe15dc07",
  "received": 2,
  "status": "queued"
}
```

The background worker claims rows atomically, runs the existing document orientation,
YOLO MRZ, crop/line split, OCR and ICAO parsing pipeline, and writes each result back to
`dbo.readmrz_passport_items`. Batch counters and terminal status are maintained in
`dbo.readmrz_passport_batches`. Items left in `processing` by an interrupted process are
returned to the queue after the configured lock timeout.

This endpoint accepts any number of file parts, subject to
`READMRZ_PASSPORT_BATCH_MAX_REQUEST_BYTES`. For a very large import, send several batches
so the HTTP request body does not need to be held in memory at once.

### Browse batches and review-ready items

List batches with optional status, keyword, paging and sort filters:

```text
GET /api/passport-batches?status=completed&page=1&page_size=20&sort=created_desc
```

List items in one batch. Both processing and review status can be filtered:

```text
GET /api/passport-batches/{batch_id}/items?status=succeeded&review_status=pending&page=1&page_size=50
```

Navigate through successfully processed items that are still pending review:

```text
GET /api/passport-batches/{batch_id}/review/next
GET /api/passport-batches/{batch_id}/review/next?current_item_key={item_key}
GET /api/passport-batches/{batch_id}/review/previous?current_item_key={item_key}
```

The initial `next` request returns the most recently processed pending item. Navigation
uses `ProcessedAt` and item ID as a stable cursor. A response with `"item": null` means
there is no matching item in that direction.

Preview the original image without embedding large base64 values in list responses:

```text
GET /api/passport-batches/items/{item_key}/image
```

Submit a review decision after the user has checked an item:

```text
POST /api/passport-batches/items/review
Content-Type: application/json
```

```json
{
  "id": "6c9b9bf1-2874-4a61-a5b1-1130dc00ac3e",
  "status": "approved",
  "reviewed_by": "operator-01",
  "note": "Optional review note"
}
```

`id` accepts either the numeric database item ID or public `item_key` UUID. The accepted
decisions are `approved`, `corrected` and `rejected`. Only successfully processed items
can be reviewed; other processing states return HTTP 409. The response contains the
updated item and refreshed batch statistics.

Batch and item-list responses include processing and review statistics. `reviewed` is the
sum of approved, corrected and rejected items; `unreviewed` is the number of pending items.
When `READMRZ_API_KEY` is configured, send it in the `X-API-Key` header for these routes.

Face match uses InsightFace model packs: SCRFD for face detection and ArcFace for recognition. Face match 1-1 keeps the same request/response shape as the older backend:

```text
POST http://127.0.0.1:8080/api/passport-face-match/verify
```

```json
{
  "api_key": "...",
  "passport_face_base64": "...",
  "passport_face_file_name": "passport_face.jpg",
  "uploaded_face_base64": "...",
  "uploaded_face_file_name": "uploaded_face.jpg"
}
```

Batch face/passport matching accepts a list of base64 images. The API classifies each image, reads MRZ for passport pages, matches faces to passports, and returns only paired objects:

```text
POST http://127.0.0.1:8080/api/passport-face-match/batch
```

```json
{
  "api_key": "...",
  "items": [
    { "file_name": "face_001.jpg", "base64": "..." },
    { "file_name": "passport_001.jpg", "base64": "..." }
  ]
}
```

```json
{
  "data": [
    {
      "face": { "file_name": "face_001.jpg", "base64": "..." },
      "passport": {
        "file_name": "passport_001.jpg",
        "base64": "...",
        "face_base64": "...",
        "match": { "score": 0.72, "decision": "match" },
        "mrz": {
          "found": true,
          "confidence": 0.88,
          "lines": [
            { "text": "P<VNM...", "confidence": 0.91 }
          ]
        },
        "parsed": {
          "passport_no": "B1234567",
          "full_name": "NGUYEN VAN A",
          "date_of_birth": "1990-01-01",
          "sex": "M",
          "nationality": "VNM",
          "expiry_date": "2030-01-01"
        }
      }
    }
  ]
}
```

Unmatched input is still returned as `{ "face": {...}, "passport": null }` or `{ "face": null, "passport": {...} }`.

The API also accepts data URLs:

```json
{
  "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ...",
  "filename": "passport.jpg"
}
```

## Output

The tool returns JSON with:

- `mrz_raw`
- `document_type`
- `confidence`
- `ocr_confidence`
- `detector_confidence`
- `checksum_pass`
- `latency_ms`
- parsed `fields`
- per-field checksum results

## Architecture

```text
PNG/JPG
-> document orientation normalization
-> YOLO MRZ detector
-> OpenCV crop, deskew and line splitting
-> custom CTC MRZ recognizer
-> MRZ charset normalization
-> ICAO-style format parsing and checksum validation
-> JSON output
```

## Generate YOLO MRZ Labels

Configure `.env`:

```text
READMRZ_SOURCE_IMAGE_DIR=D:/path/to/original/images
READMRZ_YOLO_DATASET_DIR=D:/path/to/output/mrz_yolo
READMRZ_OCR_ENGINE=paddle
READMRZ_PADDLE_PYTHON=D:/DocHochieu/Backend/.venv/Scripts/python.exe
PADDLE_TEXT_DETECTION_MODEL_DIR=D:/DocHochieu/Backend/models/paddle/PP-OCRv5_server_det
PADDLE_TEXT_RECOGNITION_MODEL_DIR=D:/DocHochieu/Backend/models/paddle/en_PP-OCRv5_mobile_rec
PADDLE_DOC_ORIENTATION_MODEL_DIR=C:/Users/Admin/.paddlex/official_models/PP-LCNet_x1_0_doc_ori
PADDLE_TEXTLINE_ORIENTATION_MODEL_DIR=C:/Users/Admin/.paddlex/official_models/PP-LCNet_x1_0_textline_ori
```

Run pseudo-label generation:

```powershell
python tools/generate_mrz_yolo_dataset.py
```

Output:

```text
generated_datasets/mrz_yolo/
  images/train/
  images/val/
  images/test/
  labels/train/
  labels/val/
  labels/test/
  review/no_mrz/
  data.yaml
  processed.json
  last_run_annotations.jsonl
  last_run_summary.json
```

The script uses PaddleOCR text boxes to find MRZ-like lines and writes one YOLO label per detected MRZ block. Images and labels are written one by one; `processed.json` metadata is flushed in batches controlled by `READMRZ_PROCESSED_BATCH_SIZE`. Review the generated boxes before using them as final detector training data.

## Review YOLO MRZ Labels

Run the local API:

```powershell
python main.py
```

Review endpoints:

```text
GET  http://127.0.0.1:8080/label-review/next
POST http://127.0.0.1:8080/label-review/decision
```

Decision body:

```json
{
  "key": "relative/source/image.jpg",
  "decision": "approved"
}
```

Use `decision=rejected` to move both generated image and label into:

```text
generated_datasets/mrz_yolo/review/rejected/images/
generated_datasets/mrz_yolo/review/rejected/labels/
```
