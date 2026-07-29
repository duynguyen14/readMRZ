# READMRZ

CPU-only MRZ reader for passport and visa images.

## Run

Run local API server from source:

```powershell
python main.py
```

The API listens at:

```text
http://127.0.0.1:8080
```

```powershell
python -m mrz_reader.cli "path\to\passport.jpg" --pretty
```

Portable build target:

```powershell
.\build_portable.ps1
.\dist\readmrz\readmrz.exe "path\to\passport.jpg" --pretty
```

No-setup portable usage after build:

```powershell
.\read_image.bat "path\to\passport.jpg"
.\run_server.bat
```

Server mode keeps the OCR model loaded in RAM:

```powershell
.\dist\readmrz\readmrz.exe --server 8080
```

Then send a base64 image:

```powershell
$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("path\to\passport.jpg"))
$body = @{ image_base64 = $b64; filename = "passport.jpg" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/read -Body $body -ContentType "application/json"
```

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
-> OpenCV MRZ candidate detector
-> RapidOCR ONNX Runtime CPU OCR
-> MRZ charset normalization
-> ICAO-style format parsing and checksum validation
-> JSON output
```

The CLI interface is stable so the OCR model can later be replaced with a fine-tuned MRZ ONNX model without changing callers.

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

The script uses RapidOCR/PaddleOCR text boxes to find MRZ-like lines and writes one YOLO label per detected MRZ block. Images and labels are written one by one; `processed.json` metadata is flushed in batches controlled by `READMRZ_PROCESSED_BATCH_SIZE`. Review the generated boxes before using them as final detector training data.

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
