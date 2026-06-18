# MMAY Image Verification API — Documentation

> **Version:** 1.0.0  
> **Base URL:** `http://<host>:8000`  
> **Interactive Docs:** `/docs` (Swagger UI) · `/redoc` (ReDoc)

---

## Overview

The **Mukhyamantri Avas Yojana (MMAY) Image Verification API** verifies construction-stage photographs against the expected stage for a given scheme level. It uses a trained Keras model (`MMAY_Image_2-0.h5`) to classify submitted images and returns a structured verification result.

### Construction Level → Stage Mapping

| Level (`level`) | Expected Stage |
|:--------------:|----------------|
| `2`            | `plinth`       |
| `3`            | `roof_cast`    |
| `4`            | `completion`   |

### Confidence Threshold

The verification checks two conditions in order:
1. The predicted stage must exactly match the expected stage for the submitted level.
2. The model must produce a confidence score ≥ **70%** (default).

Anything below that confidence triggers a `manual_check_needed` outcome, even if the predicted stage is correct. This threshold is configurable via the `CONFIDENCE_THRESHOLD` environment variable.

---

## Authentication

No authentication is required. CORS is open to all origins (`*`) in the default configuration — restrict `allow_origins` in production.

---

## Endpoints

### 1. `GET /health`

**Summary:** Health check — reports service liveness and model readiness.

**Tags:** `Utility`

#### Request

No parameters, no body.

```
GET /health HTTP/1.1
```

#### Response `200 OK`

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "MMAY_Modelv2",
  "version": "1.0.0"
}
```

| Field          | Type    | Description                                          |
|----------------|---------|------------------------------------------------------|
| `status`       | string  | `"ok"` when model is ready, `"degraded"` otherwise  |
| `model_loaded` | boolean | `true` if the Keras model was loaded at startup      |
| `model_name`   | string  | Name of the loaded ML model file (without extension) |
| `version`      | string  | API version string (always `"1.0.0"`)               |

#### Example — Model Not Ready

```json
{
  "status": "degraded",
  "model_loaded": false,
  "model_name": "MMAY_Modelv2",
  "version": "1.0.0"
}
```

---

### 2. `POST /predict`

**Summary:** Verify a construction-stage photograph against a submitted scheme level.

**Tags:** `Prediction`  
**Content-Type:** `multipart/form-data`

#### Request

| Field      | Type    | Required | Description                                                            
     |
|------------|---------|----------|-----------------------------------------------------------------------------|
| `level`    | integer | ✅       | Construction level: `2` (plinth), `3` (roof_cast), `4` (completion)       |
| `image_id` | string  | ✅       | Caller-supplied image identifier, echoed back in the response             |
| `image`    | file    | ✅       | Site photograph — accepted formats: **JPEG**, **PNG**, **WebP** (max 10 MB)|

#### Example cURL

```bash
curl -X POST http://localhost:8000/predict \
  -F "level=2" \
  -F "image_id=23456P21" \
  -F "image=@/path/to/site_photo.jpg"
```

#### Example — Python `requests`

```python
import requests

with open("site_photo.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        data={"level": 2, "image_id": "23456P21"},
        files={"image": ("site_photo.jpg", f, "image/jpeg")},
    )

print(response.json())
```

---

#### Responses

##### `200 OK` — Verification successful

Returned when the model confidence meets the threshold **and** the predicted class matches the expected stage.

```json
{
  "status": "success",
  "predicted_class": "plinth",
  "predicted_confidence": 92.3,
  "predicted_level": 2,
  "submitted_level": 2,
  "expected_class": "plinth",
  "expected_confidence": 92.3,
  "image_id": "23456P21",
  "message": "Verification successful. Stage 'plinth' confirmed."
}
```

##### `200 OK` — Manual check needed

Returned when the predicted class does not match the expected stage, or if the class matches but the confidence score is below the threshold.

```json
{
  "status": "manual_check_needed",
  "predicted_class": "plinth",
  "predicted_confidence": 55.0,
  "predicted_level": 2,
  "submitted_level": 3,
  "expected_class": "roof_cast",
  "expected_confidence": 42.5,
  "image_id": "78901R03",
  "message": "Manual review required. Predicted stage 'plinth' does not match expected stage 'roof_cast'."
}
```

#### Response Schema — `PredictResponse`

| Field                  | Type              | Description                                                        |
|------------------------|-------------------|--------------------------------------------------------------------|
| `status`               | `PredictionStatus`| `"success"` or `"manual_check_needed"`                            |
| `predicted_class`      | string \| null    | Actual class label predicted by the model (null only if model index is unknown)|
| `predicted_confidence` | float [0–100]     | Confidence (%) for the predicted class                             |
| `predicted_level`      | integer \| null   | Construction level corresponding to `predicted_class`               |
| `submitted_level`      | integer           | The `level` value sent by the caller                               |
| `expected_class`       | string            | The stage label that maps to `submitted_level`                     |
| `expected_confidence`  | float [0–100]     | Confidence (%) for the expected class                              |
| `image_id`             | string            | Caller-supplied image identifier, echoed back verbatim             |
| `message`              | string            | Human-readable summary of the outcome                              |

#### `PredictionStatus` Enum

| Value                  | Meaning                                                       |
|------------------------|---------------------------------------------------------------|
| `"success"`            | Image verified — predicted stage matches the submitted level  |
| `"manual_check_needed"`| Requires human review (stage mismatch)      |

---

### 3. `POST /predict/batch`

**Summary:** Batch verify multiple construction-stage photographs from disk under a single scheme level.

**Tags:** `Prediction`  
**Content-Type:** `application/json`

#### Request

The request body is a JSON object with the following fields:

| Field   | Type              | Required | Description                                                            |
|---------|-------------------|----------|------------------------------------------------------------------------|
| `label`  | integer           | ✅       | Construction level: `2` (plinth), `3` (roof_cast), `4` (completion)     |
| `images` | array of objects  | ✅       | List of image entries to verify (minimum 1, maximum 10 items)           |

Each object in the `images` array contains:

| Field      | Type   | Required | Description                                                               |
|------------|--------|----------|---------------------------------------------------------------------------|
| `image_id` | string | ✅       | Caller-supplied image identifier, echoed back in the response             |
| `image`    | string | ✅       | File path to the image on the server (absolute or relative)               |

#### Example Request Body

```json
{
  "label": 2,
  "images": [
    {
      "image_id": "image_1",
      "image": "/images/image_1.jpg"
    },
    {
      "image_id": "image_2",
      "image": "/images/image_2.jpg"
    },
    {
      "image_id": "image_3",
      "image": "/images/image_3.jpg"
    }
  ]
}
```

#### Example cURL

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "label": 2,
    "images": [
      {"image_id": "image_1", "image": "/images/image_1.jpg"},
      {"image_id": "image_2", "image": "/images/image_2.jpg"},
      {"image_id": "image_3", "image": "/images/image_3.jpg"}
    ]
  }'
```

#### Responses

##### `200 OK`

Returns an object containing a list of `results`. Each result represents the verification outcome for a single image in the request.

```json
{
  "results": [
    {
      "status": "success",
      "predicted_class": "plinth",
      "predicted_confidence": 92.3,
      "predicted_level": 2,
      "submitted_level": 2,
      "expected_class": "plinth",
      "expected_confidence": 92.3,
      "image_id": "image_1",
      "message": "Verification successful. Stage 'plinth' confirmed."
    },
    {
      "status": "success",
      "predicted_class": "plinth",
      "predicted_confidence": 91.8,
      "predicted_level": 2,
      "submitted_level": 2,
      "expected_class": "plinth",
      "expected_confidence": 91.8,
      "image_id": "image_2",
      "message": "Verification successful. Stage 'plinth' confirmed."
    },
    {
      "status": "success",
      "predicted_class": "plinth",
      "predicted_confidence": 93.1,
      "predicted_level": 2,
      "submitted_level": 2,
      "expected_class": "plinth",
      "expected_confidence": 93.1,
      "image_id": "image_3",
      "message": "Verification successful. Stage 'plinth' confirmed."
    }
  ]
}
```

#### Response Schema — `BatchPredictResponse`

| Field     | Type                     | Description                                         |
|-----------|--------------------------|-----------------------------------------------------|
| `results` | array of `BatchImageResult` | List of verification results for each input image  |

Each `BatchImageResult` contains:

| Field                  | Type              | Description                                                        |
|------------------------|-------------------|--------------------------------------------------------------------|
| `status`               | `PredictionStatus`| `"success"` or `"manual_check_needed"`                            |
| `predicted_class`      | string \| null    | Actual class label predicted by the model                          |
| `predicted_confidence` | float [0–100]     | Confidence (%) for the predicted class                             |
| `predicted_level`      | integer \| null   | Construction level corresponding to `predicted_class`               |
| `submitted_level`      | integer           | The `label` value sent by the caller                               |
| `expected_class`       | string            | The stage label that maps to `submitted_level`                     |
| `expected_confidence`  | float [0–100]     | Confidence (%) for the expected class                              |
| `image_id`             | string            | Caller-supplied image identifier                                   |
| `message`              | string            | Human-readable summary of the outcome                              |

---

#### Error Responses

| HTTP Status | Condition                                                       | Example Detail                                                   |
|:-----------:|-----------------------------------------------------------------|------------------------------------------------------------------|
| `400`       | Uploaded file is empty or the image data is corrupt/unreadable | `"Uploaded image is empty."`                                     |
| `413`       | Image exceeds the 10 MB size limit                             | `"Image size (12.3 MB) exceeds the limit of 10 MB."`            |
| `415`       | Unsupported MIME type (not JPEG / PNG / WebP)                  | `"Unsupported image type 'image/gif'. Allowed: image/jpeg, ..."` |
| `422`       | `level` value is not in `{2, 3, 4}`                            | `"Level 5 is not recognised. Valid levels: [2, 3, 4]."`         |
| `503`       | ML model failed to load at startup                             | `"ML model is not available. Please try again later."`           |

##### Error Response Body (standard FastAPI format)

```json
{
  "detail": "<error message>"
}
```

---

## Decision Logic

The API applies the following decision table after inference:

```
predicted_label ≠ expected_class
  → status: manual_check_needed
  → reason: "Predicted stage 'X' does not match expected stage 'Y'."

predicted_label == expected_class AND confidence < CONFIDENCE_THRESHOLD
  → status: manual_check_needed
  → reason: "Stage 'X' matched, but confidence (X.XX) is below the required threshold (Y.YY)."

predicted_label == expected_class AND confidence ≥ CONFIDENCE_THRESHOLD
  → status: success
  → reason: "Stage 'X' confirmed."
```

---

## Configuration

The API is configured entirely through environment variables (or a `.env` file in the project root).

| Variable               | Default                                                   | Description                                             |
|------------------------|-----------------------------------------------------------|---------------------------------------------------------|
| `MODEL_PATH`           | `MMAY_Image_2-0.h5`                                       | Path to the Keras model file                            |
| `IMG_HEIGHT`           | `224`                                                     | Input image height (pixels) expected by the model       |
| `IMG_WIDTH`            | `224`                                                     | Input image width (pixels) expected by the model        |
| `CONFIDENCE_THRESHOLD` | `0.70`                                                    | Minimum confidence (0–1) to accept a prediction         |
| `LEVEL_CLASS_MAP`      | `{"2":"plinth","3":"roof_cast","4":"completion"}`         | JSON map from level integer to class label              |
| `CLASS_INDEX_MAP`      | `{"0":"completion","1":"plinth","2":"roof_cast"}` | JSON map from model output index to class label |
| `ALLOWED_IMAGE_TYPES`  | `image/jpeg,image/png,image/webp`                         | Comma-separated list of accepted MIME types             |
| `MAX_IMAGE_SIZE_MB`    | `10`                                                      | Maximum allowed upload size in megabytes                |
| `APP_HOST`             | `0.0.0.0`                                                 | Uvicorn bind address                                    |
| `APP_PORT`             | `8000`                                                    | Uvicorn listen port                                     |
| `LOG_LEVEL`            | `info`                                                    | Uvicorn/Python logging level                            |

---

## Image Pre-processing Pipeline

Before inference, every uploaded image goes through the following steps internally:

1. **Read bytes** — the full file content is read into memory.
2. **Empty check** — rejects zero-byte uploads (`400`).
3. **Size check** — rejects uploads above `MAX_IMAGE_SIZE_MB` (`413`).
4. **MIME type check** — rejects types not in `ALLOWED_IMAGE_TYPES` (`415`).
5. **Decode** — `PIL.Image.open()` decodes the bytes; corrupt data raises `400`.
6. **Convert to RGB** — strips alpha channels and palette modes.
7. **Resize** — bilinear resize to `IMG_HEIGHT × IMG_WIDTH` (default 224×224).
8. **Normalise** — pixel values divided by 255, yielding floats in `[0.0, 1.0]`.
9. **Batch dim** — array is expanded to shape `(1, H, W, 3)` for `model.predict`.

---

## Running the API

### Prerequisites

```
Python 3.10+
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start the server

```bash
python main.py
```

Or with Uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.  
Interactive Swagger UI: `http://localhost:8000/docs`

---

## Project Structure

```
MMAY/
├── main.py                          # Entry point — creates app + runs uvicorn
├── requirements.txt
├── .env                             # Environment overrides (not committed)
├── .gitignore
├── API_DOCS.md                      # This documentation file
├── MMAY.ipynb                       # Training / experimentation notebook
├── model/
│   ├── MMAY_Image_2-0.h5           # Trained Keras model (active)
│   └── construction_model.h5       # Legacy / alternate model
└── app/
    ├── application.py               # App factory, lifespan (startup/shutdown)
    ├── api/
    │   ├── dependencies.py          # Shared FastAPI Depends() providers
    │   └── routes/
    │       ├── batch_predict.py     # POST /predict/batch
    │       ├── health.py            # GET /health
    │       └── predict.py           # POST /predict
    ├── core/
    │   ├── config.py                # Settings (env-driven)
    │   ├── logging.py               # Logger factory
    │   └── model_store.py           # In-process model singleton
    ├── schemas/
    │   └── prediction.py            # Pydantic request/response models
    └── services/
        ├── image_validator.py       # Upload validation (size, MIME)
        ├── model_loader.py          # Keras model loading
        ├── predictor.py             # Inference + decision logic
        └── preprocessing.py         # Image decode, resize, normalise
```

---

## Changelog

| Version | Notes 
|---------|--------------------------------------------------------------------|
| 1.2.0   | Added `New Model "MMAY_Modelv2"`, `Accuracy 83%`  
|---------|--------------------------------------------------------------------|
| 1.1.0   | Renamed `confidence` → `predicted_confidence`, added `expected_confidence`, values now in % (0–100) |
| 1.0.0   | Initial release                |
