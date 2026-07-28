# Arabic API Wrapper Contract

## What the wrapper is

The wrapper is a small HTTP API layer around Ali's Arabic FastAPI module. Joe's Flask app should call this API with an audio file and optional Arabic reference text, then receive a JSON result.

This keeps both systems independent:

- Joe keeps the existing Flask backend for the deployed platform.
- Ali keeps the Arabic FastAPI service with its own dependencies and heavy speech models.
- The two apps communicate over HTTP, so Flask never imports FastAPI code or loads Arabic models.

## Endpoints

Health check:

```text
GET http://<arabic-service-url>/api/arabic-wrapper/health
```

Evaluation:

```text
POST http://<arabic-service-url>/api/arabic-wrapper/evaluate
Content-Type: multipart/form-data
```

Form fields:

- `audio`: required audio file. Browser `webm` works, and `wav`, `m4a`, or `mp3` work if ffmpeg can decode them.
- `reference_text`: optional for transcript-only mode, required for full pronunciation checking.
- `mode`: `auto`, `light`, or `full`. Default is `auto`.
- `exercise_id`: optional identifier from the main app.
- `word_spans_json`: optional forced-alignment spans if another service already has word timestamps.

Modes:

- `light`: returns Arabic ASR transcript and diacritized/tanween text hypothesis only. This is lighter for deployment.
- `full`: returns transcript plus final-haraka pronunciation findings. This uses the Arabic acoustic model and needs the heavier Arabic service.
- `auto`: uses `full` when `reference_text` is present, otherwise `light`.

## Optional security

If `ARABIC_WRAPPER_API_KEY` is set in the Arabic service environment, Joe must send:

```text
X-API-Key: <same key>
```

If `ARABIC_WRAPPER_API_KEY` is not set, the wrapper is open.

## Flask example

```python
import requests

ARABIC_API_URL = "http://127.0.0.1:8000"

def call_arabic_wrapper(audio_bytes, reference_text, exercise_id=None):
    files = {
        "audio": ("student.webm", audio_bytes, "audio/webm"),
    }
    data = {
        "reference_text": reference_text,
        "mode": "auto",
        "exercise_id": exercise_id or "",
    }
    response = requests.post(
        f"{ARABIC_API_URL}/api/arabic-wrapper/evaluate",
        files=files,
        data=data,
        timeout=180,
    )
    response.raise_for_status()
    return response.json()
```

## Response shape

The response contains:

- `transcript.raw`: what the ASR heard.
- `transcript.diacritized`: corrected/diacritized text hypothesis.
- `pronunciation.available`: whether acoustic pronunciation checking ran.
- `pronunciation.summary`: counts for correct, incorrect, uncertain, and accuracy.
- `pronunciation.findings`: word-level final-haraka/tanween results.
- `delivery`: speech-rate and pause metrics when available.
- `model_provenance`: which models/signals were used.

This API is the integration boundary. The deployed platform only needs the URL and request/response contract.
