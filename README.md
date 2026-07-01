# ETIB Arabic I'rab & Translation Feedback System
**Université Saint-Joseph de Beyrouth — 2026**

## Architecture overview

This project merges two complementary approaches:

### Your existing work (acoustic detection engine)
- **Signal 1** — Competing Whisper hypothesis acoustic scoring (reads `avg_logprob`, never text)
- **Signal 2** — 3× wav2vec2 CTC ensemble, ending-letter detection vote
- **Signal 3** — Log-Likelihood Ratio (LLR) logit analysis on word-boundary frames

### Dr's proposed additions (Layer 1 + Layer 2 enhancements)
- **Layer 1** — LLM translation quality evaluation (Groq LLaMA-3.3-70B) with 6-dimension rubric, RAG citations, wrong-form matcher
- **Layer 2 enhancements**:
  - Expected-ending extraction from the diacritized reference (replaces grammar corrector)
  - **Silero VAD** delivery metrics (pauses, speech rate, repetitions, filled pauses)
  - Confidence-gated i'rab status: `correct | incorrect | uncertain` — damma/tanwin_damm reported `uncertain` when confidence < 55%
  - Layer 2 JSON schema matches Dr's `layer2_delivery` specification exactly

---

## Folder structure

```
etib/
├── backend/
│   ├── main.py                        ← FastAPI entry point
│   ├── routers/
│   │   ├── analysis.py                ← POST /api/analyze (Layer 2)
│   │   ├── exercises.py               ← GET /api/exercises
│   │   └── translation_eval.py        ← POST /api/evaluate-translation (Layer 1)
│   ├── services/
│   │   ├── model_loader.py            ← wav2vec2 + Silero VAD startup loader
│   │   ├── acoustic_detector.py       ← 3-signal ensemble (your approach + Dr's confidence gating)
│   │   ├── delivery_metrics.py        ← Silero VAD delivery analysis (Dr's addition)
│   │   └── translation_evaluator.py   ← Layer 1 LLM evaluator (Dr's addition)
│   └── utils/
│       ├── arabizi.py                 ← Arabic diacritic ↔ phonetic Latin converter
│       └── audio_utils.py             ← WebM → WAV conversion (ffmpeg)
├── frontend/
│   ├── templates/index.html           ← Single-page frontend
│   └── static/
│       ├── css/style.css
│       └── js/app.js
├── data/
│   └── exercises/exercises.json       ← Your 7 i'rab exercises
├── requirements.txt
└── README.md
```

---

## Prerequisites

### 1. Python 3.10 or higher

### 2. ffmpeg (required for audio conversion)

**Windows:**
```
winget install ffmpeg
# or download from https://ffmpeg.org/download.html and add to PATH
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

### 3. wav2vec2 models (your existing setup)

The system auto-detects models at:
- `C:\Projects\model`   (m1 — jonatasgrosman/wav2vec2-large-xlsr-53-arabic)
- `C:\Projects\model2`  (m2 — elgeish/wav2vec2-large-xlsr-53-arabic)
- `C:\Projects\model3`  (m3 — kmfoda/wav2vec2-large-xlsr-arabic)

If not found locally, it falls back to downloading from HuggingFace (~1.2 GB each).

---

## Setup

```bash
# 1. Clone / copy the project
cd etib

# 2. Create virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Groq API key (needed for Whisper hypothesis scoring AND Layer 1 LLM)
# Windows:
set GROQ_API_KEY=your_key_here

# macOS/Linux:
export GROQ_API_KEY=your_key_here
```

---

## Running the server

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** in your browser.

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/exercises` | List all 7 i'rab exercises |
| GET | `/api/exercises/{id}` | Get single exercise |
| POST | `/api/analyze` | Layer 2: analyze audio recording |
| POST | `/api/evaluate-translation` | Layer 1: evaluate student translation |

### POST /api/analyze — form data

| Field | Type | Description |
|-------|------|-------------|
| `audio` | File | WebM audio from browser |
| `reference_sentence` | string | Fully diacritized Arabic sentence |
| `focus_words` | JSON string | Array of focus word strings |
| `exercise_id` | string | Optional exercise ID |
| `feedback_language` | string | `ar` (default) / `en` / `fr` |

Response: `layer2_delivery` JSON object (Dr's schema)

### POST /api/evaluate-translation — JSON body

```json
{
  "source_language": "en",
  "target_language": "ar",
  "feedback_language": "ar",
  "source_text": "...",
  "student_translation": "...",
  "reference_materials": [],
  "known_interference_matches": []
}
```

Response: `layer1_translation` JSON object (Dr's schema)

---

## How the two layers complement each other

```
Student reads Arabic sentence aloud
           │
           ▼
    ┌──────────────┐
    │   Layer 2    │  ← YOUR acoustic detection + Dr's enhancements
    │  (i'rab +    │
    │  delivery)   │
    └──────┬───────┘
           │
           │  confirmed Arabic text (from exercise)
           │
    ┌──────▼───────┐
    │   Layer 1    │  ← Dr's LLM translation evaluator
    │ (translation │
    │  quality)    │
    └──────────────┘
           │
           ▼
    Merged report: layer1_translation + layer2_delivery
```

### Why accuracy improves

| Problem | Your solution | Dr's addition |
|---------|--------------|---------------|
| Damma/tanwin_damm hard to detect | LLR + hypothesis scoring | Confidence gating → uncertain (not wrong) |
| Arbitrary confidence cutoffs | Ensemble vote weights | Per-haraka threshold (damma=55%, others=45%) |
| No expected-ending ground truth | Reference Arabizi comparison | Direct diacritic extraction from reference |
| Delivery metrics missing | Silences + repetitions count | Full Silero VAD: pauses, speech rate, false starts |
| No translation evaluation | Not in scope | Full LLM rubric Layer 1 with citations |

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes (for full function) | Groq API key for Whisper + LLaMA |

Without `GROQ_API_KEY`, the system falls back to wav2vec2 + LLR only (no Whisper hypothesis scoring, no Layer 1 evaluation).

---

## Known limitations (from your technical report)

- Damma (u) and tanwin UN: 20–30% accuracy — reported as `uncertain` per Dr's guidance
- Background noise degrades all signals — close microphone recommended
- Word-level forced alignment (MFA) is optional — audio_span will be `null` without it
- Groq API adds ~3s latency per focus word for hypothesis scoring
