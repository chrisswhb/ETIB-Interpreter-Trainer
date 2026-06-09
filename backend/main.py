"""
main.py — ETIB Arabic I'rab Tester
Receives audio → Whisper transcription → Arabizi comparison → feedback
"""

import os, json, tempfile, traceback
from pathlib import Path
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from groq import Groq
from analyzer import analyze, build_payload

# ── Config ────────────────────────────────────────────────────────────────────
_here = Path(__file__).parent
load_dotenv(_here / ".env")
load_dotenv(_here.parent / ".env")

GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
EXERCISES_PATH = _here.parent / "exercises" / "exercises.json"
FRONTEND_PATH  = _here.parent / "frontend"

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="ETIB Arabic Tester")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(FRONTEND_PATH)), name="static")

@app.get("/")
def index():
    return FileResponse(str(FRONTEND_PATH / "index.html"))

@app.get("/health")
def health():
    return {"status": "ok", "groq_key_set": bool(GROQ_API_KEY)}


# ── Exercises ─────────────────────────────────────────────────────────────────
def load_exercises():
    with open(EXERCISES_PATH, encoding="utf-8") as f:
        return json.load(f)

@app.get("/exercises")
def get_exercises():
    return load_exercises()


# ── Transcription ─────────────────────────────────────────────────────────────
def transcribe(audio_bytes: bytes, reference_text: str = "") -> dict:
    """
    Send audio to Groq Whisper with the DIACRITIZED reference as the prompt.
    This forces Whisper to output diacritics, which we then use for Arabizi
    comparison. Without this, Whisper strips all diacritics.
    Returns: { text, segments, words }
    """
    if not GROQ_API_KEY:
        raise HTTPException(500, "GROQ_API_KEY not set in .env")

    client = Groq(api_key=GROQ_API_KEY)

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            kwargs = dict(
                file=(tmp_path, f, "audio/webm"),
                model="whisper-large-v3",
                language="ar",
                response_format="verbose_json",
                temperature=0.0,
            )
            # KEY: pass the DIACRITIZED reference as prompt
            # Whisper uses this as a style/vocabulary hint and tends to
            # mirror the diacritization in its output
            if reference_text:
                kwargs["prompt"] = reference_text[:224]
            response = client.audio.transcriptions.create(**kwargs)
    finally:
        os.unlink(tmp_path)

    def seg(s):
        if isinstance(s, dict): return s
        return {k: getattr(s, k, None) for k in
                ["text", "start", "end", "avg_logprob", "no_speech_prob"]}

    def word(w):
        if isinstance(w, dict): return w
        return {k: getattr(w, k, None) for k in ["word", "start", "end"]}

    return {
        "text":     response.text,
        "segments": [seg(s) for s in (getattr(response, "segments", None) or [])],
        "words":    [word(w) for w in (getattr(response, "words", None) or [])],
    }


def get_word_confidences(segments: list, num_words: int) -> dict:
    """Map word index → confidence from segment avg_logprob."""
    import math
    if not segments:
        return {}
    logprob = segments[0].get("avg_logprob")
    if logprob is None:
        return {}
    conf = round(max(0.0, min(1.0, math.exp(logprob))), 3)
    return {i: conf for i in range(num_words)}


# ── Main analyze endpoint ─────────────────────────────────────────────────────
@app.post("/analyze")
async def analyze_endpoint(
    audio: UploadFile = File(...),
    exercise_id: str = Form(...),
):
    try:
        return await _analyze(audio, exercise_id)
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print("ERROR:\n", tb)
        raise HTTPException(500, detail=f"{type(e).__name__}: {e}\n\n{tb}")


async def _analyze(audio: UploadFile, exercise_id: str):
    # Load exercise
    exercise = next((e for e in load_exercises() if e["id"] == exercise_id), None)
    if not exercise:
        raise HTTPException(404, f"Exercise {exercise_id} not found")

    # Read audio
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio")

    # Transcribe
    tr = transcribe(audio_bytes, exercise["reference"])
    raw_text = tr["text"].strip()

    # Word confidences
    num_words = len(raw_text.split())
    word_confs = get_word_confidences(tr["segments"], num_words)

    # Analyze
    results = analyze(
        reference_diacritized=exercise["reference"],
        stt_output=raw_text,
        focus_words=exercise.get("focus", []),
        word_confidences=word_confs,
    )
    payload = build_payload(results)

    return {
        "exercise_id":  exercise_id,
        "transcript":   raw_text,
        "reference":    exercise["reference"],
        "feedback":     payload,
        "tip":          exercise.get("tip", ""),
        "debug": {
            "segment_logprob": tr["segments"][0].get("avg_logprob") if tr["segments"] else None,
            "num_words_transcribed": num_words,
        }
    }