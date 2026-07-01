# ETIB Arabic Pronunciation / Tashkeel Detection Handoff

Date: 2026-06-29
Project root: `C:\Users\Ali\Downloads\etib_project\etib`

## Scope

Ali is responsible for the Arabic part of the ETIB interpreter self-training tool:

- detect Arabic final tashkeel / i'rab errors
- detect tanween errors
- give pronunciation/delivery feedback for Arabic recordings
- support arbitrary Arabic reference text, not only fixed exercises

This is not the full cahier des charges platform. It is the Arabic evaluation module.

## Current App

FastAPI backend + single-page HTML/JS frontend.

Run URL:

```text
http://127.0.0.1:8000
```

Important paths:

- `backend/main.py`: FastAPI app, model startup, `.env` loader
- `backend/routers/analysis.py`: `POST /api/analyze`, main audio analysis flow
- `backend/routers/diacritize.py`: `POST /api/diacritize` and `GET /api/groq-status`
- `backend/services/model_loader.py`: persistent local model loading
- `backend/services/acoustic_detector.py`: Arabic ending/tanween detection logic
- `backend/services/diacritizer.py`: Groq-based Arabic reference diacritization
- `backend/services/delivery_metrics.py`: pause/speech-rate/repetition metrics
- `backend/utils/audio_utils.py`: WebM -> WAV conversion using bundled `imageio-ffmpeg`
- `frontend/templates/index.html`: UI
- `frontend/static/js/app.js`: recorder, custom text, results rendering
- `data/exercises/exercises.json`: 7 demo exercises
- `scripts/download_models.py`: one-time model downloader

## Local Models

Models were downloaded once to:

```text
C:\Users\Ali\Desktop\etib_models
```

The loader checks this folder first:

- `wav2vec2_jonatasgrosman_arabic`
- `wav2vec2_elgeish_arabic`
- `wav2vec2_kmfoda_arabic`
- `silero-vad`

The app should no longer download these models on every run.

## Current Pipeline

1. Browser records audio as WebM/Opus.
2. Backend converts audio to 16 kHz mono WAV.
3. Three wav2vec2 Arabic CTC models transcribe the recording.
4. For each Arabic reference word with explicit final haraka/tanween:
   - extract expected ending directly from the diacritized reference
   - match it to the best transcript word using fuzzy Arabic normalization
   - infer ending from transcript spelling
   - compute word-aware LLR from wav2vec2 logits
   - optionally use Groq Whisper hypothesis scoring if `GROQ_API_KEY` is set
5. Combine signals into:
   - `correct`
   - `incorrect`
   - `uncertain`
6. Return JSON with findings, delivery metrics, transcript, and debug info.

## Signals Used

### Signal 1: Groq Whisper Hypothesis Scoring

Status: implemented but disabled unless `GROQ_API_KEY` exists.

For each target word, the app builds competing hypotheses:

- fatha
- damma
- kasra

or for tanween:

- tanwin_fath
- tanwin_damm
- tanwin_kasr

It sends WAV audio to Groq Whisper and reads `avg_logprob`, not Whisper's text output.

This is expected to be the strongest signal, but current logs showed:

```text
GROQ_API_KEY not set - Whisper hypothesis scoring disabled
```

To enable:

Create `.env` in project root:

```text
GROQ_API_KEY=your_key_here
```

Then restart the server.

The same key also enables `POST /api/diacritize`, used by the frontend's "شكّل النص تلقائيًا" button to create a diacritized reference from unvocalized Arabic text.

### Signal 2: wav2vec2 Ensemble Spelling

Three models transcribe the audio. The code detects written endings:

- final `ا`, `ى`, `ة` -> fatha
- final `و` -> damma
- final `ي` -> kasra
- final `ان` -> tanwin_fath
- final `ون` -> tanwin_damm
- final `ين` -> tanwin_kasr

It also does expected-aware tanween interpretation:

- reference expects `طَالِبًا`, ASR writes `طالبا` -> tanwin_fath
- reference expects `طَالِبٍ`, ASR writes `طالبين` -> tanwin_kasr
- reference expects `طَالِبٌ`, ASR writes `طالبون` -> tanwin_damm
- reference expects tanween, ASR writes bare `طالب` -> missing tanween error

### Signal 3: Word-Aware LLR Logit Analysis

Originally, LLR checked only the last frames of the whole recording. This was wrong for middle words.

Now it estimates each word's frame region from the reference word index and computes LLR on that word segment.

It averages LLR scores across all available wav2vec2 models.

### Implicit Short Vowel Fallback

Arabic ASR often writes:

- `ذَهَبْتُ` as `ذهبت`
- `مَكْتَبِ` as `مكتب`

That does not mean the student omitted the final vowel; ASR often drops short vowels.

If the transcript word matches the same Arabic base word and has no explicit wrong long-vowel ending, the app treats the expected short vowel as detected with capped confidence.

Example:

- reference `مَكْتَبِ`
- ASR `مكتب`
- expected `kasra`
- result: `correct`, confidence up to about `84%` if all three wav2vec2 models agree

## Frontend Features

The UI now has:

- ready-made 7 exercises
- custom Arabic text box for arbitrary diacritized reference sentences
- automatic "شكّل النص تلقائيًا" button using Groq LLaMA when `GROQ_API_KEY` is configured
- Groq status indicator on the main page
- recorder
- analysis results cards
- delivery metrics
- debug JSON

Custom text note:

The custom text should be fully diacritized, especially final endings. If text is not diacritized, the system cannot know the correct expected ending unless a diacritizer/LLM step is added.

## Accuracy Reality

For general Arabic text, stable 80%+ is hard without specialized training because:

- most Arabic ASR models do not output harakat
- final short vowels are acoustically short
- tanween is often dropped or normalized
- there is no true forced alignment yet

Expected practical accuracy:

- Local wav2vec2-only: roughly 55-70%, sometimes higher on clean controlled examples
- With Groq/Whisper hypothesis scoring enabled: 70-85% is realistic for controlled, diacritized reference text
- General open text with no diacritized reference: not realistically solvable without diacritizer plus more model support
- 85%+ robustly: likely needs fine-tuning or a specialized Arabic pronunciation/diacritized speech dataset

Recommended project claim:

```text
The system targets approximately 80% accuracy on controlled, diacritized Arabic reference texts using reference-guided acoustic hypothesis scoring, wav2vec2 ensemble evidence, word-aware LLR, and confidence gating. Low-confidence damma/tanween cases are marked uncertain rather than forced.
```

## Remaining Best Improvements

1. Enable `GROQ_API_KEY`.
   This is the biggest practical accuracy gain.

2. Add a diacritization step for unvocalized arbitrary text.
   Possible methods:
   - Groq LLM diacritizer prompt
   - CAMeL Tools / Farasa / Mishkal if installable
   - require the user to paste fully diacritized text

3. Add real word-level forced alignment.
   Current word frames are estimated. MFA or another aligner would improve LLR.

4. Build an evaluation set.
   Record 5-10 attempts per exercise with known correct/wrong endings and measure real accuracy.

5. Keep uncertainty honest.
   Do not claim 100% because the model cannot always acoustically prove short final vowels.

## Current Server State

A server was started on port 8000 during the previous session. If needed, restart:

```powershell
cd C:\Users\Ali\Downloads\etib_project\etib\backend
..\venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

## Important Warning

If `GROQ_API_KEY` is missing, the app is running without the primary acoustic hypothesis scorer. In that mode, accuracy is limited by wav2vec2 transcript spelling and heuristic fallbacks.
