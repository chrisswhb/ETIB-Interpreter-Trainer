"""
ETIB Arabic I'rab Detection System — Combined Architecture
Merges:
  - Ali's acoustic detection engine (wav2vec2 ensemble + Whisper hypothesis scoring + LLR)
  - Dr's Layer 1 (LLM translation evaluation with RAG) and Layer 2 enhancements
    (CATT diacritizer, MFA forced alignment, Silero VAD delivery metrics,
     confidence-gated i'rab status)
"""

import os
import logging
from contextlib import asynccontextmanager


def _load_dotenv_file():
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key == "GROQ_API_KEY" and "COHERE_API_KEY=" in value:
                groq_value, rest = value.split("COHERE_API_KEY=", 1)
                if groq_value and not os.environ.get("GROQ_API_KEY"):
                    os.environ["GROQ_API_KEY"] = groq_value
                if rest and not os.environ.get("COHERE_API_KEY"):
                    cohere_value = rest.split("COHERE_ARABIC_ASR_MODEL=", 1)[0]
                    os.environ["COHERE_API_KEY"] = cohere_value
                if "COHERE_ARABIC_ASR_MODEL=" in rest and not os.environ.get("COHERE_ARABIC_ASR_MODEL"):
                    os.environ["COHERE_ARABIC_ASR_MODEL"] = rest.split("COHERE_ARABIC_ASR_MODEL=", 1)[1]
                continue
            if value and not os.environ.get(key):
                os.environ[key] = value


_load_dotenv_file()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from routers import analysis, diacritize, exercises, expert_pipeline, speech_diacritize, trained_analysis, translation_eval
from services.model_loader import ModelLoader
from services.trained_ending_detector import TrainedEndingDetector

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models once at startup and keep in memory."""
    logger.info("⏳ Loading wav2vec2 ensemble (3 models)...")
    ModelLoader.load_wav2vec2_ensemble()

    logger.info("⏳ Initialising Silero VAD...")
    ModelLoader.load_silero_vad()

    logger.info("⏳ Loading trained Arabic ending classifier...")
    try:
        TrainedEndingDetector.load()
    except Exception as exc:
        logger.warning("Could not load trained ending classifier at startup: %s", exc)

    logger.info("✅ All models ready.")
    yield
    logger.info("Shutting down — releasing model memory.")
    ModelLoader.unload_all()


@asynccontextmanager
async def fast_lifespan(app: FastAPI):
    """Fast local startup for the trained Arabic endpoint.

    Set PRELOAD_FULL_MODELS=1 when the older /api/analyze ensemble endpoint
    should be preloaded at startup.
    """
    if os.environ.get("PRELOAD_TRAINED_DETECTOR") == "1":
        try:
            logger.info("Loading trained Arabic ending classifier...")
            TrainedEndingDetector.load()
        except Exception as exc:
            logger.warning("Could not load trained ending classifier at startup: %s", exc)

    if os.environ.get("PRELOAD_FULL_MODELS") == "1":
        logger.info("Loading wav2vec2 ensemble (3 models)...")
        ModelLoader.load_wav2vec2_ensemble()
        logger.info("Initialising Silero VAD...")
        ModelLoader.load_silero_vad()

    logger.info("Models ready.")
    yield
    logger.info("Shutting down - releasing model memory.")
    ModelLoader.unload_all()


app = FastAPI(
    title="ETIB Arabic I'rab & Translation Feedback API",
    version="2.0.0",
    lifespan=fast_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(analysis.router,          prefix="/api")   # Layer 2 audio analysis
app.include_router(trained_analysis.router,  prefix="/api")   # Trained ending classifier demo
app.include_router(diacritize.router,        prefix="/api")   # Arabic reference diacritization
app.include_router(speech_diacritize.router, prefix="/api")   # Speech transcription + tashkeel hypothesis
app.include_router(expert_pipeline.router,   prefix="/api")   # Cohere ASR + Gemma-style tanween correction
app.include_router(exercises.router,         prefix="/api")   # Exercise CRUD
app.include_router(translation_eval.router,  prefix="/api")   # Layer 1 translation eval

# ── Static files (frontend) ───────────────────────────────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=os.path.join(FRONTEND_DIR, "static")), name="static")

# Serve index.html at root
from fastapi.responses import FileResponse

@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "templates", "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
