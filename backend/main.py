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
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv_file()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from routers import analysis, diacritize, exercises, translation_eval
from services.model_loader import ModelLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models once at startup and keep in memory."""
    logger.info("⏳ Loading wav2vec2 ensemble (3 models)...")
    ModelLoader.load_wav2vec2_ensemble()

    logger.info("⏳ Initialising Silero VAD...")
    ModelLoader.load_silero_vad()

    logger.info("✅ All models ready.")
    yield
    logger.info("Shutting down — releasing model memory.")
    ModelLoader.unload_all()


app = FastAPI(
    title="ETIB Arabic I'rab & Translation Feedback API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(analysis.router,          prefix="/api")   # Layer 2 audio analysis
app.include_router(diacritize.router,        prefix="/api")   # Arabic reference diacritization
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
