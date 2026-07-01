"""
Model Loader — loads all heavy models once at startup.

wav2vec2 ensemble  : 3 × ~1.2 GB Arabic XLSR models (your existing approach)
Silero VAD         : lightweight torch model for delivery metrics (Dr's Layer 2)
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_ROOT = Path(
    os.environ.get(
        "ETIB_MODEL_DIR",
        Path.home() / "Desktop" / "etib_models",
    )
)

# ── Model path resolution (matches your existing C:\Projects\model* paths on Windows) ──
def _find_model_path(candidates: list[str]) -> Optional[str]:
    for p in candidates:
        if os.path.isdir(p):
            return p
    return None


WAV2VEC2_CANDIDATES = [
    [
        str(MODEL_ROOT / "wav2vec2_jonatasgrosman_arabic"),
        r"C:\Projects\model",
        str(Path.home() / "Desktop" / "model"),
    ],
    [
        str(MODEL_ROOT / "wav2vec2_elgeish_arabic"),
        r"C:\Projects\model2",
        str(Path.home() / "Desktop" / "model2"),
    ],
    [
        str(MODEL_ROOT / "wav2vec2_kmfoda_arabic"),
        r"C:\Projects\model3",
        str(Path.home() / "Desktop" / "model3"),
    ],
]

# HuggingFace fallback IDs if local paths not found
WAV2VEC2_HF_IDS = [
    "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "elgeish/wav2vec2-large-xlsr-53-arabic",
    "kmfoda/wav2vec2-large-xlsr-arabic",
]


class ModelLoader:
    _wav2vec2_models: list = []
    _wav2vec2_processors: list = []
    _silero_vad = None
    _silero_utils = None

    @classmethod
    def load_wav2vec2_ensemble(cls):
        """Load 3 wav2vec2 models. Tries local paths first, then HuggingFace."""
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            import torch

            cls._wav2vec2_models = []
            cls._wav2vec2_processors = []

            for i, (candidates, hf_id) in enumerate(zip(WAV2VEC2_CANDIDATES, WAV2VEC2_HF_IDS)):
                local_path = _find_model_path(candidates)
                source = local_path if local_path else hf_id
                logger.info(f"  Loading m{i+1} from: {source}")

                try:
                    kwargs = {"local_files_only": True} if local_path else {}
                    processor = Wav2Vec2Processor.from_pretrained(source, **kwargs)
                    model_kwargs = dict(kwargs)
                    if local_path and not (
                        Path(local_path, "pytorch_model.bin").exists()
                        or Path(local_path, "model.safetensors").exists()
                    ):
                        model_kwargs["from_flax"] = Path(local_path, "flax_model.msgpack").exists()
                    model = Wav2Vec2ForCTC.from_pretrained(source, **model_kwargs)
                    model.eval()
                    cls._wav2vec2_models.append(model)
                    cls._wav2vec2_processors.append(processor)
                    logger.info(f"  ✅ m{i+1} loaded")
                except Exception as e:
                    logger.warning(f"  ⚠️  Could not load m{i+1} ({source}): {e}")
                    cls._wav2vec2_models.append(None)
                    cls._wav2vec2_processors.append(None)

        except ImportError as e:
            logger.warning(f"transformers not installed — wav2vec2 disabled: {e}")

    @classmethod
    def load_silero_vad(cls):
        """Load Silero VAD for delivery metrics (Dr's Layer 2 addition)."""
        try:
            import torch
            silero_dir = MODEL_ROOT / "silero-vad"
            model, utils = torch.hub.load(
                repo_or_dir=str(silero_dir) if silero_dir.exists() else "snakers4/silero-vad",
                model="silero_vad",
                source="local" if silero_dir.exists() else "github",
                force_reload=False,
                trust_repo=True,
            )
            cls._silero_vad = model
            cls._silero_utils = utils
            logger.info("  ✅ Silero VAD loaded")
        except Exception as e:
            logger.warning(f"  ⚠️  Silero VAD not available: {e}")

    @classmethod
    def get_wav2vec2_ensemble(cls):
        return list(zip(cls._wav2vec2_models, cls._wav2vec2_processors))

    @classmethod
    def get_silero_vad(cls):
        return cls._silero_vad, cls._silero_utils

    @classmethod
    def unload_all(cls):
        cls._wav2vec2_models.clear()
        cls._wav2vec2_processors.clear()
        cls._silero_vad = None
        cls._silero_utils = None
