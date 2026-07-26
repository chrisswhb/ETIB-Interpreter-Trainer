"""
audio_utils.py — Audio preprocessing helpers.

Converts browser WebM/Opus recordings to 16kHz mono WAV
for wav2vec2 and Silero VAD consumption.
"""

import os
import subprocess
import tempfile
import shutil
import logging
import numpy as np
from io import BytesIO
from pathlib import Path

logger = logging.getLogger(__name__)

TARGET_SR = 16000  # wav2vec2 requires 16kHz


def _get_ffmpeg_executable() -> str:
    """Return system ffmpeg if installed, otherwise the bundled imageio-ffmpeg binary."""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError(
            "ffmpeg is not installed and imageio-ffmpeg fallback is unavailable. "
            "Install ffmpeg or run: pip install imageio-ffmpeg"
        ) from e


def webm_to_wav(webm_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Convert raw WebM/Opus bytes (from browser MediaRecorder) to
    a 16kHz mono float32 numpy array.

    Returns (audio_array, sample_rate)
    """
    if not webm_bytes or len(webm_bytes) < 512:
        raise RuntimeError("empty or too-short audio recording")

    with tempfile.TemporaryDirectory() as tmp:
        in_path  = os.path.join(tmp, "input.recording")
        out_path = os.path.join(tmp, "output.wav")

        with open(in_path, "wb") as f:
            f.write(webm_bytes)

        cmd = [
            _get_ffmpeg_executable(), "-y",
            "-i", in_path,
            "-ar", str(TARGET_SR),
            "-ac", "1",
            "-f", "wav",
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace").strip()
            if len(stderr) > 1200:
                stderr = stderr[-1200:]
            raise RuntimeError(
                f"ffmpeg conversion failed: {stderr}"
            )

        return load_wav(out_path)


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Load a WAV file into a float32 numpy array."""
    try:
        import soundfile as sf
        audio, sr = sf.read(path, dtype="float32")
    except ImportError:
        import scipy.io.wavfile as wav
        sr, audio = wav.read(path)
        audio = audio.astype(np.float32)
        if audio.max() > 1.0:
            audio /= 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def audio_array_to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    """Encode a mono float32 audio array as WAV bytes."""
    try:
        import soundfile as sf

        buf = BytesIO()
        sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()
    except ImportError:
        import scipy.io.wavfile as wav

        clipped = np.clip(audio, -1.0, 1.0)
        pcm = (clipped * 32767).astype(np.int16)
        buf = BytesIO()
        wav.write(buf, sr, pcm)
        return buf.getvalue()


def resample_if_needed(audio: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 16kHz if necessary."""
    if sr == TARGET_SR:
        return audio
    try:
        import librosa
        return librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    except ImportError:
        logger.warning("librosa not installed — cannot resample. Ensure audio is 16kHz.")
        return audio


def trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Simple energy-based silence trimming from both ends."""
    energy = np.abs(audio)
    mask = energy > threshold
    if not mask.any():
        return audio
    start = np.argmax(mask)
    end   = len(mask) - np.argmax(mask[::-1])
    return audio[start:end]


def normalize_for_asr(audio: np.ndarray, peak: float = 0.92) -> np.ndarray:
    """
    Remove DC offset and normalize peak level before ASR/acoustic scoring.
    This makes quiet recordings less fragile while avoiding clipping.
    """
    if audio is None or len(audio) == 0:
        return audio
    normalized = audio.astype(np.float32, copy=True)
    normalized = normalized - float(np.mean(normalized))
    max_abs = float(np.max(np.abs(normalized)))
    if max_abs > 1e-6:
        normalized = normalized / max_abs * peak
    return np.clip(normalized, -1.0, 1.0)
