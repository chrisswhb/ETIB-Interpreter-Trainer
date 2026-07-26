from __future__ import annotations

import shutil
import subprocess
import tempfile
from math import gcd
from pathlib import Path

import numpy as np


TARGET_SR = 16000


def ffmpeg_exe() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("Need ffmpeg or imageio-ffmpeg to read non-WAV audio.") from exc


def convert_to_wav(path: Path, target_sr: int = TARGET_SR) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    out_path = Path(tmp.name)
    cmd = [
        ffmpeg_exe(),
        "-y",
        "-i",
        str(path),
        "-ar",
        str(target_sr),
        "-ac",
        "1",
        "-f",
        "wav",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="replace"))
    return out_path


def load_audio(path: Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    import soundfile as sf
    from scipy.signal import resample_poly

    converted_path: Path | None = None
    try:
        try:
            audio, sr = sf.read(path, dtype="float32")
        except Exception:
            converted_path = convert_to_wav(path, target_sr)
            audio, sr = sf.read(converted_path, dtype="float32")
    finally:
        if converted_path and converted_path.exists():
            converted_path.unlink(missing_ok=True)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        factor = gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // factor, sr // factor).astype("float32")
        sr = target_sr
    return normalize_audio(audio), sr


def normalize_audio(audio: np.ndarray, peak: float = 0.92) -> np.ndarray:
    if len(audio) == 0:
        return audio.astype("float32")
    normalized = audio.astype("float32", copy=True)
    normalized -= float(np.mean(normalized))
    max_abs = float(np.max(np.abs(normalized)))
    if max_abs > 1e-6:
        normalized = normalized / max_abs * peak
    return np.clip(normalized, -1.0, 1.0)


def resolve_path(workspace_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return workspace_root / path

