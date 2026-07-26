from __future__ import annotations

import numpy as np


def ending_acoustic_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """Compact high-resolution acoustic cues for final Arabic endings.

    These features are intentionally classical and cheap: they focus on the
    final frames where short vowels/tanween live, complementing wav2vec2 hidden
    states with explicit energy, spectral, and pitch-like measurements.
    """
    if audio is None or len(audio) == 0 or sr <= 0:
        return np.zeros(92, dtype="float32")

    x = np.asarray(audio, dtype="float32")
    x = x - float(np.mean(x))
    peak = float(np.max(np.abs(x)))
    if peak > 1e-6:
        x = x / peak

    frame = max(32, int(sr * 0.025))
    hop = max(8, int(sr * 0.005))
    frames = _frames(x, frame, hop)
    if len(frames) == 0:
        frames = x.reshape(1, -1)

    window = np.hanning(frames.shape[1]).astype("float32")
    win_frames = frames * window
    energy = np.sqrt(np.mean(win_frames * win_frames, axis=1) + 1e-9)
    zcr = np.mean(np.abs(np.diff(np.signbit(win_frames), axis=1)), axis=1)

    spec = np.abs(np.fft.rfft(win_frames, axis=1)) + 1e-8
    freqs = np.fft.rfftfreq(win_frames.shape[1], 1.0 / sr).astype("float32")
    spec_sum = spec.sum(axis=1) + 1e-8
    centroid = (spec * freqs).sum(axis=1) / spec_sum
    bandwidth = np.sqrt((spec * (freqs[None, :] - centroid[:, None]) ** 2).sum(axis=1) / spec_sum)

    low_band = _band_energy(spec, freqs, 250, 900)
    mid_band = _band_energy(spec, freqs, 900, 2200)
    high_band = _band_energy(spec, freqs, 2200, 3800)
    pitch = _pitch_autocorr(frames, sr)

    coeffs = _dct_log_spectrum(spec, n_coeffs=8)
    series = [energy, zcr, centroid / max(1.0, sr / 2), bandwidth / max(1.0, sr / 2), low_band, mid_band, high_band, pitch]
    out: list[np.ndarray] = []
    for values in series:
        out.append(_series_stats(np.asarray(values, dtype="float32")))

    for i in range(coeffs.shape[1]):
        out.append(_series_stats(coeffs[:, i]))

    duration = np.array([len(x) / sr], dtype="float32")
    features = np.concatenate([*out, duration]).astype("float32")
    if len(features) != 92:
        fixed = np.zeros(92, dtype="float32")
        fixed[: min(len(features), len(fixed))] = features[: len(fixed)]
        return fixed
    return features


def _frames(audio: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(audio) < frame:
        padded = np.zeros(frame, dtype="float32")
        padded[: len(audio)] = audio
        return padded.reshape(1, -1)
    starts = range(0, len(audio) - frame + 1, hop)
    return np.stack([audio[start : start + frame] for start in starts]).astype("float32")


def _band_energy(spec: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any():
        return np.zeros(spec.shape[0], dtype="float32")
    total = spec.sum(axis=1) + 1e-8
    return (spec[:, mask].sum(axis=1) / total).astype("float32")


def _pitch_autocorr(frames: np.ndarray, sr: int) -> np.ndarray:
    min_lag = max(1, int(sr / 400))
    max_lag = max(min_lag + 1, int(sr / 70))
    values = []
    for frame in frames:
        centered = frame - float(np.mean(frame))
        denom = float(np.dot(centered, centered)) + 1e-8
        best = 0.0
        for lag in range(min_lag, min(max_lag, len(centered) - 1)):
            score = float(np.dot(centered[:-lag], centered[lag:]) / denom)
            if score > best:
                best = score
        values.append(best)
    return np.asarray(values, dtype="float32")


def _dct_log_spectrum(spec: np.ndarray, n_coeffs: int) -> np.ndarray:
    log_spec = np.log(spec)
    n_bins = log_spec.shape[1]
    basis = np.cos(
        np.pi / n_bins * (np.arange(n_bins, dtype="float32") + 0.5)[:, None] * np.arange(n_coeffs, dtype="float32")[None, :]
    )
    return (log_spec @ basis / max(1, n_bins)).astype("float32")


def _series_stats(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(6, dtype="float32")
    values = values.astype("float32")
    delta = np.diff(values) if len(values) > 1 else np.array([0.0], dtype="float32")
    return np.array(
        [
            float(np.mean(values)),
            float(np.std(values)),
            float(np.min(values)),
            float(np.max(values)),
            float(values[-1] - values[0]),
            float(np.mean(delta)),
        ],
        dtype="float32",
    )
