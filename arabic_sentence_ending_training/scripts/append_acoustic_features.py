from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from audio_io import TARGET_SR, load_audio, resolve_path


ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = ROOT.parent
sys.path.insert(0, str(APP_ROOT / "backend"))

from services.ending_acoustic_features import ending_acoustic_features


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tail-ms", type=int, default=650)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    features_path = ROOT / args.features
    meta_path = ROOT / args.meta
    out_path = ROOT / args.out
    data = np.load(features_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    groups = data["groups"]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if len(meta) != len(X):
        raise SystemExit(f"meta/features length mismatch: {len(meta)} != {len(X)}")
    if args.max_samples and args.max_samples > 0:
        n = min(args.max_samples, len(X))
        X = X[:n]
        y = y[:n]
        groups = groups[:n]
        meta = meta[:n]

    audio_cache: dict[str, tuple[np.ndarray, int]] = {}
    by_audio = defaultdict(list)
    for idx, row in enumerate(meta):
        by_audio[row["audio_path"]].append((idx, row))

    acoustic = np.zeros((len(X), 92), dtype="float32")
    for audio_i, (audio_path_text, items) in enumerate(by_audio.items(), 1):
        audio_path = resolve_path(APP_ROOT, audio_path_text)
        if not audio_path.exists():
            print(f"missing audio: {audio_path}")
            continue
        audio, sr = audio_cache.get(str(audio_path), (None, None))
        if audio is None:
            audio, sr = load_audio(audio_path, TARGET_SR)
            audio_cache[str(audio_path)] = (audio, sr)
        for idx, row in items:
            try:
                start = max(0, int(float(row["start_ms"]) / 1000.0 * sr))
                end = min(len(audio), int(float(row["end_ms"]) / 1000.0 * sr))
            except Exception:
                continue
            if end <= start:
                continue
            tail_samples = int(args.tail_ms / 1000.0 * sr)
            segment = audio[max(start, end - tail_samples) : end]
            acoustic[idx] = ending_acoustic_features(segment, sr)
        if audio_i % 25 == 0:
            print(f"processed audio files: {audio_i}/{len(by_audio)}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=np.concatenate([X, acoustic], axis=1), y=y, groups=groups)
    print(f"old_dim: {X.shape[1]}")
    print(f"new_dim: {X.shape[1] + acoustic.shape[1]}")
    print(f"samples: {len(X)}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
