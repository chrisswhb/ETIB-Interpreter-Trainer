from __future__ import annotations

import argparse
import csv
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

MODEL_DIR = Path.home() / "Desktop" / "etib_models" / "wav2vec2_jonatasgrosman_arabic"
TRAINABLE_LABELS = {
    "fatha",
    "damma",
    "kasra",
    "none",
    "tanwin_fath",
    "tanwin_damm",
    "tanwin_kasr",
}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/manual_sentence_manifest.csv")
    parser.add_argument("--out", default="data/features/sentence_word_features.npz")
    parser.add_argument("--meta", default="data/features/sentence_word_features_meta.json")
    parser.add_argument("--tail-ms", type=int, default=650)
    parser.add_argument(
        "--whole-word",
        action="store_true",
        help="Use the full aligned word span instead of only the final tail window.",
    )
    parser.add_argument(
        "--augment-acoustic",
        action="store_true",
        help="Append 5ms high-resolution ending energy/spectral/pitch cues.",
    )
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    out_path = ROOT / args.out
    meta_path = ROOT / args.meta

    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")
    if not MODEL_DIR.exists():
        raise SystemExit(f"wav2vec2 model folder not found: {MODEL_DIR}")

    import torch
    from transformers import Wav2Vec2Model, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(str(MODEL_DIR), local_files_only=True)
    model = Wav2Vec2Model.from_pretrained(str(MODEL_DIR), local_files_only=True)
    model.eval()

    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8-sig")))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["audio_path"]].append(row)

    vectors: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    meta: list[dict[str, str]] = []

    for audio_path_text, audio_rows in grouped.items():
        audio_path = resolve_path(APP_ROOT, audio_path_text)
        if not audio_path.exists():
            print(f"missing audio: {audio_path}")
            continue
        audio, sr = load_audio(audio_path, TARGET_SR)

        for row in audio_rows:
            label = row["expected_ending"]
            if label not in TRAINABLE_LABELS:
                continue
            if not row.get("start_ms") or not row.get("end_ms"):
                print(f"skip unaligned word: {row['utterance_id']} #{row['word_index']} {row['word']}")
                continue

            start = max(0, int(float(row["start_ms"]) / 1000.0 * sr))
            end = min(len(audio), int(float(row["end_ms"]) / 1000.0 * sr))
            if end <= start:
                continue

            tail_samples = int(args.tail_ms / 1000.0 * sr)
            segment_start = start if args.whole_word else max(start, end - tail_samples)
            segment = audio[segment_start:end]
            if len(segment) < int(0.08 * sr):
                continue

            inputs = processor(segment, sampling_rate=sr, return_tensors="pt", padding=True)
            with torch.no_grad():
                hidden = model(**inputs).last_hidden_state[0].cpu().numpy()

            vector = _multi_tail_vector(hidden, len(segment) / sr)
            if args.augment_acoustic:
                vector = np.concatenate(
                    [vector, ending_acoustic_features(segment, sr)]
                ).astype("float32")
            vectors.append(vector)
            labels.append(label)
            groups.append(row["utterance_id"])
            meta.append(row)
            print(f"ok {row['utterance_id']}:{row['word_index']} {row['word']} -> {label}")

    if not vectors:
        raise SystemExit("no aligned trainable word segments found")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=np.stack(vectors),
        y=np.array(labels),
        groups=np.array(groups),
    )
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out_path}")
    print(f"saved: {meta_path}")


def _multi_tail_vector(hidden: np.ndarray, duration_s: float) -> np.ndarray:
    pieces = []
    for frac in (0.15, 0.25, 0.40, 0.65, 1.0):
        start = max(0, int(len(hidden) * (1.0 - frac)))
        window = hidden[start:] if start < len(hidden) else hidden
        pieces.extend([window.mean(axis=0), window.std(axis=0)])
    duration_feature = np.array([duration_s], dtype="float32")
    return np.concatenate([*pieces, duration_feature]).astype("float32")


if __name__ == "__main__":
    main()
