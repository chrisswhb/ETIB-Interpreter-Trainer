"""Collect real Cohere ASR transcripts for the Gemma correction dataset.

This is the API-backed version of the labelled dataset. It sends a limited
number of existing ASC/ETIB audio files to Cohere, stores the ASR transcript,
and pairs it with the existing gold diacritized reference.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from services.cohere_transcriber import transcribe_arabic_with_cohere  # noqa: E402
from utils.audio_utils import audio_array_to_wav_bytes, load_wav, normalize_for_asr, trim_silence  # noqa: E402
from utils.arabizi import extract_ending_from_arabic  # noqa: E402

TANWEEN_MARKER_RE = __import__("re").compile(r"([\u0600-\u06ff]+)([FNK])(?=$|[\s،,.!?؛:)\]}])")
TANWEEN_MARKERS = {"F": "\u064b", "N": "\u064c", "K": "\u064d"}


def normalize_asc_tanween_markers(text: str) -> str:
    return TANWEEN_MARKER_RE.sub(lambda m: m.group(1) + TANWEEN_MARKERS[m.group(2)], text or "")


def load_dotenv():
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        if not line or line.strip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if value and not os.environ.get(key):
            os.environ[key] = value


def final_endings(text: str) -> list[dict]:
    out = []
    for idx, raw in enumerate(text.split()):
        word = raw.strip("،,.!?؛:()[]{}\"'")
        if word:
            out.append({"word_index": idx, "word": word, "ending": extract_ending_from_arabic(word)})
    return out


def has_tanween(text: str) -> bool:
    normalized = normalize_asc_tanween_markers(text or "")
    return any(ch in normalized for ch in ("\u064b", "\u064c", "\u064d"))


def iter_unique_utterances(path: Path, require_tanween: bool = False):
    seen: OrderedDict[str, dict] = OrderedDict()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row.get("utterance_id", "")
            if not uid or uid in seen:
                continue
            if require_tanween and not has_tanween(row.get("reference_text", "")):
                continue
            seen[uid] = {
                "id": uid,
                "audio_path": row.get("audio_path", ""),
                "gold_diacritized_text": normalize_asc_tanween_markers(row.get("reference_text", "")),
                "source": row.get("source", path.stem),
            }
    return list(seen.values())


async def transcribe_file(audio_path: Path) -> dict:
    audio, sr = load_wav(str(audio_path))
    audio = normalize_for_asr(trim_silence(audio, threshold=0.008))
    wav_bytes = audio_array_to_wav_bytes(audio, sr)
    return await transcribe_arabic_with_cohere(wav_bytes)


async def main_async(args) -> int:
    load_dotenv()
    rows = iter_unique_utterances(args.manifest, require_tanween=args.require_tanween)
    rows = rows[args.skip : args.skip + args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            audio_path = ROOT / row["audio_path"]
            if not audio_path.exists():
                print(f"skip missing audio: {audio_path}")
                continue
            result = await transcribe_file(audio_path)
            transcript = result.get("text", "")
            record = {
                **row,
                "audio_path": str(audio_path),
                "cohere_transcript": transcript,
                "cohere_model": result.get("model"),
                "cohere_error": result.get("error"),
                "final_endings": final_endings(row["gold_diacritized_text"]),
                "messages": [
                    {"role": "system", "content": "You restore Arabic tashkeel and final tanween from ASR text."},
                    {
                        "role": "user",
                        "content": (
                            "Correct this Arabic ASR transcript for Modern Standard Arabic. "
                            "Return only the same sentence with full tashkeel and final tanween.\n\n"
                            f"ASR transcript:\n{transcript}"
                        ),
                    },
                    {"role": "assistant", "content": row["gold_diacritized_text"]},
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            preview = (transcript or result.get("error") or "").encode("unicode_escape").decode("ascii")
            print(f"{written}/{len(rows)} {row['id']} -> {preview[:160]}")

    print(json.dumps({"output": str(args.output), "examples": written}, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "arabic_sentence_ending_training/data/manifests/asc_wordtier_manifest.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "arabic_sentence_ending_training/data/gemma_tanween/cohere_gemma_pilot.jsonl")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--require-tanween", action="store_true")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
