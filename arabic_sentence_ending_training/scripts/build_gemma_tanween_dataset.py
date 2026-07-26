"""Build text-labelled data for the ASR -> Gemma tanween correction step.

The expert proposal needs supervised examples shaped like:

    ASR-like transcript without tashkeel -> corrected Arabic with tashkeel/tanween

This script creates those examples from already-labelled ETIB/ASC references.
By default it simulates ASR input by stripping diacritics from the gold text.
That is useful for the first QLoRA experiment. A later script can replace
the simulated input with real Cohere transcripts from audio.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from utils.arabizi import extract_ending_from_arabic, strip_diacritics  # noqa: E402

PUNCT_RE = re.compile(r"\s+")
TANWEEN_MARKER_RE = re.compile(r"([\u0600-\u06ff]+)([FNK])(?=$|[\s،,.!?؛:)\]}])")
TANWEEN_MARKERS = {"F": "\u064b", "N": "\u064c", "K": "\u064d"}


def normalize_spaces(text: str) -> str:
    return PUNCT_RE.sub(" ", (text or "").strip())


def normalize_asc_tanween_markers(text: str) -> str:
    return TANWEEN_MARKER_RE.sub(lambda m: m.group(1) + TANWEEN_MARKERS[m.group(2)], text or "")


def final_endings(text: str) -> list[dict]:
    endings = []
    for idx, raw_word in enumerate(normalize_spaces(text).split()):
        word = raw_word.strip("،,.!?؛:()[]{}\"'")
        if not word:
            continue
        ending = extract_ending_from_arabic(word)
        endings.append({"word_index": idx, "word": word, "ending": ending})
    return endings


def instruction_for(asr_text: str) -> str:
    return (
        "Correct this Arabic ASR transcript for Modern Standard Arabic. "
        "Return the same sentence with full tashkeel, focusing on final i'rab "
        "endings and tanween. Do not add explanations.\n\n"
        f"ASR transcript:\n{asr_text}"
    )


def iter_sentence_manifest(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref = normalize_spaces(row.get("reference_text", ""))
            if ref:
                yield {
                    "id": row.get("utterance_id") or row.get("id") or path.stem,
                    "audio_path": row.get("audio_path", ""),
                    "source": row.get("source", path.stem),
                    "gold_text": ref,
                }


def iter_wordtier_manifest(path: Path):
    grouped: OrderedDict[str, dict] = OrderedDict()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row.get("utterance_id", "")
            if not uid:
                continue
            item = grouped.setdefault(
                uid,
                {
                    "id": uid,
                    "audio_path": row.get("audio_path", ""),
                    "source": row.get("source", path.stem),
                    "gold_text": normalize_spaces(row.get("reference_text", "")),
                },
            )
            if not item["gold_text"]:
                item["gold_text"] = normalize_spaces(row.get("reference_text", ""))
    for item in grouped.values():
        if item["gold_text"]:
            yield item


def build_example(item: dict) -> dict:
    gold = normalize_spaces(normalize_asc_tanween_markers(item["gold_text"]))
    asr_like = normalize_spaces(strip_diacritics(gold))
    return {
        "id": item["id"],
        "source": item["source"],
        "audio_path": item.get("audio_path", ""),
        "asr_transcript": asr_like,
        "gold_diacritized_text": gold,
        "final_endings": final_endings(gold),
        "messages": [
            {"role": "system", "content": "You restore Arabic tashkeel and final tanween from ASR text."},
            {"role": "user", "content": instruction_for(asr_like)},
            {"role": "assistant", "content": gold},
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--etib", type=Path, default=ROOT / "arabic_sentence_ending_training/data/manifests/etib_validation_sentences.csv")
    parser.add_argument("--asc-wordtier", type=Path, default=ROOT / "arabic_sentence_ending_training/data/manifests/asc_wordtier_manifest.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "arabic_sentence_ending_training/data/gemma_tanween/gemma_tanween_train.jsonl")
    parser.add_argument("--max-asc", type=int, default=800)
    parser.add_argument("--skip-asc", type=int, default=0)
    parser.add_argument("--include-etib", action="store_true", default=False)
    args = parser.parse_args()

    examples = []
    if args.include_etib and args.etib.exists():
        examples.extend(build_example(item) for item in iter_sentence_manifest(args.etib))
    if args.asc_wordtier.exists():
        for i, item in enumerate(iter_wordtier_manifest(args.asc_wordtier)):
            if i < args.skip_asc:
                continue
            if i >= args.skip_asc + args.max_asc:
                break
            examples.append(build_example(item))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(json.dumps({"output": str(args.output), "examples": len(examples)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
