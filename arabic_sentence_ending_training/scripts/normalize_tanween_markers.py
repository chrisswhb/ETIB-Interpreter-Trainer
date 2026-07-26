"""Normalize ASC-style tanween markers in Gemma datasets.

Some Arabic Speech Corpus text uses Buckwalter tanween markers after Arabic
words, e.g. جاذِبN / دَخلN, where:

    F -> tanwin fath
    N -> tanwin damm
    K -> tanwin kasr

This script converts those markers to real Arabic Unicode tanween characters
inside gold references, final ending labels, and assistant messages.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TANWEEN_MARKERS = {
    "F": "\u064b",
    "N": "\u064c",
    "K": "\u064d",
}

MARKER_RE = re.compile(r"([\u0600-\u06ff]+)([FNK])(?=$|[\s،,.!?؛:)\]}])")


def normalize_text(text: str) -> str:
    def repl(match: re.Match) -> str:
        return match.group(1) + TANWEEN_MARKERS[match.group(2)]

    return MARKER_RE.sub(repl, text or "")


def ending_from_word(word: str) -> str:
    for ch in reversed(word):
        if ch == "\u064b":
            return "tanwin_fath"
        if ch == "\u064c":
            return "tanwin_damm"
        if ch == "\u064d":
            return "tanwin_kasr"
        if ch == "\u064e":
            return "fatha"
        if ch == "\u064f":
            return "damma"
        if ch == "\u0650":
            return "kasra"
        if ch == "\u0652":
            return "none"
        if "\u0600" <= ch <= "\u06ff":
            break
    return "none"


def final_endings(text: str) -> list[dict]:
    out = []
    for idx, raw in enumerate((text or "").split()):
        word = raw.strip("،,.!?؛:()[]{}\"'")
        if word:
            out.append({"word_index": idx, "word": word, "ending": ending_from_word(word)})
    return out


def normalize_record(record: dict) -> dict:
    gold = normalize_text(record.get("gold_diacritized_text", ""))
    record["gold_diacritized_text"] = gold
    record["final_endings"] = final_endings(gold)
    for message in record.get("messages", []):
        if message.get("role") == "assistant":
            message["content"] = gold
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    tanween = 0
    with args.input.open("r", encoding="utf-8") as src, args.output.open("w", encoding="utf-8", newline="\n") as dst:
        for line in src:
            record = normalize_record(json.loads(line))
            tanween += sum(1 for item in record.get("final_endings", []) if item.get("ending", "").startswith("tanwin"))
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(json.dumps({"output": str(args.output), "examples": count, "tanween_labels": tanween}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
