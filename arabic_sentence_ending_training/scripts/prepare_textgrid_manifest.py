from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

from arabic_endings import expected_ending_from_word


ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = ROOT.parent


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description=(
            "Create a word-level manifest from a folder containing WAV files and "
            "Praat TextGrid word intervals. This is the first importer to try for "
            "Arabic Speech Corpus-style aligned datasets."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--audio-glob", default="**/*.wav")
    parser.add_argument("--textgrid-glob", default="**/*.TextGrid")
    parser.add_argument("--out", default="data/manifests/textgrid_sentence_manifest.csv")
    parser.add_argument("--source", default="textgrid_dataset")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_path = ROOT / args.out
    audio_by_stem = {p.stem: p for p in dataset_root.glob(args.audio_glob)}
    textgrids = list(dataset_root.glob(args.textgrid_glob))
    rows: list[dict[str, str]] = []

    for textgrid in textgrids:
        audio_path = audio_by_stem.get(textgrid.stem)
        if not audio_path:
            print(f"missing matching wav for {textgrid}")
            continue
        intervals = parse_textgrid_intervals(textgrid)
        words = [item["text"] for item in intervals if item["text"].strip()]
        reference_text = " ".join(words)
        utterance_id = textgrid.stem

        for word_index, item in enumerate(intervals):
            word = item["text"].strip()
            if not word:
                continue
            rows.append(
                {
                    "utterance_id": utterance_id,
                    "audio_path": str(audio_path.relative_to(APP_ROOT)) if _is_relative_to(audio_path, APP_ROOT) else str(audio_path),
                    "reference_text": reference_text,
                    "word_index": str(word_index),
                    "word": word,
                    "expected_ending": expected_ending_from_word(word),
                    "start_ms": f"{item['xmin'] * 1000:.1f}",
                    "end_ms": f"{item['xmax'] * 1000:.1f}",
                    "speaker": "",
                    "source": args.source,
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows: {out_path}")


def parse_textgrid_intervals(path: Path) -> list[dict[str, float | str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.findall(
        r"intervals \[\d+\]:\s*xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = \"(.*?)\"",
        text,
        flags=re.DOTALL,
    )
    intervals = []
    for xmin, xmax, label in blocks:
        clean = label.replace('\\"', '"').strip()
        if not clean or clean in {"sil", "sp", "<sil>", "<SIL>"}:
            continue
        if not re.search(r"[\u0600-\u06ff]", clean):
            continue
        intervals.append({"xmin": float(xmin), "xmax": float(xmax), "text": clean})
    return intervals


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


FIELDNAMES = [
    "utterance_id",
    "audio_path",
    "reference_text",
    "word_index",
    "word",
    "expected_ending",
    "start_ms",
    "end_ms",
    "speaker",
    "source",
]


if __name__ == "__main__":
    main()

