from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from arabic_endings import split_labeled_words


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Expand sentence-level metadata into one manifest row per word."
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "CSV with columns utterance_id,audio_path,reference_text,start_ms,end_ms,"
            "speaker,source. start_ms/end_ms are optional whole-sentence bounds."
        ),
    )
    parser.add_argument("--out", default="data/manifests/manual_sentence_manifest.csv")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_path = ROOT / args.out
    rows_out: list[dict[str, str]] = []

    if not input_path.exists():
        raise SystemExit(f"input CSV not found: {input_path}")

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels = split_labeled_words(row["reference_text"])
            sent_start = _float_or_none(row.get("start_ms"))
            sent_end = _float_or_none(row.get("end_ms"))
            usable_words = [item for item in labels if item.expected_ending != "ignore"]
            n = max(1, len(labels))

            for item in labels:
                if sent_start is not None and sent_end is not None and sent_end > sent_start:
                    # Bootstrap fallback when no word alignment exists yet. Good enough
                    # to create a first manifest, but replace with real alignment ASAP.
                    start_ms = sent_start + (sent_end - sent_start) * item.word_index / n
                    end_ms = sent_start + (sent_end - sent_start) * (item.word_index + 1) / n
                else:
                    start_ms = ""
                    end_ms = ""

                rows_out.append(
                    {
                        "utterance_id": row["utterance_id"],
                        "audio_path": row["audio_path"],
                        "reference_text": row["reference_text"],
                        "word_index": str(item.word_index),
                        "word": item.word,
                        "expected_ending": item.expected_ending,
                        "start_ms": _fmt(start_ms),
                        "end_ms": _fmt(end_ms),
                        "speaker": row.get("speaker", ""),
                        "source": row.get("source", "manual"),
                    }
                )

            if not usable_words:
                print(f"warning: no trainable endings in {row['utterance_id']}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"wrote {len(rows_out)} word rows: {out_path}")


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _fmt(value: object) -> str:
    if value == "":
        return ""
    return f"{float(value):.1f}"


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
