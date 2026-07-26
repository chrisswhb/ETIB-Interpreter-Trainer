"""Build an ASR fine-tuning manifest from Gemma/GOLD diacritized annotations.

This prepares the final stage mentioned in the expert proposal: once Gemma can
label ASR transcripts with tanween/diacritics, those labels can become training
pairs for an ASR model that directly outputs diacritized Arabic.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-csv")
    parser.add_argument(
        "--text-field",
        default="gemma_diacritized_text",
        choices=["gemma_diacritized_text", "gold_diacritized_text"],
        help="Use Gemma annotations for unlabelled data, or gold text for supervised datasets.",
    )
    parser.add_argument(
        "--fallback-to-gold",
        action="store_true",
        help="If the selected text field is absent, use gold_diacritized_text.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_jsonl = Path(args.output_jsonl)
    output_csv = Path(args.output_csv) if args.output_csv else None
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = 0
    with input_path.open("r", encoding="utf-8") as src:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            audio_path = record.get("audio_path")
            text = record.get(args.text_field)
            label_source = args.text_field
            if not text and args.fallback_to_gold:
                text = record.get("gold_diacritized_text")
                label_source = "gold_diacritized_text"
            if not audio_path or not text:
                skipped += 1
                continue
            rows.append(
                {
                    "id": record.get("id") or record.get("utterance_id"),
                    "audio_path": audio_path,
                    "text": text,
                    "source": record.get("source"),
                    "label_source": label_source,
                    "cohere_transcript": record.get("cohere_transcript"),
                }
            )

    with output_jsonl.open("w", encoding="utf-8", newline="\n") as dst:
        for row in rows:
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")

    if output_csv:
        with output_csv.open("w", encoding="utf-8-sig", newline="") as dst:
            writer = csv.DictWriter(
                dst,
                fieldnames=["id", "audio_path", "text", "source", "label_source", "cohere_transcript"],
            )
            writer.writeheader()
            writer.writerows(rows)

    print(
        json.dumps(
            {
                "input": str(input_path),
                "output_jsonl": str(output_jsonl),
                "output_csv": str(output_csv) if output_csv else None,
                "examples": len(rows),
                "skipped": skipped,
                "text_field": args.text_field,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
