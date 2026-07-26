from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
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
    parser.add_argument("--manifest", default="data/manifests/asc_word_manifest.csv")
    parser.add_argument("--out", default="data/manifests/asc_balanced_word_manifest.csv")
    parser.add_argument("--max-per-label", type=int, default=500)
    parser.add_argument(
        "--max-utterance-number",
        type=int,
        default=None,
        help="Optional ASC filter. Example: 1100 keeps ARA NORM 0002..1100.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    out_path = ROOT / args.out
    rng = random.Random(args.seed)

    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8-sig")))
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        label = row["expected_ending"]
        if args.max_utterance_number is not None:
            number = _utterance_number(row["utterance_id"])
            if number is None or number > args.max_utterance_number:
                continue
        if label in TRAINABLE_LABELS:
            by_label[label].append(row)

    sampled: list[dict[str, str]] = []
    for label in sorted(by_label):
        label_rows = by_label[label]
        rng.shuffle(label_rows)
        selected = label_rows[: args.max_per_label]
        sampled.extend(selected)
        print(f"{label}: {len(selected)} / {len(label_rows)}")

    sampled.sort(key=lambda row: (row["utterance_id"], int(row["word_index"])))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(sampled)

    print(f"wrote {len(sampled)} rows: {out_path}")


def _utterance_number(utterance_id: str) -> int | None:
    import re

    match = re.search(r"(\d+)$", utterance_id)
    return int(match.group(1)) if match else None


if __name__ == "__main__":
    main()
