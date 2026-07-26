"""Merge Gemma tanween JSONL datasets while avoiding duplicate IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-duplicates", action="store_true")
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    count = 0
    tanween = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as dst:
        for path in args.inputs:
            with path.open("r", encoding="utf-8") as src:
                for line in src:
                    record = json.loads(line)
                    key = record.get("id") or f"{path}:{count}"
                    if key in seen and not args.allow_duplicates:
                        continue
                    seen.add(key)
                    tanween += sum(
                        1
                        for item in record.get("final_endings", [])
                        if item.get("ending", "").startswith("tanwin")
                    )
                    dst.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
    print(json.dumps({"output": str(args.output), "examples": count, "tanween_labels": tanween}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
