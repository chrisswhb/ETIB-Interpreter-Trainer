from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_cases(path: Path) -> list[dict]:
    if path.is_file():
        return [json.loads(path.read_text(encoding="utf-8"))]
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(path.glob("*.json"))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved /api/analyze JSON files against expected statuses."
    )
    parser.add_argument("path", help="A result JSON file or a folder of result JSON files")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Ignore findings below this confidence when computing accuracy.",
    )
    args = parser.parse_args()

    cases = load_cases(Path(args.path))
    total = 0
    correct = 0
    uncertain = 0
    incorrect = 0
    rows = []

    for case_idx, case in enumerate(cases, start=1):
        for finding in case.get("iraab_findings", []):
            conf = float(finding.get("detection_confidence", 0))
            if conf < args.min_confidence:
                continue
            total += 1
            status = finding.get("status")
            if status == "correct":
                correct += 1
            elif status == "uncertain":
                uncertain += 1
            else:
                incorrect += 1
            rows.append({
                "case": case_idx,
                "word": finding.get("word"),
                "expected": finding.get("expected_ending"),
                "detected": finding.get("detected_ending"),
                "status": status,
                "confidence": conf,
            })

    decided = correct + incorrect
    overall = correct / total if total else 0.0
    decided_accuracy = correct / decided if decided else 0.0

    print(f"files: {len(cases)}")
    print(f"findings: {total}")
    print(f"correct: {correct}")
    print(f"incorrect: {incorrect}")
    print(f"uncertain: {uncertain}")
    print(f"overall_accuracy_including_uncertain: {overall:.1%}")
    print(f"decided_accuracy_excluding_uncertain: {decided_accuracy:.1%}")
    print()
    for row in rows:
        print(
            f"case={row['case']} word={row['word']} expected={row['expected']} "
            f"detected={row['detected']} status={row['status']} conf={row['confidence']:.0%}"
        )


if __name__ == "__main__":
    main()
