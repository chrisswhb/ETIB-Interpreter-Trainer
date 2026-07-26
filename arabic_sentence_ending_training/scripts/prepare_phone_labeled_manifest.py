from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = ROOT.parent

TRAINABLE_LABELS = {
    "fatha",
    "damma",
    "kasra",
    "none",
    "tanwin_fath",
    "tanwin_damm",
    "tanwin_kasr",
}

SILENCE_LABELS = {"sil", "sp", "<sil>", "dist", ""}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Relabel ASC word manifest from the actual final phone in TextGrid."
    )
    parser.add_argument("--manifest", default="data/manifests/asc_word_manifest.csv")
    parser.add_argument("--out", default="data/manifests/asc_phone_labeled_word_manifest.csv")
    parser.add_argument(
        "--keep-reference-tanween",
        action="store_true",
        help="Only emit tanween labels when the written reference also expects tanween.",
    )
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    out_path = ROOT / args.out
    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8-sig")))

    phone_cache: dict[Path, list[tuple[float, float, str]]] = {}
    out_rows: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    skipped = 0

    for row in rows:
        if row.get("expected_ending") not in TRAINABLE_LABELS:
            skipped += 1
            continue
        textgrid_path = _textgrid_from_audio(row["audio_path"])
        if not textgrid_path.exists():
            skipped += 1
            continue
        if textgrid_path not in phone_cache:
            phone_cache[textgrid_path] = parse_phone_intervals(textgrid_path)

        start = float(row["start_ms"]) / 1000.0
        end = float(row["end_ms"]) / 1000.0
        phones = [
            label
            for xmin, xmax, label in phone_cache[textgrid_path]
            if xmax > start and xmin < end and label.strip() not in SILENCE_LABELS
        ]
        pronounced = pronounced_ending_from_phones(
            phones,
            reference_expected=row.get("expected_ending", ""),
            keep_reference_tanween=args.keep_reference_tanween,
        )
        if pronounced not in TRAINABLE_LABELS:
            skipped += 1
            continue

        new_row = dict(row)
        new_row["written_ending"] = row["expected_ending"]
        new_row["expected_ending"] = pronounced
        new_row["final_phones"] = " ".join(phones[-4:])
        counts[pronounced] = counts.get(pronounced, 0) + 1
        out_rows.append(new_row)

    fieldnames = list(rows[0].keys())
    for name in ("written_ending", "final_phones"):
        if name not in fieldnames:
            fieldnames.append(name)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    for label, count in sorted(counts.items()):
        print(f"{label}: {count}")
    print(f"wrote {len(out_rows)} rows: {out_path}")
    print(f"skipped rows: {skipped}")


def parse_phone_intervals(path: Path) -> list[tuple[float, float, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.findall(
        r"intervals \[\d+\]:\s*xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = \"(.*?)\"",
        text,
        flags=re.DOTALL,
    )
    return [
        (float(xmin), float(xmax), label.strip())
        for xmin, xmax, label in blocks
        if label.strip() not in SILENCE_LABELS
    ]


def pronounced_ending_from_phones(
    phones: list[str],
    reference_expected: str,
    keep_reference_tanween: bool,
) -> str:
    normalized = [_normalize_phone(phone) for phone in phones if _normalize_phone(phone)]
    if not normalized:
        return "ignore"

    last = normalized[-1]
    previous = normalized[-2] if len(normalized) >= 2 else ""

    if last == "n" and previous in {"a", "u", "i"}:
        tanween_label = {"a": "tanwin_fath", "u": "tanwin_damm", "i": "tanwin_kasr"}[previous]
        if not keep_reference_tanween or reference_expected == tanween_label:
            return tanween_label
        return {"a": "fatha", "u": "damma", "i": "kasra"}[previous]

    if last in {"a", "u", "i"}:
        return {"a": "fatha", "u": "damma", "i": "kasra"}[last]

    return "none"


def _normalize_phone(phone: str) -> str:
    phone = phone.strip()
    if not phone or phone in SILENCE_LABELS:
        return ""
    if phone in {"n", "nn"}:
        return "n"
    if phone[0] in {"a", "A"}:
        return "a"
    if phone[0] in {"u", "U"}:
        return "u"
    if phone[0] in {"i", "I"}:
        return "i"
    return "c"


def _textgrid_from_audio(audio_path_text: str) -> Path:
    audio_path = (APP_ROOT / audio_path_text).resolve()
    subset_root = audio_path.parent.parent
    return subset_root / "textgrid" / f"{audio_path.stem}.TextGrid"


if __name__ == "__main__":
    main()
