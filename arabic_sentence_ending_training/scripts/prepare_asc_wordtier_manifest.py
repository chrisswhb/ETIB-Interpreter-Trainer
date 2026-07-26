from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = ROOT.parent

BUCKWALTER_TO_ARABIC = {
    "'": "ء",
    "|": "آ",
    ">": "أ",
    "&": "ؤ",
    "<": "إ",
    "}": "ئ",
    "A": "ا",
    "b": "ب",
    "p": "ة",
    "t": "ت",
    "v": "ث",
    "j": "ج",
    "H": "ح",
    "x": "خ",
    "d": "د",
    "*": "ذ",
    "r": "ر",
    "z": "ز",
    "s": "س",
    "$": "ش",
    "S": "ص",
    "D": "ض",
    "T": "ط",
    "Z": "ظ",
    "E": "ع",
    "g": "غ",
    "_": "ـ",
    "f": "ف",
    "q": "ق",
    "k": "ك",
    "l": "ل",
    "m": "م",
    "n": "ن",
    "h": "ه",
    "w": "و",
    "Y": "ى",
    "y": "ي",
    "F": "ً",
    "N": "ٌ",
    "K": "ٍ",
    "a": "َ",
    "u": "ُ",
    "i": "ِ",
    "~": "ّ",
    "o": "ْ",
    "`": "ٰ",
    "{": "ٱ",
}

ENDING_BY_BUCKWALTER = {
    "a": "fatha",
    "u": "damma",
    "i": "kasra",
    "o": "none",
    "F": "tanwin_fath",
    "N": "tanwin_damm",
    "K": "tanwin_kasr",
}

SILENCE_LABELS = {"sil", "sp", "<sil>", "dist", ""}
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
    "buckwalter_word",
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Prepare ASC manifest from the TextGrid words tier."
    )
    parser.add_argument(
        "--dataset-root",
        default="arabic_sentence_ending_training/data/raw/arabic_speech_corpus/arabic-speech-corpus",
    )
    parser.add_argument("--out", default="data/manifests/asc_wordtier_manifest.csv")
    args = parser.parse_args()

    dataset_root = (APP_ROOT / args.dataset_root).resolve()
    textgrids = sorted(dataset_root.rglob("*.TextGrid"))
    rows: list[dict[str, str]] = []
    skipped = 0

    for textgrid_path in textgrids:
        wav_path = matching_wav_file(textgrid_path)
        if wav_path is None:
            skipped += 1
            continue
        intervals = parse_words_tier(textgrid_path)
        words = [item for item in intervals if item[2] not in SILENCE_LABELS]
        if not words:
            skipped += 1
            continue

        reference_text = " ".join(buckwalter_to_arabic(token) for _, _, token in words)
        utterance_id = asc_utterance_id(textgrid_path)
        for word_index, (start, end, token) in enumerate(words):
            ending = expected_ending_from_buckwalter(token)
            if ending == "ignore":
                continue
            rows.append(
                {
                    "utterance_id": utterance_id,
                    "audio_path": _display_path(wav_path),
                    "reference_text": reference_text,
                    "word_index": str(word_index),
                    "word": buckwalter_to_arabic(token),
                    "expected_ending": ending,
                    "start_ms": f"{start * 1000:.1f}",
                    "end_ms": f"{end * 1000:.1f}",
                    "speaker": "asc",
                    "source": "arabic_speech_corpus_wordtier",
                    "buckwalter_word": token,
                }
            )

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["expected_ending"]] = counts.get(row["expected_ending"], 0) + 1
    for label, count in sorted(counts.items()):
        print(f"{label}: {count}")
    print(f"wrote {len(rows)} rows: {out_path}")
    print(f"skipped files: {skipped}")


def parse_words_tier(path: Path) -> list[tuple[float, float, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r'name = "words".*?intervals: size = \d+(.*?)(?:\n    item \[\d+\]:|\Z)',
        text,
        flags=re.DOTALL,
    )
    if not match:
        return []
    blocks = re.findall(
        r"intervals \[\d+\]:\s*xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = \"(.*?)\"",
        match.group(1),
        flags=re.DOTALL,
    )
    return [
        (float(xmin), float(xmax), label.strip())
        for xmin, xmax, label in blocks
    ]


def matching_wav_file(textgrid_path: Path) -> Path | None:
    subset_root = textgrid_path.parent.parent
    wav_path = subset_root / "wav" / f"{textgrid_path.stem}.wav"
    return wav_path if wav_path.exists() else None


def asc_utterance_id(textgrid_path: Path) -> str:
    subset = "test" if "test set" in {part.lower() for part in textgrid_path.parts} else "main"
    return f"asc_{subset}_{textgrid_path.stem}"


def expected_ending_from_buckwalter(token: str) -> str:
    token = token.strip()
    if not token:
        return "ignore"
    if token[-1:] in {"A", "Y"} and "F" in token[-4:]:
        return "tanwin_fath"
    for ch in reversed(token):
        if ch in ENDING_BY_BUCKWALTER:
            return ENDING_BY_BUCKWALTER[ch]
        if ch in {"~", "`", "A", "Y"}:
            continue
        break
    return "ignore"


def buckwalter_to_arabic(text: str) -> str:
    return "".join(BUCKWALTER_TO_ARABIC.get(ch, ch) for ch in text)


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(APP_ROOT.resolve()))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
