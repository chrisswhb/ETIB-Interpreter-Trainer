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

SKIP_TOKENS = {"-", ""}
SILENCE_LABELS = {"sil", "sp", "<sil>", "dist"}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Prepare Arabic Speech Corpus word manifest from lab + TextGrid files."
    )
    parser.add_argument(
        "--dataset-root",
        default="arabic_sentence_ending_training/data/raw/arabic_speech_corpus/arabic-speech-corpus",
    )
    parser.add_argument("--out", default="data/manifests/asc_word_manifest.csv")
    args = parser.parse_args()

    dataset_root = (APP_ROOT / args.dataset_root).resolve()
    if not dataset_root.exists():
        raise SystemExit(f"dataset root not found: {dataset_root}")

    lab_files = sorted(dataset_root.rglob("*.lab"))

    rows: list[dict[str, str]] = []
    skipped = 0
    for lab_path in lab_files:
        wav_path, textgrid_path = matching_asc_files(lab_path)
        if wav_path is None or textgrid_path is None:
            skipped += 1
            continue

        tokens = [
            token
            for token in lab_path.read_text(encoding="utf-8", errors="replace").split()
            if token not in SKIP_TOKENS
        ]
        intervals = parse_phone_intervals(textgrid_path)
        if not tokens or not intervals:
            skipped += 1
            continue

        phone_weights = [max(1, estimate_phone_count(token)) for token in tokens]
        spans = allocate_spans(intervals, phone_weights)
        reference_text = " ".join(buckwalter_to_arabic(token) for token in tokens)
        utterance_id = asc_utterance_id(lab_path)

        for word_index, (token, span) in enumerate(zip(tokens, spans)):
            rows.append(
                {
                    "utterance_id": utterance_id,
                    "audio_path": _display_path(wav_path),
                    "reference_text": reference_text,
                    "word_index": str(word_index),
                    "word": buckwalter_to_arabic(token),
                    "expected_ending": expected_ending_from_buckwalter(token),
                    "start_ms": f"{span[0] * 1000:.1f}",
                    "end_ms": f"{span[1] * 1000:.1f}",
                    "speaker": "asc",
                    "source": "arabic_speech_corpus",
                }
            )

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows: {out_path}")
    print(f"skipped utterances: {skipped}")


def parse_phone_intervals(path: Path) -> list[tuple[float, float, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.findall(
        r"intervals \[\d+\]:\s*xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = \"(.*?)\"",
        text,
        flags=re.DOTALL,
    )
    intervals = []
    for xmin, xmax, label in blocks:
        label = label.strip()
        if not label or label in SILENCE_LABELS:
            continue
        intervals.append((float(xmin), float(xmax), label))
    return intervals


def matching_asc_files(lab_path: Path) -> tuple[Path | None, Path | None]:
    """Match ASC lab/wav/TextGrid inside either main corpus or test set.

    Main files live in:
      lab/<stem>.lab, wav/<stem>.wav, textgrid/<stem>.TextGrid

    Test files live in:
      test set/lab/<stem>.lab, test set/wav/<stem>.wav, test set/textgrid/<stem>.TextGrid

    Matching only by stem can mix these two subsets because filenames repeat.
    """
    subset_root = lab_path.parent.parent
    wav_path = subset_root / "wav" / f"{lab_path.stem}.wav"
    textgrid_path = subset_root / "textgrid" / f"{lab_path.stem}.TextGrid"
    return (
        wav_path if wav_path.exists() else None,
        textgrid_path if textgrid_path.exists() else None,
    )


def asc_utterance_id(lab_path: Path) -> str:
    subset = "test" if "test set" in {part.lower() for part in lab_path.parts} else "main"
    return f"asc_{subset}_{lab_path.stem}"


def allocate_spans(
    intervals: list[tuple[float, float, str]],
    weights: list[int],
) -> list[tuple[float, float]]:
    total_weight = sum(weights)
    total_intervals = len(intervals)
    spans = []
    cursor = 0

    for i, weight in enumerate(weights):
        if i == len(weights) - 1:
            end_cursor = total_intervals
        else:
            end_cursor = round(sum(weights[: i + 1]) / total_weight * total_intervals)
            end_cursor = max(cursor + 1, min(total_intervals, end_cursor))

        chunk = intervals[cursor:end_cursor] or intervals[max(0, cursor - 1):cursor]
        spans.append((chunk[0][0], chunk[-1][1]))
        cursor = end_cursor

    return spans


def estimate_phone_count(token: str) -> int:
    count = 0
    previous_base_was_consonant = False
    previous_char = ""
    for ch in token:
        if ch == "~":
            if previous_base_was_consonant:
                count += 1
            continue
        if ch in {"`", "o"}:
            continue
        if ch in {"F", "N", "K"}:
            # Tanween is acoustically vowel + /n/.
            count += 2
            previous_base_was_consonant = False
            previous_char = ch
            continue
        if ch in {"a", "u", "i"}:
            count += 1
            previous_base_was_consonant = False
            previous_char = ch
            continue
        if ch == "A" and previous_char in {"a", "F"}:
            previous_char = ch
            continue
        if ch == "w" and previous_char == "u":
            previous_char = ch
            continue
        if ch in {"y", "Y"} and previous_char == "i":
            previous_char = ch
            continue
        if ch in BUCKWALTER_TO_ARABIC:
            count += 1
            previous_base_was_consonant = True
            previous_char = ch
    return count


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
