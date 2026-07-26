from __future__ import annotations

import re
from dataclasses import dataclass


ARABIC_DIACRITICS_RE = re.compile(r"[\u064b-\u0652\u0670]")
TRAILING_PUNCT_RE = re.compile(r"^[^\u0600-\u06ff]+|[^\u0600-\u06ff\u064b-\u0652\u0670]+$")

FATHA = "\u064e"
DAMMA = "\u064f"
KASRA = "\u0650"
SUKUN = "\u0652"
TANWIN_FATH = "\u064b"
TANWIN_DAMM = "\u064c"
TANWIN_KASR = "\u064d"
SHADDA = "\u0651"
DAGGER_ALIF = "\u0670"

ENDING_BY_MARK = {
    FATHA: "fatha",
    DAMMA: "damma",
    KASRA: "kasra",
    SUKUN: "none",
    TANWIN_FATH: "tanwin_fath",
    TANWIN_DAMM: "tanwin_damm",
    TANWIN_KASR: "tanwin_kasr",
}


@dataclass(frozen=True)
class LabeledWord:
    word_index: int
    word: str
    expected_ending: str


def clean_word(word: str) -> str:
    return TRAILING_PUNCT_RE.sub("", word.strip())


def strip_diacritics(text: str) -> str:
    return ARABIC_DIACRITICS_RE.sub("", text)


def expected_ending_from_word(word: str) -> str:
    """Return the final pronounced ending encoded in a diacritized Arabic word."""
    word = clean_word(word)
    if not word:
        return "ignore"

    marks: list[str] = []
    reversed_chars = list(reversed(word))
    for ch in reversed_chars:
        if ch in ENDING_BY_MARK or ch in {SHADDA, DAGGER_ALIF}:
            if ch in ENDING_BY_MARK:
                marks.append(ch)
            continue
        break

    if not marks:
        # Accusative tanween is often written before a silent final alif:
        # نَصًّا, كِتَابًا. The pronounced ending is still tanwin_fath.
        if word[-1:] in {"ا", "ى"} and TANWIN_FATH in word[-5:]:
            return "tanwin_fath"
        return "ignore"

    # The last visible short vowel/tanween is the relevant final ending. Shadda
    # and dagger alif are pronunciation details, not the i'rab case ending.
    return ENDING_BY_MARK.get(marks[0], "ignore")


def split_labeled_words(reference_text: str) -> list[LabeledWord]:
    labeled: list[LabeledWord] = []
    for i, raw_word in enumerate(reference_text.split()):
        word = clean_word(raw_word)
        if not word:
            continue
        labeled.append(
            LabeledWord(
                word_index=i,
                word=word,
                expected_ending=expected_ending_from_word(word),
            )
        )
    return labeled
