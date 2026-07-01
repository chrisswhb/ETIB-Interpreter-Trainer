"""
arabizi.py — Arabic diacritic → phonetic Latin converter.
Your existing module (Signal 1 reference framework).

Maps damma→u, fatha→a, kasra→i, tanween variants, etc.
Strips non-diacritized consonants to their base form for comparison.
"""

import re
import unicodedata

# Arabic Unicode ranges
ARABIC_DIACRITICS = {
    "\u064e": "a",   # fatha   َ
    "\u064f": "u",   # damma   ُ
    "\u0650": "i",   # kasra   ِ
    "\u064b": "an",  # tanwin fath  ً
    "\u064c": "un",  # tanwin damm  ٌ
    "\u064d": "in",  # tanwin kasr  ٍ
    "\u0652": "",    # sukun   ْ  (no vowel)
    "\u0651": "",    # shadda  ّ  (consonant doubling, skip for ending detection)
}

DIACRITIC_RE = re.compile(r"[\u064b-\u0652]")

# Ending vowel letter → haraka
ENDING_LETTER_MAP = {
    "\u0627": "a",   # alef ا → fatha
    "\u0649": "a",   # alef maqsura ى → fatha
    "\u0648": "u",   # waw  و → damma
    "\u064a": "i",   # ya   ي → kasra
}

HARAKA_LABELS = {
    "a":  "fatha",
    "u":  "damma",
    "i":  "kasra",
    "an": "tanwin_fath",
    "un": "tanwin_damm",
    "in": "tanwin_kasr",
}


def arabic_to_arabizi(text: str) -> str:
    """
    Convert a fully diacritized Arabic string to a phonetic Latin string.
    Example: الْمُدِيرُ  → almudiiru
    """
    result = []
    chars = list(text)
    i = 0
    while i < len(chars):
        ch = chars[i]
        cp = ord(ch)
        # Skip tatweel
        if ch == "\u0640":
            i += 1
            continue
        if ch in ARABIC_DIACRITICS:
            result.append(ARABIC_DIACRITICS[ch])
        elif "\u0600" <= ch <= "\u06FF":
            result.append(_arabic_letter_to_latin(ch))
        elif ch == " ":
            result.append(" ")
        i += 1
    return "".join(result)


def extract_ending(arabizi_word: str) -> str:
    """
    Given an Arabizi word string, return the final vowel marker (a/u/i/an/un/in/'').
    """
    for suffix in ("an", "un", "in"):
        if arabizi_word.endswith(suffix):
            return suffix
    for suffix in ("a", "u", "i"):
        if arabizi_word.endswith(suffix):
            return suffix
    return ""


def extract_ending_from_arabic(word: str) -> str:
    """
    Extract the final haraka/ending directly from a diacritized Arabic word.
    Returns one of: fatha, damma, kasra, tanwin_fath, tanwin_damm, tanwin_kasr, none
    """
    # Walk backwards through characters looking for the last diacritic
    for ch in reversed(word):
        if ch in ARABIC_DIACRITICS:
            arabizi = ARABIC_DIACRITICS[ch]
            return HARAKA_LABELS.get(arabizi, "none")
        # Skip non-diacritic Arabic letters to find the final vowel mark
        if "\u0600" <= ch <= "\u06FF" and ch not in ARABIC_DIACRITICS:
            break
    return "none"


def strip_diacritics(text: str) -> str:
    """Remove Arabic harakaat/shadda/sukun/tanween from text."""
    return DIACRITIC_RE.sub("", text)


def normalize_arabic_word(word: str) -> str:
    """Normalize Arabic word spelling enough to compare reference with ASR output."""
    text = strip_diacritics(word)
    text = re.sub(r"[^\u0600-\u06FF]", "", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي")
    return text


def remove_final_long_vowel_marker(word: str) -> str:
    """Remove common ASR-written ending markers used to represent final short vowels."""
    text = normalize_arabic_word(word)
    for suffix in ("ون", "ين", "ان"):
        if text.endswith(suffix):
            return text[:-len(suffix)]
    if text.endswith(("ا", "و", "ي", "ة")) and len(text) > 1:
        return text[:-1]
    return text


def arabizi_to_haraka_label(ending: str) -> str:
    """Convert arabizi ending string to label used in JSON schema."""
    return HARAKA_LABELS.get(ending, "none")


def haraka_to_arabizi(haraka: str) -> str:
    """Reverse: label → arabizi ending string."""
    rev = {v: k for k, v in HARAKA_LABELS.items()}
    return rev.get(haraka, "")


def _arabic_letter_to_latin(ch: str) -> str:
    """Very rough consonant mapping for Arabizi output (good enough for ending comparison)."""
    MAP = {
        "ا": "a", "ب": "b", "ت": "t", "ث": "th", "ج": "j", "ح": "h",
        "خ": "kh", "د": "d", "ذ": "dh", "ر": "r", "ز": "z", "س": "s",
        "ش": "sh", "ص": "s", "ض": "d", "ط": "t", "ظ": "dh", "ع": "'",
        "غ": "gh", "ف": "f", "ق": "q", "ك": "k", "ل": "l", "م": "m",
        "ن": "n", "ه": "h", "و": "w", "ي": "y", "ة": "a", "ء": "'",
        "أ": "a", "إ": "i", "آ": "aa", "ى": "a", "ئ": "y", "ؤ": "w",
        "لا": "la",
        "ال": "al",
    }
    return MAP.get(ch, ch)
