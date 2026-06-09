"""
arabizi.py
==========
Converts fully-diacritized Arabic to Arabizi (phonetic Latin).
Rules:
  fatha  (َ) = a      kasra  (ِ) = i      damma  (ُ) = u
  tanwin fath (ً) = an  tanwin kasr (ٍ) = in  tanwin damm (ٌ) = un
  sukun  (ْ) = (nothing)
  long vowels: اَ=aa  يِ=ii  وُ=uu

This is used to compare what was WRITTEN (reference) vs what was SAID
(Whisper transcript) at the phonetic level — catching i'rab and tanween errors.
"""

# ── Unicode diacritics ────────────────────────────────────────────────────────
FATHA       = '\u064e'   # َ
KASRA       = '\u0650'   # ِ
DAMMA       = '\u064f'   # ُ
SUKUN       = '\u0652'   # ْ
SHADDA      = '\u0651'   # ّ
TANWIN_FATH = '\u064b'   # ً
TANWIN_KASR = '\u064d'   # ٍ
TANWIN_DAMM = '\u064c'   # ٌ

ALL_DIACRITICS = {FATHA, KASRA, DAMMA, SUKUN, SHADDA,
                  TANWIN_FATH, TANWIN_KASR, TANWIN_DAMM}

VOWEL_MAP = {
    FATHA:       'a',
    KASRA:       'i',
    DAMMA:       'u',
    SUKUN:       '',
    TANWIN_FATH: 'an',
    TANWIN_KASR: 'in',
    TANWIN_DAMM: 'un',
}

# Consonant transliteration (simplified for phonetic comparison)
CONS = {
    'ب':'b',  'ت':'t',  'ث':'th', 'ج':'j',  'ح':'h',  'خ':'kh',
    'د':'d',  'ذ':'dh', 'ر':'r',  'ز':'z',  'س':'s',  'ش':'sh',
    'ص':'s',  'ض':'d',  'ط':'t',  'ظ':'dh', 'ع':'3',  'غ':'gh',
    'ف':'f',  'ق':'q',  'ك':'k',  'ل':'l',  'م':'m',  'ن':'n',
    'ه':'h',  'ء':"'",  'ئ':"'",  'ؤ':"'",
    'ة':'',   # ta marbuta handled separately
}

ALEF_VARIANTS = {'ا', 'أ', 'إ', 'آ', 'ٱ'}


def to_arabizi(word: str) -> str:
    """
    Convert a diacritized Arabic word to its Arabizi phonetic string.
    Example: مُدِيرَ → mudiira   طَالِبٌ → taalibun
    """
    chars = list(word)
    n = len(chars)
    out = ''
    i = 0
    prev_vowel = ''   # tracks the vowel of the previous letter

    while i < n:
        c = chars[i]

        # ── Skip bare diacritics (processed with their letter) ──
        if c in ALL_DIACRITICS:
            i += 1
            continue

        # ── Read any diacritic(s) that follow this letter ──
        haraka = ''
        has_shadda = False
        j = i + 1
        while j < n and chars[j] in ALL_DIACRITICS:
            if chars[j] == SHADDA:
                has_shadda = True
            else:
                haraka = chars[j]
            j += 1

        vowel = VOWEL_MAP.get(haraka, '')

        # ── Alef variants (ا أ إ آ) ──
        if c in ALEF_VARIANTS:
            if c == 'آ':
                out += 'aa'
            elif haraka == TANWIN_FATH:
                # طَالِباً — the ا carries tanwin fath
                out += 'an'
            elif prev_vowel == FATHA:
                # Long vowel: fatha + alef = 'aa'
                out += 'a'
            elif prev_vowel in (KASRA, DAMMA):
                out += 'a'
            else:
                out += 'a'
            prev_vowel = haraka
            i = j
            continue

        # ── Waw و ──
        if c == 'و':
            if prev_vowel == DAMMA:
                # Long vowel: damma + waw = 'uu'
                out += 'u'
            else:
                out += 'w'
                out += vowel
            prev_vowel = haraka
            i = j
            continue

        # ── Ya ي ──
        if c == 'ي':
            if prev_vowel == KASRA:
                # Long vowel: kasra + ya = 'ii'
                out += 'i'
            else:
                out += 'y'
                out += vowel
            prev_vowel = haraka
            i = j
            continue

        # ── Ta marbuta ة ──
        if c == 'ة':
            # At end of word, silent or 'a' in construct state
            out += 'a' if vowel else ''
            prev_vowel = haraka
            i = j
            continue

        # ── Regular consonant ──
        cons = CONS.get(c, c)
        if has_shadda:
            out += cons + cons   # gemination
        else:
            out += cons
        out += vowel
        prev_vowel = haraka
        i = j

    # Clean up leading/trailing apostrophes
    return out.strip("'")


def strip_diacritics(text: str) -> str:
    return ''.join(c for c in text if c not in ALL_DIACRITICS)


def normalize_alef(text: str) -> str:
    result = ''
    for c in text:
        result += 'ا' if c in ALEF_VARIANTS else c
    return result


def word_to_arabizi(word: str) -> str:
    """Public interface — handles undiacritized input gracefully."""
    return to_arabizi(word)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    cases = [
        ("مُدِيرُ",    "mudiiru"),
        ("مُدِيرَ",    "mudiira"),
        ("مُدِيرِ",    "mudiiri"),
        ("طَالِبٌ",    "taalibun"),
        ("طَالِباً",   "taaliban"),
        ("طَالِبٍ",    "taalibin"),
        ("الْمُدِيرُ", "almudiiru"),
        ("ذَهَبَ",     "dhahaba"),
        ("الاجْتِمَاعِ","alajtimaa3i"),
    ]
    all_ok = True
    for word, expected in cases:
        got = to_arabizi(word)
        ok = got == expected
        all_ok = all_ok and ok
        print(f"  {'✓' if ok else '✗'} {word:18s} → {got:18s}  (expected {expected})")
    print("ALL PASS" if all_ok else "SOME FAILED")