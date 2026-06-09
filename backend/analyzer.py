"""
analyzer.py
===========
Compares the reference (diacritized Arabic) with the student's speech
(Whisper transcript) using Arabizi phonetic conversion.

Flow:
  1. Convert reference word → Arabizi   e.g. مُدِيرَ → "mudiira"
  2. Attempt to convert transcript word → Arabizi
     (Whisper usually strips diacritics, so we detect via confidence + word identity)
  3. Compare endings to catch i'rab / tanween errors

Error types:
  ok               — correct
  wrong_ending     — ending clearly different (e.g. mudiiru vs mudiira)
  wrong_word       — different root word entirely
  uncertain        — same root, diacritics stripped, low Whisper confidence
  omission         — word not said at all
"""

import re
from difflib import SequenceMatcher
from dataclasses import dataclass
from typing import Optional
from arabizi import (
    to_arabizi, strip_diacritics, normalize_alef,
    ALL_DIACRITICS, FATHA, KASRA, DAMMA, SUKUN,
    TANWIN_FATH, TANWIN_KASR, TANWIN_DAMM,
    VOWEL_MAP
)

# Confidence below this = Whisper was unsure = likely wrong ending
UNCERTAIN_THRESHOLD = 0.80

ENDING_HARAKA = {
    DAMMA:       ("u",  "مرفوع  nominative"),
    FATHA:       ("a",  "منصوب  accusative"),
    KASRA:       ("i",  "مجرور  genitive"),
    TANWIN_DAMM: ("un", "تنوين ضم  tanwin damm"),
    TANWIN_FATH: ("an", "تنوين فتح  tanwin fath"),
    TANWIN_KASR: ("in", "تنوين كسر  tanwin kasr"),
}


@dataclass
class WordResult:
    index: int
    arabic: str           # stripped Arabic display
    ref_arabizi: str      # full reference Arabizi  e.g. "mudiira"
    ref_ending: str       # just the ending sound   e.g. "a"
    said_arabizi: str     # what student said (Arabizi) or "?" if unknown
    said_ending: str      # ending student said
    status: str           # ok | wrong_ending | wrong_word | uncertain | omission
    irab: str             # grammar note
    confidence: float
    is_focus: bool


def get_final_haraka(word: str):
    for c in reversed(word):
        if c in ALL_DIACRITICS and c != '\u0651':  # skip shadda
            return c
    return None


def root_of(word: str) -> str:
    """Strip diacritics + alef normalization + definite article for root comparison."""
    w = normalize_alef(strip_diacritics(word)).strip()
    w = re.sub(r'^ال', '', w)
    w = w.rstrip('ةه')
    return w


def ending_sound(haraka) -> str:
    if haraka is None:
        return ''
    return VOWEL_MAP.get(haraka, '')


def tokenize(text: str):
    return [w for w in text.split() if w.strip()]


def align(ref_tokens, hyp_tokens):
    ref_norm = [root_of(w) for w in ref_tokens]
    hyp_norm = [root_of(w) for w in hyp_tokens]
    matcher = SequenceMatcher(None, ref_norm, hyp_norm, autojunk=False)
    pairs = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for ri, ji in zip(range(i1, i2), range(j1, j2)):
                pairs.append((ref_tokens[ri], hyp_tokens[ji]))
        elif tag == 'replace':
            for ri in range(i1, i2):
                ji = j1 + (ri - i1)
                pairs.append((ref_tokens[ri], hyp_tokens[ji] if ji < j2 else None))
            for ji in range(j1 + (i2 - i1), j2):
                pairs.append((None, hyp_tokens[ji]))
        elif tag == 'delete':
            for ri in range(i1, i2):
                pairs.append((ref_tokens[ri], None))
        elif tag == 'insert':
            for ji in range(j1, j2):
                pairs.append((None, hyp_tokens[ji]))
    return pairs


def analyze(
    reference_diacritized: str,
    stt_output: str,
    focus_words: list = None,
    word_confidences: dict = None,
) -> list:
    focus_words      = focus_words or []
    word_confidences = word_confidences or {}

    ref_tokens = tokenize(reference_diacritized)
    hyp_tokens = tokenize(stt_output)
    pairs = align(ref_tokens, hyp_tokens)
    results = []

    for idx, (ref_w, hyp_w) in enumerate(pairs):
        if ref_w is None:
            continue

        arabic_plain = strip_diacritics(ref_w).strip()
        is_focus = any(root_of(f) == root_of(ref_w) for f in focus_words)
        conf = word_confidences.get(idx, 1.0)

        # Reference Arabizi
        ref_az  = to_arabizi(ref_w)
        ref_h   = get_final_haraka(ref_w)
        ref_end = ending_sound(ref_h)
        _, irab = ENDING_HARAKA.get(ref_h, ('', ''))

        # ── Omission ──
        if hyp_w is None:
            results.append(WordResult(
                index=idx, arabic=arabic_plain,
                ref_arabizi=ref_az, ref_ending=ref_end,
                said_arabizi='(omitted)', said_ending='',
                status='omission', irab=irab, confidence=0.0, is_focus=is_focus,
            ))
            continue

        # ── Root comparison ──
        ref_root = root_of(ref_w)
        hyp_root = root_of(hyp_w)

        if ref_root != hyp_root:
            # Completely different word
            hyp_az = to_arabizi(hyp_w) if any(c in ALL_DIACRITICS for c in hyp_w) else strip_diacritics(hyp_w)
            results.append(WordResult(
                index=idx, arabic=arabic_plain,
                ref_arabizi=ref_az, ref_ending=ref_end,
                said_arabizi=hyp_az, said_ending='',
                status='wrong_word', irab=irab, confidence=conf, is_focus=is_focus,
            ))
            continue

        # ── Same root — check ending ──
        hyp_has_diacritics = any(c in ALL_DIACRITICS for c in hyp_w)

        if hyp_has_diacritics:
            # Whisper preserved diacritics — direct Arabizi comparison
            hyp_az  = to_arabizi(hyp_w)
            hyp_h   = get_final_haraka(hyp_w)
            hyp_end = ending_sound(hyp_h)

            if ref_h == hyp_h:
                status = 'ok'
            else:
                status = 'wrong_ending'
            results.append(WordResult(
                index=idx, arabic=arabic_plain,
                ref_arabizi=ref_az, ref_ending=ref_end,
                said_arabizi=hyp_az, said_ending=hyp_end,
                status=status, irab=irab, confidence=conf, is_focus=is_focus,
            ))
        else:
            # Whisper stripped diacritics — we cannot know the exact ending
            # Mark ALL i'rab words as uncertain so teacher/student knows
            # to verify. The diacritized prompt improves this over time.
            hyp_az = strip_diacritics(hyp_w)
            if conf < UNCERTAIN_THRESHOLD:
                said_end = f'unclear ({int(conf*100)}% conf)'
            else:
                said_end = 'ending stripped by STT'
            status = 'uncertain'

            results.append(WordResult(
                index=idx, arabic=arabic_plain,
                ref_arabizi=ref_az, ref_ending=ref_end,
                said_arabizi=hyp_az, said_ending=said_end,
                status=status, irab=irab, confidence=conf, is_focus=is_focus,
            ))

    return results


def build_payload(results: list) -> dict:
    words = []
    errors = []
    for r in results:
        w = {
            'word':         r.arabic,
            'ref_arabizi':  r.ref_arabizi,
            'ref_ending':   r.ref_ending,
            'said_arabizi': r.said_arabizi,
            'said_ending':  r.said_ending,
            'status':       r.status,
            'irab':         r.irab,
            'confidence':   round(r.confidence, 3),
            'is_focus':     r.is_focus,
        }
        words.append(w)
        if r.status != 'ok':
            errors.append(w)

    score = round(
        sum(1 for w in words if w['status'] == 'ok') / len(words) * 100
        if words else 0, 1
    )
    return {'words': words, 'errors': errors, 'score': score}