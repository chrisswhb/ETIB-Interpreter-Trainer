"""
UN document sourcing via the documents.un.org symbol API.
=========================================================

WHY THIS EXISTS (16 Sep 2026)
-----------------------------
The UN put the whole of digitallibrary.un.org behind an AWS WAF JavaScript
challenge. Every path on that host -- the home page, /record/<id>, the PDF
files and the /search API the platform used -- now answers:

    HTTP/1.1 202 Accepted
    Server: awselb/2.0
    x-amzn-waf-action: challenge

with either an empty body or the AWS `gokuProps` challenge page. Passing it
requires executing the challenge JavaScript and earning an `aws-waf-token`
cookie. The old three-tier bypass in module_library (curl_cffi -> curl ->
requests) was built for TLS-FINGERPRINT blocking, which is a different and
weaker WAF mode; no amount of TLS impersonation runs JavaScript. So every UN
search silently returned [] and Module A fell through to its Wikipedia
fallback on every single generation -- grounding interpreter-training
speeches in Wikipedia instead of real UN documents.

WHAT STILL WORKS
----------------
documents.un.org is a different host and is NOT challenged. Its symbol API

    https://documents.un.org/api/symbol/access?s=<SYMBOL>&l=<E|F|A>&t=pdf

serves the real PDFs, verified in English, French and Arabic. That is the
source this module uses.

WHY A CATALOG INSTEAD OF A SEARCH
---------------------------------
No keyword search survives: search.un.org/api/search is broken (HTTP 400 on
every parameter spelling), documents.un.org/api/search wants an auth token
(403), and the digitallibrary OAI-PMH interface -- which IS reachable -- only
exposes a single `sanctions` set, so it cannot answer a topic query either.

The symbol API, though, is addressable: if you know the symbol you get the
document. So this module ships a catalog of verified symbols and ranks it
locally against the topic. Every entry below was fetched on 16 Sep 2026 and
its date read out of page 1 of the PDF itself -- nothing here is guessed.

The catalog is deliberately made of VERBATIM RECORDS (the ...PV... symbols).
Those are transcripts of speeches actually delivered at the UN, which is
exactly the register interpreter trainees need -- and a far better grounding
source than the summary documents the old keyword search often returned.

ADDING MORE DOCUMENTS
---------------------
Append to _GA_PLENARY / _SC_MEETINGS. Verify a symbol first -- the API answers
HTTP 200 with a ZERO-length body for a symbol that is not published yet (all
of A/80/PV.3+ behaves this way today), so status code alone proves nothing.
`validate_catalog()` at the bottom of this file checks a symbol properly.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import tempfile
import threading

# ── Endpoints ────────────────────────────────────────────────────────────────
UN_SYMBOL_API  = 'https://documents.un.org/api/symbol/access'
UN_DOCS_VIEWER = 'https://docs.un.org/en'

# UN language codes used by the symbol API (?l=). The platform speaks en/fr/ar;
# the ISO 639-2 spellings are what module_library already passes around.
UN_LANG_CODE = {
    'eng': 'E', 'en': 'E',
    'fre': 'F', 'fra': 'F', 'fr': 'F',
    'ara': 'A', 'ar': 'A',
    'spa': 'S', 'es': 'S',
    'rus': 'R', 'ru': 'R',
    'chi': 'C', 'zh': 'C',
}

# The symbol API answers 200 with an empty body for an unpublished symbol, so
# callers must check the payload size, never just the status code.
MIN_PDF_BYTES = 20_000


def symbol_pdf_url(symbol: str, un_lang: str = 'eng') -> str:
    """Direct PDF URL for a UN symbol in the requested language."""
    code = UN_LANG_CODE.get((un_lang or 'eng').lower(), 'E')
    # The symbol's slashes must be percent-encoded or the API 404s.
    return f'{UN_SYMBOL_API}?s={symbol.replace("/", "%2F")}&l={code}&t=pdf'


def symbol_web_url(symbol: str) -> str:
    """Human-readable viewer URL, for the source chip in the UI."""
    return f'{UN_DOCS_VIEWER}/{symbol}'


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f'{n}th'
    return f'{n}{ {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th") }'.replace(' ', '')


# ── General Assembly plenary verbatim records ────────────────────────────────
# (symbol, date read from page 1, is_general_debate)
# Sessions 78 and 79 are complete through the general debate. Session 80 has
# only published PV.1 and PV.2 so far -- PV.3 onwards return 0 bytes.
_GA_PLENARY = [
    ('A/78/PV.1',  '2023-09-05', False), ('A/78/PV.2',  '2023-09-08', False),
    ('A/78/PV.3',  '2023-09-11', True),  ('A/78/PV.4',  '2023-09-19', True),
    ('A/78/PV.5',  '2023-09-19', True),  ('A/78/PV.6',  '2023-09-20', True),
    ('A/78/PV.7',  '2023-09-20', True),  ('A/78/PV.8',  '2023-09-21', True),
    ('A/78/PV.9',  '2023-09-21', True),  ('A/78/PV.10', '2023-09-22', True),
    ('A/78/PV.11', '2023-09-22', True),  ('A/78/PV.12', '2023-09-23', True),
    ('A/78/PV.13', '2023-09-23', True),  ('A/78/PV.14', '2023-09-26', True),
    ('A/78/PV.15', '2023-09-29', True),
    ('A/79/PV.1',  '2024-09-10', False), ('A/79/PV.2',  '2024-09-13', False),
    ('A/79/PV.3',  '2024-09-22', True),  ('A/79/PV.4',  '2024-09-22', True),
    ('A/79/PV.5',  '2024-09-23', True),  ('A/79/PV.6',  '2024-09-23', True),
    ('A/79/PV.7',  '2024-09-24', True),  ('A/79/PV.8',  '2024-09-24', True),
    ('A/79/PV.9',  '2024-09-25', True),  ('A/79/PV.10', '2024-09-25', True),
    ('A/79/PV.11', '2024-09-26', True),  ('A/79/PV.12', '2024-09-26', True),
    ('A/79/PV.13', '2024-09-27', True),  ('A/79/PV.14', '2024-09-27', True),
    ('A/79/PV.15', '2024-09-28', True),
    ('A/80/PV.1',  '2025-09-09', False), ('A/80/PV.2',  '2025-09-12', False),
]

# Heads of State and ministers speaking in the general debate range across the
# whole UN agenda in a single sitting, so these records genuinely do carry
# material for every domain the platform offers. That is why they get the full
# domain spread rather than one label.
_GENERAL_DEBATE_TOPICS = (
    'general debate climate change sustainable development peace security human rights '
    'humanitarian conflict economy trade finance poverty health pandemic education gender '
    'equality women migration refugees terrorism disarmament nuclear food security hunger '
    'international law multilateralism reform technology artificial intelligence digital'
)

_ORGANISATIONAL_TOPICS = (
    'opening of the session election of officers credentials organization of work '
    'general assembly procedure multilateralism'
)

# ── Security Council verbatim records ────────────────────────────────────────
# (symbol, date, agenda subject as printed on page 1, topic keywords)
_SC_MEETINGS = [
    ('S/PV.9400', '2023-08-21',
     'The situation in the Middle East, including the Palestinian question',
     'middle east palestine palestinian israel conflict peace security humanitarian occupation'),
    ('S/PV.9450', '2023-10-23',
     'Security Council resolutions on Kosovo; Report of the Secretary-General on UNMIK',
     'kosovo balkans peacekeeping unmik political mission stabilization europe security'),
    ('S/PV.9500', '2023-12-11',
     'The situation concerning the Democratic Republic of the Congo; Report on MONUSCO',
     'democratic republic congo africa peacekeeping monusco stabilization humanitarian conflict'),
    ('S/PV.9550', '2024-02-15',
     'Threats to international peace and security caused by terrorist acts (ISIL/Da’esh)',
     'terrorism terrorist isil daesh counter-terrorism extremism international peace security'),
    ('S/PV.9600', '2024-04-11',
     'Maintenance of peace and security of Ukraine',
     'ukraine russia war aggression european security sovereignty territorial integrity'),
    ('S/PV.9650', '2024-06-10',
     'The situation in the Middle East, including the Palestinian question',
     'middle east palestine palestinian israel gaza ceasefire humanitarian civilians protection'),
    ('S/PV.9700', '2024-08-07',
     'Women and peace and security',
     'women gender equality peace security participation peacekeeping drawdown empowerment'),
]


def _build_catalog() -> list[dict]:
    """Assemble the in-memory catalog once, at import time."""
    catalog: list[dict] = []

    for symbol, date, is_debate in _GA_PLENARY:
        match = re.match(r'A/(\d+)/PV\.(\d+)', symbol)
        session, meeting = (match.group(1), match.group(2)) if match else ('', '')
        if is_debate:
            title = (f'General Assembly, {_ordinal(int(session))} session, '
                     f'{_ordinal(int(meeting))} plenary meeting — general debate')
            topics = _GENERAL_DEBATE_TOPICS
            description = ('Verbatim record of the general debate: statements by Heads of State, '
                           'Heads of Government and ministers across the full UN agenda.')
        else:
            title = (f'General Assembly, {_ordinal(int(session))} session, '
                     f'{_ordinal(int(meeting))} plenary meeting')
            topics = _ORGANISATIONAL_TOPICS
            description = 'Verbatim record of a plenary meeting of the General Assembly.'
        catalog.append({
            'symbol': symbol, 'title': title, 'date': date,
            'topics': topics, 'description': description,
            'body': 'General Assembly', 'general_debate': is_debate,
        })

    for symbol, date, subject, topics in _SC_MEETINGS:
        meeting = symbol.split('.')[-1]
        catalog.append({
            'symbol': symbol,
            'title': f'Security Council, {_ordinal(int(meeting))} meeting — {subject}',
            'date': date, 'topics': topics,
            'description': f'Verbatim record of a Security Council meeting on: {subject}.',
            'body': 'Security Council', 'general_debate': False,
        })

    return catalog


UN_CATALOG: list[dict] = _build_catalog()

# Words that carry no topical signal and would otherwise match everything.
_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'for', 'to', 'with', 'about',
    'at', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'that', 'this',
    'these', 'those', 'it', 'its', 'as', 'un', 'united', 'nations', 'speech',
    'discours', 'sur', 'le', 'la', 'les', 'des', 'du', 'de', 'et', 'un', 'une',
}


def _tokens(text: str) -> list[str]:
    """Distinct lowercase word tokens of 3+ characters, stopwords removed."""
    raw = re.findall(r'[a-zA-ZÀ-ɏ]{3,}', (text or '').lower())
    seen: dict[str, None] = {}
    for word in raw:
        # Deduplicated: a topic that repeats a word ("women" in both the query
        # and the domain) must not score twice for saying it twice.
        if word not in _STOPWORDS:
            seen.setdefault(word, None)
    return list(seen)


# A hit in the TITLE means the document is actually about the topic.
_TITLE_WEIGHT = 3.0
# A hit in a topic-specific record's keywords is strong evidence too.
_TOPIC_WEIGHT = 2.0
# ...but a hit in a GENERAL-DEBATE record's keywords is weak evidence, because
# that list deliberately spans the entire UN agenda. Matching "terrorism"
# there means "a head of state probably mentioned it", not "this sitting was
# about terrorism". Scoring both at the same weight made the catch-all records
# win every query and buried the Council record that was genuinely on topic.
_BROAD_TOPIC_WEIGHT = 0.3
# Keeps a general-debate record eligible when nothing matches at all, while
# staying below any real hit, so specificity always wins when it exists.
_GENERAL_DEBATE_FLOOR = 0.5
# Matched on a 6-character prefix so "terrorism" finds "terrorist" and
# "migration" finds "migrant" without pulling in a stemming dependency.
_STEM_LEN = 6


def _matches(token: str, haystack: str) -> bool:
    return token in haystack or (len(token) >= _STEM_LEN and token[:_STEM_LEN] in haystack)


def _score(entry: dict, query_tokens: list[str]) -> float:
    """Score a catalog entry against the query. Higher is more relevant."""
    title_hay = f"{entry['title']} {entry['description']}".lower()
    topic_hay = entry['topics'].lower()
    topic_weight = _BROAD_TOPIC_WEIGHT if entry['general_debate'] else _TOPIC_WEIGHT

    score = 0.0
    for token in query_tokens:
        if _matches(token, title_hay):
            score += _TITLE_WEIGHT
        elif _matches(token, topic_hay):
            score += topic_weight

    # A general-debate record carries statements on every topic, so it stays a
    # usable grounding source even when nothing matches literally. Without this
    # floor an unusual topic would find no UN document at all and fall through
    # to Wikipedia -- the exact failure this module exists to end.
    if entry['general_debate']:
        score += _GENERAL_DEBATE_FLOOR

    return score


def search_catalog(query: str = '', domain: str = '', un_lang: str = 'eng',
                   limit: int = 8, domain_keywords: str = '') -> list[dict]:
    """
    Rank the catalog against a topic and return results shaped exactly like the
    old _search_un_api() output, so callers need no changes.

    `domain_keywords` is the caller's DOMAIN_QUERIES[domain] expansion; it is
    optional so this module does not import module_library (circular import).
    """
    query_tokens = _tokens(f'{query} {domain} {domain_keywords}')

    scored = [(_score(entry, query_tokens), entry) for entry in UN_CATALOG]
    scored = [(s, e) for s, e in scored if s > 0]
    if not scored:
        return []

    # Shuffle before the sort so entries on the same score come back in a
    # different order each time -- the professor asked that regenerating the
    # same topic bring fresh material rather than the identical document.
    random.shuffle(scored)
    scored.sort(key=lambda pair: pair[0], reverse=True)

    results = []
    for _, entry in scored[:limit]:
        results.append({
            'un_id':       entry['symbol'],
            'title':       entry['title'],
            'date':        entry['date'],
            'languages':   ['eng', 'fre', 'ara'],
            'web_url':     symbol_web_url(entry['symbol']),
            'pdf_url':     symbol_pdf_url(entry['symbol'], un_lang),
            'description': entry['description'],
        })
    return results


# ── Extracted-text cache ─────────────────────────────────────────────────────
# These verbatim records are ~1 MB PDFs of 50-80 pages, and a cold fetch plus
# extraction measured 27-90 seconds on a normal connection. That is far too
# long to sit in front of a student pressing "generate", and with a whole class
# working through the same handful of catalogued documents it is also the same
# work repeated over and over. The catalog is fixed, published material that
# never changes, so extracted text is cached on disk and reused.
#
# On Hugging Face Spaces the cache lives for the life of the container: the
# first student to touch a document pays the download, everyone after that gets
# it instantly. Set UN_CACHE_DIR to relocate it.
_CACHE_DIR = os.getenv('UN_CACHE_DIR') or os.path.join(tempfile.gettempdir(), 'etib_un_cache')
_CACHE_LOCK = threading.Lock()
# Below this, extraction clearly failed; don't poison the cache with it.
_MIN_CACHEABLE_CHARS = 500


def _cache_path(symbol: str, un_lang: str) -> str:
    key = hashlib.sha1(f'{symbol}|{un_lang}'.encode('utf-8')).hexdigest()[:16]
    safe = re.sub(r'[^A-Za-z0-9]+', '_', symbol).strip('_')
    return os.path.join(_CACHE_DIR, f'{safe}_{key}.txt')


def fetch_symbol_text(symbol: str, un_lang: str, extractor, timeout: int = 45) -> str:
    """
    Return the extracted text of a catalogued UN document, from cache when
    possible.

    `extractor` is the caller's download-and-extract callable (it takes a URL
    and a timeout). Passing it in keeps this module free of a module_library
    import, which would be circular.
    """
    path = _cache_path(symbol, un_lang)

    try:
        if os.path.exists(path):
            with open(path, encoding='utf-8') as handle:
                cached = handle.read()
            if len(cached) >= _MIN_CACHEABLE_CHARS:
                return cached
    except OSError:
        pass  # An unreadable cache is not an error; just fetch it again.

    text = extractor(symbol_pdf_url(symbol, un_lang), timeout=timeout)

    if text and len(text) >= _MIN_CACHEABLE_CHARS:
        try:
            with _CACHE_LOCK:
                os.makedirs(_CACHE_DIR, exist_ok=True)
                # Write to a temp file and replace, so a crash or two workers
                # racing can never leave a half-written document in the cache.
                tmp = f'{path}.{os.getpid()}.tmp'
                with open(tmp, 'w', encoding='utf-8') as handle:
                    handle.write(text)
                os.replace(tmp, path)
        except OSError:
            pass  # Read-only filesystem: still return the text, just uncached.

    return text


def prewarm_cache(extractor, symbols: list[str] | None = None,
                  un_lang: str = 'eng', timeout: int = 45) -> None:
    """
    Fill the cache in the background so the first student of the day does not
    pay the cold-fetch cost (measured at 17-90 seconds).

    Call this from a daemon thread at startup -- it must never block boot, and
    every failure is swallowed: a cold cache is slow, not broken.
    """
    import logging
    log = logging.getLogger(__name__)

    if symbols is None:
        # The topic-specific Security Council records plus a couple of general
        # debate sittings: between them they answer most queries, and warming
        # all 39 would hammer documents.un.org on every container start.
        symbols = [entry['symbol'] for entry in UN_CATALOG
                   if entry['body'] == 'Security Council']
        symbols += ['A/79/PV.3', 'A/78/PV.3']

    for symbol in symbols:
        try:
            if os.path.exists(_cache_path(symbol, un_lang)):
                continue
            fetch_symbol_text(symbol, un_lang, extractor, timeout=timeout)
            log.info('[UN cache] warmed %s', symbol)
        except Exception as exc:
            log.warning('[UN cache] could not warm %s: %s', symbol, exc)


def validate_catalog(un_lang: str = 'eng', timeout: int = 30) -> list[tuple[str, str]]:
    """
    Check every catalogued symbol still serves a real PDF. Not called at
    runtime -- run it by hand after adding entries:

        python -c "from utils.un_documents import validate_catalog as v; print(v())"

    Returns a list of (symbol, problem) for anything broken; empty means healthy.
    """
    import requests

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0'}
    problems: list[tuple[str, str]] = []

    for entry in UN_CATALOG:
        url = symbol_pdf_url(entry['symbol'], un_lang)
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
        except Exception as exc:
            problems.append((entry['symbol'], f'request failed: {exc}'))
            continue
        if resp.status_code != 200:
            problems.append((entry['symbol'], f'HTTP {resp.status_code}'))
        elif len(resp.content) < MIN_PDF_BYTES:
            # The tell-tale unpublished-symbol response: 200 with nothing in it.
            problems.append((entry['symbol'], f'empty body ({len(resp.content)} bytes)'))

    return problems
