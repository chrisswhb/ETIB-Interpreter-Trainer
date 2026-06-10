"""
Module Library — UN Digital Library Integration
================================================
Searches the UN Digital Library (digitallibrary.un.org) for real official
speeches, downloads the PDF, extracts the text, and returns it ready to be
used as grounding material in Module A speech generation.

Endpoints:
  GET  /api/library/search        — search UN Digital Library
  POST /api/library/fetch         — download + extract text from a UN document
  GET  /api/library/saved         — list cached speeches from MongoDB
  POST /api/library/generate      — generate a training speech grounded in a saved UN speech
"""
import io
import re
import hashlib
import subprocess
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify

from services.llm_service import generate_text

module_library_bp = Blueprint('module_library', __name__)

# ── UN Digital Library endpoints ─────────────────────────────────────────────
UN_BASE          = 'https://digitallibrary.un.org'
UN_SEARCH_URL    = f'{UN_BASE}/search'

MARC_NS = '{http://www.loc.gov/MARC21/slim}'

HEADERS = {
    'User-Agent': 'ETIB-Interpreter-Trainer/1.0 (academic research; contact: etib@usj.edu.lb)',
    'Accept': 'application/xml',
}

# Language codes: app code → UN ISO 639-2
LANG_MAP = {'ar': 'ara', 'fr': 'fre', 'en': 'eng'}

# UN ISO 639-2 → file suffix used in UN Digital Library PDF filenames
LANG_FILE_SUFFIX = {'ara': 'AR', 'eng': 'EN', 'fre': 'FR', 'rus': 'RU', 'spa': 'ES', 'chi': 'ZH'}

# Domain → UN subject keywords
DOMAIN_QUERIES = {
    'climate':     'climate change emissions renewable energy sustainability',
    'politics':    'general assembly political declaration summit',
    'diplomacy':   'diplomatic relations international peace security council',
    'economics':   'economic development trade finance growth GDP',
    'health':      'health pandemic disease WHO public health',
    'human rights':'human rights humanitarian law protection refugees',
    'education':   'education UNESCO literacy development',
}

FETCH_TIMEOUT = 30   # seconds for API calls
PDF_TIMEOUT   = 60   # seconds for PDF downloads
MAX_PAGES     = 8    # max PDF pages to extract (speeches rarely exceed this)


def _curl_get(url: str, params: dict, accept: str = 'application/xml') -> str | None:
    """
    GET a URL via the system curl binary.

    digitallibrary.un.org sits behind an AWS WAF that bot-challenges
    requests made with python-requests' TLS fingerprint (returns an
    empty 202 response), but allows plain curl through. Shelling out
    to curl is the simplest reliable workaround.
    """
    query = '&'.join(f'{k}={requests.utils.quote(str(v), safe="")}' for k, v in params.items())
    full_url = f'{url}?{query}'
    try:
        result = subprocess.run(
            ['curl', '-s', '--max-time', str(FETCH_TIMEOUT),
             '-H', f'Accept: {accept}', full_url],
            capture_output=True, timeout=FETCH_TIMEOUT + 5,
        )
        if result.returncode != 0:
            print(f'[Library] curl error (exit {result.returncode}): {result.stderr.decode(errors="ignore")[:200]}')
            return None
        return result.stdout.decode('utf-8', errors='replace')
    except (subprocess.SubprocessError, OSError) as exc:
        print(f'[Library] curl subprocess failed: {exc}')
        return None


# ── MongoDB helpers (optional — graceful if Mongo not available) ─────────────

def _get_db():
    try:
        from pymongo import MongoClient
        from config import MONGODB_URI, MONGODB_DB
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=2000)
        client.admin.command('ping')
        return client[MONGODB_DB]['library_speeches']
    except Exception:
        return None


def _save_speech(doc: dict):
    col = _get_db()
    if col is None:
        return
    doc['saved_at'] = datetime.now(timezone.utc).isoformat()
    col.update_one({'un_id': doc['un_id']}, {'$set': doc}, upsert=True)


def _load_saved(limit: int = 50) -> list:
    col = _get_db()
    if col is None:
        return []
    return list(col.find({}, {'_id': 0}).sort('saved_at', -1).limit(limit))


# ── Search ───────────────────────────────────────────────────────────────────

@module_library_bp.route('/search', methods=['GET'])
def search_un_library():
    """
    Search the UN Digital Library for speeches.

    Query params:
      q        str   topic / keywords          required
      language str   'ar'|'fr'|'en'            default 'ar'
      domain   str   domain keyword preset      optional
      limit    int   max results (1–20)         default 10
    """
    q        = request.args.get('q', '').strip()
    language = request.args.get('language', 'ar')
    domain   = request.args.get('domain', '')
    limit    = min(int(request.args.get('limit', 10)), 20)

    if not q and not domain:
        return jsonify({'error': 'q or domain is required'}), 400

    # The UN Digital Library catalog is indexed in English/French — Arabic
    # search terms match nothing, so translate Arabic queries before searching.
    if q and _looks_like_arabic(q):
        q = _translate_query_to_english(q)

    # Build search query
    query_parts = []
    if q:
        query_parts.append(q)
    if domain and domain in DOMAIN_QUERIES:
        query_parts.append(DOMAIN_QUERIES[domain])

    un_lang = LANG_MAP.get(language, 'ara')
    full_query = ' '.join(query_parts)

    # Try the UN Digital Library JSON API
    results = _search_un_api(full_query, un_lang, limit)

    if not results:
        # Fallback: try English query to get at least something
        results = _search_un_api(full_query, 'eng', limit)

    return jsonify({
        'query':    full_query,
        'language': language,
        'count':    len(results),
        'results':  results,
    })


def _search_un_api(query: str, un_lang: str, limit: int) -> list:
    """Query the UN Digital Library legacy search and normalise the response."""
    common_params = {
        'p':  query,
        'rg': limit,
        'ln': un_lang,
        'sf': 'date',
        'so': 'd',               # sort descending by date
    }

    # MARCXML gives us proper title/date/language/abstract fields
    xml_text = _curl_get(UN_SEARCH_URL, {**common_params, 'of': 'xm'}, accept='application/xml')
    if not xml_text:
        print('[Library] UN search returned no data')
        return []
    try:
        records = _parse_marcxml(xml_text)
    except ET.ParseError:
        print('[Library] UN search returned invalid XML')
        return []

    if not records:
        return []

    # recjson gives us the downloadable file links per record
    files_by_recid = {}
    json_text = _curl_get(UN_SEARCH_URL, {**common_params, 'of': 'recjson'}, accept='application/json')
    if json_text:
        try:
            import json as _json
            for item in _json.loads(json_text):
                recid = str(item.get('recid', ''))
                if recid:
                    files_by_recid[recid] = item.get('files', [])
        except ValueError as exc:
            print(f'[Library] UN recjson parse error: {exc}')

    results = []
    for rec in records:
        files = files_by_recid.get(rec['recid'], [])
        pdf_url = _extract_pdf_url(files, un_lang)
        if not pdf_url:
            continue
        results.append({
            'un_id':       rec['recid'],
            'title':       rec['title'] or 'Untitled UN Document',
            'date':        rec['date'],
            'languages':   rec['languages'],
            'web_url':     f'{UN_BASE}/record/{rec["recid"]}',
            'pdf_url':     pdf_url,
            'description': rec['abstract'][:300],
        })

    return results


def _parse_marcxml(xml_text: str) -> list:
    """Parse a MARCXML search response into a list of record dicts."""
    root = ET.fromstring(xml_text)
    records = []

    for record in root.findall(f'{MARC_NS}record'):
        recid = ''
        for cf in record.findall(f'{MARC_NS}controlfield'):
            if cf.get('tag') == '001':
                recid = (cf.text or '').strip()

        if not recid:
            continue

        title_parts = []
        date = ''
        languages = []
        abstract = ''

        for df in record.findall(f'{MARC_NS}datafield'):
            tag = df.get('tag')
            if tag == '245':
                for sf in df.findall(f'{MARC_NS}subfield'):
                    if sf.get('code') in ('a', 'b') and sf.text:
                        title_parts.append(sf.text.strip())
            elif tag in ('269', '260') and not date:
                for sf in df.findall(f'{MARC_NS}subfield'):
                    if sf.get('code') in ('a', 'c') and sf.text:
                        date = sf.text.strip()
            elif tag == '041':
                for sf in df.findall(f'{MARC_NS}subfield'):
                    if sf.get('code') == 'a' and sf.text:
                        lang_str = sf.text.strip()
                        languages = [lang_str[i:i + 3] for i in range(0, len(lang_str), 3)]
            elif tag == '520':
                for sf in df.findall(f'{MARC_NS}subfield'):
                    if sf.get('code') == 'a' and sf.text:
                        abstract = sf.text.strip()

        title = ' '.join(title_parts).strip().rstrip(':/').strip()

        records.append({
            'recid':     recid,
            'title':     title,
            'date':      date,
            'languages': languages,
            'abstract':  abstract,
        })

    return records


def _extract_pdf_url(files: list, un_lang: str) -> str:
    """Pick the PDF link matching the requested language from a recjson 'files' array."""
    suffix = LANG_FILE_SUFFIX.get(un_lang, 'EN')

    # Prefer a file whose name ends with the language suffix (e.g. ..._AR.pdf)
    for f in files:
        if not isinstance(f, dict):
            continue
        name = (f.get('name') or f.get('full_name') or '').upper()
        url = f.get('url', '')
        if url.lower().endswith('.pdf') and name.endswith(f'-{suffix}'):
            return url

    # Fall back to any PDF file
    for f in files:
        if isinstance(f, dict):
            url = f.get('url', '')
            if url.lower().endswith('.pdf'):
                return url

    return ''


# ── Fetch + Extract ──────────────────────────────────────────────────────────

@module_library_bp.route('/fetch', methods=['POST'])
def fetch_un_document():
    """
    Download a UN document PDF and extract its text.

    Request body (JSON):
      pdf_url   str   direct PDF URL from search results   required (or)
      web_url   str   UN record page URL                   required (or)
      un_id     str   UN record ID (will build URL)        optional
      title     str   document title for caching           optional
      language  str   expected language                    default 'ar'

    Response (JSON):
      text      str   extracted speech text
      title     str
      un_id     str
      pdf_url   str
      word_count int
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON body required'}), 400

    pdf_url  = data.get('pdf_url', '').strip()
    web_url  = data.get('web_url', '').strip()
    un_id    = data.get('un_id', '').strip()
    title    = data.get('title', 'UN Document').strip()
    language = data.get('language', 'ar')

    if not pdf_url and not web_url and not un_id:
        return jsonify({'error': 'pdf_url, web_url, or un_id is required'}), 400

    # If only web_url or un_id, try to find the PDF link
    if not pdf_url:
        pdf_url = _resolve_pdf_url(web_url, un_id, language)
        if not pdf_url:
            return jsonify({'error': 'Could not find a PDF link for this document. Try providing pdf_url directly.'}), 400

    # Download and extract
    try:
        text = _download_and_extract(pdf_url)
    except Exception as exc:
        return jsonify({'error': f'PDF extraction failed: {exc}'}), 500

    if not text or len(text.strip()) < 100:
        return jsonify({'error': 'Extracted text is too short — document may be scanned image or inaccessible.'}), 400

    text = _clean_extracted_text(text)
    word_count = len(text.split())

    if language == 'ar' and not _looks_like_arabic(text):
        return jsonify({
            'error': 'This PDF uses a font encoding that could not be decoded into readable Arabic text. '
                     'Please try a different document.'
        }), 422

    # Cache in MongoDB
    speech_doc = {
        'un_id':      un_id or hashlib.md5(pdf_url.encode()).hexdigest()[:10],
        'title':      title,
        'language':   language,
        'pdf_url':    pdf_url,
        'web_url':    web_url,
        'text':       text,
        'word_count': word_count,
    }
    _save_speech(speech_doc)

    return jsonify({
        'un_id':      speech_doc['un_id'],
        'title':      title,
        'pdf_url':    pdf_url,
        'language':   language,
        'text':       text,
        'word_count': word_count,
    })


def _resolve_pdf_url(web_url: str, un_id: str, language: str = 'ar') -> str:
    """Try to find the PDF URL from a record ID via the recjson API."""
    if un_id and un_id.isdigit():
        un_lang = LANG_MAP.get(language, 'ara')
        json_text = _curl_get(UN_SEARCH_URL, {'p': f'recid:{un_id}', 'of': 'recjson'},
                               accept='application/json')
        if json_text:
            try:
                import json as _json
                items = _json.loads(json_text)
                if items:
                    pdf_url = _extract_pdf_url(items[0].get('files', []), un_lang)
                    if pdf_url:
                        return pdf_url
            except ValueError:
                pass

    return ''


def _download_and_extract(pdf_url: str) -> str:
    """Download a PDF (via curl, to avoid the AWS WAF block on python-requests) and extract its text."""
    try:
        result = subprocess.run(
            ['curl', '-sL', '--max-time', str(PDF_TIMEOUT), '-H', 'Accept: application/pdf', pdf_url],
            capture_output=True, timeout=PDF_TIMEOUT + 5,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise ValueError(f'Failed to download PDF: {exc}')

    if result.returncode != 0:
        raise ValueError(f'curl error downloading PDF (exit {result.returncode})')

    pdf_bytes = result.stdout[:10 * 1024 * 1024]
    if not pdf_bytes.startswith(b'%PDF'):
        raise ValueError('URL did not return a valid PDF file.')

    return _extract_text_from_pdf(pdf_bytes)


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber."""
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = pdf.pages[:MAX_PAGES]
            text = '\n'.join(
                page.extract_text() or ''
                for page in pages
            )
        return text
    except ImportError:
        raise RuntimeError('pdfplumber not installed. Run: pip install pdfplumber')
    except Exception as exc:
        raise RuntimeError(f'pdfplumber extraction error: {exc}')


def _clean_extracted_text(text: str) -> str:
    """Clean up extracted PDF text: remove headers/footers, fix whitespace."""
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip obvious header/footer lines (page numbers, document codes)
        if re.match(r'^[\d\s\-–]+$', line):
            continue
        if re.match(r'^(A/|S/|E/|ST/|DP/)\d', line):   # UN doc codes
            continue
        if len(line) < 4:
            continue
        cleaned.append(line)

    return '\n'.join(cleaned)


def _looks_like_arabic(text: str, threshold: float = 0.2) -> bool:
    """
    Heuristic check that extracted text is real Arabic, not mojibake.

    Some old UN PDFs embed Arabic with custom CID fonts that pdfplumber
    decodes into garbage Latin-1/CJK characters. If fewer than `threshold`
    of the non-space characters fall in the Arabic Unicode block, treat
    the extraction as failed.
    """
    letters = [c for c in text if not c.isspace()]
    if not letters:
        return False
    arabic = sum(1 for c in letters if '؀' <= c <= 'ۿ' or 'ݐ' <= c <= 'ݿ')
    return (arabic / len(letters)) >= threshold


def _translate_query_to_english(query: str) -> str:
    """Translate an Arabic search query into English keywords for the UN catalog."""
    try:
        translation = generate_text(
            messages=[
                {
                    'role': 'system',
                    'content': (
                        'Translate the user\'s search query into a short English keyword '
                        'phrase suitable for searching a UN document catalog. '
                        'Reply with only the translated keywords, nothing else.'
                    ),
                },
                {'role': 'user', 'content': query},
            ],
            max_tokens=50,
            temperature=0,
        )
        translated = translation.strip().strip('"').strip()
        return translated or query
    except Exception:
        return query


# ── Saved speeches ───────────────────────────────────────────────────────────

@module_library_bp.route('/saved', methods=['GET'])
def get_saved_speeches():
    """List speeches previously fetched and cached in MongoDB."""
    language = request.args.get('language', '')
    domain   = request.args.get('domain', '')
    saved    = _load_saved(limit=50)

    if language:
        saved = [s for s in saved if s.get('language') == language]
    if domain:
        saved = [s for s in saved if domain.lower() in s.get('title', '').lower()]

    # Return without full text (too large for list view)
    summary = [
        {k: v for k, v in s.items() if k != 'text'}
        for s in saved
    ]
    return jsonify({'saved': summary, 'count': len(summary)})


@module_library_bp.route('/saved/<un_id>', methods=['GET'])
def get_saved_speech(un_id: str):
    """Retrieve a single cached speech including its full text."""
    col = _get_db()
    if col is None:
        return jsonify({'error': 'Database not available'}), 503

    doc = col.find_one({'un_id': un_id}, {'_id': 0})
    if not doc:
        return jsonify({'error': 'Speech not found'}), 404

    return jsonify(doc)
