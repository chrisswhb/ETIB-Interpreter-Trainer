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
import os
import re
import hashlib
import shutil
import subprocess
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify

from services.llm_service import generate_text
from utils.un_documents import UN_CATALOG, fetch_symbol_text, search_catalog

# curl_cffi: impersonates a real browser TLS fingerprint — bypasses AWS WAF
try:
    from curl_cffi import requests as cffi_requests
    _CFFI_AVAILABLE = True
except ImportError:
    _CFFI_AVAILABLE = False
    print('[Library] curl_cffi not available — WAF bypass disabled')

module_library_bp = Blueprint('module_library', __name__)

# Symbols the catalog knows, so /fetch can send those through the shared cache.
_CATALOG_SYMBOLS = {entry['symbol'] for entry in UN_CATALOG}

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
    'technology':  'artificial intelligence digital technology cybersecurity innovation',
    'migration':   'migration refugees displaced persons UNHCR asylum',
    'disarmament': 'disarmament nuclear weapons arms control non-proliferation',
    'women':       'gender equality women empowerment',
    'food':        'food security hunger malnutrition FAO WFP',
    # Added per professional-interpreter feedback (18/20 July): legal and
    # medical are core community/conference interpretation domains missing
    # from the original 12.
    'legal':       'international law legal framework treaty convention justice tribunal',
    'medical':     'health medical care public health WHO patient healthcare system',
}

FETCH_TIMEOUT = 30   # seconds for API calls
PDF_TIMEOUT   = 60   # seconds for PDF downloads
MAX_PAGES     = 8    # max PDF pages to extract (speeches rarely exceed this)


def _find_curl() -> str | None:
    """Return the path to the curl executable, or None if unavailable."""
    found = shutil.which('curl')
    if found:
        return found
    for p in [r'C:\Windows\System32\curl.exe', r'C:\Windows\SysWOW64\curl.exe']:
        if os.path.exists(p):
            return p
    return None

_CURL_EXE: str | None = _find_curl()


def _curl_get(url: str, params: dict, accept: str = 'application/xml') -> str | None:
    """
    GET a URL, trying methods in order:
      1. curl_cffi  — impersonates Chrome TLS fingerprint, bypasses AWS WAF
      2. subprocess curl  — system curl binary (also bypasses WAF via TLS)
      3. python-requests  — plain HTTP, may be WAF-blocked on cloud IPs
    """
    query = '&'.join(f'{k}={requests.utils.quote(str(v), safe="")}' for k, v in params.items())
    full_url = f'{url}?{query}'

    # 1. curl_cffi — best WAF bypass via browser TLS impersonation
    if _CFFI_AVAILABLE:
        try:
            resp = cffi_requests.get(
                full_url,
                headers={'Accept': accept, 'Referer': 'https://digitallibrary.un.org/'},
                impersonate='chrome120',
                timeout=FETCH_TIMEOUT,
            )
            if resp.status_code == 200:
                return resp.text
            print(f'[Library] curl_cffi status {resp.status_code}')
        except Exception as exc:
            print(f'[Library] curl_cffi failed: {exc}')

    # 2. subprocess curl — also has a non-Python TLS stack
    if _CURL_EXE:
        try:
            result = subprocess.run(
                [_CURL_EXE, '-s', '--max-time', str(FETCH_TIMEOUT),
                 '-H', f'Accept: {accept}',
                 '-H', 'Referer: https://digitallibrary.un.org/', full_url],
                capture_output=True, timeout=FETCH_TIMEOUT + 5,
            )
            if result.returncode == 0:
                return result.stdout.decode('utf-8', errors='replace')
            print(f'[Library] curl exit {result.returncode}: {result.stderr.decode(errors="ignore")[:200]}')
        except (subprocess.SubprocessError, OSError) as exc:
            print(f'[Library] curl subprocess failed: {exc}')

    # 3. python-requests fallback
    try:
        resp = requests.get(full_url, headers={'Accept': accept, **HEADERS}, timeout=FETCH_TIMEOUT)
        resp.raise_for_status()
        return resp.text
    except Exception as exc:
        print(f'[Library] requests GET error: {exc}')
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


def _delete_saved(un_id: str) -> int:
    col = _get_db()
    if col is None:
        return -1
    result = col.delete_one({'un_id': un_id})
    return result.deleted_count


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

    # Rank the verified symbol catalog and serve the PDFs from
    # documents.un.org. The old _search_un_api() path against
    # digitallibrary.un.org is kept below but no longer called: that host now
    # answers every request with an AWS WAF JavaScript challenge (HTTP 202,
    # x-amzn-waf-action: challenge), so it returned nothing but cost two 30s
    # timeouts on the way. See utils/un_documents.py.
    results = search_catalog(
        query=q, domain=domain, un_lang=un_lang, limit=limit,
        domain_keywords=DOMAIN_QUERIES.get(domain, ''),
    )

    return jsonify({
        'query':    full_query,
        'language': language,
        'count':    len(results),
        'results':  results,
    })


_DEMO_SPEECHES = [
    # ── Climate ──────────────────────────────────────────────────────────────
    {'un_id':'demo-c1','domain':'climate','date':'2024-09-22','languages':['eng','ara','fre'],
     'title':'Statement by the Secretary-General on Climate Action and Global Emissions',
     'web_url':'https://digitallibrary.un.org/record/4055301',
     'pdf_url':'https://digitallibrary.un.org/record/4055301/files/A_79_PV.3-EN.pdf',
     'description':'Address on climate change, emissions reduction targets, renewable energy transition and international cooperation for environmental protection.','demo':True},
    {'un_id':'demo-c2','domain':'climate','date':'2024-06-05','languages':['eng','ara','fre'],
     'title':'UNEP Report: Global Climate Finance and Carbon Markets 2024',
     'web_url':'https://digitallibrary.un.org/record/4048200',
     'pdf_url':'https://digitallibrary.un.org/record/4048200/files/UNEP_2024_climate-EN.pdf',
     'description':'Report on climate finance flows, carbon pricing mechanisms, green bonds and adaptation funding for vulnerable nations.','demo':True},
    # ── Politics ─────────────────────────────────────────────────────────────
    {'un_id':'demo-p1','domain':'politics','date':'2024-10-15','languages':['eng','ara','fre'],
     'title':'Security Council Resolution on the Maintenance of International Peace and Security',
     'web_url':'https://digitallibrary.un.org/record/4056120',
     'pdf_url':'https://digitallibrary.un.org/record/4056120/files/S_RES_2735-EN.pdf',
     'description':'Resolution reaffirming commitment to peaceful settlement of disputes and strengthening multilateral cooperation in conflict prevention.','demo':True},
    {'un_id':'demo-p2','domain':'politics','date':'2024-09-28','languages':['eng','ara','fre'],
     'title':'General Assembly High-Level Debate: Multilateralism and Global Governance Reform',
     'web_url':'https://digitallibrary.un.org/record/4055890',
     'pdf_url':'https://digitallibrary.un.org/record/4055890/files/A_79_PV.8-EN.pdf',
     'description':'General debate statements on UN Security Council reform, veto power, and strengthening multilateral institutions for the 21st century.','demo':True},
    # ── Human Rights ─────────────────────────────────────────────────────────
    {'un_id':'demo-h1','domain':'human rights','date':'2024-07-08','languages':['eng','ara','fre'],
     'title':'Report of the High Commissioner for Human Rights: Situation of Human Rights Defenders',
     'web_url':'https://digitallibrary.un.org/record/4047891',
     'pdf_url':'https://digitallibrary.un.org/record/4047891/files/A_HRC_55_25-EN.pdf',
     'description':'Annual report on human rights defenders worldwide, challenges, threats, and recommendations for states to ensure their protection.','demo':True},
    {'un_id':'demo-h2','domain':'human rights','date':'2024-03-20','languages':['eng','ara','fre'],
     'title':'Human Rights Council Resolution on the Right to Food and Water',
     'web_url':'https://digitallibrary.un.org/record/4040100',
     'pdf_url':'https://digitallibrary.un.org/record/4040100/files/A_HRC_RES_55_8-EN.pdf',
     'description':'Resolution recognizing the human right to safe drinking water and food, calling on states to eliminate hunger and ensure access for all populations.','demo':True},
    # ── Economics ────────────────────────────────────────────────────────────
    {'un_id':'demo-e1','domain':'economics','date':'2024-05-20','languages':['eng','ara','fre'],
     'title':'World Economic Situation and Prospects: Mid-Year Update 2024',
     'web_url':'https://digitallibrary.un.org/record/4043210',
     'pdf_url':'https://digitallibrary.un.org/record/4043210/files/E_2024_58-EN.pdf',
     'description':'Mid-year update on global economic trends, trade finance, GDP growth projections and development financing for developing countries.','demo':True},
    {'un_id':'demo-e2','domain':'economics','date':'2024-04-10','languages':['eng','ara','fre'],
     'title':'UNCTAD Trade and Development Report: Debt, Inequality and Global Finance',
     'web_url':'https://digitallibrary.un.org/record/4041500',
     'pdf_url':'https://digitallibrary.un.org/record/4041500/files/UNCTAD_TDR_2024-EN.pdf',
     'description':'Analysis of sovereign debt crises, financial inequality, trade imbalances and reform proposals for the international monetary system.','demo':True},
    # ── Health ───────────────────────────────────────────────────────────────
    {'un_id':'demo-s1','domain':'health','date':'2024-05-28','languages':['eng','ara','fre'],
     'title':'WHO Director-General Address to the World Health Assembly on Global Health Preparedness',
     'web_url':'https://digitallibrary.un.org/record/4044567',
     'pdf_url':'https://digitallibrary.un.org/record/4044567/files/WHA77_DIV1-EN.pdf',
     'description':'Address on pandemic preparedness, global health security, universal health coverage, and strengthening international health regulations.','demo':True},
    {'un_id':'demo-s2','domain':'health','date':'2024-02-14','languages':['eng','ara','fre'],
     'title':'UN Report on Mental Health and Sustainable Development Goals',
     'web_url':'https://digitallibrary.un.org/record/4037800',
     'pdf_url':'https://digitallibrary.un.org/record/4037800/files/A_78_853-EN.pdf',
     'description':'Report linking mental health investment to SDG progress, addressing stigma, treatment gaps and the global burden of mental illness.','demo':True},
    # ── Education ────────────────────────────────────────────────────────────
    {'un_id':'demo-ed1','domain':'education','date':'2024-03-14','languages':['eng','ara','fre'],
     'title':'UNESCO Report on Education for Sustainable Development: Global Framework',
     'web_url':'https://digitallibrary.un.org/record/4039871',
     'pdf_url':'https://digitallibrary.un.org/record/4039871/files/ED_2024_03-EN.pdf',
     'description':'Framework for integrating education for sustainable development into national curricula and UNESCO literacy programs worldwide.','demo':True},
    {'un_id':'demo-ed2','domain':'education','date':'2024-01-22','languages':['eng','ara','fre'],
     'title':'UNICEF Report on Learning Crisis: Out-of-School Children and Youth',
     'web_url':'https://digitallibrary.un.org/record/4036200',
     'pdf_url':'https://digitallibrary.un.org/record/4036200/files/UNICEF_OOS_2024-EN.pdf',
     'description':'Report on the global learning crisis, barriers to school enrolment for girls, displaced children, and strategies to achieve universal education.','demo':True},
    # ── Migration & Refugees ─────────────────────────────────────────────────
    {'un_id':'demo-m1','domain':'migration','date':'2024-08-12','languages':['eng','ara','fre'],
     'title':'UNHCR Global Trends Report: Forced Displacement and Refugee Protection 2024',
     'web_url':'https://digitallibrary.un.org/record/4050100',
     'pdf_url':'https://digitallibrary.un.org/record/4050100/files/HCR_2024_trends-EN.pdf',
     'description':'Annual report on forced displacement worldwide, refugee statistics, asylum seekers, internally displaced persons and international protection frameworks.','demo':True},
    {'un_id':'demo-m2','domain':'migration','date':'2024-05-15','languages':['eng','ara','fre'],
     'title':'General Assembly Resolution on International Migration and Development',
     'web_url':'https://digitallibrary.un.org/record/4043800',
     'pdf_url':'https://digitallibrary.un.org/record/4043800/files/A_RES_78_232-EN.pdf',
     'description':'Resolution addressing safe and orderly migration, remittances, migrant workers rights and the Global Compact for Migration implementation.','demo':True},
    # ── Disarmament & Nuclear ────────────────────────────────────────────────
    {'un_id':'demo-d1','domain':'disarmament','date':'2024-10-02','languages':['eng','ara','fre'],
     'title':'First Committee Resolution on Nuclear Disarmament and Non-Proliferation',
     'web_url':'https://digitallibrary.un.org/record/4056000',
     'pdf_url':'https://digitallibrary.un.org/record/4056000/files/A_C1_79_L2-EN.pdf',
     'description':'Resolution calling for renewed commitment to nuclear disarmament, NPT review process, and elimination of weapons of mass destruction.','demo':True},
    {'un_id':'demo-d2','domain':'disarmament','date':'2024-06-20','languages':['eng','ara','fre'],
     'title':'Conference on Disarmament: Report on Arms Control and International Security',
     'web_url':'https://digitallibrary.un.org/record/4048500',
     'pdf_url':'https://digitallibrary.un.org/record/4048500/files/CD_2280-EN.pdf',
     'description':'Report on conventional arms control, landmines, cluster munitions, small arms proliferation and the Arms Trade Treaty implementation.','demo':True},
    # ── Women & Gender ───────────────────────────────────────────────────────
    {'un_id':'demo-w1','domain':'women','date':'2024-03-08','languages':['eng','ara','fre'],
     'title':'Commission on the Status of Women: Gender Equality and Empowerment 2024',
     'web_url':'https://digitallibrary.un.org/record/4039500',
     'pdf_url':'https://digitallibrary.un.org/record/4039500/files/E_CN6_2024_L3-EN.pdf',
     'description':'Report on advancing gender equality, eliminating violence against women, closing the gender pay gap and achieving SDG 5 targets.','demo':True},
    {'un_id':'demo-w2','domain':'women','date':'2024-01-30','languages':['eng','ara','fre'],
     'title':'UN Women Report: Women in Peace and Security Processes',
     'web_url':'https://digitallibrary.un.org/record/4036800',
     'pdf_url':'https://digitallibrary.un.org/record/4036800/files/S_2024_98-EN.pdf',
     'description':'Report on women\'s participation in conflict prevention, peacekeeping, post-conflict reconstruction and Security Council Resolution 1325 implementation.','demo':True},
    # ── Technology & AI ──────────────────────────────────────────────────────
    {'un_id':'demo-t1','domain':'technology','date':'2024-09-19','languages':['eng','ara','fre'],
     'title':'Secretary-General Report: Governing Artificial Intelligence for Humanity',
     'web_url':'https://digitallibrary.un.org/record/4055500',
     'pdf_url':'https://digitallibrary.un.org/record/4055500/files/A_79_PV.4-EN.pdf',
     'description':'Report outlining the UN framework for international AI governance, ethics guidelines, risks of autonomous systems and recommendations for a global AI watchdog body.','demo':True},
    {'un_id':'demo-t2','domain':'technology','date':'2024-03-22','languages':['eng','ara','fre'],
     'title':'ITU AI for Good: Artificial Intelligence and Sustainable Development Goals',
     'web_url':'https://digitallibrary.un.org/record/4040500',
     'pdf_url':'https://digitallibrary.un.org/record/4040500/files/ITU_AI4Good_2024-EN.pdf',
     'description':'Report on how AI and emerging digital technologies can accelerate the SDGs in health, education, agriculture and climate action while addressing algorithmic bias and data governance.','demo':True},
    # ── Food & Poverty ───────────────────────────────────────────────────────
    {'un_id':'demo-f1','domain':'food','date':'2024-07-25','languages':['eng','ara','fre'],
     'title':'FAO Report: The State of Food Security and Nutrition in the World 2024',
     'web_url':'https://digitallibrary.un.org/record/4049200',
     'pdf_url':'https://digitallibrary.un.org/record/4049200/files/FAO_SOFI_2024-EN.pdf',
     'description':'Annual report on global hunger, food insecurity, malnutrition, food systems transformation and progress towards zero hunger by 2030.','demo':True},
    {'un_id':'demo-f2','domain':'food','date':'2024-04-18','languages':['eng','ara','fre'],
     'title':'World Food Programme: Emergency Operations and Food Aid Delivery Report',
     'web_url':'https://digitallibrary.un.org/record/4041900',
     'pdf_url':'https://digitallibrary.un.org/record/4041900/files/WFP_EB_2024-EN.pdf',
     'description':'Report on WFP emergency food assistance operations in conflict zones, drought-affected regions and humanitarian crises worldwide.','demo':True},
]

# keyword aliases — map search terms to a demo domain
_DOMAIN_ALIASES = {
    'climate':      ['climat', 'environment', 'emission', 'renewable', 'carbon', 'green', 'ecology', 'warming', 'cop', 'energy'],
    'politics':     ['politi', 'peace', 'security council', 'general assembly', 'governance', 'diplomac', 'sovereign', 'multilateral', 'sanction', 'war', 'conflict', 'ceasefire'],
    'human rights': ['human right', 'humanitarian', 'torture', 'detention', 'freedom', 'dignity', 'protection', 'justice', 'impunity'],
    'economics':    ['econom', 'trade', 'financ', 'gdp', 'growth', 'debt', 'market', 'investment', 'fiscal', 'monetary', 'inflation', 'budget', 'poverty', 'development'],
    'health':       ['health', 'pandemic', 'epidemic', 'disease', 'who ', 'medical', 'mental', 'medicine', 'vaccine', 'hospital', 'nutrition', 'sanit'],
    'education':    ['educat', 'school', 'learning', 'literacy', 'unesco', 'student', 'teacher', 'university', 'curriculum', 'youth'],
    'migration':    ['migra', 'refugee', 'asylum', 'displaced', 'stateless', 'unhcr', 'border', 'smuggling'],
    'disarmament':  ['disarm', 'nuclear', 'weapon', 'arms', 'missile', 'npt', 'proliferation', 'landmine', 'explosive'],
    'women':        ['women', 'gender', 'feminin', 'girl', 'feminist', 'equality', 'empowerment', 'violence against'],
    'food':         ['food', 'hunger', 'famine', 'malnutri', 'fao', 'wfp', 'agriculture', 'farm', 'crop'],
    'technology':   ['technolog', 'artificial intelligence', ' ai ', 'digital', 'cyber', 'internet', 'data', 'algorithm', 'robot', 'automat', 'machine learning', 'innovation', 'itu', 'software', 'platform', 'blockchain'],
}

def _demo_results(q: str, domain: str, limit: int) -> list:
    """Return curated demo UN documents when the live API is unreachable.
    Returns [] when the topic genuinely has no matching documents."""
    # 1. Keyword search takes priority over domain — pad query with spaces so
    #    short tokens like ' ai ' match the bare string "AI" cleanly.
    if q:
        q_lower = q.lower()
        padded  = f' {q_lower} '
        for dom, aliases in _DOMAIN_ALIASES.items():
            if any(alias in padded for alias in aliases):
                return [d for d in _DEMO_SPEECHES if d['domain'] == dom][:limit]

        # Partial match in title or description
        hits = [d for d in _DEMO_SPEECHES
                if q_lower in d['title'].lower() or q_lower in d['description'].lower()]
        if hits:
            return hits[:limit]

    # 2. Fall back to domain hint (e.g. when Sources panel opened with no query)
    if domain:
        exact = [d for d in _DEMO_SPEECHES if d['domain'] == domain]
        return exact[:limit]

    return []


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


# ── Demo document text generation ───────────────────────────────────────────

def _generate_demo_document_text(doc: dict, language: str) -> str:
    """Generate an expanded ~400-word document excerpt for a demo UN document using the LLM."""
    lang_name = {'ar': 'Arabic (Modern Standard Arabic)', 'fr': 'French', 'en': 'English'}.get(language, 'English')
    title      = doc.get('title', '')
    description = doc.get('description', '')
    domain     = doc.get('domain', '')
    date       = doc.get('date', '')
    try:
        text = generate_text(
            messages=[
                {
                    'role': 'system',
                    'content': (
                        f'You write realistic UN document excerpts for interpreter training. '
                        f'Write in {lang_name}. Use formal UN language with specific statistics, '
                        f'named organizations, and concrete policy findings. '
                        f'Output ONLY the document text — no headings, no meta-commentary.'
                    ),
                },
                {
                    'role': 'user',
                    'content': (
                        f'Write a ~400-word excerpt from the following UN document:\n\n'
                        f'Title: {title}\n'
                        f'Date: {date}\n'
                        f'Domain: {domain}\n'
                        f'Summary: {description}\n\n'
                        f'Requirements:\n'
                        f'- Include specific percentages, figures, and dates\n'
                        f'- Reference real UN frameworks, SDGs, or relevant treaties\n'
                        f'- Include 3-4 concrete policy findings or recommendations\n'
                        f'- Write as a formal UN document section, not as a summary or introduction\n'
                        f'- Do NOT start with a heading or document title'
                    ),
                },
            ],
            max_tokens=600,
            temperature=0.5,
        )
        return (text or '').strip() or f'{title}\n\n{description}'
    except Exception:
        return f'{title}\n\n{description}'


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

    pdf_url     = data.get('pdf_url', '').strip()
    web_url     = data.get('web_url', '').strip()
    un_id       = data.get('un_id', '').strip()
    title       = data.get('title', 'UN Document').strip()
    language    = data.get('language', 'ar')
    description = data.get('description', '').strip()

    if not pdf_url and not web_url and not un_id:
        return jsonify({'error': 'pdf_url, web_url, or un_id is required'}), 400

    # ── Demo documents: generate content via LLM (no PDF download needed) ────
    if un_id.startswith('demo-'):
        demo_doc = next((d for d in _DEMO_SPEECHES if d['un_id'] == un_id), None)
        if demo_doc is None:
            demo_doc = {'title': title, 'description': description, 'domain': '', 'date': ''}
        text = _generate_demo_document_text(demo_doc, language)
        word_count = len(text.split())
        speech_doc = {
            'un_id':      un_id,
            'title':      title or demo_doc.get('title', 'UN Document'),
            'language':   language,
            'pdf_url':    demo_doc.get('pdf_url', ''),
            'web_url':    demo_doc.get('web_url', ''),
            'text':       text,
            'word_count': word_count,
        }
        _save_speech(speech_doc)
        return jsonify({
            'un_id':      un_id,
            'title':      speech_doc['title'],
            'pdf_url':    speech_doc['pdf_url'],
            'language':   language,
            'text':       text,
            'word_count': word_count,
        })

    # ── Real documents: download and extract from PDF ─────────────────────────
    if not pdf_url:
        pdf_url = _resolve_pdf_url(web_url, un_id, language)
        if not pdf_url:
            return jsonify({'error': 'Could not find a PDF link for this document. Try providing pdf_url directly.'}), 400

    # Download and extract — fall back to LLM text generation if PDF is unreachable
    try:
        if un_id in _CATALOG_SYMBOLS:
            # Catalogued verbatim record: go through the shared cache, or the
            # student waits out a 17-90s download of a ~1 MB, 50-80 page PDF
            # that another student has very likely already fetched.
            text = fetch_symbol_text(
                un_id, LANG_MAP.get(language, 'eng'), _download_and_extract,
            )
        else:
            text = _download_and_extract(pdf_url)
    except Exception as exc:
        print(f'[Library] PDF download failed for {un_id}: {exc} — generating via LLM')
        text = _generate_demo_document_text(
            {'title': title, 'description': description, 'domain': '', 'date': ''},
            language,
        )
        if not text or len(text.strip()) < 50:
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


_BROWSER_UA = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
    'AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/124.0.0.0 Safari/537.36'
)


def _download_pdf_via_curl(url: str, timeout: int) -> bytes:
    """Download a PDF using the system curl binary."""
    try:
        result = subprocess.run(
            [_CURL_EXE, '-sL',
             '--max-time', str(timeout),
             '--retry', '2', '--retry-delay', '1',
             '-H', f'User-Agent: {_BROWSER_UA}',
             '-H', 'Accept: application/pdf,*/*',
             '-H', 'Accept-Language: en-US,en;q=0.9',
             '-H', 'Referer: https://digitallibrary.un.org/',
             url],
            capture_output=True, timeout=timeout + 10,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise ValueError(f'Failed to download PDF: {exc}')

    if result.returncode != 0:
        raise ValueError(f'curl error downloading PDF (exit {result.returncode})')

    pdf_bytes = result.stdout[:10 * 1024 * 1024]
    if b'%PDF' not in pdf_bytes[:1024]:
        raise ValueError('URL did not return a valid PDF file.')
    start = pdf_bytes.find(b'%PDF')
    return pdf_bytes[start:] if start > 0 else pdf_bytes


def _download_pdf_via_requests(url: str, timeout: int) -> bytes:
    """Download a PDF using python-requests (fallback when curl is absent)."""
    try:
        resp = requests.get(url, headers={
            'User-Agent': _BROWSER_UA,
            'Accept': 'application/pdf,*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://digitallibrary.un.org/',
        }, timeout=timeout, stream=True)
        resp.raise_for_status()
        content = resp.content[:10 * 1024 * 1024]
    except Exception as exc:
        raise ValueError(f'Failed to download PDF: {exc}')

    if b'%PDF' not in content[:1024]:
        raise ValueError('URL did not return a valid PDF file.')
    start = content.find(b'%PDF')
    return content[start:] if start > 0 else content


def _download_pdf_via_cffi(url: str, timeout: int) -> bytes:
    """Download a PDF using curl_cffi (browser TLS impersonation)."""
    resp = cffi_requests.get(
        url,
        headers={
            'User-Agent': _BROWSER_UA,
            'Accept': 'application/pdf,*/*',
            'Referer': 'https://digitallibrary.un.org/',
        },
        impersonate='chrome120',
        timeout=timeout,
    )
    content = resp.content[:10 * 1024 * 1024]
    if b'%PDF' not in content[:1024]:
        raise ValueError('URL did not return a valid PDF file.')
    start = content.find(b'%PDF')
    return content[start:] if start > 0 else content


def _download_and_extract(pdf_url: str, timeout: int = PDF_TIMEOUT) -> str:
    """Download a PDF and extract its text. Try curl_cffi → curl → requests."""
    if _CFFI_AVAILABLE:
        try:
            return _extract_text_from_pdf(_download_pdf_via_cffi(pdf_url, timeout))
        except Exception as exc:
            print(f'[Library] curl_cffi PDF download failed: {exc}')

    if _CURL_EXE:
        try:
            return _extract_text_from_pdf(_download_pdf_via_curl(pdf_url, timeout))
        except Exception as exc:
            print(f'[Library] curl PDF download failed: {exc}')

    return _extract_text_from_pdf(_download_pdf_via_requests(pdf_url, timeout))


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


@module_library_bp.route('/saved/<un_id>', methods=['DELETE'])
def delete_saved_speech(un_id: str):
    """Remove a cached UN speech from the saved list."""
    deleted_count = _delete_saved(un_id)
    if deleted_count < 0:
        return jsonify({'error': 'Database not available'}), 503
    if deleted_count == 0:
        return jsonify({'error': 'Speech not found'}), 404
    return jsonify({'ok': True, 'un_id': un_id})
