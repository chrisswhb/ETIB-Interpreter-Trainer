"""
Tests for local document retrieval helpers.

These tests do not use Flask and do not call any LLM provider.
"""
from pathlib import Path
import sys


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from utils.document_grounding import (  # noqa: E402
    RETRIEVAL_METHOD,
    chunk_text,
    normalize_text,
    score_chunks,
    select_relevant_chunks_with_metadata,
)


FIXTURE_DIR = Path(__file__).parent / 'fixtures' / 'rag'


def fixture_text(filename):
    return (FIXTURE_DIR / filename).read_text(encoding='utf-8')


def records_from_fixture(filename, source_order=0):
    text = normalize_text(fixture_text(filename))
    chunks = chunk_text(text, chunk_size=520, overlap=0)
    return [
        {
            'text': chunk,
            'source_filename': filename,
            'source_type': '.txt',
            'chunk_index': index,
            'source_order': source_order,
        }
        for index, chunk in enumerate(chunks)
    ]


def select(records, params, max_excerpts=1):
    return select_relevant_chunks_with_metadata(
        records,
        params,
        max_excerpts=max_excerpts,
        max_total_characters=5000,
    )


def test_query_terms_strongly_influence_selected_chunks():
    text = normalize_text(fixture_text('climate_en.txt'))
    records = [
        {
            'text': chunk,
            'source_filename': 'climate_en.txt',
            'source_type': '.txt',
            'chunk_index': index,
            'source_order': 0,
        }
        for index, chunk in enumerate(chunk_text(text, chunk_size=280, overlap=0))
    ]

    selected = select(records, {
        'query': 'renewable energy technology transfer',
        'language': 'en',
    })

    text = selected[0]['text'].lower()
    assert 'technology transfer' in text
    assert 'solar and wind energy' in text

    scored = score_chunks(records, {'query': 'renewable energy technology transfer'})
    top_score = max(item['score'] for item in scored)
    irrelevant_score = min(item['score'] for item in scored)
    assert top_score > irrelevant_score


def test_domain_and_scenario_metadata_influence_selection():
    records = records_from_fixture('health_fr.txt')

    selected = select(records, {
        'domain': 'health sante vaccination',
        'scenario': 'Organisation mondiale de la Sante',
        'difficulty': 'intermediate',
        'mode': 'consecutive',
        'language': 'fr',
    })

    text = selected[0]['text'].lower()
    assert 'vaccination' in text
    assert 'systèmes de santé' in text


def test_number_density_high_prefers_chunks_with_numbers():
    records = records_from_fixture('numbers_names_en.txt')

    selected = select(records, {
        'query': 'World Bank United Nations climate resilience',
        'domain': 'finance',
        'number_density': 'high',
        'language': 'en',
    })

    text = selected[0]['text']
    assert '42 percent' in text
    assert '2023' in text
    assert '1.8 billion' in text


def test_selected_chunks_include_required_metadata():
    records = records_from_fixture('climate_en.txt')

    selected = select(records, {'query': 'climate finance'}, max_excerpts=1)
    chunk = selected[0]

    assert set(chunk) == {
        'text',
        'source_filename',
        'source_type',
        'chunk_index',
        'score',
        'retrieval_method',
    }
    assert chunk['source_filename'] == 'climate_en.txt'
    assert chunk['source_type'] == '.txt'
    assert isinstance(chunk['chunk_index'], int)
    assert isinstance(chunk['score'], (int, float))
    assert chunk['retrieval_method'] == RETRIEVAL_METHOD


def test_max_chunk_selection_is_respected():
    records = (
        records_from_fixture('climate_en.txt', source_order=0)
        + records_from_fixture('health_fr.txt', source_order=1)
        + records_from_fixture('numbers_names_en.txt', source_order=2)
    )

    selected = select(records, {'query': 'finance health climate'}, max_excerpts=2)

    assert len(selected) == 2


def test_basic_multilingual_retrieval_english_french_arabic():
    records = (
        records_from_fixture('climate_en.txt', source_order=0)
        + records_from_fixture('health_fr.txt', source_order=1)
        + records_from_fixture('diplomacy_ar.txt', source_order=2)
    )

    english = select(records, {'query': 'climate finance renewable energy', 'language': 'en'})
    french = select(records, {'query': 'vaccination prevention financement', 'language': 'fr'})
    arabic = select(records, {'query': 'الجامعة العربية التعاون الإقليمي المفاوضات', 'language': 'ar'})

    assert 'climate finance' in english[0]['text'].lower()
    assert 'vaccination' in french[0]['text'].lower()
    assert 'الجامعة العربية' in arabic[0]['text']


def test_french_accent_insensitive_matching():
    records = records_from_fixture('health_fr.txt')

    selected = select(records, {
        'query': 'sante prevention financement',
        'language': 'fr',
    })

    text = selected[0]['text'].lower()
    assert 'santé' in text
    assert 'prévention' in text
    assert 'financement' in text


def test_arabic_alef_and_diacritic_normalization():
    records = records_from_fixture('diplomacy_ar.txt')

    selected = select(records, {
        'query': 'الامن الغذائي',
        'language': 'ar',
    })

    assert 'الأَمْن الغذائي' in selected[0]['text']


def test_numbers_percentages_and_decimals_are_preserved():
    records = records_from_fixture('numbers_names_en.txt')

    selected = select(records, {
        'query': '42 2026 18% 1.8',
        'number_density': 'high',
        'language': 'en',
    })

    text = selected[0]['text']
    assert '42 percent' in text
    assert '2026' in text
    assert '18%' in text
    assert '1.8 billion' in text
