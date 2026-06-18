"""Production dense RAG wiring tests for the ui-orange Module A branch."""
from io import BytesIO
from types import SimpleNamespace
import sys

import pytest

from app import app
from utils import document_grounding
from utils import embedding_retrieval
from utils.embedding_retrieval import (
    DENSE_RETRIEVAL_METHOD,
    KEYWORD_FALLBACK_RETRIEVAL_METHOD,
    DenseEmbeddingUnavailable,
    _load_sentence_transformer,
    select_production_relevant_chunks,
)


LONG_ENGLISH_TEXT = (
    'Climate finance supports developing countries through renewable energy '
    'investment, technology transfer, solar and wind training, and transparent '
    'reporting for vulnerable communities. '
) * 3

LONG_FRENCH_TEXT = (
    'Organisation mondiale de la Sante, vaccination, prevention, financement '
    'des systemes de sante et protection des populations vulnerables. '
) * 3

LONG_ARABIC_TEXT = (
    'أكدت الجامعة العربية أن التعاون الإقليمي والمفاوضات الهادئة ضرورية '
    'لحماية الأمن الغذائي ودعم التنمية المستدامة في المنطقة. '
) * 4


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as test_client:
        yield test_client


def _txt_upload(text, filename):
    return BytesIO(text.encode('utf-8')), filename


def _dense_selector(records, params, max_excerpts, max_total_characters, logger=None):
    return [{
        'text': records[0]['text'],
        'source_filename': records[0]['source_filename'],
        'source_type': records[0]['source_type'],
        'chunk_index': records[0]['chunk_index'],
        'score': 0.91,
        'retrieval_method': DENSE_RETRIEVAL_METHOD,
    }]


def _fake_generation_text():
    words = ' '.join(['generated'] * 120)
    return (
        '{"script": "' + words + '", '
        '"summary": "Dense summary.", '
        '"mcqs": [{"question": "What was discussed?", "options": ["A. Climate", "B. Sport"], "answer": "A"}], '
        '"glossary": [{"term": "climate finance", "arabic": "تمويل المناخ", '
        '"french": "financement climatique", "english": "climate finance", '
        '"definition": "Funding for climate action."}], '
        '"metadata": {"pressure_factors": []}}'
    )


def test_production_wrapper_uses_dense_when_dense_succeeds(monkeypatch):
    monkeypatch.setattr(
        embedding_retrieval,
        'select_relevant_chunks_dense_with_metadata',
        lambda records, params, max_excerpts, max_total_characters: [{
            'text': records[0]['text'],
            'source_filename': 'sample.txt',
            'source_type': '.txt',
            'chunk_index': 0,
            'score': 0.8,
            'retrieval_method': DENSE_RETRIEVAL_METHOD,
        }],
    )

    selected = select_production_relevant_chunks(
        [{'text': 'climate finance', 'source_order': 0, 'chunk_index': 0}],
        {'query': 'climate'},
        max_excerpts=1,
        max_total_characters=1000,
    )

    assert selected[0]['retrieval_method'] == DENSE_RETRIEVAL_METHOD


def test_production_wrapper_falls_back_when_dense_unavailable(monkeypatch, caplog):
    def raise_unavailable(*args, **kwargs):
        raise DenseEmbeddingUnavailable('model unavailable')

    monkeypatch.setattr(
        embedding_retrieval,
        'select_relevant_chunks_dense_with_metadata',
        raise_unavailable,
    )
    monkeypatch.setattr(
        document_grounding,
        'select_relevant_chunks_with_metadata',
        lambda records, params, max_excerpts, max_total_characters: [{
            'text': records[0]['text'],
            'source_filename': 'fallback.txt',
            'source_type': '.txt',
            'chunk_index': 0,
            'score': 1,
            'retrieval_method': 'keyword_overlap',
        }],
    )

    selected = select_production_relevant_chunks(
        [{'text': 'fallback context', 'source_order': 0, 'chunk_index': 0}],
        {'query': 'context'},
        max_excerpts=1,
        max_total_characters=1000,
    )

    assert selected[0]['retrieval_method'] == KEYWORD_FALLBACK_RETRIEVAL_METHOD
    assert 'Dense multilingual retrieval failed' in caplog.text


def test_sentence_transformer_loader_is_cached(monkeypatch):
    _load_sentence_transformer.cache_clear()
    calls = []

    class FakeSentenceTransformer:
        def __init__(self, model_name):
            calls.append(model_name)

    monkeypatch.setitem(
        sys.modules,
        'sentence_transformers',
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    first = _load_sentence_transformer('fake-model')
    second = _load_sentence_transformer('fake-model')

    assert first is second
    assert calls == ['fake-model']
    _load_sentence_transformer.cache_clear()


def test_retrieve_document_context_uses_dense_without_selector(monkeypatch, client):
    import modules.module_a as module_a

    monkeypatch.setattr(module_a, 'select_production_relevant_chunks', _dense_selector)
    monkeypatch.setattr(
        module_a,
        'generate_text',
        lambda **kwargs: (_ for _ in ()).throw(AssertionError('LLM should not be called')),
    )

    response = client.post('/api/module-a/retrieve-document-context', data={
        'document': _txt_upload(LONG_ENGLISH_TEXT, 'climate.txt'),
        'query': 'renewable energy',
        'language': 'en',
    })

    assert response.status_code == 200
    data = response.get_json()
    assert data['mode'] == 'retrieval_only'
    assert data['query_used'] == 'renewable energy'
    assert data['selected_chunk_count'] == 1
    assert data['selected_chunks'][0]['retrieval_method'] == DENSE_RETRIEVAL_METHOD
    assert 'script' not in data
    assert {'mode', 'query_used', 'selected_chunks', 'documents_processed',
            'document_errors', 'selected_chunk_count'}.issubset(data)


def test_retrieve_document_context_handles_multiple_documents(monkeypatch, client):
    import modules.module_a as module_a

    captured = {}

    def fake_selector(records, params, max_excerpts, max_total_characters, logger=None):
        captured['filenames'] = [record['source_filename'] for record in records]
        return [{
            'text': records[-1]['text'],
            'source_filename': records[-1]['source_filename'],
            'source_type': records[-1]['source_type'],
            'chunk_index': records[-1]['chunk_index'],
            'score': 0.9,
            'retrieval_method': DENSE_RETRIEVAL_METHOD,
        }]

    monkeypatch.setattr(module_a, 'select_production_relevant_chunks', fake_selector)

    response = client.post('/api/module-a/retrieve-document-context', data={
        'documents': [
            _txt_upload(LONG_ENGLISH_TEXT, 'english.txt'),
            _txt_upload(LONG_FRENCH_TEXT, 'french.txt'),
        ],
        'query': 'sante',
        'language': 'fr',
    })

    assert response.status_code == 200
    assert set(captured['filenames']) == {'english.txt', 'french.txt'}
    assert response.get_json()['selected_chunks'][0]['source_filename'] == 'french.txt'


@pytest.mark.parametrize(
    ('text', 'query', 'language'),
    [
        (LONG_ENGLISH_TEXT, 'climate finance', 'en'),
        (LONG_FRENCH_TEXT, 'vaccination prevention', 'fr'),
        (LONG_ARABIC_TEXT, 'الأمن الغذائي', 'ar'),
    ],
)
def test_retrieve_document_context_language_cases(monkeypatch, client, text, query, language):
    import modules.module_a as module_a

    monkeypatch.setattr(module_a, 'select_production_relevant_chunks', _dense_selector)

    response = client.post('/api/module-a/retrieve-document-context', data={
        'document': _txt_upload(text, f'{language}.txt'),
        'query': query,
        'language': language,
    })

    assert response.status_code == 200
    assert response.get_json()['selected_chunks'][0]['retrieval_method'] == DENSE_RETRIEVAL_METHOD


def test_retrieve_document_context_reports_hidden_fallback(monkeypatch, client):
    import modules.module_a as module_a

    def fallback_selector(records, params, max_excerpts, max_total_characters, logger=None):
        return [{
            'text': records[0]['text'],
            'source_filename': records[0]['source_filename'],
            'source_type': records[0]['source_type'],
            'chunk_index': records[0]['chunk_index'],
            'score': 0.3,
            'retrieval_method': KEYWORD_FALLBACK_RETRIEVAL_METHOD,
        }]

    monkeypatch.setattr(module_a, 'select_production_relevant_chunks', fallback_selector)

    response = client.post('/api/module-a/retrieve-document-context', data={
        'document': _txt_upload(LONG_ENGLISH_TEXT, 'fallback.txt'),
        'query': 'climate',
    })

    data = response.get_json()
    assert response.status_code == 200
    assert data['fallback_used'] is True
    assert 'retrieval_warning' in data
    assert data['selected_chunks'][0]['retrieval_method'] == KEYWORD_FALLBACK_RETRIEVAL_METHOD


def test_from_document_passes_dense_selected_context_to_ui_orange_generation(monkeypatch, client):
    import modules.module_a as module_a

    captured = {}

    def fake_selector(records, params, max_excerpts, max_total_characters, logger=None):
        return [{
            'text': 'DENSE SELECTED CONTEXT',
            'source_filename': records[0]['source_filename'],
            'source_type': records[0]['source_type'],
            'chunk_index': 0,
            'score': 0.9,
            'retrieval_method': DENSE_RETRIEVAL_METHOD,
        }]

    def fake_prompt(params, topic, excerpts=None):
        captured['excerpts'] = excerpts
        captured['topic'] = topic
        return 'prompt from dense context'

    monkeypatch.setattr(module_a, 'select_production_relevant_chunks', fake_selector)
    monkeypatch.setattr(module_a, 'build_structured_material_prompt', fake_prompt)
    monkeypatch.setattr(module_a, 'generate_text', lambda **kwargs: _fake_generation_text())

    response = client.post('/api/module-a/from-document', data={
        'document': _txt_upload(LONG_ENGLISH_TEXT, 'source.txt'),
        'language': 'en',
        'domain': 'climate',
        'word_count': '100',
    })

    assert response.status_code == 200
    data = response.get_json()
    assert captured['excerpts'] == ['DENSE SELECTED CONTEXT']
    assert captured['topic'] == 'Document-grounded speech from source.txt'
    assert data['selected_excerpt_count'] == 1
    assert data['mode'] == 'document_grounded'


def test_from_document_reports_fallback_truthfully(monkeypatch, client):
    import modules.module_a as module_a

    def fallback_selector(records, params, max_excerpts, max_total_characters, logger=None):
        return [{
            'text': 'FALLBACK CONTEXT',
            'source_filename': records[0]['source_filename'],
            'source_type': records[0]['source_type'],
            'chunk_index': 0,
            'score': 0.2,
            'retrieval_method': KEYWORD_FALLBACK_RETRIEVAL_METHOD,
        }]

    monkeypatch.setattr(module_a, 'select_production_relevant_chunks', fallback_selector)
    monkeypatch.setattr(module_a, 'generate_text', lambda **kwargs: _fake_generation_text())

    response = client.post('/api/module-a/from-document', data={
        'document': _txt_upload(LONG_ENGLISH_TEXT, 'source.txt'),
        'language': 'en',
        'word_count': '100',
    })

    data = response.get_json()
    assert response.status_code == 200
    assert data['fallback_used'] is True
    assert 'retrieval_warning' in data


def test_generate_preserves_un_auto_grounding_with_dense_selection(monkeypatch, client):
    import modules.module_a as module_a

    captured = {}

    def fake_selector(records, params, max_excerpts, max_total_characters, logger=None):
        captured['records'] = records
        return [{
            'text': 'UN DENSE SELECTED CONTEXT',
            'source_filename': records[0]['source_filename'],
            'source_type': records[0]['source_type'],
            'chunk_index': 0,
            'score': 0.95,
            'retrieval_method': DENSE_RETRIEVAL_METHOD,
        }]

    def fake_prompt(params, topic, excerpts=None):
        captured['excerpts'] = excerpts
        return 'prompt with UN dense context'

    monkeypatch.setattr(module_a, 'find_un_grounding_source', lambda params: {
        'text': LONG_ENGLISH_TEXT,
        'title': 'UN climate source',
        'un_id': 'A/TEST/1',
        'web_url': 'https://example.test/un',
        'pdf_url': 'https://example.test/un.pdf',
        'date': '2026',
        'query': 'climate finance',
    })
    monkeypatch.setattr(module_a, 'select_production_relevant_chunks', fake_selector)
    monkeypatch.setattr(module_a, 'build_structured_material_prompt', fake_prompt)
    monkeypatch.setattr(module_a, 'generate_text', lambda **kwargs: _fake_generation_text())

    response = client.post('/api/module-a/generate', json={
        'topic': 'Climate finance',
        'language': 'en',
        'target_language': 'fr',
        'domain': 'climate',
        'word_count': 100,
    })

    assert response.status_code == 200
    data = response.get_json()
    assert data['mode'] == 'un_library_grounded'
    assert data['source_speech']['un_id'] == 'A/TEST/1'
    assert captured['excerpts'] == ['UN DENSE SELECTED CONTEXT']
    assert captured['records'][0]['source_type'] == 'un_library'
