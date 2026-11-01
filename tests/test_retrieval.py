from datetime import date
import json
import numpy as np
import pytest
from fastapi.testclient import TestClient
from nexus.api import create_app
from nexus.auth import DEMO_CUSTOMER_PASSWORD, DEMO_OPERATOR_PASSWORD
from nexus.db import initialize
from nexus.embeddings import RetrievalUnavailable
from nexus.retrieval import RetrievalService, bm25_scores, eligible_documents, evaluate, make_chunks, parse_policy, tokenize


class FakeEncoder:
    """Deterministic vectors for logic tests only; not semantic-quality evidence."""
    model_id='unit-test-encoder'
    revision='1'
    dimensions=4
    max_sequence_length=256

    def count(self,text): return len(text.split())+2

    def encode(self,texts):
        if any(self.count(t)>256 for t in texts):
            raise ValueError('Text exceeds the embedding model token limit')
        result=[]
        for text in texts:
            words=tokenize(text)
            vector=np.asarray([1,sum(w in ('payment','refund','invoice','charge') for w in words),
                               sum(w in ('access','feature','subscription') for w in words),
                               sum(w in ('identity','account','policy') for w in words)],dtype=np.float32)
            result.append(vector/np.linalg.norm(vector))
        return np.asarray(result)


def policy(folder,name='billing',version=1,effective='2026-10-01',body=None,until=None):
    path=folder/f'{name}-v{version}.md'
    text=f'# {name.title()} policy\n\nPolicy ID: {name}-v{version}\nVersion: {version}\nEffective from: {effective}\n'
    if until: text+=f'Effective until: {until}\n'
    text+='\n'+(body or '## Refund rule\n\nA duplicate payment requires refund investigation and operator approval.')
    path.write_text(text,encoding='utf-8')
    return path


@pytest.fixture
def service(tmp_path):
    path=tmp_path/'retrieval.db'
    initialize(path)
    folder=tmp_path/'policies'
    folder.mkdir()
    policy(folder)
    policy(folder,'access',body='## Feature access\n\nAn active subscription can have disabled feature access.')
    return RetrievalService(path,folder,FakeEncoder())


def test_parse_and_token_bounded_heading_chunks(tmp_path):
    body=' '.join(f'word{i}' for i in range(220))
    doc=parse_policy(policy(tmp_path,body='## Long rule\n\n'+body))
    chunks=make_chunks(doc,FakeEncoder(),max_tokens=80)
    assert doc['family']=='billing' and doc['version']==1
    assert len(chunks)>1 and all(c['token_count']<=80 for c in chunks)
    assert ' '.join(c['text'] for c in chunks)==body
    assert chunks==make_chunks(doc,FakeEncoder(),max_tokens=80)
    assert len({c['chunk_id'] for c in chunks})==len(chunks)


def test_fixed_chunk_overlap_is_explicit(tmp_path):
    doc=parse_policy(policy(tmp_path,body='## Long rule\n\n'+' '.join(f'word{i}' for i in range(220))))
    chunks=make_chunks(doc,FakeEncoder(),'fixed',80,12)
    assert chunks[0]['text'].split()[-12:]==chunks[1]['text'].split()[:12]
    assert all(c['token_count']<=80 for c in chunks)


def test_bad_metadata_is_rejected(tmp_path):
    path=policy(tmp_path)
    path.write_text(path.read_text().replace('Version: 1','Version: 2'))
    with pytest.raises(ValueError): parse_policy(path)


def test_bm25_relevance_and_no_match():
    scores=bm25_scores('duplicate invoice',['A duplicate invoice payment','A feature access rule'])
    assert scores[0]>scores[1]==0
    assert bm25_scores('zzzzzz',['A duplicate invoice payment'])==[0]
    assert 'not' in tokenize('A pending payment is not a successful charge')


def test_index_roundtrip_and_hybrid_components(service):
    built=service.build()
    status=service.status()
    assert status['ready'] and status['id']==built['index_id'] and status['documents']==2
    result=service.search('duplicate payment refund',as_of=date(2026,11,1))
    assert result['results'] and result['score_is_probability'] is False
    for row in result['results']:
        expected=(1/(60+row['bm25_rank']) if row['bm25_rank'] else 0)+(1/(60+row['dense_rank']) if row['dense_rank'] else 0)
        assert row['rrf_score']==pytest.approx(expected)
        assert 'embedding_json' not in row
        assert row['chunk_id'] and row['source'].endswith('.md') and row['text']
    reopened=RetrievalService(service.path,service.policies_dir,FakeEncoder())
    assert reopened.search('duplicate payment refund',as_of=date(2026,11,1))['results']==result['results']


def test_version_filtering_precedes_ranking_and_expiry_does_not_revive_v1(service):
    policy(service.policies_dir,version=2,effective='2026-11-01',until='2026-12-01',
           body='## New rule\n\nA new duplicate payment refund investigation rule.')
    service.build()
    old=service.search('payment refund',as_of=date(2026,10,31),family='billing')
    new=service.search('payment refund',as_of=date(2026,11,1),family='billing')
    expired=service.search('payment refund',as_of=date(2026,12,1),family='billing')
    assert {r['policy_id'] for r in old['results']}=={'billing-v1'}
    assert {r['policy_id'] for r in new['results']}=={'billing-v2'}
    assert expired['status']=='no_applicable_policy' and expired['results']==[]
    early=service.search('payment refund',as_of=date(2026,9,1))
    assert early['results']==[]


def test_family_filter_and_lexical_zero_match(service):
    service.build()
    result=service.search('payment refund',method='dense',family='access',as_of=date(2026,11,1))
    assert {r['family'] for r in result['results']}=={'access'}
    assert service.search('zzzzzz',method='bm25')['status']=='no_lexical_match'


def test_failed_rebuild_preserves_published_snapshot(service):
    built=service.build()
    (service.policies_dir/'broken.md').write_text('Missing metadata')
    with pytest.raises(ValueError): service.build()
    assert service.status()['id']==built['index_id']
    assert service.search('refund payment')['results']


def test_comparison_indexes_do_not_replace_live_index(service):
    active=service.build()
    candidate=service.build('fixed',publish=False)
    assert candidate['index_id']!=active['index_id']
    assert service.status()['id']==active['index_id']


def test_missing_index_and_model_mismatch_are_explicit(service):
    assert service.status()['ready'] is False
    with pytest.raises(RetrievalUnavailable): service.search('payment refund')
    service.build()
    service.encoder.revision='different'
    with pytest.raises(RetrievalUnavailable): service.search('payment refund')
    assert service.search('payment refund',method='bm25')['results']


def test_duplicate_versions_and_invalid_vectors_are_not_published(service):
    path=service.policies_dir/'copy.md'
    path.write_text((service.policies_dir/'billing-v1.md').read_text())
    with pytest.raises(ValueError,match='Duplicate'): service.build()
    assert service.status()['ready'] is False


def test_query_limits_and_no_silent_embedding_truncation(service):
    service.build()
    with pytest.raises(ValueError): service.search('a')
    with pytest.raises(ValueError): service.search('refund',top_k=11)
    with pytest.raises(ValueError,match='token limit'): service.search('fee '*260)


def test_zero_vectors_cannot_replace_a_working_index(service):
    active=service.build()
    service.encoder.encode=lambda texts: np.zeros((len(texts),service.encoder.dimensions),dtype=np.float32)
    with pytest.raises(ValueError,match='zero'): service.build()
    assert service.status()['id']==active['index_id']


def test_evaluation_checks_supporting_quotes_and_reports_context_volume(service,tmp_path):
    service.build()
    cases=tmp_path/'cases.json'
    cases.write_text(json.dumps([{'id':'one','query':'duplicate payment refund','as_of':'2026-11-01',
        'relevant':[{'policy_id':'billing-v1','quote':'requires refund investigation'}]}]))
    result=evaluate(service,cases_path=cases)
    for method in result['methods'].values():
        assert method['hit_at_k']==1
        assert 0<method['mrr_at_k']<=1
        assert method['mean_retrieved_embedding_tokens']>0


def test_retrieval_api_enforces_session_and_operator_reindex(tmp_path):
    path=tmp_path/'api.db'
    with TestClient(create_app(path,retrieval_encoder=FakeEncoder())) as client:
        assert client.get('/api/retrieval/status').status_code==401
        assert client.post('/api/retrieval/search',json={'query':'refund policy'}).status_code==401
        customer=client.post('/api/auth/login',json={'email':'asha@example.test','password':DEMO_CUSTOMER_PASSWORD}).json()
        client.headers['Authorization']='Bearer '+customer['access_token']
        assert client.get('/api/retrieval/status').json()['ready'] is False
        assert client.post('/api/retrieval/search',json={'query':'refund policy'}).status_code==503
        assert client.post('/api/retrieval/reindex',json={}).status_code==403
        assert client.post('/api/retrieval/evaluate').status_code==403
        operator=client.post('/api/auth/login',json={'email':'support@example.test','password':DEMO_OPERATOR_PASSWORD}).json()
        client.headers['Authorization']='Bearer '+operator['access_token']
        built=client.post('/api/retrieval/reindex',json={})
        assert built.status_code==200,built.text
        assert client.get('/api/retrieval/chunks').json()['chunks']
        assert client.post('/api/retrieval/search',json={'query':'refund policy','method':'invented'}).status_code==422
        assert client.post('/api/retrieval/search',json={'query':'refund policy','as_of':'not-a-date'}).status_code==422
        client.headers['Authorization']='Bearer '+customer['access_token']
        response=client.post('/api/retrieval/search',json={'query':'I was charged twice. Can I get a refund?'})
        assert response.status_code==200 and response.json()['results']
