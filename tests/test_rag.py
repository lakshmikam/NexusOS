import json
import httpx
import pytest
from fastapi.testclient import TestClient
from nexus.api import create_app
from nexus.auth import DEMO_CUSTOMER_PASSWORD,DEMO_OPERATOR_PASSWORD
from nexus.db import connect
from nexus.llm import GeminiProvider,GenerationError,LLMConfig
from tests.test_retrieval import FakeEncoder


class FixtureProvider:
    """Tests only. The application never silently substitutes this for Gemini."""
    def __init__(self,mode='good'):
        self.mode=mode
        self.contexts=[]
    def status(self): return {'provider':'test-fixture','model':'fixture','configured':True}
    def ensure_ready(self):
        if self.mode=='unconfigured': raise GenerationError('configuration','Set up the provider.',503)
    def stream(self,system,context,schema):
        self.contexts.append(context)
        if self.mode=='timeout': raise GenerationError('provider_timeout','Synthetic timeout',504)
        source=next((s for s in context['current_sources'] if 'no prior refund' in s['text']),context['current_sources'][0])
        claim={'text':'The cited policy specifies the requirements that support must check.',
               'citations':[{'source_id':source['source_id'],'quote':source['text'][:180]}]}
        if self.mode=='unknown': claim['citations'][0]['source_id']='S5'
        if self.mode=='badquote':
            claim['text']='NEVER-RELEASE-THIS'
            claim['citations'][0]['quote']='This fabricated quote is not in the current source.'
        if self.mode=='wrongmeaning': claim['text']='No operator approval is required.'
        if self.mode=='action': claim['text']='I have refunded your payment.'
        result={'status':'answered','claims':[claim],'clarification':None}
        if self.mode=='insufficient': result={'status':'insufficient_evidence','claims':[],'clarification':None}
        if self.mode=='clarify': result={'status':'needs_clarification','claims':[],'clarification':'Which billing event are you asking about?'}
        raw='not valid json' if self.mode=='malformed' else json.dumps(result)
        for offset in range(0,len(raw),31): yield {'type':'delta','text':raw[offset:offset+31]}
        yield {'type':'usage','usage':{'promptTokenCount':123,'candidatesTokenCount':45}}


@pytest.fixture
def environment(tmp_path):
    provider=FixtureProvider()
    app=create_app(tmp_path/'rag.db',retrieval_encoder=FakeEncoder(),generation_provider=provider)
    with TestClient(app) as client:
        operator=client.post('/api/auth/login',json={'email':'support@example.test','password':DEMO_OPERATOR_PASSWORD}).json()
        client.headers['Authorization']='Bearer '+operator['access_token']
        response=client.post('/api/retrieval/reindex',json={})
        assert response.status_code==200,response.text
        user=client.post('/api/auth/login',json={'email':'asha@example.test','password':DEMO_CUSTOMER_PASSWORD}).json()
        client.headers['Authorization']='Bearer '+user['access_token']
        yield client,app,provider,user


def question(**changes):
    return {'question':'What evidence is required for a duplicate payment refund?','method':'bm25','as_of':'2026-10-31',**changes}


def conversation(client):
    response=client.post('/api/conversations',json={'title':'Billing support'})
    assert response.status_code==201
    return response.json()['id']


def test_prompt_preview_is_explicit_and_has_no_generation_side_effect(environment):
    client,app,provider,_=environment
    result=client.post('/api/rag/preview',json=question()).json()
    assert 'NO account investigation tools' in result['system_instruction']
    assert result['context']['current_sources'] and result['context']['history_unverified']==[]
    assert result['context_info']['budget_unit']=='characters, not exact Gemini tokens'
    assert provider.contexts==[]
    with connect(app.state.db_path) as db:
        assert db.execute('SELECT COUNT(*) FROM conversation_turns').fetchone()[0]==0


def test_stateless_answer_has_checked_citations_and_no_persistence(environment):
    client,app,_,_=environment
    response=client.post('/api/rag/answer',json=question())
    assert response.status_code==200,response.text
    result=response.json()
    assert result['status']=='answered' and result['claims']
    assert '[S' in result['answer'] and result['memory_persisted'] is False
    assert result['citation_checks']['semantic_entailment']=='not automatically verified'
    with connect(app.state.db_path) as db:
        assert db.execute('SELECT COUNT(*) FROM conversation_turns').fetchone()[0]==0


def test_stream_withholds_text_until_validation(environment):
    client,_,_,_=environment
    response=client.post('/api/rag/stream',json=question())
    events=[json.loads(line) for line in response.text.splitlines()]
    types=[e['type'] for e in events]
    assert types.index('provider_progress')<types.index('validated')<types.index('answer_delta')
    assert ''.join(e['text'] for e in events if e['type']=='answer_delta')==events[-1]['result']['answer']
    assert events[-1]['type']=='done'


@pytest.mark.parametrize('mode,code',[('unknown','unknown_citation'),('badquote','unsupported_quote'),
    ('malformed','invalid_structure'),('action','action_claim'),('timeout','provider_timeout')])
def test_bad_answers_are_not_published_or_remembered(environment,mode,code):
    client,_,provider,_=environment
    provider.mode=mode
    conversation_id=conversation(client)
    response=client.post('/api/rag/stream',json=question(conversation_id=conversation_id))
    events=[json.loads(line) for line in response.text.splitlines()]
    assert events[-1]['type']=='error' and events[-1]['code']==code
    assert not any(e['type']=='answer_delta' for e in events)
    assert 'NEVER-RELEASE-THIS' not in response.text
    turn=client.get('/api/conversations/'+conversation_id).json()['turns'][0]
    assert turn['status']=='failed' and turn['answer'] is None


def test_valid_quote_does_not_prove_claim_meaning(environment):
    client,_,provider,_=environment
    provider.mode='wrongmeaning'
    response=client.post('/api/rag/answer',json=question())
    assert response.status_code==200
    assert response.json()['citation_checks']['semantic_entailment']=='not automatically verified'
    # Deliberate counterexample: deterministic quote integrity is not a semantic fact checker.


def test_no_applicable_policy_skips_provider_even_without_key(environment):
    client,_,provider,_=environment
    provider.mode='unconfigured'
    result=client.post('/api/rag/answer',json=question(as_of='2026-09-01')).json()
    assert result['status']=='insufficient_evidence' and result['provider_used'] is False
    assert provider.contexts==[]
    assert client.post('/api/rag/answer',json=question()).status_code==503


def test_model_can_abstain_or_ask_a_question(environment):
    client,_,provider,_=environment
    provider.mode='insufficient'
    result=client.post('/api/rag/answer',json=question()).json()
    assert result['status']=='insufficient_evidence' and result['claims']==[]
    provider.mode='clarify'
    result=client.post('/api/rag/answer',json=question()).json()
    assert result['status']=='needs_clarification' and result['answer'].endswith('?')


def test_memory_followup_is_scoped_and_bounded(environment):
    client,_,provider,_=environment
    conversation_id=conversation(client)
    first=client.post('/api/rag/answer',json=question(conversation_id=conversation_id)).json()
    second=client.post('/api/rag/answer',json=question(question='Does that need operator approval?',conversation_id=conversation_id)).json()
    assert first['memory_persisted'] and second['context_info']['included_history_turns']==1
    assert second['retrieval']['query_expanded_from_recent_user_question'] is True
    assert '[S1]' not in provider.contexts[-1]['history_unverified'][0]['previous_answer']
    for _ in range(2):
        assert client.post('/api/rag/answer',json=question(conversation_id=conversation_id)).status_code==200
    preview=client.post('/api/rag/preview',json=question(conversation_id=conversation_id)).json()
    assert preview['context_info']['stored_completed_turns']==4
    assert preview['context_info']['included_history_turns']==3
    assert preview['context_info']['history_is_bounded'] is True


def test_conversation_ownership_and_role_fields_cannot_be_spoofed(environment):
    client,_,_,_=environment
    conversation_id=conversation(client)
    assert client.post('/api/conversations',json={'title':'Bad owner','account_id':'acct_healthy'}).status_code==404
    assert client.post('/api/conversations',json={'title':'Bad owner','owner_user_id':'usr_meera'}).status_code==422
    for email,password in [('meera@example.test',DEMO_CUSTOMER_PASSWORD),('support@example.test',DEMO_OPERATOR_PASSWORD)]:
        login=client.post('/api/auth/login',json={'email':email,'password':password}).json()
        client.headers['Authorization']='Bearer '+login['access_token']
        assert client.get('/api/conversations/'+conversation_id).status_code==404
        assert client.post('/api/rag/preview',json=question(conversation_id=conversation_id)).status_code==404
        assert client.get('/api/conversations').json()['conversations']==[]


def test_history_survives_restart_and_incomplete_turn_is_recovered(environment):
    client,app,provider,user=environment
    conversation_id=conversation(client)
    assert client.post('/api/rag/answer',json=question(conversation_id=conversation_id)).status_code==200
    from nexus.auth import Principal
    principal=Principal(**user['user'])
    job=app.state.rag.start(principal, __import__('nexus.rag',fromlist=['RAGRequest']).RAGRequest(**question(conversation_id=conversation_id)))
    assert client.post('/api/rag/answer',json=question(conversation_id=conversation_id)).status_code==409
    with TestClient(create_app(app.state.db_path,retrieval_encoder=FakeEncoder(),generation_provider=provider)) as restarted:
        restarted.headers['Authorization']='Bearer '+user['access_token']
        turns=restarted.get('/api/conversations/'+conversation_id).json()['turns']
        assert turns[0]['status']=='completed'
        assert turns[1]['status']=='failed' and turns[1]['error_code']=='server_restarted'


def test_rag_endpoints_require_identity(environment):
    client,_,_,_=environment
    client.headers.pop('Authorization')
    assert client.get('/api/rag/status').status_code==401
    assert client.get('/api/conversations').status_code==401
    assert client.post('/api/rag/answer',json=question()).status_code==401
    assert client.post('/api/rag/preview',json=question()).status_code==401


def sse(payload):
    return 'data: '+json.dumps(payload)+'\n\n'


def test_gemini_transport_uses_secret_header_and_stream_schema():
    def handle(request):
        assert request.headers['x-goog-api-key']=='fixture-secret'
        assert 'fixture-secret' not in str(request.url)
        body=json.loads(request.content)
        assert body['generationConfig']['responseMimeType']=='application/json'
        assert 'responseJsonSchema' in body['generationConfig']
        frames=sse({'candidates':[{'content':{'parts':[{'text':'hidden reasoning','thought':True},{'text':'{"ok":true}'}]},'finishReason':'STOP'}]})
        return httpx.Response(200,text=frames,headers={'content-type':'text/event-stream'})
    provider=GeminiProvider(LLMConfig(api_key='fixture-secret'),httpx.MockTransport(handle))
    result=list(provider.stream('system',{'question':'test'},{'type':'object'}))
    assert ''.join(e['text'] for e in result if e['type']=='delta')=='{"ok":true}'
    assert 'fixture-secret' not in repr(provider.config)


@pytest.mark.parametrize('status,code',[(403,'provider_auth'),(404,'model_unavailable'),(429,'provider_quota'),(500,'provider_unavailable')])
def test_provider_errors_never_echo_secret_response_bodies(status,code):
    transport=httpx.MockTransport(lambda request:httpx.Response(status,text='fixture-secret provider internals'))
    provider=GeminiProvider(LLMConfig(api_key='fixture-secret'),transport)
    with pytest.raises(GenerationError) as caught:
        list(provider.stream('system',{},{}))
    assert caught.value.code==code and 'fixture-secret' not in str(caught.value)


def test_truncated_provider_response_is_rejected():
    transport=httpx.MockTransport(lambda request:httpx.Response(200,text=sse({'candidates':[{'content':{'parts':[{'text':'partial'}]},'finishReason':'MAX_TOKENS'}]})))
    provider=GeminiProvider(LLMConfig(api_key='fixture-secret'),transport)
    with pytest.raises(GenerationError,match='complete response'):
        list(provider.stream('system',{},{}))
