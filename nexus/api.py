"""NexusOS local support API: Part 2 adds identity and account authorization."""
import uuid
import json
import time
from collections import deque
from datetime import date
from typing import Literal
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse,FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from nexus.db import connect, default_path, initialize, seed_demo
from nexus.auth import (SESSION_TTL, LoginRequest, LoginResponse, Principal, authenticated_user,
                        bearer, login, require_account_access, require_operator, revoke)
from nexus.retrieval import RetrievalService, evaluate
from nexus.embeddings import RetrievalUnavailable
from nexus.llm import GenerationError
from nexus.rag import RAGRequest, RAGService
from nexus.workflow import InvestigationService
from nexus.actions import ActionService
from nexus.tools import AccountTools
from nexus.retrieval import today_india


class TicketCreate(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    account_id: str = Field(min_length=1, max_length=80)
    subject: str = Field(min_length=3, max_length=120)
    description: str = Field(min_length=10, max_length=4000)


class PolicySearch(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    query: str = Field(min_length=3,max_length=1200)
    method: Literal['bm25','dense','hybrid'] = 'hybrid'
    top_k: int = Field(default=3,ge=1,le=10)
    as_of: date | None = None
    family: str | None = Field(default=None,pattern=r'^[a-z][a-z0-9-]{0,60}$')


class IndexBuild(BaseModel):
    model_config = ConfigDict(extra='forbid')
    strategy: Literal['heading','fixed'] = 'heading'
    max_tokens: int = Field(default=180,ge=40,le=220)
    overlap_words: int = Field(default=12,ge=0,le=30)


class ConversationCreate(BaseModel):
    model_config=ConfigDict(extra='forbid',str_strip_whitespace=True)
    title: str = Field(default='Support conversation',min_length=3,max_length=120)
    account_id: str | None = Field(default=None,min_length=1,max_length=80)


class InvestigationCreate(BaseModel):
    model_config=ConfigDict(extra='forbid',str_strip_whitespace=True)
    account_id: str = Field(min_length=1,max_length=80)
    complaint: str = Field(min_length=10,max_length=4000)
    event_date: date = Field(default_factory=today_india)


class ActionDecision(BaseModel):
    model_config=ConfigDict(extra='forbid',str_strip_whitespace=True)
    decision: Literal['approve','reject']
    note: str = Field(min_length=3,max_length=1000)


class ActionExecution(BaseModel):
    model_config=ConfigDict(extra='forbid',str_strip_whitespace=True)
    idempotency_key: str = Field(min_length=8,max_length=100,pattern=r'^[A-Za-z0-9_-]+$')


def require_account(db, account_id):
    row = db.execute('SELECT * FROM accounts WHERE id = ?', (account_id,)).fetchone()
    if row is None:
        raise HTTPException(404, 'Account not found')
    return dict(row)


def create_app(db_path=None, demo=True, session_ttl=SESSION_TTL, retrieval_encoder=None, policies_dir=None, generation_provider=None):
    path = Path(db_path) if db_path is not None else default_path()
    if not isinstance(session_ttl, int) or not 1 <= session_ttl <= 86400:
        raise ValueError('Session TTL must be 1..86400 seconds')

    @asynccontextmanager
    async def lifespan(app):
        initialize(path)
        if demo:
            seed_demo(path)
        rag.memory.recover_interrupted()
        investigations.recover_interrupted()
        yield

    app = FastAPI(title='NexusOS — OrbitDesk Support', version='0.8.0', lifespan=lifespan,
                  description='Synthetic support application: policy RAG, private conversation memory, bounded account investigation and approval-gated simulated actions. Login and Authorize first. No real money moves.')
    app.state.db_path = path
    retrieval=RetrievalService(path,policies_dir,retrieval_encoder)
    app.state.retrieval=retrieval
    rag=RAGService(path,retrieval,generation_provider)
    app.state.rag=rag
    tools=AccountTools(path)
    investigations=InvestigationService(path,retrieval,tools)
    actions=ActionService(path)
    app.state.investigations=investigations
    app.state.actions=actions
    app.state.metrics={'started':time.monotonic(),'requests':0,'server_errors':0,'latencies_ms':deque(maxlen=1000)}

    @app.middleware('http')
    async def request_metrics(request: Request,call_next):
        started=time.perf_counter()
        response=await call_next(request)
        elapsed=(time.perf_counter()-started)*1000
        metrics=app.state.metrics
        metrics['requests']+=1
        metrics['server_errors']+=int(response.status_code>=500)
        metrics['latencies_ms'].append(elapsed)
        response.headers['X-Request-ID']=uuid.uuid4().hex
        response.headers['Server-Timing']=f'app;dur={elapsed:.2f}'
        response.headers['X-Content-Type-Options']='nosniff'
        if request.url.path.startswith('/api/'):
            response.headers['Cache-Control']='no-store'
        return response

    @app.get('/api/system/status',tags=['System'])
    def system_status(user: Principal = Depends(require_operator)):
        with connect(path) as db:
            counts={table:db.execute('SELECT COUNT(*) FROM '+table).fetchone()[0] for table in ('investigations','action_proposals','refunds','audit_events')}
            usage={}
            for row in db.execute("SELECT answer_json FROM conversation_turns WHERE status='completed' ORDER BY rowid DESC LIMIT 100"):
                for key,value in json.loads(row[0]).get('usage',{}).items():
                    if isinstance(value,int):usage[key]=usage.get(key,0)+value
        metrics=app.state.metrics;latencies=sorted(metrics['latencies_ms'])
        return {'mode':'synthetic local demo','single_worker':True,'uptime_seconds':round(time.monotonic()-metrics['started'],1),
                'requests_since_restart':metrics['requests'],'server_errors_since_restart':metrics['server_errors'],
                'recent_request_p95_ms':round(latencies[min(len(latencies)-1,int(.95*len(latencies)))],2) if latencies else None,
                'database_counts':counts,'retrieval':retrieval.status(),'generator':rag.status(),
                'saved_generation_usage_last_100_completed_turns':usage,
                'cost_note':'Token usage is not a currency bill. Failed/stateless generations are not included in this saved-turn aggregate.'}
    static=Path(__file__).parent/'static'
    app.mount('/static',StaticFiles(directory=static),name='static')

    @app.get('/',include_in_schema=False)
    def dashboard():
        return FileResponse(static/'index.html',headers={'Cache-Control':'no-store',
            'Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'",
            'X-Content-Type-Options':'nosniff'})

    @app.get('/health', tags=['System'])
    def health():
        with connect(path) as db:
            db.execute('SELECT 1').fetchone()
        return {'status':'ok', 'phase':8, 'mode':'synthetic demo' if demo else 'local', 'authentication':'expiring bearer sessions'}

    @app.get('/api/tools/account/{account_id}',tags=['Investigation tools'])
    def inspect_account_tool(account_id: str,user: Principal = Depends(authenticated_user)):
        return tools.inspect_account(user,account_id)

    @app.get('/api/tools/billing/{account_id}',tags=['Investigation tools'])
    def inspect_billing_tool(account_id: str,user: Principal = Depends(authenticated_user)):
        return tools.inspect_billing(user,account_id)

    @app.post('/api/investigations',status_code=201,tags=['Investigations'])
    def run_investigation(payload: InvestigationCreate,user: Principal = Depends(authenticated_user)):
        return investigations.run(user,payload.account_id,payload.complaint,payload.event_date)

    @app.get('/api/investigations',tags=['Investigations'])
    def list_investigations(user: Principal = Depends(authenticated_user)):
        return {'investigations':investigations.list(user)}

    @app.get('/api/investigations/{investigation_id}',tags=['Investigations'])
    def inspect_investigation(investigation_id: str,user: Principal = Depends(authenticated_user)):
        return investigations.inspect(user,investigation_id)

    @app.get('/api/proposals',tags=['Approval and actions'])
    def list_proposals(user: Principal = Depends(authenticated_user)):
        return {'proposals':actions.list(user)}

    @app.post('/api/proposals/{proposal_id}/decision',tags=['Approval and actions'])
    def decide_proposal(proposal_id: str,payload: ActionDecision,user: Principal = Depends(require_operator)):
        return actions.decide(user,proposal_id,payload.decision,payload.note)

    @app.post('/api/proposals/{proposal_id}/execute',tags=['Approval and actions'])
    def execute_proposal(proposal_id: str,payload: ActionExecution,user: Principal = Depends(require_operator)):
        return actions.execute(user,proposal_id,payload.idempotency_key)

    @app.get('/api/accounts/{account_id}/audit',tags=['Approval and actions'])
    def account_audit(account_id: str,user: Principal = Depends(authenticated_user)):
        return {'events':actions.history(user,account_id)}

    def retrieval_call(function,*args,**kwargs):
        try:
            return function(*args,**kwargs)
        except GenerationError as exc:
            raise HTTPException(exc.status,{'code':exc.code,'message':str(exc)}) from exc
        except RetrievalUnavailable as exc:
            raise HTTPException(503,str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400,str(exc)) from exc

    @app.get('/api/rag/status',tags=['Policy answers'])
    def rag_status(user: Principal = Depends(authenticated_user)):
        return rag.status()

    @app.post('/api/rag/preview',tags=['Policy answers'])
    def rag_preview(payload: RAGRequest,user: Principal = Depends(authenticated_user)):
        return retrieval_call(rag.preview,user,payload)

    @app.post('/api/rag/answer',tags=['Policy answers'])
    def rag_answer(payload: RAGRequest,user: Principal = Depends(authenticated_user)):
        job=retrieval_call(rag.start,user,payload)
        return retrieval_call(rag.answer,job)

    @app.post('/api/rag/stream',tags=['Policy answers'])
    def rag_stream(payload: RAGRequest,user: Principal = Depends(authenticated_user)):
        job=retrieval_call(rag.start,user,payload)
        def stream():
            for event in rag.events(job):
                yield json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n'
        return StreamingResponse(stream(),media_type='application/x-ndjson',headers={'Cache-Control':'no-store','X-Accel-Buffering':'no'})

    @app.post('/api/conversations',status_code=201,tags=['Conversation memory'])
    def create_conversation(payload: ConversationCreate,user: Principal = Depends(authenticated_user)):
        return rag.memory.create(user,payload.title,payload.account_id)

    @app.get('/api/conversations',tags=['Conversation memory'])
    def list_conversations(user: Principal = Depends(authenticated_user)):
        return {'conversations':rag.memory.list(user)}

    @app.get('/api/conversations/{conversation_id}',tags=['Conversation memory'])
    def inspect_conversation(conversation_id: str,user: Principal = Depends(authenticated_user)):
        return rag.memory.inspect(user,conversation_id)

    @app.get('/api/retrieval/status',tags=['Policy retrieval'])
    def retrieval_status(user: Principal = Depends(authenticated_user)):
        return retrieval.status()

    @app.post('/api/retrieval/search',tags=['Policy retrieval'])
    def search_policies(payload: PolicySearch,user: Principal = Depends(authenticated_user)):
        return retrieval_call(retrieval.search,**payload.model_dump())

    @app.get('/api/retrieval/chunks',tags=['Policy retrieval'])
    def inspect_chunks(as_of: date | None = None,family: str | None = None,user: Principal = Depends(authenticated_user)):
        return retrieval_call(retrieval.chunks,as_of=as_of,family=family)

    @app.post('/api/retrieval/reindex',tags=['Policy retrieval'])
    def reindex(payload: IndexBuild,user: Principal = Depends(require_operator)):
        return retrieval_call(retrieval.build,**payload.model_dump())

    @app.post('/api/retrieval/evaluate',tags=['Policy retrieval'])
    def evaluate_retrieval(user: Principal = Depends(require_operator)):
        return retrieval_call(evaluate,retrieval)

    @app.post('/api/auth/login', response_model=LoginResponse, tags=['Authentication'])
    def sign_in(payload: LoginRequest, request: Request, response: Response):
        response.headers['Cache-Control'] = 'no-store'
        response.headers['Pragma'] = 'no-cache'
        return login(path,payload,request.client.host if request.client else 'unknown',session_ttl)

    @app.get('/api/auth/me', response_model=Principal, tags=['Authentication'])
    def me(user: Principal = Depends(authenticated_user)):
        return user

    @app.post('/api/auth/logout', tags=['Authentication'])
    def sign_out(user: Principal = Depends(authenticated_user), credentials: HTTPAuthorizationCredentials = Depends(bearer)):
        revoke(path,credentials.credentials)
        return {'message':'Session revoked'}

    @app.get('/api/demo/scenarios', tags=['Demo'])
    def scenarios(user: Principal = Depends(require_operator)):
        return {'product':'OrbitDesk', 'currency':'INR', 'money_unit':'paise',
                'pro_monthly_price_minor':99900,
                'scenarios':[
                    {'account_id':'acct_healthy','name':'Healthy Pro account','complaint':'Can you confirm my subscription and access?',
                     'expected_facts':['One successful payment','Pro subscription active','Both Pro features enabled']},
                    {'account_id':'acct_pending','name':'Pending payment','complaint':'I tried paying twice. Did you charge me twice?',
                     'expected_facts':['One pending attempt and one failed attempt','No confirmed successful payment','No duplicate-charge conclusion']},
                    {'account_id':'acct_duplicate','name':'Duplicate charge and missing access','complaint':'I upgraded yesterday, was charged twice, and still cannot access the paid features.',
                     'expected_facts':['Two successful payments for one invoice','Pro subscription active','Pro features disabled']},
                ]}

    @app.get('/api/accounts', tags=['Accounts'])
    def list_accounts(user: Principal = Depends(authenticated_user)):
        with connect(path) as db:
            if user.role == 'operator':
                rows=db.execute('SELECT id,name,email FROM accounts ORDER BY id')
            else:
                rows=db.execute('SELECT id,name,email FROM accounts WHERE id=?',(user.account_id,))
            return {'accounts':[dict(row) for row in rows]}

    @app.get('/api/accounts/{account_id}', tags=['Accounts'])
    def account_details(account_id: str, user: Principal = Depends(authenticated_user)):
        require_account_access(user,account_id)
        with connect(path) as db:
            account = require_account(db,account_id)
            subscription = db.execute('SELECT * FROM subscriptions WHERE account_id = ?', (account_id,)).fetchone()
            entitlements = [{'feature':r['feature'],'enabled':bool(r['enabled'])} for r in db.execute(
                'SELECT feature,enabled FROM entitlements WHERE account_id = ? ORDER BY feature',(account_id,))]
        return {'account':account,'subscription':dict(subscription) if subscription else None,'entitlements':entitlements}

    @app.get('/api/accounts/{account_id}/invoices', tags=['Billing'])
    def invoices(account_id: str, user: Principal = Depends(authenticated_user)):
        data=tools.inspect_billing(user,account_id)
        for invoice in data['invoices']:
            invoice['outstanding_minor']=max(0,invoice['amount_minor']-invoice['net_paid_minor'])
            invoice['excess_paid_minor']=invoice['excess_minor']
        return {'account_id':account_id,**data}

    @app.post('/api/tickets', status_code=201, tags=['Tickets'])
    def create_ticket(payload: TicketCreate, user: Principal = Depends(authenticated_user)):
        require_account_access(user,payload.account_id)
        ticket_id = 'tkt_' + uuid.uuid4().hex
        with connect(path) as db:
            require_account(db,payload.account_id)
            db.execute('INSERT INTO tickets(id,account_id,subject,description) VALUES (?,?,?,?)',
                       (ticket_id,payload.account_id,payload.subject,payload.description))
            return dict(db.execute('SELECT * FROM tickets WHERE id = ?', (ticket_id,)).fetchone())

    @app.get('/api/tickets/{ticket_id}', tags=['Tickets'])
    def get_ticket(ticket_id: str, user: Principal = Depends(authenticated_user)):
        with connect(path) as db:
            if user.role == 'operator':
                row=db.execute('SELECT * FROM tickets WHERE id = ?', (ticket_id,)).fetchone()
            else:
                row=db.execute('SELECT * FROM tickets WHERE id=? AND account_id=?',(ticket_id,user.account_id)).fetchone()
            if row is None:
                raise HTTPException(404,'Ticket not found')
            return dict(row)

    @app.get('/api/accounts/{account_id}/tickets', tags=['Tickets'])
    def account_tickets(account_id: str, user: Principal = Depends(authenticated_user)):
        require_account_access(user,account_id)
        with connect(path) as db:
            require_account(db,account_id)
            return {'account_id':account_id,'tickets':[dict(row) for row in db.execute(
                'SELECT * FROM tickets WHERE account_id = ? ORDER BY created_at,id',(account_id,))]}

    return app


app = create_app()
