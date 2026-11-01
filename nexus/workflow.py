"""Bounded LangGraph investigation. Financial decisions are deterministic rules."""
import json
import time
import uuid
from typing import TypedDict
from fastapi import HTTPException
from langgraph.graph import StateGraph,START,END
from nexus.auth import require_account_access
from nexus.db import connect
from nexus.tools import AccountTools,eligible_actions


class InvestigationState(TypedDict,total=False):
    snapshot: dict
    policies: list
    findings: list
    candidates: list
    outcome: str
    summary: str
    policy_ready: bool


def authorized_investigation(db,user,investigation_id):
    row=db.execute('SELECT * FROM investigations WHERE id=?',(investigation_id,)).fetchone()
    if row is None: raise HTTPException(404,'Investigation not found')
    require_account_access(user,row['account_id'])
    return dict(row)


def audit(db,user,account_id,investigation_id,event,data):
    db.execute('INSERT INTO audit_events(actor_id,account_id,investigation_id,event,data_json) VALUES (?,?,?,?,?)',
               (user.id,account_id,investigation_id,event,json.dumps(data,allow_nan=False)))


class InvestigationService:
    def __init__(self,path,retrieval,tools=None):
        self.path,self.retrieval=path,retrieval
        self.tools=tools or AccountTools(path)

    def inspect(self,user,investigation_id):
        with connect(self.path) as db:
            row=authorized_investigation(db,user,investigation_id)
            state=row.pop('state_json')
            row['state']=json.loads(state) if state else None
            row['steps']=[{**dict(r),'data':json.loads(r['data_json'])} for r in db.execute(
                'SELECT id,stage,data_json,created_at FROM investigation_steps WHERE investigation_id=? ORDER BY id',(investigation_id,))]
            for step in row['steps']: step.pop('data_json')
            row['proposals']=[dict(r) for r in db.execute('SELECT * FROM action_proposals WHERE investigation_id=? ORDER BY created_at,id',(investigation_id,))]
            for proposal in row['proposals']: proposal['policies']=json.loads(proposal.pop('policy_json'))
            row['ticket']=dict(db.execute('SELECT * FROM tickets WHERE id=?',(row['ticket_id'],)).fetchone())
        return row

    def list(self,user):
        with connect(self.path) as db:
            sql='SELECT id,ticket_id,account_id,status,created_at,complaint FROM investigations'
            rows=db.execute(sql+' ORDER BY created_at DESC,id DESC LIMIT 100') if user.role=='operator' else db.execute(
                sql+' WHERE account_id=? ORDER BY created_at DESC,id DESC LIMIT 100',(user.account_id,))
            return [dict(r) for r in rows]

    def recover_interrupted(self):
        with connect(self.path) as db:
            db.execute("UPDATE tickets SET status='escalated' WHERE id IN (SELECT ticket_id FROM investigations WHERE status='running')")
            db.execute("UPDATE investigations SET status='failed',error_code='server_restarted',completed_at=CURRENT_TIMESTAMP WHERE status='running'")

    def run(self,user,account_id,complaint,event_date):
        require_account_access(user,account_id)
        # Authorization and existence before any persisted case.
        self.tools.inspect_account(user,account_id)
        investigation_id='case_'+uuid.uuid4().hex
        ticket_id='tkt_'+uuid.uuid4().hex
        with connect(self.path) as db:
            db.execute("INSERT INTO tickets(id,account_id,subject,description,status) VALUES (?,?,?,?,'investigating')",
                       (ticket_id,account_id,complaint[:120],complaint))
            db.execute("INSERT INTO investigations(id,ticket_id,account_id,created_by,complaint,event_date,status) VALUES (?,?,?,?,?,?,'running')",
                       (investigation_id,ticket_id,account_id,user.id,complaint,str(event_date)))
            audit(db,user,account_id,investigation_id,'investigation_started',{'ticket_id':ticket_id})

        def record(stage,update):
            with connect(self.path) as db:
                db.execute('INSERT INTO investigation_steps(investigation_id,stage,data_json) VALUES (?,?,?)',
                           (investigation_id,stage,json.dumps(update,allow_nan=False)))
            return update

        def inspect(state):
            return record('inspect_records',{'snapshot':self.tools.snapshot(user,account_id)})

        def retrieve(state):
            # Controlled policy queries; customer text never selects an account, tool or amount.
            results=[]
            required=(('billing','refund eligibility and approval'),('access','access correction and approval'),('escalation','missing evidence and escalation'))
            for family,query in required:
                found=self.retrieval.search(query,method='bm25',top_k=3,as_of=event_date,family=family)
                results.extend(found['results'])
            unique={p['chunk_id']:p for p in results}
            ready=all(any(p['policy_id']==family+'-v1' and p['version']==1 and p['heading'].lower()==heading for p in unique.values()) for family,heading in required)
            return record('retrieve_policies',{'policies':list(unique.values()),'policy_ready':ready})

        def assess(state):
            snapshot=state['snapshot'];findings=[]
            for invoice in snapshot['invoices']:
                succeeded=[p for p in invoice['payments'] if p['status']=='succeeded']
                findings.append({'type':'billing','invoice_id':invoice['id'],'successful_payments':len(succeeded),
                    'pending_attempts':sum(p['status']=='pending' for p in invoice['payments']),
                    'failed_attempts':sum(p['status']=='failed' for p in invoice['payments']),
                    'net_paid_minor':invoice['net_paid_minor'],'excess_minor':invoice['excess_minor']})
            findings.append({'type':'access','subscription':snapshot['subscription'],
                             'disabled_features':sorted({'team_workspace','advanced_reports'}-{e['feature'] for e in snapshot['entitlements'] if e['enabled']})})
            candidates=eligible_actions(snapshot) if state['policy_ready'] else []
            outcome='pending_approval' if candidates else 'resolved'
            if not state['policy_ready'] or not snapshot['invoices'] or any(i['net_paid_minor']<i['amount_minor'] or i['excess_minor'] for i in snapshot['invoices']):
                if not candidates: outcome='escalated'
            if findings[-1]['disabled_features'] and not candidates: outcome='escalated'
            summary=('Verified account records support proposed simulated fixes. Operator approval is required; no action has executed.' if candidates else
                     'No eligible automatic fix was established. Escalated for human review.' if outcome=='escalated' else
                     'The checked account has no outstanding billing or access discrepancy under the supported rules.')
            return record('assess_evidence',{'findings':findings,'candidates':candidates,'outcome':outcome,'summary':summary})

        def persist(state):
            with connect(self.path) as db:
                db.execute('BEGIN IMMEDIATE')
                for candidate in state['candidates']:
                    db.execute('INSERT INTO action_proposals(id,investigation_id,account_id,kind,target_id,amount_minor,reason,policy_json) VALUES (?,?,?,?,?,?,?,?)',
                        ('act_'+uuid.uuid4().hex,investigation_id,account_id,candidate['kind'],candidate['target_id'],candidate['amount_minor'],candidate['reason'],json.dumps(state['policies'])))
                db.execute("UPDATE investigations SET status='completed',state_json=?,completed_at=CURRENT_TIMESTAMP WHERE id=?",(json.dumps(state),investigation_id))
                db.execute('UPDATE tickets SET status=? WHERE id=?',(state['outcome'],ticket_id))
                audit(db,user,account_id,investigation_id,'investigation_completed',{'outcome':state['outcome'],'proposal_count':len(state['candidates'])})
                db.execute('INSERT INTO investigation_steps(investigation_id,stage,data_json) VALUES (?,?,?)',(investigation_id,'persist_result','{}'))
            return {}

        graph=StateGraph(InvestigationState)
        for name,node in [('inspect',inspect),('retrieve',retrieve),('assess',assess),('persist',persist)]: graph.add_node(name,node)
        for left,right in [(START,'inspect'),('inspect','retrieve'),('retrieve','assess'),('assess','persist'),('persist',END)]: graph.add_edge(left,right)
        try:
            from langsmith import tracing_context
            with tracing_context(enabled=False):
                graph.compile().invoke({},config={'recursion_limit':8,'callbacks':[]})
        except Exception:
            with connect(self.path) as db:
                db.execute("UPDATE investigations SET status='failed',error_code='investigation_failed',completed_at=CURRENT_TIMESTAMP WHERE id=?",(investigation_id,))
                db.execute("UPDATE tickets SET status='escalated' WHERE id=?",(ticket_id,))
                audit(db,user,account_id,investigation_id,'investigation_failed',{'reason':'Tool, policy retrieval or workflow failure; no accepted resolution.'})
            # Persisted failure is inspectable without exposing exception bodies.
        return self.inspect(user,investigation_id)
