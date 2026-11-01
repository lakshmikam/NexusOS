"""Approval and simulated execution, serialized with eligibility revalidation."""
import hashlib
import json
import uuid
from fastapi import HTTPException
from nexus.db import connect
from nexus.auth import require_account_access
from nexus.tools import account_snapshot,eligible_actions,unresolved
from nexus.workflow import audit
from nexus.retrieval import today_india


def policies_current(db,item):
    expected={p['policy_id']:p['source_hash'] for p in json.loads(item['policy_json'])}
    active=db.execute('SELECT active_index_id FROM retrieval_state WHERE id=1').fetchone()
    if not active: return False
    for family in ('billing','access','escalation'):
        row=db.execute('SELECT * FROM policy_documents WHERE index_id=? AND family=? AND effective_from<=? ORDER BY effective_from DESC,version DESC LIMIT 1',
                       (active[0],family,str(today_india()))).fetchone()
        if not row or row['version']!=1 or expected.get(row['policy_id'])!=row['source_hash']: return False
        if row['effective_until'] and row['effective_until']<=str(today_india()): return False
    return True


def operator(user):
    if user.role!='operator': raise HTTPException(403,'Operator access required')


def proposal(db,user,proposal_id):
    row=db.execute('SELECT * FROM action_proposals WHERE id=?',(proposal_id,)).fetchone()
    if row is None: raise HTTPException(404,'Proposal not found')
    require_account_access(user,row['account_id'])
    return dict(row)


def synchronize_ticket(db,investigation_id):
    states=[r['status'] for r in db.execute('SELECT status FROM action_proposals WHERE investigation_id=?',(investigation_id,))]
    status='escalated' if any(s in ('rejected','stale') for s in states) else 'pending_approval' if any(s in ('pending','approved') for s in states) else 'resolved'
    if status=='resolved':
        account_id=db.execute('SELECT account_id FROM investigations WHERE id=?',(investigation_id,)).fetchone()[0]
        if unresolved(account_snapshot(db,account_id)):status='escalated'
    db.execute('UPDATE tickets SET status=? WHERE id=(SELECT ticket_id FROM investigations WHERE id=?)',(status,investigation_id))


class ActionService:
    def __init__(self,path): self.path=path

    def list(self,user):
        with connect(self.path) as db:
            rows=db.execute('SELECT * FROM action_proposals ORDER BY created_at DESC,id LIMIT 200') if user.role=='operator' else db.execute(
                'SELECT * FROM action_proposals WHERE account_id=? ORDER BY created_at DESC,id LIMIT 200',(user.account_id,))
            return [{k:r[k] for k in r.keys() if k!='policy_json'} for r in rows]

    def decide(self,user,proposal_id,decision,note):
        operator(user)
        with connect(self.path) as db:
            db.execute('BEGIN IMMEDIATE')
            item=proposal(db,user,proposal_id)
            if db.execute('SELECT status FROM investigations WHERE id=?',(item['investigation_id'],)).fetchone()[0]!='completed':
                raise HTTPException(409,'A completed investigation is required')
            if item['status']!='pending': raise HTTPException(409,'Only a pending proposal can be decided')
            status='approved' if decision=='approve' else 'rejected'
            db.execute('UPDATE action_proposals SET status=?,approved_by=?,decision_note=?,decided_at=CURRENT_TIMESTAMP WHERE id=?',
                       (status,user.id,note,proposal_id))
            audit(db,user,item['account_id'],item['investigation_id'],'proposal_'+status,{'proposal_id':proposal_id,'note':note})
            synchronize_ticket(db,item['investigation_id'])
            return proposal(db,user,proposal_id) | {'policy_json':None}

    def execute(self,user,proposal_id,key):
        operator(user)
        digest=hashlib.sha256(json.dumps({'proposal_id':proposal_id,'operation':'execute'},sort_keys=True).encode()).hexdigest()
        with connect(self.path) as db:
            db.execute('BEGIN IMMEDIATE')
            cached=db.execute('SELECT * FROM action_requests WHERE actor_id=? AND idempotency_key=?',(user.id,key)).fetchone()
            if cached:
                if cached['request_hash']!=digest: raise HTTPException(409,'Idempotency key was already used for another request')
                return json.loads(cached['result_json']) | {'replayed':True}
            item=proposal(db,user,proposal_id)
            if item['status']!='approved': raise HTTPException(409,'Execution requires an approved, unexecuted proposal')
            candidates=eligible_actions(account_snapshot(db,item['account_id']))
            eligible=policies_current(db,item) and any(all(c[k]==item[k] for k in ('kind','target_id','amount_minor')) for c in candidates)
            if not eligible:
                db.execute("UPDATE action_proposals SET status='stale' WHERE id=?",(proposal_id,))
                result={'proposal_id':proposal_id,'status':'stale','simulated':True,'message':'Account facts or active policies changed; run a new investigation.','replayed':False}
                audit(db,user,item['account_id'],item['investigation_id'],'execution_blocked',result)
            else:
                result={'proposal_id':proposal_id,'status':'executed','simulated':True,'kind':item['kind'],'amount_minor':item['amount_minor'],'replayed':False}
                if item['kind']=='refund':
                    refund_id='ref_'+uuid.uuid4().hex
                    db.execute('INSERT INTO refunds(id,proposal_id,payment_id,account_id,amount_minor) VALUES (?,?,?,?,?)',
                               (refund_id,proposal_id,item['target_id'],item['account_id'],item['amount_minor']))
                    result['refund_id']=refund_id
                else:
                    for feature in ('team_workspace','advanced_reports'):
                        db.execute('INSERT INTO entitlements(account_id,feature,enabled) VALUES (?,?,1) ON CONFLICT(account_id,feature) DO UPDATE SET enabled=1',(item['account_id'],feature))
                db.execute("UPDATE action_proposals SET status='executed',executed_at=CURRENT_TIMESTAMP WHERE id=?",(proposal_id,))
                audit(db,user,item['account_id'],item['investigation_id'],'simulated_action_executed',result)
            synchronize_ticket(db,item['investigation_id'])
            db.execute('INSERT INTO action_requests(actor_id,idempotency_key,request_hash,result_json) VALUES (?,?,?,?)',
                       (user.id,key,digest,json.dumps(result)))
            return result

    def history(self,user,account_id):
        require_account_access(user,account_id)
        with connect(self.path) as db:
            if not db.execute('SELECT 1 FROM accounts WHERE id=?',(account_id,)).fetchone(): raise HTTPException(404,'Account not found')
            return [{**{k:r[k] for k in r.keys() if k!='data_json'},'data':json.loads(r['data_json'])} for r in db.execute(
                'SELECT * FROM audit_events WHERE account_id=? ORDER BY id DESC LIMIT 200',(account_id,))]
