"""Read-only, identity-bound tools shared by the workflow and API/MCP adapter."""
import hashlib
import json
from fastapi import HTTPException
from nexus.auth import require_account_access
from nexus.db import connect


def account_snapshot(db, account_id):
    account=db.execute('SELECT id,name,email FROM accounts WHERE id=?',(account_id,)).fetchone()
    if account is None: raise HTTPException(404,'Account not found')
    subscription=db.execute('SELECT * FROM subscriptions WHERE account_id=?',(account_id,)).fetchone()
    entitlements=[dict(r) for r in db.execute('SELECT feature,enabled FROM entitlements WHERE account_id=? ORDER BY feature',(account_id,))]
    invoices=[]
    for row in db.execute('SELECT * FROM invoices WHERE account_id=? ORDER BY issued_at,id',(account_id,)):
        invoice=dict(row)
        invoice['payments']=[dict(r) for r in db.execute('SELECT * FROM payments WHERE invoice_id=? AND account_id=? ORDER BY received_at,id',(row['id'],account_id))]
        invoice['refunds']=[dict(r) for r in db.execute('SELECT r.* FROM refunds r JOIN payments p ON p.id=r.payment_id WHERE p.invoice_id=? AND r.account_id=? ORDER BY r.id',(row['id'],account_id))]
        invoice['successful_paid_minor']=sum(p['amount_minor'] for p in invoice['payments'] if p['status']=='succeeded')
        invoice['refunded_minor']=sum(r['amount_minor'] for r in invoice['refunds'])
        invoice['net_paid_minor']=invoice['successful_paid_minor']-invoice['refunded_minor']
        invoice['excess_minor']=max(0,invoice['net_paid_minor']-invoice['amount_minor'])
        invoices.append(invoice)
    return {'account':dict(account),'subscription':dict(subscription) if subscription else None,
            'entitlements':entitlements,'invoices':invoices,'money_unit':'paise','currency':'INR'}


def eligible_actions(snapshot):
    """Backend rules, never LLM-generated payment amounts or permissions."""
    actions=[]
    for invoice in snapshot['invoices']:
        if invoice['status']!='paid': continue
        succeeded=[p for p in invoice['payments'] if p['status']=='succeeded']
        already={r['payment_id'] for r in invoice['refunds']}
        remaining=invoice['excess_minor']
        if len({p['provider_reference'] for p in succeeded})<2: continue
        # Full duplicate charges only; ambiguous partial amounts must be reviewed manually.
        for payment in reversed(succeeded):
            if payment['id'] not in already and payment['amount_minor']==invoice['amount_minor'] and remaining>=payment['amount_minor']:
                actions.append({'kind':'refund','target_id':payment['id'],'amount_minor':payment['amount_minor'],
                                'reason':'Distinct successful duplicate payment; refund leaves the invoice fully funded.'})
                remaining-=payment['amount_minor']
    sub=snapshot['subscription']
    missing={'team_workspace','advanced_reports'}-{e['feature'] for e in snapshot['entitlements'] if e['enabled']}
    relevant=max((i for i in snapshot['invoices'] if i['status']!='void'),key=lambda i:(i['issued_at'],i['id']),default=None)
    if sub and sub['plan']=='pro' and sub['status']=='active' and missing and relevant and (
        relevant['status']=='paid' and relevant['net_paid_minor']>=relevant['amount_minor']):
        actions.append({'kind':'restore_access','target_id':sub['id'],'amount_minor':0,
                        'reason':'Active paid Pro subscription has disabled Pro entitlements.'})
    return actions


def unresolved(snapshot):
    if not snapshot['invoices']: return True
    if any(i['net_paid_minor']!=i['amount_minor'] for i in snapshot['invoices'] if i['status']!='void'): return True
    sub=snapshot['subscription']
    return bool(sub and sub['plan']=='pro' and sub['status']=='active' and
                {'team_workspace','advanced_reports'}-{e['feature'] for e in snapshot['entitlements'] if e['enabled']})


class AccountTools:
    def __init__(self,path): self.path=path

    def snapshot(self,user,account_id):
        require_account_access(user,account_id)
        with connect(self.path) as db:
            # Consistent multi-table read snapshot.
            db.execute('BEGIN')
            result=account_snapshot(db,account_id)
        result['snapshot_hash']=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()
        return result

    def inspect_account(self,user,account_id):
        data=self.snapshot(user,account_id)
        return {k:data[k] for k in ('account','subscription','entitlements','snapshot_hash')}

    def inspect_billing(self,user,account_id):
        data=self.snapshot(user,account_id)
        return {k:data[k] for k in ('invoices','money_unit','currency','snapshot_hash')}
