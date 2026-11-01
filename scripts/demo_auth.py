"""Live Part 2 smoke test. Prints results, never passwords or session tokens."""
import json
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nexus.auth import DEMO_CUSTOMER_PASSWORD, DEMO_OPERATOR_PASSWORD

BASE = 'http://127.0.0.1:8766'

def call(path, body=None, token=None):
    headers={}
    if token:
        headers['Authorization']='Bearer '+token
    if body is not None:
        headers['Content-Type']='application/json'
    request=Request(BASE+path,data=json.dumps(body).encode() if body is not None else None,headers=headers)
    try:
        with urlopen(request,timeout=15) as response:
            return response.status,json.load(response)
    except HTTPError as error:
        return error.code,json.load(error)

def main():
    results={}
    results['anonymous_account_read']=call('/api/accounts')[0]
    status,customer=call('/api/auth/login',{'email':'asha@example.test','password':DEMO_CUSTOMER_PASSWORD})
    assert status==200
    token=customer['access_token']
    results['customer_own_invoices']=call('/api/accounts/acct_duplicate/invoices',token=token)[0]
    results['customer_other_invoices']=call('/api/accounts/acct_healthy/invoices',token=token)[0]
    results['customer_operator_overview']=call('/api/demo/scenarios',token=token)[0]
    status,operator=call('/api/auth/login',{'email':'support@example.test','password':DEMO_OPERATOR_PASSWORD})
    assert status==200
    operator_token=operator['access_token']
    status,accounts=call('/api/accounts',token=operator_token)
    results['operator_account_count']=len(accounts['accounts'])
    results['operator_other_invoices']=call('/api/accounts/acct_healthy/invoices',token=operator_token)[0]
    results['logout']=call('/api/auth/logout',{},token)[0]
    results['revoked_session']=call('/api/auth/me',token=token)[0]
    call('/api/auth/logout',{},operator_token)
    expected={'anonymous_account_read':401,'customer_own_invoices':200,'customer_other_invoices':404,
              'customer_operator_overview':403,'operator_account_count':3,'operator_other_invoices':200,
              'logout':200,'revoked_session':401}
    assert results==expected,results
    print(json.dumps(results,indent=2))

if __name__=='__main__':
    main()
