"""Verify real local retrieval through authenticated HTTP requests; never print tokens."""
import json
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request,urlopen
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nexus.auth import DEMO_CUSTOMER_PASSWORD

BASE='http://127.0.0.1:8766'

def call(path,body=None,token=None):
    headers={}
    if token: headers['Authorization']='Bearer '+token
    if body is not None: headers['Content-Type']='application/json'
    request=Request(BASE+path,data=json.dumps(body).encode() if body is not None else None,headers=headers)
    try:
        with urlopen(request,timeout=120) as response:
            return response.status,json.load(response)
    except HTTPError as error:
        return error.code,json.load(error)

def main():
    query='I was charged twice. What evidence is needed for a refund?'
    assert call('/api/retrieval/search',{'query':query})[0]==401
    status,login=call('/api/auth/login',{'email':'asha@example.test','password':DEMO_CUSTOMER_PASSWORD})
    assert status==200
    token=login['access_token']
    summary={}
    try:
        for method in ('bm25','dense','hybrid'):
            status,response=call('/api/retrieval/search',{'query':query,'method':method,'as_of':'2026-10-31'},token)
            assert status==200,response
            assert response['score_is_probability'] is False
            assert any(r['policy_id']=='billing-v1' for r in response['results'])
            summary[method]=[{'rank':r['rank'],'policy_id':r['policy_id'],'heading':r['heading']} for r in response['results']]
        status,early=call('/api/retrieval/search',{'query':query,'as_of':'2026-09-01'},token)
        assert status==200 and early['status']=='no_applicable_policy'
        assert call('/api/retrieval/reindex',{},token)[0]==403
        summary['historical_filter']='no policy before the effective date'
        print(json.dumps(summary,indent=2))
    finally:
        call('/api/auth/logout',{},token)

if __name__=='__main__': main()
