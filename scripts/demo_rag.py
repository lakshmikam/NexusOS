"""Explicit live Gemini smoke test through the local API; no credentials are printed."""
import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request,urlopen
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nexus.auth import DEMO_CUSTOMER_PASSWORD

BASE='http://127.0.0.1:8766'

def call(path,body=None,token=None,stream=False):
    headers={}
    if token: headers['Authorization']='Bearer '+token
    if body is not None: headers['Content-Type']='application/json'
    request=Request(BASE+path,data=json.dumps(body).encode() if body is not None else None,headers=headers)
    try:
        with urlopen(request,timeout=180) as response:
            if stream: return response.status,[json.loads(line) for line in response if line.strip()]
            return response.status,json.load(response)
    except HTTPError as error:
        return error.code,json.load(error)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true',required=True,help='Makes three requests to the configured Gemini model; consumes provider quota')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    tokens=[]
    try:
        status,login=call('/api/auth/login',{'email':'asha@example.test','password':DEMO_CUSTOMER_PASSWORD})
        assert status==200
        token=login['access_token'];tokens.append(token)
        status,configuration=call('/api/rag/status',token=token)
        if not configuration.get('configured'):
            raise RuntimeError('Configure Gemini in the local .env file before the live test')
        status,conversation=call('/api/conversations',{'title':'Part 4: live policy and memory demo'},token)
        assert status==201
        conversation_id=conversation['id']
        base={'method':'bm25','as_of':'2026-10-31','conversation_id':conversation_id}
        question='What evidence and approval are required for a duplicate-charge refund?'
        status,events=call('/api/rag/stream',{**base,'question':question},token,True)
        assert status==200
        terminal=events[-1]
        if terminal['type']=='error': raise RuntimeError(terminal['code']+': '+terminal['message'])
        assert terminal['type']=='done'
        first=terminal['result']
        kinds=[e['type'] for e in events]
        assert kinds.index('validated')<kinds.index('answer_delta')
        assert first['status']=='answered'
        status,followup=call('/api/rag/answer',{**base,'question':'Does that mean a refund has already been issued to me?'},token)
        if status!=200: raise RuntimeError(str(followup))
        assert followup['context_info']['included_history_turns']==1
        status,unsupported=call('/api/rag/answer',{'question':'Does OrbitDesk support exporting invoices to XML accounting files?','method':'bm25','as_of':'2026-10-31'},token)
        if status!=200: raise RuntimeError(str(unsupported))
        status,stored=call('/api/conversations/'+conversation_id,token=token)
        assert status==200 and len(stored['turns'])==2
        status,other=call('/api/auth/login',{'email':'meera@example.test','password':DEMO_CUSTOMER_PASSWORD})
        assert status==200
        other_token=other['access_token'];tokens.append(other_token)
        assert call('/api/conversations/'+conversation_id,token=other_token)[0]==404
        report={'label':'Three synthetic live development cases; manually review meaning, not only citation format.',
                'provider':configuration['provider'],'model':configuration['model'],'conversation_id':conversation_id,
                'stream_event_types':kinds,'first':first,'followup':followup,'unsupported':unsupported,
                'cross_user_access':'404','stored_completed_turns':len(stored['turns'])}
        if args.out:
            args.out.parent.mkdir(parents=True,exist_ok=True)
            args.out.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps({'model':configuration['model'],'conversation_id':conversation_id,
                         'first_status':first['status'],'first_answer':first['answer'],
                         'followup_status':followup['status'],'followup_answer':followup['answer'],
                         'included_history_turns':followup['context_info']['included_history_turns'],
                         'unsupported_status':unsupported['status'],'unsupported_answer':unsupported['answer'],
                         'stored_completed_turns':2,'cross_user_access':404},indent=2))
    finally:
        for token in tokens: call('/api/auth/logout',{},token)

if __name__=='__main__': main()
