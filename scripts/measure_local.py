"""Bounded read-only local HTTP measurement. Not a production load benchmark."""
import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path
from urllib.request import Request,urlopen
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nexus.auth import DEMO_CUSTOMER_PASSWORD


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests',type=int,default=24)
    parser.add_argument('--concurrency',type=int,default=4)
    parser.add_argument('--out',type=Path,default=Path('evaluation/results/local-http.json'))
    args=parser.parse_args()
    if not 1<=args.requests<=100 or not 1<=args.concurrency<=8:parser.error('Use 1..100 requests and 1..8 concurrency')
    base='http://127.0.0.1:8766'
    body=json.dumps({'email':'asha@example.test','password':DEMO_CUSTOMER_PASSWORD}).encode()
    with urlopen(Request(base+'/api/auth/login',data=body,headers={'Content-Type':'application/json'}),timeout=15) as response:
        token=json.load(response)['access_token']
    headers={'Authorization':'Bearer '+token}
    def measure(_):
        start=time.perf_counter()
        with urlopen(Request(base+'/api/tools/billing/acct_duplicate',headers=headers),timeout=15) as response:
            data=json.load(response)
        assert data['money_unit']=='paise'
        return (time.perf_counter()-start)*1000
    try:
        started=time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            values=sorted(pool.map(measure,range(args.requests)))
        elapsed=time.perf_counter()-started
        report={'label':'Local loopback read-only billing requests; includes HTTP overhead, no LLM or dense search; not production capacity.',
                'requests':len(values),'concurrency':args.concurrency,'p50_ms':round(values[len(values)//2],2),
                'p95_ms':round(values[min(len(values)-1,int(len(values)*.95))],2),'max_ms':round(max(values),2),
                'total_seconds':round(elapsed,3),'errors':0}
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))
    finally:
        with urlopen(Request(base+'/api/auth/logout',data=b'{}',headers={**headers,'Content-Type':'application/json'}),timeout=15):pass


if __name__=='__main__':main()
