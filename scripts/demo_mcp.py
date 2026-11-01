"""Live MCP stdio integration check against the local API; no generation calls."""
import asyncio
import json
import os
import sys
from pathlib import Path
from urllib.request import Request,urlopen
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nexus.auth import DEMO_CUSTOMER_PASSWORD
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client


def api(path,body,token=None):
    headers={'Content-Type':'application/json'}
    if token:headers['Authorization']='Bearer '+token
    with urlopen(Request('http://127.0.0.1:8766'+path,data=json.dumps(body).encode(),headers=headers),timeout=15) as response:
        return json.load(response)


async def main():
    login=api('/api/auth/login',{'email':'asha@example.test','password':DEMO_CUSTOMER_PASSWORD})
    token=login['access_token']
    try:
        params=StdioServerParameters(command=sys.executable,args=['-m','nexus.mcp_server'],
            env={**os.environ,'NEXUS_SESSION_TOKEN':token,'NEXUS_API_URL':'http://127.0.0.1:8766'})
        async with stdio_client(params) as (read,write):
            async with ClientSession(read,write) as session:
                await session.initialize()
                names=[t.name for t in (await session.list_tools()).tools]
                assert set(names)=={'inspect_account','inspect_billing','search_policies','inspect_investigation'}
                account=await session.call_tool('inspect_account',{'account_id':'acct_duplicate'})
                assert not account.isError
                billing=await session.call_tool('inspect_billing',{'account_id':'acct_duplicate'})
                assert not billing.isError
                denied=await session.call_tool('inspect_account',{'account_id':'acct_healthy'})
                assert denied.isError
                api('/api/auth/logout',{},token)
                revoked=await session.call_tool('inspect_account',{'account_id':'acct_duplicate'})
                assert revoked.isError
                print(json.dumps({'tools':names,'authorized_reads':'passed','cross_account_denial':'passed','revoked_session_denial':'passed'}))
    finally:
        from urllib.error import HTTPError
        try: api('/api/auth/logout',{},token)
        except HTTPError as error:
            if error.code!=401: raise


if __name__=='__main__':asyncio.run(main())
