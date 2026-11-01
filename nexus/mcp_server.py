"""Read-only MCP stdio server. Uses a real expiring NexusOS API session.

Run: python -m nexus.mcp_server
Set NEXUS_SESSION_TOKEN in the launching environment, never command-line arguments.
"""
import os
from urllib.parse import urlparse
import httpx
from mcp.server.fastmcp import FastMCP

mcp=FastMCP('NexusOS account investigation')


def request(path,body=None):
    base=os.environ.get('NEXUS_API_URL','http://127.0.0.1:8766').rstrip('/')
    parsed=urlparse(base)
    if parsed.scheme!='http' or parsed.hostname not in ('127.0.0.1','localhost') or parsed.username or parsed.password:
        raise ValueError('This local adapter requires a loopback HTTP API URL')
    token=os.environ.get('NEXUS_SESSION_TOKEN','')
    if not token: raise ValueError('Configure an expiring NEXUS_SESSION_TOKEN in the environment')
    try:
        with httpx.Client(timeout=20,follow_redirects=False) as client:
            response=client.request('POST' if body is not None else 'GET',base+path,
                headers={'Authorization':'Bearer '+token},json=body)
    except httpx.HTTPError:
        raise ValueError('NexusOS API is unreachable') from None
    if response.status_code!=200:
        raise ValueError(f'NexusOS denied or could not complete this tool request (HTTP {response.status_code})')
    return response.json()


def identifier(value):
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',value): raise ValueError('Invalid resource identifier')
    return value


@mcp.tool()
def inspect_account(account_id: str) -> dict:
    """Read authorized subscription and feature entitlements; does not change access."""
    return request('/api/tools/account/'+identifier(account_id))


@mcp.tool()
def inspect_billing(account_id: str) -> dict:
    """Read authorized invoices, payment attempts and simulated refunds. Money is integer paise."""
    return request('/api/tools/billing/'+identifier(account_id))


@mcp.tool()
def search_policies(query: str,as_of: str='2026-10-31') -> dict:
    """Search shared policies with BM25 at the specified ISO date; scores are not confidence."""
    return request('/api/retrieval/search',{'query':query,'method':'bm25','top_k':3,'as_of':as_of})


@mcp.tool()
def inspect_investigation(investigation_id: str) -> dict:
    """Read an already-persisted authorized investigation; never approve or execute actions."""
    return request('/api/investigations/'+identifier(investigation_id))


if __name__=='__main__': mcp.run(transport='stdio')
