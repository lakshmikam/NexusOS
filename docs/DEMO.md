# Run and demonstrate NexusOS

1. Install `requirements-app.txt` and `requirements-retrieval.txt` in the project environment. Explicitly download the pinned model if absent: `python -m nexus.retrieval download-model`.
2. Build policies with `python -m nexus.retrieval index`.
3. Keep Gemini credentials only in ignored `.env` using `.env.example` as a template. Investigations work without a generator key once policies are indexed; policy chat needs Gemini.
4. Start `python -m uvicorn nexus.api:app --host 127.0.0.1 --port 8766 --workers 1` and open http://127.0.0.1:8766/.

## Five-minute walkthrough

Sign in as Asha using the demo selector. Inspect her two successful payments, INR 999 excess and disabled Pro features. Run the prefilled complaint at policy date 2026-10-31. Open the findings and source passages: they are records/policies, not assumptions from complaint wording. Two proposed fixes should await approval on an unmodified demo database.

Sign out and choose Support operator. Select Asha if necessary, open Approvals and examine the investigation evidence. Enter a review note and approve each proposed action. Approval alone changes no money/access. Explicitly execute each simulated fix. Overview should then show INR 999 net paid, zero excess and two enabled features. Activity log records who investigated, reviewed and executed. The original case retains its initial findings while proposal/ticket states update.

Inspect Kabir: pending and failed attempts do not establish payment, so no refund is proposed and the case escalates. Inspect Meera: a healthy account produces no action.

Open Policy assistant for a cited rule explanation. Try the suggested evidence/approval question. Inspect prompt before generation to see the system instruction, source passages and recent history. Follow up in the same conversation, then reopen it from Saved conversations. Account investigation facts are not silently inserted into this policy-only chat. Gemini may return quota/network/timeouts; errors are explicit.

## Fresh replay without deleting history

Demo seeding does not undo simulated actions. Stop the current server, select a new database filename and index it before serving:

```powershell
$env:NEXUS_DB_PATH = Join-Path $PWD 'data/replay-01.db'
python -m nexus.db --seed
python -m nexus.retrieval index
python -m uvicorn nexus.api:app --host 127.0.0.1 --port 8766 --workers 1
```

Use an unused filename for another fresh replay. The isolated evaluation script provides a repeatable complete flow without modifying your main database.

## External MCP client

Launch `python -m nexus.mcp_server` from the project environment with `NEXUS_SESSION_TOKEN` set to an existing login token and `NEXUS_API_URL=http://127.0.0.1:8766`. Supply the token privately through the client's environment configuration, not command arguments or a committed file. It expires/revokes exactly like other API sessions. Four tools are discoverable: inspect_account, inspect_billing, search_policies, inspect_investigation. No approval/execution capability is exposed. `python scripts/demo_mcp.py` verifies the real stdio transport without printing credentials.
