# Application verification — 2026-11-01

## Implemented and exercised locally

- Full automated suite: **81 passed**. A final focused run after unifying billing balances: **28 passed**. Python compilation, JavaScript syntax checking and Git whitespace checks passed.
- Isolated development resolution evaluation: **4/4 passed**, using the real active policy index copied into temporary seeded databases. Healthy, pending, duplicate/access and injected-complaint cases behaved as specified. No live account state was changed by that evaluator; no generator API calls were made.
- Real MCP stdio transport: initialization, four-tool discovery, authorized account/billing reads, cross-account denial and denial after session revocation passed.
- Browser: customer sign-in, invoice/account figures, case creation with real policy retrieval, operator sign-in, separate approval/execution of both proposed fixes and updated current account figures were verified.
- Asha's main local demo now contains a completed simulated refund/access repair: net paid 99900 paise, excess zero and both Pro features enabled. The initial investigation and its evidence remain stored; seeding does not undo these changes.
- Narrow 390-pixel layout was inspected; mobile sign-out is present. Desktop screenshot: [workspace](screenshots/workspace.png).
- Local read-only HTTP sample: **24 requests**, concurrency **4**, **0 errors**, p50 **33.73 ms**, p95 **51.14 ms**. This sample includes loopback HTTP overhead and excludes LLM/dense-search work; it is not a capacity claim.

Reports: [isolated resolution](../evaluation/results/resolution.json), [local HTTP sample](../evaluation/results/local-http.json), [final live application state](../evaluation/results/application-live.json), [earlier partial Gemini check](../evaluation/results/part4-partial-live.json).

## What the automated checks establish

The suite exercises authentication/account boundaries, schema integrity, retrieval/versioning, citation format/quotes, memory isolation, failure handling, read-only tools, persisted investigation stages, approval permissions, changed evidence, rejected/stale proposals, concurrent execution, idempotent replay, audit append-only triggers and ticket outcomes. Additional cases cover old paid invoices with a newer unpaid invoice, unhandled partial excess and missing entitlement rows.

A throttle regression test now freezes its clock before issuing attempts. Its previous failure occurred when real wall time crossed the 60-second window under machine load; the fixed-clock test isolates the intended window behaviour without changing runtime throttling.

## Limits and unverified claims

- Docker/Compose and GitHub Actions configuration are included. Docker is unavailable locally, so no local container build/run is claimed. CI has not run remotely because these changes have not been pushed.
- Gemini produced a correctly qualified cited refund-policy answer in earlier live checks. Provider timeouts prevented completing that earlier follow-up/unsupported-question exercise; the final bounded retry also timed out on its first question. Current provider availability remains a demo limitation. See [answer verification](VERIFIED_ANSWERS.md) for live evidence and updates.
- Quote integrity is not semantic entailment. Deterministic provider tests do not establish real-model answer quality.
- The graph calls shared read tools directly. MCP is a separately verified external read-only interface, not an internal network hop. The graph is a bounded deterministic investigation workflow, not an autonomous Gemini planner.
- No real payment provider, real money, distributed queue, cryptographically tamper-proof audit system, autonomous multi-agent comparison, neural reranker or completed support-model fine-tuning run is claimed.
- This is a single-worker, local synthetic-data application. Demo credentials are public; production security/provisioning/deployment are separate work.
