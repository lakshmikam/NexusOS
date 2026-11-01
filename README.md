# NexusOS — AI Support Resolution System

An evidence-backed support system for **OrbitDesk**, a fictional subscription product. The workflow investigates a billing/access complaint using account records and company policies, records the findings, proposes eligible simulated actions for operator approval, or escalates.

## Current implementation: Parts 1–8

Open the [local workspace](http://127.0.0.1:8766/) for the customer/operator interface. Start an investigation from Overview, inspect findings and sources, then use the operator identity to review and explicitly execute eligible simulated fixes. [Demo guide](docs/DEMO.md) · [Learning guide](docs/learning-log.md) · [Verified application results](docs/VERIFIED_APPLICATION.md).

- FastAPI application with interactive API documentation.
- Transactional SQLite schema migration and repeatable synthetic demo seeding.
- Accounts, current subscriptions, invoices, payment attempts, entitlements and tickets.
- Invoice responses distinguish successful charges from failed or pending attempts.
- Three reproducible support scenarios and persistent ticket creation.
- Versioned fictional policy documents for later retrieval.
- Seeded customer/operator identities with salted scrypt password hashes.
- Expiring opaque bearer sessions, hashed token storage and logout/revocation.
- Customer account/ticket isolation and explicit operator permissions.
- Basic login-attempt throttling and generic invalid-credential responses.
- Versioned policy ingestion with heading-aware or fixed-window token-bounded chunks.
- Local 384-dimensional sentence embeddings, inspectable BM25 and dense search, and reciprocal rank fusion.
- Date/family filtering before ranking, immutable index snapshots and source provenance.
- A twelve-question development evaluation and a measured chunking comparison.
- Gemini policy answers with structured claims, checked source IDs and exact supporting quotes.
- Validation-first answer streaming, provider error handling and insufficient-evidence responses.
- Private persistent conversations with bounded recent history, fresh retrieval and restart recovery.
- A four-node bounded LangGraph investigation with persisted steps, verified account facts and policy evidence.
- Read-only account tools plus an official MCP stdio adapter using existing expiring API sessions.
- Operator review, separate execution, transactional eligibility/policy revalidation and idempotent simulated actions.
- Refund ledger, entitlement repair, ticket outcomes and append-only audit events.
- Responsive customer/operator workspace, saved policy conversations and inspectable prompt context.
- Isolated resolution evaluation, bounded HTTP measurements, request metrics, Docker/Compose and CI configuration.
- Automated tests covering billing, sessions, access boundaries, retrieval, generation validation and memory.

This is a local synthetic-data development application. Gemini explains policies; a separate bounded LangGraph workflow reads account facts and applies explicit backend eligibility rules. The workflow calls shared tools directly; the MCP stdio server exposes read-only tools to external clients through authenticated HTTP. No autonomous LLM planner or LLM-selected money movement is claimed. Simulated changes require separate operator approval and execution. The publicly documented credentials are only for fictional local accounts; do not use them with real data or public hosting. Docker and CI files are included, but their execution has not been verified on this machine.

The existing `experiments/sft/` code is preserved separately. It contains LoRA, quantization, dataset preparation, QLoRA training, evaluation and serving experiments. The presence of a script is not evidence of a completed model-training run.

## Run

Python 3.10+:

```powershell
python -m pip install -r requirements-app.txt
python -m uvicorn nexus.api:app --host 127.0.0.1 --port 8766
```

Open the [workspace](http://127.0.0.1:8766/) or [API documentation](http://127.0.0.1:8766/docs). The application initializes `data/nexus.db` and seeds demo data at startup without resetting existing changes. Set `NEXUS_DB_PATH` to select another SQLite file. Run one server worker. Policy indexing is a separate explicit setup step below.

Manual initialization is also available:

```powershell
python -m nexus.db --seed
python -m pytest -q
```

See [requests.http](requests.http) for example requests, [Part 1 learning notes](docs/part-1-learning.md) for the database walkthrough, and [Part 2 learning notes](docs/part-2-learning.md) for authentication and access control.

### Set up local policy retrieval

```powershell
python -m pip install -r requirements-retrieval.txt
# Explicit download if the pinned model is not already cached:
python -m nexus.retrieval download-model
python -m nexus.retrieval index
```

The encoder is `sentence-transformers/all-MiniLM-L6-v2`, pinned to revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`. Normal indexing/search uses local files only. It creates 384-dimensional normalized vectors and uses a 180-token chunk budget including title/heading. It rejects text that would exceed the encoder limit instead of silently truncating it.

After login and authorization in `/docs`, try `POST /api/retrieval/search`:

```json
{"query":"I was charged twice. What evidence is needed for a refund?","method":"hybrid","top_k":3,"as_of":"2026-10-31"}
```

Change `method` to `bm25` or `dense` and inspect the returned source passages, versions, scores and ranks. Search returns candidates, not generated answers; scores are not probabilities. Shared policies contain no private account records. The first dense query lazily loads the model and can be slower than subsequent queries.

`GET /api/retrieval/chunks` shows the stored passage snapshots. `GET /api/retrieval/status` reports the active index. Operator-only reindex/evaluation endpoints are available. SQLite persists vectors as JSON and NumPy performs exact similarity search over eligible chunks; no external vector database is required for this small corpus.

```powershell
python -m nexus.retrieval compare --out data/retrieval-comparison.json
python scripts/demo_retrieval.py
```

Comparison builds unpublished snapshots without switching the live index. [Part 3 learning notes](docs/part-3-learning.md) explain each concept, alternatives and trade-offs. [Measured results](docs/VERIFIED_RETRIEVAL.md) describe the development comparison and its limits.

### Connect Gemini and try an answer

Copy `.env.example` to `.env` only if you do not already have a local configuration, and set `GEMINI_API_KEY` there. The real `.env` is ignored by Git. The configured model is `gemini-3.8-flash`; select a model available to your own Gemini account through `NEXUS_LLM_MODEL`. The adapter sends requests to Google's Gemini API, including the question, selected shared policies and selected conversation history. It does not send the account database.

After login, `GET /api/rag/status` reports configuration without revealing the key. Use `POST /api/rag/preview` to inspect the assembled context without a generation call or saved turn. Then call `/api/rag/answer` or `/api/rag/stream`:

```json
{"question":"What evidence and approval are required for a duplicate-charge refund?","method":"bm25","as_of":"2026-10-31"}
```

Create a conversation with `POST /api/conversations` and include its returned `id` as `conversation_id` on answer requests to enable memory. Stored conversations belong to their creator, including for operators. Only the last three completed turns, within 4,000 serialized characters, enter the next prompt. Sources have a separate 6,000-character budget; these are application budgets, not exact Gemini token counts. Retrieval runs again on each question. There is no automatic history summary or training on conversations.

The stream uses newline-delimited JSON. Progress events arrive while the provider generates, but answer text is buffered until schema, source-ID and quote checks pass. This delays the first answer text. A matching quote does **not** prove that a claim follows from it; semantic correctness still needs evaluation. There is no automatic retry on quota/provider errors. Run this development server with one worker: startup recovery marks unfinished turns failed.

```powershell
# Explicitly makes three real generation requests using your provider quota:
python scripts/demo_rag.py --live --out data/part4-live-smoke.json
```

[Part 4 learning notes](docs/part-4-learning.md) explain prompts, context, citations, streaming, evaluation and memory. [Answer verification](docs/VERIFIED_ANSWERS.md) separates automated checks from the small live development exercise.

### Login in the interactive documentation

1. Use `POST /api/auth/login` with a demo email/password from the table below.
2. Copy the response's `access_token`.
3. Click **Authorize** at the top of `/docs`, paste the token into **SessionToken**, and authorize. The UI adds the `Bearer` prefix.
4. Use `/api/auth/me`, then inspect the permitted accounts/invoices/tickets.
5. Use `POST /api/auth/logout` to revoke the current token.

| User | Email | Demo password | Access |
|---|---|---|---|
| Asha | `asha@example.test` | `OrbitDesk-Demo-2026!` | `acct_duplicate` only |
| Meera | `meera@example.test` | `OrbitDesk-Demo-2026!` | `acct_healthy` only |
| Kabir | `kabir@example.test` | `OrbitDesk-Demo-2026!` | `acct_pending` only |
| Support operator | `support@example.test` | `OrbitDesk-Operator-2026!` | All demo accounts and scenario overview |

Login sessions default to one hour. The database stores only a SHA-256 digest of the random token. Restarting preserves unexpired sessions; logout and user deactivation invalidate access. Password hashes use independent random salts and scrypt. Seeding does not overwrite existing users' passwords.

Five failed logins for the same normalized email/client address within one minute trigger 429 with `Retry-After`. This is a local safeguard, not production-grade distributed abuse protection. There is no signup/password-reset flow yet.

To verify the running server without printing credentials or tokens:

```powershell
python scripts/demo_auth.py
```

## The demo cases

OrbitDesk Pro costs **INR 999/month** in the simulation. Database amounts are integer paise: `99900` means INR 999.00. No real money moves.

| Account | Billing facts | Subscription/access |
|---|---|---|
| `acct_healthy` | One invoice and one successful payment | Active Pro, both features enabled |
| `acct_pending` | One invoice, one pending attempt and one failed attempt | Pending subscription, features disabled |
| `acct_duplicate` | One invoice and two distinct successful payments | Active Pro, features incorrectly disabled |

The third case starts from: **"I upgraded yesterday, was charged twice, and still cannot access the paid features."** These are fixed fictional fixtures; seed timestamps are not shifted daily.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Database readiness and current phase |
| POST | `/api/auth/login` | Verify credentials and issue a session |
| GET | `/api/auth/me` | Verified user's public profile |
| POST | `/api/auth/logout` | Revoke the current session |
| GET | `/api/demo/scenarios` | Expected facts for all three cases (operator only) |
| GET | `/api/accounts` | Own account for customers; all accounts for operators |
| GET | `/api/accounts/{id}` | Account, subscription and entitlements |
| GET | `/api/accounts/{id}/invoices` | Invoices, payment attempts and confirmed amounts |
| POST | `/api/tickets` | Create an open support ticket |
| GET | `/api/tickets/{id}` | Retrieve a saved ticket |
| GET | `/api/accounts/{id}/tickets` | List an account's tickets |
| GET | `/api/retrieval/status` | Active policy index configuration |
| GET | `/api/retrieval/chunks` | Eligible shared policy passages |
| POST | `/api/retrieval/search` | BM25, dense or hybrid policy search |
| POST | `/api/retrieval/reindex` | Build/publish an index (operator only) |
| POST | `/api/retrieval/evaluate` | Run development evaluation (operator only) |
| GET | `/api/rag/status` | Safe provider status and context budgets |
| POST | `/api/rag/preview` | Inspect prompt/context without generation |
| POST | `/api/rag/answer` | Generate and validate a policy answer |
| POST | `/api/rag/stream` | Stream progress, then checked answer text as NDJSON |
| POST | `/api/conversations` | Create a private conversation |
| GET | `/api/conversations` | List your conversations |
| GET | `/api/conversations/{id}` | Read your persisted turns |
| GET | `/api/tools/account/{id}` | Authorized subscription/access tool |
| GET | `/api/tools/billing/{id}` | Authorized invoices, attempts and refund ledger |
| POST | `/api/investigations` | Run a bounded investigation and propose eligible fixes |
| GET | `/api/investigations` | List authorized cases |
| GET | `/api/investigations/{id}` | Persisted evidence, steps and proposals |
| GET | `/api/proposals` | Authorized proposed actions |
| POST | `/api/proposals/{id}/decision` | Approve or reject with a note (operator) |
| POST | `/api/proposals/{id}/execute` | Revalidate and execute a simulated fix (operator) |
| GET | `/api/accounts/{id}/audit` | Account-scoped audit history |
| GET | `/api/system/status` | Readiness, counters and saved usage (operator) |

All business-record routes require a live session. Customer reads and ticket creation enforce ownership; operators have explicit cross-account investigation access. Inaccessible account/ticket IDs return the same 404 as missing records. The operator-only overview returns 403 for customers. Health, login and API documentation are public.

## Database decisions

- An **invoice** records what is owed; **payments** record individual attempts.
- Only `succeeded` payments contribute to the confirmed paid amount.
- A subscription's plan/status and actual feature entitlements are separate facts.
- A composite foreign key prevents associating a payment with another account's invoice. This is a data-integrity constraint; it does not replace request authorization.
- Users have a trusted role and account relationship. Session authentication retrieves these from the database; a caller cannot grant themselves operator permissions through request fields.
- Payment provider references and refunded payment IDs are unique. Execution idempotency records and action changes commit together. Refunds preserve historical payments and reduce net paid through a separate simulated ledger.
- Integer money values, status checks, foreign keys and transactional migrations protect basic consistency.
- Seeding uses stable IDs and does not reset existing records. It is intended for a fresh demo database, not as a production synchronization mechanism.

## Implementation plan

1. **Product, database and first API — implemented.**
2. **Login, expiring sessions and account authorization — implemented.**
3. **Versioned policies, chunking, BM25, dense/hybrid retrieval and development evaluation — implemented.**
4. **Cited RAG responses, validation-first streaming, insufficient-evidence handling and bounded conversation memory — implemented.**
5. **Account tools, read-only MCP adapter, bounded LangGraph workflow and persisted investigations — implemented.**
6. **Operator review, simulated actions, revalidation, idempotency and audit history — implemented.**
7. **Customer/operator interface and synthetic end-to-end resolution evaluation — implemented.**
8. **Health/request metrics, local measurement, demo/learning notes and Docker/CI configuration — implemented.** Container build/runtime and remote CI verification remain pending because Docker is unavailable locally and these changes have not been pushed.

Reranking, an investigator/reviewer comparison, an autonomous LLM tool planner and a measured new fine-tuning experiment are follow-up extensions, not implemented claims.

## Verification commands

```powershell
python -m pytest -q
python scripts/evaluate_resolution.py
python scripts/demo_mcp.py
python scripts/measure_local.py
node --check nexus/static/app.js
```

The resolution evaluator uses temporary accounts and the existing active real policy index; it does not change the running demo or call Gemini. MCP/HTTP scripts require the running local API. See [Part 8](docs/part-8-learning.md) for Docker setup and the measurement limits. Real Gemini calls require explicit `python scripts/demo_rag.py --live` and consume provider quota; earlier follow-up checks encountered provider timeouts.
