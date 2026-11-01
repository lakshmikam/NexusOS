# Learning path

We will learn one part at a time after implementation. Each session: explain the user problem in plain language, walk a concrete example, trace the relevant request/data, explain the code, compare alternatives and trade-offs, then practise explaining a failure case. No prerequisite knowledge is assumed.

| Part | Guide | Concepts |
|---|---|---|
| 1 | [Database and API](part-1-learning.md) | HTTP, FastAPI, tables, keys, joins, money units, migrations |
| 2 | [Identity and ownership](part-2-learning.md) | Password hashes, sessions, authentication vs authorization |
| 3 | [Retrieval](part-3-learning.md) | Chunks, tokens, BM25, embeddings, cosine, hybrid RRF and evaluation |
| 4 | [Answers and memory](part-4-learning.md) | Prompts, context, hallucinations, citations, streaming, bounded history |
| 5 | [Investigation](part-5-learning.md) | Tools, MCP, LangGraph state/nodes/edges, verified facts, persistence |
| 6 | [Controlled actions](part-6-learning.md) | Approval, revalidation, transactions, races, idempotency, audit |
| 7 | [Interface and evaluation](part-7-learning.md) | DOM, fetch, streaming parsing, frontend/backend boundaries, full scenarios |
| 8 | [Packaging and operations](part-8-learning.md) | Docker, volumes, CI, health, latency percentiles, usage/cost limits |

Start the next teaching session with Part 5. Parts 1–4 notes remain available for revision. [Demo instructions](DEMO.md) and [verification record](VERIFIED_APPLICATION.md) distinguish implemented behaviour from unverified deployment or model-quality claims.
