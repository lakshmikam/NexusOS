# Part 8 — reproducibility and operations

## What packaging does

The application needs Python dependencies, policy files, static assets, a database and optionally a cached embedding model plus Gemini configuration. A Docker image packages the executable environment and source. A running container is an instance of that image. Volumes preserve SQLite data and downloaded model files outside the disposable container filesystem.

The Dockerfile uses Python 3.11, CPU PyTorch, application/retrieval dependencies and a non-root application user. It does not copy the real .env or existing data. Compose passes local configuration at runtime, mounts named volumes and binds the port to loopback only. The publicly documented synthetic demo identities are not suitable for public real-data hosting.

Docker is not installed on the development machine, so this configuration has not been built/run locally. CI includes a container build and health check, but no remote CI run is claimed until the changes are pushed and that workflow runs.

## Running with Docker when available

Create a local .env from the example only if one does not already exist; keep the real key out of Git. Stop any other server using port 8766, then run:

```powershell
docker compose up --build -d
# Explicit model download; requires access to Hugging Face:
docker compose exec nexus python -m nexus.retrieval download-model
docker compose exec nexus python -m nexus.retrieval index
docker compose logs --tail 50 nexus
```

Open http://127.0.0.1:8766/. The app can start before an index exists, but investigation/retrieval cannot succeed until indexing is complete. Normal model use is offline after download. Gemini answers require provider network access and a configured key. Use one application worker: startup recovery of unfinished jobs is designed for that mode.

## CI

Continuous integration runs automated checks for source changes. Our workflow installs lightweight application dependencies, runs deterministic tests without Gemini credentials or model downloads, compiles Python and syntax-checks JavaScript. A separate job builds the CPU container and checks HTTP readiness. Full dense-model quality and external API availability are not established by those checks.

## Health, latency and cost

Health confirms database readiness; it does not prove Gemini or policy retrieval is ready. The operator-only system status endpoint reports policy index state, sanitized provider configuration, record counts, request/error counts, recent request latency and token usage from up to 100 saved completed conversation turns. Process counters reset on restart. Streaming request middleware timing measures response setup, not full generation duration; the RAG result separately records generation and total service duration.

The local measurement script sends bounded concurrent read-only HTTP requests and records p50/p95 latency. p95 means roughly 95% of the measured requests finished within that duration. Twenty-four loopback requests say little about internet latency, database write contention or production capacity. Do not extrapolate millions of transactions from this test.

The investigation workflow makes no generation API calls. Policy chat records actual provider token usage when supplied. Tokens are not a currency bill: thinking, cached input, model-specific pricing, failed calls and stateless calls complicate billing. The saved-turn aggregate deliberately does not claim full provider spending. Consult provider billing for actual charges; no invented cost estimate is supplied.

## Production extensions

A production service would need secure user provisioning, HTTPS, secret management, backups/restore testing, privacy/retention controls, monitored jobs, distributed abuse protection and deployment-specific security review. Moving to multiple workers requires leases or durable queue ownership instead of startup-wide recovery. PostgreSQL and a job queue are possible next steps when measured concurrency justifies them. These are documented extensions, not implemented capabilities of the local demo.
