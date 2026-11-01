# Part 5 — verified account investigation

## The problem this part solves

The policy assistant knows company rules but does not know whether a customer's claim is true. 'I paid twice' is a statement, not two confirmed payment records. This part reads the authorized account's actual synthetic records and records an investigation before proposing anything.

An investigation creates a support ticket and a case ID. Its event date selects applicable policies. A fixed LangGraph workflow then runs four nodes: inspect records, retrieve policies, assess evidence, and persist the result. Nodes are Python functions; edges specify what runs next. A state dictionary carries records, policies, findings and candidate actions between them. There are no cycles, a recursion limit of eight, and no automatic retries. LangSmith tracing is explicitly disabled for these private records.

## What is a tool?

A tool is a named operation with structured inputs and outputs. Our read-only tools inspect subscription/access and billing. Account IDs are validated and authorization uses the authenticated principal. Writing another account ID in the complaint does not grant access. The workflow calls AccountTools in-process, using one consistent multi-table snapshot for its actual investigation.

The same read operations are exposed to external agents through an official MCP stdio server. MCP is a standard protocol for discovering and calling tools; it is not a language model or a permission grant. `python -m nexus.mcp_server` communicates through standard input/output. It forwards reads to the HTTP API using an existing expiring session from its environment. The API still checks that session and account ownership on every call. No approval or execution tool is exposed through MCP.

Be precise in an interview: the LangGraph workflow calls the shared Python tools directly; the external MCP adapter is a separately verified access path. It does not perform a network MCP round trip internally. This avoids unnecessary local transport overhead while preserving one authorization model.

## What the investigation concludes

For each invoice, only succeeded attempts contribute funds. Recorded simulated refunds reduce net paid. Pending and failed attempts remain visible but do not count as charges. Duplicate-charge candidates must be distinct successful full invoice payments, with enough excess remaining to refund a complete charge while leaving the invoice funded. Ambiguous partial excess requires human review.

Access repair requires an active Pro subscription, missing Pro entitlements and a fully funded latest non-void invoice. An old paid invoice does not authorize access when the latest invoice is unpaid. Required v1 policy sections must be retrieved at the event date before proposals are generated. Unsupported or missing policies lead to escalation or a failed, escalated case; new policy versions require reviewing the executable rules.

The complaint is saved for context, but it cannot choose tools, change the authenticated identity, set amounts or execute actions. The financial findings and recommendations are deterministic backend rules. Gemini remains available for cited policy explanations; this graph is not an autonomous LLM planner. That choice gives reproducibility, no generation API dependency for investigation and a much smaller action surface.

## Persistence and failure

Every successful node records an inspectable step in SQLite. The final findings, proposals, case completion and ticket outcome are committed together. If a tool or workflow step fails, the case is marked failed and the ticket escalated. Raw exception details are not returned to the customer. Startup marks interrupted investigations failed; this recovery assumes one application worker. Persisted results survive restart, but this is not a LangGraph checkpoint/resume implementation: a failed investigation is inspected, then a new case can be run.

## Other designs and trade-offs

A plain Python function would also implement this linear flow and be simpler. LangGraph makes the stages/state explicit and leaves room for bounded conditional paths. An LLM choosing tools could handle more varied issues but adds model latency, tool-selection errors and injection risks. Multiple agents could divide work but introduce coordination and evaluation costs; neither extension is claimed here. A queue with durable jobs would support long operations and multiple workers but requires leases, retries and additional infrastructure.

Trace the implementation in `nexus/tools.py`, `nexus/workflow.py`, `nexus/mcp_server.py` and migration 005. Practise explaining why a pending payment is not a successful charge, why a complaint cannot authorize another account, and why a completed investigation is not an executed refund.

References: [LangGraph StateGraph reference](https://reference.langchain.com/python/langgraph/graph/state/) and the [official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk).
