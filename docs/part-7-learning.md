# Part 7 — using the product and evaluating the whole flow

## Browser and server responsibilities

The browser renders HTML/CSS, collects form input and makes HTTP requests using JavaScript fetch. FastAPI handles validation, authenticated identity, business rules and database access. The browser cannot grant itself operator permissions; hiding a button is only a presentation choice. The server independently rejects unauthorized requests.

We chose plain JavaScript and CSS rather than adding React and a separate build chain. For this bounded interface that keeps setup small and exposes HTTP/DOM fundamentals. A larger product could benefit from React's component/state ecosystem, routing and a dedicated design system, at the cost of dependencies and build complexity.

The session token stays in JavaScript memory. Reloading requires signing in again. It is never placed in URLs or persisted in localStorage. Text from complaints, records and model responses is inserted using textContent, not interpreted as HTML. A Content Security Policy permits local scripts/styles only. Those controls reduce certain risks; they do not establish production hardening.

## The screens

Overview shows verified account state, net paid, excess funds, feature access and invoices/payment attempts. Start an investigation there. Investigations shows saved case steps, findings, initial evidence and proposal states. A case's findings remain its historical snapshot; the overview reflects current account state.

Policy assistant creates private conversations, streams provider progress and displays accepted cited answers. Saved conversations can be reopened. Inspect prompt shows the exact assembled question, selected sources and bounded recent history without a generation call. This remains policy-only; it does not silently acquire account facts from an investigation.

Approvals lists proposals and exposes review/execution controls for operators. Activity log shows account-scoped audit events. A customer cannot perform an operator action by constructing the HTTP request manually. Request epochs and stream cancellation discard stale responses when the session/account/conversation changes.

## Streaming in the interface

The browser reads newline-delimited JSON events from the response stream, retaining incomplete lines across network chunks. Progress indicates that generation is underway. Answer text is displayed only from the final accepted result, after backend checks. A lost connection displays an error instead of inventing an answer. Reopen the saved conversation before retrying an uncertain generation result. Provider availability and quota remain external dependencies.

## Evaluation is more than a happy-path demo

The isolated resolution evaluator creates temporary accounts, copies only the active real policy index and runs four development cases. Healthy: no fix. Pending/failed attempts: no confirmed charge, escalate. Duplicate/missing access: two proposals, customer denial, operator approval, execution, replay and resolved state. Injection complaint: account, amount and approval rules remain controlled by the server.

Automated tests also cover concurrent refunds, stale records/policies, rejected proposals, missing historical policies, latest-invoice access eligibility, partial excess and persistence. These tests target specific failure modes; they are not evidence that all adversarial inputs are safe.

Retrieval hit rates, answer semantic accuracy and final resolution correctness are separate measures. Our scripted synthetic cases are development fixtures, not an unseen benchmark. Real Gemini follow-up and unsupported-question quality were not verified in the earlier live exercise because of provider timeouts. See `docs/VERIFIED_APPLICATION.md` for exact observed results and limits.

## Suggested demo

Sign in as Asha, inspect attempts, run a case, review policy evidence, and see pending proposals. Sign out and enter as Support operator. Review the same case, approve with a note, and explicitly execute each simulated fix. Refresh Overview and inspect Activity log. Then examine Kabir's pending account and explain why two attempts do not justify a refund. Existing demo state is persistent; use a separate database file for a fresh replay rather than deleting history.
