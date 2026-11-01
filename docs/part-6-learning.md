# Part 6 — approval, execution and repeat safety

## A recommendation must not be a database write

An action proposal is a record saying what could be done and why. It has a type (refund or restore_access), a server-selected target, amount, policy evidence and status. Initially it is pending. Investigation itself does not move funds or enable features.

An authenticated operator can approve or reject a pending proposal with a review note. Approval changes its status but does not execute it. Execution is a separate explicit operator request. One operator may review and execute; a two-person separation-of-duties policy is not implemented. Customers can see their account's proposals but cannot approve or execute them.

## Why recheck after approval?

An investigation is a snapshot in time. Between review and execution, a payment might fail, a subscription might be cancelled, or another case might already have refunded the charge. Approval is therefore not enough. Execution reads current records, calculates eligible actions again, and checks the target and amount against the approved proposal. It also checks that the current applicable policy versions and content hashes match the proposal's evidence. Changed or unsupported evidence marks the proposal stale and requires a new investigation.

SQLite `BEGIN IMMEDIATE` serializes writers before these checks. The eligibility calculation and resulting writes occur inside the same transaction. A transaction groups changes so either all commit or all roll back. This prevents another local writer from slipping a competing refund between the check and insert.

## Idempotency

Imagine an operator clicks Execute, the server commits, and the network response is lost. Retrying must not create another refund. The client sends an idempotency key: a unique label for that intended request. The backend stores the actor, key, request hash and result in the same transaction as the action. Repeating the same actor/key/request returns the stored result with replayed=true. Reusing that key for another proposal returns 409.

Different keys alone do not make two requests safe. A unique constraint permits at most one refund row per payment, and eligibility is recalculated after earlier refunds. Two investigations proposing the same refund cannot both move the simulated amount. Concurrent execution is tested. This is a local transaction guarantee, not distributed payment-provider exactly-once delivery.

## What changes when execution succeeds?

A simulated refund adds a refund record; it does not delete or rewrite the original payment. Gross successful paid remains historical truth, while net paid subtracts refunds. Access correction upserts the two supported Pro entitlements as enabled, including creating a missing entitlement row. The action status, result and audit event commit with the change.

The ticket remains pending while proposals await review/execution. Rejected or stale proposals escalate. After all proposals execute, the backend checks for remaining discrepancies; unresolved partial excess is escalated rather than marked resolved.

## Audit history

An audit event records the authenticated actor, account, case, event type, details and timestamp. Database triggers reject updates and deletes through ordinary SQL. This supports traceability but is not cryptographic tamper evidence: a database administrator could alter schema/triggers. External immutable logs would be a stronger production option.

## Alternatives

Automatic execution is faster but inappropriate for a learning support system handling financial changes. An external payment provider would require its own idempotency, asynchronous state, webhooks, reconciliation and recovery for uncertain outcomes. PostgreSQL offers row-level locking and more write concurrency; SQLite is sufficient for this single-worker local demonstration. A unique key without a transaction would leave gaps between changes and saved results. See `nexus/actions.py` and the race/authorization tests in `tests/test_resolution.py`.
