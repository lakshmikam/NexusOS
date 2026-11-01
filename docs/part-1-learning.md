# Part 1 — Understand the records before adding AI

Historical Part 1 walkthrough. The current Part 2 API requires login: follow `docs/part-2-learning.md` first to obtain and authorize a session token. The billing model and calculations below remain the same; authenticated identity and ownership checks now run before record access.

## The product

OrbitDesk is a fictional subscription application. Pro includes a team workspace and advanced reports. NexusOS is the support system we are building around it. The LLM will eventually investigate facts; it must not invent them or decide its own permissions.

## Trace the main complaint

1. Open `/api/accounts/acct_duplicate`.
2. Observe an active Pro subscription but disabled feature entitlements.
3. Open `/api/accounts/acct_duplicate/invoices`.
4. Observe one invoice for 99900 paise and two distinct successful payments of 99900 each.
5. The response calculates 199800 paise confirmed paid and 99900 paise excess paid.
6. Create a support ticket with `/api/tickets`, then fetch it using its returned ID.
7. Restart the app. The ticket still exists because it was stored in SQLite, not a Python list.

This phase returns raw facts and simple accounting totals. It does not generate an AI diagnosis, authorize a refund, or correct access.

## Files to read in order

1. `nexus/migrations/001_initial.sql`: tables, keys, relationships and constraints.
2. `nexus/db.py`: connection lifecycle, migrations and fixed seed scenarios.
3. `nexus/api.py`: request schemas, endpoints and parameterized SQL queries.
4. `tests/test_part1.py`: expected behaviours and intentionally invalid cases.
5. `policies/`: the fictional rules that the later RAG system will retrieve.

## Follow one request

`GET /api/accounts/acct_duplicate/invoices`

```text
HTTP request
  → FastAPI matches the route and reads account_id
  → endpoint opens a SQLite connection
  → require_account checks that the requested account exists
  → SQL retrieves this account's invoices and payment attempts
  → Python sums only succeeded payments
  → endpoint returns dictionaries
  → FastAPI serializes them into a JSON HTTP response
  → connection closes
```

Checking existence is not authentication or authorization. Every seeded account is currently inspectable because Part 1 is explicitly a local demo. Part 2 adds identity and ownership checks before any LLM tools are introduced.

## Why these entities are separate

- **Account:** whose records are these?
- **Subscription:** what plan is currently recorded, and is it active?
- **Invoice:** what amount was owed?
- **Payment:** did a particular attempt succeed, fail or remain pending?
- **Entitlement:** which feature is actually enabled?
- **Ticket:** what issue was reported and what is its support status?

One account has many invoices and tickets. One invoice has many payment attempts. This first version keeps one current subscription row per account; subscription change history is not yet modelled.

## The important mechanisms

**Primary key:** identifies one row, such as `inv_duplicate`.

**Foreign key:** links a child record to an existing parent. The payment's `(invoice_id, account_id)` must match the same invoice/account pair.

**Constraint:** the database rejects invalid states such as a negative invoice amount or an unrecognized payment status. These are integrity checks, not the full business-rule engine.

**Transaction:** related changes commit together or roll back on failure. A test inserts an account and then attempts an invalid invoice; neither survives.

**Parameterized SQL:** values are passed separately from the SQL statement. A quote in an account ID is treated as data, not executable SQL.

**Pydantic validation:** requests need a valid account ID, subject and description. Extra fields such as a customer-supplied ticket status are rejected. Valid JSON alone does not prove a requested action is allowed.

**Integer money:** 99900 paise represents INR 999.00 without binary floating-point arithmetic. This phase has no FX or real payment gateway.

**Persistent state:** restarting Python does not erase the SQLite file. Repeated demo seeding does not delete tickets.

## Deliberately break things

- Request an account that does not exist: expect 404.
- Create a ticket with a one-character subject: expect 422.
- Include `status: resolved` in ticket creation: expect 422 because status is server-controlled.
- Inspect the pending-payment scenario: two attempts must still produce zero confirmed paid.
- Try inserting a payment pointing to another account's invoice in a test: expect a foreign-key error.

## Explain these back

1. Why are two payment attempts not necessarily two successful charges?
2. How can the subscription be active while Pro features are disabled?
3. What preserves a ticket after the server restarts?
4. What is the difference between checking an account exists and checking the caller owns it?
5. Why is 99900 stored instead of 999.0?
6. What rolls back when one statement in a transaction fails?

## Interview explanation after Part 1

“I started NexusOS by modelling the source of truth. I separated invoices, payment attempts, subscriptions and entitlements, seeded reproducible support cases, and exposed them through a FastAPI API. SQLite constraints protect record relationships, and tests verify confirmed-payment calculations, persistence, validation and rollback. The AI workflow will use these records through authorized tools in later phases.”
