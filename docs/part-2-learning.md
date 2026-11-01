# Part 2 — Who is asking, and what may they access?

## Start in the interactive documentation

1. Refresh http://127.0.0.1:8766/docs.
2. Expand `POST /api/auth/login`, click **Try it out**, and send:

```json
{"email":"asha@example.test","password":"OrbitDesk-Demo-2026!"}
```

3. Copy only the `access_token` string from the response.
4. Click **Authorize** near the top of the documentation. Paste the token into **SessionToken**. Do not add `Bearer ` yourself; this UI adds the scheme.
5. Request `/api/auth/me` to see the verified identity.
6. Request `/api/accounts/acct_duplicate/invoices`: 200, because this is Asha's account.
7. Request `/api/accounts/acct_healthy/invoices`: 404, because Asha cannot inspect Meera's account.
8. Request `/api/demo/scenarios`: 403, because the all-account teaching overview is operator-only.

For the support operator, log in with `support@example.test` and password `OrbitDesk-Operator-2026!`, then replace the token in **Authorize**. The operator can inspect all three accounts and the scenario overview.

These are intentionally public synthetic demo credentials, not real user credentials. The server remains bound to localhost. There is no signup endpoint, and roles are not accepted from the login request.

## The two new domain tables

- `users`: login identity, salted password hash, role, account relationship and active flag.
- `sessions`: hash of a random session token, linked user, creation/expiry timestamps and optional revocation timestamp.

One account may have several customer users. The seeded example has one customer user per account. An operator has no customer account and receives explicit cross-account read/investigation permissions. Operators can create tickets on behalf of accounts; refunds and other actions do not exist yet.

`login_throttles` is an additional technical table for the basic login-attempt limiter. It is not customer billing data.

## Trace login

```text
JSON email + password
  → Pydantic rejects extra fields and normalizes email
  → basic attempt limit for this email + client address
  → find user by normalized email
  → verify the password using scrypt and constant-time digest comparison
  → generate a cryptographically random token
  → store SHA-256(token), user ID and expiry in SQLite
  → return the raw token once with the public user profile
```

Passwords are not trimmed, lowercased or stored as plaintext. Email addresses are normalized. Passwords use a random salt and scrypt, a deliberately expensive password derivation function. Two identical passwords receive different stored hashes because their salts differ.

A session token is already random and has high entropy; its SHA-256 digest supports lookup without storing the bearer credential itself. Passwords need slow salted hashing because human-chosen passwords can be guessed. These are two different uses of hashing.

The session defaults to one hour. `expires_at` is a Unix UTC timestamp in seconds. Login responses have `Cache-Control: no-store`. Password hashes and session digests are never returned in user profiles.

## Trace a protected request

```text
Authorization: Bearer <token>
  → FastAPI dependency extracts the credential
  → hash token and look up a live, unrevoked session
  → join to an active user row
  → obtain trusted role and account_id from the database
  → check permission for the requested account
  → retrieve only permitted records
  → return JSON
```

`Depends(authenticated_user)` means FastAPI resolves the verified caller before invoking the endpoint. Missing, invalid, expired or revoked sessions return 401. Deactivating a user also blocks previously issued sessions.

`require_account_access` checks the verified principal, not an account ID claimed by a message or model. Customer account listings are filtered to their own account. Ticket reads include the ownership filter in SQL, so guessing a ticket ID does not expose its contents or confirm its existence.

## Authentication versus authorization

- Authentication: this token corresponds to Asha's active login session.
- Authorization: Asha may access `acct_duplicate` but not `acct_healthy`.

A valid login does not give unrestricted account access. A user-supplied `role: operator` is rejected as an unexpected login field. The trusted role comes from SQLite.

## Why opaque sessions instead of JWTs here?

The token has no readable embedded account claims. The server looks it up in the database on each request. That makes logout/revocation and user deactivation immediate, and persists sessions across restarts. The trade-off is a database lookup per authenticated request. We do not need distributed token verification for this single-process learning application.

## What logout does

`POST /api/auth/logout` sets the current session's `revoked_at` timestamp. The same token then receives 401. It revokes only that session; other login sessions are independent. Repeating logout with the revoked token also receives 401.

## Response codes to expect

| Status | Example |
|---|---|
| 200 | Valid login or allowed read |
| 201 | Allowed ticket creation |
| 401 | Missing/invalid/expired/revoked session or invalid credentials |
| 403 | A customer requests the operator-only demo overview |
| 404 | Account/ticket not found or not visible to this customer |
| 422 | Invalid request shape, including a user-supplied role |
| 429 | Five failed attempts in the same one-minute email/client window; respect Retry-After |

404 is deliberate for other customers' account/ticket IDs: the response does not reveal whether the resource exists. The operator-only route returns 403 because it is a known capability the customer role cannot use.

## The migration preserves Part 1

Migration `002_auth.sql` adds tables and indexes. It does not drop billing or ticket records. Demo-user seeding skips existing users and never silently resets a changed password hash. `demo=False` creates no seeded users or accounts.

## Tests and limitations

Tests cover password salting, session storage, successful login, missing/invalid/expired credentials, user deactivation, logout, customer isolation, ticket-ID guessing, operator permissions, role injection, restart persistence, throttling and schema relationships. Part 1's billing tests now authenticate as an operator.

This remains a local synthetic-data application. No public signup, password reset, MFA, HTTPS termination, production abuse controls or operator activity audit log is implemented. The fixed-window login limiter is a small local safeguard, not a distributed rate-limiting service. Operator action approval and auditing arrive in Part 6.

## Files to read in order

1. `nexus/migrations/002_auth.sql` — new tables and constraints.
2. `nexus/auth.py` — hashes, login, sessions, dependencies and permission checks.
3. `nexus/api.py` — authentication endpoints and protected account/ticket routes.
4. `tests/test_auth.py` — expected access boundaries and failure examples.

After the walkthrough, explain why knowing an account ID is insufficient, why login returns a random token instead of a password hash, and why a logged-out token no longer works.
