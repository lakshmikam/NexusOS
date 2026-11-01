CREATE TABLE users (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('customer', 'operator')),
    account_id TEXT REFERENCES accounts(id),
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    CHECK ((role = 'customer' AND account_id IS NOT NULL)
        OR (role = 'operator' AND account_id IS NULL))
);

-- Store only a digest of the random bearer token, never the raw credential.
CREATE TABLE sessions (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    created_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL CHECK (expires_at > created_at),
    revoked_at INTEGER
);
CREATE INDEX sessions_user_idx ON sessions(user_id);

CREATE TABLE login_throttles (
    scope TEXT PRIMARY KEY,
    window_started INTEGER NOT NULL,
    attempts INTEGER NOT NULL CHECK (attempts > 0)
);
