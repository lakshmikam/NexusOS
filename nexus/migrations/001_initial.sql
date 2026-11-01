-- Money is stored in integer paise: 99900 = INR 999.00.
CREATE TABLE accounts (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE subscriptions (
    id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL UNIQUE REFERENCES accounts(id),
    plan TEXT NOT NULL CHECK (plan IN ('free', 'pro')),
    status TEXT NOT NULL CHECK (status IN ('active', 'pending', 'cancelled')),
    monthly_amount_minor INTEGER NOT NULL CHECK (typeof(monthly_amount_minor) = 'integer' AND monthly_amount_minor >= 0),
    currency TEXT NOT NULL DEFAULT 'INR' CHECK (currency = 'INR')
);

CREATE TABLE invoices (
    id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL REFERENCES accounts(id),
    reference TEXT NOT NULL UNIQUE,
    amount_minor INTEGER NOT NULL CHECK (typeof(amount_minor) = 'integer' AND amount_minor > 0),
    currency TEXT NOT NULL DEFAULT 'INR' CHECK (currency = 'INR'),
    status TEXT NOT NULL CHECK (status IN ('open', 'paid', 'void')),
    issued_at TEXT NOT NULL,
    UNIQUE (id, account_id)
);

-- An invoice says what is owed; each payment says what happened to an attempt.
CREATE TABLE payments (
    id TEXT PRIMARY KEY,
    invoice_id TEXT NOT NULL,
    account_id TEXT NOT NULL,
    provider_reference TEXT NOT NULL UNIQUE,
    amount_minor INTEGER NOT NULL CHECK (typeof(amount_minor) = 'integer' AND amount_minor > 0),
    currency TEXT NOT NULL DEFAULT 'INR' CHECK (currency = 'INR'),
    status TEXT NOT NULL CHECK (status IN ('pending', 'succeeded', 'failed')),
    received_at TEXT NOT NULL,
    FOREIGN KEY (invoice_id, account_id) REFERENCES invoices(id, account_id)
);

CREATE TABLE entitlements (
    account_id TEXT NOT NULL REFERENCES accounts(id),
    feature TEXT NOT NULL CHECK (feature IN ('team_workspace', 'advanced_reports')),
    enabled INTEGER NOT NULL CHECK (enabled IN (0, 1)),
    PRIMARY KEY (account_id, feature)
);

CREATE TABLE tickets (
    id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL REFERENCES accounts(id),
    subject TEXT NOT NULL CHECK (length(subject) BETWEEN 3 AND 120),
    description TEXT NOT NULL CHECK (length(description) BETWEEN 10 AND 4000),
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'investigating', 'pending_approval', 'resolved', 'escalated')),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX invoices_account_idx ON invoices(account_id);
CREATE INDEX payments_invoice_idx ON payments(invoice_id, account_id);
CREATE INDEX tickets_account_idx ON tickets(account_id, created_at);
