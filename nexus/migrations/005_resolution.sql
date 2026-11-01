CREATE TABLE investigations (
 id TEXT PRIMARY KEY, ticket_id TEXT NOT NULL UNIQUE REFERENCES tickets(id),
 account_id TEXT NOT NULL REFERENCES accounts(id), created_by TEXT NOT NULL REFERENCES users(id),
 complaint TEXT NOT NULL, event_date TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('running','completed','failed')),
 state_json TEXT, error_code TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 completed_at TEXT
);
CREATE TABLE investigation_steps (
 id INTEGER PRIMARY KEY AUTOINCREMENT, investigation_id TEXT NOT NULL REFERENCES investigations(id),
 stage TEXT NOT NULL, data_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE action_proposals (
 id TEXT PRIMARY KEY, investigation_id TEXT NOT NULL REFERENCES investigations(id),
 account_id TEXT NOT NULL REFERENCES accounts(id),
 kind TEXT NOT NULL CHECK(kind IN ('refund','restore_access')),
 target_id TEXT NOT NULL, amount_minor INTEGER NOT NULL DEFAULT 0 CHECK(typeof(amount_minor)='integer' AND amount_minor>=0),
 status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','approved','rejected','executed','stale')),
 reason TEXT NOT NULL, policy_json TEXT NOT NULL, approved_by TEXT REFERENCES users(id), decision_note TEXT,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, decided_at TEXT, executed_at TEXT,
 UNIQUE(investigation_id,kind,target_id)
);
CREATE TABLE refunds (
 id TEXT PRIMARY KEY, proposal_id TEXT NOT NULL UNIQUE REFERENCES action_proposals(id),
 payment_id TEXT NOT NULL UNIQUE REFERENCES payments(id), account_id TEXT NOT NULL REFERENCES accounts(id),
 amount_minor INTEGER NOT NULL CHECK(typeof(amount_minor)='integer' AND amount_minor>0),
 mode TEXT NOT NULL DEFAULT 'simulated' CHECK(mode='simulated'),
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE action_requests (
 actor_id TEXT NOT NULL REFERENCES users(id), idempotency_key TEXT NOT NULL,
 request_hash TEXT NOT NULL, result_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 PRIMARY KEY(actor_id,idempotency_key)
);
CREATE TABLE audit_events (
 id INTEGER PRIMARY KEY AUTOINCREMENT, actor_id TEXT NOT NULL REFERENCES users(id),
 account_id TEXT NOT NULL REFERENCES accounts(id), investigation_id TEXT REFERENCES investigations(id),
 event TEXT NOT NULL, data_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TRIGGER audit_no_update BEFORE UPDATE ON audit_events BEGIN SELECT RAISE(ABORT,'Audit events are append-only'); END;
CREATE TRIGGER audit_no_delete BEFORE DELETE ON audit_events BEGIN SELECT RAISE(ABORT,'Audit events are append-only'); END;
CREATE INDEX investigations_account ON investigations(account_id,created_at);
CREATE INDEX proposals_account ON action_proposals(account_id,status);
CREATE INDEX audit_account ON audit_events(account_id,id);
