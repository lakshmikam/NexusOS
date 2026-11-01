CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    owner_user_id TEXT NOT NULL REFERENCES users(id),
    account_id TEXT REFERENCES accounts(id),
    title TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE conversation_turns (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    question TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running','completed','failed')),
    answer_json TEXT,
    error_code TEXT,
    created_at INTEGER NOT NULL,
    completed_at INTEGER
);
CREATE UNIQUE INDEX one_running_turn_per_conversation ON conversation_turns(conversation_id) WHERE status='running';
CREATE INDEX conversation_owner_idx ON conversations(owner_user_id,updated_at);
CREATE INDEX conversation_turn_order_idx ON conversation_turns(conversation_id,created_at,id);
