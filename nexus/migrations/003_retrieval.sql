CREATE TABLE retrieval_indexes (
    id TEXT PRIMARY KEY,
    created_at INTEGER NOT NULL,
    strategy TEXT NOT NULL CHECK (strategy IN ('heading', 'fixed')),
    max_tokens INTEGER NOT NULL,
    overlap_words INTEGER NOT NULL,
    model_id TEXT NOT NULL,
    model_revision TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    corpus_hash TEXT NOT NULL
);

CREATE TABLE policy_documents (
    index_id TEXT NOT NULL REFERENCES retrieval_indexes(id),
    policy_id TEXT NOT NULL,
    family TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version > 0),
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    effective_from TEXT NOT NULL,
    effective_until TEXT,
    PRIMARY KEY (index_id, policy_id),
    UNIQUE (index_id, family, version)
);

CREATE TABLE policy_chunks (
    index_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    policy_id TEXT NOT NULL,
    heading TEXT NOT NULL,
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    embedding_json TEXT NOT NULL,
    PRIMARY KEY (index_id, chunk_id),
    FOREIGN KEY (index_id, policy_id) REFERENCES policy_documents(index_id, policy_id)
);

CREATE TABLE retrieval_state (
    id INTEGER PRIMARY KEY CHECK (id=1),
    active_index_id TEXT NOT NULL REFERENCES retrieval_indexes(id)
);
CREATE INDEX policy_chunks_document_idx ON policy_chunks(index_id, policy_id);
