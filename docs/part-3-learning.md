# Part 3 — Policy retrieval, starting from zero

These notes accompany the chat walkthrough. No prior RAG knowledge is assumed. We will explain the mechanisms in the chat; this file is a reference, not homework required before the explanation.

## 1. What problem are we solving?

The database says what happened to a customer. Policy documents say what the company permits. A customer asking about a duplicate charge needs both. Part 3 makes the shared policy documents searchable. It does not yet generate answers or authorize actions.

For three tiny documents we could read every document on every request. We are building retrieval now to learn the mechanism and make the later application work as the policy collection grows.

RAG means **Retrieval-Augmented Generation**: find relevant source information, supply it as context to an LLM, then have the LLM generate a response using that context. This phase implements retrieval. Part 4 adds generation. The local embedding model already uses learned AI features, but it produces vectors rather than customer-facing answers. Retrieving a passage does not guarantee that a later answer will be correct.

## 2. A document becomes passages

A chunk is a passage chosen as a unit of search. A whole document can mix price, payment status, refunds and approval conditions. Returning only its title does not identify the relevant rule; returning the full document can include unrelated material.

Our default **heading strategy** keeps each Markdown `##` section together when it fits the budget. A long section is split at word boundaries using actual tokenizer counts. The limit includes the document title and section heading, which are included in the embedding input. No overlap is added for heading chunks. The current three policy documents produce eleven chunks at the default 180-token limit.

Tokens are pieces used by a particular model's tokenizer, not necessarily words. `team_workspace`, punctuation or an unfamiliar word may use several tokens. This encoder accepts up to 256 wordpieces by default; we keep chunks within 180 tokens including context and reject overlong query inputs instead of silently truncating them. These are embedding-token counts, not automatically the token counts of a future answer-generating LLM.

Our comparison **fixed strategy** packs text from the whole document into token-bounded windows at word boundaries, with 12 words of overlap. It deliberately ignores the section boundaries. Overlap repeats context around cuts, increasing storage and potential duplicate results. Its overlap unit is words; its maximum size is measured in embedding tokens.

Other choices:

| Choice | Strength | Trade-off |
|---|---|---|
| Whole documents | Easiest baseline, no boundary cuts | More irrelevant text and larger contexts as documents grow |
| Paragraph chunks | Simple and easy to inspect | Paragraph lengths vary; a rule and its exception may be separate |
| Fixed windows | Predictable upper bound, works without headings | Can split conditions; overlap adds redundancy |
| Heading-aware (our default) | Uses structure already written by the policy author | Depends on useful headings; long sections still need splitting |
| Semantic chunking | Can use embedding changes to propose boundaries | More inference, thresholds and evaluation; not automatically better |

## 3. Metadata is evidence about the passage

Each document has a policy ID, family, version, effective date, optional expiry date, source filename and source hash. Each chunk has its own stable content-derived ID, section heading, body and token count. The active index also records the model ID, model revision, strategy and corpus hash.

For a question about an event on a particular date, the search selects the latest effective version in each policy family **before ranking**. If that latest version has expired, the older version is not silently revived. Different families, such as billing and access, can each contribute their applicable version.

New indexes are built as snapshots. The active pointer changes only after the complete index transaction succeeds. A failed rebuild leaves the previous snapshot available. Source edits require explicit reindexing; an existing snapshot is not silently rewritten.

The source hash identifies the ingested file content. The chunk text in SQLite is the exact passage snapshot used for retrieval, so later citations can reference what was actually supplied.

## 4. BM25: rank using words

BM25 is a lexical ranking formula. We lowercase and split text into terms, remove common functional words, and count remaining terms. Negations such as `not` and `no` are retained. There is no stemming or language-specific parser in this version.

The score considers:

- Which query terms occur in the passage.
- How rare each term is in the eligible passages.
- How often it occurs in this passage, with diminishing returns.
- Passage length, so long passages do not win simply by containing more words.

Example: `duplicate payment` should reward the matching billing section. But `money was taken twice` may use different wording. BM25 does not directly learn that these expressions can describe the same situation.

We implement the formula locally with k1=1.5 and b=0.75 so it can be traced. For eleven chunks, computing statistics over eligible passages is small enough. This is exact in-process scoring, not a production inverted-index search service. A larger corpus could use SQLite FTS5, OpenSearch or another search engine instead.

Alternatives: substring search is simpler but brittle; TF-IDF/cosine is a useful lexical baseline; an indexed search engine adds faster lookup and richer features but requires more setup. None removes the need to evaluate relevance.

## 5. Embeddings: represent text with learned numeric features

An embedding is a vector, a list of numbers produced by a trained model. We use a pretrained sentence encoder, `sentence-transformers/all-MiniLM-L6-v2`, pinned to a specific revision. It maps a sentence or short passage to 384 numbers.

We are not assigning human labels to individual dimensions. The model learned patterns during its earlier training. Related texts can end up pointing in similar directions in this numeric space, although mismatches still occur.

At ingestion we embed each passage's title, heading and body. At search time we embed the question using the **same model**. We compare its vector with the stored vectors using cosine similarity. All vectors are normalized to length one, so the dot product computes cosine similarity.

This helps with paraphrases, but similarity is not proof that a passage answers the question. An unrelated query can still have a highest-scoring result. Cosine similarity is not a calibrated probability of correctness.

Why this model? It is already cached locally, works on CPU, needs no API key, and is suitable for an English short-passage baseline. It is not a Hindi/Hinglish quality guarantee. We would need representative multilingual questions and a model comparison before claiming that capability.

An embedding API could reduce local model management but introduces a network dependency, provider costs and transmission of text. A larger or multilingual local model could help some queries but changes memory/latency and requires rebuilding embeddings. We would choose through measurement, not model size alone.

Changing policies requires rebuilding the passage index, not retraining the encoder. Changing the encoder or its revision requires rebuilding the stored vectors; mixing models is rejected.

## 6. Why SQLite and exact vector search?

SQLite persists passage metadata, text and normalized vectors as JSON. NumPy computes exact similarities over the date/family-filtered candidates. There is no FAISS, pgvector or remote vector service in this implementation.

| Choice | Strength | Trade-off |
|---|---|---|
| SQLite + exact scan (our choice) | Few moving parts; easy to inspect; exact ranking for this small corpus | Linear scan and JSON decoding do not scale indefinitely |
| FAISS-style vector index | Efficient vector search; approximate indexes can reduce work | Additional index lifecycle; metadata/access filtering still needs design |
| PostgreSQL + pgvector | Transactional metadata and vector search in one database | Database service/setup; tuning depends on workload |
| Managed vector/search service | Hosted operational features and scaling options | Network, billing and service-specific behaviour |

Dense vectors do not contain customer billing records here. Only the controlled `policies/` directory is ingested. All users can search these shared policies; account data remains behind the Part 2 ownership checks.

## 7. Hybrid search: merge rankings

BM25 can be strong for exact terms and identifiers. Dense search can help when wording differs. Hybrid search combines both ranked lists with **reciprocal rank fusion**, or RRF.

For each passage, our score adds:

```text
1 / (60 + BM25 rank), if it appears in that list
1 / (60 + dense rank), if it appears in that list
```

Ranks start at one. We fuse up to twenty candidates per method and return up to the requested top_k. Sixty is an initial smoothing constant, not a tuned optimum for OrbitDesk.

Why not add the raw scores? A BM25 score and a cosine similarity have different ranges and meanings. Rank fusion avoids assuming they share a scale. A weighted score blend is another option, but its normalization and weights need evaluation. RRF discards score-gap information: being barely first and being far ahead both count as rank one. A reranker could later examine the query and candidate passages more closely, at extra inference cost; it is not implemented here.

## 8. What the API returns

Log in as Asha and authorize the token. Then call `POST /api/retrieval/search`:

```json
{
  "query": "I was charged twice. What evidence is needed for a refund?",
  "method": "hybrid",
  "top_k": 3,
  "as_of": "2026-10-31"
}
```

Inspect policy ID, heading, passage text, source filename/hash, effective date, BM25 rank/score, dense rank/cosine, and fused score. Try `bm25` and `dense` on the same question. Scores should not be compared numerically across methods.

The status `candidates_only` deliberately does not mean sufficient evidence or an answer. `no_applicable_policy` means date/family filtering left no valid passage; `no_lexical_match` means BM25 found no matching terms. Dense/hybrid retrieval has no calibrated relevance threshold and may return candidates for unsupported questions. Part 4 must handle evidence support and uncertainty explicitly.

`GET /api/retrieval/chunks` exposes the stored passage snapshots. `GET /api/retrieval/status` reports index readiness/configuration, not a promise that the embedding model is already warm. The first dense query loads the local model and can take longer.

Reindex and evaluation endpoints require an operator. Indexing uses offline cached files by default; the explicit CLI `download-model` command is the only configured model-download path. No answer-generation API is called in Part 3.

## 9. How the comparison is evaluated

`evaluation/retrieval_cases.json` contains twelve development questions with a source policy ID and a short supporting quote. Those labels are never ingested as policy documents.

- Hit@3: fraction of questions with a labelled supporting quote in at least one top-three passage.
- MRR@3: average reciprocal rank of the first relevant result, counting missing results as zero.
- Mean retrieved embedding tokens: a rough view of context volume returned, including each chunk's embedded title/heading.

Example: relevant passage at rank 1 contributes 1 to reciprocal rank; rank 2 contributes 1/2; absent from the top three contributes 0.

This is a small, known development set, not a held-out quality guarantee. It does not measure answer correctness or sufficient evidence. Larger passages may get more hits by containing more facts, so compare context volume and inspect outputs as well. All methods use the same corpus/model/date/labels. Heading and fixed strategies have the same maximum token budget but different boundaries, overlap and actual passage lengths.

```powershell
python -m nexus.retrieval compare --out data/retrieval-comparison.json
```

Comparison builds unpublished snapshots and does not replace the live heading index. The report records each case, source passages and measured scores. No model training is required.

## 10. Code walkthrough order

1. `policies/billing-v1.md` — one ordinary document and its metadata.
2. `parse_policy` and `make_chunks` in `nexus/retrieval.py` — document to passages.
3. `nexus/embeddings.py` — local model and tokenizer.
4. `RetrievalService.build` — snapshot creation and persistence.
5. `bm25_scores`, `eligible_documents`, and `RetrievalService.search` — filtering, ranking and fusion.
6. Retrieval routes in `nexus/api.py` — validated requests and authorization.
7. `evaluate` and the labelled development questions — checking actual retrieval behaviour.

## Primary references used for implementation choices

- Model card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 — encoder use, 384 dimensions, short-passage use and 256-wordpiece default truncation.
- RRF explanation: https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking — combining reciprocal ranks and the distinction between retrieval scores and reranking.

We use local NumPy ranking, not Azure Search. These references explain the model and fusion mechanism, not an external service dependency.
