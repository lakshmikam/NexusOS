# Part 3 — Verified retrieval results

The local encoder is `sentence-transformers/all-MiniLM-L6-v2`, revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`, producing 384-dimensional vectors. Model files were already cached. Policy text was embedded locally on CPU, with no paid API or answer-generation call.

Corpus: three fictional OrbitDesk policies. The active heading index has eleven chunks. The alternate fixed-window index has five chunks. Both use a maximum of 180 embedding tokens including title/heading; fixed windows overlap by twelve words. Comparison indexes do not replace the active snapshot.

Corpus SHA-256: `7005dd68e27548470063397c3d1cb49b84994f05da45db2ef1e157eee9ceb33b`.

## Twelve-question development set

Labels identify a policy ID and supporting quote. A result is relevant only when its passage contains that quote after case/whitespace/backtick normalization. Questions and labels are not ingested as policy documents.

| Chunking | Search | Hit@3 | MRR@3 | Mean returned embedding tokens |
|---|---|---:|---:|---:|
| Heading | BM25 | 1.0000 | 0.9583 | 165.50 |
| Heading | Dense | 0.9167 | 0.8750 | 171.17 |
| Heading | Hybrid RRF | 1.0000 | 0.9028 | 171.17 |
| Fixed | BM25 | 1.0000 | 0.9167 | 349.50 |
| Fixed | Dense | 0.9167 | 0.9167 | 369.67 |
| Fixed | Hybrid RRF | 1.0000 | 0.9583 | 386.75 |

These are **development measurements on twelve known questions and a tiny corpus**, not a held-out benchmark or proof of production reliability. The questions often share words with the policies. BM25 is a strong baseline here; adding dense search did not improve its heading-index MRR. Larger fixed chunks return more text and can score a hit simply by containing more facts. The comparison does not establish a universally best strategy.

Heading chunks remain the readable demo default, and all three retrieval methods are exposed for inspection. Hybrid combines independent rankings; it is not assumed to outperform BM25 on every query.

The complete per-question comparison is in `evaluation/results/part3-comparison.json`. Timings are machine-specific, and should not be compared as an isolated performance study. The comparison warms the encoder before building candidate indexes.

## Verified behaviour

All 42 automated tests passed. Unit retrieval tests use a deterministic fake encoder to isolate ranking, versioning and persistence logic; the real model was additionally exercised through local CLI comparisons and authenticated HTTP searches.

The live smoke test verified all three search methods, rejected an anonymous search, blocked a customer reindex attempt, and returned no applicable policy for a date before the rules became effective. The refund example ranked `Refund eligibility and approval` first and `Duplicate-payment investigation` second across all three methods. No LLM-generated answer was produced.

- Missing sessions cannot access policy search; customers cannot rebuild indexes or invoke the operator evaluation route.
- The latest effective policy version is selected before ranking. Expired versions do not revive older rules.
- Malformed rebuilds and invalid vectors cannot replace the active index.
- Heading chunks preserve section boundaries when possible, and all chunks are checked against tokenizer limits.
- Query and stored vectors must use the same model ID, revision and dimensions.
- BM25 with no matching terms returns an explicit empty result. Dense/hybrid results remain labelled candidates; there is no calibrated answerability threshold.
- Billing records and customer messages are not part of the policy index.

Primary references: [encoder model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [RRF mechanism](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking). Local NumPy implements our ranking; Azure is not an application dependency.
