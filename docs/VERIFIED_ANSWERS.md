# Part 4 verification

## Automated checks

On 2026-11-01, `python -m pytest -q` completed with **63 passing tests** across Parts 1–4. Compilation and `git diff --check` also passed. The generation tests inject deterministic fixtures or an HTTP mock; they do not require provider credits and do not measure Gemini's reasoning quality.

Coverage includes structured output, unknown source IDs, fabricated quotes, some obvious action claims, provider failure sanitization, incomplete output, validation-before-text stream ordering, bounded recent history, fresh retrieval for follow-ups, persistence, interrupted-turn recovery, overlapping-turn rejection and conversation ownership. A deliberately incorrect claim with a real quote is accepted by the integrity checker in one test, demonstrating that exact-quote validation does not establish semantic support.

## Live development checks

The configured provider is Gemini, model `gemini-3.8-flash`. An initial direct generation produced a structured policy answer with checked citations about duplicate-payment evidence and approval requirements. The first HTTP smoke attempt timed out. One bounded retry completed its first streamed answer: it preserved the requirements for distinct successful payment records, prior-refund checks, backend eligibility checks and operator approval, and did not claim an account check or completed refund. That retry's follow-up generation timed out, so the third unsupported-question generation was not run. The three-case live exercise did **not** complete successfully.

A subsequent preview verified that the completed turn entered follow-up context, the failed turn was excluded, and the recent question expanded the follow-up retrieval query. Another customer's conversation read returned 404. [Partial live report](../evaluation/results/part4-partial-live.json) records this evidence without credentials. Unsupported-answer handling and multi-turn behaviour pass deterministic automated tests, but real-provider follow-up and abstention quality remain unverified in this exercise. Provider timeout errors published no accepted partial answer.

After Parts 5–8 were integrated, a final bounded retry with `gemini-3.8-flash` timed out on its first question. No successful three-case live report was produced. Current provider availability is therefore a demo limitation; the application surfaces the error, and the deterministic account-investigation/action workflow remains independent of generation calls.

`python scripts/demo_rag.py --live --out data/part4-live-smoke.json` exercises three synthetic cases: evidence/approval for a duplicate-charge refund, a remembered follow-up about whether a refund has actually happened, and an invoice XML-export question absent from the policies. It also checks two persisted turns and another customer's 404 access response. It makes three provider requests when successful and must be explicitly invoked. A failed run does not create a completed report; its absence must not be interpreted as a pass.

Review generated claims against the full policy, not only copied quotes. Record model, context, usage and latency with the observed answers. These few development questions are neither a held-out evaluation nor a production accuracy estimate. No automatic semantic correctness, prompt-injection immunity or executed billing action is claimed.

## Limits relevant to a demo

- The generator receives policies and selected chat history, not trusted billing rows or action tools.
- Source IDs are local to each answer; current sources are retrieved again.
- Prompt/history budgets use characters; provider usage uses its own tokens.
- Streaming buffers generated answer text until the implemented checks pass.
- Quota, network and timeout errors are surfaced without automatic retries or fabricated fallback answers.
- Persistent memory stores text in SQLite and does not train a model.
- Run one API worker; startup interrupted-turn recovery is not a distributed job system.

Provider interface references: [Gemini generation API](https://ai.google.dev/api/generate-content), [structured output](https://ai.google.dev/gemini-api/docs/structured-output), and [text generation](https://ai.google.dev/gemini-api/docs/text-generation).
