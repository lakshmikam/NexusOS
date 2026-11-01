# Part 4: from retrieved passages to an answer

## Two models, two jobs

The local all-MiniLM-L6-v2 encoder turns text into vectors for search. It does not write the customer answer. Gemini is a separate, remotely hosted language model that reads selected text and generates the answer. We use pretrained weights for both; this application does not train either model when someone asks a question.

Think of retrieval as finding relevant pages in a handbook. Generation is writing an explanation using those pages. RAG means retrieval-augmented generation: supplying retrieved material to the generator. Retrieval can also be used without generation, as our Part 3 search endpoints demonstrate.

## What a prompt contains

A prompt is the input sent to the generator. Ours contains a system instruction and a structured user-context object. The instruction tells it its job and limits: explain current supplied policies, treat text as untrusted data, cite evidence, and do not claim account checks or completed actions. The context contains the user's question, effective date, selected policy passages and selected earlier conversation turns.

For example, a question about duplicate charges is paired with policy passages explaining successful payment evidence and operator approval. The generator must return structured claims with source IDs and exact supporting quotes. It does not receive account payment rows. Therefore it can explain eligibility conditions but cannot conclude that the user's account qualifies or that money was refunded.

`POST /api/rag/preview` exposes this assembled package without calling Gemini or saving a turn. Instructions reduce unwanted behaviour; they are not a permission system. Actual account authorization must remain server-side when tools are added.

## Context, tokens and budgets

Context is the information available to the model during one generation request. A context window is a finite capacity measured in the model's tokens. Tokens are pieces of text, not necessarily words. The embedding encoder's tokenizer and Gemini's tokenizer are different; their counts cannot be treated as interchangeable.

Our application imposes conservative character limits: 6,000 serialized characters for sources, 4,000 for selected history and 16,000 for the assembled instruction/context. It keeps whole selected source passages rather than silently chopping their ends. It requests at most 2,048 output tokens. Actual provider usage is recorded when returned, including thinking tokens where available. Character budgets are not claims about exact Gemini input tokens or a currency-cost calculation.

Sending the entire database or entire chat would consume more context, time and potentially money, and introduce irrelevant or private material. Selecting evidence reduces that burden but can omit something important. Retrieval quality and memory selection therefore affect answer quality.

## Why an answer can sound right and still be wrong

A language model learns parameters during earlier training and generates plausible token sequences conditioned on input. Fluency is not an evidence check. It can rely on general learned patterns, misunderstand a passage, miss an exception, confuse history with verified facts, or receive poor retrieval results. Low temperature reduces variation; it does not establish truth.

We supply policies, require citations, allow insufficient-evidence and clarification responses, and reject malformed outputs. These reduce some failures but do not eliminate unsupported reasoning. A user saying 'I paid twice' remains an unverified statement until a trusted account tool checks records. Earlier model answers are also unverified history.

## Citations and their limits

The application assigns temporary IDs such as S1 to the current retrieved passages. The generator returns each claim with its source ID and an exact quote. The server checks the structured shape, supplied source membership and quote substring (with whitespace/Markdown normalization), then renders citation markers itself. It also rejects some obvious claims of completed actions. This pattern matching is not comprehensive action-claim detection.

Two different checks matter:

1. Citation integrity: does this ID exist, and is the quote really in that passage? Implemented automatically.
2. Semantic support: does that passage actually justify this claim, including conditions and exceptions? Not automatically established.

For instance, quoting 'refunds require operator approval' while saying 'approval is unnecessary' has valid copied text and wrong meaning. A regression test intentionally demonstrates that limitation. Human review and a labelled answer evaluation are still needed. Another LLM could act as a reviewer, but it would introduce cost, latency and its own mistakes; it would not become proof.

## Structured output

Instead of asking for unrestricted prose, we request JSON with a status, claims, citations and optional clarification. A schema describes the allowed shape. Python validation checks the received object again because provider support alone is not sufficient. Truncated JSON, unknown citations and invented quotes fail without publishing an accepted answer. Structured output makes software integration easier; it does not guarantee correct content.

## Streaming

An ordinary response arrives as one completed object. A streamed response arrives in pieces over a connection. Provider token production, network chunks and our displayed text chunks are different units.

Our Gemini adapter receives provider events, but buffers raw generated JSON until the completed answer passes structural and citation checks. The client receives retrieval/status/progress events while waiting, then a validated event, accepted answer chunks and a final result. This is validation-first streaming, not immediate raw token display. It delays first answer text in exchange for withholding outputs that fail these checks. A disconnect or provider failure must not add a partial accepted answer to future memory. There is no automatic retry that might unexpectedly consume more quota.

## Memory without training

Gemini does not automatically know the previous request. We save conversation turns in SQLite and explicitly include selected previous questions and accepted answers in the next prompt. All completed turns remain stored, but only up to three recent completed turns fitting the history budget are sent. Failed turns are recorded as failures and excluded from prompt history. A restart preserves stored conversations.

A short follow-up such as 'Does that mean it was refunded?' can use the previous question to improve retrieval. Sources are retrieved again; old S1 citation labels are not treated as current evidence. Memory helps interpret a reference like 'that'. It does not verify the underlying account claim or update model weights.

Conversation ownership comes from the authenticated database identity, not a user-supplied owner field. Another customer or an operator cannot read someone else's private conversation. Single-running-turn constraints and revision checks prevent overlapping writes. Startup interruption recovery currently assumes one server worker.

Alternatives include a stateless API (simpler, loses follow-up context), the whole transcript (more context/cost), summaries (compact but lossy), structured verified case state (suited to later investigations), and semantic retrieval over old messages (useful for long histories but adds retrieval and privacy concerns). We chose a bounded recent history because it is explicit and inspectable for this stage. Persistent memory is distinct from a model download cache, runtime RAM and fine-tuning.

## Evaluating the system

Retrieval evaluation asks whether supporting passages appeared in the results. Answer evaluation asks whether the response follows those passages, preserves conditions, abstains appropriately and avoids invented account facts. System tests ask whether ownership, failure handling, persistence and stream ordering work.

The automated suite uses deterministic test providers for reproducibility. These test protocol and safeguards, not Gemini intelligence. The explicit live demo calls the real provider on three synthetic development cases and records answers, evidence, timings and provider usage. This tiny exercise is not a held-out benchmark or production reliability estimate. Review each claim against its quote and full policy, not merely whether the status says answered.

Next, tools can supply verified account facts. Even then, reading records, proposing an action, approving it and executing it must remain separate operations with server-side permissions and checks.
