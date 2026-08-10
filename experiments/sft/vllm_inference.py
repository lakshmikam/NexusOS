"""
Day 8: vLLM Inference Benchmark

Loads the merged model using vLLM and performs inference.

Run:

    python -m experiments.sft.vllm_inference

Install first:

    pip install vllm
"""

import time

from vllm import LLM, SamplingParams


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "./merged_model"

PROMPTS = [
    "Write a Python function to reverse a linked list.",
    "Explain binary search and its time complexity.",
    "Write a Java function to check whether a string is a palindrome.",
    "Explain the difference between a stack and a queue.",
    "Write a Python function to find the maximum element in an array.",
]


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("=" * 70)
    print("Loading merged model with vLLM...")
    print("=" * 70)

    llm = LLM(
        model=MODEL_PATH,

        # vLLM handles the KV cache / paged attention
        # internally.
        trust_remote_code=True,
    )

    print("✅ vLLM model loaded.")

    return llm


# ============================================================
# GENERATION
# ============================================================

def generate(
    llm,
    prompts,
):

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
    )

    start_time = time.perf_counter()

    outputs = llm.generate(
        prompts,
        sampling_params,
    )

    end_time = time.perf_counter()

    elapsed = (
        end_time - start_time
    )

    return outputs, elapsed


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    outputs,
    elapsed,
):

    print("\n" + "=" * 70)
    print("GENERATED RESPONSES")
    print("=" * 70)

    total_tokens = 0

    for index, output in enumerate(
        outputs,
        start=1,
    ):

        print(
            f"\n{'=' * 70}"
        )

        print(
            f"PROMPT {index}"
        )

        print(
            f"{'=' * 70}"
        )

        print(
            output.prompt
        )

        if output.outputs:

            generated = output.outputs[0]

            print("\nRESPONSE:")
            print(
                generated.text
            )

            total_tokens += len(
                generated.token_ids
            )

    print("\n" + "=" * 70)
    print("BENCHMARK")
    print("=" * 70)

    print(
        f"Requests: {len(outputs)}"
    )

    print(
        f"Total generated tokens: "
        f"{total_tokens}"
    )

    print(
        f"Total time: "
        f"{elapsed:.4f} seconds"
    )

    if elapsed > 0:

        print(
            f"Throughput: "
            f"{total_tokens / elapsed:.2f} tokens/sec"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DAY 8 — vLLM INFERENCE")
    print("=" * 70)

    llm = load_model()

    outputs, elapsed = generate(
        llm,
        PROMPTS,
    )

    print_results(
        outputs,
        elapsed,
    )

    print("\n" + "=" * 70)
    print("🎉 vLLM BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()