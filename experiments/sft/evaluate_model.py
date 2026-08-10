"""
Day 7: Quantitative & Qualitative Evaluation

Evaluates:
1. Validation Perplexity using sliding-window evaluation
2. Base vs fine-tuned generation
3. ROUGE-1 / ROUGE-2 / ROUGE-L

The trained LoRA adapter is loaded on top of the base Qwen model.

Run from repository root:

    python -m experiments.sft.evaluate_model
"""

import os
import math
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import PeftModel


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B"

# Change this after training if your output directory differs.
ADAPTER_PATH = "./outputs/qwen2.5-1.5b-codealpaca-lora"

MAX_LENGTH = 2048
STRIDE = 512

MAX_NEW_TOKENS = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# TEST PROMPTS
# ============================================================

TEST_PROMPTS = [
    "Write a Python function to reverse a linked list.",
    "Explain binary search and give its time complexity.",
    "Write a Java function to check whether a string is a palindrome.",
    "Explain the difference between a stack and a queue.",
    "Write a Python function to find the maximum element in an array.",
]


# ============================================================
# 1. LOAD TOKENIZER
# ============================================================

def load_tokenizer():

    print("=" * 70)
    print("Loading tokenizer...")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


# ============================================================
# 2. LOAD BASE MODEL
# ============================================================

def load_base_model():

    print("\n" + "=" * 70)
    print("Loading base model...")
    print("=" * 70)

    if torch.cuda.is_available():

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    else:

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    model.eval()

    return model


# ============================================================
# 3. LOAD LoRA MODEL
# ============================================================

def load_finetuned_model():

    print("\n" + "=" * 70)
    print("Loading fine-tuned LoRA model...")
    print("=" * 70)

    if not os.path.exists(ADAPTER_PATH):

        raise FileNotFoundError(
            f"\nLoRA adapter not found:\n"
            f"{ADAPTER_PATH}\n\n"
            "Run Day 6 training first."
        )

    base_model = load_base_model()

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
    )

    model.eval()

    print("✅ LoRA adapter loaded.")

    return model


# ============================================================
# 4. SLIDING-WINDOW PERPLEXITY
# ============================================================

def compute_perplexity(
    model,
    tokenizer,
    text,
    max_len=2048,
    stride=512,
):
    """
    Compute perplexity using overlapping sliding windows.

    Perplexity:

        PPL = exp(mean negative log likelihood)

    Only newly introduced tokens in each window contribute
    to the loss. This avoids repeatedly counting the context
    tokens from overlapping windows.
    """

    encodings = tokenizer(
        text,
        return_tensors="pt",
    )

    input_ids = encodings.input_ids

    seq_len = input_ids.size(1)

    if seq_len == 0:
        return float("inf")

    nlls = []

    previous_end = 0

    for begin_loc in range(
        0,
        seq_len,
        stride,
    ):

        end_loc = min(
            begin_loc + max_len,
            seq_len,
        )

        input_ids_window = input_ids[
            :,
            begin_loc:end_loc,
        ]

        target_ids = input_ids_window.clone()

        # Number of new target tokens introduced
        # by this window.
        trg_len = end_loc - previous_end

        if trg_len <= 0:
            break

        # Ignore context tokens.
        target_ids[:, :-trg_len] = -100

        input_ids_window = input_ids_window.to(
            model.device
        )

        target_ids = target_ids.to(
            model.device
        )

        with torch.no_grad():

            outputs = model(
                input_ids=input_ids_window,
                labels=target_ids,
            )

        negative_log_likelihood = (
            outputs.loss * trg_len
        )

        nlls.append(
            negative_log_likelihood
        )

        previous_end = end_loc

        if end_loc >= seq_len:
            break

    total_nll = torch.stack(nlls).sum()

    perplexity = torch.exp(
        total_nll / seq_len
    )

    return perplexity.item()


# ============================================================
# 5. GENERATION
# ============================================================

def generate_response(
    model,
    tokenizer,
    prompt,
):
    """
    Deterministic generation for reproducible evaluation.
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,

            max_new_tokens=MAX_NEW_TOKENS,

            do_sample=False,

            pad_token_id=tokenizer.pad_token_id,

            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[1]:,
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return response.strip()


# ============================================================
# 6. BASE VS LoRA GENERATION
# ============================================================

def compare_generations(
    tokenizer,
    base_model,
    finetuned_model,
):

    print("\n" + "=" * 70)
    print("BASE vs FINE-TUNED GENERATION")
    print("=" * 70)

    results = []

    for index, prompt in enumerate(
        TEST_PROMPTS,
        start=1,
    ):

        print(
            f"\n{'=' * 70}"
        )

        print(
            f"TEST PROMPT {index}"
        )

        print(
            f"{'=' * 70}"
        )

        print("\nPROMPT:")
        print(prompt)

        base_response = generate_response(
            base_model,
            tokenizer,
            prompt,
        )

        finetuned_response = generate_response(
            finetuned_model,
            tokenizer,
            prompt,
        )

        print("\n--- BASE MODEL ---")
        print(base_response)

        print("\n--- FINE-TUNED MODEL ---")
        print(finetuned_response)

        results.append(
            {
                "prompt": prompt,
                "base": base_response,
                "finetuned": finetuned_response,
            }
        )

    return results


# ============================================================
# 7. ROUGE
# ============================================================

def compute_rouge(
    predictions,
    references,
):
    """
    Compute ROUGE-1, ROUGE-2 and ROUGE-L.

    The Hugging Face evaluate library is used as specified
    by the curriculum.
    """

    try:

        import evaluate

    except ImportError:

        raise ImportError(
            "Install the evaluation dependency with:\n"
            "pip install evaluate"
        )

    rouge = evaluate.load("rouge")

    results = rouge.compute(
        predictions=predictions,
        references=references,
    )

    print("\n" + "=" * 70)
    print("ROUGE RESULTS")
    print("=" * 70)

    print(
        f"ROUGE-1: {results['rouge1']:.4f}"
    )

    print(
        f"ROUGE-2: {results['rouge2']:.4f}"
    )

    print(
        f"ROUGE-L: {results['rougeL']:.4f}"
    )

    return results


# ============================================================
# 8. MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DAY 7 — MODEL EVALUATION")
    print("=" * 70)

    print(
        "\nDevice:",
        DEVICE,
    )

    tokenizer = load_tokenizer()

    # --------------------------------------------------------
    # Load base
    # --------------------------------------------------------

    base_model = load_base_model()

    # --------------------------------------------------------
    # Load fine-tuned
    # --------------------------------------------------------

    finetuned_model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
    )

    finetuned_model.eval()

    # --------------------------------------------------------
    # Perplexity
    # --------------------------------------------------------

    evaluation_text = """
    Python is a programming language widely used for
    machine learning, data science, automation, and software
    development. Neural networks contain layers of trainable
    parameters. Fine-tuning adapts a pretrained model to a
    specific downstream task.
    """

    print("\n" + "=" * 70)
    print("PERPLEXITY")
    print("=" * 70)

    base_ppl = compute_perplexity(
        base_model,
        tokenizer,
        evaluation_text,
        max_len=MAX_LENGTH,
        stride=STRIDE,
    )

    finetuned_ppl = compute_perplexity(
        finetuned_model,
        tokenizer,
        evaluation_text,
        max_len=MAX_LENGTH,
        stride=STRIDE,
    )

    print(
        f"\nBase model PPL:      {base_ppl:.4f}"
    )

    print(
        f"Fine-tuned model PPL: {finetuned_ppl:.4f}"
    )

    # --------------------------------------------------------
    # Generation comparison
    # --------------------------------------------------------

    generation_results = compare_generations(
        tokenizer,
        base_model,
        finetuned_model,
    )

    # --------------------------------------------------------
    # ROUGE
    #
    # These reference answers are intentionally simple.
    # Replace them with held-out reference answers from your
    # actual evaluation dataset before reporting final results.
    # --------------------------------------------------------

    references = [
        "Reverse a linked list by iterating through the nodes and changing each node's next pointer.",
        "Binary search repeatedly divides a sorted search range in half and has O(log n) time complexity.",
        "A palindrome string reads the same forward and backward.",
        "A stack follows LIFO while a queue follows FIFO.",
        "Iterate through the array while keeping track of the maximum element.",
    ]

    predictions = [
        result["finetuned"]
        for result in generation_results
    ]

    try:

        compute_rouge(
            predictions,
            references,
        )

    except Exception as exc:

        print(
            "\n⚠️ ROUGE evaluation skipped:"
        )

        print(exc)

    print("\n" + "=" * 70)
    print("DAY 7 EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()