"""
Day 8: Export and Serve

Workflow:

    Base FP16 model
          +
    trained LoRA adapter
          |
          v
    merge_and_unload()
          |
          v
    merged FP16 model
          |
          v
    ./merged_model

Run from repository root:

    python -m experiments.sft.export_and_serve
"""

import os
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from peft import PeftModel


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B"

ADAPTER_PATH = (
    "./outputs/qwen2.5-1.5b-codealpaca-lora"
)

MERGED_MODEL_PATH = "./merged_model"


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

    tokenizer.save_pretrained(
        MERGED_MODEL_PATH
    )

    print("✅ Tokenizer loaded.")

    return tokenizer


# ============================================================
# 2. LOAD FP16 BASE MODEL
# ============================================================

def load_fp16_base_model():

    print("\n" + "=" * 70)
    print("Loading base model in FP16...")
    print("=" * 70)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,

        torch_dtype=torch.float16,

        device_map="auto",

        trust_remote_code=True,
    )

    print("✅ FP16 base model loaded.")

    return model


# ============================================================
# 3. LOAD LoRA ADAPTER
# ============================================================

def load_lora_model():

    if not os.path.exists(
        ADAPTER_PATH
    ):

        raise FileNotFoundError(
            f"\nLoRA adapter not found:\n"
            f"{ADAPTER_PATH}\n\n"
            "Run Day 6 training first."
        )

    base_model = load_fp16_base_model()

    print("\n" + "=" * 70)
    print("Loading trained LoRA adapter...")
    print("=" * 70)

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
    )

    print("✅ LoRA adapter loaded.")

    return model


# ============================================================
# 4. MERGE LoRA
# ============================================================

def merge_lora_weights(model):

    print("\n" + "=" * 70)
    print("Merging LoRA weights...")
    print("=" * 70)

    print(
        "\nMathematical operation:"
    )

    print(
        "W_final = W_base + (alpha / r) * B * A"
    )

    merged_model = model.merge_and_unload()

    print(
        "\n✅ LoRA adapters merged into base weights."
    )

    return merged_model


# ============================================================
# 5. SAVE MERGED MODEL
# ============================================================

def save_merged_model(
    model,
    tokenizer,
):

    print("\n" + "=" * 70)
    print("Saving merged FP16 model...")
    print("=" * 70)

    os.makedirs(
        MERGED_MODEL_PATH,
        exist_ok=True,
    )

    model.save_pretrained(
        MERGED_MODEL_PATH,

        safe_serialization=True,
    )

    tokenizer.save_pretrained(
        MERGED_MODEL_PATH
    )

    print(
        f"\n✅ Merged model saved to:"
        f"\n{MERGED_MODEL_PATH}"
    )


# ============================================================
# 6. VALIDATE MERGED STATE DICT
# ============================================================

def validate_merged_model():

    print("\n" + "=" * 70)
    print("Validating merged model...")
    print("=" * 70)

    model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    keys = list(
        model.state_dict().keys()
    )

    lora_keys = [
        key
        for key in keys
        if "lora_" in key
    ]

    if lora_keys:

        raise RuntimeError(
            "❌ Unmerged LoRA keys found:\n"
            + "\n".join(
                lora_keys[:20]
            )
        )

    print(
        "✅ Validation passed:"
        " no LoRA keys found."
    )

    print(
        f"Total state_dict keys: {len(keys)}"
    )

    del model


# ============================================================
# 7. MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DAY 8 — EXPORT AND SERVE")
    print("=" * 70)

    # --------------------------------------------------------
    # Create output directory before tokenizer save
    # --------------------------------------------------------

    os.makedirs(
        MERGED_MODEL_PATH,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    # --------------------------------------------------------
    # Base + LoRA
    # --------------------------------------------------------

    model = load_lora_model()

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    merged_model = merge_lora_weights(
        model
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_merged_model(
        merged_model,
        tokenizer,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validate_merged_model()

    print("\n" + "=" * 70)
    print("🎉 DAY 8 EXPORT COMPLETE")
    print("=" * 70)

    print(
        "\nMerged model:"
        f" {MERGED_MODEL_PATH}"
    )


if __name__ == "__main__":
    main()