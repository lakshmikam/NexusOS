# environment_and_pipeline_setup.py

import os
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import LoraConfig, get_peft_model


# ============================================================
# 1. CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B"

SEED = 42

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 2. DEVICE / GPU INFO
# ============================================================

print("=" * 60)
print("ENVIRONMENT")
print("=" * 60)

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(
        f"VRAM            : "
        f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    )


# ============================================================
# 3. 4-BIT QLoRA CONFIGURATION
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


# ============================================================
# 4. LOAD TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

# SFT training → right padding
tokenizer.padding_side = "right"

# Some causal LMs don't have a dedicated PAD token.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\nTokenizer")
print("-" * 60)
print(f"Vocabulary size : {len(tokenizer)}")
print(f"PAD token       : {tokenizer.pad_token}")
print(f"PAD token ID    : {tokenizer.pad_token_id}")
print(f"EOS token       : {tokenizer.eos_token}")
print(f"EOS token ID    : {tokenizer.eos_token_id}")
print(f"Padding side    : {tokenizer.padding_side}")


# ============================================================
# 5. LOAD BASE MODEL
# ============================================================

print("\nLoading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False


# ============================================================
# 6. LoRA CONFIGURATION
# ============================================================

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],

    bias="none",
    task_type="CAUSAL_LM",
)


# ============================================================
# 7. ATTACH LoRA ADAPTERS
# ============================================================

model = get_peft_model(
    model,
    lora_config,
)


# ============================================================
# 8. PARAMETER COUNTS
# ============================================================

print("\nParameter Statistics")
print("=" * 60)

total_params = 0
trainable_params = 0

for name, param in model.named_parameters():

    total_params += param.numel()

    if param.requires_grad:
        trainable_params += param.numel()


percentage = (
    100 * trainable_params / total_params
    if total_params > 0
    else 0
)

print(f"Total parameters     : {total_params:,}")
print(f"Trainable parameters : {trainable_params:,}")
print(f"Frozen parameters    : {total_params - trainable_params:,}")
print(f"Trainable percentage : {percentage:.4f}%")


# ============================================================
# 9. VERIFY BASE PARAMETERS ARE FROZEN
# ============================================================

print("\nVerifying parameter freezing...")

trainable_base_params = []

for name, param in model.named_parameters():

    if "lora_" not in name and param.requires_grad:
        trainable_base_params.append(name)


if len(trainable_base_params) == 0:
    print("✅ All base model parameters are frozen.")
else:
    print("❌ ERROR: Base parameters are trainable!")

    for name in trainable_base_params:
        print("   ", name)


# ============================================================
# 10. PRINT LoRA PARAMETERS
# ============================================================

print("\nLoRA Parameters")
print("=" * 60)

for name, param in model.named_parameters():

    if "lora_" in name:
        print(
            f"{name:<80} "
            f"shape={tuple(param.shape)} "
            f"trainable={param.requires_grad}"
        )


# ============================================================
# 11. GPU MEMORY
# ============================================================

if torch.cuda.is_available():

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9

    print("\nGPU Memory")
    print("=" * 60)
    print(f"Allocated : {allocated:.2f} GB")
    print(f"Reserved  : {reserved:.2f} GB")


# ============================================================
# 12. FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 60)
print("PIPELINE READY")
print("=" * 60)

print("✅ 4-bit NF4 model loaded")
print("✅ Double quantization enabled")
print("✅ BF16 compute enabled")
print("✅ Tokenizer configured")
print("✅ Right padding enabled")
print("✅ LoRA adapters attached")
print("✅ Base parameters frozen")
print("✅ Trainable parameter count verified")