# train_sft.py
#
# Day 6: Full QLoRA SFT Training Engine
#
# Day 5 pipeline:
# CodeAlpaca
#     ↓
# DataCollatorForSFT
#     ↓
# input_ids / labels / attention_mask
#
# Day 6:
# Qwen 1.5B
#     ↓
# 4-bit NF4
#     ↓
# LoRA
#     ↓
# SFTTrainer
#     ↓
# TRAIN
#     ↓
# trained LoRA adapter


import os
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from trl import SFTTrainer, SFTConfig

# ============================================================
# IMPORT DAY 5 PIPELINE
# ============================================================

from experiments.sft.dataset_pipeline import (
    load_sft_dataset,
    DataCollatorForSFT,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B"

OUTPUT_DIR = "./outputs/qwen2.5-1.5b-codealpaca-lora"

MAX_LENGTH = 2048

SEED = 42


# ============================================================
# 1. ENVIRONMENT CHECK
# ============================================================

print("=" * 70)
print("DAY 6 — FULL QLoRA SFT TRAINING")
print("=" * 70)

print("\nPyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required for QLoRA training."
    )

print(
    "GPU:",
    torch.cuda.get_device_name(0),
)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ============================================================
# 2. TOKENIZER
# ============================================================

print("\n" + "=" * 70)
print("Loading tokenizer...")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

tokenizer.padding_side = "right"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded.")
print("Vocabulary size:", len(tokenizer))
print("PAD token:", tokenizer.pad_token)
print("EOS token:", tokenizer.eos_token)


# ============================================================
# 3. 4-BIT NF4 CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("Configuring 4-bit NF4...")
print("=" * 70)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,

    # NormalFloat4
    bnb_4bit_quant_type="nf4",

    # Computation happens in BF16
    bnb_4bit_compute_dtype=torch.bfloat16,

    # Double quantization
    bnb_4bit_use_double_quant=True,
)


# ============================================================
# 4. LOAD BASE MODEL
# ============================================================

print("\nLoading Qwen 1.5B in 4-bit NF4...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

print("✅ Base model loaded.")

print(
    "GPU memory allocated:",
    f"{torch.cuda.memory_allocated() / 1e9:.2f} GB",
)


# ============================================================
# 5. PREPARE FOR K-BIT TRAINING
# ============================================================

print("\n" + "=" * 70)
print("Preparing model for k-bit training...")
print("=" * 70)

model = prepare_model_for_kbit_training(model)

print("✅ Model prepared.")


# ============================================================
# 6. LoRA CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("Configuring LoRA...")
print("=" * 70)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,

    # Apply LoRA to all linear layers
    target_modules="all-linear",

    bias="none",

    task_type="CAUSAL_LM",
)


# ============================================================
# 7. ATTACH LoRA
# ============================================================

print("\nAttaching LoRA adapters...")

model = get_peft_model(
    model,
    lora_config,
)

print("✅ LoRA adapters attached.")


# ============================================================
# 8. PARAMETER VERIFICATION
# ============================================================

print("\n" + "=" * 70)
print("PARAMETER SUMMARY")
print("=" * 70)

total_params = 0
trainable_params = 0

for name, param in model.named_parameters():

    total_params += param.numel()

    if param.requires_grad:
        trainable_params += param.numel()


trainable_percentage = (
    100.0 * trainable_params / total_params
)

print(
    f"Total parameters:     {total_params:,}"
)

print(
    f"Trainable parameters: {trainable_params:,}"
)

print(
    f"Trainable percentage: {trainable_percentage:.4f}%"
)


# Verify that base parameters are frozen.
base_trainable = []

for name, param in model.named_parameters():

    if "lora_" not in name and param.requires_grad:
        base_trainable.append(name)


if len(base_trainable) != 0:

    raise RuntimeError(
        "Some base model parameters are still trainable:\n"
        + "\n".join(base_trainable[:20])
    )


print("✅ Base model parameters are frozen.")


# ============================================================
# 9. LOAD THE SAME DATASET FROM DAY 5
# ============================================================

print("\n" + "=" * 70)
print("Loading Day 5 SFT dataset...")
print("=" * 70)

dataset = load_sft_dataset()

train_dataset = dataset["train"]

print("\nDataset:")
print(train_dataset)

print("\nColumns:")
print(train_dataset.column_names)

print("\nFirst example:")
print(train_dataset[0])


# ============================================================
# 10. REUSE THE DAY 5 COLLATOR
# ============================================================

print("\n" + "=" * 70)
print("Creating Day 5 DataCollatorForSFT...")
print("=" * 70)

data_collator = DataCollatorForSFT(
    tokenizer=tokenizer,
    max_length=MAX_LENGTH,
)

print("✅ Day 5 collator connected to Day 6.")


# ============================================================
# 11. TRAINING CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("Configuring training...")
print("=" * 70)

training_args = SFTConfig(

    output_dir=OUTPUT_DIR,

    # --------------------------------------------------------
    # Batch
    # --------------------------------------------------------

    per_device_train_batch_size=2,

    gradient_accumulation_steps=8,

    # Effective batch size:
    #
    # 2 × 8 × 1 GPU = 16
    #

    # --------------------------------------------------------
    # Epochs
    # --------------------------------------------------------

    num_train_epochs=1,

    # --------------------------------------------------------
    # Learning rate
    # --------------------------------------------------------

    learning_rate=2e-4,

    lr_scheduler_type="cosine",

    warmup_ratio=0.03,

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    bf16=True,

    fp16=False,

    # --------------------------------------------------------
    # Memory
    # --------------------------------------------------------

    gradient_checkpointing=True,

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logging_strategy="steps",

    logging_steps=5,

    # We can enable W&B later after the pipeline is verified.
    report_to="none",

    # --------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------

    save_strategy="steps",

    save_steps=50,

    save_total_limit=2,

    # --------------------------------------------------------
    # Sequence length
    # --------------------------------------------------------

    max_length=MAX_LENGTH,

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    seed=SEED,

    # --------------------------------------------------------
    # Do not automatically pack multiple examples together.
    # Our Day 5 collator handles each example explicitly.
    # --------------------------------------------------------

    packing=False,

    # Keep prompt/completion columns available for our
    # custom Day 5 collator.
    remove_unused_columns=False,
)


# ============================================================
# 12. CREATE SFT TRAINER
# ============================================================

print("\n" + "=" * 70)
print("Creating SFTTrainer...")
print("=" * 70)

trainer = SFTTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    # THIS IS THE IMPORTANT PART:
    #
    # Day 5 → Day 6
    #
    # Our custom collator performs:
    #
    # prompt tokenization
    # response tokenization
    # concatenation
    # -100 masking
    # dynamic padding
    # attention mask
    #

    data_collator=data_collator,

    # We deliberately do not ask SFTTrainer to create
    # another tokenizer/collator pipeline.
    processing_class=None,
)


print("✅ SFTTrainer created.")


# ============================================================
# 13. TRAINING SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)

effective_batch_size = (
    2 * 8 * 1
)

print("Model:", MODEL_NAME)

print("Dataset:", train_dataset)

print("Quantization: 4-bit NF4")

print("Double quantization: enabled")

print("Compute dtype: BF16")

print("LoRA rank:", 16)

print("LoRA alpha:", 32)

print("LoRA dropout:", 0.05)

print("Per-device batch size:", 2)

print("Gradient accumulation:", 8)

print("Number of GPUs:", 1)

print("Effective batch size:", effective_batch_size)

print("Learning rate:", 2e-4)

print("Scheduler:", "cosine")

print("Warmup ratio:", 0.03)

print("Epochs:", 1)


# ============================================================
# 14. TRAIN
# ============================================================

print("\n" + "=" * 70)
print("🚀 STARTING QLoRA TRAINING")
print("=" * 70)

train_result = trainer.train()


# ============================================================
# 15. SAVE TRAINED LoRA ADAPTER
# ============================================================

print("\n" + "=" * 70)
print("Saving trained LoRA adapter...")
print("=" * 70)

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(
    OUTPUT_DIR
)

print(
    f"✅ Trained adapter saved to:\n{OUTPUT_DIR}"
)


# ============================================================
# 16. PRINT METRICS
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

if train_result is not None:

    print("\nTraining metrics:")

    for key, value in train_result.metrics.items():

        print(
            f"{key}: {value}"
        )


print("\n🎉 DAY 6 QLoRA TRAINING COMPLETE!")