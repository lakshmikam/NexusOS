# dataset_pipeline.py
#
# Day 5: Deterministic Instruction Dataset Engineering & Collators
#
# Pipeline:
# Hugging Face Dataset
#        ↓
# Prompt / Response formatting
#        ↓
# Separate tokenization
#        ↓
# input_ids = prompt_ids + response_ids
#        ↓
# labels = [-100] * prompt_length + response_ids
#        ↓
# Dynamic padding
#        ↓
# attention_mask
#        ↓
# PyTorch training batch


import torch

from torch.nn.utils.rnn import pad_sequence

from datasets import load_dataset

from transformers import AutoTokenizer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B"

MAX_LENGTH = 2048

DATASET_NAME = "HuggingFaceH4/CodeAlpaca_20K"


# ============================================================
# 1. LOAD TOKENIZER
# ============================================================

print("=" * 70)
print("Loading tokenizer...")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

# Right padding for SFT training.
tokenizer.padding_side = "right"

# Some causal language models do not have a dedicated PAD token.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


print(f"Tokenizer: {MODEL_NAME}")
print(f"Vocabulary size: {len(tokenizer)}")
print(f"PAD token: {tokenizer.pad_token}")
print(f"PAD token ID: {tokenizer.pad_token_id}")
print(f"EOS token: {tokenizer.eos_token}")
print(f"EOS token ID: {tokenizer.eos_token_id}")


# ============================================================
# 2. LOAD REAL SFT DATASET
# ============================================================

def load_sft_dataset():
    """
    Load the real instruction-tuning dataset.

    CodeAlpaca contains instruction/prompt examples paired
    with expected code-oriented completions.
    """

    print("\n" + "=" * 70)
    print("Loading SFT dataset...")
    print("=" * 70)

    dataset = load_dataset(DATASET_NAME)

    print(dataset)

    print("\nDataset columns:")
    print(dataset["train"].column_names)

    print("\nFirst raw example:")
    print(dataset["train"][0])

    return dataset


# ============================================================
# 3. JINJA2-STYLE CHAT FORMATTER
# ============================================================

def format_chat(prompt: str, response: str):
    """
    Convert a raw prompt/response pair into a structured
    user/assistant conversation.

    We keep prompt and response as separate strings because
    the collator needs the exact prompt token length.
    """

    prompt_string = (
        "<|im_start|>user\n"
        f"{prompt}"
        "\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    response_string = (
        f"{response}"
        "\n<|im_end|>"
    )

    return prompt_string, response_string


# ============================================================
# 4. CUSTOM SFT DATA COLLATOR
# ============================================================

class DataCollatorForSFT:

    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
    ):

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):

        input_ids_list = []
        labels_list = []

        # ----------------------------------------------------
        # Process every example in the batch
        # ----------------------------------------------------

        for item in batch:

            prompt_str = item["prompt"]

            response_str = item["completion"]

            # ------------------------------------------------
            # Format prompt and response
            # ------------------------------------------------

            prompt_str, response_str = format_chat(
                prompt_str,
                response_str,
            )

            # ------------------------------------------------
            # Tokenize separately
            # ------------------------------------------------

            prompt_ids = self.tokenizer.encode(
                prompt_str,
                add_special_tokens=False,
            )

            response_ids = self.tokenizer.encode(
                response_str,
                add_special_tokens=False,
            )

            # Add EOS so the model learns when the answer ends.
            response_ids = response_ids + [
                self.tokenizer.eos_token_id
            ]

            # ------------------------------------------------
            # Concatenate
            # ------------------------------------------------

            input_ids = prompt_ids + response_ids

            # ------------------------------------------------
            # Prompt-masked labels
            # ------------------------------------------------
            #
            # Prompt tokens → -100
            # Response tokens → actual token IDs
            #
            # CrossEntropyLoss ignores -100.
            # Therefore the model is trained on the response,
            # not on reproducing the prompt.
            # ------------------------------------------------

            labels = (
                [-100] * len(prompt_ids)
                + response_ids
            )

            # ------------------------------------------------
            # Truncate
            # ------------------------------------------------

            if len(input_ids) > self.max_length:

                input_ids = input_ids[:self.max_length]

                labels = labels[:self.max_length]

            # ------------------------------------------------
            # Convert to tensors
            # ------------------------------------------------

            input_ids_list.append(
                torch.tensor(
                    input_ids,
                    dtype=torch.long,
                )
            )

            labels_list.append(
                torch.tensor(
                    labels,
                    dtype=torch.long,
                )
            )

        # ====================================================
        # DYNAMIC PADDING
        # ====================================================

        # Pad input IDs with the tokenizer's PAD token.
        padded_inputs = pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        # Pad labels with -100.
        #
        # This is important because padding positions must
        # NOT contribute to the training loss.
        padded_labels = pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=-100,
        )

        # ====================================================
        # ATTENTION MASK
        # ====================================================

        # Real tokens → 1
        # Padding tokens → 0
        attention_mask = (
            padded_inputs != self.tokenizer.pad_token_id
        ).long()

        # ====================================================
        # RETURN TRAINING BATCH
        # ====================================================

        return {
            "input_ids": padded_inputs,
            "labels": padded_labels,
            "attention_mask": attention_mask,
        }


# ============================================================
# 5. SANITY CHECK
# ============================================================

def run_sanity_check(dataset):

    print("\n" + "=" * 70)
    print("Running DataCollator sanity check...")
    print("=" * 70)

    # Take three real examples from the dataset.
    samples = [
        dataset["train"][0],
        dataset["train"][1],
        dataset["train"][2],
    ]

    collator = DataCollatorForSFT(
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    batch = collator(samples)

    # --------------------------------------------------------
    # Print batch structure
    # --------------------------------------------------------

    print("\nBatch keys:")
    print(batch.keys())

    print("\nTensor information:")

    for key, tensor in batch.items():

        print(
            f"{key:<20}"
            f"shape={tuple(tensor.shape)} "
            f"dtype={tensor.dtype}"
        )

    # --------------------------------------------------------
    # Verify shapes
    # --------------------------------------------------------

    assert batch["input_ids"].dim() == 2

    assert (
        batch["labels"].shape
        == batch["input_ids"].shape
    )

    assert (
        batch["attention_mask"].shape
        == batch["input_ids"].shape
    )

    # --------------------------------------------------------
    # Verify prompt masking exists
    # --------------------------------------------------------

    assert (
        batch["labels"] == -100
    ).any(), "ERROR: No -100 prompt masking found."

    # --------------------------------------------------------
    # Verify attention mask is binary
    # --------------------------------------------------------

    unique_attention_values = torch.unique(
        batch["attention_mask"]
    ).tolist()

    assert set(unique_attention_values).issubset(
        {0, 1}
    ), "ERROR: Attention mask is not binary."

    # --------------------------------------------------------
    # Print an interpretable example
    # --------------------------------------------------------

    print("\nFirst example:")
    print("-" * 70)

    first_input = batch["input_ids"][0]

    first_labels = batch["labels"][0]

    print(
        "Number of input tokens:",
        first_input.shape[0],
    )

    print(
        "Number of ignored (-100) labels:",
        (first_labels == -100).sum().item(),
    )

    print(
        "Number of response labels:",
        (first_labels != -100).sum().item(),
    )

    # --------------------------------------------------------
    # Decode the first example
    # --------------------------------------------------------

    print("\nDecoded first example:")
    print("-" * 70)

    decoded = tokenizer.decode(
        first_input,
        skip_special_tokens=False,
    )

    print(decoded)

    # --------------------------------------------------------
    # Final success
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("✅ DAY 5 DATA PIPELINE SANITY CHECK PASSED")
    print("=" * 70)


# ============================================================
# 6. MAIN
# ============================================================

if __name__ == "__main__":

    dataset = load_sft_dataset()

    run_sanity_check(dataset)