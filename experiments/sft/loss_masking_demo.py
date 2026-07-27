"""
loss_masking_demo.py

A step-by-step demonstration of how Supervised Fine-Tuning (SFT)
uses loss masking to train only on the assistant's response.

This script is educational and intentionally avoids using
pretrained tokenizers or transformer models so that each step
can be understood clearly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

USER_ROLE = "User:"
ASSISTANT_ROLE = "Assistant:"


# user sample interaction
# using two distinct variables to represent the user prompt and the assistant response in the dataset
user_prompt = "What is AI?"

assistant_response = "AI stands for Artificial Intelligence."

# combine the user prompt and assistant response into a single conversation string
conversation = (
    USER_ROLE + " " + user_prompt +
    "\n\n" + ASSISTANT_ROLE + " " + assistant_response
)

# start tokenize the conversation string into a list of tokens
tokens = conversation.split()

token_to_id = {}

for token in tokens:
    if token not in token_to_id:
        token_to_id[token] = len(token_to_id) + 1

# convert the conversation into a list of token IDs
input_ids = []

for token in tokens:
    input_ids.append(token_to_id[token])

labels = input_ids.copy()

assistant_start = tokens.index("Assistant:")

# mask the user prompt tokens in the labels by setting them to -100
for i in range(assistant_start + 1):
    labels[i] = -100


print("\nToken Mapping")
for token, idx in token_to_id.items():
    print(f"{token:15} -> {idx}")

print("\nInput IDs")
print(input_ids)

print("\nLabels")
print(labels)


# ==========================================
# Create Mock Logits
# ==========================================

vocab_size = len(token_to_id) + 1

logits = torch.randn(
    len(input_ids),
    vocab_size,
    requires_grad=True
)

labels = torch.tensor(labels)


# ==========================================
# Built-in Cross Entropy
# ==========================================

loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

builtin_loss = loss_fn(logits, labels)

print(f"\nBuilt-in Loss : {builtin_loss.item():6f}")


# ==========================================
# Backward Pass
# ==========================================

builtin_loss.backward()


# ==========================================
# Gradient Inspection
# ==========================================

print("\nGradient Shape:")
print(logits.grad.shape)

print("\nGradient Verification\n")

for i, target in enumerate(labels):

    gradient_sum = logits.grad[i].abs().sum().item()

    print(
        f"Position {i:2d} | "
        f"Target = {target.item():4d} | "
        f"Gradient Sum = {gradient_sum:.6f}"
    )


# ==========================================
# Manual Cross Entropy (No F.cross_entropy)
# ==========================================

# Convert logits into log probabilities
log_probs = F.log_softmax(logits, dim=-1)

# Create a mask for valid (non -100) targets
mask = labels != -100

# Keep only valid target indices
valid_targets = labels[mask]

# Keep only valid log probability rows
valid_log_probs = log_probs[mask]

# Pick the log probability corresponding to the correct token
selected_log_probs = valid_log_probs.gather(
    dim=1,
    index=valid_targets.unsqueeze(1)
).squeeze(1)

# Cross Entropy = - mean(log probability of correct tokens)
manual_loss = -selected_log_probs.mean()

print(f"\nManual Loss   : {manual_loss.item():6f}")


# ==========================================
# Compare Both Losses
# ==========================================

difference = abs(
    builtin_loss.item() -
    manual_loss.item()
)

print("\nLoss Comparison")
print(f"Built-in Loss : {builtin_loss.item():6f}")
print(f"Manual Loss   : {manual_loss.item():6f}")
print(f"Difference    : {difference:.10f}")

# ==========================================
# Manual Perplexity
# ==========================================

# Perplexity = exp(Cross Entropy)
manual_perplexity = torch.exp(manual_loss)

print(f"\nManual Perplexity : {manual_perplexity.item():6f}")

builtin_perplexity = torch.exp(builtin_loss)

print(f"Built-in Perplexity: {builtin_perplexity.item():6f}")

print(
    "Difference:",
    abs(
        manual_perplexity.item()
        - builtin_perplexity.item()
    )
)

# ==========================================
# Batch Size = 2 Example
# ==========================================

print("\n\n==============================")
print("Batch Size = 2 Example")
print("==============================")

sample1 = torch.tensor([
    -100,
    -100,
    -100,
    6,
    7,
    8,
])

sample2 = torch.tensor([
    -100,
    -100,
    5,
    6,
    7,
    -100,
])

batch_labels = torch.stack([sample1, sample2])

print(batch_labels)

batch_size = batch_labels.shape[0]
sequence_length = batch_labels.shape[1]

batch_logits = torch.randn(
    batch_size,
    sequence_length,
    vocab_size
)

batch_loss = F.cross_entropy(
    batch_logits.view(-1, vocab_size),
    batch_labels.view(-1),
    ignore_index=-100
)

print(f"\nBatch Loss : {batch_loss.item():6f}")

# ==========================================
# Right Padding Example
# ==========================================

print("\n\n==============================")
print("Right Padding Example")
print("==============================")

PAD = -100

sample1 = torch.tensor([
    -100,
    -100,
    6,
    7,
    8,
    PAD,
    PAD
])

sample2 = torch.tensor([
    -100,
    -100,
    -100,
    5,
    6,
    7,
    8
])

padded_labels = torch.stack([
    sample1,
    sample2
])

print(padded_labels)

batch_size = padded_labels.shape[0]
sequence_length = padded_labels.shape[1]

padded_logits = torch.randn(
    batch_size,
    sequence_length,
    vocab_size
)

padding_loss = F.cross_entropy(
    padded_logits.view(-1, vocab_size),
    padded_labels.view(-1),
    ignore_index=-100
)

print(f"\nPadding Loss : {padding_loss.item():6f}")
