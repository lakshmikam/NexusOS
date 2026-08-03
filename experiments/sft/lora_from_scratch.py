"""
lora_from_scratch.py

Day 2 - LoRA from Scratch

Features:
- LoRALinear implementation
- Frozen base weights
- Low-rank adapters (A and B)
- Alpha scaling
- Weight merging
- Automatic replacement of Linear layers
- Save / Load LoRA adapters
- Merge verification
"""

from __future__ import annotations

import copy
from typing import Iterable, Tuple

import torch
import torch.nn as nn


# ============================================================
# LoRA Linear Layer
# ============================================================

class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear layer with LoRA adapters.

    Original:
        y = W₀x

    LoRA:
        y = W₀x + (α/r) BAx
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
    ):
        super().__init__()

        if not isinstance(base_layer, nn.Linear):
            raise TypeError("base_layer must be nn.Linear")

        self.base_layer = base_layer

        # Freeze pretrained weights
        self.base_layer.weight.requires_grad = False

        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # LoRA parameters
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))

        # Recommended initialization
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward:

        base = W₀x

        lora = xAᵀBᵀ

        output = base + scaling * lora
        """

        base_out = self.base_layer(x)

        if self.merged:
            return base_out

        lora_out = (
            x @ self.lora_A.T
            @ self.lora_B.T
        ) * self.scaling

        return base_out + lora_out

    @torch.no_grad()
    def merge(self):
        """
        Merge LoRA weights into base weights.

        W ← W + (α/r)BA

        After merging, inference behaves like
        a normal Linear layer.
        """

        if self.merged:
            return

        delta_weight = (
            self.lora_B @ self.lora_A
        ) * self.scaling

        self.base_layer.weight += delta_weight

        self.merged = True

    @torch.no_grad()
    def unmerge(self):
        """
        Restore original frozen weights.
        """

        if not self.merged:
            return

        delta_weight = (
            self.lora_B @ self.lora_A
        ) * self.scaling

        self.base_layer.weight -= delta_weight

        self.merged = False


# ============================================================
# Replace Linear Layers
# ============================================================

def replace_with_lora(
    module: nn.Module,
    target_modules: Iterable[str] = ("q_proj", "v_proj"),
    rank: int = 8,
    alpha: float = 16.0,
):
    """
    Recursively replace selected Linear layers
    with LoRALinear.

    Example:
        q_proj
        v_proj
    """

    for name, child in module.named_children():

        if (
            isinstance(child, nn.Linear)
            and name in target_modules
        ):
            setattr(
                module,
                name,
                LoRALinear(
                    child,
                    rank=rank,
                    alpha=alpha,
                ),
            )

        else:
            replace_with_lora(
                child,
                target_modules,
                rank,
                alpha,
            )


# ============================================================
# Save only LoRA adapters
# ============================================================

def save_lora(model: nn.Module, path: str):

    adapter_state = {}

    for name, param in model.state_dict().items():

        if (
            "lora_A" in name
            or "lora_B" in name
        ):
            adapter_state[name] = param.cpu()

    torch.save(adapter_state, path)


# ============================================================
# Load LoRA adapters
# ============================================================

def load_lora(model: nn.Module, path: str):

    state = torch.load(path, map_location="cpu")

    model.load_state_dict(
        state,
        strict=False,
    )


# ============================================================
# Merge every LoRA layer
# ============================================================

def merge_all_lora(model: nn.Module):

    for module in model.modules():

        if isinstance(module, LoRALinear):
            module.merge()


# ============================================================
# Count Parameters
# ============================================================

def count_parameters(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


# ============================================================
# Demo Model
# ============================================================

class TinyAttention(nn.Module):

    def __init__(self):

        super().__init__()

        self.q_proj = nn.Linear(64, 64)
        self.k_proj = nn.Linear(64, 64)
        self.v_proj = nn.Linear(64, 64)
        self.o_proj = nn.Linear(64, 64)

    def forward(self, x):

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        return self.o_proj(q + k + v)


# ============================================================
# Verification
# ============================================================

def verify_merge():

    print("=" * 60)
    print("VERIFYING MERGE")
    print("=" * 60)

    torch.manual_seed(42)

    base = nn.Linear(32, 64)

    lora = LoRALinear(
        copy.deepcopy(base),
        rank=8,
        alpha=16,
    )

    # Simulate training
    nn.init.normal_(lora.lora_B)

    x = torch.randn(5, 32)

    out_before = lora(x)

    lora.merge()

    out_after = lora(x)

    max_diff = (out_before - out_after).abs().max()

    print(f"Maximum difference: {max_diff:.10f}")

    assert torch.allclose(
        out_before,
        out_after,
        atol=1e-6,
    )

    print("Merge verified successfully!\n")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    torch.manual_seed(42)

    print("=" * 60)
    print("LoRA FROM SCRATCH")
    print("=" * 60)

    model = TinyAttention()

    print("\nOriginal Model\n")
    print(model)

    replace_with_lora(
        model,
        target_modules=("q_proj", "v_proj"),
        rank=8,
        alpha=16,
    )

    print("\nAfter LoRA Replacement\n")
    print(model)

    total, trainable = count_parameters(model)

    print("\nParameter Statistics")
    print("--------------------")
    print(f"Total Parameters     : {total:,}")
    print(f"Trainable Parameters : {trainable:,}")

    x = torch.randn(2, 64)

    y = model(x)

    print("\nForward Pass")
    print("------------")
    print("Input Shape :", x.shape)
    print("Output Shape:", y.shape)

    save_lora(model, "adapter.pt")
    print("\nSaved adapter -> adapter.pt")

    load_lora(model, "adapter.pt")
    print("Loaded adapter successfully")

    verify_merge()

    print("=" * 60)
    print("Everything completed successfully!")
    print("=" * 60)