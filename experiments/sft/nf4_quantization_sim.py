"""
Day 3 - NF4 Quantization Simulation

Topics Covered
--------------
✓ Uniform INT4 Quantization
✓ NF4 Quantization
✓ Block-wise Scaling
✓ Dequantization
✓ Mean Squared Error Comparison
✓ Memory Footprint Calculation

Author: Lakshmi Lekhya Kambala
"""

import torch
import torch.nn.functional as F


# ============================================================
# NF4 Lookup Table
# (bitsandbytes implementation values)
# ============================================================

NF4_CODEBOOK = torch.tensor([
    -1.0000,
    -0.6962,
    -0.5251,
    -0.3949,
    -0.2844,
    -0.1848,
    -0.0911,
     0.0000,
     0.0796,
     0.1609,
     0.2461,
     0.3379,
     0.4407,
     0.5626,
     0.7230,
     1.0000
], dtype=torch.float32)


# ============================================================
# Generate Gaussian Weight Matrix
# ============================================================

def generate_weights():

    return torch.randn(4096, 4096) * 0.02


# ============================================================
# Block Scale
# ============================================================

def compute_block_scale(block):

    return block.abs().max().clamp(min=1e-8)


# ============================================================
# NF4 Quantization
# ============================================================

def nf4_quantize(weights, block_size=64):

    flat = weights.flatten()

    quantized_indices = []
    scales = []

    for start in range(0, flat.numel(), block_size):

        block = flat[start:start + block_size]

        scale = compute_block_scale(block)

        normalized = block / scale

        distance = torch.abs(
            normalized.unsqueeze(1) - NF4_CODEBOOK.unsqueeze(0)
        )

        indices = distance.argmin(dim=1)

        quantized_indices.append(indices)
        scales.append(scale)

    indices = torch.cat(quantized_indices)

    scales = torch.stack(scales)

    return indices, scales


# ============================================================
# NF4 Dequantization
# ============================================================

def nf4_dequantize(indices, scales, shape, block_size=64):

    recovered = []

    pointer = 0

    for scale in scales:

        block_indices = indices[pointer:pointer + block_size]

        values = NF4_CODEBOOK[block_indices] * scale

        recovered.append(values)

        pointer += block_size

    recovered = torch.cat(recovered)

    return recovered.view(shape)


# ============================================================
# Uniform INT4 Quantization
# ============================================================

def uniform_int4_quantize(weights):

    min_val = weights.min()

    max_val = weights.max()

    levels = torch.linspace(min_val, max_val, 16)

    distance = torch.abs(
        weights.unsqueeze(-1) - levels
    )

    indices = distance.argmin(dim=-1)

    return indices, levels


# ============================================================
# Uniform INT4 Dequantization
# ============================================================

def uniform_int4_dequantize(indices, levels):

    return levels[indices]


# ============================================================
# Memory Calculator
# ============================================================

def quantized_memory(shape,
                     block_size=64,
                     double_quant=True):

    total_params = shape[0] * shape[1]

    weight_bits = total_params * 4

    num_blocks = total_params / block_size

    if double_quant:

        scale_bits = num_blocks * 8

        second_scale_bits = (num_blocks / 256) * 32

        total_bits = (
            weight_bits
            + scale_bits
            + second_scale_bits
        )

    else:

        scale_bits = num_blocks * 32

        total_bits = weight_bits + scale_bits

    return total_bits / 8 / (1024 ** 3)


# ============================================================
# Main Demonstration
# ============================================================

def main():

    print("=" * 60)
    print("NF4 QUANTIZATION SIMULATION")
    print("=" * 60)

    print("\nGenerating Gaussian Weight Matrix...")

    weights = generate_weights()

    print("Shape :", tuple(weights.shape))

    print("\nRunning NF4 Quantization...")

    nf4_indices, nf4_scales = nf4_quantize(weights)

    nf4_recovered = nf4_dequantize(
        nf4_indices,
        nf4_scales,
        weights.shape
    )

    print("Done.")

    print("\nRunning Uniform INT4 Quantization...")

    int4_indices, int4_levels = uniform_int4_quantize(weights)

    int4_recovered = uniform_int4_dequantize(
        int4_indices,
        int4_levels
    )

    print("Done.")

    nf4_mse = F.mse_loss(
        nf4_recovered,
        weights
    )

    int4_mse = F.mse_loss(
        int4_recovered,
        weights
    )

    print("\n================ RESULTS ================")

    print(f"NF4 MSE           : {nf4_mse:.10f}")

    print(f"Uniform INT4 MSE  : {int4_mse:.10f}")

    print("\nMemory Estimates")

    print("----------------")

    fp16 = weights.numel() * 2 / (1024 ** 3)

    nf4 = quantized_memory(
        weights.shape,
        double_quant=True
    )

    int4 = quantized_memory(
        weights.shape,
        double_quant=False
    )

    print(f"FP16              : {fp16:.4f} GB")

    print(f"NF4 + DQ          : {nf4:.4f} GB")

    print(f"INT4              : {int4:.4f} GB")

    print("\nComparison")

    if nf4_mse < int4_mse:

        print("✓ NF4 produces lower reconstruction error.")

    else:

        print("✓ Uniform INT4 unexpectedly performed better.")

    print("\nSimulation Complete!")


if __name__ == "__main__":
    main()