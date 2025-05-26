
## FouRA: Fourier Low Rank Adaptation
#### (Unofficial PyTorch Implementation)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Inspired by [FouRA: Fourier Low Rank Adaptation](https://arxiv.org/abs/2406.08798v1), this is a community implementation for the paper. The project scaffolding was mostly written out by gemini so yeah hf with the AI gen code-I'll try and fix a bunch of stuff later

## What is FouRA (Fourier Low-Rank Adaptation)?

FouRA builds upon the concept of Low-Rank Adaptation (LoRA) for efficiently fine-tuning large pre-trained models. While LoRA is effective, it can sometimes suffer from issues like information loss during projection, overfitting (especially with higher ranks or smaller datasets), and a tendency towards generating less diverse outputs (distribution collapse).

The core innovation of FouRA is to perform the low-rank adaptation in the **frequency domain** rather than the standard feature space. This is motivated by the idea that the frequency domain offers an inherently more compact and decorrelated representation of the input features, potentially leading to better generalization and more robust adaptation.

### Key Concepts:

1.  **Frequency Domain Transformation**: 
    Instead of directly applying LoRA matrices (A and B) to the input features `z_in`, FouRA first transforms `z_in` into the frequency domain using a Discrete Fourier Transform (DFT) or Discrete Cosine Transform (DCT). Let's denote this transform as `ℱ`.

2.  **Low-Rank Adaptation in Frequency Space**:
    The down-projection (matrix `A` to rank `r`) and up-projection (matrix `B` back to the original dimension) are learned and applied to this frequency-transformed representation.
    A scaling factor `α` (alpha), similar to LoRA, controls the strength of the adaptation. This `α` is often folded into the low-rank subspace.

3.  **Inverse Transformation**: 
    After the adaptation in the frequency domain, the result is transformed back to the original feature space using the inverse frequency transform `ℱ⁻¹`.

4.  **Overall Formulation (Simplified)**:
    The output `z_out` from a FouRA-adapted layer can be conceptualized as:

    `z_out = W₀ * z_in + ℱ⁻¹( B * α * A * ℱ(z_in) )`

    Where:
    *   `W₀` is the original pre-trained weight matrix.
    *   `z_in` is the input to the layer.
    *   `ℱ` and `ℱ⁻¹` are the forward and inverse frequency transforms.
    *   `A` is the down-projection matrix in the frequency domain.
    *   `B` is the up-projection matrix in the frequency domain.
    *   `α` is the adapter strength scaling factor.

5.  **Adaptive Rank Gating (Optional Enhancement)**:
    FouRA also introduces an adaptive gating mechanism `𝒢` that operates within the low-rank frequency subspace. This mechanism dynamically adjusts the effective rank of the adaptation based on the input, allowing the model to be more flexible during both training and inference.
    The equation with gating becomes:

    `z_out = W₀ * z_in + ℱ⁻¹( B * α * 𝒢(z_lr) ⋅ (A * ℱ(z_in)) )`

    Where `z_lr = A * ℱ(z_in)` is the low-rank representation in the frequency domain, and `𝒢(z_lr)` is the learned gate.

By operating in the frequency domain and optionally using adaptive rank gating, FouRA aims to achieve improved performance, better generalization, and more diverse outputs compared to standard LoRA, particularly for tasks sensitive to overfitting and representation quality.

This repository provides an implementation of the FouRA technique, allowing for its application to various models.

## Project Structure

```
Foura/
├── src/foura/    # core code (DCT, adapters, helpers)
├── notebooks/    # experiments & plots
├── examples/     # quick-start scripts
├── tests/        # pytest tests
├── requirements.txt
└── README.md
```

