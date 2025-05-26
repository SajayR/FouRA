import torch
import torch.nn as nn
import math

# 1D Discrete Cosine Transform (DCT)


class DCT1D(nn.Module):
    """Orthonormal DCT‑II basis"""
    def __init__(self, N):
        super().__init__()
        # basis B[k,n] = sqrt(2/N) · cos(pi*(n+0.5)*k/N), row 0 scaled by 1/√2
        n = torch.arange(N).reshape(1, -1).float()
        k = torch.arange(N).reshape(-1, 1).float()
        B = math.sqrt(2.0 / N) * torch.cos(math.pi * (n + 0.5) * k / N)
        B[0] /= math.sqrt(2.0)
        self.register_buffer('BT', B.t())    # [N, N] forward does x @ Bᵀ

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.shape
        flat = x.reshape(-1, orig[-1])
        out  = flat @ self.BT             # [*, N]
        return out.reshape(*orig)

    def inverse(self, X: torch.Tensor) -> torch.Tensor:
        # B orthonormal, inverse == BTᵀ mat‑mul
        orig = X.shape
        flat = X.reshape(-1, orig[-1])
        out  = flat @ self.BT.t()
        return out.reshape(*orig) 