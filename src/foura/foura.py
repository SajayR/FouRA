import torch
import torch.nn as nn
import torch.nn.functional as F
from .dct import DCT1D

class FouRA(nn.Module):
    """
    Args:
        N (int): Original embedding dim per token.
        rank (int): Bottleneck rank for low-rank projection.
        transform_type (str): One of {'none', 'fft', 'dct'} specifying the frequency transform.
        use_gate (bool): If True, enable per-rank entropy-based gating.
        alpha_init (float): Initial scale for sigmoid gating threshold.
        beta_init (float): Initial shift for sigmoid gating threshold.

    Forward:
        x: Tensor[B, T, N]  # B=batch, T=tokens, N=features
        Returns: Tensor[B, T, N] adapted with low-rank + gating
    """
    def __init__(self, N, rank, transform_type="none", use_gate=False,
                 alpha_init=1.0, beta_init=0.0):
        super().__init__()
        self.transform_type = transform_type
        self.use_gate = use_gate

        # - FFT gives (N//2+1) complex coefficients (real + imag)
        # - DCT preserves length but yields energy-compacted real coefficients
        if transform_type == "fft":
            self.freq_dim = N // 2 + 1
        elif transform_type == "dct":
            self.freq_dim = N
            self.dct = DCT1D(N) 
        else:
            self.freq_dim = N

        dtype = torch.cfloat if transform_type == "fft" else torch.float
        self.down = nn.Linear(self.freq_dim, rank, bias=False).to(dtype)
        self.up   = nn.Linear(rank, self.freq_dim, bias=False).to(dtype)

        if use_gate:
            # learnable per-rank scale and shift
            # alpha_i scales sensitivity; beta_i shifts activation threshold
            self.alpha = nn.Parameter(torch.ones(rank) * alpha_init)
            self.beta  = nn.Parameter(torch.ones(rank) * beta_init)

    def forward(self, x):
        """
        x: Tensor of shape [B, T, N]
           B = batch size, T = number of tokens (or sequence length),
           N = embedding / feature dimension per token.
        returns: [B, T, N]
        """
        B, T, N = x.shape
        if self.transform_type == "fft":
            X = torch.fft.rfft(x, dim=-1)
        elif self.transform_type == "dct":
            X = self.dct(x)
        else:
            X = x
        # X: [B, T, freq_dim]
        z = self.down(X)  # shape [B, T, rank]

        if self.use_gate:
            # distribution over tokens for each rank-channel
            if torch.is_complex(z): # torch softmax doesn't support complex yet-abs is workable for this
                real_scores = z.abs()
            else: 
                real_scores = z
                
            z_perm = real_scores.transpose(1, 2) # [B, rank, T]
            # yields distribution p_{i,k} per rank i
            p = F.softmax(z_perm, dim=-1)           # [B, rank, T]
            H = -(p * torch.log(p + 1e-12)).sum(dim=-1)  # entropy per rank: [B, rank]
            g = torch.sigmoid(self.alpha * (H - self.beta))  # [B, rank]
            # gate to each channel across tokens
            z = z * g.unsqueeze(1)  # [B, T, rank]

        Y = self.up(z)  # [B, T, freq_dim]

        if self.transform_type == "fft":
            out = torch.fft.irfft(Y, n=N, dim=-1)
        elif self.transform_type == "dct":
            out = self.dct.inverse(Y)
        else:
            out = Y

        return out 