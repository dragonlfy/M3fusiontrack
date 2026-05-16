"""
Cross-Modal Frequency Attention (CMFA) and Uncertainty-Aware Gating (UAG).

These are the two central blocks of the paper:

* CMFA (§3.4) -- a transformer block whose attention operates over the
  joint (modality, band) token grid and incorporates a learned relative
  bias ``phi(k_i, b_i, k_j, b_j)``.

* UAG (§3.5)  -- each (modality, band) branch outputs both a feature and an
  aleatoric uncertainty.  The gate is

      g_{k,b} = softmax( s_{k,b} − σ²_{k,b} / τ_g )

  so an uncertain branch is automatically down-weighted.

The implementations below are deliberately compact (no flash-attention,
no fused kernels) — they are meant to be read.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CMFA
# ---------------------------------------------------------------------------
class CrossModalFrequencyAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_modalities: int = 4,
        num_bands: int = 3,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.M = num_modalities
        self.B = num_bands
        # Learned relative bias  phi[k_i, b_i, k_j, b_j]  shared across heads.
        self.rel_bias = nn.Parameter(
            torch.zeros(num_modalities, num_bands,
                        num_modalities, num_bands)
        )
        nn.init.trunc_normal_(self.rel_bias, std=0.02)
        self.layers = nn.ModuleList([
            _CMFABlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])

    def _build_bias_table(self, n_tokens_per_cell: int, device, dtype) -> torch.Tensor:
        """Expand the (M, B, M, B) bias into a per-token (T, T) matrix."""
        # The 12 cells each contribute n_tokens_per_cell tokens. We tile the
        # bias so all token pairs (i, j) within the same (k_i, b_i, k_j, b_j)
        # share the same bias value.
        bias = self.rel_bias.reshape(self.M * self.B, self.M * self.B)
        bias = bias.repeat_interleave(n_tokens_per_cell, dim=0)
        bias = bias.repeat_interleave(n_tokens_per_cell, dim=1)
        return bias.to(device=device, dtype=dtype)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:           # noqa: D401
        """Apply L cross-modal frequency attention layers.

        Args:
            tokens: (N, M * B, L, D) — token sequence for each (modality, band)
                cell.  L includes the CLS token.

        Returns:
            Refined tokens of the same shape.
        """
        n, mb, l, d = tokens.shape
        assert mb == self.M * self.B
        # Flatten the cell grid into a single token sequence so a vanilla
        # attention layer can mix them.
        x = tokens.reshape(n, mb * l, d)
        bias = self._build_bias_table(l, x.device, x.dtype)            # (T, T)
        for layer in self.layers:
            x = layer(x, attn_bias=bias)
        return x.reshape(n, mb, l, d)


class _CMFABlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = _BiasedMHA(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_bias)
        x = x + self.mlp(self.norm2(x))
        return x


class _BiasedMHA(nn.Module):
    """Multi-head attention with an additive bias on the logits."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        n, t, d = x.shape
        qkv = self.qkv(x).reshape(n, t, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)                           # (N, H, T, Dh)
        logits = (q @ k.transpose(-1, -2)) * self.scale
        logits = logits + attn_bias.unsqueeze(0).unsqueeze(0)
        attn = logits.softmax(dim=-1)
        out = attn @ v                                                 # (N, H, T, Dh)
        out = out.transpose(1, 2).reshape(n, t, d)
        return self.proj(out)


# ---------------------------------------------------------------------------
# UAG
# ---------------------------------------------------------------------------
class UncertaintyAwareGating(nn.Module):
    """Per-cell tracking-quality score + aleatoric variance, fused into a gate.

    For each (modality, band) cell we read its CLS token, produce a scalar
    score s_{k,b} and a non-negative variance σ²_{k,b}, and combine them with
        g = softmax( s − σ² / τ_g )
    """

    def __init__(self, embed_dim: int, num_modalities: int = 4,
                 num_bands: int = 3, temperature: float = 1.0) -> None:
        super().__init__()
        self.M = num_modalities
        self.B = num_bands
        self.score_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )
        self.logvar_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )
        # Learnable inverse temperature, parameterised in log-space so it stays
        # positive.
        self.log_inv_tau = nn.Parameter(
            torch.tensor(math.log(1.0 / temperature))
        )

    def forward(self, tokens: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute fused feature, gate weights and per-cell variance.

        Args:
            tokens: (N, M*B, L, D) — output of CMFA.

        Returns:
            fused:    (N, L, D)       — uncertainty-weighted fusion.
            gate:     (N, M*B)        — softmax weights.
            log_var:  (N, M*B)        — log of σ², for the regulariser.
        """
        n, mb, l, d = tokens.shape
        cls = tokens[:, :, 0]                                          # (N, M*B, D)
        score = self.score_head(cls).squeeze(-1)                       # (N, M*B)
        log_var = self.logvar_head(cls).squeeze(-1)                    # (N, M*B)
        inv_tau = self.log_inv_tau.exp()
        var = log_var.exp()
        logits = score - var * inv_tau
        gate = logits.softmax(dim=-1)                                  # (N, M*B)
        fused = (gate.unsqueeze(-1).unsqueeze(-1) * tokens).sum(dim=1) # (N, L, D)
        return fused, gate, log_var

    @staticmethod
    def spectral_consistency_loss(gate: torch.Tensor,
                                  target_entropy: float = 1.6) -> torch.Tensor:
        """Eq. (7): encourage moderate gate entropy.

        Strong winner-take-all is unstable under modality dropout, but a
        flat gate wastes capacity. We push the entropy toward a target.
        """
        eps = 1e-8
        entropy = -(gate * (gate + eps).log()).sum(dim=-1).mean()
        return (entropy - target_entropy).pow(2)
