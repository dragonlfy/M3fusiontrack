"""
Frequency-Aware Correlation (Section 3.6 of the paper).

Given fused template tokens T and search tokens S, we compute a correlation
volume in the frequency domain so that the network can express
spatial-frequency-dependent matching rather than the uniform inner product
used in OSTrack / ARTrack.

Concretely:

    F_T = FFT2(T_grid),  F_S = FFT2(S_grid)
    M(u, v) = Beta(u, v; α, β)        # learnable band-pass mask
    C = IFFT2( M ⊙ F_T conj(F_S) ).real
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyAwareCorrelation(nn.Module):
    def __init__(self, embed_dim: int, init_alpha: float = 2.0,
                 init_beta: float = 2.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # Beta-distribution parameters of the radial band-pass mask. Stored
        # in log-space so they stay positive after gradient updates.
        self.log_alpha = nn.Parameter(torch.tensor(float(init_alpha)).log())
        self.log_beta = nn.Parameter(torch.tensor(float(init_beta)).log())
        # Light per-channel projection before correlation (helps with scale).
        self.proj = nn.Linear(embed_dim, embed_dim)

    def _radial_mask(self, h: int, w: int, device, dtype) -> torch.Tensor:
        """Build a (H, W) Beta-shaped radial mask in [0, 1]."""
        fy = torch.fft.fftfreq(h, d=1.0, device=device).to(dtype)
        fx = torch.fft.fftfreq(w, d=1.0, device=device).to(dtype)
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        r = (yy.pow(2) + xx.pow(2)).sqrt()
        # Normalise r into (0, 1) so Beta(α, β) is well-defined.
        r_norm = (r / (r.max() + 1e-8)).clamp(1e-4, 1 - 1e-4)
        a = self.log_alpha.exp()
        b = self.log_beta.exp()
        mask = r_norm.pow(a - 1) * (1 - r_norm).pow(b - 1)
        # Normalise the mask peak to 1 — its absolute scale is absorbed by
        # the linear projection.
        return mask / (mask.amax() + 1e-8)

    def forward(self, template: torch.Tensor, search: torch.Tensor,
                grid_size: int) -> torch.Tensor:
        """
        Args:
            template: (N, L_t, D) — patch tokens (no CLS).
            search:   (N, L_s, D) — patch tokens.
            grid_size: spatial grid side, so L_s = grid_size ** 2.

        Returns:
            (N, D, grid_size, grid_size) correlation map.
        """
        n, l_s, d = search.shape
        assert l_s == grid_size * grid_size, (
            f"search has {l_s} tokens but grid_size={grid_size}"
        )
        # We pool the template to a single descriptor (mean over its tokens),
        # then "spread" it across the search grid via FFT correlation.
        t_pooled = self.proj(template.mean(dim=1))                 # (N, D)
        t_grid = t_pooled.view(n, d, 1, 1).expand(
            n, d, grid_size, grid_size).contiguous()
        s_grid = self.proj(search).transpose(1, 2).reshape(
            n, d, grid_size, grid_size)

        f_t = torch.fft.fft2(t_grid)
        f_s = torch.fft.fft2(s_grid)
        mask = self._radial_mask(grid_size, grid_size,
                                 device=t_grid.device,
                                 dtype=t_grid.real.dtype)
        corr_freq = mask * (f_t * f_s.conj())
        corr = torch.fft.ifft2(corr_freq).real
        return corr
