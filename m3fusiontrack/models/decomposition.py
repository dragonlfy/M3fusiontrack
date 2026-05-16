"""
Learnable Spectral Decomposition (Section 3.2 of the paper).

Each modality input is decomposed into B frequency bands using a bank of
learnable 1-D filters that are applied separably along H and W. We softly
constrain the filter bank toward orthonormality (Daubechies-like) by
adding a penalty `orthonormality_penalty()` which the trainer adds to
the total loss.

This is a teaching-purposes simplification of the full DWT-based
formulation in the paper; it preserves the key idea (learnable, regularised
multi-band filters) without depending on `pytorch_wavelets`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableSpectralDecomposition(nn.Module):
    """Decompose an image-like tensor into ``num_bands`` frequency bands.

    Args:
        in_channels: number of input channels (e.g., 3 for RGB, 1 for TIR).
        num_bands:   number of frequency bands B (default 3: low/mid/high).
        kernel_size: 1-D filter length used along H and W (must be odd).

    Input shape:  ``(B_, C, H, W)``.
    Output shape: ``(B_, num_bands, C, H, W)``.
    """

    def __init__(
        self,
        in_channels: int,
        num_bands: int = 3,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.in_channels = in_channels
        self.num_bands = num_bands
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Learnable 1-D filter bank: (num_bands, kernel_size).
        # Initialised toward a low-pass / band-pass / high-pass split so the
        # network does not have to discover this structure from scratch.
        filt = torch.zeros(num_bands, kernel_size)
        center = kernel_size // 2
        for b in range(num_bands):
            sigma = (b + 1) * 0.7  # increasing sigma -> lower frequency
            xs = torch.arange(kernel_size, dtype=torch.float32) - center
            gauss = torch.exp(-xs.pow(2) / (2 * sigma ** 2))
            gauss = gauss / gauss.sum()
            if b == 0:
                filt[b] = gauss                     # low-pass
            elif b == num_bands - 1:
                hp = -gauss.clone()
                hp[center] = hp[center] + 1.0       # high-pass = δ − low-pass
                filt[b] = hp
            else:
                # band-pass: difference of two Gaussians at adjacent scales
                sigma_lo = b * 0.7
                xs2 = torch.arange(kernel_size, dtype=torch.float32) - center
                g_lo = torch.exp(-xs2.pow(2) / (2 * sigma_lo ** 2))
                g_lo = g_lo / g_lo.sum()
                filt[b] = gauss - g_lo
        self.filter_1d = nn.Parameter(filt)         # (B, K)

        # Per-band channel-wise scale / shift after decomposition (LayerNorm-like).
        self.gain = nn.Parameter(torch.ones(num_bands, in_channels))
        self.bias = nn.Parameter(torch.zeros(num_bands, in_channels))

    def orthonormality_penalty(self) -> torch.Tensor:
        """Soft Daubechies constraint:  ‖F F^T − I‖_F^2 .

        Encourages the B filters to span (approximately) orthonormal
        directions so different bands do not collapse to the same response.
        """
        f = self.filter_1d                                  # (B, K)
        gram = f @ f.t()                                    # (B, B)
        identity = torch.eye(self.num_bands, device=f.device, dtype=f.dtype)
        return (gram - identity).pow(2).sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:     # noqa: D401
        """Apply separable 2-D filtering for every band."""
        if x.ndim != 4:
            raise ValueError(f"expected (N, C, H, W), got {tuple(x.shape)}")
        n, c, h, w = x.shape
        # Build (num_bands * C, 1, K, 1) and (num_bands * C, 1, 1, K) kernels
        # for depthwise separable convolution per band.
        f_v = self.filter_1d.view(self.num_bands, 1, self.kernel_size, 1)
        f_h = self.filter_1d.view(self.num_bands, 1, 1, self.kernel_size)
        f_v = f_v.expand(self.num_bands, c, self.kernel_size, 1).reshape(
            self.num_bands * c, 1, self.kernel_size, 1
        )
        f_h = f_h.expand(self.num_bands, c, 1, self.kernel_size).reshape(
            self.num_bands * c, 1, 1, self.kernel_size
        )
        # Tile input so groups = num_bands * C
        x_tiled = x.repeat(1, self.num_bands, 1, 1)         # (N, B*C, H, W)
        y = F.conv2d(x_tiled, f_v, padding=(self.padding, 0),
                     groups=self.num_bands * c)
        y = F.conv2d(y, f_h, padding=(0, self.padding),
                     groups=self.num_bands * c)
        y = y.view(n, self.num_bands, c, h, w)              # (N, B, C, H, W)
        # Apply per-band channel-wise affine.
        y = y * self.gain.view(1, self.num_bands, c, 1, 1)
        y = y + self.bias.view(1, self.num_bands, c, 1, 1)
        return y
