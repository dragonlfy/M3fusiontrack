"""
Tracking head.

Consumes the correlation map and produces:
  * a classification (object presence) map ``(N, 1, H, W)``,
  * a bounding-box offset map ``(N, 4, H, W)`` in (l, t, r, b) format.

Loss-side weighting (GIoU + focal) lives in `utils/losses.py`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TrackingHead(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 256) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )
        self.cls = nn.Conv2d(hidden, 1, kernel_size=1)
        # Bounding-box branch outputs raw offsets; we map them to positive
        # numbers via a Softplus when computing predictions.
        self.bbox = nn.Conv2d(hidden, 4, kernel_size=1)
        self.softplus = nn.Softplus()

    def forward(self, corr_map: torch.Tensor) -> dict:
        feat = self.shared(corr_map)
        cls_logits = self.cls(feat)
        bbox = self.softplus(self.bbox(feat))
        return {"cls_logits": cls_logits, "bbox_ltrb": bbox}
