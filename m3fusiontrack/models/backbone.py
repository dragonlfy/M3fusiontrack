"""
Shared Foundation Backbone (Section 3.3 of the paper).

In the full model we wrap a frozen DINOv2-B ViT and inject lightweight LoRA
adapters into every Q/V projection. For this simplified reference repo we
provide:

    * A small Vision Transformer (`SimpleViT`) that has the same input
      contract as DINOv2 (patch tokens + CLS).
    * A drop-in DINOv2 wrapper (`DinoV2Backbone`) that lazily loads weights
      via `torch.hub` when the user has internet access.

Both expose the same `forward(x) -> tokens` interface so the rest of the
model is backbone-agnostic.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# LoRA adapter
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Standard LoRA wrapping a frozen `nn.Linear`."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scaling = alpha / max(rank, 1)
        self.A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:                # noqa: D401
        out = self.base(x)
        delta = (x @ self.A.t()) @ self.B.t()
        return out + self.scaling * delta


# ---------------------------------------------------------------------------
# A tiny ViT — fast to run, useful as a default when DINOv2 weights are not
# available locally. Embed dim and depth are kept small so unit tests fit in
# a CPU-only CI runner.
# ---------------------------------------------------------------------------
class SimpleViT(nn.Module):
    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:               # noqa: D401
        n = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)            # (N, L, D)
        cls = self.cls_token.expand(n, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        return x                                                      # (N, L+1, D)


# ---------------------------------------------------------------------------
# Optional DINOv2 wrapper
# ---------------------------------------------------------------------------
class DinoV2Backbone(nn.Module):
    """Thin wrapper around torch.hub DINOv2 with optional LoRA adapters."""

    def __init__(self, name: str = "dinov2_vitb14", apply_lora: bool = True,
                 lora_rank: int = 8) -> None:
        super().__init__()
        try:
            self.model = torch.hub.load("facebookresearch/dinov2", name)
        except Exception as exc:  # pragma: no cover - depends on network
            raise RuntimeError(
                "DINOv2 weights could not be loaded via torch.hub; either "
                "preload them or fall back to SimpleViT."
            ) from exc
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.embed_dim = self.model.embed_dim
        if apply_lora:
            self._inject_lora(lora_rank)

    def _inject_lora(self, rank: int) -> None:
        for module in self.model.modules():
            if hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear):
                module.qkv = LoRALinear(module.qkv, rank=rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:               # noqa: D401
        # torch.hub DINOv2 returns the patch tokens + CLS through `forward_features`.
        feats = self.model.forward_features(x)
        tokens = torch.cat([feats["x_norm_clstoken"].unsqueeze(1),
                            feats["x_norm_patchtokens"]], dim=1)
        return tokens


# ---------------------------------------------------------------------------
# Factory — pick the backbone that the user configured
# ---------------------------------------------------------------------------
def build_backbone(
    name: str = "simple",
    img_size: int = 128,
    in_channels: int = 3,
    lora_rank: Optional[int] = 8,
) -> nn.Module:
    """Build either the lightweight ``SimpleViT`` or DINOv2."""
    if name == "simple":
        return SimpleViT(img_size=img_size, in_channels=in_channels)
    if name.startswith("dinov2"):
        return DinoV2Backbone(name=name, apply_lora=lora_rank is not None,
                              lora_rank=lora_rank or 0)
    raise ValueError(f"unknown backbone {name!r}")


# ---------------------------------------------------------------------------
# Shared backbone over (modality, band) — all 12 inputs go through the SAME
# network. We add a learnable (modality, band) prompt that is summed onto the
# CLS token, following Eq. (4) in the paper.
# ---------------------------------------------------------------------------
class SharedFoundationBackbone(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_modalities: int = 4,
        num_bands: int = 3,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_modalities = num_modalities
        self.num_bands = num_bands
        embed_dim = getattr(backbone, "embed_dim", 768)
        self.embed_dim = embed_dim
        # Learnable (modality, band) prompts, shape (M, B, D).
        self.prompts = nn.Parameter(
            torch.zeros(num_modalities, num_bands, embed_dim)
        )
        nn.init.trunc_normal_(self.prompts, std=0.02)

    def forward(self, x: torch.Tensor, modality_idx: int, band_idx: int) -> torch.Tensor:
        """Forward a single (modality, band) feature map.

        Args:
            x:            (N, C, H, W)
            modality_idx: int in [0, num_modalities)
            band_idx:     int in [0, num_bands)

        Returns:
            Token sequence with prompt added to CLS, shape (N, L+1, D).
        """
        tokens = self.backbone(x)                       # (N, L+1, D)
        prompt = self.prompts[modality_idx, band_idx]   # (D,)
        tokens = tokens.clone()
        tokens[:, 0] = tokens[:, 0] + prompt
        return tokens
