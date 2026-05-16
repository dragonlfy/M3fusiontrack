"""
The top-level ``M3FusionTrack`` model.

Pipeline (matching Figure 1 of the paper):

    template / search images for K modalities
            │
            ▼
    LearnableSpectralDecomposition  →  (K modalities) × (B bands) feature maps
            │
            ▼
    SharedFoundationBackbone        →  (M*B) token sequences
            │
            ▼
    CrossModalFrequencyAttention    →  refined tokens with cross-cell mixing
            │
            ▼
    UncertaintyAwareGating          →  fused tokens + per-cell variance
            │
            ▼
    FrequencyAwareCorrelation       →  template-search correlation
            │
            ▼
    TrackingHead                    →  cls map + bbox map

Use ``build_m3fusiontrack(cfg)`` to construct from a config dict.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .backbone import SharedFoundationBackbone, build_backbone
from .correlation import FrequencyAwareCorrelation
from .decomposition import LearnableSpectralDecomposition
from .fusion import CrossModalFrequencyAttention, UncertaintyAwareGating
from .head import TrackingHead


# Default channel counts for the four modalities.
DEFAULT_MODALITIES: Dict[str, int] = {
    "rgb": 3,
    "tir": 1,
    "event": 3,        # event-frame representation (e.g., voxel grid → 3ch)
    "depth": 1,
}


class M3FusionTrack(nn.Module):
    """Multi-modal multi-frequency foundation tracker."""

    def __init__(
        self,
        modalities: Sequence[str] = ("rgb", "tir", "event", "depth"),
        num_bands: int = 3,
        img_size: int = 128,
        backbone_name: str = "simple",
        lora_rank: Optional[int] = 8,
        cmfa_layers: int = 2,
        cmfa_heads: int = 8,
        gate_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.modality_names = list(modalities)
        self.num_modalities = len(self.modality_names)
        self.num_bands = num_bands
        self.img_size = img_size

        # 1. Per-modality learnable spectral decomposition.
        self.decomposers = nn.ModuleDict({
            m: LearnableSpectralDecomposition(
                in_channels=DEFAULT_MODALITIES[m], num_bands=num_bands)
            for m in self.modality_names
        })

        # 2. Shared backbone (DINOv2 / SimpleViT).
        in_channels_first = DEFAULT_MODALITIES[self.modality_names[0]]
        backbone = build_backbone(
            name=backbone_name, img_size=img_size,
            in_channels=in_channels_first, lora_rank=lora_rank,
        )
        # SimpleViT expects a fixed in-channel count. To keep the backbone
        # truly shared, every (modality, band) feature is mapped to a common
        # channel dimension before patch-embedding.
        self.modality_stems = nn.ModuleDict({
            m: nn.Conv2d(DEFAULT_MODALITIES[m], in_channels_first,
                         kernel_size=1)
            for m in self.modality_names
        })
        self.backbone = SharedFoundationBackbone(
            backbone, num_modalities=self.num_modalities, num_bands=num_bands)

        # 3. CMFA + 4. UAG.
        d = self.backbone.embed_dim
        self.cmfa = CrossModalFrequencyAttention(
            embed_dim=d, num_modalities=self.num_modalities,
            num_bands=num_bands, num_heads=cmfa_heads,
            num_layers=cmfa_layers,
        )
        self.uag = UncertaintyAwareGating(
            embed_dim=d, num_modalities=self.num_modalities,
            num_bands=num_bands, temperature=gate_temperature,
        )

        # 5. Frequency-aware correlation + 6. tracking head.
        self.correlation = FrequencyAwareCorrelation(embed_dim=d)
        self.head = TrackingHead(in_channels=d)

        # Patch grid size on backbone output, used by correlation.
        self.grid_size = img_size // 16 if backbone_name == "simple" else 14

    # ------------------------------------------------------------------
    # Building blocks of forward
    # ------------------------------------------------------------------
    def _encode(self, inputs: Dict[str, torch.Tensor],
                modality_mask: Optional[Sequence[bool]] = None
                ) -> torch.Tensor:
        """Decompose → backbone for every (modality, band) cell.

        Returns:
            tokens: (N, M*B, L+1, D)
        """
        tokens_list: List[torch.Tensor] = []
        for k, m in enumerate(self.modality_names):
            x = inputs.get(m, None)
            if x is None or (modality_mask is not None and not modality_mask[k]):
                # Modality is missing or dropped — we still need (B) outputs
                # so the grid is full. We fill with a learned zero-prompt
                # response: feed an all-zero image through the stem so the
                # backbone produces a valid (but uninformative) token set.
                ref = next(v for v in inputs.values() if v is not None)
                x = torch.zeros(
                    ref.size(0), DEFAULT_MODALITIES[m],
                    self.img_size, self.img_size,
                    device=ref.device, dtype=ref.dtype,
                )
            decomposed = self.decomposers[m](x)            # (N, B, C, H, W)
            for b in range(self.num_bands):
                band_feat = decomposed[:, b]               # (N, C, H, W)
                band_feat = self.modality_stems[m](band_feat)
                tokens = self.backbone(band_feat, k, b)    # (N, L+1, D)
                tokens_list.append(tokens)
        return torch.stack(tokens_list, dim=1)             # (N, M*B, L+1, D)

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------
    def forward(
        self,
        template: Dict[str, torch.Tensor],
        search: Dict[str, torch.Tensor],
        modality_mask: Optional[Sequence[bool]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run one forward pass.

        Args:
            template: dict ``{modality: (N, C, H, W)}``
            search:   dict ``{modality: (N, C, H, W)}``
            modality_mask: optional list of bool, length = num_modalities.
                Entries set to False simulate a missing modality.

        Returns:
            dict with cls_logits, bbox_ltrb, gate, log_var and aux losses
            (orthonormality, spectral consistency).
        """
        # Encode template and search independently — the backbone is shared
        # and so is the prompt table.
        t_tokens = self._encode(template, modality_mask)
        s_tokens = self._encode(search, modality_mask)

        # CMFA mixes tokens across (modality, band) cells.
        t_refined = self.cmfa(t_tokens)
        s_refined = self.cmfa(s_tokens)

        # UAG fuses cells into a single representation per stream.
        t_fused, t_gate, t_logvar = self.uag(t_refined)    # (N, L+1, D)
        s_fused, s_gate, s_logvar = self.uag(s_refined)

        # Correlation: drop CLS, use patch tokens only.
        corr_map = self.correlation(
            t_fused[:, 1:], s_fused[:, 1:], grid_size=self.grid_size,
        )                                                  # (N, D, G, G)

        out = self.head(corr_map)
        # Auxiliary terms used by the trainer to build the total loss.
        ortho = sum(self.decomposers[m].orthonormality_penalty()
                    for m in self.modality_names)
        sc_loss = UncertaintyAwareGating.spectral_consistency_loss(s_gate)
        out.update({
            "gate_template": t_gate,
            "gate_search":   s_gate,
            "log_var":       s_logvar,
            "aux_orthonormality":      ortho,
            "aux_spectral_consistency": sc_loss,
        })
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_m3fusiontrack(cfg: dict) -> M3FusionTrack:
    """Build a model from a config dict (e.g., loaded from YAML).

    Recognised keys (with defaults shown in parens):
        modalities       (rgb, tir, event, depth)
        num_bands        (3)
        img_size         (128)
        backbone_name    ("simple")  -- "simple" or e.g. "dinov2_vitb14"
        lora_rank        (8)
        cmfa_layers      (2)
        cmfa_heads       (8)
        gate_temperature (1.0)
    """
    return M3FusionTrack(
        modalities=tuple(cfg.get("modalities", ("rgb", "tir", "event", "depth"))),
        num_bands=int(cfg.get("num_bands", 3)),
        img_size=int(cfg.get("img_size", 128)),
        backbone_name=str(cfg.get("backbone_name", "simple")),
        lora_rank=cfg.get("lora_rank", 8),
        cmfa_layers=int(cfg.get("cmfa_layers", 2)),
        cmfa_heads=int(cfg.get("cmfa_heads", 8)),
        gate_temperature=float(cfg.get("gate_temperature", 1.0)),
    )
