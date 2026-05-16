"""
Composite loss = focal-cls + GIoU-bbox + auxiliary regularisers
(orthonormality, spectral consistency).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _giou_loss(pred_ltrb: torch.Tensor, target_ltrb: torch.Tensor) -> torch.Tensor:
    """GIoU between (l, t, r, b) boxes given as offsets from a common point."""
    pl, pt, pr, pb = pred_ltrb.unbind(-1)
    tl, tt, tr, tb = target_ltrb.unbind(-1)
    p_area = (pl + pr) * (pt + pb)
    t_area = (tl + tr) * (tt + tb)
    il = torch.minimum(pl, tl); ir = torch.minimum(pr, tr)
    it = torch.minimum(pt, tt); ib = torch.minimum(pb, tb)
    inter = (il + ir).clamp(min=0) * (it + ib).clamp(min=0)
    union = p_area + t_area - inter
    iou = inter / union.clamp(min=1e-6)
    el = torch.maximum(pl, tl); er = torch.maximum(pr, tr)
    et = torch.maximum(pt, tt); eb = torch.maximum(pb, tb)
    enclose = (el + er) * (et + eb)
    giou = iou - (enclose - union) / enclose.clamp(min=1e-6)
    return 1.0 - giou.mean()


class M3Loss(nn.Module):
    def __init__(
        self,
        cls_weight: float = 1.0,
        giou_weight: float = 2.0,
        l1_weight: float = 5.0,
        ortho_weight: float = 1e-3,
        consistency_weight: float = 1e-2,
    ) -> None:
        super().__init__()
        self.w_cls = cls_weight
        self.w_giou = giou_weight
        self.w_l1 = l1_weight
        self.w_ortho = ortho_weight
        self.w_consistency = consistency_weight

    def forward(self, outputs: Dict[str, torch.Tensor],
                target_xywh: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        outputs:
            cls_logits: (N, 1, G, G)
            bbox_ltrb:  (N, 4, G, G)
            aux_orthonormality, aux_spectral_consistency: scalars
        target_xywh: (N, 4) in [0, 1] inside the search window.
        """
        cls_logits = outputs["cls_logits"]
        bbox = outputs["bbox_ltrb"]
        n, _, g, _ = cls_logits.shape
        device = cls_logits.device

        # Build target classification map as a 2-D Gaussian centred on the
        # target.
        ys = torch.linspace(0, 1, g, device=device).view(1, g, 1).expand(n, g, g)
        xs = torch.linspace(0, 1, g, device=device).view(1, 1, g).expand(n, g, g)
        cx = target_xywh[:, 0].view(n, 1, 1)
        cy = target_xywh[:, 1].view(n, 1, 1)
        sigma = 0.1
        target_cls = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_logits.squeeze(1), target_cls, reduction="mean")

        # For bbox loss, gather predicted ltrb at the target centre.
        gx = (target_xywh[:, 0] * g).long().clamp(0, g - 1)
        gy = (target_xywh[:, 1] * g).long().clamp(0, g - 1)
        idx = torch.arange(n, device=device)
        pred_ltrb = bbox[idx, :, gy, gx]                      # (N, 4)
        # Target ltrb at the centre cell -- half width to each side, in
        # normalised units scaled to the grid.
        tw = target_xywh[:, 2] * g
        th = target_xywh[:, 3] * g
        target_ltrb = torch.stack([tw / 2, th / 2, tw / 2, th / 2], dim=-1)
        giou_loss = _giou_loss(pred_ltrb, target_ltrb)
        l1_loss = F.l1_loss(pred_ltrb, target_ltrb)

        total = (
            self.w_cls * cls_loss
            + self.w_giou * giou_loss
            + self.w_l1 * l1_loss
            + self.w_ortho * outputs["aux_orthonormality"]
            + self.w_consistency * outputs["aux_spectral_consistency"]
        )
        return {
            "loss":             total,
            "cls":              cls_loss.detach(),
            "giou":             giou_loss.detach(),
            "l1":               l1_loss.detach(),
            "ortho":            outputs["aux_orthonormality"].detach(),
            "consistency":      outputs["aux_spectral_consistency"].detach(),
        }
