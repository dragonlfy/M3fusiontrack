"""
Inference-time wrapper around ``M3FusionTrack``.

Given an initial bounding box on frame 0, ``M3Tracker.track(frame)`` returns
the predicted bounding box on each subsequent frame.

The implementation is the minimal version: a Hann-window prior on the
classification map and a single-step box decoding. No update strategy
(template memory, online learner) is implemented here -- the paper's full
HiPTrack-style memory module is left out of this simplified reference.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..data.transforms import _crop_to_tensor
from ..models.m3fusiontrack import M3FusionTrack


class M3Tracker:
    def __init__(
        self,
        model: M3FusionTrack,
        modalities: Sequence[str] = ("rgb", "tir", "event", "depth"),
        device: str = "cuda",
        template_area_factor: float = 2.0,
        search_area_factor: float = 4.0,
        window_influence: float = 0.21,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.modalities = list(modalities)
        self.template_area_factor = template_area_factor
        self.search_area_factor = search_area_factor
        self.window_influence = window_influence
        self._template_tensors: Optional[Dict[str, torch.Tensor]] = None
        self._state: Optional[np.ndarray] = None    # last bbox (x, y, w, h)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def initialise(self, frame: Dict[str, Optional[Image.Image]],
                   bbox_xywh: Sequence[float]) -> None:
        """Build and cache the template from frame 0."""
        bbox = np.asarray(bbox_xywh, dtype=np.float32)
        tens = {
            m: _crop_to_tensor(frame.get(m), bbox,
                               self.template_area_factor,
                               self.model.img_size).unsqueeze(0).to(self.device)
            for m in self.modalities
        }
        self._template_tensors = tens
        self._state = bbox.copy()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def track(self, frame: Dict[str, Optional[Image.Image]]
              ) -> np.ndarray:
        """Predict the bounding box for the current frame."""
        if self._template_tensors is None or self._state is None:
            raise RuntimeError("call initialise() before track().")
        # 1) build the search-area crop around the previous prediction
        search_tens = {
            m: _crop_to_tensor(frame.get(m), self._state,
                               self.search_area_factor,
                               self.model.img_size).unsqueeze(0).to(self.device)
            for m in self.modalities
        }
        out = self.model(self._template_tensors, search_tens)
        cls_map = out["cls_logits"][0, 0]                 # (G, G)
        bbox_map = out["bbox_ltrb"][0]                    # (4, G, G)

        # 2) apply a cosine window to discourage drift to image corners
        g = cls_map.shape[-1]
        win = self._hann_window(g, cls_map.device)
        scored = cls_map.sigmoid() * (1 - self.window_influence) + \
                 win * self.window_influence
        cy, cx = torch.unravel_index(scored.argmax(), scored.shape)

        # 3) decode the bounding box
        ltrb = bbox_map[:, cy, cx]                        # (4,)
        side = max(self._state[2], self._state[3]) * self.search_area_factor
        # cell centre in normalised search-window coords
        ncx = (cx.item() + 0.5) / g
        ncy = (cy.item() + 0.5) / g
        l, t, r, b = (ltrb / g).tolist()
        x1 = (ncx - l) * side + self._state[0] + self._state[2] / 2 - side / 2
        y1 = (ncy - t) * side + self._state[1] + self._state[3] / 2 - side / 2
        x2 = (ncx + r) * side + self._state[0] + self._state[2] / 2 - side / 2
        y2 = (ncy + b) * side + self._state[1] + self._state[3] / 2 - side / 2
        w = max(2.0, x2 - x1)
        h = max(2.0, y2 - y1)
        self._state = np.array([x1, y1, w, h], dtype=np.float32)
        return self._state.copy()

    # ------------------------------------------------------------------
    @staticmethod
    def _hann_window(g: int, device) -> torch.Tensor:
        w = torch.hann_window(g, periodic=False, device=device)
        return w[:, None] * w[None, :]
