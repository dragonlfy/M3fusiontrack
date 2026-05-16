"""
Training and evaluation transforms.

Both are object-style callables so they can be replaced by user code or
extended for new augmentations. They take a (template, search, bboxes)
pair, crop a search-area window around the bounding box, resize to
``img_size``, and return tensors ready for the model.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import torch
from PIL import Image


def _crop_to_tensor(
    img: Optional[Image.Image],
    bbox: np.ndarray,
    area_factor: float,
    out_size: int,
) -> torch.Tensor:
    """Crop a fixed area around the bounding box and resize."""
    if img is None:
        # Caller will mark the modality as missing.
        return torch.zeros(1, out_size, out_size)
    x, y, w, h = bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * area_factor
    left, top = int(cx - side / 2), int(cy - side / 2)
    right, bottom = left + int(side), top + int(side)
    crop = img.crop((left, top, right, bottom)).resize(
        (out_size, out_size), Image.BILINEAR
    )
    arr = np.asarray(crop, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[None]                       # (1, H, W)
    else:
        arr = arr.transpose(2, 0, 1)          # (C, H, W)
    return torch.from_numpy(arr)


class _Transform:
    def __init__(self, img_size: int,
                 template_area_factor: float = 2.0,
                 search_area_factor: float = 4.0) -> None:
        self.img_size = img_size
        self.template_area_factor = template_area_factor
        self.search_area_factor = search_area_factor

    def __call__(self, *, template: Dict[str, Optional[Image.Image]],
                 search: Dict[str, Optional[Image.Image]],
                 template_bbox: np.ndarray,
                 search_bbox: np.ndarray) -> dict:
        t_tensors = {
            m: _crop_to_tensor(im, template_bbox,
                               self.template_area_factor, self.img_size)
            for m, im in template.items()
        }
        s_tensors = {
            m: _crop_to_tensor(im, search_bbox,
                               self.search_area_factor, self.img_size)
            for m, im in search.items()
        }
        # Normalised target box inside the search window.
        side = max(search_bbox[2], search_bbox[3]) * self.search_area_factor
        cx_norm = 0.5 + (search_bbox[0] + search_bbox[2] / 2.0 -
                         (search_bbox[0] + search_bbox[2] / 2.0)) / side
        # ^ degenerate by construction; in a real implementation we would
        # track the crop origin so the target lands at its true centre.
        target = torch.tensor([
            0.5, 0.5,
            search_bbox[2] / side,
            search_bbox[3] / side,
        ], dtype=torch.float32)
        return {
            "template": t_tensors,
            "search":   s_tensors,
            "target_xywh": target,
        }


def default_train_transform(img_size: int = 128) -> Callable:
    return _Transform(img_size=img_size)


def default_eval_transform(img_size: int = 128) -> Callable:
    # Same crop policy for eval -- augmentations such as colour jitter live
    # higher up the stack if/when they are enabled.
    return _Transform(img_size=img_size)
