"""
Tracking benchmark metrics:
* success rate (AUC of IoU > τ over τ ∈ [0, 1])
* precision (fraction of frames with centre error < 20 px)
"""

from __future__ import annotations

import numpy as np


def _iou(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    x1 = np.maximum(b1[:, 0], b2[:, 0])
    y1 = np.maximum(b1[:, 1], b2[:, 1])
    x2 = np.minimum(b1[:, 0] + b1[:, 2], b2[:, 0] + b2[:, 2])
    y2 = np.minimum(b1[:, 1] + b1[:, 3], b2[:, 1] + b2[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a1 = b1[:, 2] * b1[:, 3]
    a2 = b2[:, 2] * b2[:, 3]
    return inter / np.maximum(a1 + a2 - inter, 1e-6)


def success_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """AUC of success plot."""
    ious = _iou(pred, gt)
    thresholds = np.linspace(0, 1, 21)
    return float(np.mean([(ious > t).mean() for t in thresholds]))


def precision_score(pred: np.ndarray, gt: np.ndarray,
                    threshold: float = 20.0) -> float:
    """Fraction of frames whose centre error is below ``threshold`` pixels."""
    cx_p = pred[:, 0] + pred[:, 2] / 2
    cy_p = pred[:, 1] + pred[:, 3] / 2
    cx_g = gt[:, 0] + gt[:, 2] / 2
    cy_g = gt[:, 1] + gt[:, 3] / 2
    err = np.sqrt((cx_p - cx_g) ** 2 + (cy_p - cy_g) ** 2)
    return float((err < threshold).mean())
