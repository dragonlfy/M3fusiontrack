"""
Multi-modal tracking dataset loader.

The repo is set up so that any of the public benchmarks discussed in the
paper -- LasHeR (RGB+T), VisEvent (RGB+E), EventVOT (RGB+E), DepthTrack
(RGB+D) -- can be loaded through a single class by pointing it at the
right root directory and specifying which modalities are available.

For modalities that are missing in a given benchmark, the dataset returns
``None`` for that key; the model's ``modality_mask`` argument then
handles the absence transparently.

The expected directory layout is::

    <root>/
        sequences/
            <seq_name>/
                rgb/       000001.jpg, 000002.jpg, ...
                tir/       (optional)
                event/     (optional)
                depth/     (optional)
                groundtruth.txt    # x, y, w, h  per frame
        splits/
            train.txt          # one sequence name per line
            test.txt

Switch dataset just by changing ``modalities`` to match what is on disk
for that benchmark.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import default_train_transform


def _read_groundtruth(path: Path) -> np.ndarray:
    """Parse a (N, 4) groundtruth file in `x,y,w,h` format. Comma or
    whitespace separated."""
    raw = path.read_text().strip().splitlines()
    rows = []
    for line in raw:
        line = line.replace(",", " ").split()
        rows.append([float(v) for v in line[:4]])
    return np.asarray(rows, dtype=np.float32)


class MultiModalTrackingDataset(Dataset):
    """One sample = (template, search) pair from the same sequence."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        modalities: Sequence[str] = ("rgb", "tir"),
        img_size: int = 128,
        max_gap: int = 100,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.modalities = list(modalities)
        self.img_size = img_size
        self.max_gap = max_gap
        self.transform = transform or default_train_transform(img_size)

        split_file = self.root / "splits" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(
                f"split file {split_file} not found — expected one sequence "
                "name per line"
            )
        self.sequences: List[str] = [
            ln.strip() for ln in split_file.read_text().splitlines() if ln.strip()
        ]

        # Cache (seq_name, num_frames) for sampling.
        self._index: List[Tuple[str, int]] = []
        for seq in self.sequences:
            primary = self.root / "sequences" / seq / self.modalities[0]
            if not primary.exists():
                continue
            n = len(sorted(primary.glob("*.jpg")) + sorted(primary.glob("*.png")))
            if n >= 2:
                self._index.append((seq, n))
        if not self._index:
            raise RuntimeError(
                f"No usable sequences found under {self.root}. Check that the "
                f"primary modality '{self.modalities[0]}' exists for at least "
                "one sequence in the split."
            )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._index)

    def _load_frame(self, seq: str, idx: int) -> Dict[str, Optional[Image.Image]]:
        frame: Dict[str, Optional[Image.Image]] = {}
        for m in self.modalities:
            d = self.root / "sequences" / seq / m
            if not d.exists():
                frame[m] = None
                continue
            # Try both jpg and png; idx is 1-based on disk by convention.
            candidates = (
                d / f"{idx + 1:06d}.jpg",
                d / f"{idx + 1:06d}.png",
                d / f"{idx + 1:08d}.jpg",
            )
            for p in candidates:
                if p.exists():
                    frame[m] = Image.open(p).convert(
                        "L" if m in ("tir", "depth") else "RGB"
                    )
                    break
            else:
                frame[m] = None
        return frame

    def __getitem__(self, item: int) -> dict:
        seq, n = self._index[item]
        rng = np.random.default_rng()
        i = int(rng.integers(0, n - 1))
        j = int(rng.integers(max(0, i - self.max_gap),
                             min(n, i + self.max_gap + 1)))
        if i == j:
            j = min(n - 1, i + 1)

        template = self._load_frame(seq, i)
        search = self._load_frame(seq, j)
        gt = _read_groundtruth(
            self.root / "sequences" / seq / "groundtruth.txt"
        )

        sample = self.transform(template=template, search=search,
                                template_bbox=gt[i], search_bbox=gt[j])
        sample["sequence"] = seq
        return sample


def build_dataset(cfg: dict) -> MultiModalTrackingDataset:
    """Construct a dataset from a config dict."""
    return MultiModalTrackingDataset(
        root=cfg["root"],
        split=cfg.get("split", "train"),
        modalities=cfg.get("modalities", ("rgb", "tir")),
        img_size=int(cfg.get("img_size", 128)),
        max_gap=int(cfg.get("max_gap", 100)),
    )
