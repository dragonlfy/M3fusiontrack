#!/usr/bin/env python
"""Evaluate a trained M3FusionTrack checkpoint on a test split.

For each sequence we initialise on frame 0 with the groundtruth bounding
box, then run the tracker frame-by-frame and report success / precision.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from m3fusiontrack import M3Tracker
from m3fusiontrack.data.dataset import _read_groundtruth
from m3fusiontrack.models import build_m3fusiontrack
from m3fusiontrack.utils import precision_score, success_score


def _load_seq(root: Path, seq: str, modalities: list, n: int) -> list:
    frames = []
    for i in range(n):
        f = {}
        for m in modalities:
            cands = [
                root / "sequences" / seq / m / f"{i + 1:06d}.jpg",
                root / "sequences" / seq / m / f"{i + 1:06d}.png",
            ]
            f[m] = next((Image.open(p).convert(
                "L" if m in ("tir", "depth") else "RGB")
                for p in cands if p.exists()), None)
        frames.append(f)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--split", default="test", type=str)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    test_cfg = cfg["data"]["test"]
    root = Path(test_cfg["root"])
    modalities = list(test_cfg["modalities"])

    model = build_m3fusiontrack(cfg["model"])
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"], strict=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tracker = M3Tracker(model, modalities=modalities, device=device)

    sequences = (root / "splits" / f"{args.split}.txt").read_text().splitlines()
    sequences = [s.strip() for s in sequences if s.strip()]

    all_success, all_precision = [], []
    for seq in sequences:
        gt = _read_groundtruth(root / "sequences" / seq / "groundtruth.txt")
        n = gt.shape[0]
        frames = _load_seq(root, seq, modalities, n)
        tracker.initialise(frames[0], gt[0])
        preds = [gt[0].copy()]
        for k in range(1, n):
            preds.append(tracker.track(frames[k]))
        preds_arr = np.stack(preds, axis=0)
        s = success_score(preds_arr, gt)
        p = precision_score(preds_arr, gt)
        all_success.append(s)
        all_precision.append(p)
        print(f"{seq:>40s}  success={s:.4f}  precision={p:.4f}")

    print("─" * 60)
    print(f"average success   : {np.mean(all_success):.4f}")
    print(f"average precision : {np.mean(all_precision):.4f}")


if __name__ == "__main__":
    main()
