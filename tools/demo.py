#!/usr/bin/env python
"""Run M3FusionTrack on a single sequence and draw the predicted boxes.

Usage:
    python tools/demo.py --checkpoint cps/best.pt --config configs/lasher.yaml \
        --sequence path/to/seq

The sequence directory should contain one sub-folder per modality and
``groundtruth.txt`` with the initial box on its first line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from m3fusiontrack import M3Tracker
from m3fusiontrack.data.dataset import _read_groundtruth
from m3fusiontrack.models import build_m3fusiontrack


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--sequence", required=True, type=str,
                        help="path to a single sequence directory")
    parser.add_argument("--output", default="demo_out", type=str)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    modalities = list(cfg["data"]["test"]["modalities"])
    primary = modalities[0]

    seq = Path(args.sequence)
    gt = _read_groundtruth(seq / "groundtruth.txt")
    frames_dir = seq / primary
    files = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))

    model = build_m3fusiontrack(cfg["model"])
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"], strict=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tracker = M3Tracker(model, modalities=modalities, device=device)

    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Initialise on frame 0
    first_frame = {m: Image.open(seq / m / files[0].name).convert(
        "L" if m in ("tir", "depth") else "RGB") if (seq / m / files[0].name).exists()
        else None for m in modalities}
    tracker.initialise(first_frame, gt[0])

    for i, fp in enumerate(files):
        frame = {m: Image.open(seq / m / fp.name).convert(
            "L" if m in ("tir", "depth") else "RGB") if (seq / m / fp.name).exists()
            else None for m in modalities}
        if i == 0:
            box = gt[0]
        else:
            box = tracker.track(frame)
        rgb = frame[primary].convert("RGB") if frame[primary] is not None \
            else Image.new("RGB", (300, 300), "black")
        draw = ImageDraw.Draw(rgb)
        x, y, w, h = box
        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
        if i < len(gt):
            gx, gy, gw, gh = gt[i]
            draw.rectangle([gx, gy, gx + gw, gy + gh], outline="green", width=2)
        rgb.save(Path(args.output) / fp.name)

    print(f"saved {len(files)} frames to {args.output}")


if __name__ == "__main__":
    main()
