#!/usr/bin/env python
"""Train M3FusionTrack from a YAML config file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from m3fusiontrack.data import build_dataset
from m3fusiontrack.models import build_m3fusiontrack
from m3fusiontrack.trainer import Trainer
from m3fusiontrack.utils import M3Loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str,
                        help="path to YAML config file")
    parser.add_argument("--output", default="checkpoints", type=str,
                        help="where to save checkpoints")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Build dataset & loader
    train_ds = build_dataset(cfg["data"]["train"])
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"].get("batch_size", 8),
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 4),
        pin_memory=True,
    )

    # Build model
    model = build_m3fusiontrack(cfg["model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"trainable params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Loss + optimiser + (optional) scheduler
    loss_fn = M3Loss(**cfg["loss"])
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 1e-4),
    )
    scheduler = None
    if "cosine_t_max" in cfg["train"]:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=cfg["train"]["cosine_t_max"]
        )

    trainer = Trainer(
        model=model,
        loader=train_loader,
        optimizer=optim,
        loss_fn=loss_fn,
        device=device,
        epochs=cfg["train"]["epochs"],
        modality_dropout_max=cfg["train"].get("modality_dropout_max", 0.3),
        log_every=cfg["train"].get("log_every", 50),
        scheduler=scheduler,
    )
    trainer.fit()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    out = Path(args.output) / "m3fusiontrack_final.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg}, out)
    print(f"saved checkpoint to {out}")


if __name__ == "__main__":
    main()
