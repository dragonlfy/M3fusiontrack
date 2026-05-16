"""
Minimal training loop.

The trainer implements the modality-dropout curriculum from Section 3.7
of the paper: with probability ``p(epoch)`` we randomly drop one modality
per sample. p ramps linearly from 0 to ``modality_dropout_max`` over the
first half of training, then stays constant.
"""

from __future__ import annotations

import random
from typing import Optional

import torch
from torch.utils.data import DataLoader

from .models.m3fusiontrack import M3FusionTrack
from .utils.losses import M3Loss


class Trainer:
    def __init__(
        self,
        model: M3FusionTrack,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: M3Loss,
        device: str = "cuda",
        epochs: int = 50,
        modality_dropout_max: float = 0.3,
        log_every: int = 50,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        self.model = model.to(device)
        self.loader = loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.epochs = epochs
        self.modality_dropout_max = modality_dropout_max
        self.log_every = log_every
        self.scheduler = scheduler

    def _dropout_prob(self, epoch: int) -> float:
        half = max(1, self.epochs // 2)
        return self.modality_dropout_max * min(1.0, epoch / half)

    def _sample_mask(self, p: float) -> list:
        m = self.model.num_modalities
        if random.random() > p:
            return [True] * m
        # Drop exactly one modality.
        drop = random.randrange(m)
        return [i != drop for i in range(m)]

    def fit(self) -> None:
        global_step = 0
        for epoch in range(self.epochs):
            p_drop = self._dropout_prob(epoch)
            self.model.train()
            for batch in self.loader:
                template = {k: v.to(self.device) for k, v in batch["template"].items()}
                search = {k: v.to(self.device) for k, v in batch["search"].items()}
                target = batch["target_xywh"].to(self.device)
                mask = self._sample_mask(p_drop)

                outputs = self.model(template, search, modality_mask=mask)
                losses = self.loss_fn(outputs, target)

                self.optimizer.zero_grad()
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                if global_step % self.log_every == 0:
                    msg = (f"[ep {epoch:3d} step {global_step:6d}] "
                           f"p_drop={p_drop:.2f} "
                           f"loss={losses['loss'].item():.4f} "
                           f"cls={losses['cls'].item():.4f} "
                           f"giou={losses['giou'].item():.4f} "
                           f"ortho={losses['ortho'].item():.4f} "
                           f"cons={losses['consistency'].item():.4f}")
                    print(msg, flush=True)
                global_step += 1
