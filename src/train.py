"""One training function, parameterized by config. Used for both directions
of the bidirectional cross-dataset experiment.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import freeze_batchnorm


@dataclass
class TrainReport:
    epochs_run: int
    final_train_loss: float
    final_train_acc: float
    wall_seconds: float


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
    log_prefix: str = "",
) -> TrainReport:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    t0 = time.time()
    last_loss = 0.0
    last_acc = 0.0
    for epoch in range(epochs):
        model.train()
        freeze_batchnorm(model)  # keep BN stats fixed on frozen backbone
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"{log_prefix}epoch {epoch + 1}/{epochs}", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        scheduler.step()
        last_loss = running_loss / max(total, 1)
        last_acc = correct / max(total, 1)
        print(f"{log_prefix}epoch {epoch + 1}: loss={last_loss:.4f} acc={last_acc:.4f}")

    return TrainReport(
        epochs_run=epochs,
        final_train_loss=last_loss,
        final_train_acc=last_acc,
        wall_seconds=time.time() - t0,
    )
