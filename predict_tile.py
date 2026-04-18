"""Sliding-window inference across a full tile.

    python predict_tile.py --model models/rareplanes_to_xview.pt --tile data/.../1038.tif
    python predict_tile.py --model models/xview_to_rareplanes.pt --tile data/.../some.png --stride 96 --out heatmap.png

The model sees 224x224 chips, so for a whole tile we slide the window,
classify each crop, and emit:
  - a summary: how many windows predicted aircraft
  - a heatmap PNG: the tile with a red overlay of aircraft-probability

Stride defaults to CHIP_SIZE // 2 (50% overlap).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from src.data import CHIP_SIZE, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.model import build_model  # noqa: E402
from predict import load_image  # noqa: E402


def preprocess_batch(chips: list[np.ndarray]) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    t = torch.from_numpy(np.stack(chips)).float().permute(0, 3, 1, 2) / 255.0
    return (t - mean) / std


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, type=Path, help="path to .pt weights")
    p.add_argument("--tile", required=True, type=Path, help="path to tile image (TIF/PNG)")
    p.add_argument("--stride", type=int, default=CHIP_SIZE // 2, help=f"window stride (default {CHIP_SIZE // 2})")
    p.add_argument("--out", type=Path, default=None, help="output heatmap PNG (default: <tile>.heatmap.png)")
    p.add_argument("--threshold", type=float, default=0.5, help="aircraft probability threshold for counting positives")
    p.add_argument("--device", default=None)
    p.add_argument("--batch", type=int, default=64)
    args = p.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    out = args.out or args.tile.with_suffix(".heatmap.png")

    print(f"loading tile: {args.tile}")
    tile = load_image(args.tile)
    h, w = tile.shape[:2]
    if h < CHIP_SIZE or w < CHIP_SIZE:
        raise ValueError(f"tile is {w}x{h}; need at least {CHIP_SIZE}x{CHIP_SIZE}")

    xs = list(range(0, w - CHIP_SIZE + 1, args.stride))
    ys = list(range(0, h - CHIP_SIZE + 1, args.stride))
    if xs[-1] != w - CHIP_SIZE:
        xs.append(w - CHIP_SIZE)
    if ys[-1] != h - CHIP_SIZE:
        ys.append(h - CHIP_SIZE)
    n_windows = len(xs) * len(ys)
    print(f"tile: {w}x{h} -> sliding {n_windows} windows ({len(xs)} x {len(ys)}, stride={args.stride})")

    print(f"loading model: {args.model}")
    model = build_model(num_classes=2, pretrained=False)
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    chips: list[np.ndarray] = []
    positions: list[tuple[int, int]] = []
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            chip = tile[y:y + CHIP_SIZE, x:x + CHIP_SIZE]
            chips.append(chip)
            positions.append((yi, xi))

    t0 = time.time()
    probs_flat = np.zeros(len(chips), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(chips), args.batch):
            batch = preprocess_batch(chips[i:i + args.batch]).to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs_flat[i:i + len(probs)] = probs
    elapsed = time.time() - t0

    grid = np.zeros((len(ys), len(xs)), dtype=np.float32)
    for (yi, xi), p in zip(positions, probs_flat):
        grid[yi, xi] = p

    n_pos = int((probs_flat > args.threshold).sum())
    max_p = float(probs_flat.max())
    mean_p = float(probs_flat.mean())

    print(f"inference: {elapsed:.2f}s on {device}")
    print(f"positives (p>{args.threshold}): {n_pos}/{n_windows} windows")
    print(f"max aircraft probability:  {max_p:.3f}")
    print(f"mean aircraft probability: {mean_p:.3f}")

    heatmap = np.zeros((h, w), dtype=np.float32)
    for (yi, xi), p in zip(positions, probs_flat):
        y, x = ys[yi], xs[xi]
        heatmap[y:y + CHIP_SIZE, x:x + CHIP_SIZE] = np.maximum(
            heatmap[y:y + CHIP_SIZE, x:x + CHIP_SIZE], p
        )

    fig, ax = plt.subplots(figsize=(max(8, w / 200), max(8, h / 200)))
    ax.imshow(tile)
    ax.imshow(heatmap, cmap="Reds", alpha=0.45, vmin=0, vmax=1)
    ax.set_title(f"{args.tile.name} — {n_pos}/{n_windows} windows positive (p>{args.threshold})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved heatmap -> {out}")


if __name__ == "__main__":
    main()
