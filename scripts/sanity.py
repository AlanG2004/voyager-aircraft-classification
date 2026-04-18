"""Two fast pre-flight checks before the full run:

  1. Save ~24 sample chips (positives + negatives from each dataset) so we can
     visually confirm the chip-extraction logic is correct.
  2. Overfit a model on 100 chips for 15 epochs — if training accuracy doesn't
     reach ~0.95+ the pipeline has a real bug.

Run with: python sanity.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (  # noqa: E402
    Chip,
    ChipDataset,
    extract_rareplanes_chips,
    extract_xview_chips,
    load_xview_labels,
    rareplanes_tile_ids,
)
from src.model import build_model  # noqa: E402
from src.train import train_model  # noqa: E402

DATA_ROOT = PROJECT_ROOT.parent / "data"
XVIEW_TILES = DATA_ROOT / "xview" / "train_images" / "train_images"
XVIEW_LABELS = DATA_ROOT / "xview" / "train_labels" / "xView_train.geojson"
RP_TRAIN_IMG = DATA_ROOT / "rareplanes" / "train" / "PS-RGB_tiled"
RP_TRAIN_GJ = DATA_ROOT / "rareplanes" / "train" / "geojson_aircraft_tiled"
SAMPLE_DIR = PROJECT_ROOT / "figures" / "sample_chips"


def save_samples(chips: list[Chip], prefix: str, n_per_class: int = 6) -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    pos = [c for c in chips if c.label == 1][:n_per_class]
    neg = [c for c in chips if c.label == 0][:n_per_class]
    for i, c in enumerate(pos):
        Image.fromarray(c.array).save(SAMPLE_DIR / f"{prefix}_pos_{i:02d}.png")
    for i, c in enumerate(neg):
        Image.fromarray(c.array).save(SAMPLE_DIR / f"{prefix}_neg_{i:02d}.png")
    print(f"saved {len(pos)} positives + {len(neg)} negatives to {SAMPLE_DIR} ({prefix}_*)")


def extract_for_inspection() -> tuple[list[Chip], list[Chip]]:
    rng = random.Random(42)
    # RarePlanes — grab 4 tiles
    rp_tiles = sorted(rareplanes_tile_ids(RP_TRAIN_GJ, "train"))
    rng.shuffle(rp_tiles)
    rp_chips = extract_rareplanes_chips(
        RP_TRAIN_IMG, RP_TRAIN_GJ, rp_tiles[:4], negatives_per_tile=3, rng=rng
    )
    # xView — grab 4 aircraft tiles + 4 non-aircraft
    xview_labels = load_xview_labels(XVIEW_LABELS)
    all_tifs = sorted(p.name for p in XVIEW_TILES.iterdir() if p.suffix == ".tif")
    aircraft_ids = sorted(xview_labels.keys())[:4]
    non_air = [t for t in all_tifs if t not in xview_labels]
    rng.shuffle(non_air)
    xview_chips = extract_xview_chips(
        XVIEW_TILES, xview_labels, aircraft_ids, non_air[:4],
        n_negatives=8, rng=rng,
    )
    return rp_chips, xview_chips


def overfit_test(chips: list[Chip], device: str) -> None:
    # balance to 1:1 and keep just 100 chips
    pos = [c for c in chips if c.label == 1]
    neg = [c for c in chips if c.label == 0]
    k = min(len(pos), len(neg), 50)
    rng = random.Random(7)
    rng.shuffle(pos); rng.shuffle(neg)
    sub = pos[:k] + neg[:k]
    rng.shuffle(sub)
    print(f"overfit test on {len(sub)} chips ({k} pos / {k} neg)")

    loader = DataLoader(ChipDataset(sub, train=True), batch_size=32, shuffle=True)
    model = build_model(num_classes=2, pretrained=True)
    report = train_model(model, loader, epochs=15, lr=5e-4, device=device, log_prefix="[overfit] ")
    print(f"overfit final: loss={report.final_train_loss:.4f}  acc={report.final_train_acc:.4f}")
    if report.final_train_acc < 0.9:
        print("!! WARNING: training accuracy < 0.90 after 15 epochs. Pipeline may have a bug.")
    else:
        print("OK: overfit passed — model can memorize the training set, so gradients flow and labels are consistent.")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device = {device}")

    print("\n=== 1. Extracting chips for visual inspection ===")
    rp_chips, xview_chips = extract_for_inspection()
    print(f"RarePlanes: {sum(c.label for c in rp_chips)} pos / {sum(1 - c.label for c in rp_chips)} neg")
    print(f"xView:      {sum(c.label for c in xview_chips)} pos / {sum(1 - c.label for c in xview_chips)} neg")
    save_samples(rp_chips, "rp")
    save_samples(xview_chips, "xview")

    print("\n=== 2. Overfit test on xView training chips ===")
    overfit_test(xview_chips, device)


if __name__ == "__main__":
    main()
