"""Generate the writeup hero chip strip from 8 real aircraft chips.

Reads the same datasets as run.py; writes `writeup/chip_strip.png`
(1792x224 = 8 positive chips of 224x224 tiled horizontally, RP + xView mix).

Run with: python chip_strip.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (  # noqa: E402
    extract_rareplanes_chips,
    extract_xview_chips,
    load_xview_labels,
    rareplanes_tile_ids,
)

DATA = PROJECT_ROOT.parent / "data"
RP_IMG = DATA / "rareplanes" / "train" / "PS-RGB_tiled"
RP_GJ = DATA / "rareplanes" / "train" / "geojson_aircraft_tiled"
XVIEW_TILES = DATA / "xview" / "train_images" / "train_images"
XVIEW_LABELS = DATA / "xview" / "train_labels" / "xView_train.geojson"
OUT = PROJECT_ROOT / "writeup" / "chip_strip.png"


def main() -> None:
    rng = random.Random(42)

    rp_tiles = sorted(rareplanes_tile_ids(RP_GJ, "train"))
    rng.shuffle(rp_tiles)
    rp_chips = extract_rareplanes_chips(RP_IMG, RP_GJ, rp_tiles[:8], negatives_per_tile=0, rng=rng)
    rp_pos = [c.array for c in rp_chips if c.label == 1][:4]

    xv_labels = load_xview_labels(XVIEW_LABELS)
    aircraft_ids = sorted(xv_labels.keys())[:8]
    xv_chips = extract_xview_chips(XVIEW_TILES, xv_labels, aircraft_ids, [], n_negatives=0, rng=rng)
    xv_pos = [c.array for c in xv_chips if c.label == 1][:4]

    arrs = rp_pos + xv_pos
    if len(arrs) < 8:
        raise RuntimeError(f"expected 8 aircraft chips, got {len(arrs)} (RP={len(rp_pos)}, xView={len(xv_pos)})")
    rng.shuffle(arrs)

    strip = np.concatenate(arrs[:8], axis=1)
    Image.fromarray(strip).save(OUT)
    print(f"saved {OUT}  shape={strip.shape}  (4 RP + 4 xView positives, shuffled)")


if __name__ == "__main__":
    main()
