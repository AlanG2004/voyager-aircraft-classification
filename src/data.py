"""Chip extraction and Dataset for xView and RarePlanes.

Both datasets feed into the same binary aircraft/non-aircraft classifier via a
shared chip-extraction pipeline. xView gives us pixel-coord bounding boxes
directly (bounds_imcoords); RarePlanes gives us WGS84 polygons that we
project to pixel coords via each tile's GeoTransform (parsed from the .aux.xml
sidecar).

Aircraft classes in xView: 11 Fixed-wing, 12 Small, 13 Cargo, 15 Helicopter.
"""
from __future__ import annotations

import json
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

XVIEW_AIRCRAFT_IDS = {11, 12, 13, 15}
# Asymmetric no-data tolerance: be strict on negatives so the model doesn't
# learn "black pixels ≈ not-aircraft"; be lenient on positives so aircraft
# near the edge of valid imagery (common in RarePlanes' blacked-out tiles)
# aren't all discarded.
POSITIVE_NODATA_RATIO = 0.25
NEGATIVE_NODATA_RATIO = 0.10
CHIP_SIZE = 224


@dataclass
class Chip:
    """One training example. `array` is HxWx3 uint8 RGB."""
    array: np.ndarray
    label: int  # 1 = aircraft, 0 = non-aircraft
    source: str  # "xview" or "rareplanes"
    meta: dict


def _is_valid_chip(arr: np.ndarray, is_positive: bool) -> bool:
    """Reject chips that are mostly zero-padded no-data regions."""
    if arr.size == 0:
        return False
    zero_pixels = np.all(arr == 0, axis=-1)
    threshold = POSITIVE_NODATA_RATIO if is_positive else NEGATIVE_NODATA_RATIO
    return zero_pixels.mean() < threshold


def _extract_chip(
    image: np.ndarray, cx: int, cy: int, size: int = CHIP_SIZE
) -> np.ndarray | None:
    """Crop a square chip centered at (cx, cy). Returns None if out of bounds."""
    h, w = image.shape[:2]
    half = size // 2
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    chip = image[y0:y1, x0:x1]
    if chip.shape[0] != size or chip.shape[1] != size:
        return None
    return chip


# ---------- xView ----------

def load_xview_labels(geojson_path: Path) -> dict[str, list[dict]]:
    """Return a mapping image_id -> list of aircraft feature dicts.

    We filter to aircraft-only at load time so downstream code never has to
    re-scan 600k non-aircraft features.
    """
    with open(geojson_path) as f:
        data = json.load(f)
    per_image: dict[str, list[dict]] = {}
    for feat in data["features"]:
        props = feat["properties"]
        if props.get("type_id") not in XVIEW_AIRCRAFT_IDS:
            continue
        image_id = props["image_id"]
        xmin, ymin, xmax, ymax = (int(round(float(v))) for v in props["bounds_imcoords"].split(","))
        per_image.setdefault(image_id, []).append(
            {"bbox": (xmin, ymin, xmax, ymax), "type_id": props["type_id"]}
        )
    return per_image


def _read_tif(path: Path) -> np.ndarray:
    """Read an xView GeoTIFF as HxWx3 uint8."""
    arr = tifffile.imread(str(path))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[0] < arr.shape[-1]:
        arr = arr.transpose(1, 2, 0)  # band-first -> band-last
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def extract_xview_chips(
    tiles_dir: Path,
    labels: dict[str, list[dict]],
    aircraft_image_ids: list[str],
    non_aircraft_image_ids: list[str],
    n_negatives: int,
    rng: random.Random,
) -> list[Chip]:
    """Extract aircraft chips (from labels) + background-negative chips."""
    chips: list[Chip] = []
    # positives
    for image_id in aircraft_image_ids:
        tif = tiles_dir / image_id
        if not tif.exists():
            continue
        try:
            img = _read_tif(tif)
        except Exception:
            continue
        for feat in labels.get(image_id, []):
            xmin, ymin, xmax, ymax = feat["bbox"]
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2
            chip = _extract_chip(img, cx, cy)
            if chip is None or not _is_valid_chip(chip, is_positive=True):
                continue
            chips.append(Chip(chip, 1, "xview", {"image_id": image_id, "type_id": feat["type_id"]}))
    # negatives drawn from tiles without aircraft
    neg_pool = non_aircraft_image_ids[:]
    rng.shuffle(neg_pool)
    picked = 0
    for image_id in neg_pool:
        if picked >= n_negatives:
            break
        tif = tiles_dir / image_id
        if not tif.exists():
            continue
        try:
            img = _read_tif(tif)
        except Exception:
            continue
        h, w = img.shape[:2]
        attempts = 0
        while picked < n_negatives and attempts < 30:
            attempts += 1
            cx = rng.randint(CHIP_SIZE // 2, w - CHIP_SIZE // 2)
            cy = rng.randint(CHIP_SIZE // 2, h - CHIP_SIZE // 2)
            chip = _extract_chip(img, cx, cy)
            if chip is None or not _is_valid_chip(chip, is_positive=False):
                continue
            chips.append(Chip(chip, 0, "xview", {"image_id": image_id}))
            picked += 1
    return chips


# ---------- RarePlanes ----------

def _parse_geotransform(aux_xml: Path) -> tuple[float, float, float, float, float, float] | None:
    """Read the GDAL PAMDataset sidecar and pull GeoTransform coefficients."""
    try:
        root = ET.parse(aux_xml).getroot()
    except Exception:
        return None
    node = root.find("GeoTransform")
    if node is None or not node.text:
        return None
    coeffs = [float(v) for v in node.text.strip().split(",")]
    return tuple(coeffs)  # (ox, a, b, oy, d, e)


def _lonlat_to_pixel(
    lon: float, lat: float, gt: tuple[float, float, float, float, float, float]
) -> tuple[int, int]:
    """Invert the affine transform. Assumes b == d == 0 (axis-aligned tiles)."""
    ox, a, _b, oy, _d, e = gt
    col = (lon - ox) / a
    row = (lat - oy) / e
    return int(round(col)), int(round(row))


def extract_rareplanes_chips(
    imagery_dir: Path,
    geojson_dir: Path,
    tile_ids: list[str],
    negatives_per_tile: int,
    rng: random.Random,
) -> list[Chip]:
    """For each tile, take every aircraft centroid as a positive chip, and
    sample `negatives_per_tile` background chips (centroids far from aircraft).
    """
    chips: list[Chip] = []
    for tile_id in tile_ids:
        png = imagery_dir / f"{tile_id}.png"
        aux = imagery_dir / f"{tile_id}.png.aux.xml"
        gj = geojson_dir / f"{tile_id}.geojson"
        if not (png.exists() and aux.exists() and gj.exists()):
            continue
        gt = _parse_geotransform(aux)
        if gt is None:
            continue
        try:
            img = np.asarray(Image.open(png).convert("RGB"))
        except Exception:
            continue
        with open(gj) as f:
            data = json.load(f)

        positive_centroids: list[tuple[int, int]] = []
        for feat in data.get("features", []):
            geom = feat.get("geometry") or {}
            coords = geom.get("coordinates")
            if not coords:
                continue
            ring = coords[0] if geom.get("type") == "Polygon" else coords
            xs = [pt[0] for pt in ring]
            ys = [pt[1] for pt in ring]
            lon_c = sum(xs) / len(xs)
            lat_c = sum(ys) / len(ys)
            cx, cy = _lonlat_to_pixel(lon_c, lat_c, gt)
            positive_centroids.append((cx, cy))
            chip = _extract_chip(img, cx, cy)
            if chip is None or not _is_valid_chip(chip, is_positive=True):
                continue
            chips.append(Chip(chip, 1, "rareplanes", {"tile_id": tile_id}))

        h, w = img.shape[:2]
        picked = 0
        attempts = 0
        while picked < negatives_per_tile and attempts < negatives_per_tile * 10:
            attempts += 1
            cx = rng.randint(CHIP_SIZE // 2, w - CHIP_SIZE // 2)
            cy = rng.randint(CHIP_SIZE // 2, h - CHIP_SIZE // 2)
            if any(abs(cx - px) < CHIP_SIZE and abs(cy - py) < CHIP_SIZE for px, py in positive_centroids):
                continue
            chip = _extract_chip(img, cx, cy)
            if chip is None or not _is_valid_chip(chip, is_positive=False):
                continue
            chips.append(Chip(chip, 0, "rareplanes", {"tile_id": tile_id}))
            picked += 1
    return chips


def rareplanes_tile_ids(geojson_dir: Path, split: str) -> list[str]:
    """Walk the tile geojsons and return tile_ids whose features are tagged
    with the matching Public_Train / Public_Test flag. A tile with any train
    feature goes into train; same for test. No overlap because the split is
    labeled per-feature and tile-id is stable.
    """
    flag = "Public_Train" if split == "train" else "Public_Test"
    out: list[str] = []
    for gj in geojson_dir.iterdir():
        if not gj.name.endswith(".geojson"):
            continue
        try:
            with open(gj) as f:
                data = json.load(f)
        except Exception:
            continue
        if any(feat.get("properties", {}).get(flag) == 1 for feat in data.get("features", [])):
            out.append(gj.stem)
    return out


# ---------- PyTorch Dataset ----------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(train: bool) -> transforms.Compose:
    ops = []
    if train:
        ops.append(transforms.RandomHorizontalFlip())
        ops.append(transforms.RandomVerticalFlip())
        ops.append(transforms.ColorJitter(0.15, 0.15, 0.15))
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(ops)


class ChipDataset(Dataset):
    def __init__(self, chips: list[Chip], train: bool):
        self.chips = chips
        self.transform = _build_transform(train)

    def __len__(self) -> int:
        return len(self.chips)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        chip = self.chips[idx]
        img = Image.fromarray(chip.array)
        return self.transform(img), chip.label
