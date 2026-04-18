"""Run one of the trained models on a single image chip.

    python predict.py --model models/rareplanes_to_xview.pt --image some_chip.png
    python predict.py --model models/xview_to_rareplanes.pt --image tile.tif

Accepts .png / .jpg / .tif. Images larger than 224x224 are center-cropped
(the nearest 224x224 region at the tile center). Use `predict_tile.py` if
you want sliding-window inference over a whole tile.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from src.data import CHIP_SIZE, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.model import build_model  # noqa: E402

CLASS_NAMES = ["not-aircraft", "aircraft"]


def load_image(path: Path) -> np.ndarray:
    """Load a PNG/JPG/TIF as HxWx3 uint8."""
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        arr = tifffile.imread(str(path))
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[0] < arr.shape[-1]:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            arr = np.clip(arr / max(arr.max(), 1) * 255.0, 0, 255).astype(np.uint8)
        return arr
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def center_crop_to_chip(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    if h < CHIP_SIZE or w < CHIP_SIZE:
        raise ValueError(f"image too small ({w}x{h}); need at least {CHIP_SIZE}x{CHIP_SIZE}")
    cx, cy = w // 2, h // 2
    half = CHIP_SIZE // 2
    return arr[cy - half:cy + half, cx - half:cx + half]


def preprocess(chip: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(chip)).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return ((t - mean) / std).unsqueeze(0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, type=Path, help="path to .pt weights")
    p.add_argument("--image", required=True, type=Path, help="path to chip image")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    model = build_model(num_classes=2, pretrained=False)
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    arr = load_image(args.image)
    chip = center_crop_to_chip(arr)
    x = preprocess(chip).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred = int(probs.argmax())
    conf = float(probs[pred])

    print(f"image:    {args.image}")
    print(f"model:    {args.model.name}  (device={device})")
    print(f"chip:     {CHIP_SIZE}x{CHIP_SIZE} from {arr.shape[1]}x{arr.shape[0]} input")
    print(f"predict:  {CLASS_NAMES[pred]}  (p={conf:.3f})")
    print(f"scores:   not-aircraft={probs[0]:.3f}  aircraft={probs[1]:.3f}")


if __name__ == "__main__":
    main()
