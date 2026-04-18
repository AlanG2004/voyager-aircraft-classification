"""End-to-end pipeline entry point.

Usage:
    python run.py                    # both directions, writes figures + results.json
    python run.py --smoke            # tiny end-to-end test
    python run.py --direction a      # only rareplanes->xview
    python run.py --direction b      # only xview->rareplanes

Each direction runs a three-way split:
    source (80% tiles)   -> train
    source (20% tiles)   -> within-dataset validation (reference baseline)
    other dataset        -> cross-dataset test (the domain-gap measurement)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from src.data import (  # noqa: E402
    Chip,
    ChipDataset,
    extract_rareplanes_chips,
    extract_xview_chips,
    load_xview_labels,
    rareplanes_tile_ids,
)
from src.eval import (  # noqa: E402
    EvalReport,
    evaluate,
    save_comparison_bar,
    save_confusion_plot,
)
from src.model import build_model, trainable_params  # noqa: E402
from src.plot_style import apply_theme  # noqa: E402
from src.train import train_model  # noqa: E402

DATA_ROOT = HERE.parent / "data"
XVIEW_TILES = DATA_ROOT / "xview" / "train_images" / "train_images"
XVIEW_LABELS = DATA_ROOT / "xview" / "train_labels" / "xView_train.geojson"
RP_TRAIN_IMG = DATA_ROOT / "rareplanes" / "train" / "PS-RGB_tiled"
RP_TRAIN_GJ = DATA_ROOT / "rareplanes" / "train" / "geojson_aircraft_tiled"
RP_TEST_IMG = DATA_ROOT / "rareplanes" / "test" / "PS-RGB_tiled"
RP_TEST_GJ = DATA_ROOT / "rareplanes" / "test" / "geojson_aircraft_tiled"
FIG_DIR = HERE / "figures"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _balance(chips: list[Chip], rng: random.Random) -> list[Chip]:
    """Truncate the majority class so positives == negatives. Keeps the task
    symmetric and makes accuracy interpretable (baseline is ~0.5)."""
    pos = [c for c in chips if c.label == 1]
    neg = [c for c in chips if c.label == 0]
    k = min(len(pos), len(neg))
    rng.shuffle(pos)
    rng.shuffle(neg)
    out = pos[:k] + neg[:k]
    rng.shuffle(out)
    return out


def _tile_split(tile_ids: list[str], val_ratio: float, rng: random.Random) -> tuple[list[str], list[str]]:
    """Deterministic 80/20 tile-level split. Sort for cross-machine reproducibility."""
    sorted_ids = sorted(tile_ids)
    rng.shuffle(sorted_ids)
    n_val = max(1, int(len(sorted_ids) * val_ratio))
    return sorted_ids[n_val:], sorted_ids[:n_val]


def run_direction_a(cfg: dict, device: str, smoke: bool) -> dict:
    """Train on RarePlanes real -> within-val on held-out RP -> cross-test on xView."""
    rng = random.Random(cfg["seed"])
    prefix = "[A RP->xV] "

    # ---- RP tile split (train / within-val) ----
    print(f"{prefix}loading RarePlanes train tile list (scanning geojsons for Public_Train)...")
    rp_all = rareplanes_tile_ids(RP_TRAIN_GJ, "train")
    n_rp = 30 if smoke else cfg["train"]["n_rareplanes_tiles"]
    rp_all = sorted(rp_all)
    rng_pool = random.Random(cfg["seed"])
    rng_pool.shuffle(rp_all)
    rp_pool = rp_all[:n_rp]
    rp_train_tiles, rp_val_tiles = _tile_split(rp_pool, val_ratio=0.2, rng=random.Random(cfg["seed"] + 1))
    print(f"{prefix}RP pool: {len(rp_train_tiles)} train tiles, {len(rp_val_tiles)} within-val tiles")

    print(f"{prefix}extracting RP train chips...")
    train_chips = extract_rareplanes_chips(
        RP_TRAIN_IMG, RP_TRAIN_GJ, rp_train_tiles, cfg["train"]["negatives_per_tile"], rng,
    )
    train_chips = _balance(train_chips, rng)
    print(f"{prefix}train chips: {len(train_chips)} (balanced 1:1)")

    print(f"{prefix}extracting RP within-val chips...")
    within_chips = extract_rareplanes_chips(
        RP_TRAIN_IMG, RP_TRAIN_GJ, rp_val_tiles, cfg["train"]["negatives_per_tile"], rng,
    )
    within_chips = _balance(within_chips, rng)
    print(f"{prefix}within-val chips: {len(within_chips)} (balanced 1:1)")

    # ---- xView cross-test ----
    print(f"{prefix}loading xView aircraft labels...")
    xview_labels = load_xview_labels(XVIEW_LABELS)
    all_tifs = sorted(p.name for p in XVIEW_TILES.iterdir() if p.suffix == ".tif")
    n_air = 8 if smoke else cfg["test"]["n_xview_aircraft_tiles"]
    n_non = 10 if smoke else cfg["test"]["n_xview_non_aircraft_tiles"]
    aircraft_ids = sorted(xview_labels.keys())[:n_air]
    non_air_pool = [t for t in all_tifs if t not in xview_labels]
    rng.shuffle(non_air_pool)
    non_air_ids = non_air_pool[:n_non]
    print(f"{prefix}extracting xView cross-test chips from {len(aircraft_ids)} aircraft + {len(non_air_ids)} non-aircraft tiles...")
    cross_chips = extract_xview_chips(
        XVIEW_TILES, xview_labels, aircraft_ids, non_air_ids,
        n_negatives=20 if smoke else 1500, rng=rng,
    )
    cross_chips = _balance(cross_chips, rng)
    print(f"{prefix}cross-test chips: {len(cross_chips)} (balanced 1:1)")

    return _train_and_eval(train_chips, within_chips, cross_chips, cfg, device, prefix, smoke)


def run_direction_b(cfg: dict, device: str, smoke: bool) -> dict:
    """Train on xView -> within-val on held-out xView -> cross-test on RarePlanes."""
    rng = random.Random(cfg["seed"])
    prefix = "[B xV->RP] "

    print(f"{prefix}loading xView aircraft labels...")
    xview_labels = load_xview_labels(XVIEW_LABELS)
    all_tifs = sorted(p.name for p in XVIEW_TILES.iterdir() if p.suffix == ".tif")
    n_air_total = 20 if smoke else cfg["train"]["n_xview_aircraft_tiles"]
    n_non_total = 20 if smoke else cfg["train"]["n_xview_non_aircraft_tiles"]

    aircraft_pool = sorted(xview_labels.keys())[:n_air_total]
    non_aircraft_pool = [t for t in all_tifs if t not in xview_labels]
    rng.shuffle(non_aircraft_pool)
    non_aircraft_pool = non_aircraft_pool[:n_non_total]

    air_train, air_val = _tile_split(aircraft_pool, val_ratio=0.2, rng=random.Random(cfg["seed"] + 2))
    non_train, non_val = _tile_split(non_aircraft_pool, val_ratio=0.2, rng=random.Random(cfg["seed"] + 3))
    print(f"{prefix}xView train: {len(air_train)} aircraft + {len(non_train)} non-aircraft; within-val: {len(air_val)}+{len(non_val)}")

    print(f"{prefix}extracting xView train chips...")
    train_chips = extract_xview_chips(
        XVIEW_TILES, xview_labels, air_train, non_train,
        n_negatives=20 if smoke else 2000, rng=rng,
    )
    train_chips = _balance(train_chips, rng)
    print(f"{prefix}train chips: {len(train_chips)} (balanced 1:1)")

    print(f"{prefix}extracting xView within-val chips...")
    within_chips = extract_xview_chips(
        XVIEW_TILES, xview_labels, air_val, non_val,
        n_negatives=8 if smoke else 500, rng=rng,
    )
    within_chips = _balance(within_chips, rng)
    print(f"{prefix}within-val chips: {len(within_chips)} (balanced 1:1)")

    # ---- RP cross-test ----
    print(f"{prefix}loading RarePlanes test tile list...")
    rp_test = rareplanes_tile_ids(RP_TEST_GJ, "test")
    rp_test = sorted(rp_test)
    rng.shuffle(rp_test)
    n_rp = 15 if smoke else cfg["test"]["n_rareplanes_tiles"]
    rp_test = rp_test[:n_rp]
    print(f"{prefix}extracting RP cross-test chips from {len(rp_test)} tiles...")
    cross_chips = extract_rareplanes_chips(
        RP_TEST_IMG, RP_TEST_GJ, rp_test, cfg["test"]["negatives_per_tile"], rng,
    )
    cross_chips = _balance(cross_chips, rng)
    print(f"{prefix}cross-test chips: {len(cross_chips)} (balanced 1:1)")

    return _train_and_eval(train_chips, within_chips, cross_chips, cfg, device, prefix, smoke)


def _train_and_eval(
    train_chips: list[Chip],
    within_chips: list[Chip],
    cross_chips: list[Chip],
    cfg: dict,
    device: str,
    prefix: str,
    smoke: bool,
) -> dict:
    for name, cs in [("train", train_chips), ("within-val", within_chips), ("cross-test", cross_chips)]:
        if not cs:
            raise RuntimeError(f"{prefix}{name} chip set is EMPTY — cannot proceed")
        labels = [c.label for c in cs]
        if len(set(labels)) < 2:
            raise RuntimeError(f"{prefix}{name} has only one class (labels={set(labels)})")

    train_ds = ChipDataset(train_chips, train=True)
    within_ds = ChipDataset(within_chips, train=False)
    cross_ds = ChipDataset(cross_chips, train=False)

    bs = cfg["train"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
    within_loader = DataLoader(within_ds, batch_size=bs, shuffle=False, num_workers=0)
    cross_loader = DataLoader(cross_ds, batch_size=bs, shuffle=False, num_workers=0)

    model = build_model(num_classes=2, pretrained=True)
    print(f"{prefix}trainable params: {trainable_params(model):,}")

    epochs = 1 if smoke else cfg["train"]["epochs"]
    train_report = train_model(model, train_loader, epochs, cfg["train"]["lr"], device, prefix)
    within_report = evaluate(model, within_loader, device)
    cross_report = evaluate(model, cross_loader, device)
    gap = within_report.f1_macro - cross_report.f1_macro
    print(f"{prefix}within-val F1={within_report.f1_macro:.3f}  cross-test F1={cross_report.f1_macro:.3f}  gap={gap:+.3f}")

    models_dir = HERE / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / f"{cfg['name']}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"{prefix}saved weights → {model_path.relative_to(HERE)}")

    probs_dir = HERE / "figures" / "probs"
    probs_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        probs_dir / f"{cfg['name']}_within_val.npz",
        probs_pos=within_report.probs_pos,
        labels_true=within_report.labels_true,
    )
    np.savez(
        probs_dir / f"{cfg['name']}_cross_test.npz",
        probs_pos=cross_report.probs_pos,
        labels_true=cross_report.labels_true,
    )
    print(f"{prefix}saved probabilities → figures/probs/{cfg['name']}_*.npz")

    return {
        "config_name": cfg["name"],
        "train": {
            "epochs": train_report.epochs_run,
            "final_train_loss": round(train_report.final_train_loss, 4),
            "final_train_acc": round(train_report.final_train_acc, 4),
            "wall_seconds": round(train_report.wall_seconds, 1),
            "n_chips": len(train_chips),
        },
        "within_val": within_report.as_dict(),
        "cross_test": cross_report.as_dict(),
        "generalization_gap_f1": round(gap, 4),
        "_within_report": within_report,
        "_cross_report": cross_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=["a", "b", "both"], default="both")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    FIG_DIR.mkdir(exist_ok=True)
    apply_theme()

    device = args.device or (
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"device = {device}")

    cfg_a = yaml.safe_load(open(HERE / "configs" / "rareplanes_to_xview.yaml"))
    cfg_b = yaml.safe_load(open(HERE / "configs" / "xview_to_rareplanes.yaml"))
    set_seed(cfg_a["seed"])

    results: dict[str, dict] = {}
    t0 = time.time()
    if args.direction in ("a", "both"):
        results["rareplanes_to_xview"] = run_direction_a(cfg_a, device, args.smoke)
    if args.direction in ("b", "both"):
        results["xview_to_rareplanes"] = run_direction_b(cfg_b, device, args.smoke)

    within_reports: dict[str, EvalReport] = {}
    cross_reports: dict[str, EvalReport] = {}
    for name, r in results.items():
        within_reports[name] = r.pop("_within_report")
        cross_reports[name] = r.pop("_cross_report")
        save_confusion_plot(
            r["within_val"]["confusion"], f"{name}  within-val",
            FIG_DIR / f"confusion_{name}_within.png",
        )
        save_confusion_plot(
            r["cross_test"]["confusion"], f"{name}  cross-test",
            FIG_DIR / f"confusion_{name}_cross.png",
        )
    save_comparison_bar(cross_reports, FIG_DIR / "comparison_f1.png")

    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nwrote results.json and figures in {FIG_DIR}")
    print(f"total wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
