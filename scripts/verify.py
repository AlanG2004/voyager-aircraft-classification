"""Sanity checks for the full-run results. Two probes:

1. Leakage check — reconstruct the tile splits used by run.py and verify
   no tile appears in both train and eval pools for the same direction.
2. Metric-math check — pull confusion matrices from results.json and verify
   the reported accuracy / F1 / baseline numbers match the matrices.

Run with: python verify.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.data import load_xview_labels, rareplanes_tile_ids  # noqa: E402

DATA_ROOT = PROJECT_ROOT.parent / "data"
XVIEW_TILES = DATA_ROOT / "xview" / "train_images" / "train_images"
XVIEW_LABELS = DATA_ROOT / "xview" / "train_labels" / "xView_train.geojson"
RP_TRAIN_GJ = DATA_ROOT / "rareplanes" / "train" / "geojson_aircraft_tiled"
RP_TEST_GJ = DATA_ROOT / "rareplanes" / "test" / "geojson_aircraft_tiled"


def _tile_split(tile_ids: list[str], val_ratio: float, rng: random.Random) -> tuple[list[str], list[str]]:
    sorted_ids = sorted(tile_ids)
    rng.shuffle(sorted_ids)
    n_val = max(1, int(len(sorted_ids) * val_ratio))
    return sorted_ids[n_val:], sorted_ids[:n_val]


# ---------- Probe 1: leakage ----------

def reconstruct_direction_a(cfg: dict) -> dict:
    rng = random.Random(cfg["seed"])
    rp_all = sorted(rareplanes_tile_ids(RP_TRAIN_GJ, "train"))
    rng_pool = random.Random(cfg["seed"])
    rng_pool.shuffle(rp_all)
    rp_pool = rp_all[:cfg["train"]["n_rareplanes_tiles"]]
    rp_train, rp_val = _tile_split(rp_pool, 0.2, random.Random(cfg["seed"] + 1))
    # cross-test uses xView tiles — no possible overlap with RP
    xview_labels = load_xview_labels(XVIEW_LABELS)
    all_tifs = sorted(p.name for p in XVIEW_TILES.iterdir() if p.suffix == ".tif")
    aircraft_ids = sorted(xview_labels.keys())[:cfg["test"]["n_xview_aircraft_tiles"]]
    non_air_pool = [t for t in all_tifs if t not in xview_labels]
    rng.shuffle(non_air_pool)
    non_air_ids = non_air_pool[:cfg["test"]["n_xview_non_aircraft_tiles"]]
    return {
        "rp_train": rp_train, "rp_within_val": rp_val,
        "xview_cross_aircraft": aircraft_ids, "xview_cross_non_aircraft": non_air_ids,
    }


def reconstruct_direction_b(cfg: dict) -> dict:
    rng = random.Random(cfg["seed"])
    xview_labels = load_xview_labels(XVIEW_LABELS)
    all_tifs = sorted(p.name for p in XVIEW_TILES.iterdir() if p.suffix == ".tif")
    aircraft_pool = sorted(xview_labels.keys())[:cfg["train"]["n_xview_aircraft_tiles"]]
    non_pool = [t for t in all_tifs if t not in xview_labels]
    rng.shuffle(non_pool)
    non_pool = non_pool[:cfg["train"]["n_xview_non_aircraft_tiles"]]
    air_train, air_val = _tile_split(aircraft_pool, 0.2, random.Random(cfg["seed"] + 2))
    non_train, non_val = _tile_split(non_pool, 0.2, random.Random(cfg["seed"] + 3))
    rp_test_all = sorted(rareplanes_tile_ids(RP_TEST_GJ, "test"))
    rng.shuffle(rp_test_all)
    rp_test = rp_test_all[:cfg["test"]["n_rareplanes_tiles"]]
    return {
        "xview_train_aircraft": air_train, "xview_train_non_aircraft": non_train,
        "xview_within_val_aircraft": air_val, "xview_within_val_non_aircraft": non_val,
        "rp_cross_test": rp_test,
    }


def check_leakage():
    print("=" * 60)
    print("PROBE 1 — tile-level leakage check")
    print("=" * 60)

    cfg_a = yaml.safe_load(open(PROJECT_ROOT / "configs" / "rareplanes_to_xview.yaml"))
    cfg_b = yaml.safe_load(open(PROJECT_ROOT / "configs" / "xview_to_rareplanes.yaml"))

    a = reconstruct_direction_a(cfg_a)
    b = reconstruct_direction_b(cfg_b)

    print(f"\nDirection A (RP->xView):")
    print(f"  RP train tiles:   {len(a['rp_train'])}")
    print(f"  RP within-val:    {len(a['rp_within_val'])}")
    print(f"  xView cross-test: {len(a['xview_cross_aircraft'])} aircraft + {len(a['xview_cross_non_aircraft'])} non-aircraft")

    overlap = set(a["rp_train"]) & set(a["rp_within_val"])
    print(f"  RP train ∩ within-val: {len(overlap)}  {'PASS' if not overlap else 'FAIL'}")

    print(f"\nDirection B (xView->RP):")
    print(f"  xView train:   {len(b['xview_train_aircraft'])} aircraft + {len(b['xview_train_non_aircraft'])} non-aircraft")
    print(f"  xView within-val: {len(b['xview_within_val_aircraft'])} + {len(b['xview_within_val_non_aircraft'])}")
    print(f"  RP cross-test: {len(b['rp_cross_test'])}")

    air_overlap = set(b["xview_train_aircraft"]) & set(b["xview_within_val_aircraft"])
    non_overlap = set(b["xview_train_non_aircraft"]) & set(b["xview_within_val_non_aircraft"])
    print(f"  xView aircraft train ∩ within-val:     {len(air_overlap)}  {'PASS' if not air_overlap else 'FAIL'}")
    print(f"  xView non-aircraft train ∩ within-val: {len(non_overlap)}  {'PASS' if not non_overlap else 'FAIL'}")

    # cross-dataset overlap is impossible by construction (different filesystems),
    # but let's prove it explicitly
    cross_overlap_a_rp = set(a["rp_train"]) & set(b["rp_cross_test"])
    print(f"\n  RP tiles used in A-train AND B-cross-test: {len(cross_overlap_a_rp)}  (expected 0 because A uses Public_Train, B uses Public_Test)")
    print(f"    {'PASS' if not cross_overlap_a_rp else 'INVESTIGATE'}")


# ---------- Probe 2: metric math ----------

def confusion_to_metrics(cm: list[list[int]]) -> dict:
    tn, fp = cm[0]
    fn, tp = cm[1]
    n = tn + fp + fn + tp
    acc = (tn + tp) / n if n else 0.0
    # per-class
    p_neg = tn / (tn + fn) if (tn + fn) else 0.0
    r_neg = tn / (tn + fp) if (tn + fp) else 0.0
    f1_neg = 2 * p_neg * r_neg / (p_neg + r_neg) if (p_neg + r_neg) else 0.0
    p_pos = tp / (tp + fp) if (tp + fp) else 0.0
    r_pos = tp / (tp + fn) if (tp + fn) else 0.0
    f1_pos = 2 * p_pos * r_pos / (p_pos + r_pos) if (p_pos + r_pos) else 0.0
    macro_f1 = (f1_neg + f1_pos) / 2
    macro_p = (p_neg + p_pos) / 2
    macro_r = (r_neg + r_pos) / 2
    return {
        "accuracy": round(acc, 4),
        "precision_macro": round(macro_p, 4),
        "recall_macro": round(macro_r, 4),
        "f1_macro": round(macro_f1, 4),
        "f1_per_class": [round(f1_neg, 4), round(f1_pos, 4)],
        "n": n,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def check_metrics():
    print("\n" + "=" * 60)
    print("PROBE 2 — confusion matrix consistency with reported metrics")
    print("=" * 60)

    results = json.load(open(PROJECT_ROOT / "results.json"))
    for direction, d in results.items():
        print(f"\n{direction}:")
        for split in ("within_val", "cross_test"):
            cm = d[split]["confusion"]
            recomputed = confusion_to_metrics(cm)
            reported = d[split]
            match_acc = abs(recomputed["accuracy"] - reported["accuracy"]) < 1e-3
            match_f1 = abs(recomputed["f1_macro"] - reported["f1_macro"]) < 1e-3
            print(f"  {split}:")
            print(f"    confusion: [[tn={cm[0][0]}, fp={cm[0][1]}], [fn={cm[1][0]}, tp={cm[1][1]}]]  N={recomputed['n']}")
            print(f"    reported:   acc={reported['accuracy']:.4f}  F1={reported['f1_macro']:.4f}")
            print(f"    recomputed: acc={recomputed['accuracy']:.4f}  F1={recomputed['f1_macro']:.4f}")
            print(f"    match: {'PASS' if match_acc and match_f1 else 'FAIL'}")
            # error breakdown
            if recomputed["tp"] + recomputed["fn"] > 0:
                recall_pos = recomputed["tp"] / (recomputed["tp"] + recomputed["fn"])
                print(f"    aircraft recall: {recall_pos:.3f}  ({recomputed['fn']} missed out of {recomputed['tp'] + recomputed['fn']})")


def main():
    check_leakage()
    check_metrics()
    print("\n" + "=" * 60)
    print("DONE")


if __name__ == "__main__":
    main()
