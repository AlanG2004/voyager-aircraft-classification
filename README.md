# Remote Sensing ML Pipeline: Cross-Dataset Aircraft Classification

A bidirectional cross-dataset chip-classification pipeline on **xView** and **RarePlanes (real)**. One model is trained on each dataset and evaluated in two ways: on held-out tiles from the **same** dataset (within-val) and on tiles from the **other** dataset (cross-test). The gap between the two is what matters. It names the domain shift between two overhead imagery sources captured by the same sensor but annotated by different teams for different goals.

## Run it

```bash
./run.sh                        # one command: creates .venv, installs deps, runs the pipeline
```

That's it. `run.sh` is idempotent: the second call skips venv setup and dependency install and goes straight to the run. Device auto-detects (MPS on Apple Silicon → CUDA → CPU); pass `--device cpu` to force. For a one-minute wiring check instead of the full run, use `./run.sh --smoke`.

Expected wall time on MPS / Apple Silicon: ~8 minutes. Writes `results.json` plus confusion matrices and a comparison chart under `figures/`.

Manual equivalent (if you'd rather do it step-by-step):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py --device mps      # or --device cuda, or --device cpu
```

## Predict on a new image

After `./run.sh` finishes, two model checkpoints live under `models/`. You can run them against any image without retraining.

**Chip-level prediction.** Takes a 224×224 chip, or any larger image (it gets center-cropped):

```bash
python predict.py --model models/rareplanes_to_xview.pt --image figures/sample_chips/xview_pos_00.png
# image:    figures/sample_chips/xview_pos_00.png
# model:    rareplanes_to_xview.pt  (device=mps)
# chip:     224x224 from 224x224 input
# predict:  aircraft  (p=0.98)
# scores:   not-aircraft=0.02  aircraft=0.98
```

**Tile-level prediction.** Slides a 224×224 window across a whole tile (xView .tif or RarePlanes .png) and saves a per-window aircraft-probability heatmap overlaid on the original tile:

```bash
python predict_tile.py --model models/xview_to_rareplanes.pt --tile ../data/rareplanes/test/PS-RGB_tiled/some_tile.png
# tile: 512x512 -> sliding 9 windows (3 x 3, stride=112)
# inference: 0.18s on mps
# positives (p>0.5): 2/9 windows
# max aircraft probability:  0.91
# mean aircraft probability: 0.34
# saved heatmap -> some_tile.heatmap.png
```

Both scripts auto-detect device (MPS → CUDA → CPU). They work on either dataset's native file format.

## What's in the box

```
project/
├── run.sh              one-script repro: venv + deps + pipeline in a single command
├── run.py              pipeline entry point; runs both directions end-to-end
├── predict.py          chip-level inference: one image in, prediction out
├── predict_tile.py     tile-level inference: sliding-window heatmap across a whole tile
├── requirements.txt
├── README.md
├── results.json        per-direction metrics: within-val, cross-test, generalization gap
├── configs/
│   ├── rareplanes_to_xview.yaml   direction A: train RP → test xView
│   └── xview_to_rareplanes.yaml   direction B: train xView → test RP
├── src/                library code (imported by entry scripts)
│   ├── data.py         chip extraction (both datasets) + validity filter + PyTorch Dataset
│   ├── model.py        ResNet-18 backbone, mostly-frozen, 2-layer binary head, BN freeze utility
│   ├── train.py        AdamW + cosine annealing; keeps BN in eval mode while training
│   ├── eval.py         accuracy + precision + recall + macro-F1 + per-class F1 + confusion + majority baseline
│   └── plot_style.py   matplotlib theme (navy + brass, paper context)
├── scripts/            auxiliary scripts (run when needed, not part of the main pipeline)
│   ├── sanity.py       pre-flight: sample chips + overfit test
│   ├── verify.py       post-run: tile-level leakage + metric-math consistency
│   ├── chip_strip.py   regenerates the writeup hero chip strip
│   └── plot_curves.py  PR + ROC curves from saved per-sample probabilities
├── models/             trained weight files (populated by run.sh): rareplanes_to_xview.pt, xview_to_rareplanes.pt
├── figures/            output figures (confusion matrices, gap chart, PR/ROC curves, heatmap demo, sample chips)
└── writeup/
    └── index.html      polished one-page writeup with embedded results
```

## Data expectations

The pipeline reads from the project's sibling `data/` directory:

```
data/
├── xview/
│   ├── train_images/train_images/*.tif         846 Kaggle GeoTIFFs
│   └── train_labels/xView_train.geojson        601k object labels, 62 classes
└── rareplanes/
    ├── train/PS-RGB_tiled/*.png                5815 PS-RGB tiles (512×512)
    ├── train/geojson_aircraft_tiled/*.geojson  per-tile aircraft annotations
    ├── test/PS-RGB_tiled/*.png
    └── test/geojson_aircraft_tiled/*.geojson
```

## Problem framing

The prompt is deliberately open: two satellite-imagery datasets, build a functional pipeline. Instead of training two independent classifiers and reporting separate accuracies, I framed it as a **cross-dataset generalization study**: each model is deliberately tested on both its own held-out set and the other dataset, so the result is a measurable domain-gap number, not two free-floating accuracies.

Both datasets come from Maxar WorldView-3 imagery at ~0.3 m/pixel pan-sharpened. That's a quiet-but-critical match. Aircraft at the same real-world size have roughly the same pixel footprint across both datasets, so the cross-test comparison is actually a test of *content-distribution* and *annotation-protocol* shift, not a sensor/resolution shift.

## Approach

**Task**: binary classification. *"Does this 224×224 chip contain an aircraft centered in it?"*

**Positive chips**: 224×224 crops centered on the centroid of an aircraft polygon. For xView, centroid comes from the pre-computed `bounds_imcoords`. For RarePlanes, I parse the GeoTransform from the `.aux.xml` sidecar of each PNG and invert it to take WGS84 polygon coords into pixel space.

**Negative chips**: 224×224 crops from background regions. Extraction differs by dataset because the datasets are shaped differently:
- **xView negatives** are drawn from the 703 tiles that contain *no* aircraft labels, so they include buildings, roads, ships, rural land. Diverse satellite content.
- **RarePlanes negatives** are drawn from tiles far from aircraft centroids. Because every RP tile contains aircraft, negatives here are airport-adjacent ground: tarmac, hangars, runway edges. Narrow by construction.

This asymmetry is honest: RarePlanes is an aircraft-specific dataset; xView is a general object dataset. The cross-dataset gap that falls out of this is therefore two-directional: a RP-trained model sees only airport-environment negatives during training, then faces an entire satellite-imagery distribution at test time.

**Validity filter**: chips where more than 25% (positives) or 10% (negatives) of pixels are pure black (0,0,0) are rejected. Both datasets use `(0,0,0)` as their no-data sentinel. Asymmetric thresholds let aircraft near blacked-out tile edges survive while preventing the model from learning "black pixels ≈ not-aircraft."

**Class balancing**: after extraction each split is truncated to 1:1 positive:negative. Simple, makes the baseline interpretable (close to 0.5), lets us report macro-F1 without weight tricks.

**Model**: ResNet-18 pretrained on ImageNet. Layers 1–3 frozen; layer 4 and a new 2-layer head (`512→128→2` with ReLU + Dropout(0.3)) trained. BatchNorm layers are held in `eval()` mode throughout training so that frozen-backbone running statistics don't silently drift during fine-tune. 8.4M trainable params total.

**Training**: AdamW(lr=3e-4, weight_decay=1e-4) with cosine annealing over 6 epochs, batch 64. No hyperparameter tuning; defaults chosen once.

**Per direction, three-way split at the tile level** (no chip-level leakage across splits):
1. Source-dataset training tiles → train the model.
2. Source-dataset held-out tiles → within-val: *"does it fit its own domain?"*
3. Other-dataset tiles → cross-test: *"how far does it transfer?"*

The generalization gap = within-val macro-F1 − cross-test macro-F1, reported per direction.

## Eval metrics and why

- **Macro-F1** handles the slight class imbalance (after 1:1 balancing, still near-balanced but not exact) without hiding per-class behavior.
- **Per-class F1** shows which class the model struggles with. Useful for reasoning about the negative-distribution asymmetry.
- **Precision + recall** exposes asymmetric errors (high precision, low recall ≠ low precision, high recall).
- **Confusion matrix** per direction per split. Most direct visual of "where did it get confused."
- **Majority-class baseline** (always predict the most common label in the test set). The honesty check. If the model can't beat this, it learned nothing.

## Results

Full metrics in `results.json`. Per-direction summary (macro-F1, 1:1 balanced test sets, seed 42):

| Direction | Train chips | Within-val F1 | Cross-test F1 | Gap | Cross-test N |
|---|---|---|---|---|---|
| RP → xView | 712 | 0.952 | **0.948** | +0.004 | 2,058 |
| xView → RP | 1,534 | 0.968 | **0.888** | +0.079 | 262 |
| Majority baseline | · | 0.333 | 0.333 | · | · |

**The surprise + its explanation**: I predicted RP → xView would have the larger gap because RP's negatives are narrow. The reverse was true. Direction A transfers almost perfectly (gap +0.004); direction B drops noticeably (+0.079). Running `verify.py` on the confusion matrices revealed *why*: the two directions fail in qualitatively different ways:

- **Direction A (RP → xView)** is *conservative*: 96 false negatives, 10 false positives. Aircraft recall 91%. RP-trained model rejects anything that doesn't match its narrow template.
- **Direction B (xView → RP)** is *permissive*: 1 false negative out of 131 aircraft, 28 false positives on backgrounds. Aircraft recall 99%. xView-trained model flags every RP aircraft but misfires on airport-adjacent backgrounds that share visual structure with aircraft surroundings.

The gap asymmetry is therefore not a simple "one direction is harder". It's two different failure modes driven by the training-set negative distributions. Exactly what you'd expect given RP's narrow vs. xView's broad negative content.

Per-split confusion matrices and the cross-test comparison chart live under `figures/`. The polished one-page narrative is at `writeup/index.html`.

## What I'd do with another day

1. Add the synthetic arm: train a third model on RarePlanes-synthetic, evaluate on both reals. Sim-to-real is the EMSI problem.
2. Coarse multi-class aircraft subtypes using a defensible 3-way taxonomy mapped between xView and RarePlanes.
3. Simple feature-alignment loss (DANN or a MMD regularizer on the penultimate layer) to see if the bidirectional gap shrinks.
4. Test-time augmentation (horizontal flip average). A free +0.5% F1 on most tasks.

## Reproducibility notes

- Seed fixed at 42 (in both configs). All rngs (Python `random`, numpy, torch) seeded from it.
- Tile lists sorted alphabetically before shuffling, so the subset is deterministic across machines.
- Pinned dependency versions in `requirements.txt`.
- Single entry point: `./run.sh` creates the venv, installs deps, and runs both directions end-to-end. No hidden state, no `cache/`.
- ImageNet mean/std normalization (not dataset-specific) because the backbone is pretrained on ImageNet and expects that input distribution.
