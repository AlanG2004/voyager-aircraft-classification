# Remote Sensing ML Pipeline: Cross-Dataset Aircraft Classification

A bidirectional cross-dataset chip-classification pipeline on **xView** and **RarePlanes (real)**. Each model is trained on one dataset, then evaluated on a held-out split of the same dataset (within-val) and the other dataset (cross-test). The gap between the two is the reported metric.

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

## Repository layout

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

## Results

| Direction | Train chips | Within-val F1 | Cross-test F1 | Gap | Cross-test N |
|---|---|---|---|---|---|
| RP → xView | 712 | 0.952 | 0.948 | +0.004 | 2,058 |
| xView → RP | 1,534 | 0.968 | 0.888 | +0.079 | 262 |

Full metrics in `results.json`. Confusion matrices and the gap-comparison chart under `figures/`. Narrative writeup — framing, surprise, and the per-direction failure-mode breakdown — at `writeup/index.html`.

## Reproducibility notes

- Seed fixed at 42 (in both configs). All rngs (Python `random`, numpy, torch) seeded from it.
- Tile lists sorted alphabetically before shuffling, so the subset is deterministic across machines.
- Pinned dependency versions in `requirements.txt`.
- Single entry point: `./run.sh` creates the venv, installs deps, and runs both directions end-to-end. No hidden state, no `cache/`.
- ImageNet mean/std normalization (not dataset-specific) because the backbone is pretrained on ImageNet and expects that input distribution.
