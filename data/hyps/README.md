# Hyperparameter presets

## `hyp.hardneg-focal.yaml` — Focal Loss + hard negative mining

Preset for reducing false positives on confusing backgrounds (wildlife / ecological monitoring).

### Training

```bash
# Single GPU
python train.py --hyp data/hyps/hyp.hardneg-focal.yaml --data your_data.yaml --weights yolov5s.pt --img 640

# Multi-GPU DDP (batch-size = global batch, must divide by GPU count)
python -m torch.distributed.run --nproc_per_node 4 --master_port 29500 \
  train.py --hyp data/hyps/hyp.hardneg-focal.yaml --data your_data.yaml --weights yolov5s.pt \
  --img 640 --batch-size 64 --device 0,1,2,3
```

DDP constraints (unchanged from stock YOLOv5): do not use `--image-weights`, `--evolve`, or `--batch-size -1`.

### Dataset: `background` class

1. Add a `background` entry to `names` in your `data.yaml` (typically the last class).
2. Annotate confusing regions (rocks, leaves, shadows, etc.) with bounding boxes labeled `background`.
3. Mix 2k–5k background crops with normal target images; aim for ~4:1 negative-to-positive boxes in the training set.

Example for **mylts_birdanimal v0.0.8** export: [`data/mylts_birdanimal_hardneg.example.yaml`](data/mylts_birdanimal_hardneg.example.yaml) (7 ecological classes + `background`).

At inference, filter out the background class (see below).

### Key hyperparameters

| Key | Default (preset) | Description |
|-----|------------------|-------------|
| `fl_gamma` | `2.0` | Focal Loss γ; try `2.5` if false positives persist |
| `fl_alpha` | `0.25` | Focal Loss α |
| `obj` | `1.1` | Objectness loss gain (scaled in `train.py`) |
| `hnm_enabled` | `true` | Enable hard negative mining on obj loss |
| `neg_pos_ratio` | `4.5` | Cap negatives per positive (per GPU batch) |
| `hnm_topk_ratio` | `0.3` | Keep top fraction of negatives by obj confidence |
| `hnm_hard_weight` | `3.0` | Weight multiplier for high-conf negatives |
| `hnm_conf_thresh` | `0.3` | Confidence threshold for hard-negative weighting |
| `background_cls` | `-1` | Class id for background; `-1` = auto from name `background` |
| `background_skip_box` | `true` | Skip CIoU box loss for background boxes |

### Other presets

| File | Use case |
|------|----------|
| `hyp.scratch-low.yaml` | Default low-augmentation training |
| `hyp.scratch-med.yaml` | Medium augmentation |
| `hyp.scratch-high.yaml` | High augmentation |
