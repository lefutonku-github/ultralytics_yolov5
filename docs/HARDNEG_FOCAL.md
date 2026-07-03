# Hard negative mining + Focal Loss (xlab_inat / YOLOv5)

This fork extends stock YOLOv5 with **Focal Loss**, **hard negative mining (HNM)** on objectness, and an optional **`background` training class** to reduce false positives on confusing textures (rocks, leaves, shadows, etc.).

## Quick start

### 1. Dataset

Add `background` to your `data.yaml` (example: last class index):

```yaml
nc: 4
names:
  0: bird
  1: mammal
  2: reptile
  3: background
```

Annotate confusing regions with boxes labeled `background`. Mix ~2k–5k background images/boxes with normal target data.

### 2. Train

**Single GPU:**

```bash
python train.py \
  --hyp data/hyps/hyp.hardneg-focal.yaml \
  --data your_data.yaml \
  --weights yolov5s.pt \
  --img 640 \
  --epochs 100
```

**Multi-GPU DDP (single machine):**

```bash
python -m torch.distributed.run --nproc_per_node 4 --master_port 29500 \
  train.py \
  --hyp data/hyps/hyp.hardneg-focal.yaml \
  --data your_data.yaml \
  --weights yolov5s.pt \
  --img 640 \
  --batch-size 64 \
  --device 0,1,2,3
```

- `--batch-size` is the **global** batch; each GPU gets `batch_size / nproc`.
- `batch_size` must divide evenly by GPU count.
- Do **not** use `--image-weights`, `--evolve`, or `--batch-size -1` with DDP.

### 3. Inference

`detect.py` **automatically drops** the `background` class when it appears in model `names`:

```bash
python detect.py --weights runs/train/exp/weights/best.pt --source images/ --data your_data.yaml
```

To keep background detections (debug only):

```bash
python detect.py ... --no-exclude-background
```

**xlab_inat** `Yolov5DetectionBackend.postprocess()` filters background by default (`exclude_background=True`).

## What changed in code

| Area | File | Behavior |
|------|------|----------|
| Hyp preset | `data/hyps/hyp.hardneg-focal.yaml` | `fl_gamma`, HNM knobs, background options |
| Loss | `utils/loss.py` | `_obj_loss()` HNM; optional skip CIoU for background |
| Inference | `utils/general.py` | `resolve_background_cls`, `filter_excluded_class_predictions` |
| detect CLI | `detect.py` | Auto-filter background after NMS |
| xlab backend | `xlab_inat/.../det_backend.py` | Same filter in `postprocess` |

## Hyperparameters (`hyp.hardneg-focal.yaml`)

| Key | Default | Notes |
|-----|---------|-------|
| `fl_gamma` | `2.0` | Focal γ; try `2.5` if FP remain high |
| `fl_alpha` | `0.25` | Focal α |
| `obj` | `1.1` | Objectness gain (still scaled in `train.py`) |
| `hnm_enabled` | `true` | Toggle HNM |
| `neg_pos_ratio` | `4.5` | Max negatives per positive (per GPU batch) |
| `hnm_topk_ratio` | `0.3` | Top fraction of negatives by obj confidence |
| `hnm_hard_weight` | `3.0` | Extra weight for conf > `hnm_conf_thresh` |
| `hnm_conf_thresh` | `0.3` | Hard-negative confidence threshold |
| `background_cls` | `-1` | Explicit id, or auto-detect name `background` |
| `background_skip_box` | `true` | Train cls+obj only on background boxes |

Stock `hyp.scratch-*.yaml` files are unchanged (`hnm_enabled` defaults to `false`).

## DDP compatibility

Phase 1 (Focal) and Phase 2 (HNM) are **DDP-safe**:

- HNM runs on each rank’s local batch (no cross-GPU sync required).
- Loss remains a scalar; existing `loss *= WORLD_SIZE` in `train.py` is unchanged.
- Use deterministic `topk` (no extra RNG).

## Tuning tips

1. Start with the preset yaml; compare FP/image on a fixed val set.
2. If recall drops, lower `hnm_hard_weight` or raise `hnm_topk_ratio` toward `0.5`.
3. If FP persist, raise `fl_gamma` to `2.5` and add more diverse background boxes.
4. Use FiftyOne mistakenness runs to mine high-confidence FPs → label as `background`.

## Related docs

- [data/hyps/README.md](data/hyps/README.md) — hyp file index
- xlab FiftyOne brain: `xlab_inat/plugins/fiftyone/brain/README.md`
