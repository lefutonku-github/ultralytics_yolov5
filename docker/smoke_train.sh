#!/usr/bin/env bash
# Single-GPU train.py smoke test (params aligned with det_train_birdanimal_master notebook).
set -euo pipefail

DOCKER_DIR="$(cd "$(dirname "$0")" && pwd)"
YOLOV5_ROOT="$(cd "${DOCKER_DIR}/.." && pwd)"
SMOKE_ROOT="${DOCKER_DIR}/smoke_data"
RUNS_DIR="${DOCKER_DIR}/smoke_runs"

DATASET_SRC="/mnt/wsl/data2_jd4t/Material/11.spiece_recognition/01.dataset/detection_fused/mylts_birdanimal_det_master-released_versions/mylts_birdanimal_det_master-gitver_v0.0.7_trainval_full"
PRETRAINED="/mnt/wsl/data1_jd4t/Material/11.spiece_recognition/11.models/11.training/07.active_det_birdanimal/20241108_075457-yolov5s1280-fulltrain_add_birdanimal_det_from_web-mylts_birdanimal_det_master-gitver_trainval_v0.0.6_full-20241107_073938-yolov5s1280_prew/weights/best.pt"

bash "${DOCKER_DIR}/prepare_smoke_data.sh" "${DATASET_SRC}"

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export YOLOv5_AUTOINSTALL=false

mkdir -p "${RUNS_DIR}"
cd /tmp

python "$(python -c "import train, os; print(os.path.join(os.path.dirname(train.__file__), 'train.py'))")" \
  --weights "${PRETRAINED}" \
  --data "${SMOKE_ROOT}/dataset.yaml" \
  --workers 4 \
  --imgsz 1280 \
  --epochs 1 \
  --device 0 \
  --batch-size 4 \
  --project "${RUNS_DIR}" \
  --name smoke_gitver_v0.0.7 \
  --save-period 1 \
  --exist-ok

echo "Smoke train finished. Runs: ${RUNS_DIR}/smoke_gitver_v0.0.7"
