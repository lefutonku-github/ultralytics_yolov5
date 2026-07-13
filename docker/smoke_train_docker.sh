#!/usr/bin/env bash
# Single-GPU train.py smoke test inside navfm_zodc_yv5 docker image.
set -euo pipefail

DOCKER_DIR="$(cd "$(dirname "$0")" && pwd)"
SMOKE_ROOT="${DOCKER_DIR}/smoke_data"
RUNS_DIR="${DOCKER_DIR}/smoke_runs_docker"
DATASET_SRC="/mnt/wsl/data2_jd4t/Material/11.spiece_recognition/01.dataset/detection_fused/mylts_birdanimal_det_master-released_versions/mylts_birdanimal_det_master-gitver_v0.0.7_trainval_full"
PRETRAINED="/mnt/wsl/data1_jd4t/Material/11.spiece_recognition/11.models/11.training/07.active_det_birdanimal/20241108_075457-yolov5s1280-fulltrain_add_birdanimal_det_from_web-mylts_birdanimal_det_master-gitver_trainval_v0.0.6_full-20241107_073938-yolov5s1280_prew/weights/best.pt"
IMAGE_TAG="${IMAGE_TAG:-registry.cn-shanghai.aliyuncs.com/xlab-inat/xlab-inat:liuw7-navfm_zodc_yv5-v0.0.1-local}"

bash "${DOCKER_DIR}/prepare_smoke_data.sh" "${DATASET_SRC}"

mkdir -p "${RUNS_DIR}" "${DOCKER_DIR}/.ultralytics_config"
if [[ ! -f "${DOCKER_DIR}/.ultralytics_config/Arial.ttf" ]]; then
  curl -fsSL -o "${DOCKER_DIR}/.ultralytics_config/Arial.ttf" "https://ultralytics.com/assets/Arial.ttf"
fi
sed 's|^path:.*|path: /smoke_data|' "${SMOKE_ROOT}/dataset.yaml" > "${SMOKE_ROOT}/dataset.docker.yaml"

docker run --rm --gpus all --shm-size=4g \
  -e WANDB_MODE=disabled \
  -e WANDB_DISABLED=true \
  -e YOLOv5_AUTOINSTALL=false \
  -v "${SMOKE_ROOT}:/smoke_data:ro" \
  -v "${PRETRAINED}:/weights/best.pt:ro" \
  -v "${RUNS_DIR}:/runs" \
  -v "${DOCKER_DIR}/.ultralytics_config:/root/.config/Ultralytics" \
  "${IMAGE_TAG}" \
  python /usr/local/lib/python3.12/dist-packages/train.py \
    --weights /weights/best.pt \
    --data /smoke_data/dataset.docker.yaml \
    --workers 2 \
    --imgsz 1280 \
    --epochs 1 \
    --device 0 \
    --batch-size 4 \
    --project /runs \
    --name smoke_gitver_v0.0.7 \
    --save-period 1 \
    --exist-ok

echo "Docker smoke train finished. Runs: ${RUNS_DIR}/smoke_gitver_v0.0.7"
