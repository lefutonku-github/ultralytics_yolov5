#!/usr/bin/env bash
# Prepare a small YOLOv5 smoke dataset from gitver_v0.0.7_trainval_full (val split sample).
set -euo pipefail

SRC_ROOT="${1:-/mnt/wsl/data2_jd4t/Material/11.spiece_recognition/01.dataset/detection_fused/mylts_birdanimal_det_master-released_versions/mylts_birdanimal_det_master-gitver_v0.0.7_trainval_full}"
OUT_ROOT="$(cd "$(dirname "$0")" && pwd)/smoke_data"
TRAIN_N="${TRAIN_N:-32}"
VAL_N="${VAL_N:-16}"

rm -rf "${OUT_ROOT}"
mkdir -p "${OUT_ROOT}/images/train" "${OUT_ROOT}/images/val" \
         "${OUT_ROOT}/labels/train" "${OUT_ROOT}/labels/val"

mapfile -t TRAIN_IMGS < <(find "${SRC_ROOT}/images/train" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | head -n "${TRAIN_N}")
mapfile -t VAL_IMGS < <(find "${SRC_ROOT}/images/val" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | head -n "${VAL_N}")

copy_pair() {
  local img="$1" split="$2"
  local base rel label
  base="$(basename "${img}")"
  rel="${split}/${base}"
  label="${SRC_ROOT}/labels/${split}/${base%.*}.txt"
  cp "${img}" "${OUT_ROOT}/images/${rel}"
  if [[ -f "${label}" ]]; then
    cp "${label}" "${OUT_ROOT}/labels/${rel%.*}.txt"
  else
    : > "${OUT_ROOT}/labels/${rel%.*}.txt"
  fi
}

for img in "${TRAIN_IMGS[@]}"; do copy_pair "${img}" train; done
for img in "${VAL_IMGS[@]}"; do copy_pair "${img}" val; done

cp "${SRC_ROOT}/dataset.yaml" "${OUT_ROOT}/dataset.yaml"
sed -i "s|^path:.*|path: ${OUT_ROOT}|" "${OUT_ROOT}/dataset.yaml"

echo "Smoke dataset ready: ${OUT_ROOT} (train=${#TRAIN_IMGS[@]}, val=${#VAL_IMGS[@]})"
