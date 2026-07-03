#!/usr/bin/env bash
# Fuyao remote build & push via `fuyao docker --push`.
# Uploads cwd (ultralytics_yolov5 repo root) as build context; Dockerfile default BASE_IMAGE is fuyao-image-convert.
#
# Usage (from anywhere):
#   bash docker/push_fuyao.sh
#   IMAGE_TAG=navfm_zodc_yv5-v0.0.2 bash docker/push_fuyao.sh
set -euo pipefail

DOCKER_DIR="$(cd "$(dirname "$0")" && pwd)"
YOLOV5_ROOT="$(cd "${DOCKER_DIR}/.." && pwd)"
DOCKERFILE="${DOCKER_DIR}/fuyao_navfm_zodc_yv5_v0.0.1.dockerfile"
OSSUTIL_ZIP="${DOCKER_DIR}/assets/ossutil-2.3.0-linux-amd64.zip"
OSSUTIL_URL="https://gosspublic.alicdn.com/ossutil/v2/2.3.0/ossutil-2.3.0-linux-amd64.zip"
IMAGE_TAG="${IMAGE_TAG:-navfm_zodc_yv5-v0.0.1}"
FUYAO_WIKI="https://xiaopeng.feishu.cn/wiki/EaCBwUWfKiLN2QkU3hxcsMHRn0f"

if [[ ! -f "${YOLOV5_ROOT}/train.py" ]]; then
  echo "ERROR: expected YOLOv5 repo root at ${YOLOV5_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${OSSUTIL_ZIP}" ]]; then
  echo "==> Downloading ossutil to ${OSSUTIL_ZIP}"
  curl -fsSL -o "${OSSUTIL_ZIP}" "${OSSUTIL_URL}"
fi

cd "${YOLOV5_ROOT}"

echo "==> Fuyao docker --push"
echo "==> cwd (build context): ${YOLOV5_ROOT}"
echo "==> dockerfile:          ./docker/fuyao_navfm_zodc_yv5_v0.0.1.dockerfile"
echo "==> image-tag:           ${IMAGE_TAG}"
echo "==> BASE_IMAGE:          Dockerfile default (fuyao-image-convert)"
echo "==> docs:                ${FUYAO_WIKI}"

fuyao docker --push \
  --dockerfile="./docker/fuyao_navfm_zodc_yv5_v0.0.1.dockerfile" \
  --image-tag "${IMAGE_TAG}"

echo "==> Fuyao push submitted: ${IMAGE_TAG}"
