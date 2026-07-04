#!/usr/bin/env bash
# Local docker build only (PyTorch 2.11.0 mirror, same stack as Fuyao fuyao-image-convert).
#
# Usage (from anywhere):
#   bash docker/build_local.sh
#   IMAGE_TAG=navfm_zodc_yv5:my-dev bash docker/build_local.sh
set -euo pipefail

DOCKER_DIR="$(cd "$(dirname "$0")" && pwd)"
YOLOV5_ROOT="$(cd "${DOCKER_DIR}/.." && pwd)"

BASE_IMAGE_LOCAL="ywvk8934o3f50v3q9i.xuanyuan.run/pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime"
DOCKERFILE="${DOCKER_DIR}/fuyao_navfm_zodc_yv5_v0.0.1.dockerfile"
OSSUTIL_ZIP="${DOCKER_DIR}/assets/ossutil-2.3.0-linux-amd64.zip"
OSSUTIL_URL="https://gosspublic.alicdn.com/ossutil/v2/2.3.0/ossutil-2.3.0-linux-amd64.zip"
IMAGE_TAG="${IMAGE_TAG:-navfm_zodc_yv5:local-v0.0.1}"

if [[ ! -f "${YOLOV5_ROOT}/train.py" ]]; then
  echo "ERROR: expected YOLOv5 repo root at ${YOLOV5_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${OSSUTIL_ZIP}" ]]; then
  echo "==> Downloading ossutil to ${OSSUTIL_ZIP}"
  curl -fsSL -o "${OSSUTIL_ZIP}" "${OSSUTIL_URL}"
fi

echo "==> Local docker build"
echo "==> BASE_IMAGE: ${BASE_IMAGE_LOCAL}"
echo "==> IMAGE_TAG:  ${IMAGE_TAG}"
echo "==> Context:    ${YOLOV5_ROOT}"

docker build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE_LOCAL}" \
  -f "${DOCKERFILE}" \
  -t "${IMAGE_TAG}" \
  "${YOLOV5_ROOT}"

echo "==> Build OK: ${IMAGE_TAG}"
echo "==> Next (optional, separate): push to Fuyao with  bash docker/push_fuyao.sh"
echo "==> Next (optional, local smoke): bash docker/smoke_train_docker.sh"
