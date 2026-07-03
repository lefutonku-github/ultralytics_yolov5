#!/usr/bin/env bash
# Local docker build only (NVCR mirror base image).
#
# Usage (from anywhere):
#   bash docker/build.sh
#   IMAGE_TAG=yolov5-train:my-dev bash docker/build.sh
set -euo pipefail

DOCKER_DIR="$(cd "$(dirname "$0")" && pwd)"
YOLOV5_ROOT="$(cd "${DOCKER_DIR}/.." && pwd)"

BASE_IMAGE_LOCAL="ywvk8934o3f50v3q9i-nvcr.xuanyuan.run/nvidia/pytorch:25.02-py3"
DOCKERFILE="${DOCKER_DIR}/fuyao_yolov5_v0.0.1.dockerfile"
OSSUTIL_ZIP="${DOCKER_DIR}/assets/ossutil-2.3.0-linux-amd64.zip"
OSSUTIL_URL="https://gosspublic.alicdn.com/ossutil/v2/2.3.0/ossutil-2.3.0-linux-amd64.zip"
IMAGE_TAG="${IMAGE_TAG:-yolov5-train:local-v0.0.1}"

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
echo "==> Fuyao remote push: bash docker/push_fuyao.sh"
