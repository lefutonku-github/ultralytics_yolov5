#!/usr/bin/env bash
# Resolve YOLOv5 training deps with uv --dry-run (skip pip dry-run unless uv unavailable).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DOCKER_DIR="$(cd "$(dirname "$0")" && pwd)"
ALIYUN_PYPI="-i https://mirrors.aliyun.com/pypi/simple"
YOLOV5_SRC="${YOLOV5_SRC:-${ROOT}}"

pip install -q uv ${ALIYUN_PYPI} 2>/dev/null || pip install -q uv ${ALIYUN_PYPI}

TORCH_VER="$(python -c 'import torch; print(torch.__version__)')"
TV_VER="$(python -c 'import torchvision; print(torchvision.__version__)')"
CONSTRAINTS="$(mktemp)"
cat "${DOCKER_DIR}/constraints-base.txt" > "${CONSTRAINTS}"

echo "==> uv dry-run (keep installed torch=${TORCH_VER}, torchvision=${TV_VER})"

DEPS=(
  "matplotlib>=3.3.0"
  "numpy>=1.23.5,<2"
  "opencv-python-headless>=4.6,<4.10"
  "pillow>=7.1.2"
  "pyyaml>=5.3.1"
  "requests>=2.23.0"
  "scipy>=1.4.1"
  "tqdm>=4.64.0"
  "psutil"
  "py-cpuinfo"
  "thop>=0.1.1"
  "gitpython>=3.1.30"
  "pandas>=1.1.4"
  "seaborn>=0.11.0"
  "albumentations>=1.0.3"
  "pycocotools>=2.0.6"
  "tensorboard>=2.13.0"
  "setuptools>=65.5.1,<81"
)

set +e
uv pip install --system ${ALIYUN_PYPI} \
  --dry-run \
  --no-upgrade \
  --constraint "${CONSTRAINTS}" \
  "${DEPS[@]}"
deps_rc=$?
uv pip install --system ${ALIYUN_PYPI} \
  --dry-run \
  --no-upgrade \
  --no-deps \
  "ultralytics>=8.0.232,<8.5"
ultra_rc=$?
uv pip install --system ${ALIYUN_PYPI} \
  --dry-run \
  --no-upgrade \
  --constraint "${CONSTRAINTS}" \
  --no-deps \
  "${YOLOV5_SRC}"
pkg_rc=$?
set -e
rm -f "${CONSTRAINTS}"

if [[ ${deps_rc} -ne 0 || ${ultra_rc} -ne 0 || ${pkg_rc} -ne 0 ]]; then
  echo "ERROR: uv dependency resolution failed (deps=${deps_rc}, ultralytics=${ultra_rc}, pkg=${pkg_rc})" >&2
  exit 1
fi
echo "==> uv dry-run OK"
