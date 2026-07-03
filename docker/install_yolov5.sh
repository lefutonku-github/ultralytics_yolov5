#!/usr/bin/env bash
# Install YOLOv5 training stack into system Python (non-editable pip install .).
set -euo pipefail

YOLOV5_SRC="${YOLOV5_SRC:-/tmp/yolov5}"
DOCKER_DIR="${DOCKER_DIR:-$(cd "$(dirname "$0")" && pwd)}"
ALIYUN_PYPI="-i https://mirrors.aliyun.com/pypi/simple"

pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip install -q --upgrade pip uv ${ALIYUN_PYPI}

TORCH_VER="$(python -c 'import torch; print(torch.__version__)')"
TV_VER="$(python -c 'import torchvision; print(torchvision.__version__)')"
CONSTRAINTS="$(mktemp)"
cat "${DOCKER_DIR}/constraints-base.txt" > "${CONSTRAINTS}"

echo "==> Keep installed torch==${TORCH_VER}, torchvision==${TV_VER} (--no-upgrade)"
pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true

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
  "pandas>=1.1.4"
  "seaborn>=0.11.0"
  "gitpython>=3.1.30"
  "albumentations>=1.0.3"
  "pycocotools>=2.0.6"
  "tensorboard>=2.13.0"
  "setuptools>=65.5.1,<81"
)

# ultralytics pulls opencv-python; install without deps then add headless separately.
uv pip install --system ${ALIYUN_PYPI} --no-upgrade --constraint "${CONSTRAINTS}" "${DEPS[@]}"
uv pip install --system ${ALIYUN_PYPI} --no-upgrade --no-deps "ultralytics>=8.0.232,<8.5"

pushd "${YOLOV5_SRC}" >/dev/null
uv pip install --system ${ALIYUN_PYPI} --no-upgrade --constraint "${CONSTRAINTS}" --no-deps .
popd >/dev/null

# Flat-layout pip install does not ship data/hyps; copy runtime assets into site-packages.
SITE_PKGS="$(python -c 'import site; print(site.getsitepackages()[0])')"
cp -a "${YOLOV5_SRC}/data" "${SITE_PKGS}/"
sed 's/opencv-python/opencv-python-headless/g' "${YOLOV5_SRC}/requirements.txt" > "${SITE_PKGS}/requirements.txt"

pip uninstall -y opencv-python 2>/dev/null || true
pip install ${ALIYUN_PYPI} "opencv-python-headless>=4.6,<4.10"

rm -f "${CONSTRAINTS}"

python -c "import torch, torchvision; assert torch.__version__=='${TORCH_VER}', f'torch changed: {torch.__version__}'; assert torchvision.__version__=='${TV_VER}', f'torchvision changed: {torchvision.__version__}'; print('torch/torchvision unchanged OK')"

python --version
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
python -c "import cv2, numpy; print('cv2', cv2.__version__, 'np', numpy.__version__)"
python -c "from ultralytics.utils.plotting import Annotator; print('ultralytics OK')"
python -c "from torch.utils.tensorboard import SummaryWriter; print('tensorboard OK')"
python -c "import importlib.util; assert importlib.util.find_spec('wandb') is None; print('wandb not installed OK')"
python -c "import train; from models.common import DetectMultiBackend; print('yolov5 package OK', train.__file__)"
