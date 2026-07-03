# YOLOv5 training image — see docker/README.md
#
# BASE_IMAGE (see docker/build.sh vs push_fuyao.sh):
#   Fuyao (Dockerfile default): infra-registry.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao-image-convert:pytorch-2.11.0-cuda12.8-cudnn9-runtime
#   Local build only:           ywvk8934o3f50v3q9i-nvcr.xuanyuan.run/nvidia/pytorch:25.02-py3  (--build-arg via docker/build.sh)
#
# Local docker build:
#   bash docker/build.sh
#
# Fuyao remote build & push (cwd must be repo root; script handles cd):
#   bash docker/push_fuyao.sh

ARG BASE_IMAGE=infra-registry.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao-image-convert:pytorch-2.11.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV MAX_JOBS=1
ENV TZ=Asia/Shanghai
ENV HF_ENDPOINT=https://hf-mirror.com
ENV NVIDIA_DRIVER_CAPABILITIES=all

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    MKL_THREADING_LAYER=GNU \
    OMP_NUM_THREADS=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TORCH_CPP_LOG_LEVEL=ERROR \
    WANDB_MODE=disabled \
    WANDB_DISABLED=true \
    YOLOv5_AUTOINSTALL=false

COPY docker/assets/ossutil-2.3.0-linux-amd64.zip /tmp/ossutil.zip
COPY docker/constraints-base.txt docker/install_yolov5.sh /tmp/docker/
COPY . /tmp/yolov5

RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone \
    && (test -f /etc/apt/sources.list.d/ubuntu.sources && sed -i 's|archive.ubuntu.com|mirrors.aliyun.com|g' /etc/apt/sources.list.d/ubuntu.sources && sed -i 's|security.ubuntu.com|mirrors.aliyun.com|g' /etc/apt/sources.list.d/ubuntu.sources || true) \
    && (test -f /etc/apt/sources.list && sed -i 's|archive.ubuntu.com|mirrors.aliyun.com|g' /etc/apt/sources.list && sed -i 's|security.ubuntu.com|mirrors.aliyun.com|g' /etc/apt/sources.list || true) \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        gcc git zip unzip wget curl htop \
        libgl1 libglib2.0-0 libsm6 gnupg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && unzip -q /tmp/ossutil.zip -d /tmp/ossutil \
    && install -m 755 /tmp/ossutil/ossutil-2.3.0-linux-amd64/ossutil /usr/local/bin/ossutil \
    && rm -rf /tmp/ossutil /tmp/ossutil.zip \
    && chmod +x /tmp/docker/install_yolov5.sh \
    && YOLOV5_SRC=/tmp/yolov5 DOCKER_DIR=/tmp/docker bash /tmp/docker/install_yolov5.sh \
    && rm -rf /tmp/yolov5 /tmp/docker

WORKDIR /workspace

# ---- Usage Example:
# NOTE: cd to ultralytics_yolov5 repository root first (NOT docker/ subdir alone)
# # fuyao build using specified dockerfile; other options see:
# # https://xiaopeng.feishu.cn/wiki/EaCBwUWfKiLN2QkU3hxcsMHRn0f
# bash docker/push_fuyao.sh
# # or: fuyao docker --push --dockerfile="./docker/fuyao_yolov5_v0.0.1.dockerfile" --image-tag "yolov5_train-v0.0.1"
#
# Local build: bash docker/build.sh
