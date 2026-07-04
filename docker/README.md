# navfm_zodc_yv5 训练 Docker 镜像

扶摇/本地构建 **navfm_zodc_yv5** 产品线训练镜像（内含 YOLOv5 fork，通过 **正式 `pip install .`** 安装，非 editable），并预装 tensorboard、ossutil；**不安装 wandb**。

- **镜像产品线名**：`navfm_zodc_yv5`（IMAGE_TAG、Dockerfile 文件名）
- **源码仓库目录**：仍为 `ultralytics_yolov5/`（不改目录名）

构建 context 为 **`ultralytics_yolov5` 仓库根目录**（本目录的上一级）。

## 两个入口脚本（职责分离）

| 脚本 | 用途 | 命令 |
|------|------|------|
| [`build_local.sh`](build_local.sh) | **本地** `docker build`（PyTorch 2.11.0 镜像 + `--build-arg BASE_IMAGE`） | `bash docker/build_local.sh` |
| [`push_fuyao.sh`](push_fuyao.sh) | **扶摇远程** `fuyao docker --push`（Dockerfile 默认 fuyao-image-convert） | `bash docker/push_fuyao.sh` |

二者互不调用。构建前若缺少 ossutil zip，各自独立下载。

## 基础镜像

| 场景 | `BASE_IMAGE` |
|------|----------------|
| 扶摇 push（Dockerfile 默认） | `infra-registry.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao-image-convert:pytorch-2.11.0-cuda12.8-cudnn9-runtime` |
| 本地 build（`build_local.sh` 传入） | `ywvk8934o3f50v3q9i.xuanyuan.run/pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime` |

默认 IMAGE_TAG：

| 场景 | 默认 tag |
|------|----------|
| 扶摇 push | `navfm_zodc_yv5-v0.0.1` |
| 本地 build | `navfm_zodc_yv5:local-v0.0.1` |

## 与 vfm_apple 扶摇构建的差异

| 项目 | vfm_apple（yolo26） | navfm_zodc_yv5 |
|------|---------------------|----------------|
| 典型 cwd | `species_recognition/docker/` | **`ultralytics_yolov5/` 仓库根** |
| 是否 COPY 业务源码 | 否（仅 pip 装 ultralytics） | **是**（`COPY . /tmp/navfm_zodc_yv5_src` → `pip install .`） |
| 扶摇命令 cwd | `cd docker` | **`cd ultralytics_yolov5`**（`push_fuyao.sh` 会自动 cd） |

扶摇 `fuyao docker --push` 会将 **cwd 下文件** 打成压缩包上传云端构建。YOLOv5 fork 源码必须在该压缩包内，故 cwd 必须是仓库根，**不能**只 `cd docker/`。

构建时源码处理：`COPY . /tmp/navfm_zodc_yv5_src` → `install_yolov5.sh` 正式安装 → 删除临时目录。训练 Job 使用镜像内 site-packages 中的 YOLOv5，不依赖 Job 再打包一份源码。

## 快速开始

### 1. 依赖预检（可选）

```bash
cd ultralytics/ultralytics_yolov5
bash docker/check_deps.sh
```

### 2. 本地构建

```bash
cd ultralytics/ultralytics_yolov5
bash docker/build_local.sh
# IMAGE_TAG=navfm_zodc_yv5:my-dev bash docker/build_local.sh
```

### 3. 扶摇推送

```bash
cd ultralytics/ultralytics_yolov5
bash docker/push_fuyao.sh
# IMAGE_TAG=navfm_zodc_yv5-v0.0.2 bash docker/push_fuyao.sh
```

等价于（需已在仓库根目录）：

```bash
fuyao docker --push \
  --dockerfile="./docker/fuyao_navfm_zodc_yv5_v0.0.1.dockerfile" \
  --image-tag "navfm_zodc_yv5-v0.0.1"
```

扶摇参数说明：[扶摇自定义镜像文档](https://xiaopeng.feishu.cn/wiki/EaCBwUWfKiLN2QkU3hxcsMHRn0f)

### 4. 构建前准备

`docker/assets/ossutil-2.3.0-linux-amd64.zip` 须存在（脚本可自动 curl；该文件 gitignore 不入库）。扶摇侧通常无法访问外网，**推送前务必在本地准备好**。

## 目录说明

| 文件 | 说明 |
|------|------|
| [`fuyao_navfm_zodc_yv5_v0.0.1.dockerfile`](fuyao_navfm_zodc_yv5_v0.0.1.dockerfile) | 主 Dockerfile |
| [`build_local.sh`](build_local.sh) | 本地 docker build |
| [`push_fuyao.sh`](push_fuyao.sh) | 扶摇 `fuyao docker --push` |
| [`install_yolov5.sh`](install_yolov5.sh) | 镜像内 uv 安装逻辑 |
| [`check_deps.sh`](check_deps.sh) | uv dry-run 依赖检查 |
| [`constraints-base.txt`](constraints-base.txt) | 版本约束 |
| [`prepare_smoke_data.sh`](prepare_smoke_data.sh) / [`smoke_train.sh`](smoke_train.sh) | 宿主机冒烟 |

## 镜像内容

- YOLOv5：`pip install .`（site-packages + `data/hyps` 拷贝）
- ultralytics（`--no-deps`）+ `opencv-python-headless`
- tensorboard；**不含 wandb**
- ossutil 2.0
- `YOLOv5_AUTOINSTALL=false`、`WANDB_MODE=disabled`

## 冒烟测试（宿主机 conda）

```bash
conda activate xlab_inat
cd ultralytics/ultralytics_yolov5
YOLOV5_SRC="$(pwd)" bash docker/install_yolov5.sh
bash docker/smoke_train.sh
```

## 多卡训练

```bash
python -m torch.distributed.run --nproc_per_node 4 --master_port 29500 \
  train.py --data /path/to/dataset.yaml --weights yolov5s6.pt \
  --img 1280 --batch 128 --epochs 300 --device 0,1,2,3
```

## 常见问题

**Q: 本地构建忘了传 BASE_IMAGE？**  
A: 只用 `bash docker/build_local.sh`，已内置与扶摇 fuyao-image-convert 同版本的 PyTorch 2.11.0 本地镜像 URL。

**Q: 扶摇 push 要在哪个目录？**  
A: `ultralytics_yolov5` 仓库根；用 `bash docker/push_fuyao.sh` 即可。

**Q: Job 里还要带 yolov5 源码吗？**  
A: 不需要。镜像构建时已 `pip install .` 进 site-packages。

**Q: 镜像名 navfm_zodc_yv5 与目录 ultralytics_yolov5 的关系？**  
A: 前者是扶摇/registry 产品线命名；后者是本地 git 仓库目录，互不影响安装与训练。
