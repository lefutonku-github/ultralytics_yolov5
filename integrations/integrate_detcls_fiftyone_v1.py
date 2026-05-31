"""
DEPRECATED: Moved to xlab_inat_lefutonku as
``xlab_inat.integrations.legacy.integrate_detcls_fiftyone_v1``.
Use ``xlab_inat.integrations.detcls_v2`` for new work.

integrate_detcls_fiftyone.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unified detection + classification pipeline for an ImageFolder-style
input directory.

Architecture
------------
Operators (atomic, easy to extract independently)
  load_image          : read one BGR image from disk
  preprocess_det      : letterbox-resize for detection model
  preprocess_cls      : centre-crop + normalise for classification model
  infer_det           : detection forward + NMS → raw boxes (xyxy, conf, cls)
  infer_cls_on_crops  : crop each detection box → classify each crop
  postprocess_det     : scale boxes back to original image size
  convert_labels      : class-id → string (with optional remapping)

Wrappers
  DetectionModelWrapper      : reused from integrate_det_fiftyone_v2.py style
  ClassificationModelWrapper : reused from integrate_cls_fiftyone.py style
  ImageFolderBatchLoader     : yields (tensor_batch, paths, ori_shapes) from a dir

Pipeline entry-points
  run_detcls          : full det→cls pipeline on an image folder
  main / create_argparser : CLI entry
"""

# ---------------------------------------------------------------
# ---- imports

## ---- stdlib
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

## ---- 3rd-party
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

## ---- ultralytics yolov5
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_boxes, xyxy2xywh

## ---- local utils sub-package
from .detcls_utils import (
    glob_images,
    load_label_mappings_from_csv,
    DetectionTransform,
    ClassificationTransform,
    draw_detcls_result,
)

# ---------------------------------------------------------------
# ---- module logger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ===============================================================
# ---- SECTION 1: Model wrappers
# ===============================================================

class DetectionModelWrapper(nn.Module):
    """Thin wrapper around DetectMultiBackend for detection inference.

    Exposes:
        ``predict(x, conf_thres, iou_thres, ori_imshapes)``
            → list of ``torch.Tensor`` shaped (N, 6) per image,
              columns: [x1, y1, x2, y2, conf, cls_id] in original pixel coords.
    """

    def __init__(self, weights: str, device: str = "", label_mappings: dict = None):
        super().__init__()
        self._device = select_device(device)
        self._model  = DetectMultiBackend(weights, device=self._device)
        self._model.eval()
        self._label_mappings = label_mappings
        
    # ----------------------------------------------------------
    @property
    def stride(self) -> int:
        return int(self._model.stride)

    @property
    def names(self) -> dict:
        return self._model.names

    # ----------------------------------------------------------
    def warmup(self, imgsz: tuple = (1, 3, 640, 640)):
        self._model.warmup(imgsz)

    # ----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._model(x.to(self._device))

    # ----------------------------------------------------------
    def predict(
        self,
        x: torch.Tensor,
        conf_thres: float = 0.25,
        iou_thres: float  = 0.45,
        ori_imshapes: Optional[list] = None,
    ) -> List[torch.Tensor]:
        """Run forward + NMS; scale boxes back to original image coordinates.

        Args:
            x:            Batched float tensor (B, C, H, W) in [0, 1].
            conf_thres:   NMS confidence threshold.
            iou_thres:    NMS IoU threshold.
            ori_imshapes: List of (H, W, C) shapes of the *original* images.
                          If None, no scaling is applied.

        Returns:
            List of tensors, one per image.  Each tensor has shape (N, 6):
            ``[x1, y1, x2, y2, conf, cls_id]`` in *original-image* pixel coords.
            Empty-detection images return a tensor of shape (0, 6).
        """
        raw = self.forward(x)
        preds = non_max_suppression(raw, conf_thres, iou_thres)

        if ori_imshapes is not None:
            for i, (pred, ori_shape) in enumerate(zip(preds, ori_imshapes)):
                if pred is not None and pred.shape[0] > 0:
                    pred[:, :4] = scale_boxes(
                        x.shape[2:], pred[:, :4], ori_shape
                    ).round()
                    preds[i] = pred

        return preds


# ---------------------------------------------------------------

class ClassificationModelWrapper(nn.Module):
    """Thin wrapper around DetectMultiBackend for classification inference.

    Exposes:
        ``predict_crops(crops_bgr)``
            → list of ``(remapped_label, conf, ori_label)`` tuples, one per crop.
    """

    def __init__(self, weights: str, device: str = "", label_mappings: dict = None):
        super().__init__()
        self._device = select_device(device)
        self._model  = DetectMultiBackend(weights, device=self._device)
        self._model.eval()
        self._label_mappings = label_mappings
        self._tfm = ClassificationTransform()  # default 224

    # ----------------------------------------------------------
    @property
    def names(self) -> dict:
        return self._model.names

    # ----------------------------------------------------------
    def warmup(self, imgsz: tuple = (1, 3, 224, 224)):
        self._model.warmup(imgsz)

    # ----------------------------------------------------------
    def set_img_size(self, img_size: int):
        """Re-initialise the classification transform for a different size."""
        self._tfm = ClassificationTransform(img_size)

    # ----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._model(x.to(self._device))

    # ----------------------------------------------------------
    def predict_crops(
        self,
        crops_bgr: List[np.ndarray],
        topk: int = 1,
        label_mappings: dict = None,
    ) -> List[Tuple]:
        """Classify a list of BGR crop images.

        Args:
            crops_bgr:      List of BGR ``np.ndarray`` crops (variable sizes).
            topk:           Number of top-K predictions to return per crop.
            label_mappings: Override the instance-level label mapping.

        Returns:
            List of length ``len(crops_bgr)``.  Each element is a list of
            ``topk`` tuples ``(remapped_label, conf, ori_label)``.
            Empty crops return ``[("?", 0.0, "?")]``.
        """
        lm = label_mappings if label_mappings is not None else self._label_mappings
        results = []

        for crop in crops_bgr:
            if crop is None or crop.size == 0:
                results.append([("?", 0.0, "?")] * topk)
                continue

            ##>>>> preprocess single crop
            im_t = self._tfm(crop).unsqueeze(0)  # (1, C, H, W)

            ##>>>> forward
            logits = self.forward(im_t)           # (1, num_classes)

            ##>>>> topk decode
            prob   = F.softmax(logits, dim=-1)[0]  # (num_classes,)
            topk_v = torch.topk(prob, k=min(topk, prob.shape[0]))
            cur = []
            for idx in topk_v.indices:
                conf      = float(prob[idx])
                ori_label = self._model.names[int(idx)]
                remap     = lm.get(ori_label, ori_label) if lm else ori_label
                cur.append((remap, conf, ori_label))
            results.append(cur)

        return results


# ===============================================================
# ---- SECTION 2: Image folder batch loader
# ===============================================================

class ImageFolderBatchLoader:
    """Yield batches of (tensor, paths, ori_shapes) from an image folder.

    Args:
        image_paths: list of absolute image file paths.
        batch_size:  number of images per batch.
        det_transform: callable that maps BGR np.ndarray → float32 tensor.
    """

    def __init__(
        self,
        image_paths: List[str],
        batch_size: int,
        det_transform: DetectionTransform,
    ):
        self._paths      = image_paths
        self._batch_size = batch_size
        self._tfm        = det_transform
        self._count      = 0

    def __len__(self) -> int:
        import math
        return math.ceil(len(self._paths) / self._batch_size)

    def __iter__(self):
        self._count = 0
        return self

    def __next__(self) -> Tuple[torch.Tensor, List[str], List[tuple]]:
        if self._count >= len(self._paths):
            raise StopIteration

        batch_paths = self._paths[self._count: self._count + self._batch_size]
        self._count += len(batch_paths)

        tensors, ori_shapes = [], []
        valid_paths = []
        for p in batch_paths:
            im0 = load_image(p)
            if im0 is None:
                logger.warning(f"[ImageFolderBatchLoader] cannot read: {p}")
                continue
            tensors.append(preprocess_det(im0, self._tfm))
            ori_shapes.append(im0.shape)
            valid_paths.append(p)

        if not tensors:
            return self.__next__()  # skip all-bad batch

        return torch.stack(tensors, 0), valid_paths, ori_shapes


# ===============================================================
# ---- SECTION 3: Atomic operators
# ===============================================================

def load_image(path: str) -> Optional[np.ndarray]:
    """Load a BGR image from *path*.  Returns None on failure."""
    im = cv2.imread(path)
    if im is None:
        logger.warning(f"[load_image] cannot read: {path}")
    return im


def preprocess_det(im0: np.ndarray, transform: DetectionTransform) -> torch.Tensor:
    """Apply detection pre-processing transform to one image.

    Args:
        im0:       BGR uint8 image.
        transform: ``DetectionTransform`` instance.

    Returns:
        Float32 tensor (C, H, W) in [0, 1].
    """
    return transform(im0)


def preprocess_cls(crop_bgr: np.ndarray, transform: ClassificationTransform) -> torch.Tensor:
    """Apply classification pre-processing transform to one crop.

    Args:
        crop_bgr:  BGR uint8 crop.
        transform: ``ClassificationTransform`` instance.

    Returns:
        Float32 tensor (C, H, W).
    """
    return transform(crop_bgr)


def infer_det(
    det_model: DetectionModelWrapper,
    batch: torch.Tensor,
    ori_imshapes: List[tuple],
    conf_thres: float = 0.25,
    iou_thres: float  = 0.45,
) -> List[torch.Tensor]:
    """Run detection inference on one pre-processed batch.

    Returns list of (N, 6) tensors ``[x1, y1, x2, y2, conf, cls_id]``
    in original-image pixel coordinates.
    """
    return det_model.predict(batch, conf_thres, iou_thres, ori_imshapes)


def extract_crops(
    im0: np.ndarray,
    boxes_xyxy: np.ndarray,
    pad_ratio: float = 0.02,
    pad_px: int = 10,
) -> List[np.ndarray]:
    """Crop detection regions from the original image.

    Args:
        im0:        BGR source image.
        boxes_xyxy: (N, 4) float array of ``[x1, y1, x2, y2]`` in pixel coords.
        pad_ratio:  Multiplicative padding applied to box dimensions.
        pad_px:     Absolute pixel padding (added after ratio padding).

    Returns:
        List of BGR crop arrays.  Crops that fall outside the image are empty.
    """
    h, w = im0.shape[:2]
    crops = []
    for box in boxes_xyxy:
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - bw * pad_ratio - pad_px))
        y1 = max(0, int(y1 - bh * pad_ratio - pad_px))
        x2 = min(w, int(x2 + bw * pad_ratio + pad_px))
        y2 = min(h, int(y2 + bh * pad_ratio + pad_px))
        crops.append(im0[y1:y2, x1:x2])
    return crops


def infer_cls_on_crops(
    cls_model: ClassificationModelWrapper,
    crops: List[np.ndarray],
    topk: int = 1,
) -> List[List[Tuple]]:
    """Run classification on a list of BGR crops.

    Returns list of per-crop topk result lists:
    ``[[(remapped_label, conf, ori_label), ...], ...]``
    """
    return cls_model.predict_crops(crops, topk=topk)


def convert_det_labels(
    pred: torch.Tensor,
    det_classnames: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Unpack a single-image detection tensor into typed arrays + string labels.

    Args:
        pred:           (N, 6) tensor ``[x1, y1, x2, y2, conf, cls_id]``.
        det_classnames: Detection class name list.

    Returns:
        ``(boxes_xyxy, confs, classids, labels)``
        *boxes_xyxy*: float32 (N, 4) ndarray
        *confs*:      float32 (N,)   ndarray
        *classids*:   int32   (N,)   ndarray
        *labels*:     list of N strings
    """
    if pred is None or pred.shape[0] == 0:
        empty = np.zeros((0, 4), dtype=np.float32)
        return empty, np.array([]), np.array([], dtype=int), []

    arr      = pred.cpu().numpy()
    boxes    = arr[:, :4].astype(np.float32)
    confs    = arr[:, 4].astype(np.float32)
    classids = arr[:, 5].astype(int)
    labels   = [
        det_classnames[cid] if 0 <= cid < len(det_classnames) else "unknown"
        for cid in classids
    ]
    return boxes, confs, classids, labels


def resize_for_output(im: np.ndarray, long_side: int = 1280) -> np.ndarray:
    """Resize *im* so the longer side equals *long_side*, keeping aspect ratio."""
    h, w = im.shape[:2]
    if max(h, w) == long_side:
        return im
    scale = long_side / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


# ===============================================================
# ---- SECTION 4: Full pipeline
# ===============================================================

def run_detcls(
    data: str,
    det_weights: str,
    cls_weights: str,
    output_dir: str,
    det_imgsz: int      = 1280,
    cls_imgsz: int      = 224,
    det_conf_thres: float = 0.25,
    det_iou_thres: float  = 0.45,
    det_batch_size: int  = 4,
    device: str          = "",
    det_label_mappings: dict = None,
    cls_label_mappings: dict = None,
    output_long_side: int    = 1280,
    topk: int            = 1,
) -> dict:
    """Full detection + classification pipeline for an ImageFolder directory.

    Steps
    -----
    1. Collect image paths from *data* (recursive).
    2. Load detection model + classification model.
    3. For each batch:
       a. Preprocess images for detection.
       b. Run detection → scale boxes to original coords.
       c. For each detected box, crop original image and run classification.
       d. Draw results (Chinese label badges) onto the image.
       e. Save annotated image.

    Args:
        data:               Input image folder (recursive).
        det_weights:        Path to detection model weights (.pt).
        cls_weights:        Path to classification model weights (.pt).
        output_dir:         Directory to write annotated images.
        det_imgsz:          Detection input long-side size (pixels).
        cls_imgsz:          Classification input square size (pixels).
        det_conf_thres:     Detection NMS confidence threshold.
        det_iou_thres:      Detection NMS IoU threshold.
        det_batch_size:     Number of images per detection batch.
        device:             Torch device string ('' = auto, 'cpu', '0' …).
        det_label_mappings: ``{ori → remapped}`` dict for detection labels.
        cls_label_mappings: ``{ori → remapped}`` dict for classification labels.
        output_long_side:   Long-side size of saved annotated images.
        topk:               Top-K classification results to retain.

    Returns:
        ``dict`` with keys:
            ``"n_images"``    – total images processed
            ``"n_detected"``  – images with ≥1 detection
            ``"output_dir"``  – absolute path of output directory
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    ## ---- 1. collect images
    image_paths = glob_images(data)
    if not image_paths:
        logger.warning(f"[run_detcls] no images found under: {data}")
        return {"n_images": 0, "n_detected": 0, "output_dir": output_dir}

    logger.info(f"[run_detcls] found {len(image_paths)} images")
    os.makedirs(output_dir, exist_ok=True)

    ## ---- 2. load models
    logger.info("[run_detcls] loading detection model …")
    det_model = DetectionModelWrapper(
        det_weights, device=device, label_mappings=det_label_mappings
    )
    det_model.warmup((1, 3, det_imgsz, det_imgsz))

    logger.info("[run_detcls] loading classification model …")
    cls_model = ClassificationModelWrapper(
        cls_weights, device=device, label_mappings=cls_label_mappings
    )
    cls_model.set_img_size(cls_imgsz)
    cls_model.warmup((1, 3, cls_imgsz, cls_imgsz))

    det_classnames = list(det_model.names.values())

    ## ---- 3. transforms
    det_tfm = DetectionTransform(img_size=det_imgsz, stride=det_model.stride)

    ## ---- 4. dataloader
    loader = ImageFolderBatchLoader(image_paths, det_batch_size, det_tfm)

    ## ---- 5. inference loop
    n_detected = 0
    for batch_tensor, batch_paths, batch_ori_shapes in tqdm.tqdm(
        loader, total=len(loader), desc="det+cls"
    ):
        ##>>>> detection
        det_preds = infer_det(
            det_model, batch_tensor, batch_ori_shapes,
            conf_thres=det_conf_thres, iou_thres=det_iou_thres,
        )

        ##>>>> per-image post-processing
        for path, pred, ori_shape in zip(batch_paths, det_preds, batch_ori_shapes):
            im0 = load_image(path)  # reload original
            if im0 is None:
                continue

            ##>>>> unpack detection results in original-image coords
            boxes, confs, classids, det_labels = convert_det_labels(
                pred, det_classnames
            )

            ##>>>> classification on crops from the ORIGINAL image / coords
            cls_results_per_box = []
            if len(boxes) > 0:
                n_detected += 1
                crops = extract_crops(im0, boxes)
                cls_results_per_box = infer_cls_on_crops(cls_model, crops, topk=topk)

            ##>>>> build top-1 label lists for drawing
            cls_top1_labels = [r[0][0] for r in cls_results_per_box]  # remapped label
            cls_top1_confs  = [r[0][1] for r in cls_results_per_box]

            ##>>>> resize to output size BEFORE drawing
            im_vis = resize_for_output(im0, long_side=output_long_side)

            ##>>>> scale boxes from original coords → resized coords for drawing
            if len(boxes) > 0:
                scale_x = im_vis.shape[1] / im0.shape[1]
                scale_y = im_vis.shape[0] / im0.shape[0]
                boxes_vis = boxes.copy()
                boxes_vis[:, 0] *= scale_x
                boxes_vis[:, 2] *= scale_x
                boxes_vis[:, 1] *= scale_y
                boxes_vis[:, 3] *= scale_y
            else:
                boxes_vis = boxes

            ##>>>> draw on the resized image
            vis = draw_detcls_result(
                im_vis,
                boxes_vis, classids, confs,
                cls_top1_labels, cls_top1_confs,
                det_classnames,
            )

            ##>>>> save
            dst = os.path.join(output_dir, os.path.basename(path))
            cv2.imwrite(dst, vis)
            logger.debug(f"[run_detcls] saved → {dst}")

    logger.info(
        f"[run_detcls] done. {n_detected}/{len(image_paths)} images had detections."
    )
    return {
        "n_images":   len(image_paths),
        "n_detected": n_detected,
        "output_dir": os.path.abspath(output_dir),
    }


# ===============================================================
# ---- SECTION 5: CLI
# ===============================================================

def create_argparser() -> argparse.ArgumentParser:
    """Build and return the argument parser.

    Exposed as a function so notebooks can reuse it::

        from integrate_detcls_fiftyone import create_argparser, run_detcls
        parser = create_argparser()
        opt = parser.parse_args(["--data", "...", "--det_weights", "..."])
    """
    p = argparse.ArgumentParser(
        description="Detection + Classification pipeline on an ImageFolder directory."
    )

    ## ---- input
    p.add_argument(
        "--data",
        type=str,
        required=True,
        help="input image folder path (searched recursively)",
    )

    ## ---- detection model
    p.add_argument(
        "--det_weights",
        type=str,
        required=True,
        help="detection model weights (.pt)",
    )
    p.add_argument(
        "--det_label_mappings",
        type=str,
        default=None,
        help="optional CSV for detection label remapping "
             "(columns: class_id, remapped_label, ori_label)",
    )
    p.add_argument(
        "--det_imgsz",
        type=int,
        default=1280,
        help="detection input long-side size in pixels (default: 1280)",
    )
    p.add_argument(
        "--det_conf_thres",
        type=float,
        default=0.25,
        help="detection NMS confidence threshold (default: 0.25)",
    )
    p.add_argument(
        "--det_iou_thres",
        type=float,
        default=0.45,
        help="detection NMS IoU threshold (default: 0.45)",
    )
    p.add_argument(
        "--det_batch_size",
        type=int,
        default=4,
        help="detection inference batch size (default: 4)",
    )

    ## ---- classification model
    p.add_argument(
        "--cls_weights",
        type=str,
        required=True,
        help="classification model weights (.pt)",
    )
    p.add_argument(
        "--cls_label_mappings",
        type=str,
        default=None,
        help="optional CSV for classification label remapping "
             "(columns: class_id, remapped_label, ori_label)",
    )
    p.add_argument(
        "--cls_imgsz",
        type=int,
        default=224,
        help="classification input square size in pixels (default: 224)",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=1,
        help="number of top-K classification results to keep (default: 1)",
    )

    ## ---- output
    p.add_argument(
        "--output_dir",
        type=str,
        default="./detcls_output",
        help="directory for annotated output images (default: ./detcls_output)",
    )
    p.add_argument(
        "--output_long_side",
        type=int,
        default=1280,
        help="long-side size of saved annotated images in pixels (default: 1280)",
    )

    ## ---- device
    p.add_argument(
        "--device",
        type=str,
        default="",
        help="torch device string: '' (auto) / 'cpu' / '0' (default: '')",
    )

    return p


def main(args=None):
    """Parse arguments and run the pipeline.

    Can be called from a notebook::

        from integrate_detcls_fiftyone import main
        result = main([
            "--data",           "/path/to/images",
            "--det_weights",    "/path/to/det.pt",
            "--cls_weights",    "/path/to/cls.pt",
            "--output_dir",     "/path/to/out",
            "--cls_label_mappings", "/path/to/labels.csv",
        ])

    Args:
        args: list of CLI argument strings; defaults to ``sys.argv[1:]``.

    Returns:
        dict returned by ``run_detcls()``.
    """
    parser = create_argparser()
    opt    = parser.parse_args(args)

    ## ---- load label mappings from CSV if provided
    det_lm = None
    if opt.det_label_mappings:
        det_lm = load_label_mappings_from_csv(opt.det_label_mappings)
        logger.info(f"[main] det_label_mappings: {len(det_lm)} entries")

    cls_lm = None
    if opt.cls_label_mappings:
        cls_lm = load_label_mappings_from_csv(opt.cls_label_mappings)
        logger.info(f"[main] cls_label_mappings: {len(cls_lm)} entries")

    return run_detcls(
        data              = opt.data,
        det_weights       = opt.det_weights,
        cls_weights       = opt.cls_weights,
        output_dir        = opt.output_dir,
        det_imgsz         = opt.det_imgsz,
        cls_imgsz         = opt.cls_imgsz,
        det_conf_thres    = opt.det_conf_thres,
        det_iou_thres     = opt.det_iou_thres,
        det_batch_size    = opt.det_batch_size,
        device            = opt.device,
        det_label_mappings= det_lm,
        cls_label_mappings= cls_lm,
        output_long_side  = opt.output_long_side,
        topk              = opt.topk,
    )


# ---------------------------------------------------------------
if __name__ == "__main__":
    main()
