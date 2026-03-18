"""
Image pre-processing transforms.

Each transform is a callable that accepts a BGR ``np.ndarray`` (H×W×C)
and returns a ``torch.Tensor`` ready to be stacked into a batch.
"""

import numpy as np
import torch

# NOTE: these imports assume the ultralytics_yolov5 root is on sys.path,
# which is handled by integrate_detcls_fiftyone.py at import time.
from utils.augmentations import classify_transforms, letterbox


# ---------------------------------------------------------------
# ---- detection transform

class DetectionTransform:
    """Letterbox-resize + normalise for a YOLOv5 detection model.

    Args:
        img_size: target long-side size (pixels).
        stride:   model stride (usually 32).
        auto:     use auto-padding (True for .pt models).
    """

    def __init__(self, img_size: int = 1280, stride: int = 32, auto: bool = True):
        self._img_size = img_size
        self._stride   = stride
        self._auto     = auto

    def __call__(self, im0: np.ndarray) -> torch.Tensor:
        """
        Args:
            im0: BGR uint8 image (H×W×C).

        Returns:
            Float32 tensor (C×H×W) in [0, 1].
        """
        im = letterbox(im0, self._img_size, stride=self._stride, auto=self._auto)[0]
        im = im.transpose((2, 0, 1))[::-1]       # HWC BGR → CHW RGB
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).float()
        im /= 255.0
        return im


# ---------------------------------------------------------------
# ---- classification transform

class ClassificationTransform:
    """Center-crop + ImageNet normalise for a YOLOv5 classification model.

    Args:
        img_size: square crop size (pixels, default 224).
    """

    def __init__(self, img_size: int = 224):
        self._tfm = classify_transforms(img_size)

    def __call__(self, im0: np.ndarray) -> torch.Tensor:
        """
        Args:
            im0: BGR uint8 image (H×W×C).

        Returns:
            Float32 tensor (C×H×W).
        """
        return self._tfm(im0)
