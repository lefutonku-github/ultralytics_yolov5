"""
PIL-based drawing helpers that support CJK (Chinese/Japanese/Korean) text.

All drawing functions accept and return BGR ``np.ndarray`` images so they
are drop-in replacements for OpenCV drawing calls.
"""

import logging
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# ---- font loader (cached module-level singleton)

_FONT_CACHE: dict = {}


def _load_font(font_size: int) -> ImageFont.FreeTypeFont:
    """Return a cached PIL FreeTypeFont at the given *font_size*.

    Search order:
        1. ``simsun.ttc``  (works on Windows / Linux with simsun installed)
        2. ``~/Library/Fonts/simsun.ttc``  (macOS user font dir)
        3. ``/System/Library/Fonts/STHeiti Light.ttc``  (macOS system fallback)
        4. PIL default bitmap font as last resort (no CJK support)
    """
    if font_size in _FONT_CACHE:
        return _FONT_CACHE[font_size]

    candidates = [
        "simsun.ttc",
        os.path.expanduser("~/Library/Fonts/simsun.ttc"),
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # Linux
    ]
    font = None
    for path in candidates:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except Exception:
            continue

    if font is None:
        logger.warning(
            "[_load_font] no CJK-capable font found; falling back to PIL default "
            "(Chinese characters will not render correctly)."
        )
        font = ImageFont.load_default()

    _FONT_CACHE[font_size] = font
    return font


# ---------------------------------------------------------------
# ---- atomic: draw one text badge (bg-filled rectangle + text)

def paint_label_with_bg(
    im: np.ndarray,
    text: str,
    pos: tuple,
    font_size: int = 26,
    bg_color: tuple  = (180, 60, 0),    # BGR — blue
    text_color: tuple = (255, 255, 255), # white
    padding: int = 4,
) -> np.ndarray:
    """Draw a filled-background text badge onto a BGR image.

    Supports CJK characters via PIL + simsun / STHeiti font.

    Args:
        im:         Input BGR ``np.ndarray`` (uint8).
        text:       Label string (may contain Chinese characters).
        pos:        ``(x, y)`` top-left corner of the badge.
        font_size:  PIL font size in px.
        bg_color:   Badge background colour in BGR order.
        text_color: Text colour in BGR order.
        padding:    Inner padding between text and badge border (px).

    Returns:
        BGR ``np.ndarray`` with the badge drawn in-place (copy returned).
    """
    if not isinstance(text, str):
        text = text.decode("utf-8")

    font = _load_font(font_size)

    # BGR → RGB for PIL
    img_pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

    # measure text bounding box (accurate for CJK)
    draw_tmp = ImageDraw.Draw(img_pil)
    bbox = draw_tmp.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x, y = int(pos[0]), int(pos[1])
    bg_x2 = min(x + text_w + padding * 2, img_pil.width  - 1)
    bg_y2 = min(y + text_h + padding * 2, img_pil.height - 1)

    # draw background rect (PIL uses RGB)
    bg_rgb   = (bg_color[2],   bg_color[1],   bg_color[0])
    text_rgb = (text_color[2], text_color[1], text_color[0])
    draw = ImageDraw.Draw(img_pil)
    draw.rectangle([x, y, bg_x2, bg_y2], fill=bg_rgb)
    draw.text((x + padding, y + padding), text, font=font, fill=text_rgb)

    return cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------
# ---- colour palette (same as sim_inference_onboard.py)

_HEXS = (
    "FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231",
    "48F90A", "92CC17", "3DDB86", "1A9334", "00D4BB",
    "2C99A8", "00C2FF", "344593", "6473FF", "0018EC",
    "8438FF", "520085", "CB38FF", "FF95C8", "FF37C7",
)


def _hex2bgr(h: str) -> tuple:
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (b, g, r)  # OpenCV BGR


BOX_COLORS = [_hex2bgr(h) for h in _HEXS]


# ---------------------------------------------------------------
# ---- composite: draw detection boxes + cls labels on one image

def draw_detcls_result(
    im_bgr: np.ndarray,
    det_boxes_xyxy: np.ndarray,
    det_classids: np.ndarray,
    det_confs: np.ndarray,
    cls_labels: list,
    cls_confs: list,
    det_classnames: list,
    font_size: int = 26,
    box_thickness: int = 2,
    bg_color: tuple  = (180, 60, 0),
    text_color: tuple = (255, 255, 255),
    label_padding: int = 4,
) -> np.ndarray:
    """Render detection bounding boxes and classification labels onto an image.

    Follows the same visual style as ``sim_inference_onboard.py /
    show_result_image()``:
      * coloured detection box per det-class
      * blue-bg / white-text badge with ``<cls_label>-<cls_conf>``
        positioned just above the detection box

    Args:
        im_bgr:          BGR uint8 image to draw on (will be copied).
        det_boxes_xyxy:  ``(N, 4)`` float array of ``[x1, y1, x2, y2]`` boxes.
        det_classids:    ``(N,)`` int array of detection class indices.
        det_confs:       ``(N,)`` float array of detection confidences.
        cls_labels:      List of N top-1 classification label strings
                         (already remapped / Chinese).
        cls_confs:       List of N top-1 classification confidences (float).
        det_classnames:  Detection class name list indexed by *det_classids*.
        font_size:       Badge font size.
        box_thickness:   Detection box line thickness.
        bg_color:        Badge background colour (BGR).
        text_color:      Badge text colour (BGR).
        label_padding:   Inner badge padding (px).

    Returns:
        New BGR ``np.ndarray`` with all annotations drawn.
    """
    im = im_bgr.copy()
    label_h_approx = font_size + label_padding * 2 + 4  # badge height estimate

    for i, (box, cid, dconf) in enumerate(
        zip(det_boxes_xyxy, det_classids, det_confs)
    ):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cid = int(cid)

        ##>>>> detection box
        color = BOX_COLORS[cid % len(BOX_COLORS)]
        cv2.rectangle(im, (x1, y1), (x2, y2), color,
                      thickness=box_thickness, lineType=cv2.LINE_AA)

        ##>>>> classification badge above box
        cls_label = cls_labels[i] if i < len(cls_labels) else "?"
        cls_conf  = float(cls_confs[i]) if i < len(cls_confs) else 0.0
        badge_text = f"{cls_label}-{cls_conf:.2f}"

        lx = max(x1, 0)
        ly = max(y1 - label_h_approx, 0)
        im = paint_label_with_bg(
            im, badge_text, (lx, ly),
            font_size=font_size,
            bg_color=bg_color,
            text_color=text_color,
            padding=label_padding,
        )

    return im
