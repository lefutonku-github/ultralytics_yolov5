"""
I/O utilities: image file collection and label-mapping CSV loader.
"""

import csv
import logging
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# ---- image file collection

_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def glob_images(root_dir: str, extensions: set = None) -> list:
    """Recursively collect all image file paths under *root_dir*.

    Args:
        root_dir:   root directory to search.
        extensions: set of lower-case file extensions to include
                    (default: jpg / jpeg / png / bmp / tiff / webp).

    Returns:
        Sorted list of absolute image file paths.
    """
    if extensions is None:
        extensions = _IMG_EXTENSIONS
    image_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                image_paths.append(os.path.join(root, file))
    image_paths.sort()
    logger.debug(f"[glob_images] found {len(image_paths)} images under {root_dir}")
    return image_paths


# ---------------------------------------------------------------
# ---- label-mapping CSV loader

def load_label_mappings_from_csv(csv_path: str) -> dict:
    """Load a label-mapping CSV and return an ``ori_label → remapped_label`` dict.

    Expected CSV format (header row auto-detected and skipped if the first
    cell cannot be converted to *int*)::

        class_id, remapped_label, ori_label
        0,        喜鹊,           Pica pica
        1,        普通雨燕,        Apus apus

    Args:
        csv_path: path to the CSV file.

    Returns:
        ``{ori_label: remapped_label}`` dict.
    """
    mappings = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            # auto-skip header: first cell must be a parseable int
            try:
                int(row[0].strip())
            except ValueError:
                continue
            ori_label      = row[2].strip()
            remapped_label = row[1].strip()
            if ori_label:
                mappings[ori_label] = remapped_label
    logger.info(
        f"[load_label_mappings_from_csv] loaded {len(mappings)} mappings "
        f"from {csv_path}"
    )
    return mappings
