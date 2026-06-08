# DEPRECATED: migrated to xlab_inat.integrations.cls_v2 / det_v1 (legacy/ for frozen reference).
""" 
wrap yolo classification model into fiftyone usage,
in a step by step implementation manner 

20240610 step1: provide functionality that can `load` and provide `predict` `embed`, `logits` from a yolo classification model to fiftyone, in common format (eg., np.ndarray), not in fiftyone class.
20260318 step2: support classification pipeline for a folder of images:
    - input:  image folder
    - output: (1) CSV label file in a general classification dataset format
              (2) ImageNet-style folder layout (label sub-dirs) with symlinked / copied images
              (3) labeled visualization images with confidence overlaid
    - entry:  run_classification() + main() + create_argparser()
"""

##-----------------------------------------------
##---- imports

##---- import std
import os
import sys
import csv
import shutil
import argparse
import logging
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

##---- import 3rdpartys
import tqdm
import numpy as np

##---- import torch related 3rdparties
import torch
import torch.nn as nn
import torch.nn.functional as F

##---- import fiftyone related 3rdparties
import fiftyone.core.models as focm

##---- import ultraylitic yolov5 related 3rdparties
import cv2
from PIL import Image, ImageDraw, ImageFont
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.augmentations import classify_transforms, letterbox
## simple dataloader for `LoadImages`
from utils.dataloaders import (
    IMG_FORMATS, 
    LoadImages, 
)

##---- import local moduls


##-----------------------------------------------
##---- vars
logger = logging.getLogger(__name__)

##-----------------------------------------------
##---- utils
class ClassificationModelWrapper(nn.Module):
    """ directly use `nn.Module` as base class for flexibility.
    
    later can be changed to `ultralytics.BaseModel` or `fiftyone.FiftyOneYOLOModel` if neccessary. Now we did'nt do that just for simplicity !!
    
    @Interfaces: try our best to make it compatible with fiftyone model, eg.,
        - load
        - predict, predict_all
        - embed, embed_all, has_embeddings
        - logits, has_logits, etc
    
    """
    def __init__(self, weights, device=None, label_mappings:dict=None):
        """ wrap a yolov5 classification model to provide embedings and logits

        Args:
            weights (_type_): _description_
            device (_type_, optional): _description_. Defaults to None.
            label_mappings: support internal label mappings for usage convenience
        """
        super(ClassificationModelWrapper, self).__init__()
        
        ##>>>> load model
        ## NOTE: use internal class_id -> class_label mapping, but provide extra mapping mechanism for user (eg., to ch name)
        self.device = select_device(device) if device is not None else device
        
        model = DetectMultiBackend(weights, device=self.device)
        
        self.model = model
        model.eval()
        
        ##>>>> internal impls
        self._backbone = model.model.features
        
        ##>>>> extra mappings
        self._label_mappings = label_mappings
        
        return
    
    def warmup(self, imgsz=(1, 3, 640, 640)):
        """ warmup the model
        """
        self.model.warmup(imgsz)
        return
    
    def forward(self, x):
        """ forward pass
        NOTE: to support embeddings, logits, etc
        """
        with torch.no_grad():
            x = x.to(self.device)
            ##>>>> logits
            logits = self.model(x)
        
        return logits
    
    def logits_to_labels(self, logits, k:int=1, label_mappings:dict=None):
        """ convert logits to labels
        
        @Args:
            logits: input logis
            k: topk, default 1, can be set to k
            label_mappings: the labels directly from model will be remap again, typically for more readable labels
            
        @Returns:
           typically with [(label, conf, logit), ...] format, sorted by conf in desending order
           if k == 1, return top1 label, in (label, conf, logit) format
           if k > 1, return topk labels, in [(label, conf, logit), ...] format
           if label_mappings is provided, generally in [(remapped_label, conf, logit, original_label), ...] format
        """
        ##>>>> get topk
        prob = F.softmax(logits, dim=-1)
        ## use torch.topk rather than numpy.argsort, as it's more efficient
        topk = torch.topk(prob, k, dim=-1, largest=True, sorted=True)
        
        ##>>>> convert to labels
        if label_mappings is None:
            label_mappings = self._label_mappings
        
        ## output: (batch_size, k, x) format
        labels = []
        for i, idxs in enumerate(topk.indices): ## the first dim of topk.indices will be the batch
            cur_topk_labels = []
            for k in idxs:
                conf = float(prob[i, k])
                ori_label = self.model.names[int(k)]
                remapped_label = label_mappings[ori_label] if label_mappings is not None and ori_label in label_mappings else ori_label
                cur_topk_labels.append((remapped_label, conf, ori_label))
                
            labels.append(cur_topk_labels)
            
        labels = np.asarray(labels) ## convert to numpy array
        
        return labels
    
    def predict(self, x):
        """ predict
        """
        return self.forward(x)
    
    def predict_all(self, x):
        """ predict all
        """
        return self.predict(x)
    
    @property
    def has_embeddings(self):
        return True
    
    @property
    def has_logits(self):
        return True
    
    def embed(self, x):
        """ forward pass
        NOTE: to support embeddings, logits, etc
        """
        with torch.no_grad():
            x = x.to(self.device)
            
            ##>>>> features
            features = self._backbone(x)
            
            ##>>>> embeddings
            ## note: for us, we only take 1 layer output, so different from yolo implementation
            embeddings = nn.functional.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)  # flatten
            
            ## note: fiftyone requires a return of numpy array, not a list, so directly convert to numpy and not reduce the first dims
            ## refer to https://docs.voxel51.com/api/fiftyone.core.models.html#fiftyone.core.models.EmbeddingsMixin.embed and https://docs.voxel51.com/api/fiftyone.core.models.html#fiftyone.core.models.EmbeddingsMixin.embed_all for detail
            # embeddings_unbind = torch.unbind(embeddings.to('cpu'), dim=0) ## convert to tuple or list of each batch.
            # embeddings_unbind = [embeddings.cpu().numpy() for embeddings in embeddings_unbind]
            embeddings = embeddings.cpu().numpy()
            
            
        return embeddings
    
    def embed_all(self, x):
        """ embed
        """
        return self.embed(x)


class BatchDataLoader:
    def __init__(self, dataset, batch_size=16, img_size=224, transforms=None) -> None:
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.count = 0
        self.nf = len(self.dataset) ## number of files
        
        self.img_size = img_size
        self.transforms = transforms
        pass
    
    def __len__(self):
        """Returns the number of files in the dataset."""
        return int(np.ceil(self.nf / self.batch_size))  # number of files
    
    def __iter__(self):
        """Initializes iterator by resetting count and returns the iterator object itself."""
        self.count = 0
        return self
    
    def __next__(self):
        """Advances to the next file in the dataset, raising StopIteration if at the end."""
        ##>>>> backup
        if self.count == self.nf:
            raise StopIteration
        
        ##>>>> batching the data
        images, paths = [], []
        for i in range(self.batch_size):
            cur_imfile = self.dataset[self.count]
            path, im, *_ = self._get_image(cur_imfile)
            self.count += 1
            
            images.append(im)
            paths.append(path)
            
            if len(images) < self.batch_size and self.count < self.nf:
                continue
            
            ##>>>> expand the batch dim for images
            image_tensor = torch.stack(images, 0)
            return image_tensor, paths
            
        image_tensor = torch.stack(images, 0)
        return images, paths
    
    def _get_image(self, img_path:str):
        im0 = cv2.imread(img_path)  # BGR
        assert im0 is not None, f"Image Not Found {img_path}"
        s = f"image {self.count}/{self.nf} {img_path}: "
        
        if self.transforms:
            im = self.transforms(im0)  # transforms
        else:
            im = letterbox(im0, self.img_size, 1, auto=True)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous
            
        return img_path, im, im0, None, s

# Function to recursively gather all image paths
def glob(
    root_dir,
    extensions={".jpg", ".jpeg", ".png", ".bmp", ".tiff"},
):
    """act as a list of list of image paths, each list is a chunk of image paths
    NOTE: support gater by chunk_size to avoid too many files in memory"""
    image_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(tuple(extensions)):
                image_paths.append(os.path.join(root, file))
    return image_paths


def load_label_mappings_from_csv(csv_path: str) -> dict:
    """Load a label mapping CSV and return an ``ori_label → remapped_label`` dict.

    Expected CSV format (no header row required; header row is auto-detected and
    skipped if the first cell cannot be converted to int)::

        class_id, remapped_label, ori_label
        0,        Magpie,         Pica pica
        1,        Common Swift,   Apus apus

    Args:
        csv_path: path to the CSV file

    Returns:
        dict mapping ``ori_label`` (column 3) → ``remapped_label`` (column 2)
    """
    mappings = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            ##>>>> auto-skip header row: if first cell is not an integer, treat as header
            try:
                int(row[0].strip())
            except ValueError:
                continue
            ori_label     = row[2].strip()
            remapped_label = row[1].strip()
            mappings[ori_label] = remapped_label
    logger.info(f"[load_label_mappings_from_csv] loaded {len(mappings)} mappings from {csv_path}")
    return mappings


##-----------------------------------------------
##---- workflows
def compute_labels(image_paths:str, model, batch_size:int=16, img_size:int=224, topk:int = 1, label_mappings:dict=None):
    """ support batch computing for convenience """
    ##>>>> dataset / data source
    if isinstance(image_paths, str) and os.path.isdir(image_paths):
        image_paths = glob(image_paths)
    
    ##>>>> dataloader
    dataloader = BatchDataLoader(
        dataset=image_paths, 
        batch_size=batch_size, 
        img_size=img_size, 
        transforms=classify_transforms(img_size)
    )
    
    ##>>>> model setup
    model.eval() ## duplicate but ensure val mode
    model.warmup()
    
    ##>>>> action loop
    results = []
    for imgs, paths in tqdm.tqdm(dataloader, total=len(dataloader), desc="batch:"):
        with torch.no_grad():
            logits = model.predict(imgs)
            labels = model.logits_to_labels(logits, k=topk, label_mappings=label_mappings)
            
            ##>>>> log the results
            for path, label, logit in zip(paths, labels, logits):
                results.append((path, label, logit.cpu().numpy()))
                # print(f"image: {path}, label: {label}")
               
    ##>>>> post processing
    return results

def compute_embeddings(image_paths:str, model, batch_size:int=16, img_size:int=224):
    """ support batch computing for convenience """
    ##>>>> dataset / data source
    if isinstance(image_paths, str) and os.path.isdir(image_paths):
        image_paths = glob(image_paths)
    
    ##>>>> dataloader
    dataloader = BatchDataLoader(
        dataset=image_paths, 
        batch_size=batch_size, 
        img_size=img_size, 
        transforms=classify_transforms(img_size)
    )
    
    ##>>>> model setup
    model.eval() ## duplicate but ensure val mode
    model.warmup()
    
    ##>>>> action loop
    results = []
    for imgs, paths in tqdm.tqdm(dataloader, total=len(dataloader), desc="batch:"):
        with torch.no_grad():
            embeddings = model.embed(imgs)
            
            results.append(embeddings)
            
    ##>>>> post processing
    results = np.concatenate(results, axis=0) ## concat along the first dim
    return results


##-----------------------------------------------
##---- output helpers

def save_results_to_csv(results: list, output_dir: str, filename: str = "predictions.csv") -> str:
    """Save classification results to a CSV file.

    CSV columns: image_path, top1_label, top1_conf, [top2_label, top2_conf, ...]
    This is a common flat format compatible with most downstream tools.

    Args:
        results: list of (path, topk_labels, logits) tuples returned by compute_labels()
        output_dir: directory to write the CSV file
        filename: output CSV filename

    Returns:
        absolute path of the written CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, filename)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        ##>>>> header — detect topk from first result
        topk = len(results[0][1]) if results else 1
        header = ["image_path"]
        for k in range(topk):
            header += [f"top{k+1}_label", f"top{k+1}_conf"]
        writer.writerow(header)

        ##>>>> rows
        for path, topk_labels, _logits in results:
            row = [path]
            for label_entry in topk_labels:
                ## label_entry: (remapped_label, conf, original_label)
                row += [label_entry[0], f"{float(label_entry[1]):.6f}"]
            writer.writerow(row)

    logger.info(f"[save_results_to_csv] saved {len(results)} records → {csv_path}")
    return csv_path


def save_results_to_imagenet_layout(results: list, output_dir: str, copy_files: bool = False) -> str:
    """Organise images into an ImageNet-style folder layout based on top-1 prediction.

    Output structure::

        output_dir/
            label_A/
                image1.jpg
                image2.jpg
            label_B/
                image3.jpg

    By default symbolic links are created (fast, saves disk).  Set
    ``copy_files=True`` to physically copy the files instead.

    Args:
        results: list of (path, topk_labels, logits) from compute_labels()
        output_dir: root directory for the imagenet-style layout
        copy_files: if True, copy files; otherwise create symlinks

    Returns:
        output_dir path
    """
    os.makedirs(output_dir, exist_ok=True)
    for path, topk_labels, _logits in results:
        top1_label = topk_labels[0][0]  ## remapped_label
        label_dir = os.path.join(output_dir, top1_label)
        os.makedirs(label_dir, exist_ok=True)

        dst = os.path.join(label_dir, os.path.basename(path))
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)

        if copy_files:
            shutil.copy2(path, dst)
        else:
            os.symlink(os.path.abspath(path), dst)

    logger.info(f"[save_results_to_imagenet_layout] organised {len(results)} images → {output_dir}")
    return output_dir


def save_labeled_images(results: list, output_dir: str,
                        font_size: int = 26,
                        bg_color: tuple = (180, 60, 0),    # BGR — blue
                        text_color: tuple = (255, 255, 255),
                        padding: int = 6) -> str:
    """Render top-1 classification label + confidence onto each image and save.

    The label is drawn in a filled-rectangle badge style (blue background,
    white text) at the top-left corner of the image.
    Supports Chinese characters via PIL (simsun.ttc font).

    Args:
        results: list of (path, topk_labels, logits) from compute_labels()
        output_dir: directory to write visualised images
        font_size: PIL font size (default: 26)
        bg_color: BGR tuple for the label background rectangle
        text_color: BGR tuple for the label text
        padding: pixels of padding around the text inside the badge

    Returns:
        output_dir path
    """
    os.makedirs(output_dir, exist_ok=True)

    ##>>>> load font once — fallback to absolute path on macOS if needed
    try:
        _font = ImageFont.truetype("simsun.ttc", font_size)
    except Exception:
        _font = ImageFont.truetype("/Users/weiliu/Library/Fonts/simsun.ttc", font_size)

    for path, topk_labels, _logits in results:
        im_bgr = cv2.imread(path)
        if im_bgr is None:
            logger.warning(f"[save_labeled_images] cannot read image: {path}")
            continue

        top1_label = topk_labels[0][0]
        top1_conf  = float(topk_labels[0][1])
        text = f"{top1_label} {top1_conf:.2f}"
        if not isinstance(text, str):
            text = text.decode("utf-8")

        ##>>>> convert BGR → RGB for PIL
        img_pil = Image.fromarray(cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB))

        ##>>>> measure text bounding box via PIL (supports CJK)
        draw_tmp = ImageDraw.Draw(img_pil)
        bbox = draw_tmp.textbbox((0, 0), text, font=_font)  # (left, top, right, bottom)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        x0, y0 = padding, padding
        bg_x2 = x0 + text_w + padding * 2
        bg_y2 = y0 + text_h + padding * 2

        ##>>>> clamp to image boundary
        img_w, img_h = img_pil.size
        bg_x2 = min(bg_x2, img_w - 1)
        bg_y2 = min(bg_y2, img_h - 1)

        ##>>>> draw filled background rectangle (PIL uses RGB)
        bg_color_rgb   = (bg_color[2],   bg_color[1],   bg_color[0])
        text_color_rgb = (text_color[2], text_color[1], text_color[0])
        draw = ImageDraw.Draw(img_pil)
        draw.rectangle([x0, y0, bg_x2, bg_y2], fill=bg_color_rgb)

        ##>>>> draw text on top of the background
        draw.text((x0 + padding, y0 + padding), text, font=_font, fill=text_color_rgb)

        ##>>>> convert back to BGR and save
        im_out = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
        dst = os.path.join(output_dir, os.path.basename(path))
        cv2.imwrite(dst, im_out)

    logger.info(f"[save_labeled_images] saved labeled images → {output_dir}")
    return output_dir


##-----------------------------------------------
##---- main pipeline

def run_classification(
    data: str,
    weights: str,
    output_dir: str,
    batch_size: int = 16,
    imgsz: int = 224,
    topk: int = 1,
    device: str = "",
    label_mappings: dict = None,
    save_csv: bool = True,
    save_imagenet_layout: bool = True,
    save_viz: bool = True,
    copy_files: bool = False,
) -> dict:
    """Full classification pipeline for an image folder.

    Steps:
        1. Collect image paths from ``data`` folder (recursively).
        2. Load model from ``weights``.
        3. Run batch classification → results list.
        4. (Optional) Save CSV label file.
        5. (Optional) Save ImageNet-style folder layout.
        6. (Optional) Save visualised labeled images.

    Args:
        data:                  path to image folder (or list of image paths)
        weights:               path to model weights (.pt)
        output_dir:            root directory for all outputs
        batch_size:            inference batch size
        imgsz:                 input image size (square)
        topk:                  number of top-K predictions to keep per image
        device:                torch device string, e.g. '' / 'cpu' / '0'
        label_mappings:        optional dict mapping model class names to human-readable labels
        save_csv:              whether to save predictions.csv
        save_imagenet_layout:  whether to organise images into label sub-dirs
        save_viz:              whether to save visualised labeled images
        copy_files:            if True, copy files in imagenet layout; else symlink

    Returns:
5        dict with keys: 'results', 'csv_path', 'layout_dir', 'viz_dir'
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ##>>>> 1. collect image paths
    if isinstance(data, (str, Path)) and os.path.isdir(data):
        image_paths = glob(str(data))
    elif isinstance(data, list):
        image_paths = data
    else:
        raise ValueError(f"data must be an image folder path or a list of paths, got: {data}")

    if not image_paths:
        logger.warning(f"[run_classification] no images found under: {data}")
        return {"results": [], "csv_path": None, "layout_dir": None, "viz_dir": None}

    logger.info(f"[run_classification] found {len(image_paths)} images, loading model …")

    ##>>>> 2. load model
    model = ClassificationModelWrapper(weights, device=device, label_mappings=label_mappings)

    ##>>>> 3. compute labels
    results = compute_labels(
        image_paths=image_paths,
        model=model,
        batch_size=batch_size,
        img_size=imgsz,
        topk=topk,
        label_mappings=label_mappings,
    )
    logger.info(f"[run_classification] inference done, {len(results)} images processed.")

    os.makedirs(output_dir, exist_ok=True)
    output = {"results": results, "csv_path": None, "layout_dir": None, "viz_dir": None}

    ##>>>> 4. save CSV
    if save_csv:
        output["csv_path"] = save_results_to_csv(results, output_dir)

    ##>>>> 5. ImageNet-style layout
    if save_imagenet_layout:
        layout_dir = os.path.join(output_dir, "imagenet_layout")
        output["layout_dir"] = save_results_to_imagenet_layout(results, layout_dir, copy_files=copy_files)

    ##>>>> 6. labeled visualization images
    if save_viz:
        viz_dir = os.path.join(output_dir, "labeled_images")
        output["viz_dir"] = save_labeled_images(results, viz_dir)

    logger.info(f"[run_classification] all outputs saved to: {output_dir}")
    return output


##-----------------------------------------------
##---- argument parser

def create_argparser():
    """ make the argument parser for this script
    NOTE:
        make it a function, so that it can be used in other scripts, eg., notebooks
    """
    parser = argparse.ArgumentParser(description="Classify images in a folder with a YOLOv5 classification model")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="input image folder path",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="model weights path (.pt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./cls_output",
        help="root output directory for all results (default: ./cls_output)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="inference batch size (default: 16)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=224,
        help="inference image size in pixels, square (default: 224)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=1,
        help="save top-K predictions per image (default: 1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="torch device string: '' (auto) / 'cpu' / '0' (default: '')",
    )
    parser.add_argument(
        "--label_mappings",
        type=str,
        default=None,
        help="optional path to a CSV file for label remapping. "
             "Expected columns (no header): class_id, remapped_label, ori_label. "
             "Builds an ori_label → remapped_label mapping used in output.",
    )
    parser.add_argument(
        "--no_csv",
        action="store_true",
        help="skip saving predictions.csv",
    )
    parser.add_argument(
        "--no_imagenet_layout",
        action="store_true",
        help="skip saving ImageNet-style folder layout",
    )
    parser.add_argument(
        "--no_viz",
        action="store_true",
        help="skip saving labeled visualization images",
    )
    parser.add_argument(
        "--copy_files",
        action="store_true",
        help="copy image files in imagenet layout instead of creating symlinks",
    )
    return parser


##-----------------------------------------------
##---- main entry

def main(args=None):
    """Parse CLI arguments and run the classification pipeline.

    Can be called from other scripts or notebooks by passing a list of
    argument strings directly::

        from integrate_cls_fiftyone import main
        results = main(["--data", "/imgs", "--weights", "model.pt",
                        "--output_dir", "/out", "--topk", "3"])

    Args:
        args: list of CLI argument strings (default: sys.argv[1:])

    Returns:
        dict returned by run_classification()
    """
    parser = create_argparser()
    opt = parser.parse_args(args)

    ##>>>> load optional label mappings from CSV file
    label_mappings = None
    if opt.label_mappings is not None:
        label_mappings = load_label_mappings_from_csv(opt.label_mappings)
        logger.info(f"[main] loaded label_mappings with {len(label_mappings)} entries")

    return run_classification(
        data=opt.data,
        weights=opt.weights,
        output_dir=opt.output_dir,
        batch_size=opt.batch_size,
        imgsz=opt.imgsz,
        topk=opt.topk,
        device=opt.device,
        label_mappings=label_mappings,
        save_csv=not opt.no_csv,
        save_imagenet_layout=not opt.no_imagenet_layout,
        save_viz=not opt.no_viz,
        copy_files=opt.copy_files,
    )



##-----------------------------------------------
##---- unit_test
def unit_test_batch_predict():
    """ unit test: compare with yolov5 predict
    
    NOTE: refer to `data/ImageNet10.yaml` for datadownloading and class names etc
    """
    work_dir = os.path.dirname(__file__)
    ##>>>> basic setup
    data_source_dir = os.path.join(work_dir, "../../datasets/imagenet10")
    weights = os.path.join(work_dir, "../data/weights/efficientnet_b0.pt")
    
    model_test = ClassificationModelWrapper(weights)
    
    imgsz = (224, 224)
    batch_size = 16
    topk = 5
    
    label_mappings = {class_label: f"remap_{class_label}" for class_id, class_label in model_test.model.names.items()}
    
    results = compute_labels(
        image_paths=data_source_dir, 
        model=model_test, 
        batch_size=batch_size, 
        img_size=imgsz[0],
        topk=topk,
        label_mappings=label_mappings,
        )
    
    embeddings = compute_embeddings(
        image_paths=data_source_dir, 
        model=model_test, 
        batch_size=batch_size, 
        img_size=imgsz[0],
        )
    
    pass

def unit_test_cmp_with_yolov5_predict():
    """ unit test: compare with yolov5 predict
    
    NOTE: refer to `data/ImageNet10.yaml` for datadownloading and class names etc
    """
    work_dir = os.path.dirname(__file__)
    ##>>>> basic setup
    imgsz = (224, 224)
    data_source_dir = os.path.join(work_dir, "../../datasets/imagenet10/train")
    dataset = LoadImages(data_source_dir, img_size=imgsz, transforms=classify_transforms(imgsz[0]))
    
    weights = os.path.join(work_dir, "../data/weights/efficientnet_b0.pt")
    
    device = select_device()
    model_bench = DetectMultiBackend(weights, device=device)
    model_test = ClassificationModelWrapper(weights)
    
    batch_size = 1
    model_bench.eval()
    model_bench.warmup()
    model_test.warmup()
    
    ##>>>> model warmup
    topk = 5
    ## note: in batch_size==1 case, note there's no batching right now
    for path, im, im0s, vid_cap, s in dataset:
        ##>>>> get bench result
        im_bench = torch.Tensor(im).to(model_bench.device)
        im_bench = im_bench.float()  # uint8 to fp16/32
        if len(im_bench.shape) == 3:
            im_bench = im_bench[None]  # expand for batch dim
            
        ## get bench
        with torch.no_grad():
            results_bench = model_bench(im_bench)
            
        results_bench = results_bench.cpu()
        pred_bench = F.softmax(results_bench, dim=1)  # probabilities
    
        ##>>> get test result
        im_test = torch.Tensor(im).to(model_test.device)
        im_test = im_test.float()  # uint8 to fp16/32
        if len(im_test.shape) == 3:
            im_test = im_test[None]  # expand for batch dim
            
        logit_test = model_test(im_test)
        ## our result present in a ndarray of shape(batch_size, k, x) format
        pred_test = model_test.logits_to_labels(logit_test, k=topk)
            
        ##>>>> check and verify, they are in same batch_size
        ## topk if neccessary
        # Process predictions
        for i in range(batch_size):  # per image
            logits_bench = results_bench[i]
            prob_bench = pred_bench[i]
            topk_bench = prob_bench.argsort(0, descending=True)[:topk].tolist()  # top 5 
            
            ## convert the topk results to [(label, conf, logit, ...), ...] format
            final_bench = np.asarray([(model_bench.names[k], float(prob_bench[k]), float(logits_bench[k])) for k in topk_bench])
            
            ## for our model, the result is already in pred_test
            final_test = pred_test[i]
            
            ##>>>> compare the results
            label_diff = np.argwhere(final_bench[::, 0] != final_test[::, 0])
            conf_diff = np.argwhere(~np.isclose(final_bench[::, 1].astype(float), final_test[::, 1].astype(float)))
            logic_diff = np.argwhere(~np.isclose(final_bench[::, 2].astype(float), final_test[::, 2].astype(float)))
            
            fail_index = set(np.concatenate([label_diff, conf_diff, logic_diff], axis=0))
            
            ss = "PASS" if len(fail_index) == 0 else "FAIL"
            print(f"status:{ss}, image: {os.path.basename(path)}, fail_index: {fail_index}")
            pass
    
    pass

##-----------------------------------------------
##---- main
if __name__ == "__main__":
    main()
