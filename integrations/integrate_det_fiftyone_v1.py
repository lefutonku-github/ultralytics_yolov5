""" 
provide v1 version of integrate detection with fiftyone.
- directly use existing `detect.py` provided by yolo, save offline results and then load to fiftyone dataset using `add_yolo_label`

why v1?
- if we want to provide batch mechnism, special care should be taken to ensure the code correctness, but we can not afford the time cost debugging procedure now.
"""

##-----------------------------------------------
##---- imports

##---- import std
from cProfile import label
from genericpath import isdir
import os
import sys
import argparse
import logging
from typing import TypeVar, Union
import shutil
from pathlib import Path
import glob

import datetime
from dateutil import tz

## NOTE: add ultrylitcs into PYTHONPATH

##---- import 3rdpartys
import tqdm
import numpy as np

##---- import torch related 3rdparties
import torch
import torch.nn as nn
import torch.nn.functional as F

##---- import fiftyone related 3rdparties
import fiftyone as fo ## try using fiftyone dataest
import fiftyone.utils as fou
import fiftyone.utils.yolo as fouy

##---- import ultraylitic yolov5 related 3rdparties
import cv2
from models.common import DetectMultiBackend
from utils.torch_utils import select_device

from detect import run as detect_run

##---- import local moduls

##-----------------------------------------------
##---- vars

##-----------------------------------------------
##---- utils
def time_str_now():
    tz_sh = tz.gettz("Asia/Shanghai")
    time_str = datetime.datetime.now(tz=tz_sh).strftime("%Y%m%d_%H%M%S")
    return time_str

##-----------------------------------------------
##---- workflows
def compute_labels(dataset: Union[fo.Dataset, fo.DatasetView], 
                   weights: str, 
                   label_field:str,
                   img_size:int=640,
                   label_mappings:dict=None,
                   tag = None,
                   conf_thres=0.25,
                   iou_thres=0.45,
                   work_dir=os.getcwd(),
                   device="cuda:0", ## prefer gpu
                   keep_temp=False,
                   keep_old_label_values=False,
                   ):
    """ although in v1, we still provide the same syntax of `compute_labels` for future compatibility
    
    @Args:
        `batch_size`: note used now
        `chunk_size`: decided by available CPU mem, indicates how frequently to update the dataset
        `tag`: whether to append specified tag to those samples / labels?
    """
    ##>>>> param check and rectify
    img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
    
    ##>>>> set field schema for the new label field
    if label_field is None:
        raise Exception("required param `label_field` can not be None, MUST be specified")
    
    ## temp annotated since `add_sample_field` may only exist in `Dataset` but not in `DatasetView`
    # if label_field not in dataset.get_field_schema():
    #     dataset.add_sample_field(
    #         label_field, ## format as "{family}-{genus_c}-{species_c}"
    #         fo.EmbeddedDocumentField,
    #         embedded_doc_type=fo.Detections,
    #     )
    
    ##>>>> setup log and workdir
    timed_work_dir = os.path.join(work_dir, f"{time_str_now()}-det_fiftyone_v1")
    if not os.path.exists(timed_work_dir):
        os.makedirs(timed_work_dir)
        print(f"create workdir: {timed_work_dir}")
        
    ##>>>> temp load model for reading names
    temp_model = DetectMultiBackend(weights, device=select_device(device=device)) ## only used for reading names
    names_dict = temp_model.names.copy() ## the basic dict
    
    if label_mappings is not None:
        names_dict = {class_id: class_name if label_mappings is None or class_name not in label_mappings else label_mappings[class_name] for class_id, class_name in names_dict.items() }
    classes = [names_dict[i] for i in range(len(names_dict))]
    print(f"classes and names after mapping: {classes}")
        
    ##>>>> construct temp source.txt
    source_txt = os.path.join(timed_work_dir, "source.txt")
    filepaths = dataset.values("filepath")
    
    ## NOTE: do not use utf-8 encodings, since the `detect.py` may not support utf-8
    np.savetxt(source_txt, np.asarray(filepaths), fmt="%s" )
    
    ##>>>> run offline detection
    detect_out_dir = os.path.join(timed_work_dir, "detect_out")
    if os.path.isdir(detect_out_dir):
        shutil.rmtree(detect_out_dir) ## ensure located in expected dir
        
    detect_run(
        weights=weights,
        source=source_txt, ## source provide a txt format
        imgsz=img_size,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        save_txt=True, ## NOTE: MUST set to true
        save_conf=True, ## NOTE: MUST set to true
        device=device, ## NOTE: prefer gpu
        augment=False,
        classes=None,
        agnostic_nms=False,
        project=timed_work_dir,
        name="detect_out",
        exist_ok=True, ## do not incr
    )
    
    ##>>>> check results manually
    out_label_dir = os.path.join(detect_out_dir, "labels")
    if not os.path.isdir(out_label_dir):
        raise Exception(f"bad output dir, not found: {out_label_dir}")
    
    ##>>>> set result mappings
    labels_info_dict = {}
    for filepath in filepaths:
        out_label_file = os.path.join(out_label_dir, f"{Path(filepath).stem}.txt")
        if not os.path.exists(out_label_file):
            continue
        
        labels_info_dict[filepath] = out_label_file
        
    print(f"total {len(labels_info_dict)}/{len(filepaths)} have detections")
    if len(labels_info_dict) == 0:
        print(f"ERROR! no detection results found!!! job done!")
        return
    
    ##>>>> using `add_yolo_labels` to load offline results
    ## refer to [`add_yolo_labels`](https://docs.voxel51.com/api/fiftyone.utils.yolo.html#fiftyone.utils.yolo.add_yolo_labels) for more detail
    if not keep_old_label_values and label_field in dataset.get_field_schema():
        dataset.clear_sample_field(label_field)
        
    # And add model predictions
    fouy.add_yolo_labels(
        dataset,
        label_field=label_field,
        labels_path=labels_info_dict, ## using dict
        classes=classes,
    )
       
    ##>>>> post process and save
    dataset.save() ## ensure save
    print("job done!")
    if not keep_temp:
        shutil.rmtree(timed_work_dir)
    return

def create_argparser():
    """ make the argument parser for this script 
    NOTE:
        make it a function, so that it can be used in other scripts, eg., notebooks
    """
    parser = argparse.ArgumentParser(description="Wrap yolo detection model into fiftyone usage")
    
    ##>>>> fiftyone setup
    parser.add_argument(
        "--fiftyone_dsname",
        type=str,
        choices=fo.list_datasets(),
        help="the dataset name in fiftyone",
    )
    parser.add_argument(
        "--fiftyone_viewname",
        type=str,
        default=None,
        help="the dataset name in fiftyone",
    )
    parser.add_argument(
        "--label_field",
        type=str,
        default="my_detections",
        help="the field to put detection results",
    )
    
    ##>>>> model setup
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov5s-cls.pt",
        help="model.pt path(s)",
    )
    parser.add_argument(
        "--imgsz",
        type=int, ## temp NOT convert to tuple
        default=640,
        help="inference size (pixels)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="cuda device, i.e. 0 or 0,1,2,3 or cpu",
    )
    
    ##>>>> other setup
    parser.add_argument(
        "--conf_thres",
        type=float,
        default=0.25,
        help="conf threshold for detection nms",
    )
    parser.add_argument(
        "--iou_thres",
        type=float,
        default=0.45,
        help="iou threshold for detection nms",
    )
    parser.add_argument(
        "--keep_temp", nargs="?", default=False, const=True, help="clear intermediate results"
    )
    parser.add_argument(
        "--keep_old_label_values", nargs="?", default=False, const=True, help="whether keep original values in label_field, by default False! "
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        required=True,
        help="where to put the temp results",
    )
    return parser

##-----------------------------------------------
##---- unit_test

##-----------------------------------------------
##---- main
if __name__ == "__main__":
    ##>>>> parse args
    parser = create_argparser()
    args = parser.parse_args()
    
    ##>>>> dataset
    if args.fiftyone_dsname not in fo.list_datasets():
        raise Exception(f"bad dataset name, not in fiftyone datasets, available datasets are: {fo.list_datasets()}")
    
    dataset = fo.load_dataset(args.fiftyone_dsname)
    label_mappings = None if "det_label_mapping" not in dataset.info else dataset.info["det_label_mapping"]
    print(f"label mappings: {label_mappings}, dataset detail:\n{dataset} ")
    
    if args.fiftyone_viewname is not None:
        if args.fiftyone_viewname not in dataset.list_saved_views():
            raise Exception(f"bad view name, not in fiftyone views, available views are: {dataset.list_saved_views()}")
        
        view = dataset.load_saved_view(args.fiftyone_viewname)
    else:
        view = dataset
    
    ##>>>> model workflow
    compute_labels(
        dataset=view,
        weights=args.weights,
        label_field=args.label_field,
        img_size=args.imgsz,
        label_mappings=label_mappings,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        device=args.device,
        keep_temp=args.keep_temp,
        keep_old_label_values=args.keep_old_label_values,
        work_dir=args.work_dir,
    )
    
    pass
