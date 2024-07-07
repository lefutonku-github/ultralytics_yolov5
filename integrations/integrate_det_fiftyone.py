""" 
wrap yolo detection model into fiftyone usage,
in a step by step implementation manner
"""

""" 
20240610 step1: provide functionality that can `load` and provide `predict` `embed`, `logits` from a yolo classification model to fiftyone, in common format (eg., np.ndarray), not in fiftyone class.
"""

##-----------------------------------------------
##---- imports

##---- import std
import os
import sys
import argparse
import logging
from typing import TypeVar, Union

## NOTE: add ultrylitcs into PYTHONPATH
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

##---- import 3rdpartys
import tqdm
import numpy as np

##---- import torch related 3rdparties
import torch
import torch.nn as nn
import torch.nn.functional as F

##---- import fiftyone related 3rdparties
import fiftyone as fo ## try using fiftyone dataest
import fiftyone.core.models as focm

##---- import ultraylitic yolov5 related 3rdparties
import cv2
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
## simple dataloader for `LoadImages`
from utils.dataloaders import (
    IMG_FORMATS, 
    LoadImages, 
)

from utils.augmentations import letterbox
from utils.general import (
    non_max_suppression, scale_boxes, xyxy2xywh,
)

##---- import local moduls


##-----------------------------------------------
##---- vars

##-----------------------------------------------
##---- utils
class DetectionTransform:
    def __init__(self, img_size:int=640, stride:int=32) -> None:
        """ stride shall use the model stride """
        self._img_size = img_size
        self._stride=stride
        self._auto=True  ## for pt model, auto is True
        pass
    
    def __call__(self, im0):
        im = letterbox(im0, self._img_size, stride=self._stride, auto=self._auto)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        
        ## NOTE: `torch.ToTensor()` directly transform np.ndarray y (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] if the numpy.ndarray has dtype = np.uint8
        ## since we already transpose the image, so no need to use `torch.ToTensor()` here
        im = torch.from_numpy(im)
        im /= 255  # 0 - 255 to 0.0 - 1.0
        return im
    
def xyxy2tlwh(x):
    """Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right.
    @Returns:
       tlwh boxes: [x, y, w, h]
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0]  # x tl
    y[..., 1] = x[..., 1]  # y tl
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y
    
class DetectionModelWrapper(nn.Module):
    """ wrap detection model to provide embedings and logits, and directly convert label and to fiftyone formats
    
    later can be changed to `ultralytics.BaseModel` or `fiftyone.FiftyOneYOLOModel` if neccessary. Now we did'nt do that just for simplicity !!
    
    @Interfaces: try our best to make it compatible with fiftyone model, eg.,
        - load
        - predict, predict_all
        - embed, embed_all, has_embeddings
        - logits, has_logits, etc
        
    @Notes:
        directly use `nn.Module` as base class for flexibility.
    
    """
    def __init__(self, weights, device=None, label_mappings:dict=None):
        """ wrap a yolov5 classification model to provide embedings and logits

        Args:
            weights (_type_): _description_
            device (_type_, optional): _description_. Defaults to None.
            label_mappings: support internal label mappings for usage convenience
        """
        super(DetectionModelWrapper, self).__init__()
        
        ##>>>> load model
        ## NOTE: use internal class_id -> class_label mapping, but provide extra mapping mechanism for user (eg., to ch name)
        self._device = select_device(device) if device is None or device == "" else device
        
        self._model = DetectMultiBackend(weights, device=self._device)
        self._model.eval()
        self._label_mappings = label_mappings ## object level label mappings
    
        return
    
    def warmup(self, imgsz=(640, 640)): ## default imgsz
        """ warmup the model
        """
        assert isinstance(imgsz, tuple) and len(imgsz) == 2, f"imgsz must be a tuple of 2, but got {imgsz}"
        
        warmup_imgsz = (1, 3, *imgsz)
        self._model.warmup(warmup_imgsz)
        return
    
    def forward(self, x):
        """ forward pass. only consider forward in current imgsz
        """
        with torch.no_grad():
            x = x.to(self._device)
            ##>>>> logits
            preds = self._model(x)
            
        return preds
    
    def predict(self, x, conf_thres=0.25, iou_thres=0.45, ori_imshapes:np.ndarray=None):
        """ when wrapped by this model , not only simple forward pass, but also post processing, lik nms, xyxy to xywhn etc
        @Args:
            - x: the input tensor, of shape (bs, c, w, h)
            - conf_thres: the confidence threshold for nms
            - iou_thres: the iou threshold for nms
            - ori_imsizes: the original image sizes, of shape (bs, 2), with type np.ndarray, if not None, will be used to convert the box to relative box
        @Notes:
            predict in fiftyone format, but not in fiftyone class, just in common format
        """
        ##>>>> param check
        if ori_imshapes is not None and len(ori_imshapes) != x.shape[0]:
            raise Exception(f"ori_imshapes must be None or of shape (bs, 2+), but got {ori_imshapes}")
            
        ##>>>> forward pass
        raw_preds = self.forward(x)
        
        ##>>>> nms
        ## NOTE: out type: list(torch.tensor((n, 6))]), of shape: (bs, n, 6), 6 is of form [xyxy, conf, cls]
        preds = non_max_suppression(raw_preds, 
                                    conf_thres,
                                    iou_thres,
                                #    max_det=300, ## use default val
                                    )
        
        ##>>>> convert to fiftyone format
        ## mainly [<top-left-x>, <top-left-y>, <width>, <height>], with  box coordinates as floats in [0, 1] relative to the dimensions of the image. 
        ## refer to [Docs > FiftyOne User Guide > Using FiftyOne Datasets > Object detection](https://docs.voxel51.com/user_guide/using_datasets.html#object-detection) for detail format
        ori_imshapes = ori_imshapes if ori_imshapes is not None else [x.shape[2:]] * x.shape[0]
        for pred, ori_imshape in zip(preds, ori_imshapes):
            if pred is None or pred.shape[0] == 0: ## nil pred, keep not changed
                continue
            
            ##>>>> rescale the box to original image size
            pred[:, :4] = scale_boxes(x.shape[2:], pred[:, :4], ori_imshape).round() ## still in xyxy
            
            ##>>>> normalize the boxes to [0, 1]
            gn = torch.tensor(ori_imshape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            ## NOTE: tlwh means top-left corner and wh, tlwhn means normalized tlwh according to original imgsize;
            pred[:, :4] = (xyxy2tlwh(torch.tensor(pred[:, :4]).view(-1, 4)) / gn)  # normalized tlwhn
        
    
        ##>>>> now just return the preds, not in fiftyone format
        ## NOTE: output shape will be of shape (bs, nboxs, 6), of type list(torch.Tensor), each tensor of shape (nboxs, 6), 6 is of form [xywhn, conf, cls]
        return preds
    
    def preds_to_fodetectionslist(self, preds: list, label_mappings:dict=None):
        """ convert preds to fo detections class

        Args:
            preds (list): list of tensor, each tensor will be batch_size * [xywhn, conf, classid]
            label_mappings (dict, optional): _description_. Defaults to None.
        """
        ##>>>> update class name dict
        ## use class level label_mappings if not specified
        label_mappings = label_mappings if label_mappings is not None else self._label_mappings
        
        names_dict = self._model.names.copy() ## the basic dict
        if label_mappings is not None:
            names_dict = {class_id: class_name if label_mappings is None or class_name not in label_mappings else label_mappings[class_name] for class_id, class_name in names_dict.items() }
            
        ##>>>> convert to fiftyone class
        fo_detections_list = []
            
        for pred in preds:
            if pred is None or pred.shape[0] == 0:
                fo_detections_list.append(fo.Detections(detections=[]))
                continue
            
            pred = pred.cpu().numpy() ## make sure in cpu
            
            ##>>>> convert to fiftyone class
            tlwhn_boxes = pred[:, :4].tolist()
            confs = pred[:, 4].tolist()
            classids = np.asarry(pred[:, 5], int).tolist()
            
            labels = [names_dict[classid] for classid in classids]
            
            detections = [fo.Detection(label=label, bounding_box=box, confidence=conf) for label, box, conf in zip(labels, tlwhn_boxes, confs)]
            
            fo_detections_list.append(fo.Detections(detections=detections))
            
        return fo_detections_list
    
class BatchDataLoader:
    def __init__(self, dataset, batch_size=64, img_size=640, transforms=None) -> None:
        """ batch data loader for yolov5 model 
        @Notes:
            better to specify the `transform` explicitly, since it's not always the same as the model's default transform
        """
        ##>>>> param setup
        ## to make code clear, ensure the transforms are set outside
        if transforms is None:
            raise Exception("required param `transforms` can not be None, MUST be specified")
        
        if not isinstance(dataset, (fo.Dataset, fo.DatasetView)):
            raise Exception(f"dataset must be of type `fiftyone.core.dataset.Dataset` or `fiftyone.core.dataset.DatasetView`, but got {type(dataset)}")
        
        if not dataset.has_field("filepath"):
            raise Exception(f"dataset must have field `filepath`, but got {dataset.get_field_schema()}")
        
        if not dataset.has_field("id"):
            raise Exception(f"dataset must have field `id`, but got {dataset.get_field_schema()}")
        
        ##>>>> basic setup
        ## NOTE: for batch loader, only need the id and filepath
        self.dataset = dataset.select_fields("id", "filepath")
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
        images, paths, ori_imshapes, sampleids = [], [], [], []
        for i in range(self.batch_size):
            ## NOTE: maybe more efficient only select the required fields
            cur_sample = self.dataset[self.count]
            cur_imfile = cur_sample.filepath
            path, im, im0, *_ = self._get_image(cur_imfile)
            self.count += 1
            
            images.append(im)
            paths.append(path)
            ## avoid passing the whold original image, just the shape
            ori_imshapes.append(im0.shape) ## NOTE: shape will be (h, w, c), not imgsize(w, h)
            sampleids.append(cur_sample.id)
            
            if len(images) < self.batch_size and self.count < self.nf:
                continue
            
            ##>>>> expand the batch dim for images
            image_tensor = torch.stack(images, 0)
            return image_tensor, paths, ori_imshapes, sampleids
            
        image_tensor = torch.stack(images, 0)
        return images, paths, ori_imshapes, sampleids
    
    def _get_image(self, img_path:str):
        im0 = cv2.imread(img_path)  # BGR
        assert im0 is not None, f"Image Not Found {img_path}"
        s = f"image {self.count}/{self.nf} {img_path}: "
        
        im = self.transforms(im0)  # transforms
        # if self.transforms:
        # else:
        #     im = letterbox(im0, self.img_size, 1, auto=True)[0]  # padded resize
        #     im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        #     im = np.ascontiguousarray(im)  # contiguous
            
        return img_path, im, im0, None, s

##-----------------------------------------------
##---- workflows
def compute_labels(dataset: Union[fo.Dataset, fo.DatasetView], 
                   model: nn.modules, 
                   label_field:str=None,
                   batch_size:int=64, 
                   img_size:int=640,
                   label_mappings:dict=None,
                   tag = None,
                   conf_thres=0.25,
                   iou_thres=0.45):
    """ support batch computing for convenience 
    @Args:
        `batch_size`: the batch_size for the model to do each prediction, decided by available GPU mem
        `chunk_size`: decided by available CPU mem, indicates how frequently to update the dataset
        `tag`: whether to append specified tag to those samples / labels?
    """
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
    
    print((
        f"inference cfg ==>\n"
        f"save to label field: {label_field}, batch_size: {batch_size}, append tag: {tag}\n"
        f"with rectified_chunk_size / total: {batch_size}/{len(dataset)}.\n"
        # f"classes:{classes}"
    ))
    
    ##>>>> model setup
    model.eval() ## duplicate but ensure val mode
    model.warmup((batch_size, 3, img_size, img_size))
    
    ##>>>> setup dataset
    det_tfm = DetectionTransform(img_size=img_size, stride=model.stride)
    
    dataloader = BatchDataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        img_size=img_size, 
        transforms=det_tfm, ## for detection now, same as ultralytics default transform
    )
    
    ##>>>> action loop
    ## NOTE: ims will be tensor with shape (batch_size, c, h, w)
    ## and better exeute in batch mode in this functions
    for imgs, paths, ori_imshapes, sampleids in tqdm.tqdm(dataloader, total=len(dataloader), desc="batch computing labels"):
        
        ##>>>> predict; output is list of tensors, each tensor for each image
        preds = model.predict(imgs, conf_thres=conf_thres, iou_thres=iou_thres, ori_imshapes=ori_imshapes)
        
        fo_detections_list = model.preds_to_fodetectionslist(preds, label_mappings=label_mappings, tag=tag)
        
        ##>>>> batch update the dataset
        batch_view = dataset.select(sample_ids=sampleids)
        
        values = {sampleid: fo_detections for sampleid, fo_detections in zip(sampleids, fo_detections_list) }
        
        ## maybe more efficient by `id` than by `filepath``
        batch_view.set_values(field_name= label_field, values=values, key_file="id")
        
       
    ##>>>> workon the whole dataset
    dataset.save() ## ensure save
    print("job done!")
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
        "--batch_size",
        type=int,
        default=32,
        help="batch size",
    )
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
    
    dataset = fo.load_dataset(args.fiftyone_dsname).take(70, seed=5151) ## for test
    
    label_mappings = None if "det_label_mapping" not in dataset.info else dataset.info["det_label_mapping"]
    print(f"label mappings: {label_mappings}, dataset detail:\n{dataset} ")
    
    ##>>>> model workflow
    model = DetectionModelWrapper(
        weights=args.weights,
        device=args.device,
        label_mappings=label_mappings,
    )
    
    compute_labels(
        dataset=dataset,
        model=model,
        label_field=args.label_field,
        batch_size=args.batch_size,
        img_size=args.imgsz,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
    )
    
    pass
