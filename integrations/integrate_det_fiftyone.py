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

##---- import local moduls


##-----------------------------------------------
##---- vars

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
        self.device = select_device(device) if device is None else device
        
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

##-----------------------------------------------
##---- workflows
def compute_labels(dataset: Union[fo.Dataset, fo.DatasetView], 
                   model: nn.modules, 
                   label_field:str=None,
                   batch_size:int=64, 
                   chunk_size:int = 3200,
                   img_size:int=640,
                   label_mappings:dict=None,
                   tag = None):
    """ support batch computing for convenience 
    @Args:
        `batch_size`: the batch_size for the model to do each prediction, decided by available GPU mem
        `chunk_size`: decided by available CPU mem, indicates how frequently to update the dataset
        `tag`: whether to append specified tag to those samples / labels?
    """
    if label_field is None:
        raise Exception("required param `label_field` can not be None, MUST be specified")
    
    chunk_size = int(np.ceil(chunk_size / batch_size)) * batch_size ## make chunk_size dividable by batch_size
    print((
        f"inference cfg ==>\n"
        f"save to label field: {label_field}, batch_size: {batch_size}, append tag: {tag}\n"
        f"with rectified_chunk_size / total: {chunk_size}/{len(dataset)}.\n"
        # f"classes:{classes}"
    ))
    
    ##>>>> predict the results
    with fo.ProgressBar() as pb:
        for step in pb(range(0, dataset.count(), chunk_size)):
            ## for each chunk
            chunk_view = dataset[step: step+chunk_size]
            
            chunk_filepaths = chunk_view.values("filepath")
            chunk_dl = trainval_dl.test_dl(chunk_filepaths, bs=bs)
            
            # Perform inference
            chunk_preds, _ = fastai_learner.get_preds(dl=chunk_dl)
            if chunk_preds is None:
                print("chunk_preds is None for chunk: {}:{}".format(step, step+chunk_size))
                continue
                
            chunk_preds = chunk_preds.numpy()

            ##>>>> save back to fiftyone
            for sample, scores, fastai_filepath in zip(
                chunk_view.select_fields(["id", "filepath"]).iter_samples(autosave=True, progress=True),
                chunk_preds,
                chunk_dl.items
            ):
                if sample["filepath"] != fastai_filepath:
                    print("potential error, fiftyone filepath {} not eq fastai filepath {}".format(sample["filepath"], fastai_filepath))
                    continue
                    
                target = np.argmax(scores)
                sample[save_to_label] = fo.Classification(
                    label=classes[target],
                    confidence=scores[target],
                    logits=np.log(scores),
                )
                if tag is not None:
                    sample.tags.append(tag)

    print("job done!")
    
    pass
    


##---- xxxx --------------------- bakup workflows
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

def create_argparser():
    """ make the argument parser for this script 
    NOTE:
        make it a function, so that it can be used in other scripts, eg., notebooks
    """
    parser = argparse.ArgumentParser(description="Wrap yolo classification model into fiftyone usage")
    parser.add_argument(
        "--data",
        type=str,
        default="../datasets/mnist",
        help="dataset dir",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov5s-cls.pt",
        help="model.pt path(s)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="batch size",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=224,
        help="inference size (pixels)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="cuda device, i.e. 0 or 0,1,2,3 or cpu",
    )
    return parser

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
    ##>>>> parse args
    # unit_test_cmp_with_yolov5_predict()
    
    unit_test_batch_predict()
    
    pass
