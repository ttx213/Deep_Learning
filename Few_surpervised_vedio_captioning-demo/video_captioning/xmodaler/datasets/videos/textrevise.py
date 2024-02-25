import os
import copy
import pickle
import random
import numpy as np
import torch
import json
from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import read_np, dict_as_tensor
from ..build import DATASETS_REGISTRY

__all__ = ["TextReviseDataset"]

@DATASETS_REGISTRY.register()
class TextReviseDataset:
    @configurable
    def __init__(
        self,
        stage: str,
        anno_file: str,
    ):
        self.stage = stage
        self.anno_file = anno_file

    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        ann_files = {
            "train": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "msrvtt.json"),
            "val": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "msrvtt.json"),
            "test": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "msrvtt.json")
        }

        ret = {
            "stage": stage,
            "anno_file": ann_files[stage],      
        }
        return ret
    
    def load_data(self, cfg): #载入标注文件
        datalist = json.load(open(self.anno_file, 'r'))
        expand_datalist = []
        i=0
        for data in datalist["annotations"]:
            if i==0:
                video_id = data['image_id']
                expand_datalist.append({
                    'video_id': str(data["image_id"]),
                    'caption': data["caption"]
                })
                i+=1
            
            if data['image_id'] != video_id:
                video_id = data['image_id']
                expand_datalist.append({
                    'video_id': str(data["image_id"]),
                    'caption': data["caption"]
                })
        
        datalist = expand_datalist
        return datalist
    
    def __call__(self, dataset_dict): #使得类实例对象可以像调用普通函数那样，以“对象名()”的形式使用
        dataset_dict = copy.deepcopy(dataset_dict) #深复制
        ret = { 
            kfg.IDS: dataset_dict['video_id'], 
            kfg.CAND_INPS_SENTS: dataset_dict['caption']
        }
        return ret