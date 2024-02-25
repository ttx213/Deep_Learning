# Copyright 2022 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import os
import copy
import pickle
import random
from tqdm import tqdm
import numpy as np
from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import read_np, dict_as_tensor, boxes_to_locfeats
from .mscoco import MSCoCoDataset
from ..build import DATASETS_REGISTRY

__all__ = ["MSCoCoCOSNetDataset"]

@DATASETS_REGISTRY.register()
class MSCoCoCOSNetDataset(MSCoCoDataset):
    @configurable
    def __init__(
        self,
        stage: str,
        anno_file: str,
        seq_per_img: int,
        max_feat_num: int,
        max_seq_len: int,
        obj_classes: int,
        feats_folder: str,
        relation_file: str,
        gv_feat_file: str,
        attribute_file: str,
        sample_prob: float,
    ):
        super(MSCoCoCOSNetDataset, self).__init__(
            stage,
            anno_file,
            seq_per_img, 
            max_feat_num,
            max_seq_len,
            feats_folder,
            relation_file,
            gv_feat_file,
            attribute_file
        )
        self.obj_classes = obj_classes
        self.sample_prob = sample_prob

    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        ann_files = {
            "train": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "cosnet", "mscoco_caption_anno_clipfilter_train.pkl"),
            "val": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "cosnet", "mscoco_caption_anno_clipfilter_val.pkl"),
            "test": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "cosnet", "mscoco_caption_anno_clipfilter_test.pkl")
        }
        ret = {
            "stage": stage,
            "anno_file": ann_files[stage],
            "seq_per_img": cfg.DATALOADER.SEQ_PER_SAMPLE,
            "max_feat_num": cfg.DATALOADER.MAX_FEAT_NUM,
            "feats_folder": cfg.DATALOADER.FEATS_FOLDER,
            "relation_file": cfg.DATALOADER.RELATION_FILE,
            "gv_feat_file": cfg.DATALOADER.GV_FEAT_FILE,
            "attribute_file": cfg.DATALOADER.ATTRIBUTE_FILE,
            "max_seq_len": cfg.MODEL.MAX_SEQ_LEN,
            "obj_classes": cfg.MODEL.COSNET.NUM_CLASSES,
            "sample_prob": cfg.DATALOADER.SAMPLE_PROB,
        }
        return ret

    def sampling(self, semantics_ids_arr, semantics_labels_arr, semantics_miss_labels_arr):
        for i in range(len(semantics_ids_arr)):
            semantics_ids = semantics_ids_arr[i] #array([ 32,  24, 290,  11])
            semantics_labels = semantics_labels_arr[i] #array([ 32, 906, 290,  11])
            semantics_miss_labels = semantics_miss_labels_arr[i] #(907)*5

            num_classes = len(semantics_miss_labels) - 1 #906
            gt_labels1 = list(np.where(semantics_miss_labels > 0)[0]) #[29, 78, 136, 148, 231, 281, 439, 627]
            gt_labels2 = list(semantics_ids[semantics_labels != num_classes]) #[32, 290, 11] 将无关语义词剔除
            gt_labels = set(gt_labels1 + gt_labels2) #{32, 290, 231, 136, 11, 78, 627, 148, 439, 281, 29}
            
            for j in range(len(semantics_ids)):
                if random.random() < self.sample_prob: #引入替换概率
                    ori_semantics_id = semantics_ids_arr[i][j] #11
                    rnd_idx = np.random.randint(num_classes) #185
                    semantics_ids_arr[i][j] = rnd_idx #将11替换为185

                    if rnd_idx in gt_labels:
                        semantics_labels_arr[i][j] = rnd_idx
                        semantics_miss_labels_arr[i][ori_semantics_id] = 1
                        semantics_miss_labels_arr[i][rnd_idx] = 0
                    else:
                        semantics_labels_arr[i][j] = num_classes #906
                        if ori_semantics_id in gt_labels:
                           semantics_miss_labels_arr[i][ori_semantics_id] = 1 #将缺失label的位置置为1

        return semantics_ids_arr, semantics_labels_arr, semantics_miss_labels_arr

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image_id = dataset_dict['image_id'] #'234713'
        
        if len(self.feats_folder) > 0:
            feat_path = os.path.join(self.feats_folder, image_id + '.npz')
            content = read_np(feat_path)
            att_feats = content['features'][0:self.max_feat_num].astype('float32')
            global_feat = content['g_feature']

            ret = { 
                kfg.IDS: image_id, 
                kfg.ATT_FEATS: att_feats,
                kfg.GLOBAL_FEATS: global_feat
            }

        else:
            # dummy ATT_FEATS
            ret = { kfg.IDS: image_id, kfg.ATT_FEATS: np.zeros((1,1)) }

        semantics_ids = dataset_dict['attr_pred'] #array([ 32,  24, 290,  11])
        semantics_labels = dataset_dict['attr_labels'] #array([ 32, 906, 290,  11])
        semantics_miss_labels_arr = dataset_dict['missing_labels'] #[248, 288, 5, 102, 555, 44, 749, 789, 120, 794]
        semantics_miss_labels = np.zeros((self.obj_classes+1, )).astype(np.int64) #0*(907)
        for sem in semantics_miss_labels_arr:
            semantics_miss_labels[sem] = 1
        
        if self.stage != 'train':
            semantics_ids = [ semantics_ids.astype(np.int64) ]

            ret.update({ 
                kfg.SEMANTICS_IDS: semantics_ids,
            })

            g_tokens_type = np.ones((self.max_seq_len,), dtype=np.int64)
            ret.update({ kfg.G_TOKENS_TYPE: g_tokens_type })
            dict_as_tensor(ret)
            return ret
        
        sent_num = len(dataset_dict['tokens_ids']) #5
        if sent_num >= self.seq_per_img: 
            selects = random.sample(range(sent_num), self.seq_per_img) #5
        else:
            selects = random.choices(range(sent_num), k = (self.seq_per_img - sent_num))
            selects += list(range(sent_num))

        semantics_ids = [ semantics_ids.astype(np.int64) for i in selects ] #array([ 32,  24, 290,  11])*5
        semantics_labels = [ semantics_labels.astype(np.int64) for i in selects ] #array([ 32, 906, 290,  11])*5
        semantics_miss_labels = [ semantics_miss_labels.astype(np.int64) for i in selects ] #(907)*5

        semantics_ids, semantics_labels, semantics_miss_labels = self.sampling(semantics_ids, semantics_labels, semantics_miss_labels)
        ret.update({ 
            kfg.SEMANTICS_IDS: semantics_ids,
            kfg.SEMANTICS_LABELS: semantics_labels,
            kfg.SEMANTICS_MISS_LABELS: semantics_miss_labels,
        })
        
        tokens_ids = [ dataset_dict['tokens_ids'][i,:].astype(np.int64) for i in selects ]
        target_ids = [ dataset_dict['target_ids'][i,:].astype(np.int64) for i in selects ]
        g_tokens_type = [ np.ones((len(dataset_dict['tokens_ids'][i,:]), ), dtype=np.int64) for i in selects ]

        ret.update({
            kfg.SEQ_PER_SAMPLE: self.seq_per_img,
            kfg.G_TOKENS_IDS: tokens_ids,
            kfg.G_TARGET_IDS: target_ids,
            kfg.G_TOKENS_TYPE: g_tokens_type,
        })
        dict_as_tensor(ret)
        return ret