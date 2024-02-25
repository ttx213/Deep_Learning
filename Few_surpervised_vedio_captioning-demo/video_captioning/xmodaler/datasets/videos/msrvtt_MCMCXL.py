# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li, Jingwen Chen
@contact: yehaoli.sysu@gmail.com, chenjingwen.sysu@gmail.com
"""
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
from memory_profiler import profile

__all__ = ["MSRVTTMCMCXLDataset"]

@DATASETS_REGISTRY.register() #调用自定义函数
class MSRVTTMCMCXLDataset:
    @configurable
    def __init__(
        self,
        stage: str,
        anno_file: str,
        seq_per_img: int,
        max_feat_num: int,
        max_seq_len: int,
        feats_folder: str,
        motion_feats_folder: str,
        train_percentage: float,
        c3d: bool,
        object_feats_folder: str,
        faster_r_cnn: bool,
        max_object_num: int,

    ):
        self.stage = stage
        self.anno_file = anno_file
        self.seq_per_img = seq_per_img
        self.max_feat_num = max_feat_num
        self.feats_folder = feats_folder
        self.max_seq_len = max_seq_len
        self.train_percentage = train_percentage
        self.motion_feats_folder = motion_feats_folder
        self.c3d = c3d
        self.object_feats_folder = object_feats_folder
        self.faster_r_cnn = faster_r_cnn
        self.max_object_num = max_object_num

    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        ann_files = {
            "train": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "msrvtt_caption_anno_train.pkl"),
            "val": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "msrvtt_caption_anno_val.pkl"),
            "test": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "msrvtt_caption_anno_test.pkl")
        }
        ret = {
            "stage": stage,
            "anno_file": ann_files[stage],
            "seq_per_img": cfg.DATALOADER.SEQ_PER_SAMPLE,
            "max_feat_num": cfg.DATALOADER.MAX_FEAT_NUM,
            "feats_folder": cfg.DATALOADER.FEATS_FOLDER,
            "motion_feats_folder": cfg.DATALOADER.MOTION_FEATS_FOLDER, 
            "max_seq_len": cfg.MODEL.MAX_SEQ_LEN,
            "train_percentage": cfg.DATALOADER.TRAIN_PERCENTAGE,
            "c3d": cfg.DATALOADER.C3D,
            "object_feats_folder": cfg.DATALOADER.OBJECT_FEATS_FOLDER,
            "faster_r_cnn": cfg.DATALOADER.FASTER_R_CNN,
            "max_object_num":cfg.DATALOADER.MAX_OBJECT_NUM,
        }
        return ret

    def load_data(self, cfg): #载入标注文件
        datalist = pickle.load(open(self.anno_file, 'rb'), encoding='bytes')
        if self.stage == 'train':
            if self.train_percentage < 1.0:
                np.random.shuffle(datalist)#随机打乱序列
                datalist = datalist[:int(len(datalist)*self.train_percentage)]

            expand_datalist = []
            for data in datalist:
                for token_id, target_id in zip(data['tokens_ids'], data['target_ids']): #(21,) (21,)
                    expand_datalist.append({
                        'video_id': data['video_id'],
                        'tokens_ids': np.expand_dims(token_id, axis=0), #扩展维度 (1,21)
                        'target_ids': np.expand_dims(target_id, axis=0)
                    })
            datalist = expand_datalist
        return datalist
        
    def _sample_frame(self, atten_feats):
        interval = atten_feats.shape[0] / self.max_feat_num #1.95
        selected_indexes = [int(i * interval) for i in range(self.max_feat_num)]
        selected_frames = atten_feats[selected_indexes, :]
        return selected_frames

    def _sample_object(self, atten_feats):
        # interval = atten_feats.shape[0] / self.max_feat_num #1.95
        # selected_indexes = [int(i * interval) for i in range(self.max_feat_num)]
        selected_objects = atten_feats[:self.max_object_num, :]
        return selected_objects

    def __call__(self, dataset_dict): #使得类实例对象可以像调用普通函数那样，以“对象名()”的形式使用
        dataset_dict = copy.deepcopy(dataset_dict) #深复制
        video_id = dataset_dict['video_id']
        if self.c3d and self.faster_r_cnn:
            feat_path  = os.path.join(self.feats_folder, video_id + '.npy')
            motion_feat_path = os.path.join(self.motion_feats_folder, video_id + '.npy')
            object_feats_path = os.path.join(self.object_feats_folder, video_id + '.npy')
            content = read_np(feat_path) #获取视频特征
            motion_content = read_np(motion_feat_path) #获取动作特征
            object_content = read_np(object_feats_path) #获取目标特征
            att_feats = content['features'].astype('float32') #(57, 1536)
            motion_att_feats = motion_content['features'].astype('float32') #(57, 2048)
            object_att_feats = object_content['features'].astype('float32') #(89, 2048)

            if self.max_feat_num > 0 and att_feats.shape[0] > self.max_feat_num: 
                att_feats = self._sample_frame(att_feats) #均匀选择帧数，避免超过最大帧数
                assert att_feats.shape[0] == self.max_feat_num

            if self.max_feat_num > 0 and motion_att_feats.shape[0] > self.max_feat_num: 
                motion_att_feats = self._sample_frame(motion_att_feats) #均匀选择帧数，避免超过最大帧数
                assert motion_att_feats.shape[0] == self.max_feat_num

            if self.max_object_num > 0 and object_att_feats.shape[0] > self.max_object_num: 
                object_att_feats = self._sample_object(object_att_feats) #均匀选择目标数，避免超过最大目标数
                assert object_att_feats.shape[0] == self.max_object_num

            ret = {kfg.IDS: video_id, kfg.ATT_FEATS: att_feats, kfg.MOTION_ATT_FEATS: motion_att_feats, kfg.OBJECT_ATT_FEATS: object_att_feats}

        elif self.faster_r_cnn:

            object_feats_path = os.path.join(self.object_feats_folder, video_id + '.npy')
            motion_feat_path = os.path.join(self.motion_feats_folder, video_id +'.npy')

            object_content = read_np(object_feats_path) #获取目标特征
            motion_content = read_np(motion_feat_path) #获取动作特征
            object_att_feats = object_content['features'].astype('float32') #(89, 2048)
            motion_att_feats = motion_content['features'].astype('float32') #(16, 2048)

            if self.max_feat_num > 0 and object_att_feats.shape[0] > self.max_object_num: 
                object_att_feats = self._sample_object(object_att_feats) #均匀选择目标数，避免超过最大目标数
                assert object_att_feats.shape[0] == self.max_object_num
            
            if self.max_feat_num > 0 and  motion_att_feats.shape[0] > self.max_feat_num: 
                motion_att_feats = self._sample_frame(motion_att_feats) #均匀选择帧数，避免超过最大帧数
                assert motion_att_feats.shape[0] == self.max_feat_num

            ret = {kfg.IDS: video_id, kfg.ATT_FEATS: object_att_feats, kfg.MOTION_ATT_FEATS: motion_att_feats}

        elif self.c3d: 
            feat_path  = os.path.join(self.feats_folder, video_id + '.npy')
            motion_feat_path = os.path.join(self.motion_feats_folder, video_id +'.npy')
            content = read_np(feat_path) #获取视频特征
            motion_content = read_np(motion_feat_path) #获取动作特征
            att_feats = content['features'].astype('float32') #(40,2048)
            motion_att_feats = motion_content['features'].astype('float32') #(16, 4096)

            if self.max_feat_num > 0 and att_feats.shape[0] > self.max_feat_num: 
                att_feats = self._sample_frame(att_feats) #均匀选择帧数，避免超过最大帧数
                assert att_feats.shape[0] == self.max_feat_num

            if self.max_feat_num > 0 and  motion_att_feats.shape[0] > self.max_feat_num: 
                motion_att_feats = self._sample_frame(motion_att_feats) #均匀选择帧数，避免超过最大帧数
                assert motion_att_feats.shape[0] == self.max_feat_num
            
            ret = {kfg.IDS: video_id, kfg.ATT_FEATS: att_feats, kfg.MOTION_ATT_FEATS: motion_att_feats}
            
        else:
            feat_path  = os.path.join(self.feats_folder, video_id + '.npy')
            content = read_np(feat_path) #获取视频特征
            att_feats = content['features'].astype('float32') #(40,2048)
            if self.max_feat_num > 0 and att_feats.shape[0] > self.max_feat_num: 
                att_feats = self._sample_frame(att_feats) #均匀选择帧数，避免超过最大帧数
                assert att_feats.shape[0] == self.max_feat_num

            ret = {kfg.IDS: video_id, kfg.ATT_FEATS: att_feats}

        if self.stage != 'train':
            g_tokens_type = np.ones((self.max_seq_len,), dtype=np.int64)
            ret.update({ kfg.G_TOKENS_TYPE: g_tokens_type })
            dict_as_tensor(ret)
            return ret #IDS: 1202, ATT_FEATS: (40, 2048), G_TOKENS_TYPE: (21)

        sent_num = len(dataset_dict['tokens_ids']) #获取语句个数
        if sent_num >= self.seq_per_img: #为每个视频随机分配一个描述语句
            selects = random.sample(range(sent_num), self.seq_per_img)#截取列表指定长度的随机数
        else:
            selects = random.choices(range(sent_num), k = (self.seq_per_img - sent_num))
            selects += list(range(sent_num))

        tokens_ids = [ dataset_dict['tokens_ids'][i,:].astype(np.int64) for i in selects ]
        target_ids = [ dataset_dict['target_ids'][i,:].astype(np.int64) for i in selects ]
        g_tokens_type = [ np.ones((len(dataset_dict['tokens_ids'][i,:]), ), dtype=np.int64) for i in selects ]
        
        ret.update({
            kfg.SEQ_PER_SAMPLE: self.seq_per_img,
            kfg.G_TOKENS_IDS: tokens_ids,
            kfg.G_TARGET_IDS: target_ids,
            kfg.G_TOKENS_TYPE: g_tokens_type,
        })
        dict_as_tensor(ret) #将array转为tensor
        return ret