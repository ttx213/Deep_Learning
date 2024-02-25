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

__all__ = ["MSVDMCMCXLCLIPDataset"]

@DATASETS_REGISTRY.register() #调用自定义函数
class MSVDMCMCXLCLIPDataset:
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
        keywords_num: int,
        sample_prob: float,
        sentence_nums: int,
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
        self.keywords_num = keywords_num
        self.sample_prob = sample_prob
        self.sentence_nums = sentence_nums

    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        ann_files = {
            "train": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "msvd_caption_anno_clipfilter_train.pkl"),
            "val": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "msvd_caption_anno_clipfilter_val.pkl"),
            "test": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "msvd_caption_anno_clipfilter_test.pkl")
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
            "sentence_nums": cfg.DATALOADER.SENTENCE_NUMS,
            "c3d": cfg.DATALOADER.C3D,
            "object_feats_folder": cfg.DATALOADER.OBJECT_FEATS_FOLDER,
            "faster_r_cnn": cfg.DATALOADER.FASTER_R_CNN,
            "max_object_num":cfg.DATALOADER.MAX_OBJECT_NUM,
            "keywords_num": cfg.MODEL.CLIP.KEYWORDS_NUM,
            "sample_prob": cfg.DATALOADER.SAMPLE_PROB,
        }
        return ret

    def load_data(self, cfg): #载入标注文件
        datalist = pickle.load(open(self.anno_file, 'rb'), encoding='bytes')
        if self.stage == 'train':
            if self.train_percentage < 1.0:
                np.random.shuffle(datalist)#随机打乱序列
                datalist = datalist[:int(len(datalist)*self.train_percentage)]

            expand_datalist = []
            sentence_nums = self.sentence_nums
            if sentence_nums > 0:
                for data in datalist:
                    for token_id, target_id in zip(data['tokens_ids'][:sentence_nums], data['target_ids'][:sentence_nums]): #(21,) (21,)
                        expand_datalist.append({
                            'video_id': data['video_id'],
                            'tokens_ids': np.expand_dims(token_id, axis=0), #扩展维度 (1,21)
                            'target_ids': np.expand_dims(target_id, axis=0),
                            'attr_pred': data['attr_pred'],
                            'attr_labels': data['attr_labels'],
                            'missing_labels': data['missing_labels'],
                        }) #将每个语句进行拆分
            else:
                for data in datalist:
                    for token_id, target_id in zip(data['tokens_ids'], data['target_ids']): #(21,) (21,)
                        expand_datalist.append({
                            'video_id': data['video_id'],
                            'tokens_ids': np.expand_dims(token_id, axis=0), #扩展维度 (1,21)
                            'target_ids': np.expand_dims(target_id, axis=0),
                            'attr_pred': data['attr_pred'],
                            'attr_labels': data['attr_labels'],
                            'missing_labels': data['missing_labels'],
                        }) #将每个语句进行拆分
            datalist = expand_datalist
        return datalist
    
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
        
        semantics_ids = dataset_dict['attr_pred'] #array([354, 395, 685, 117, 597,  87,  27,  93])
        semantics_labels = dataset_dict['attr_labels'] #array([1000, 1000, 1000,  117, 1000, 1000, 1000, 1000])
        semantics_miss_labels_arr = dataset_dict['missing_labels'] #[1, 386, 3, 4, 9, 907, 13, 17, 273, 657, 22, 25, 26, 28, ...]
        semantics_miss_labels = np.zeros((self.keywords_num+1, )).astype(np.int64) #0*(1001)
        
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
            return ret #IDS: 1202, ATT_FEATS: (40, 2048), G_TOKENS_TYPE: (21)

        sent_num = len(dataset_dict['tokens_ids']) #获取语句个数
        if sent_num >= self.seq_per_img: #为每个视频随机分配一个描述语句
            selects = random.sample(range(sent_num), self.seq_per_img)#截取列表指定长度的随机数
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
        
        dict_as_tensor(ret) #将array转为tensor
        return ret
        