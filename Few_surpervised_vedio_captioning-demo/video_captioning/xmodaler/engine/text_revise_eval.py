import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import csv
from xmodaler.config import CfgNode as CN
from ..modeling.mtl_roberta.modeling_roberta import RobertaConfig, RobertaForMTL
from ..modeling.mtl_roberta.tokenization_roberta import RobertaTokenizer
from ..utils.text_revise_utils import get_entity_mask, collate_no_tokenize, ids2sents, \
    batchify, get_ngram_topk, collate_inp_mask_after_span
from .defaults import DefaultTrainer
from .build import ENGINE_REGISTRY
import time
import xmodaler.utils.comm as comm
from xmodaler.config import kfg
import json
__all__ = ['TextReviseEvaluator']

@ENGINE_REGISTRY.register()
class TextReviseEvaluator(DefaultTrainer):
    def __init__(self, cfg):
        super(TextReviseEvaluator, self).__init__(cfg)
        self.device = cfg.MODEL.DEVICE   

    def test(self, cfg, model, test_data_loader, evaluator, epoch):
        device = self.device
        results = []
        model = self.model
        for data in tqdm(test_data_loader):
            data = comm.unwrap_model(model).preprocess_batch(data)
            ids = data[kfg.IDS]

            cand_inps_sents = data[kfg.CAND_INPS_SENTS]
            
            for id, output in zip(ids, cand_inps_sents):
                results.append({cfg.INFERENCE.ID_KEY: int(id), cfg.INFERENCE.VALUE: output}) #[{'image_id': 1297, 'caption': 'a man is cutting a UNK'}, {'image_id': 1298, 'caption': 'a man is making a UNK'}]
        
        if evaluator is not None:
            eval_res = evaluator.eval(results, epoch)
        else:
            eval_res = ''