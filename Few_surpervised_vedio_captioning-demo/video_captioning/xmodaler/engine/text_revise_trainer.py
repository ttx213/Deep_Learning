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
__all__ = ['TextReviseTrainer']

@ENGINE_REGISTRY.register()
class TextReviseTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super(TextReviseTrainer, self).__init__(cfg)
        self.K = cfg.MODEL.ROBERTA.K
        self.rbt_tknzr = RobertaTokenizer.from_pretrained(cfg.MODEL.ROBERTA.PATH, do_lower_case=False)
        self.iter_step = cfg.MODEL.ITER_STEP
        self.device = cfg.MODEL.DEVICE
        self.attribute = cfg.MODEL.ATTRIBUTE
        self.cls_thld = cfg.MODEL.CLS_THLD
        self.max_mask_ratio = cfg.MODEL.MAX_MASK_RATIO
        self.C = cfg.MODEL.C
        self.fixed_span_len = cfg.MODEL.FIXED_SPAN_LEN
        self.step_size =  cfg.MODEL.STEP_SIZE    

    def run_step(self, data):
        device = self.device
        start = time.perf_counter()
        try:
            data = next(self._train_data_loader_iter)
        except StopIteration:
            self._train_data_loader_iter = iter(self.train_data_loader)
            data = next(self._train_data_loader_iter)
        data_time = time.perf_counter() - start
        data = comm.unwrap_model(self.model).preprocess_batch(data)
        cand_inps_sents = data['CAND_INPS_SENTS']
        edit_track = [[] for _ in range(len(data))]
        for step in range(self.iter_step):#一共经历四轮修改
            torch.cuda.empty_cache()
            inps = [self.rbt_tknzr.tokenize(sent, add_prefix_space=True)[:50] for sent in cand_inps_sents] #分词
            
            if self.attribute == 'formality':
                abbr_pos = self.model.select_abbr_span(inps) #搜索缩写片段
            #获取文本序列，同时填充到相同长度
            cand_inps, cand_lens, _, _ = collate_no_tokenize(inps, self.rbt_tknzr.convert_tokens_to_ids, device=device) #(2, 48) [17, 48] (2, 48) (2)
            bsz, seqlen = cand_inps.size() #2, 48
            self.model.rbt_model.output_hidden_states = True
            
            #获取attribute head的值和hidden states
            attr_val_org, hid_states_org = self.model.cal_attr(cand_inps, hook_hid_grad=True) #(2, 2) (2, 48, 768)*12
            self.model.rbt_model.output_hidden_states = False

            #计算attribute head损失
            self.optimizer.zero_grad()
            loss = F.cross_entropy(attr_val_org, torch.ones(bsz).long().to(device))
            loss.backward()
            # self.optimizer.step()

            attr_val = F.softmax(attr_val_org, dim=1).cpu()
            del attr_val_org
            
            attr_val_mask = torch.where(attr_val[:, 1] > self.cls_thld, torch.zeros(bsz),
                                        torch.ones(bsz)).bool().tolist() #[False, True]

            attr_scores = attr_val[:, 1].tolist()
            
            for i, (sent, attr_score) in enumerate(zip(cand_inps_sents, attr_scores)): #记录每个句子的得分情况
                ex = {'sent': sent, 'score': attr_score}
                edit_track[i].append(ex)

            ent_mask = get_entity_mask(cand_inps_sents, cand_inps, self.rbt_tknzr, add_prefix_space=True) #(2, 48) 生成已命名的实体掩码
            grad_mask = (cand_inps.ne(self.model.pad_idx)) * (cand_inps.ne(self.model.bos_idx)) * \
                        (cand_inps.ne(self.model.eos_idx))
            
            del cand_inps, ent_mask

            for i, state in enumerate(hid_states_org):
                norm = torch.norm(state.grad, dim=-1).unsqueeze(2) #(2, 48, 1) 生成2范式
                norm = torch.where(norm > 0, norm, torch.full_like(norm, 1e-10)) #(2, 48, 1)
                if i == 0:
                    max_span_len = torch.floor((cand_lens - 2) * self.max_mask_ratio) #[4, 13] 获取最大文本片段长度
                    max_span_len = torch.where(max_span_len < 1, torch.ones_like(max_span_len), max_span_len) #[4, 13] 
                    emb_ngram_top1 = get_ngram_topk(norm.squeeze(-1), grad_mask, self.C, max_span_len, self.fixed_span_len) #[[7, 8], [13]]
                    break
            
            del grad_mask, hid_states_org
            self.model.zero_grad()

            if self.attribute == 'formality':
                ins_pos_l = [item if item else emb_ngram_top1[i] for i, item in enumerate(abbr_pos)]
            elif self.attribute == 'simplicity':
                ins_pos_l = emb_ngram_top1 #[[7, 8], [13]]
            #在每个需要编辑的文本片段之后添加['<lm-mask>']
            cand_inps, cand_lens, _, ins_pos = collate_inp_mask_after_span(inps, self.rbt_tknzr.convert_tokens_to_ids,
                                                                           ins_pos_l,
                                                                           device=device) #(2, 49) [18, 49] (2, 49) [tensor([0, 0, 0, 1, 1]), tensor([ 7,  8,  9, 13, 14])]
        
            bsz, seqlen = cand_inps.size() #2, 49
            
            self.model.rbt_model.output_hidden_states = True
            attr_val_org, hid_states_pad = self.model.cal_attr(cand_inps, hook_hid_grad=True) #(2, 2) (2, 49, 768)*12
            self.model.rbt_model.output_hidden_states = False

            self.optimizer.zero_grad()
            loss = F.cross_entropy(attr_val_org, torch.ones(bsz).long().to(device))
            loss.backward()
            # self.optimizer.step()
            del attr_val_org

            cand_ins_inps = cand_inps.index_put(ins_pos, torch.tensor(self.model.mask_idx).to(device)) #按索引赋值，将所有编辑文本片段改为mask

            for i, state in enumerate(hid_states_pad):
                if i == 0: continue
                norm = torch.norm(state.grad, dim=-1).unsqueeze(2) #(2, 49, 1) torch.norm 求二范式
                norm = torch.where(norm > 0, norm, torch.full_like(norm, 1e-10)) #(2, 49, 1)
                hid_states_pad[i] = state - self.step_size * state.grad / norm 

            cand_mask = cand_ins_inps.eq(self.model.mask_idx)#
            cand_inps = self.model.revise(self.K, cand_ins_inps, cand_mask, \
                                    ins_pos, memory_bank=hid_states_pad[1:]) #编辑文本片段 (2, 49)
            del hid_states_pad
            self.model.zero_grad()
        
            cand_inps = cand_inps[:bsz] #(2, 49)
            mid_cand_inps_toks = ids2sents(cand_inps.view(-1, cand_inps.size(-1)), self.rbt_tknzr, cand_lens) #将id转化为单词
            edited_cand_inps_sents = [self.rbt_tknzr.convert_tokens_to_string(x).lstrip() for x in mid_cand_inps_toks] #转化为完整句子

            if self.attribute == 'formality':
                for i, (val, has_abbr) in enumerate(zip(attr_val_mask, abbr_pos)):
                    if has_abbr or val:
                        cand_inps_sents[i] = edited_cand_inps_sents[i]
            elif self.attribute == 'simplicity':
                for i, val in enumerate(attr_val_mask):
                    if val:
                        cand_inps_sents[i] = edited_cand_inps_sents[i]

            if step == self.iter_step - 1:
                tknzd_inps = [self.rbt_tknzr.tokenize(sent, add_prefix_space=True) for sent in cand_inps_sents]
                cand_inps, cand_lens, _, _ = collate_no_tokenize(tknzd_inps, self.rbt_tknzr.convert_tokens_to_ids, device=device)
                bsz, seqlen = cand_inps.size()
                self.model.rbt_model.output_hidden_states = True
                with torch.no_grad():
                    attr_val_org = self.model.cal_attr(cand_inps, hook_hid_grad=False)[0]
                    attr_scores = F.softmax(attr_val_org, dim=1)[:, 1].cpu().tolist()
                self.model.rbt_model.output_hidden_states = False
                del attr_val_org, cand_inps
                for i, (sent, attr_score) in enumerate(zip(cand_inps_sents, attr_scores)):
                    ex = {'sent': sent, 'score': attr_score}
                    edit_track[i].append(ex)
                
            edit_spans = []
            
            for i, tknz_sent in enumerate(inps):
                span = ins_pos_l[i]
                topk_tokens = tknz_sent[span[0] - 1: span[-1]]
                edit_spans.append((topk_tokens, span)) #记录修改片段和所在位置
        
        results = []
        for sent in edit_track:
            results.append(sorted(sent, key=lambda x: x['score'])[-1]['sent'])
        
        return results