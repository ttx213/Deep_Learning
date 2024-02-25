import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from .decoder import Decoder
# from .containers import Module
from ..layers.gpt2 import GPT2Model, GPT2LMHead
from .build import DECODER_REGISTRY
import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
from torch.nn import functional as F


__all__ = ["GPT2Decoder"]

@DECODER_REGISTRY.register()

class GPT2Decoder(Decoder):
    @configurable
    def __init__(
        self, 
        *,
        padding_idx: int,
        tau: float,
        transformer,
        n_embd
    ):
        super(GPT2Decoder, self).__init__()
        self.transformer = transformer
        self.lm_head = GPT2LMHead(transformer.wte.weight, n_embd)
        self.padding_idx = padding_idx
        self.tau = tau
        # self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).bool())

    @classmethod
    def from_config(cls, cfg):
        transformer = GPT2Model(cfg)
        return{
            "transformer": transformer,
            "padding_idx": cfg.MODEL.GPT2.PADDING_IDX,
            "tau": cfg.MODEL.GPT2.TAU,
            "n_embd": cfg.MODEL.GPT2.N_EMBD,
        }
    
    @classmethod
    def add_config(cls, cfg):
        pass

    def set_tied(self):
        """ Make sure we are sharing the embeddings
        """
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    def forward(self, batched_inputs, encoder_output=None, mask_encoder=None, position_ids=None, token_type_ids=None, lm_labels=None, past=None, model_type='train'):
        ret = {}
        input_ids = batched_inputs[kfg.G_TOKENS_IDS] #(640)
        if len(input_ids.size()) == 1:
            input_ids = input_ids.unsqueeze(1) #(640, 1)
        mask_encoder = batched_inputs[kfg.EXT_ATT_MASKS].gt(0) #(128, 1, 1, 40) / (640, 1, 1, 40)
        encoder_output = batched_inputs[kfg.ATT_OUTS] #(128, 3, 40, 768)
        b_s, seq_len = input_ids.shape[:2] #128, 21/ 128, 1
        mask_queries = (input_ids != self.padding_idx).unsqueeze(-1).float() #(128, 21, 1) / (128, 1, 1)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input_ids.device),
                                         diagonal=1) #上三角矩阵 (21, 21) / (1, 1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, 21, 21) / (1, 1, 1, 1)
        mask_self_attention = mask_self_attention + (input_ids == self.padding_idx).unsqueeze(1).unsqueeze(1).bool()
        mask_self_attention = mask_self_attention.gt(0)  # (128, 1, 21, 21)/ (128, 1, 1, 1)

        if kfg.HISTORY_STATES in batched_inputs:
            past = batched_inputs[kfg.HISTORY_STATES]
        else:
            past=None
        
        hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past,mask_queries=mask_queries,
                                                    encoder_output=encoder_output,mask_encoder=mask_encoder, 
                                                    mask_self_attention= mask_self_attention, tau = self.tau) #(128, 1, 768) (2, 128, 12, 1, 64)*12
        ret.update({ kfg.G_HIDDEN_STATES: hidden_states})
        ret.update({ kfg.HISTORY_STATES: presents}) #(2, 5, 12, 1, 64), (2, 5, 12, 2, 64) ... (2, 5, 12, 20, 64)
        return ret