# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from ..layers.bert import BertLayer
from .build import ENCODER_REGISTRY

__all__ = ["TransformerEncoder"]

@ENCODER_REGISTRY.register()
class TransformerEncoder(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        num_hidden_layers: int,
        bert_layers,
    ):
        super(TransformerEncoder, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.layers = bert_layers

    @classmethod
    def from_config(cls, cfg):
        bert_layers = nn.ModuleList(
            [BertLayer(cfg) for _ in range(cfg.MODEL.BERT.NUM_HIDDEN_LAYERS)]
        )
        return {
            "num_hidden_layers": cfg.MODEL.BERT.NUM_HIDDEN_LAYERS,
            "bert_layers": bert_layers,
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs, mode=None):
        ret = {}
        if mode == None or mode == 'v':
            vfeats = batched_inputs[kfg.ATT_FEATS] #(16, 20, 768)
            ext_vmasks = batched_inputs[kfg.EXT_ATT_MASKS] #(16, 1, 1, 20)
            out = []
            for layer_module in self.layers:
                vfeats, _ = layer_module(vfeats, ext_vmasks)
                out.append(vfeats.unsqueeze(1))
            outs = torch.cat(out, 1) #(16, 6, 20, 768)
            ret.update({ kfg.ATT_FEATS: vfeats }) #(16, 20, 768)
            ret.update({ kfg.ATT_OUTS: outs }) #(16, 6, 20, 768)
        return ret