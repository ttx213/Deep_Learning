import torch
import torch.nn as nn
from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import LOSSES_REGISTRY
from .hungary import HungarianMatcher 
import numpy as np

@LOSSES_REGISTRY.register()
class SoftLoss(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        soft_loss,
    ):
        super(SoftLoss, self).__init__()
        self.soft_loss = soft_loss

    @classmethod
    def from_config(cls, cfg):
        return {
            "soft_loss": cfg.LOSSES.SOFT
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, outputs_dict):
        ret  = {}
        logits = outputs_dict[kfg.G_LOGITS]
        idxs = outputs_dict[kfg.VOCAB_IDS]
        soft_target = outputs_dict[kfg.VOCAB_PROBS]
        mask = outputs_dict[kfg.FILLMASKS]

        topk = -1.0 * logits.gather(-1, idxs) * mask[..., None] #(128, 22, 50)
        output = soft_target * topk
        loss = torch.sum(output) / torch.sum(mask)
        ret.update({'Soft Loss(G)': self.soft_loss*loss.float()})
        return ret



