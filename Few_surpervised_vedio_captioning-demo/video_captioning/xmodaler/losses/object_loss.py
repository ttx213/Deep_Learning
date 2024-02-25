import torch
import torch.nn as nn
from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import LOSSES_REGISTRY
from .hungary import HungarianMatcher 
import numpy as np

@LOSSES_REGISTRY.register()
class ObjectLoss(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        object_loss
    ):
        super(ObjectLoss, self).__init__()
        self.matcher = HungarianMatcher()
        self.eps = 1e-12
        self.object_loss = object_loss

    @classmethod
    def from_config(cls, cfg):
        return {
            'object_loss': cfg.LOSSES.OBJECT
        }

    @classmethod
    def add_config(cls, cfg):
        pass
    
    def cos_loss(self, pred, target):
        assert pred.shape == target.shape and pred.dim() == 2, \
            'expected pred.shape == target.shape, ' \
            'but got pred.shape == {} and target.shape == {}'.format(pred.shape, target.shape)
        pred_denom = torch.norm(pred, p=2, dim=-1, keepdim=True).clamp_min(self.eps).expand_as(pred) #clamp_min(a)设置下限，最小值为a (272, 768)
        pred = pred / pred_denom #(272, 768)
        target_denom = torch.norm(target, p=2, dim=-1, keepdim=True).clamp_min(self.eps).expand_as(target) #(272, 768)
        target = target / target_denom #(272, 768)
        ret = pred * target
        ret = 1.0 - ret.sum(dim=-1)
        ret = ret.sum()
        return ret

    def _get_src_permutation_idx(self, indices): #[(tensor([2, 4, 6]), tensor([1, 2, 0])), (tensor([0, 1, 7]), tensor([2, 1, 0])),...,]
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)]) # .full_like(src, i)构造值全为i，大小为src的矩阵 #(272)
        src_idx = torch.cat([src for (src, _) in indices]) #(272)
        return batch_idx, src_idx

    def forward(self, outputs_dict):
        ret  = {}
        objects = outputs_dict[kfg.OBJECT_ATT_FEATS]
        nouns = outputs_dict[kfg.G_OBJECT_IDS]
        indices = self.matcher(objects, nouns)
        src_idx = self._get_src_permutation_idx(indices) #[(272), (272)]
        new_objects = objects[src_idx] #(272, 768)
        targets = torch.cat([t['nouns_vec'][i] for t, (_, i) in zip(nouns, indices)], dim=0).cuda() #(272, 768)
        if np.any(np.isnan(new_objects.detach().cpu().numpy())):
            raise RuntimeError
        object_loss = self.cos_loss(new_objects, targets)

        ret.update({'ObjectLoss': self.object_loss*object_loss})

        return ret
    

