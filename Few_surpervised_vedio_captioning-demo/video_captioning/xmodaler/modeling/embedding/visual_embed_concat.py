# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import kfg
from ..layers.create_act import get_act_layer
from .build import EMBEDDING_REGISTRY

__all__ = ["ConcatVisualBaseEmbedding"]

@EMBEDDING_REGISTRY.register()
class ConcatVisualBaseEmbedding(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        in_dim: int,
        motion_in_dim: int,
        object_in_dim: int,
        out_dim: int,
        embeddings_dim: int,
        concat_method: str,
        **kwargs
    ):
        super(ConcatVisualBaseEmbedding, self).__init__()
        self.embeddings_act = kwargs.pop("embeddings_act", None)
        self.embeddings_norm = kwargs.pop("embeddings_norm", None)
        self.embeddings_dropout = kwargs.pop("embeddings_dropout", None)
        self.embeddings_pos = kwargs.pop('embeddings_pos', None)
        self.feats_proj = nn.Sequential(
            nn.Linear(in_dim, 2 * out_dim),
            nn.BatchNorm1d(out_dim * 2),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(out_dim * 2, out_dim)
        )
        self.motion_feats_proj = nn.Sequential(
            nn.Linear(motion_in_dim, 2 * out_dim),
            nn.BatchNorm1d(out_dim * 2),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(out_dim * 2, out_dim)
        )
        self.object_feats_proj = nn.Sequential(
            nn.Linear(object_in_dim, 2 * out_dim),
            nn.BatchNorm1d(out_dim * 2),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(out_dim * 2, out_dim)
        )

        # self.bilstm = nn.LSTM(input_size=out_dim * 2, hidden_size=out_dim//2, 
        #                     batch_first=True, bidirectional=True)
        
        self.embeddings = nn.Linear(embeddings_dim, out_dim)
        self.concat_method = concat_method
    @classmethod
    def from_config(cls, cfg):
        kwargs = {
            "in_dim": cfg.MODEL.VISUAL_EMBED.IN_DIM,
            "motion_in_dim": cfg.MODEL.VISUAL_EMBED.MOTION_IN_DIM,
            "object_in_dim": cfg.MODEL.VISUAL_EMBED.OBJECT_IN_DIM,
            "out_dim": cfg.MODEL.VISUAL_EMBED.OUT_DIM,
            "embeddings_dim": cfg.MODEL.VISUAL_EMBED.EMBEDDINGS_DIM,
            "concat_method": cfg.MODEL.VISUAL_EMBED.CONCAT_METHOD,
        }

        activation_name = (cfg.MODEL.VISUAL_EMBED.ACTIVATION).lower()
        if activation_name != "none":
            activation = get_act_layer(activation_name)
            assert activation is not None

            act_kwargs = {}
            if activation_name in { "elu", "celu" }:
                act_kwargs["alpha"] = cfg.MODEL.VISUAL_EMBED.ELU_ALPHA
            embeddings_act = activation(**act_kwargs)
            kwargs['embeddings_act'] = embeddings_act

        if cfg.MODEL.VISUAL_EMBED.DROPOUT > 0:
            embeddings_dropout = nn.Dropout(cfg.MODEL.VISUAL_EMBED.DROPOUT)
            kwargs['embeddings_dropout'] = embeddings_dropout

        if cfg.MODEL.VISUAL_EMBED.USE_NORM:
            embeddings_norm = nn.LayerNorm(cfg.MODEL.VISUAL_EMBED.OUT_DIM)
            kwargs['embeddings_norm'] = embeddings_norm

        if cfg.MODEL.VISUAL_EMBED.LOCATION_SIZE > 0:
            embeddings_pos = nn.Linear(5, cfg.MODEL.VISUAL_EMBED.OUT_DIM)
            kwargs['embeddings_pos'] = embeddings_pos

        return kwargs

    def forward(self, batched_inputs):
        ret = {}
        feats = batched_inputs[kfg.ATT_FEATS] #(30, 20, 1536) / (30, 30, 1024)
        boxes = batched_inputs[kfg.ATT_FEATS_LOC] if kfg.ATT_FEATS_LOC in batched_inputs else None
        motion_feats = batched_inputs[kfg.MOTION_ATT_FEATS] #(30, 20, 2048) / (30, 18, 1024)
        object_feats = batched_inputs[kfg.OBJECT_ATT_FEATS] #(30, 40, 2048) / (30, 19, 2048)
        bsz, sample_numb, max_objects_per_video = feats.shape[0], motion_feats.shape[1], object_feats.shape[1] #30, 20, 40
        feats = self.feats_proj(feats.view(-1, feats.shape[-1])) #(600, 768)
        feats = feats.view(bsz, sample_numb, -1).contiguous()  #(bsz, sample_numb, hidden_dim) (30, 20, 768)
        motion_feats = self.motion_feats_proj(motion_feats.view(-1, motion_feats.shape[-1])) #(600, 768)
        motion_feats = motion_feats.view(bsz, sample_numb, -1).contiguous()  #(bsz, sample_numb, hidden_dim) (30, 20, 768)
        object_feats = self.object_feats_proj(object_feats.view(-1, object_feats.shape[-1])) #(1200, 768)
        object_feats = object_feats.view(bsz, max_objects_per_video, -1).contiguous()  #(bsz, sample_numb, hidden_dim) (30, 40, 768)

        if self.concat_method == 'concat':
            content_vectors = torch.cat([feats, motion_feats, object_feats], dim=-1) #(30, 20, 1536)
        elif self.concat_method == 'transformer':
            content_vectors = torch.cat([feats, motion_feats], dim=-1) #(30, 20, 1536)
        embeddings = self.embeddings(content_vectors) #(30, 20, 768)

        if (self.embeddings_pos is not None) and (boxes is not None):
            embeddings_pos = self.embeddings_pos(boxes)
            embeddings = embeddings + embeddings_pos

        if self.embeddings_act is not None:
            embeddings = self.embeddings_act(embeddings)

        if self.embeddings_norm is not None:
            embeddings = self.embeddings_norm(embeddings)

        if self.embeddings_dropout is not None:
            embeddings = self.embeddings_dropout(embeddings)
        
        
        ret.update({ kfg.ATT_FEATS: embeddings })
        ret.update({ kfg.OBJECT_ATT_FEATS: object_feats})
        return ret #(30, 20, 768)