import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from ..layers.bert import BertLayer
from ..layers.bert import BertObjectLayer
from .build import ENCODER_REGISTRY

__all__ = ["TransformerObjectEncoder"]

@ENCODER_REGISTRY.register()
class TransformerObjectEncoder(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        num_encoder_olayers: int,
        num_decoder_olayers: int,
        num_hidden_layers: int,
        bert_encoder_olayers,
        bert_decoder_olayers,
        bert_layers,
        min_object_num: int,
        out_dim: int,
        embeddings_dim: int,
        model_type: str,
    ):
        super(TransformerObjectEncoder, self).__init__()
        self.num_encoder_olayers = num_encoder_olayers
        self.num_decoder_olayers = num_decoder_olayers
        self.num_hidden_layers = num_hidden_layers
        self.bert_encoder_olayers = bert_encoder_olayers
        self.bert_decoder_olayers = bert_decoder_olayers
        self.layers = bert_layers
        self.min_object_num = min_object_num
        self.model_type = model_type
        self.embeddings = nn.Linear(embeddings_dim, out_dim)

    @classmethod
    def from_config(cls, cfg):

        bert_layers = nn.ModuleList(
            [BertLayer(cfg) for _ in range(cfg.MODEL.BERT.NUM_HIDDEN_LAYERS)]
        )

        bert_encoder_olayers = nn.ModuleList(
            [BertLayer(cfg) for _ in range(cfg.MODEL.BERT.NUM_ENCODER_OLAYERS)]
        )

        bert_decoder_olayers = nn.ModuleList(
            [BertObjectLayer(cfg) for _ in range(cfg.MODEL.BERT.NUM_DECODER_OLAYERS)]
        )
        return {
            "num_encoder_olayers": cfg.MODEL.BERT.NUM_ENCODER_OLAYERS,
            "num_decoder_olayers": cfg.MODEL.BERT.NUM_DECODER_OLAYERS,
            "num_hidden_layers": cfg.MODEL.BERT.NUM_HIDDEN_LAYERS,
            "bert_encoder_olayers": bert_encoder_olayers,
            "bert_decoder_olayers": bert_decoder_olayers,
            "bert_layers": bert_layers,
            "min_object_num": cfg.DATALOADER.MIN_OBJECT_NUM,
            "embeddings_dim": cfg.MODEL.VISUAL_EMBED.EMBEDDINGS_DIM,
            "out_dim": cfg.MODEL.VISUAL_EMBED.OUT_DIM,
            "model_type": cfg.MODEL.TYPE,

        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs, mode=None):
        ret = {}
        if mode == None or mode == 'v':
            vfeats = batched_inputs[kfg.ATT_FEATS] #(30, 20, 768)
            ext_vmasks = batched_inputs[kfg.EXT_ATT_MASKS] #(30, 1, 1, 20)
            ofeats = batched_inputs[kfg.OBJECT_ATT_FEATS] #(30, 40, 768)
            ext_omasks = batched_inputs[kfg.EXT_object_ATT_MASKS] #(30, 1, 1, 40)

            new_vfeats = torch.max(vfeats, dim=1)[0].unsqueeze(1)  # (bsz, hidden_dim) (30, 1, 768)
            new_vfeats = new_vfeats.repeat(1, self.min_object_num, 1)  # (bsz, max_objects, hidden_dim) (30, 8, 768)
            out_v = []
            out_o = []
            for layer_module in self.layers:
                vfeats, _ = layer_module(vfeats, ext_vmasks) #(30, 20, 768)
                out_v.append(vfeats.unsqueeze(1))

            for layer_module in self.bert_encoder_olayers:
                ofeats, _ = layer_module(ofeats, ext_omasks) #(30, 40, 768)
            
            for layer_module in self.bert_decoder_olayers:
                new_vfeats = layer_module(new_vfeats, ofeats, ext_omasks) #(30, 20, 768)
                # out_o.append(new_vfeats.unsqueeze(1))
            
            if self.model_type == 'bert':
                content_vectors = torch.cat([vfeats, new_vfeats], dim=-1) #(30, 20, 1536)
                # embeddings, _ = self.bilstm(content_vectors) #(30, 20, 768)
                feats = self.embeddings(content_vectors) #(30, 20, 768)
                ret.update({ kfg.ATT_FEATS: feats, kfg.OBJECT_ATT_FEATS: new_vfeats})
            
            elif self.model_type == 'gpt2':
                outs_v = torch.cat(out_v, 1) #(30, 6, 20, 768)
                # outs_o = torch.cat(out_o, 1) #(30, 6, 20, 768)
                outs_o = new_vfeats.unsqueeze(1).repeat(1, self.num_hidden_layers, 1, 1)
                outs = torch.cat([outs_v, outs_o], dim=-1)
                outs = self.embeddings(outs)
                ret.update({ kfg.ATT_OUTS: outs, kfg.OBJECT_ATT_FEATS: new_vfeats})
        
        return ret