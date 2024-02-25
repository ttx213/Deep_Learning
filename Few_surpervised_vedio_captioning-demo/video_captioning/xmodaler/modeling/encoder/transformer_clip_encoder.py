import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from ..layers.bert import BertLayer, BertGenerationLayer, BertObjectLayer
from .build import ENCODER_REGISTRY

__all__ = ["TransformerClipEncoder"]

@ENCODER_REGISTRY.register()
class TransformerClipEncoder(nn.Module):
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
        semcomphder_layers,
        min_object_num: int,
        out_dim: int,
        embeddings_dim: int,
        model_type: str,
        hidden_size: int,
        num_semcomphder_layers: int,
        slot_size: int,
        keywords_num: int,
        max_pos: int

    ):
        super(TransformerClipEncoder, self).__init__()
        self.num_encoder_olayers = num_encoder_olayers
        self.num_decoder_olayers = num_decoder_olayers
        self.num_hidden_layers = num_hidden_layers
        self.num_semcomphder_layers = num_semcomphder_layers
        self.bert_encoder_olayers = bert_encoder_olayers
        self.bert_decoder_olayers = bert_decoder_olayers
        self.layers = bert_layers
        self.decoder_enc_layers = semcomphder_layers
        self.min_object_num = min_object_num
        self.model_type = model_type
        self.visual_embeddings = nn.Linear(embeddings_dim, out_dim)
        self.keywords_num = keywords_num
        self.slot_size = slot_size
        self.max_pos_len = max_pos

        self.semantics_pred = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, keywords_num+1)   
        )

        self.semantics_embeddings = nn.Sequential(
            nn.Embedding(keywords_num, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
        )

        self.slot_embeddings = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
        )

        self.slot = nn.Parameter(torch.FloatTensor(1, slot_size, hidden_size))
        nn.init.xavier_uniform_(self.slot)

        self.softmax = nn.Softmax(dim=-1)
        self.position = nn.Parameter(torch.FloatTensor(self.max_pos_len, hidden_size))
        nn.init.xavier_uniform_(self.position)

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

        semcomphder_layers = nn.ModuleList(
            [BertGenerationLayer(cfg) for _ in range(cfg.MODEL.CLIP.NUM_SEMCOMPHDER_LAYERS)]
        )

        return {
            "num_encoder_olayers": cfg.MODEL.BERT.NUM_ENCODER_OLAYERS,
            "num_decoder_olayers": cfg.MODEL.BERT.NUM_DECODER_OLAYERS,
            "hidden_size": cfg.MODEL.BERT.HIDDEN_SIZE,
            "num_hidden_layers": cfg.MODEL.BERT.NUM_HIDDEN_LAYERS,
            "bert_encoder_olayers": bert_encoder_olayers,
            "bert_decoder_olayers": bert_decoder_olayers,
            "bert_layers": bert_layers,
            "min_object_num": cfg.DATALOADER.MIN_OBJECT_NUM,
            "embeddings_dim": cfg.MODEL.VISUAL_EMBED.EMBEDDINGS_DIM,
            "out_dim": cfg.MODEL.VISUAL_EMBED.OUT_DIM,
            "model_type": cfg.MODEL.TYPE,
            "semcomphder_layers": semcomphder_layers,
            "slot_size": cfg.MODEL.CLIP.SLOT_SIZE,
            "keywords_num": cfg.MODEL.CLIP.KEYWORDS_NUM,
            "max_pos": cfg.MODEL.CLIP.MAX_POS,
            "num_semcomphder_layers": cfg.MODEL.CLIP.NUM_SEMCOMPHDER_LAYERS,
        }

    @classmethod
    def add_config(cls, cfg):
        cfg.MODEL.CLIP = CN()
        cfg.MODEL.CLIP.NUM_SEMCOMPHDER_LAYERS = 3
        cfg.MODEL.CLIP.SLOT_SIZE = 6
        cfg.MODEL.CLIP.KEYWORDS_NUM = 1000
        cfg.MODEL.CLIP.MAX_POS = 26
        cfg.MODEL.CLIP.FILTER_WEIGHT = 1.0
        cfg.MODEL.CLIP.RECONSTRUCT_WEIGHT = 0.1

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
                feats = self.visual_embeddings(content_vectors) #(30, 20, 768)
                ret.update({ kfg.ATT_FEATS: feats, kfg.OBJECT_ATT_FEATS: new_vfeats})
            
            elif self.model_type == 'gpt2':
                outs_v = torch.cat(out_v, 1) #(30, 6, 20, 768)
                # outs_o = torch.cat(out_o, 1) #(30, 6, 20, 768)
                outs_o = new_vfeats.unsqueeze(1).repeat(1, self.num_hidden_layers, 1, 1)
                outs = torch.cat([outs_v, outs_o], dim=-1)
                outs = self.visual_embeddings(outs)
                ret.update({ kfg.ATT_OUTS: outs, kfg.OBJECT_ATT_FEATS: new_vfeats})
        
            semantics_ids = batched_inputs[kfg.SEMANTICS_IDS] #(30, 14)
            semantics_mask = batched_inputs[kfg.SEMANTICS_MASK] #(30, 14)
            
            semantics_embed = self.semantics_embeddings(semantics_ids) #(30, 15, 768)
            slot_embed = self.slot_embeddings(self.slot) #(1, 6, 512)
            slot_embed = slot_embed.expand(semantics_embed.shape[0], slot_embed.shape[1], slot_embed.shape[2]) #(30, 6, 768)
            semantics_embed = torch.cat([slot_embed, semantics_embed], dim=1) #(30, 20, 512)

            slot_mask = torch.ones((semantics_embed.shape[0], slot_embed.shape[1]), device=slot_embed.device).to(dtype=next(self.parameters()).dtype) #(30, 6)
            semantics_mask = torch.cat([slot_mask, semantics_mask], dim=1) #(30, 20)

            semantics_mask = (1.0 - semantics_mask) * -10000.0 #(30, 20)
            semantics_mask = semantics_mask.unsqueeze(1).unsqueeze(2) #(30, 1, 1, 20)

            for layer_module in self.decoder_enc_layers:
                semantics_embed = layer_module(semantics_embed, feats, semantics_mask, ext_vmasks) #(30, 20, 768)

            semantics_pred = self.semantics_pred(semantics_embed) #(30, 20, 1001)

            ret.update({
                kfg.SEMANTICS_PRED: semantics_pred,
                kfg.SEMANTICS_FEATS: semantics_embed,
                kfg.EXT_SEMANTICS_MASKS: semantics_mask,
            })

            semantics_pos_pred = semantics_embed @ self.position.t() #(30, 20, 25)
            semantics_pos_prob = self.softmax(semantics_pos_pred) #(30, 20, 25)
            position = semantics_pos_prob @ self.position #(30, 20, 768)
            semantics_embed = semantics_embed + position #(30, 20, 768)
            ret.update({ kfg.SEMANTICS_FEATS: semantics_embed, kfg.SEMANTICS_POS_PRED: semantics_pos_pred }) #(30, 20, 768) (30, 20, 768)

        return ret