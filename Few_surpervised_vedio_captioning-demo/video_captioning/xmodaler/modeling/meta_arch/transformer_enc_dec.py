# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import pad_tensor, dict_to_cuda
from ..predictor import build_v_predictor
from .base_enc_dec import BaseEncoderDecoder
from .build import META_ARCH_REGISTRY
from .load_model import load_weight
from memory_profiler import profile

__all__ = ["TransformerEncoderDecoder"]

@META_ARCH_REGISTRY.register()
class TransformerEncoderDecoder(BaseEncoderDecoder):
    @configurable
    def __init__(
        self,
        *,
        vocab_size,
        max_seq_len,
        token_embed,
        visual_embed,
        encoder,
        decoder,
        predictor,
        greedy_decoder,
        beam_searcher,
        v_predictor,
        model_type: str,
        pre_parameters: bool
    ):
        super().__init__(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            token_embed=token_embed,
            visual_embed=visual_embed,
            encoder=encoder,
            decoder=decoder,
            predictor=predictor,
            greedy_decoder=greedy_decoder,
            beam_searcher=beam_searcher,
            model_type=model_type,
            pre_parameters=pre_parameters
        )
        self.v_predictor = v_predictor

        if self.pre_parameters:

            if self.model_type =='gpt2':
                state_dict_gpt2 = torch.load('/home/wangtao/video_caption/xmodaler-master/Pre_training_models/GPT2/gpt2-pytorch_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)
                self.decoder = load_weight(self.decoder, state_dict_gpt2, self.model_type)
            
            elif self.model_type =='bert':
                state_dict_bert = torch.load('/home/wangtao/video_caption/xmodaler-master/Pre_training_models/Bert/bert-pytorch_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)
                self.decoder = load_weight(self.decoder, state_dict_bert, self.model_type)
        
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        if cfg.MODEL.BERT.V_TARGET_SIZE > 0:
            v_predictor = build_v_predictor(cfg)
        else:
            v_predictor = None
        
        ret.update({ "v_predictor": v_predictor })
        return ret


    def get_extended_attention_mask(self, batched_inputs):
        if kfg.TOKENS_MASKS not in batched_inputs:
            batched_inputs[kfg.TOKENS_MASKS] = torch.ones((batched_inputs[kfg.ATT_MASKS].size(0), self.max_seq_len)).cuda()

        tmasks = batched_inputs[kfg.TOKENS_MASKS] #(128, 21)
        seq_length = tmasks.size(-1) #21
        tmasks = tmasks.to(dtype=next(self.parameters()).dtype)
        ext_u_tmasks = tmasks.unsqueeze(1).unsqueeze(2) #(128, 1, 1, 21)
        ext_u_tmasks = (1.0 - ext_u_tmasks) * -10000.0 #全置零

        ext_g_tmasks = torch.tril(torch.ones(
            (seq_length, seq_length), dtype=tmasks.dtype, device=tmasks.device)) #(21, 21) 返回下三角矩阵
        ext_g_tmasks = ext_g_tmasks.unsqueeze(0).expand(
            (tmasks.size(0), seq_length, seq_length)) #(128, 21, 21)
        ext_g_tmasks = ext_g_tmasks * tmasks.unsqueeze(1) #(128, 21, 21)
        ext_g_tmasks = ext_g_tmasks.to(dtype=next(self.parameters()).dtype)
        ext_g_tmasks = ext_g_tmasks.unsqueeze(1) #(128, 1, 21, 21)
        ext_g_tmasks = (1.0 - ext_g_tmasks) * -10000.0 #(128, 1, 21, 21) 将0变为-10000

        vmasks = batched_inputs[kfg.ATT_MASKS] #(128, 40)
        vmasks = vmasks.to(dtype=next(self.parameters()).dtype)
        vmasks = vmasks.unsqueeze(1).unsqueeze(2) #(128, 1, 1, 40)
        ext_vmasks = (1.0 - vmasks) * -10000.0 #全置零
        
        if kfg.OBJECT_ATT_MASKS in batched_inputs:
            omasks = batched_inputs[kfg.OBJECT_ATT_MASKS] #(128, 40)
            omasks = omasks.to(dtype=next(self.parameters()).dtype)
            omasks = omasks.unsqueeze(1).unsqueeze(2) #(128, 1, 1, 40)
            ext_omasks = (1.0 - omasks) * -10000.0 #全置零
        else:
            ext_omasks = None

        return {
            kfg.TOKENS_MASKS: tmasks, #(128, 21)
            kfg.EXT_U_TOKENS_MASKS: ext_u_tmasks, #(128, 1, 1, 21)
            kfg.EXT_G_TOKENS_MASKS: ext_g_tmasks, #(128, 1, 21, 21)
            kfg.ATT_MASKS: vmasks, #(128, 1, 1, 40)
            kfg.EXT_ATT_MASKS: ext_vmasks, #(128, 1, 1, 40)
            kfg.EXT_object_ATT_MASKS: ext_omasks
        }

    def _forward(self, batched_inputs): #保护类型只能允许其本身与子类进行访问，不能使用from xxx import * 的方式导入
        inputs = batched_inputs
        masks = self.get_extended_attention_mask(batched_inputs)
        inputs.update(masks)

        ve_out = self.visual_embed(batched_inputs) #(5, 40, 768)
        inputs.update(ve_out)

        if self.encoder is not None:
            encoder_out_v = self.encoder(inputs, mode='v') #(5, 40, 768)
            inputs.update(encoder_out_v)

        if self.decoder is not None:
            inputs = self.decoder.preprocess(inputs)

        if self.encoder is not None: 
            if self.model_type =='bert':
                te_out = self.token_embed(batched_inputs) #(5, 21, 768)
                inputs.update(te_out)
            
        
        if self.encoder is not None:
            encoder_out_t = self.encoder(inputs, mode='t') # {}
            inputs.update(encoder_out_t)
        
        if self.decoder is not None:
            decoder_out = self.decoder(inputs) #(5, 21, 768)
            inputs.update(decoder_out)

        if self.predictor is not None:
            tlogits = self.predictor(inputs) #(5, 21, 5492)
            inputs.update(tlogits)

        if self.v_predictor is not None:
            vlogits = self.v_predictor(inputs)
            inputs.update(vlogits)
        
        return inputs
