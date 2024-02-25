'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import copy
import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
from torch.nn import functional as F
# from models.containers import Module, ModuleList
from xmodaler.config import configurable
# from ..decoder.containers import Module

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = Parameter(w)
        self.bias = Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,) #[5, 21, 2304]
        x = torch.addmm(self.bias, x.contiguous().view(-1, x.size(-1)), self.weight) #(105, 2304)

        x = x.view(*size_out) #(5, 21, 2304)
        return x


class Attention(nn.Module):
    @configurable
    def __init__(
        self,
        *, 
        nx: int, 
        n_ctx: int,
        n_head: int,
        attn_drop: float,
        scale: bool,
        can_be_stateful: bool
    ):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        assert n_state % n_head == 0
        self.n_head = n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.can_be_stateful = can_be_stateful
        self.attn_drop = nn.Dropout(attn_drop)

    @classmethod
    def from_config(cls, cfg):
        return {
            "nx":  cfg.MODEL.GPT2.N_EMBD,
            "n_ctx": cfg.MODEL.GPT2.N_CTX,
            "n_head": cfg.MODEL.GPT2.N_HEAD,
            "attn_drop": cfg.MODEL.GPT2.ATTN_DROP,
            "scale": cfg.MODEL.GPT2.SCALE,
            "can_be_stateful": cfg.MODEL.GPT2.CAN_BE_STATEFUL,
        }

    def _attn(self, q, k, v,mask_self_attention):

        w = torch.matmul(q, k) #(5, 12, 21, 21)
        if self.scale:
            w = w / math.sqrt(v.size(-1))

        if mask_self_attention is not None:

            w = w.masked_fill(mask_self_attention, -10000.0) #将true填充为-10000
            # w[:,:,:,:nk] = w[:,:,:,:nk].masked_fill(mask_self_attention, -1e7)
        # nd, ns = w.size(-2), w.size(-1)
        # b = self.bias[:, :, ns-nd:ns, :ns]

        # w = w * b - 1e10 * (1 - b)
        w = nn.Softmax(dim=-1)(w)
        self.w = self.attn_drop(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head) #[5, 21, 12, 64] 12为多头数
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states (5, 21, 12, 64)
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None, mask_self_attention=None): #(5, 21, 768)
        x = self.c_attn(x) #(5, 21, 2304)
        query, key, value = x.split(self.split_size, dim=2) #(5, 21, 768), (5, 21, 768), (5, 21, 768)
        query = self.split_heads(query) #(5, 12, 21, 64) 将768拆分为12*64
        key = self.split_heads(key, k=True) #(5, 12, 64, 21)
        value = self.split_heads(value) #(5, 12, 21, 64)

        # if self.can_be_stateful and self._is_stateful:
        #     self.running_keys = torch.cat([self.running_keys, key.transpose(-2,-1)],-2)
        #     key = self.running_keys.transpose(-2,-1)
        #     self.running_values = torch.cat([self.running_values, value], -2)
        #     value = self.running_values
        
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)

        present = torch.stack((key.transpose(-2, -1), value))  # 将key和value转换为具有相同形状以进行堆叠 (2, 5, 12, 21, 64)
        a = self._attn(query, key, value, mask_self_attention) #(5, 12, 21, 64)
        a = self.merge_heads(a) #(5, 21, 768)
        a = self.c_proj(a) #(5, 21, 768)


        return a, present


class Enc_Dec_Attention(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        nx: int, 
        n_ctx: int, 
        scale: bool,
        n_head: int
    ):
        super(Enc_Dec_Attention, self).__init__()
        n_state = nx
        # n_ctx = 60
        scale = True
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % n_head == 0
        # self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = n_head
        self.split_size = n_state
        self.scale = scale
        # self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)

        self.fc_q = nn.Linear(n_state, 64 * 12)
        self.fc_k = nn.Linear(n_state, 64 * 12)
        self.fc_v = nn.Linear(n_state, 64 * 12)

        self.attn_dropout = nn.Dropout(0.2)

        self.init_weights()

    @classmethod
    def from_config(cls, cfg):
        return{
            "nx": cfg.MODEL.GPT2.N_EMBD,
            "n_ctx": cfg.MODEL.GPT2.N_CTX,
            "scale": cfg.MODEL.GPT2.SCALE,
            "n_head": cfg.MODEL.GPT2.N_HEAD,
        }

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)

        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        # nn.init.xavier_uniform_(self.fc_o.weight)


    def _attn(self, q, k, v, enc_dec_attention):
        nk = k.shape[-1]
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        if enc_dec_attention is not None:
            w = w.masked_fill(enc_dec_attention, -10000.0)


        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None,encoder_output=None, mask_encoder=None):

        query = self.fc_q(x) #(5, 21, 768)
        encoder_key = self.fc_k(encoder_output) #(5, 21, 768)
        encoder_value = self.fc_v(encoder_output) #(5, 21, 768)
        query = self.split_heads(query) #(5, 12, 21, 64)
        encoder_key = self.split_heads(encoder_key, k=True) #(5, 12, 64, 40)
        encoder_value = self.split_heads(encoder_value) #(5, 12, 40, 64)


        a = self._attn(query, encoder_key,encoder_value,mask_encoder)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a #(5, 21, 768)


class MLP(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        n_state: int,
        nx: int
    ):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu

    @classmethod
    def from_config(cls, cfg):
        return{
            'n_state': 4*cfg.MODEL.GPT2.N_EMBD,
            'nx': cfg.MODEL.GPT2.N_EMBD,
        }

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2

class Block(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        n_embd: int,
        n_ctx: int,
        layer_norm_epsilon: float,
        resid_drop: float,
        attn,
        enc_dec_attn,
        mlp
    ):
        super(Block, self).__init__()
        nx = n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=layer_norm_epsilon)
        self.attn = attn
        self.enc_dec_attn = enc_dec_attn
        self.ln_2 = nn.LayerNorm(nx, eps=layer_norm_epsilon)
        self.mlp = mlp
        self.resid_drop= nn.Dropout(resid_drop)

        # self.fc_alpha1 = nn.Linear(nx + nx, nx)
        # self.fc_alpha2 = nn.Linear(nx + nx, nx)
        # self.fc_alpha3 = nn.Linear(nx + nx, nx)

    
    @classmethod
    def from_config(cls, cfg):
        attn = Attention(cfg)
        enc_dec_attn = Enc_Dec_Attention(cfg)
        mlp = MLP(cfg)
        return{
            "n_ctx": cfg.MODEL.GPT2.N_CTX,
            "layer_norm_epsilon": cfg.MODEL.GPT2.LAYER_NORM_EPSILON,
            "n_embd": cfg.MODEL.GPT2.N_EMBD,
            "resid_drop": cfg.MODEL.GPT2.RESID_DROP,
            "attn": attn,
            "enc_dec_attn": enc_dec_attn,
            "mlp": mlp,
        }

    def forward(self, x, layer_past=None,mask_queries=None,encoder_output=None,mask_encoder=None, mask_self_attention=None, tau = 0):
        threshold = tau #0.2

        self_attention, present = self.attn(self.ln_1(x), layer_past=layer_past,
                                            mask_self_attention=mask_self_attention) #(5, 21, 768) (2, 5, 12, 21, 64)
        a = x + self_attention #(5, 21, 768)
        a = self.resid_drop(a)


        enc_att1 = self.enc_dec_attn(x=self.ln_1(a), encoder_output=self.ln_1(encoder_output[:, 0]),mask_encoder=mask_encoder) #(5, 21, 768)
     
        enc_att2 = self.enc_dec_attn(x=self.ln_1(a), encoder_output=self.ln_1(encoder_output[:, 1]),mask_encoder=mask_encoder)
     
        enc_att3 = self.enc_dec_attn(x=self.ln_1(a), encoder_output=self.ln_1(encoder_output[:, 2]),mask_encoder=mask_encoder)

        enc_att4 = self.enc_dec_attn(x=self.ln_1(a), encoder_output=self.ln_1(encoder_output[:, 3]),mask_encoder=mask_encoder) #(5, 21, 768)
     
        enc_att5 = self.enc_dec_attn(x=self.ln_1(a), encoder_output=self.ln_1(encoder_output[:, 4]),mask_encoder=mask_encoder)
     
        enc_att6 = self.enc_dec_attn(x=self.ln_1(a), encoder_output=self.ln_1(encoder_output[:, 5]),mask_encoder=mask_encoder)

        # alpha1 = torch.sigmoid(self.fc_alpha1(torch.cat([a, enc_att1], -1))) #(5, 21, 768)
        # alpha2 = torch.sigmoid(self.fc_alpha2(torch.cat([a, enc_att2], -1)))
        # alpha3 = torch.sigmoid(self.fc_alpha3(torch.cat([a, enc_att3], -1)))


        # linguistics_alpha1_mask = torch.where(alpha1 > threshold, torch.ones_like(alpha1), torch.zeros_like(alpha1)) #(5, 21, 768)
        # linguistics_alpha2_mask = torch.where(alpha2 > threshold, torch.ones_like(alpha2), torch.zeros_like(alpha2))
        # linguistics_alpha3_mask = torch.where(alpha3 > threshold, torch.ones_like(alpha3), torch.zeros_like(alpha3))



        # visual_alpha1_mask = torch.where(alpha1 < 1-threshold, torch.ones_like(alpha1), torch.zeros_like(alpha1)) #(5, 21, 768)
        # visual_alpha2_mask = torch.where(alpha2 < 1-threshold, torch.ones_like(alpha2), torch.zeros_like(alpha2))
        # visual_alpha3_mask = torch.where(alpha3 < 1-threshold, torch.ones_like(alpha3), torch.zeros_like(alpha3))



        # enc_att1 = alpha1* linguistics_alpha1_mask * a + (1-alpha1)* visual_alpha1_mask * enc_att1 #(5, 21, 768)
        # enc_att2 = alpha2* linguistics_alpha2_mask * a + (1-alpha2)* visual_alpha2_mask * enc_att2
        # enc_att3 = alpha3* linguistics_alpha3_mask * a + (1-alpha3)* visual_alpha3_mask * enc_att3


        enc_att = (enc_att1 + enc_att2 + enc_att3 + enc_att4 + enc_att5 + enc_att6) / np.sqrt(6) #(5, 21, 768)
        # a = enc_att * mask_queries #(5, 21, 768)

        # m = self.mlp(self.ln_2(a)) #(5, 21, 768)

        b = enc_att * mask_queries #(5, 21, 768)

        m = self.mlp(self.ln_2(b)) #(5, 21, 768)

        encoder_result = a + m #(5, 21, 768)

        encoder_result = self.resid_drop(encoder_result) #(5, 21, 768)

        encoder_result = encoder_result  * mask_queries #(5, 21, 768)
        return encoder_result, present #(5, 21, 768) (2, 5, 12, 21, 64)

class GPT2Model(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        n_layer: int,
        n_embd: int,
        n_positions: int,
        layer_norm_epsilon: float,
        vocab_size: int,
        block,
    ):
        super(GPT2Model, self).__init__()
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_vocab = vocab_size

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(n_positions, n_embd)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd, eps=layer_norm_epsilon)

    
    @classmethod
    def from_config(cls, cfg):
        return {
            "n_layer": cfg.MODEL.GPT2.N_LAYER,
            "n_embd": cfg.MODEL.GPT2.N_EMBD,
            "vocab_size": cfg.MODEL.VOCAB_SIZE,
            "n_positions": cfg.MODEL.GPT2.N_POSITIONS,
            "block": Block(cfg),
            "layer_norm_epsilon": cfg.MODEL.GPT2.LAYER_NORM_EPSILON,
        }

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, input_ids, position_ids=None, token_type_ids=None, past=None,mask_queries=None,encoder_output=None,mask_encoder=None, mask_self_attention = None, tau = 0):

        if past is None:
            past_length = 0
            past = [None] * len(self.h) #5
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long,
                                        device=input_ids.device) #21
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids) #(5, 21)

        input_shape = input_ids.size() #[5, 21]
        input_ids = input_ids.view(-1, input_ids.size(-1)) #(5, 21)
        position_ids = position_ids.view(-1, position_ids.size(-1)) #(5, 21)
        inputs_embeds = self.wte(input_ids) #(5, 21, 768)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds #(5, 21, 768)
        presents = []

        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past, mask_queries = mask_queries,encoder_output=encoder_output,mask_encoder=mask_encoder, mask_self_attention= mask_self_attention, tau = tau)
            presents.append(present) #(5, 21, 768) (2, 5, 12, 21, 64)
        hidden_states = self.ln_f(hidden_states) #(5, 21, 768)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape), presents #(5, 21, 768) (2, 5, 12, 21, 64)*12

class GPT2LMHead(nn.Module):
    @configurable
    def __init__(
        self, 
        model_embeddings_weights, 
        n_embd
    ):
        super(GPT2LMHead, self).__init__()
        self.n_embd = n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    @classmethod
    def from_config(cls, cfg):
        return{
            'n_embd': cfg.MODEL.GPT2.N_EMBD,
        }

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):
        lm_logits = self.decoder(hidden_state)
        return lm_logits
