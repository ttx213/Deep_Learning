import torch
import torch.nn as nn
from .build import META_ARCH_REGISTRY
from xmodaler.config import configurable
from xmodaler.config import kfg
from ..mtl_roberta.modeling_roberta import RobertaConfig, RobertaForMTL
from ..mtl_roberta.tokenization_roberta import RobertaTokenizer

ABBR = ("'s","'S","'t","'t","'re","'RE","'ve","'VE","'m","'M","'ll","'LL","'d","'D") # abbreviation
__all__ = ["TextRevise"]

@META_ARCH_REGISTRY.register()
class TextRevise(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        rbt_model, 
        rbt_tknzr, 
        k,
        device
    ):
        super(TextRevise, self).__init__()
        self.rbt_model = rbt_model
        self.tokenizer = rbt_tknzr
        self.K = k
        self.vocab_size = len(self.tokenizer) #50267
        self.device = device
        self.mask_idx = self.tokenizer.convert_tokens_to_ids(['<lm-mask>'])[0] #50265
        self.pad_idx = self.tokenizer.convert_tokens_to_ids(['<pad>'])[0] #1
        self.bos_idx = self.tokenizer.convert_tokens_to_ids(['<s>'])[0] #0
        self.eos_idx = self.tokenizer.convert_tokens_to_ids(['</s>'])[0] #2
        self.rbt_model.eval()
    
    @classmethod
    def from_config(cls, cfg):
        rbt_config = RobertaConfig.from_pretrained(cfg.MODEL.ROBERTA.PATH, cache_dir=None) #载入Roberta模型
        rbt_tknzr = RobertaTokenizer.from_pretrained(cfg.MODEL.ROBERTA.PATH, do_lower_case=False)
        rbt_model = RobertaForMTL.from_pretrained(cfg.MODEL.ROBERTA.PATH, config=rbt_config,
                                                 task_names=['bertscore', 'maskedlm', 'cls'])
        return {
            "rbt_model": rbt_model,
            "rbt_tknzr": rbt_tknzr,
            "k": cfg.MODEL.ROBERTA.K,
            "device": cfg.MODEL.DEVICE
        }
    
    @classmethod
    def add_config(cls, cfg, tmp_cfg):
        pass
    
    def preprocess_batch(self, batched_inputs):
        ret = {}
        if kfg.IDS in batched_inputs[0]:
            ids = [x[kfg.IDS]  for x in batched_inputs ]
            ret.update({ kfg.IDS: ids})
        if kfg.CAND_INPS_SENTS in batched_inputs[0]:
            cand_inps_sents = [x[kfg.CAND_INPS_SENTS]  for x in batched_inputs]
            ret.update({ kfg.CAND_INPS_SENTS: cand_inps_sents})
        return ret
    
    def cal_attr(self, cand_inps, hook_hid_grad):
        attn_mask = cand_inps.ne(self.pad_idx).float().to(self.device) #.ne()不等于
        attr_val, hid_states = self.rbt_model('cls', cand_inps,
                                            attention_mask=attn_mask,
                                            hook_hid_grad=hook_hid_grad) #(2, 2) (2, 8, 768)*13
        return attr_val, hid_states
    
    def select_abbr_span(self, tknzd_sents): #搜索缩写文本片段
        abbr_pos = []
        for sent in tknzd_sents:
            pos = []
            for i, tk in enumerate(sent):
                if tk in ABBR:
                    pos.extend([i, i + 1])
                    break
            abbr_pos.append(pos)
        return abbr_pos

    def revise(self, K, cand_inps, cand_mask, ins_pos, memory_bank=None):
        bsz, seqlen = cand_inps.size() #2, 49

        t = 0

        with torch.no_grad():
            editable = cand_mask.float() #获取可编辑文本片段位置信息

            while editable.eq(1).any():
                if t == 0:
                    attn_mask = cand_inps.ne(self.pad_idx).type(torch.float) #(2, 49)
                    outs = self.rbt_model('maskedlm', cand_inps, attention_mask=attn_mask,
                                      memory_bank=memory_bank, memory_fix_pos=ins_pos)[0] #(2, 49, 50267)
                    outs = outs.transpose(1, 2).contiguous().view(bsz, -1) #(2, 2463083)
                    outs = torch.where(editable.repeat(1, self.vocab_size).eq(1).to(self.device),
                                       outs, torch.full_like(outs, -1e10)) #(2, 2463083)
                    ins_probs, cand_words = torch.topk(outs, K, dim=-1) # _, [[497], [3933]]
                    del outs, ins_probs

                    edit_pos_t = cand_words % seqlen #[[[7], [13]]]
                    cand_words = cand_words // seqlen #[[10], [80]]
                    cand_inps = cand_inps.repeat(1, K).contiguous().view(bsz * K, seqlen) #(2, 49)
                    editable = editable.repeat(1, K).contiguous().view(bsz * K, seqlen) #(2, 49)
                    edit_pos_t = edit_pos_t.view(-1, 1) #[[7], [13]]
                    cand_words = cand_words.view(-1, 1) #[[10], [80]]
                else:
                    cand_words_all = []
                    edit_pos_t = []
                    cand_inps_bats = torch.split(cand_inps, 100, dim=0)
                    editable_bats = torch.split(editable, 100, dim=0)
                    for b, (inps, editable_t) in enumerate(zip(cand_inps_bats, editable_bats)):
                        outs = self.rbt_model('maskedlm', inps.to(self.device),
                                          attention_mask=attn_mask, memory_bank=memory_bank,
                                          memory_fix_pos=ins_pos)[0]
                        outs = outs.transpose(1, 2).contiguous().view(outs.size(0), -1)
                        outs = torch.where(editable_t.repeat(1, self.vocab_size).eq(1).to(self.device),
                                           outs, torch.full_like(outs, -1e10))
                        ins_probs, cand_words = torch.topk(outs, 1, dim=-1)
                        del outs, ins_probs, editable_t
                        edit_pos_tb = cand_words % seqlen
                        cand_words = cand_words // seqlen
                        cand_words_all.append(cand_words)
                        edit_pos_t.append(edit_pos_tb)
                    cand_words = torch.cat(cand_words_all, dim=0)
                    edit_pos_t = torch.cat(edit_pos_t, dim=0)
                t += 1
                assert cand_words.ne(self.mask_idx).all()
                new_cand_inps = cand_inps.scatter(1, edit_pos_t, cand_words)

                cand_inps = torch.where(editable.eq(1), new_cand_inps, cand_inps)

                del new_cand_inps
                edit_pos_t = edit_pos_t.view(-1, 1)
                editable = editable.scatter(1, edit_pos_t.to(self.device),
                                            torch.zeros_like(edit_pos_t).float().to(self.device))
        return cand_inps #(2, 49)