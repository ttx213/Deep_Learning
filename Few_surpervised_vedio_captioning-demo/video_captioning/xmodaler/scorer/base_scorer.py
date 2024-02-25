# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import os
import sys
import numpy as np
import pickle

from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import SCORER_REGISTRY
from xmodaler.functional import load_vocab
__all__ = ['BaseScorer']

@SCORER_REGISTRY.register()
class BaseScorer(object):
    @configurable
    def __init__(
        self,
        *,
        types,
        scorers,
        weights,
        gt_path,
        eos_id,
        vocab_path: str,
    ): 
       self.types = types
       self.scorers = scorers
       self.eos_id = eos_id
       self.weights = weights
       self.gts = pickle.load(open(gt_path, 'rb'), encoding='bytes')
       self.vocab = load_vocab(vocab_path)

    @classmethod
    def from_config(cls, cfg):
        scorers = []
        for name in cfg.SCORER.TYPES:
            scorers.append(SCORER_REGISTRY.get(name)(cfg))

        return {
            'scorers': scorers,
            'types': cfg.SCORER.TYPES,
            'weights': cfg.SCORER.WEIGHTS,
            'gt_path': cfg.SCORER.GT_PATH,
            'eos_id': cfg.SCORER.EOS_ID,
            'vocab_path': cfg.INFERENCE.VOCAB,
        }

    def get_sents(self, sent):
        words = []
        for word in sent:
            if word == self.eos_id:
                words.append(self.eos_id)
                break
            words.append(word)
        return words
    
    def decode_sequence(self, seq):
        N = len(seq)
        T = len(seq[0])
        # N, T = seq.shape
        sents = []
        for n in range(N):
            words = []
            for t in range(T):
                ix = seq[n][t]
                if ix == 0:
                    break
                elif ix == 2:
                    continue
                words.append(self.vocab[ix])
            sent = ' '.join(words)
            sents.append(sent)
        return sents

    def __call__(self, batched_inputs):
        ids = batched_inputs[kfg.IDS] #array(['1'], dtype='<U1')
        res = batched_inputs[kfg.G_SENTS_IDS] #(1, 21)
        res = res.cpu().tolist() #(1, 21)

        # hypo = [self.get_sents(r) for r in res] #生成语句
        # gts = [self.gts[i] for i in ids] #真实语句

        hypo =[[self.get_sents(r)] for r in res] #生成语句
        gts = [self.gts[i] for i in ids] #真实语句

        hypo_dict = {}
        gts_dict = {}

        for num, id in enumerate(ids):
            hypo_dict[id] = self.decode_sequence(hypo[num])
            gts_dict[id] = self.decode_sequence(gts[num])
        
        rewards_info = {}
        rewards = np.zeros(len(ids)) #(4,)
        for i, scorer in enumerate(self.scorers):
            score, scores = scorer.compute_score(gts_dict, hypo_dict) #
            if self.types[i] == 'Bleu':
                scores = np.array(scores[3]).astype(np.float64)
            # score, scores = scorer.compute_score(gts, hypo) #0.7529862761774273 array([0.71466238, 0.24184459, 0.81946665, 1.23597149])
            if (len(scores) != len(ids)):
                scorer_dict = {}
                for j, item in enumerate(scores):
                    scorer_dict[list(gts_dict.keys())[j]] = item
                scores_list = []
                for id in ids:
                    scores_list.append(scorer_dict[id])
                scores = np.array(scores_list).astype(np.float64)
            rewards += self.weights[i] * scores
            rewards_info[self.types[i]] = score[3]
        rewards_info.update({ kfg.REWARDS: rewards })
        return rewards_info