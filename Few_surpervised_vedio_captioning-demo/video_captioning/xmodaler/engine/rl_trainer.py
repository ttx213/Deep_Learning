# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import time
import copy
import torch
from .defaults import DefaultTrainer
from xmodaler.scorer import build_scorer
from xmodaler.config import kfg
from xmodaler.losses import build_rl_losses
import xmodaler.utils.comm as comm
from .build import ENGINE_REGISTRY

__all__ = ['RLTrainer']

@ENGINE_REGISTRY.register()
class RLTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super(RLTrainer, self).__init__(cfg)
        self.scorer = self.build_scorer(cfg)
        self.losses = build_rl_losses(cfg)

    @classmethod
    def build_scorer(cls, cfg):
        return build_scorer(cfg)

    def run_step(self):
        start = time.perf_counter()
        try:
            data = next(self._train_data_loader_iter) #返回可迭代对象中的元素
        except StopIteration:
            if comm.get_world_size() > 1:
                self.train_data_loader.sampler.set_epoch(self.iter//self.iters_per_epoch)

            self._train_data_loader_iter = iter(self.train_data_loader)
            data = next(self._train_data_loader_iter)
        data_time = time.perf_counter() - start

        data = comm.unwrap_model(self.model).preprocess_batch(data)#将一个batch的数据合并，并且增加'MASKS'用以区分真实数据和填充数据。

        self.model.eval()
        with torch.no_grad():
            bs_data = copy.copy(data)
            bs_outputs_dict = self.model(bs_data, use_beam_search=False, output_sents=False)
        bs_rewards = self.scorer(bs_outputs_dict) #计算cider得分和rewards {'Cider': 0.8999689271279383, 'REWARDS': array([1.72763665, 0.76176257, 0.49597235, 0.61450414])}

        self.model.train()
        data[kfg.DECODE_BY_SAMPLE] = True
        outputs_dict = self.model(data, use_beam_search=False, output_sents=False) #在分布式训练中多余数据不要输出
        rewards = self.scorer(outputs_dict)
        rewards = torch.from_numpy(rewards[kfg.REWARDS] - bs_rewards[kfg.REWARDS]).float().cuda()
        outputs_dict.update({ kfg.REWARDS: rewards })

        losses_dict = {}
        for loss in self.losses:
            loss_dict = loss(outputs_dict)
            losses_dict.update(loss_dict)
        
        losses = [losses_dict[k] for k in losses_dict if 'acc' not in k]
        losses = sum(losses)

        self.optimizer.zero_grad()
        losses.backward()

        # a=[]
        # for name, param in self.model.named_parameters():
        #     if param.grad is None:
        #         a.append(name)
        # print(a)

        bs_rewards.pop(kfg.REWARDS)
        losses_dict.update(bs_rewards)
        self._write_metrics(losses_dict, data_time)
        self.optimizer.step()
        if self.ema is not None:
            self.ema.update(self.model)