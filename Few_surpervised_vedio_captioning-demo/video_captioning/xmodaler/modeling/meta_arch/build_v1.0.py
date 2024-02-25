# -*- coding: utf-8 -*-
"""
From original at https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/meta_arch/build.py
Original copyright of Facebook code below, modifications by Yehao Li, Copyright 2021.	
"""
# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from xmodaler.utils.registry import Registry

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """ Registry for meta-architectures, i.e. the whole model """

def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    #选择是使用一个模型还是两个模型
    if len(cfg.MODEL.META_ARCHITECTURE) == 1:
        meta_arch = cfg.MODEL.META_ARCHITECTURE[0]
        model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
        model.to(torch.device(cfg.MODEL.DEVICE)) #TransformerEncoderDecoder()
        return model
    else:
        models={}
        for i, meta_arch in enumerate(cfg.MODEL.META_ARCHITECTURE):
            model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
            if meta_arch == 'TextRevise':
                model.model.to(torch.device(cfg.MODEL.DEVICE))
            else:
                model.to(torch.device(cfg.MODEL.DEVICE))
            models[meta_arch] = model
        return models

def add_config(cfg, tmp_cfg):
    meta_arch_list = tmp_cfg.MODEL.META_ARCHITECTURE
    for meta_arch in meta_arch_list:
        META_ARCH_REGISTRY.get(meta_arch).add_config(cfg, tmp_cfg) #其中的cfg和tmp_cfg是初始化算法需要传入的参数