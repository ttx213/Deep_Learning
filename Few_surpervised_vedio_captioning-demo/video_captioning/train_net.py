"""
From original at https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/__init__.py
Original copyright of Facebook code below, modifications by Yehao Li, Copyright 2021.	
"""

# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
from collections import OrderedDict
import torch
import xmodaler.utils.comm as comm
from xmodaler.checkpoint import XmodalerCheckpointer
from xmodaler.config import get_cfg
from xmodaler.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch, build_engine
from xmodaler.modeling import add_config

def setup(args, config_file):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg() #获取默认配置
    tmp_cfg = cfg.load_from_file_tmp(config_file) #获取临时配置
    add_config(cfg, tmp_cfg) #添加默认配置和临时配置

    cfg.merge_from_file(config_file) #load values from a file
    cfg.merge_from_list(args.opts) #can also load values from a list of str
    
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    i = 1
    if i==1:
        cfg = setup(args, args.config_file)
        """
        If you'd like to do anything fancier than the standard training logic,
        consider writing your own training loop (see plain_train_net.py) or
        subclassing the trainer.
        """
        trainer = build_engine(cfg)
        trainer.resume_or_load(resume=args.resume)

    else: 
        cfg = setup(args, args.config_file1)
        trainer = build_engine(cfg)
        trainer.resume_or_load(resume=args.resume)

    if args.eval_only:
        res = None
        # if trainer.val_data_loader is not None: #验证集测试
        #     res = trainer.test(trainer.cfg, trainer.model, trainer.val_data_loader, trainer.val_evaluator, epoch=-1)
        # if comm.is_main_process():
        #     print(res)
        if trainer.test_data_loader is not None: #测试集测试
            res = trainer.test(trainer.cfg, trainer.model, trainer.test_data_loader, trainer.test_evaluator, epoch=-1)
        if comm.is_main_process():
            print(res)
        return res

    return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch( #启用多GPU分布式训练
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )