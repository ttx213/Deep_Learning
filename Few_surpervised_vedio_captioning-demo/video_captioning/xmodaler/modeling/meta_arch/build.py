import torch
from xmodaler.utils.registry import Registry

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """ Registry for meta-architectures, i.e. the whole model """

def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE)) #TransformerEncoderDecoder()
    return model

def add_config(cfg, tmp_cfg):
    meta_arch = tmp_cfg.MODEL.META_ARCHITECTURE
    META_ARCH_REGISTRY.get(meta_arch).add_config(cfg, tmp_cfg) #其中的cfg和tmp_cfg是初始化算法需要传入的参数