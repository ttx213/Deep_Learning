# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import itertools
import logging
import numpy as np
import pickle
import random
import torch.utils.data as data
from torch.utils.data.sampler import Sampler

from xmodaler.utils.serialize import PicklableWrapper

__all__ = ["MapDataset", "DatasetFromList"]

class MapDataset(data.Dataset):
    """
    Map a function over the elements in a dataset. (将函数映射到数据集中的元素上。)

    Args:
        dataset: a dataset where map function is applied.
        map_func: a callable which maps the element in dataset. map_func is
            responsible for error handling, when error happens, it needs to
            return None so the MapDataset will randomly use other
            elements from the dataset.
    """

    def __init__(self, dataset, map_func):
        self._dataset = dataset
        self._map_func = PicklableWrapper(map_func)  # wrap so that a lambda will work

        self._rng = random.Random(42)
        self._fallback_candidates = set(range(len(dataset)))

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        retry_count = 0
        cur_idx = int(idx)

        while True:
            data = self._map_func(self._dataset[cur_idx]) #IDS:'609', ATT_FEATS:(40,2048), SEQ_PER_SAMLPE:[1], G_TOKENS_IDS:(21), G_TARGET_IDS:(21), G_TOKENS_TYPE:(21)
            if data is not None:
                self._fallback_candidates.add(cur_idx) #是一个​​set​​,它的特点是其中的元素是独一无二的，定义这个的作用是记录可正常读取的数据索引，
                return data

            # _map_func fails for this idx, use a random new index from the pool
            retry_count += 1
            self._fallback_candidates.discard(cur_idx) #这个坏数据的索引从​​_fallback_candidates​​中剔除
            cur_idx = self._rng.sample(self._fallback_candidates, k=1)[0]

            if retry_count >= 3:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Failed to apply `_map_func` for idx: {}, retry count: {}".format(
                        idx, retry_count
                    )
                )

class DatasetFromList(data.Dataset):
    """
    Wrap a list to a torch Dataset. It produces elements of the list as data.(将list包装到torch数据集。它生成列表元素作为数据)
    """

    def __init__(self, lst: list, copy: bool = True, serialize: bool = True):
        """
        Args:
            lst (list): a list which contains elements to produce.
            copy (bool): whether to deepcopy the element when producing it,
                so that the result can be modified in place without affecting the
                source in the list.
            serialize (bool): whether to hold memory using serialized objects, when
                enabled, data loader workers can use shared RAM from master
                process instead of making a copy.
        """
        self._lst = lst #包含video_id, tokens_ids (1, 21) 和 target_ids (1, 21)
        self._copy = copy #是否使用深复制
        self._serialize = serialize #是否使用序列化对象保持内存

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        if self._serialize:
            logger = logging.getLogger(__name__)
            logger.info(
                "Serializing {} elements to byte tensors and concatenating them all ...".format(
                    len(self._lst)
                )
            )
            self._lst = [_serialize(x) for x in self._lst] #
            self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64) #(48748,)
            self._addr = np.cumsum(self._addr) #(48748,)
            self._lst = np.concatenate(self._lst) #(19698563,)
            logger.info("Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024 ** 2))

    def __len__(self):
        if self._serialize:
            return len(self._addr)
        else:
            return len(self._lst)

    def __getitem__(self, idx):
        if self._serialize:
            start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
            end_addr = self._addr[idx].item()
            bytes = memoryview(self._lst[start_addr:end_addr])
            return pickle.loads(bytes)
        elif self._copy:
            return copy.deepcopy(self._lst[idx])
        else:
            return self._lst[idx]