"""	
From original at https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/__init__.py
Original copyright of Facebook code below, modifications by Yehao Li, Copyright 2021.	
"""
# Copyright (c) Facebook, Inc. and its affiliates.
from .build import (
    build_xmodaler_train_loader,
    build_xmodaler_valtest_loader,
    build_dataset_mapper
)

from .common import DatasetFromList, MapDataset
from .images.mscoco import MSCoCoDataset, MSCoCoSampleByTxtDataset
from .images.mscoco_bert import MSCoCoBertDataset
from .images.mscoco_cosnet import MSCoCoCOSNetDataset
from .images.mscoco_feat import MSCoCoFeatDataset
#from .images.mscoco_raw import MSCoCoRawDataset
from .images.conceptual_captions import ConceptualCaptionsDataset, ConceptualCaptionsDatasetForSingleStream
from .images.vqa import VQADataset
from .images.vcr import VCRDataset
from .images.flickr30k import Flickr30kDataset
from .images.flickr30k_single_stream import Flickr30kDatasetForSingleStream, Flickr30kDatasetForSingleStreamVal
from .videos.msvd import MSVDDataset
from .videos.msrvtt import MSRVTTDataset
from .videos.textrevise import TextReviseDataset
from .videos.msvd_clip import MSVDClipDataset
from .videos.msvd_MCMCXL_clip import MSVDMCMCXLCLIPDataset
from .videos.msvd_MCMCXL import MSVDMCMCXLDataset
from .videos.msrvtt_clip import MSRVTTClipDataset
from .videos.msrvtt_MCMCXL_clip import MSRVTTMCMCXLCLIPDataset
from .videos.msrvtt_original import MSRVTTOriginalDataset
from .videos.msvd_original import MSVDOriginalDataset
from .videos.msrvtt_MCMCXL import MSRVTTMCMCXLDataset
from .videos.vtvtt import VTVTTDataset
from .videos.charades import CharadesDataset
from .videos.charades_clip import CharadesCLIPDataset
from .videos.vatex import VatexDataset
from .videos.vatex_clip import VatexClipDataset

__all__ = [k for k in globals().keys() if not k.startswith("_")]
