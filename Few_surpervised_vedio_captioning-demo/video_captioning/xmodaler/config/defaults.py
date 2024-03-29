"""
From original at https://github.com/facebookresearch/detectron2/blob/master/detectron2/config/defaults.py
Original copyright of Facebook code below, modifications by Yehao Li, Copyright 2021.	
"""
# Copyright (c) Facebook, Inc. and its affiliates.
from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# The version number, to upgrade from old configs to new ones if any
# changes happen. It's recommended to keep a VERSION in your config file.
_C.VERSION = 1

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()

_C.DATASETS.TRAIN = ''

_C.DATASETS.VAL = ''

_C.DATASETS.TEST = ''

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()

_C.DATALOADER.TRAIN_BATCH_SIZE = 64

_C.DATALOADER.TEST_BATCH_SIZE = 64

_C.DATALOADER.NUM_WORKERS = 4

_C.DATALOADER.FEATS_FOLDER = ''

_C.DATALOADER.MOTION_FEATS_FOLDER = ''

_C.DATALOADER.OBJECT_FEATS_FOLDER = ''

_C.DATALOADER.ANNO_FOLDER = ''

_C.DATALOADER.RELATION_FILE = ''

_C.DATALOADER.GV_FEAT_FILE = ''

_C.DATALOADER.ATTRIBUTE_FILE = ''

_C.DATALOADER.SEQ_PER_SAMPLE = 5

_C.DATALOADER.MAX_FEAT_NUM = -1

_C.DATALOADER.MAX_OBJECT_NUM = -1

_C.DATALOADER.MIN_OBJECT_NUM = -1

_C.DATALOADER.NEGATIVE_SIZE = -1

_C.DATALOADER.INF_BATCH_SIZE = 200 # for single stream retrieval only, chunk size

_C.DATALOADER.USE_GLOBAL_V = True

_C.DATALOADER.SAMPLE_PROB = 0.2

_C.DATALOADER.SAMPLE_IDS = []

_C.DATALOADER.FILE_PATHS = []

_C.DATALOADER.TRAIN_PERCENTAGE = 1.0

_C.DATALOADER.C3D = False

_C.DATALOADER.FASTER_R_CNN = False

_C.DATALOADER.SENTENCE_NUMS = -1

# -----------------------------------------------------------------------------
# Engine
# -----------------------------------------------------------------------------
_C.ENGINE = CN()

_C.ENGINE.NAME = 'DefaultTrainer'

# -----------------------------------------------------------------------------
# Scheduled sampling
# -----------------------------------------------------------------------------
_C.SCHEDULED_SAMPLING = CN()

_C.SCHEDULED_SAMPLING.START_EPOCH = 0

_C.SCHEDULED_SAMPLING.INC_EVERY_EPOCH = 5

_C.SCHEDULED_SAMPLING.INC_PROB = 0.05

_C.SCHEDULED_SAMPLING.MAX_PROB = 0.25

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()

_C.MODEL.DEVICE = "cuda"

_C.MODEL.VOCAB_SIZE = 1000 # include <BOS>/<EOS>

_C.MODEL.META_ARCHITECTURE = ''

_C.MODEL.ENCODER = ''

_C.MODEL.ENCODER_DIM = 1024

_C.MODEL.DECODER = ''

_C.MODEL.DECODER_DIM = 1024

_C.MODEL.PRED_DROPOUT = 0.0

_C.MODEL.PREDICTOR = ''

_C.MODEL.V_PREDICTOR = ''

_C.MODEL.MAX_SEQ_LEN = 17

_C.MODEL.WEIGHTS = ''

_C.MODEL.ITM_NEG_PROB = 0.5

_C.MODEL.USE_EMA = False

_C.MODEL.EMA_DECAY = 0.9999

_C.MODEL.ENSEMBLE_WEIGHTS = ['']

_C.MODEL.MODEL_WEIGHTS = [1.0, 1.0]

_C.MODEL.TYPE = ''

_C.MODEL.PRE_PARAMETERS = False

_C.MODEL.ITER_STEP = 4

_C.MODEL.ATTRIBUTE = 'formality'

_C.MODEL.CLS_THLD = 0.3

_C.MODEL.MAX_MASK_RATIO = 0.3

_C.MODEL.C = 1

_C.MODEL.FIXED_SPAN_LEN = 3

_C.MODEL.STEP_SIZE = 1.6

# ----------------------------------------------------------------------------
# Token embedding
# ----------------------------------------------------------------------------
_C.MODEL.TOKEN_EMBED = CN()

_C.MODEL.TOKEN_EMBED.NAME = ''

_C.MODEL.TOKEN_EMBED.DIM = 1024

_C.MODEL.TOKEN_EMBED.ACTIVATION = 'none'

_C.MODEL.TOKEN_EMBED.ELU_ALPHA = 0.5

_C.MODEL.TOKEN_EMBED.USE_NORM = False

_C.MODEL.TOKEN_EMBED.DROPOUT = 0.0

_C.MODEL.TOKEN_EMBED.POSITION = 'none'

_C.MODEL.TOKEN_EMBED.POSITION_MAX_LEN = 5000

_C.MODEL.TOKEN_EMBED.TYPE_VOCAB_SIZE = 0

_C.MODEL.TOKEN_EMBED.EMBEDDING_WEIGHTS = ''
# ----------------------------------------------------------------------------
# Visual embedding
# ----------------------------------------------------------------------------
_C.MODEL.VISUAL_EMBED = CN()

_C.MODEL.VISUAL_EMBED.NAME = ''

_C.MODEL.VISUAL_EMBED.IN_DIM = 1536

_C.MODEL.VISUAL_EMBED.MOTION_IN_DIM = 2048

_C.MODEL.VISUAL_EMBED.OBJECT_IN_DIM = 2048

_C.MODEL.VISUAL_EMBED.EMBEDDINGS_DIM = 1536

_C.MODEL.VISUAL_EMBED.OUT_DIM = 1024

_C.MODEL.VISUAL_EMBED.G_IN_DIM = 512

_C.MODEL.VISUAL_EMBED.ACTIVATION = 'none'

_C.MODEL.VISUAL_EMBED.ELU_ALPHA = 0.5

_C.MODEL.VISUAL_EMBED.USE_NORM = False

_C.MODEL.VISUAL_EMBED.DROPOUT = 0.0

_C.MODEL.VISUAL_EMBED.LOCATION_SIZE = 0

_C.MODEL.VISUAL_EMBED.CONCAT_METHOD = 'transformer'

# ----------------------------------------------------------------------------
# Pre-training
# ----------------------------------------------------------------------------
_C.MODEL.PRETRAINING = CN()

_C.MODEL.PRETRAINING.MODEL_NAME = 'bert-base-uncased'

_C.MODEL.PRETRAINING.FROM_PRETRAINED = 'bert-base-uncased'

_C.MODEL.PRETRAINING.DO_LOWER_CASE = True

# ----------------------------------------------------------------------------
# BERT
# ----------------------------------------------------------------------------
_C.MODEL.BERT = CN()

_C.MODEL.BERT.HIDDEN_SIZE = 512

_C.MODEL.BERT.HIDDEN_DROPOUT_PROB = 0.1

_C.MODEL.BERT.HIDDEN_ACT = "gelu"

_C.MODEL.BERT.NUM_ATTENTION_HEADS = 8

_C.MODEL.BERT.INTERMEDIATE_SIZE = 2048

_C.MODEL.BERT.INTERMEDIATE_DROP = 0.1

_C.MODEL.BERT.FFN_DROPOUT_PROB = 0.1

_C.MODEL.BERT.ATTENTION_PROBS_DROPOUT_PROB = 0.1

_C.MODEL.BERT.V_TARGET_SIZE = 0

_C.MODEL.BERT.NUM_HIDDEN_LAYERS = 12

_C.MODEL.BERT.NUM_ENCODER_OLAYERS = 12

_C.MODEL.BERT.NUM_DECODER_OLAYERS = 12

_C.MODEL.BERT.LAYER_DROP = 0.0

_C.MODEL.BERT.V_NUM_HIDDEN_LAYERS = 6

_C.MODEL.BERT.V_LAYER_DROP = 0.0

_C.MODEL.BERT.NUM_UNDERSTANDING_LAYERS = 6

_C.MODEL.BERT.U_LAYER_DROP = 0.0

_C.MODEL.BERT.NUM_GENERATION_LAYERS = 6

_C.MODEL.BERT.G_LAYER_DROP = 0.0
# ----------------------------------------------------------------------------
# GPT2
# ----------------------------------------------------------------------------
_C.MODEL.GPT2 = CN()

_C.MODEL.GPT2.N_CTX = 60

_C.MODEL.GPT2.N_POSITIONS = 1024

_C.MODEL.GPT2.N_EMBD = 768

_C.MODEL.GPT2.N_LAYER = 12

_C.MODEL.GPT2.N_HEAD = 12

_C.MODEL.GPT2.LAYER_NORM_EPSILON = 1e-5

_C.MODEL.GPT2.INITIALIZER_RANGE = 0.02

_C.MODEL.GPT2.ATTN_DROP = 0.1

_C.MODEL.GPT2.RESID_DROP = 0.1

_C.MODEL.GPT2.INTERMEDIATE_DROP = 0.1

_C.MODEL.GPT2.PADDING_IDX = -1

_C.MODEL.GPT2.TAU = 0.2

_C.MODEL.GPT2.SCALE = False

_C.MODEL.GPT2.CAN_BE_STATEFUL = True

_C.MODEL.GPT2.INTERMEDIATE_SIZE = 2048

# ----------------------------------------------------------------------------
# ROBERTA
# ----------------------------------------------------------------------------

_C.MODEL.ROBERTA = CN()

_C.MODEL.ROBERTA.PATH = ''

_C.MODEL.ROBERTA.K = 1 


# ----------------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------------
_C.SOLVER = CN()

_C.SOLVER.NAME = 'Adam'

_C.SOLVER.EPOCH = 10

_C.SOLVER.CHECKPOINT_PERIOD = 1

_C.SOLVER.EVAL_PERIOD = 1

_C.SOLVER.BASE_LR = 0.0005

_C.SOLVER.BIAS_LR_FACTOR = 1.0

_C.SOLVER.LR_DECAY = 0.0

_C.SOLVER.WEIGHT_DECAY = 0.0

_C.SOLVER.WEIGHT_DECAY_NORM = 0.0

_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0

_C.SOLVER.INITIAL_ACCUMULATOR_VALUE = 0.0

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.DAMPENING = 0.0

_C.SOLVER.NESTEROV = 0.0

_C.SOLVER.ALPHA = 0.99

_C.SOLVER.BETAS = [0.9, 0.999]

_C.SOLVER.EPS = 1e-8

_C.SOLVER.AMSGRAD = False

_C.SOLVER.CENTERED = False

_C.SOLVER.GRAD_CLIP_TYPE = 'norm' # norm, value

_C.SOLVER.GRAD_CLIP = 0.1

_C.SOLVER.NORM_TYPE = 2.0

_C.SOLVER.WRITE_PERIOD = 20

# ----------------------------------------------------------------------------
# lr scheduler
# ----------------------------------------------------------------------------
_C.LR_SCHEDULER = CN()

_C.LR_SCHEDULER.NAME = 'StepLR'

_C.LR_SCHEDULER.STEP_SIZE = 3

_C.LR_SCHEDULER.GAMMA = 0.1

_C.LR_SCHEDULER.MODEL_SIZE = -1 # for Noam only

_C.LR_SCHEDULER.FACTOR = 1.0 # for Noam only

_C.LR_SCHEDULER.WARMUP = 0 # epoch, for WarmupXXX; iteration, for Noam

_C.LR_SCHEDULER.MIN_LR = 0.00001 

_C.LR_SCHEDULER.STEPS = (3,) # for WarmupMultiStep only

_C.LR_SCHEDULER.WARMUP_FACTOR = 0.0 # for WarmupMultiStep only

_C.LR_SCHEDULER.WARMUP_METHOD = "linear" # for WarmupMultiStep only

# ---------------------------------------------------------------------------- #
# Losses
# ---------------------------------------------------------------------------- #
_C.LOSSES = CN()

_C.LOSSES.NAMES = ['']

_C.LOSSES.LABELSMOOTHING = 0.1

_C.LOSSES.MARGIN = 0.2

_C.LOSSES.MAX_VIOLATION = True

_C.LOSSES.OBJECT = 0.2

_C.LOSSES.SOFT = 0.5


# ---------------------------------------------------------------------------- #
# SCORER options
# ---------------------------------------------------------------------------- #
_C.SCORER = CN()

_C.SCORER.NAME = ''

_C.SCORER.TYPES = ['']

_C.SCORER.WEIGHTS = [1.0]

_C.SCORER.GT_PATH = 'coco_train_gts.pkl'

_C.SCORER.CIDER_CACHED = 'coco_train_cider.pkl'

_C.SCORER.EOS_ID = 0

# ---------------------------------------------------------------------------- #
# Decode strategy
# ---------------------------------------------------------------------------- #
_C.DECODE_STRATEGY = CN()

_C.DECODE_STRATEGY.NAME = 'BeamSearcher'

_C.DECODE_STRATEGY.BEAM_SIZE = 1

_C.DECODE_STRATEGY.BOS_TOKEN_ID = 5493

_C.DECODE_STRATEGY.EOS_TOKEN_ID = 0

# ---------------------------------------------------------------------------- #
# INFERENCE options
# ---------------------------------------------------------------------------- #
_C.INFERENCE = CN()

_C.INFERENCE.NAME = ''

_C.INFERENCE.VOCAB = 'coco_vocabulary.txt'

_C.INFERENCE.ID_KEY = 'image_id'

_C.INFERENCE.VALUE = 'caption'

_C.INFERENCE.VAL_ANNFILE = 'captions_val5k.json'

_C.INFERENCE.TEST_ANNFILE = 'captions_test5k.json'

_C.INFERENCE.GENERATION_MODE = True

_C.INFERENCE.VAL_EVAL_START = -1

_C.INFERENCE.TEST_EVAL_START = -1

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "./output"

_C.SEED = -1

_C.CUDNN_BENCHMARK = True