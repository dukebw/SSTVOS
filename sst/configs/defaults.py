import os

from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the maximum image side during training will be
# INPUT.MAX_SIZE_TRAIN, while for testing it will be
# INPUT.MAX_SIZE_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.EXP_NAME = "resnet101_cfbi"

_C.DIR_ROOT = "./"
_C.DIR_DATA = os.path.join(_C.DIR_ROOT, "datasets")
_C.DIR_DAVIS = os.path.join(_C.DIR_DATA, "DAVIS")
_C.DIR_YTB = os.path.join(_C.DIR_DATA, "YTB/train")
_C.DIR_YTB_EVAL = os.path.join(_C.DIR_DATA, "YTB/valid")
_C.DIR_RESULT = os.path.join(_C.DIR_ROOT, "result", _C.EXP_NAME)
_C.DIR_CKPT = os.path.join(_C.DIR_RESULT, "ckpt")
_C.DIR_LOG = os.path.join(_C.DIR_RESULT, "log")
_C.DIR_IMG_LOG = os.path.join(_C.DIR_RESULT, "log", "img")
_C.DIR_TB_LOG = os.path.join(_C.DIR_RESULT, "log", "tensorboard")
_C.DIR_EVALUATION = os.path.join(_C.DIR_RESULT, "eval")

_C.DATASETS = ["davis2017"]

_C.DATA = CN()
_C.DATA.WORKERS = 4
_C.DATA.RANDOMCROP = (465, 465)
_C.DATA.RANDOMFLIP = 0.5
_C.DATA.MAX_CROP_STEPS = 5
_C.DATA.MIN_SCALE_FACTOR = 1.0
_C.DATA.MAX_SCALE_FACTOR = 1.3
_C.DATA.SHORT_EDGE_LEN = 480
_C.DATA.RANDOM_REVERSE_SEQ = True
_C.DATA.DAVIS_REPEAT = 30
_C.DATA.CURR_SEQ_LEN = 3
_C.DATA.RANDOM_GAP_DAVIS = 3
_C.DATA.RANDOM_GAP_YTB = 3

_C.PRETRAIN = True
_C.PRETRAIN_FULL = False
_C.PRETRAIN_MODEL = "./pretrain_models/resnet101-deeplabv3p.pth.tar"

_C.MODEL = CN()
_C.MODEL.BACKBONE = "resnet"
_C.MODEL.MODULE = "networks.cfbi.cfbi"
_C.MODEL.OUTPUT_STRIDE = 16
_C.MODEL.ASPP_OUTDIM = 256
_C.MODEL.SHORTCUT_DIM = 48
_C.MODEL.SEMANTIC_EMBEDDING_DIM = 100
_C.MODEL.HEAD_EMBEDDING_DIM = 256
_C.MODEL.PRE_HEAD_EMBEDDING_DIM = 64
_C.MODEL.GN_GROUPS = 32
_C.MODEL.GN_EMB_GROUPS = 25
_C.MODEL.MULTI_LOCAL_DISTANCE = [2, 4, 6, 8, 10, 12]
_C.MODEL.LOCAL_DOWNSAMPLE = True
_C.MODEL.REFINE_CHANNELS = 64  # n * 32
_C.MODEL.LOW_LEVEL_INPLANES = 256 if _C.MODEL.BACKBONE == "resnet" else 24
_C.MODEL.RELATED_CHANNELS = 64
_C.MODEL.EPSILON = 1e-5
_C.MODEL.MATCHING_BACKGROUND = True
_C.MODEL.GCT_BETA_WD = True
_C.MODEL.FLOAT16_MATCHING = False
_C.MODEL.FREEZE_BN = True
_C.MODEL.FREEZE_BACKBONE = False
_C.MODEL.IS_TRANSFORMER = False
_C.MODEL.USE_PREV_EMBEDDING_AND_LABEL_AS_FEATURES = True
_C.MODEL.USE_LOCAL_MATCHING = True

_C.MODEL.TRANSFORMER = CN()
_C.MODEL.TRANSFORMER.ATTENTION_FEATURES = "No"
_C.MODEL.TRANSFORMER.ATTENTION_NORM_TYPE = "FilterResponseNorm"
_C.MODEL.TRANSFORMER.ATTENTION_NORMALIZATION = "Distance"
_C.MODEL.TRANSFORMER.ATTENTION_STRIDE_SPATIAL = [-1]
_C.MODEL.TRANSFORMER.ATTENTION_STRIDE_TEMPORAL = "Dense"
_C.MODEL.TRANSFORMER.ATTENTION_TYPE = "Full"
_C.MODEL.TRANSFORMER.ATTENTION_SPARSITY = "Local"
_C.MODEL.TRANSFORMER.HISTORY_SIZE = 1
_C.MODEL.TRANSFORMER.NUM_ATTN_HEADS = 8
_C.MODEL.TRANSFORMER.NUM_ATTN_LAYERS = 1
_C.MODEL.TRANSFORMER.POSITIONAL_ENCODING = "NonPositional"
_C.MODEL.TRANSFORMER.POSITIONAL_ENCODING_DIM = 64
_C.MODEL.TRANSFORMER.POSITIONAL_ENCODING_MAX_LEN_T = 300
_C.MODEL.TRANSFORMER.POSITIONWISE_FEEDFORWARD_NONLINEARITY = "FRN"
_C.MODEL.TRANSFORMER.USE_RELATIVE_POSITIONAL_EMBEDDINGS = False
_C.MODEL.TRANSFORMER.USE_WARPED_FGBG_FEATURES = True
_C.MODEL.TRANSFORMER.ARE_FEATURES_DOWNSAMPLED = False

_C.TRAIN = CN()
_C.TRAIN.TOTAL_STEPS = 25000
_C.TRAIN.START_STEP = 0
_C.TRAIN.LR = 0.01
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.COSINE_DECAY = False
_C.TRAIN.WARM_UP_STEPS = 1000
_C.TRAIN.WEIGHT_DECAY = 15e-5
_C.TRAIN.POWER = 0.9
_C.TRAIN.GPUS = 4
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.START_SEQ_TRAINING_STEPS = _C.TRAIN.TOTAL_STEPS // 2
_C.TRAIN.TBLOG = False
_C.TRAIN.TBLOG_STEP = 60
_C.TRAIN.LOG_STEP = 100
_C.TRAIN.IMG_LOG = False
_C.TRAIN.TOP_K_PERCENT_PIXELS = 0.15
_C.TRAIN.HARD_MINING_STEP = _C.TRAIN.TOTAL_STEPS // 2
_C.TRAIN.CLIP_GRAD_NORM = 5.0
_C.TRAIN.SAVE_STEP = 1000
_C.TRAIN.MAX_KEEP_CKPT = 8
_C.TRAIN.RESUME = False
_C.TRAIN.RESUME_CKPT = None
_C.TRAIN.RESUME_STEP = 0
_C.TRAIN.AUTO_RESUME = True
_C.TRAIN.GLOBAL_ATROUS_RATE = 1
_C.TRAIN.LOCAL_ATROUS_RATE = 1
_C.TRAIN.LOCAL_PARALLEL = True
_C.TRAIN.GLOBAL_CHUNKS = 20
_C.TRAIN.DATASET_FULL_RESOLUTION = True
_C.TRAIN.DO_EVAL_DURING_TRAINING = True

_C.TEST = CN()
_C.TEST.GPU_ID = 0
_C.TEST.DATASET = "davis2017"
_C.TEST.DATASET_FULL_RESOLUTION = False
_C.TEST.DATASET_SPLIT = ["val"]
_C.TEST.CKPT_PATH = None
_C.TEST.CKPT_STEP = None  # if "None", evaluate the latest checkpoint.
_C.TEST.FLIP = False
_C.TEST.MULTISCALE = [1]
_C.TEST.MIN_SIZE = None
_C.TEST.MAX_SIZE = 800 * 1.3 if _C.TEST.MULTISCALE == [1.0] else 800
_C.TEST.WORKERS = 4
_C.TEST.GLOBAL_CHUNKS = 4
_C.TEST.GLOBAL_ATROUS_RATE = 1
_C.TEST.LOCAL_ATROUS_RATE = 1
_C.TEST.LOCAL_PARALLEL = True

# dist
_C.DIST = CN()
_C.DIST.ENABLE = True
_C.DIST.BACKEND = "gloo"
_C.DIST.URL = "file:///home/ubuntu/work/CFBI/sharefile"
_C.DIST.START_GPU = 0
