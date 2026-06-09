!mkdir /kaggle/temp
!cp -r /kaggle/input/byu-final-dataset/byu-main/byu-main /kaggle/working/byu
!cp -r /kaggle/input/dataset-czii-pkgs/pkgs /kaggle/temp


!pwd
!ls -la /kaggle/working/byu


%%time
!cp -r /kaggle/input/czii-additional-packages/* /kaggle/temp/pkgs
!pip install --no-index --find-links=/kaggle/temp/pkgs/ /kaggle/temp/pkgs/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4
!pip install --no-index --find-links=/kaggle/temp/pkgs/ /kaggle/temp/pkgs/fvcore-0.1.5.post20221221/fvcore-0.1.5.post20221221
!pip install --no-index --find-links=/kaggle/temp/pkgs/ /kaggle/temp/pkgs/pytorchvideo-0.1.5/pytorchvideo
!pip install --no-index --find-links=/kaggle/temp/pkgs/ /kaggle/temp/pkgs/detectron2-0.6/detectron2

!pip install /kaggle/temp/pkgs/asciitree-0.3.3/asciitree-0.3.3
!pip install --no-index --find-links=/kaggle/temp/pkgs/ hydra-core copick lightning monai simplejson iopath fvcore connected-components-3d
!pip install --no-index /kaggle/input/tensorrt-10-1-0/tensorrt*.whl
!pip install --no-index --find-links=/kaggle/input/onnx-wheels/ onnx onnxoptimizer onnxsim onnxruntime-gpu


%cd /kaggle/working/byu


import sys
sys.path.extend([
    '/kaggle/working/byu/yagm/src',
    '/kaggle/working/byu/src',
    '/kaggle/working/byu/',
    '/kaggle/working/byu/third_party/segmentation_models_pytorch_3d',
    '/kaggle/working/byu/third_party/slowfast',
    '/kaggle/working/byu/third_party/timm_3d'
])
print('SYS PATH:', sys.path, sep = '\n')








!rm -rf /kaggle/working/byu/assets
!ln -s /kaggle/input/byu-final-dataset /kaggle/working/byu/assets


# import gc
# import logging
# import os
# import random

# import torch
# from omegaconf import OmegaConf
# from torch import nn
# from torch.nn import functional as F
# from yagm.utils import lightning as l_utils
# from yagm.utils.torch2onnx import onnx2trt, torch2onnx

# from byu.data.datasets.heatmap_3d_dataset import Heatmap3dDataset
# from byu.data.io import OpencvTomogramLoader

# logging.basicConfig(level=logging.INFO)

# # DATA_DIR = "/home/dangnh36/datasets/.comp/byu/"
# # DATASET_3D_CONFIG = f"""
# # env:
# #     data_dir: {DATA_DIR}
# # cv:
# #     strategy: skf4_rd42
# #     num_folds: 4
# #     fold_idx: 0
# #     train_on_all: False
# # loader:
# #     num_workers: 1
# # data:
# #     label_fname: gt_v2  # gt | gt_v2 | gt_v3 | all_gt | all_gt_v3
# #     patch_size: [224, 448, 448]
# #     start: [0, 0, 0]
# #     overlap: [0, 0, 0]
# #     border: [0, 0, 0]

# #     sigma: 0.2
# #     fast_val_workers: 1
# #     fast_val_prefetch: 1
# #     io_backend: cv2  # cv2, cv2_seq, npy, cache
# #     crop_outside: True
# #     ensure_fg: False
# #     label_smooth: [0.0,1.0]
# #     filter_rule: null  # null | eq1 | le1

# #     sampling:
# #         method: pre_patch  # pre_patch | rand_crop
# #         pre_patch:
# #             fg_max_dup: 1
# #             bg_ratio: 0.0
# #             bg_from_pos_ratio: 0.01
# #             overlap: [0, 0, 0]
# #         rand_crop:
# #             random_center: True
# #             margin: 0.25
# #             auto_correct_center: True
# #             pos_weight: 1
# #             neg_weight: 1
# #     transform:
# #         resample_mode: trilinear # F.grid_sample() mode
# #         target_spacing: [16,16,16]

# #         heatmap_mode: gaussian
# #         heatmap_stride: [1,1,1]
# #         heatmap_same_sigma: False
# #         heatmap_same_std: False
# #         lazy: True
# #         device: null
# #     aug:
# #         enable: True
# #         zoom_prob: 0.4
# #         zoom_range: [[0.6, 1.2], [0.6, 1.2], [0.6, 1.2]]  # (X, Y, Z) or (H, W, D)
# #         # affine1
# #         affine1_prob: 0.5
# #         affine1_scale: 0.3  # max_skew_xy = 1.3 / 0.7 = 1.86
# #         # affine2
# #         affine2_prob: 0.25
# #         affine2_rotate_xy: 15 # degrees
# #         affine2_scale: 0.3  # max_skew_xy = 1.3 / 0.7 = 1.86
# #         affine2_shear: 0.2

# #         rand_shift: False  # only used in rand_crop, very useful if random_center=auto_correct_center=False

# #         # no lazy, can't properly transform points
# #         grid_distort_prob: 0.0
# #         smooth_deform_prob: 0.0

# #         intensity_prob: 0.5
# #         smooth_prob: 0.0
# #         hist_equalize: False
# #         downsample_prob: 0.2
# #         coarse_dropout_prob: 0.1

# #         # MIXER
# #         mixup_prob: 0.0
# #         cutmix_prob: 0.0
# #         mixer_alpha: 1.0
# #         mixup_target_mode: max
# #     tta:
# #         # enable: [zyx]
# #         enable: [zyx, zxy, zyx_x, zyx_y]
# # """


# # global_cfg = OmegaConf.create(DATASET_3D_CONFIG)
# # dataset = Heatmap3dDataset(global_cfg, stage="train")
# # if dataset.stage != "train":
# #     dataset.fast_val_tomo_loader.start()

# # imgs = []
# # random.seed(42)
# # # select_idxs = random.choices(list(range(len(dataset))), k = 50)
# # select_idxs = list(range(1))
# # print("SELECT RANDOM INDICES:", select_idxs)
# # for idx in select_idxs:
# #     sample = dataset[idx]
# #     img = sample["image"]
# #     assert img.dtype == torch.uint8
# #     print(idx, img.shape, img.dtype)
# #     imgs.append(img[None].float().contiguous())
# # del dataset
# # gc.collect()


# tomo_loader = OpencvTomogramLoader()
# # TOMO_ROOT_DIR = "/home/dangnh36/datasets/.comp/byu/raw/train/"
# TOMO_ROOT_DIR = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train/'
# tomo = tomo_loader.load(os.path.join(TOMO_ROOT_DIR, "tomo_bfd5ea"))
# tomo = (
#     torch.from_numpy(tomo[None, 10:13, 0:896, 0:896]).float().contiguous()
# )  # (1, 1, Z, Y, X)
# print(tomo.shape)
# imgs = [tomo]


# ############### LOAD MODEL ###############
# DEVICE = "cuda:0"
# CONVERT_ONNX = True
# CONVERT_TRT = True






# CONFIG_PATH = "/kaggle/input/byu-final-dataset/EXP25_COATLITEMEDIUM_ALLGTV3_config.yaml"
# WEIGHT_PATH = "/kaggle/input/byu-final-dataset/EXP25_COATLITEMEDIUM_ALLGTV3_ep4_step20000.ckpt"
# TRT_SAVE_PATH = '/kaggle/working/EXP25_COATLITEMEDIUM_ALLGTV3_ep4_step20000.engine'
# WORKSPACE_SIZE = int(5 * (2**30))
# PRECISION = 'fp16'



# ONNX_SAVE_PATH = None
# MIN_BS, OPT_BS, MAX_BS = 8, 8, 8

# if TRT_SAVE_PATH is None:
#     TRT_SAVE_PATH = WEIGHT_PATH.replace(".ckpt", ".engine")
    
# if ONNX_SAVE_PATH is None:
#     ONNX_SAVE_PATH = TRT_SAVE_PATH.replace(".engine", ".onnx")

# os.makedirs(os.path.dirname(ONNX_SAVE_PATH), exist_ok=True)
# os.makedirs(os.path.dirname(TRT_SAVE_PATH), exist_ok=True)


# if CONVERT_ONNX:
#     device = torch.device(DEVICE)
#     cfg = OmegaConf.load(CONFIG_PATH)

#     ######### OVERWRITE SOME CONFIGS ##########
#     cfg.misc.log_model = False
#     cfg.ema.val_decays = [0.99]
#     cfg.ckpt.strict = False

#     ######### OVERWRITE SOME CONFIGS ##########
#     # OVERWRITE SOME CONFIG
#     try:
#         cfg.model.encoder.pretrained = False
#     except:
#         pass

#     # for backward-compatibility
#     OmegaConf.set_readonly(cfg, False)
#     # just keep outputing heatmap
#     cfg.model.reg_head.enable = False
#     cfg.model.dsnt = OmegaConf.create()
#     cfg.model.dsnt.enable = False
#     OmegaConf.set_readonly(cfg, True)
#     ##########

#     task = l_utils.build_task(cfg)
#     print(task.device)
#     l_utils.load_lightning_state_dict(
#         model=task,
#         ckpt_path=WEIGHT_PATH,
#         cfg=cfg,
#     )
#     print(f"Loaded state dict")

#     class ModelWrapper(nn.Module):
#         def __init__(self, model):
#             super().__init__()
#             self._model = model

#         def forward(self, x):
#             ret = self._model(x)
#             assert isinstance(ret, (list, tuple)) and len(ret) == 4
#             return ret[2]

#     torch_model = ModelWrapper(task.model)
#     print(torch_model)
#     torch_model.eval().to(device)
#     print('DEVICE:', device)

#     torch2onnx(
#         torch_model=torch_model,
#         sample_inputs=[imgs[0]],
#         input_names=["image"],
#         output_names=["heatmap"],
#         save_path=ONNX_SAVE_PATH,
#         precision="fp32",
#         device=DEVICE,
#         dynamic_batching=True,
#         batch_axis=0,
#         validate_fn=None,
#         rtol=1e-3,
#         atol=1e-4,
#         verbose=True,
#     )
#     del torch_model
#     gc.collect()
#     torch.cuda.empty_cache()


# if CONVERT_TRT:
#     try:
#         engine = onnx2trt(
#             onnx_file_path=ONNX_SAVE_PATH,
#             engine_file_path=TRT_SAVE_PATH,
#             input_shape=imgs[0].shape[1:],
#             precision=PRECISION,
#             min_batch=MIN_BS,
#             opt_batch=OPT_BS,
#             max_batch=MAX_BS,
#             enable_dynamic_batching=True,
#             workspace_size=WORKSPACE_SIZE,
#         )

#         if engine is not None:
#             print("Conversion successful!")
#         else:
#             print("Conversion failed!")
#     except Exception as e:
#         print(f"Conversion error: {e}")





























%%writefile /kaggle/working/byu/submit_3d.py

import sys
sys.path.extend([
    '/kaggle/working/byu/yagm/src',
    '/kaggle/working/byu/src',
    '/kaggle/working/byu/',
    '/kaggle/working/byu/third_party/segmentation_models_pytorch_3d',
    '/kaggle/working/byu/third_party/slowfast',
    '/kaggle/working/byu/third_party/timm_3d'
])
print('SYS PATH:', sys.path, sep = '\n')


import warnings

warnings.simplefilter("ignore")
import gc
import json
import logging
import multiprocessing as mp
import os
import queue
import shutil
import sys
import threading
import time
from functools import partial

import cc3d
import cv2
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from torch.utils.data import default_collate
from tqdm import tqdm
from yagm.tasks.base_task import BaseTask
from yagm.transforms.keypoints.decode import decode_heatmap_3d, decode_segment_mask_3d
from yagm.transforms.sliding_window import get_sliding_patch_positions
from yagm.transforms.tta_3d import build_tta
from yagm.utils import hydra as hydra_utils
from yagm.utils import lightning as l_utils
from yagm.utils.concurrent import ShmTensor

# Setting up custom logging
from yagm.utils.logging import init_logging, setup_logging

from byu.data.io import MultithreadOpencvTomogramLoader
from byu.inference import zoo
from byu.inference.tensorrt_engine import ThreadsafeTRTEngine
from byu.utils.data import SubmissionDataFrame
from byu.utils.metrics import compute_metrics, kaggle_score
from byu.utils.viz import viz_byu
from byu.utils.wbf import weighted_boxes_fusion_3d

try:
    hydra_utils.init_hydra()
except:
    print("SKIP RE-INIT HYDRA")

### SETUP LOGGER FOR IPYTHON ###
# ref: https://github.com/ipython/ipykernel/issues/111
# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Create STDERR handler
handler = logging.StreamHandler(sys.stderr)
# Create formatter and add it to the handler
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
# Set STDERR handler as the only handler
logger.handlers = [handler]


##################################################
###### MODEL SIGNATURES OVERRIDES
##################################################
ALL_SUPPORTED_TTAS = [
    "zyx",
    "zyx_x",
    "zyx_y",
    "zyx_z",
    "zyx_xy",
    "zyx_xz",
    "zyx_yz",
    "zyx_xyz",
    "zxy",
    "zxy_x",
    "zxy_y",
    "zxy_z",
    "zxy_xy",
    "zxy_xz",
    "zxy_yz",
    "zxy_xyz",
]

_SIG_TEMPLATE = {
    "backend": "torch",  # torch | trt
    "trt_path": "",
    "config_path": "",
    "torch_path": "",
    "ema": 0.99,
}

# DENSENET121 (LB 86.0)
# https://www.kaggle.com/code/dangnh0611/submit-heatmap-final?scriptVersionId=243314913
EXP28_DENSNET121_ALLGTV3 = {
    "backend": "torch",  # torch | trt
    "trt_path": "",
    "config_path": "/kaggle/input/byu-final-dataset/EXP28_DENSNET121_ALLGTV3_config.yaml",
    "torch_path": "/kaggle/input/byu-final-dataset/EXP28_DENSNET121_ALLGTV3_ep1_step32000.ckpt",
    "ema": 0.99,
}

# X3D (LB 85.8)
# https://www.kaggle.com/code/dangnh0611/submit-heatmap-final?scriptVersionId=242747515
EXP16_X3DM_ALL_TRAIN_EXT = {
    "backend": "torch",  # torch | trt
    "trt_path": "",
    "config_path": "/kaggle/input/byu-final-dataset/EXP16_X3DM_ALL_TRAIN_EXT_config.yaml",
    "torch_path": "/kaggle/input/byu-final-dataset/EXP16_X3DM_ALL_TRAIN_EXT_ep1_step30000.ckpt",
    "ema": 0.99,
}

# RESNEXT50 (LB 85.8)
# https://www.kaggle.com/code/dangnh0611/submit-heatmap-final?scriptVersionId=243403282
EXP31_RESNEXT50_ALLGTV3 = {
    "backend": "torch",  # torch | trt
    "trt_path": "",
    "config_path": "/kaggle/input/byu-final-dataset/EXP31_RESNEXT50_ALLGTV3_config.yaml",
    "torch_path": "/kaggle/input/byu-final-dataset/EXP31_RESNEXT50_ALLGTV3_ep1_step32000.ckpt",
    "ema": 0.99,
}

# R50 (LB 85.4)
# https://www.kaggle.com/code/dangnh0611/submit-heatmap-final?scriptVersionId=243127441
EXP26_R50_ALLGTV3 = {
    "backend": "torch",  # torch | trt
    "trt_path": "",
    "config_path": "/kaggle/input/byu-final-dataset/EXP26_R50_ALLGTV3_config.yaml",
    "torch_path": "/kaggle/input/byu-final-dataset/EXP26_R50_ALLGTV3_ep1_step32000.ckpt",
    "ema": 0.99,
}

# 2.5D CONVNEXT_TINY_AVG (LB 83.7), for diversity
# https://www.kaggle.com/code/dangnh0611/submit-heatmap-final?scriptVersionId=243123609
EXP24_CONVNEXT_TINY_AVG_ALLGTV3 = {
    "backend": "torch",  # torch | trt
    "trt_path": "",
    "config_path": "/kaggle/input/byu-final-dataset/EXP24_CONVNEXT_TINY_AVG_ALLGTV3_config.yaml",
    "torch_path": "/kaggle/input/byu-final-dataset/EXP24_CONVNEXT_TINY_AVG_ALLGTV3_ep2_step36000.ckpt",
    "ema": 0.99,
}



JOBS = [
    [
        {
            "sig": EXP28_DENSNET121_ALLGTV3,
            "tta": ["zyx", "zxy"],
            "weight": [1.0, 1.0],
            "agg_name": ["DENSENET-zyx", "DENSENET-zxy"],
        },
        {
            "sig": EXP16_X3DM_ALL_TRAIN_EXT,
            "tta": ["zyx_x", "zyx_y"],
            "weight": [1.0, 1.0],
            "agg_name": ["X3D-zyx_x", "X3D-zyx_y"],
        },
        {
            "sig": EXP31_RESNEXT50_ALLGTV3,
            "tta": ["zyx_xy"],
            "weight": [1.0],
            "agg_name": ["RESNEXT50-zyx_xy"],
        },
        # {
        #     "sig": EXP26_R50_ALLGTV3,
        #     "tta": ["zxy_z"],
        #     "weight": [1.0],
        #     "agg_name": ["R50-zyx_xy"],
        # },
        {
            "sig": EXP24_CONVNEXT_TINY_AVG_ALLGTV3,
            "tta": ["zxy_xy"],
            "weight": [1.0],
            "agg_name": ["CONVNEXT-zxy_xy"],
        },
    ]
]



# JOBS = [
#     [
#         {
#             "sig": EXP28_DENSNET121_ALLGTV3,
#             "tta": ["zyx", "zxy"],
#             "weight": [1.0, 1.0],
#             "agg_name": ["DENSENET-zyx", "DENSENET-zxy"],
#         },
#         {
#             "sig": EXP26_R50_ALLGTV3,
#             "tta": ["zyx_x", "zyx_y"],
#             "weight": [1.0, 1.0],
#             "agg_name": ["R50-zyx_x", "R50-zyx_y"],
#         },
#         {
#             "sig": EXP16_X3DM_ALL_TRAIN_EXT,
#             "tta": ["zyx_xy", "zxy_xy"],
#             "weight": [1.0, 1.0],
#             "agg_name": ["X3D-zyx_xy", "X3D-zxy_xy"],
#         },
#     ]
# ]


##################################################
###### GLOBAL VARS
##################################################
_CV2_TOMO_LOADER_NUM_THREADS = 8

PATCH_SIZE = [224, 448, 448]
PATCH_BORDER = [0, 0, 0]
PATCH_OVERLAP = [0, 0, 0]

IO_BACKEND = "cv2"
BASE_SPACING = 13.1  # avg 15.6
# TARGET_SPACINGS = [13.1, 16.0, 19.7]
TARGET_SPACINGS = [[16.0, 16.0, 16.0]]
TOMO_SPACING_DEVICE = "cpu"
TOMO_SPACING_MODE = "trilinear"

HEATMAP_AGG_MODE = "avg"
HEATMAP_AGG_LOGITS = True
HEATMAP_INTERPOLATION_MODE = "trilinear"
FORWARD_SAVE_MEMORY = False
BATCH_SIZE = 1
HEATMAP_STRIDE = 8

# MASK CC3D DECODING
DECODE_CC3D_CONF_THRES = 0.05
DECODE_CC3D_RADIUS_FACTOR = 0.1
DECODE_CC3D_CONF_MODE = "prob"  # volume | prob | fuse
DECODE_CC3D_PROB_MODE = "center"  # center | mean | max
# HEATMAP NMS DECODING
DECODE_NMS_BLUR_SIGMA = None
###### MAIN DECODE PARAMS ######
DECODE_METHOD = "nms"  # nms, cc3d
DECODE_CONF_THRES = 0.05
QUANTILE_THRES = 0.55

SAVE_HEATMAP_NPY = False
DECODE_HEATMAP_TO_CSV = True
CSV_ENSEMBLE_MODE = "wbf"  # max | wbf
WBF_CONF_TYPE = "avg"  # avg | max
WBF_CONF_THRES = 0.2

VIZ_ENABLE = False


# automatic determine the MODE
if os.path.isdir("/kaggle/input/byu-locating-bacterial-flagellar-motors-2025"):
    MODE = "KAGGLE"
else:
    MODE = "LOCAL"
# manual override
# MODE = "KAGGLE_VAL"

# LOCAL DEVELOPMENT ENV
if MODE == "LOCAL":
    DEVICES = [0]
    ASSETS_DIR = "assets/"
    WORKING_DIR = "outputs/submit/working/"

    DATA_DIR = "/home/dangnh36/datasets/.comp/byu/raw/train"
    VAL_GT_CSV_PATH = "/home/dangnh36/datasets/.comp/byu/processed/gt_v3.csv"
    with open(
        "/home/dangnh36/datasets/.comp/byu/processed/cv/v3/skf4_rd42.json", "r"
    ) as f:
        cv_meta = json.load(f)
        ALL_TOMO_IDS = cv_meta["folds"][0]["val"][:]  # fold 0

    # DATA_DIR = "/home/dangnh36/datasets/.comp/byu/processed/pseudo_test/tomograms"
    # VAL_GT_CSV_PATH = "/home/dangnh36/datasets/.comp/byu/processed/pseudo_test/gt.csv"
    # ALL_TOMO_IDS = sorted(os.listdir(DATA_DIR)) * 10

    GT_DF = pd.read_csv(VAL_GT_CSV_PATH)
    GT_DF = GT_DF[GT_DF["tomo_id"].isin(ALL_TOMO_IDS)].reset_index(drop=True)
    _tomo2spacing = {}
    for i, row in GT_DF.iterrows():
        _tomo2spacing[row["tomo_id"]] = row["voxel_spacing"]
    ALL_TOMO_SPACINGS = [_tomo2spacing[tomo_id] for tomo_id in ALL_TOMO_IDS]

    # TOMO_PRODUCER_NUM_WORKERS = 8
    # TOMO_PRODUCER_PREFETCH = 8
    # PATCHES_PRODUCER_NUM_WORKERS = 2
    # PATCHES_PRODUCER_PREFETCH = 8
    # DATALOADER_PREFETCH = 8

    TOMO_PRODUCER_NUM_WORKERS = 1
    TOMO_PRODUCER_PREFETCH = 1
    PATCHES_PRODUCER_NUM_WORKERS = 1
    PATCHES_PRODUCER_PREFETCH = 2
    DATALOADER_PREFETCH = 8

# VALIDATION ON KAGGLE
elif MODE == "KAGGLE_VAL":
    DEVICES = [0, 1]
    ASSETS_DIR = "/kaggle/input/byu-final-dataset/"
    DATA_DIR = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train"
    WORKING_DIR = "/kaggle/working/"
    VAL_GT_CSV_PATH = "/kaggle/input/byu-checkpoints/gt_v2.csv"
    with open("/kaggle/input/byu-checkpoints/skf4_rd42.json", "r") as f:
        cv_meta = json.load(f)
        ALL_TOMO_IDS = cv_meta["folds"][0]["val"][:50]  # fold 0, top 10 first tomo
    GT_DF = pd.read_csv(VAL_GT_CSV_PATH)
    GT_DF = GT_DF[GT_DF["tomo_id"].isin(ALL_TOMO_IDS)].reset_index(drop=True)
    _tomo2spacing = {}
    for i, row in GT_DF.iterrows():
        _tomo2spacing[row["tomo_id"]] = row["voxel_spacing"]
    ALL_TOMO_SPACINGS = [_tomo2spacing[tomo_id] for tomo_id in ALL_TOMO_IDS]

    TOMO_PRODUCER_NUM_WORKERS = 1
    TOMO_PRODUCER_PREFETCH = 1
    PATCHES_PRODUCER_NUM_WORKERS = 1
    PATCHES_PRODUCER_PREFETCH = 2
    DATALOADER_PREFETCH = 8

# PRIVATE TEST SUBMISSION
elif MODE == "KAGGLE":
    DEVICES = [0, 1]
    ASSETS_DIR = "/kaggle/input/byu-final-dataset/"
    DATA_DIR = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/test"
    WORKING_DIR = "/kaggle/working/"
    ALL_TOMO_IDS = sorted(os.listdir(DATA_DIR))
    ALL_TOMO_SPACINGS = [BASE_SPACING] * len(ALL_TOMO_IDS)

    TOMO_PRODUCER_NUM_WORKERS = 1
    TOMO_PRODUCER_PREFETCH = 1
    PATCHES_PRODUCER_NUM_WORKERS = 1
    PATCHES_PRODUCER_PREFETCH = 2
    DATALOADER_PREFETCH = 8

else:
    raise ValueError

TMP_CSV_DIR = os.path.join(WORKING_DIR, "tmp_csv")
TMP_VIZ_DIR = os.path.join(WORKING_DIR, "tmp_viz")
SAVE_HEATMAP_NPY_DIR = os.path.join("/kaggle/temp/tmp_heatmap")

# RECHECK SOME CONDITIONS
assert len(ALL_TOMO_IDS) == len(ALL_TOMO_SPACINGS)
# assert not (SAVE_HEATMAP_NPY and DECODE_HEATMAP_TO_CSV) and (SAVE_HEATMAP_NPY or DECODE_HEATMAP_TO_CSV)


###################################################
###### FUNCTIONS/CLASSES DEFINITION
###################################################
def exception_printer(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception("Exception %s in %s: %s", type(e), func.__name__, e)
            import traceback

            traceback.print_exc()
            raise e

    return wrapper


def clear_dir_content(folder_path):
    if os.path.isdir(folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)


class Model3dWrapper(nn.Module):
    def __init__(self, sig, gpu_id, act="sigmoid"):
        super().__init__()
        if sig["backend"] == "trt":
            logger.info("Loading TRT model with config:\n%s", sig)
            self.model = ThreadsafeTRTEngine(sig["trt_path"], gpu_id)
        elif sig["backend"] == "torch":
            cfg = OmegaConf.load(sig["config_path"])
            cfg.misc.log_model = False
            cfg.ema.val_decays = [sig["ema"] if sig["ema"] is not None else 0]
            cfg.model.head.ms = [0]
            cfg.ckpt.strict = True

            ######### OVERWRITE SOME CONFIGS ##########
            if (
                "x3d" in cfg.model.encoder._target_
                or "i3d" in cfg.model.encoder._target_
            ):
                cfg.model.encoder.pretrained = None
            elif "smp" in cfg.model.encoder._target_:
                cfg.model.encoder.weights = None
                if "resnet101" in cfg.model.encoder.model_name:
                    cfg.model.encoder.model_name = "resnet101"
            else:
                pass
            # for 2.5D encoder
            try:
                cfg.model.encoder.encoder_2d.pretrained = False
            except:
                pass
            try:
                if (
                    cfg.model.encoder.encoder_2d.model_name
                    == "convnext_nano.r384_ad_in12k"
                ):
                    cfg.model.encoder.encoder_2d.model_name = "convnext_nano"
            except:
                pass
            ##########

            task: BaseTask = l_utils.build_task(cfg)
            l_utils.load_lightning_state_dict(
                model=task,
                ckpt_path=sig["torch_path"],
                cfg=cfg,
            )
            logger.info("Loaded Pytorch state dict from %s", sig["torch_path"])
            # self.model = TorchModelWrapper(task.model)
            self.model = task.model
            del task
            gc.collect()
        else:
            raise ValueError

        self.act_name = act
        if act == "sigmoid":
            self.act = F.sigmoid
            self.nan_to_num = partial(torch.nan_to_num, nan=0.0)
        elif act == "identity":
            self.act = lambda x: x
            self.nan_to_num = partial(
                torch.nan_to_num, nan=torch.finfo(torch.float32).min
            )
        else:
            raise ValueError

    def forward(self, x):
        heatmap = self.act(self.model(x)[0])
        heatmap = self.nan_to_num(heatmap)
        # if torch.isnan(heatmap).any():
        #     print('\n\nNAN!!!')
        #     raise Exception
        return heatmap


def decode_heatmap(pred_heatmap, target_spacing, stride, blur_operator=None):
    radius_voxel = [1000.0 / e / stride for e in target_spacing]
    radius_thres = max(radius_voxel)
    # @TODO - currently, using fixed pool_ksize=3 due to performance issue
    # when using large kernel size with Pytorch (>10 sec on 224x448x448)

    # # maximum ood number which <= radius_thres
    # # 0.7071067811865475 = 1 / sqrt(2)
    # pool_ksize = int((2 * radius_thres * 0.7071067811865475 - 1) // 2 * 2 + 1)
    # # at least 3, [1,1,1] == no pooling, which increase maxRecall but significantly reduce other metrics
    # pool_ksize = max(3, pool_ksize)
    # pool_ksize = [pool_ksize, pool_ksize, pool_ksize]

    pool_ksize = [3, 3, 3]
    logger.debug(
        "Heatmap decode with radius=%s, pool_ksize=%s, radius_thres=%s",
        radius_voxel,
        pool_ksize,
        radius_thres,
    )
    del target_spacing
    assert pred_heatmap.shape[0] == 1
    ret = []
    for channel_idx, heatmap in enumerate(pred_heatmap):
        outputs = decode_heatmap_3d(
            heatmap=heatmap,
            pool_ksize=pool_ksize,
            nms_radius_thres=radius_thres,
            blur_operator=blur_operator,
            conf_thres=DECODE_CONF_THRES,
            # max_dets=5 if (VIZ_ENABLE and MODE != "KAGGLE") else 1,
            max_dets=100,
            timeout=None,
        )
        outputs = outputs.cpu()
        ret.append(outputs)
    return ret


def decode_segment_mask(
    pred_heatmap,
    target_spacing,
    stride,
    prob_thres,
    radius_factor_thres,
    conf_mode="fuse",
    prob_mode="avg",
):
    radius_voxel_thres = [
        1000 / e / stride * radius_factor_thres for e in target_spacing
    ]
    volume_thres = (
        4
        / 3
        * np.pi
        * radius_voxel_thres[0]
        * radius_voxel_thres[1]
        * radius_voxel_thres[2]
    )
    assert len(pred_heatmap.shape) == 4 and pred_heatmap.shape[0] == 1
    ret = []
    for channel_idx, heatmap in enumerate(pred_heatmap):
        keypoints = decode_segment_mask_3d(
            heatmap,
            prob_thres,
            volume_thres,
            max_dets=100,
            conf_mode=conf_mode,
            prob_mode=prob_mode,
        )
        ret.append(keypoints)
    return ret


def crop_tomo_patch(tomo, patch_position):
    # start = time.time()
    tomo_shape = tomo.shape
    _roi_start, _roi_end, patch_start, patch_end = patch_position
    top_pad_z, top_pad_y, top_pad_x = [max(-start, 0) for start in patch_start]
    bot_pad_z, bot_pad_y, bot_pad_x = [
        max(0, end - size) for end, size in zip(patch_end, tomo_shape)
    ]
    actual_crop_start = [max(0, start) for start in patch_start]
    actual_crop_end = [min(end, size) for end, size in zip(patch_end, tomo_shape)]
    crop_slices = tuple(
        slice(start, end) for start, end in zip(actual_crop_start, actual_crop_end)
    )
    crop = tomo[crop_slices]
    # pad if needed
    pad = (top_pad_x, bot_pad_x, top_pad_y, bot_pad_y, top_pad_z, bot_pad_z)
    if any(pad):
        crop = F.pad(crop, pad, mode="constant", value=0)
    # end = time.time()
    # logger.debug("Crop take %f sec: %s", end - start, patch_position)
    return crop


@torch.inference_mode()
def spacing_torch(
    ori_tomo, ori_spacing, target_spacing, device="cuda", mode="trilinear"
):
    if tuple(ori_spacing) == tuple(target_spacing):
        return ori_tomo
    # NOTE: https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d
    # shrink -> AREA, enlarge -> CUBIC or LINEAR
    assert mode in ["trilinear", "nearest", "area", "nearest-exact"]
    assert len(ori_tomo.shape) == 3
    ori_tomo = ori_tomo[None, None].to(device)  # 11ZYX
    if mode != "nearest":
        ori_tomo = ori_tomo.float()

    scale_factor = tuple(ori / tgt for ori, tgt in zip(ori_spacing, target_spacing))
    ##### ISSUES #####
    ### with interpolation mode `nearest`
    # RuntimeError: upsample_nearest3d only supports output tensors with less than INT_MAX elements, but got [1, 1, 1178, 1367, 1367]
    # ref: https://github.com/pytorch/pytorch/issues/144855
    ### with interpolation mode `trilinear` on GPU
    # RuntimeError: CUDA error: invalid configuration argument
    ##### CURRENT SIMPLE FIX: if CUDA-interpolation-kernel fails, use CPU instead
    try:
        start = time.time()
        spaced_tomo = F.interpolate(
            ori_tomo,
            size=None,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=False if mode == "trilinear" else None,
            recompute_scale_factor=False,
        )[0, 0]
        end = time.time()
        logger.debug("INTERPOLATE WITH MODE=%s TAKE %.2f sec", mode, end - start)
    except RuntimeError as e:
        logger.warning(
            "EXCEPTION with torch.interpolate():\n%s\nAttempt CPU interpolation..", e
        )
        spaced_tomo = F.interpolate(
            ori_tomo.cpu(),
            size=None,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=False if mode == "trilinear" else None,
            recompute_scale_factor=False,
        )[0, 0]
    if mode != "nearest":
        spaced_tomo = spaced_tomo.to(torch.uint8)  # ZYX
    spaced_tomo = spaced_tomo.cpu()
    assert spaced_tomo.dtype == torch.uint8
    return spaced_tomo


def scale_range(start, end, scale, minv, maxv):
    n = round((end - start) * scale)
    if round(end * scale) - round(start * scale) == n:
        return round(start * scale), round(end * scale)
    start1 = int(start * scale)
    start2 = round(start * scale)
    candidate_starts = []
    if minv <= start1 < start1 + n <= maxv:
        candidate_starts.append(start1)
    if minv <= start2 < start2 + n <= maxv:
        candidate_starts.append(start2)
    if len(candidate_starts) == 1:
        return candidate_starts[0], candidate_starts[0] + n
    elif len(candidate_starts) == 2:
        if candidate_starts[0] == candidate_starts[1]:
            return candidate_starts[0], candidate_starts[0] + n
        diffs = [
            abs(s - start * scale) + abs(s + n - end * scale) for s in candidate_starts
        ]
        if diffs[0] < diffs[1]:
            return candidate_starts[0], candidate_starts[0] + n
        else:
            return candidate_starts[1], candidate_starts[1] + n
    else:
        raise ValueError


@exception_printer
def tomo_producer_worker_func(
    in_queue: mp.Queue,
    out_queue: queue.Queue,
    num_remain_workers: mp.Value,
    lock: mp.Lock,
    device: str = "cpu",
    mode: str = "trilinear",
    worker_id: str = "",
):
    """Read tomo, spacing, then save to shared memory.
    Should be run at process level (multiprocessing)
    """
    logger.info("[TOMO PRODUCER %s] STARTING..", worker_id)
    _old_num_threads = torch.get_num_threads()
    if _old_num_threads < 16:
        torch.set_num_threads(16)
    logger.info(
        "[TOMO PRODUCER %s] Number of threads change from %d to %d",
        worker_id,
        _old_num_threads,
        torch.get_num_threads(),
    )
    tomo_loader = MultithreadOpencvTomogramLoader(
        num_workers=_CV2_TOMO_LOADER_NUM_THREADS
    )
    while True:
        item = in_queue.get()
        if item is None:
            with lock:
                num_remain_workers.value -= 1
                if num_remain_workers.value == 0:
                    # all other workers are stopped, this worker is the last one alive
                    # put None to indicate stopping to out_queue's consumer
                    out_queue.put(None)
            # other workers should stop too..
            in_queue.put(None)
            logger.debug(
                "[TOMO PRODUCER %s] STOPPING..",
                worker_id,
            )
            return
        start = time.time()
        tomo_idx, tomo_id, ori_spacing = item
        # fetch new item
        tomo = tomo_loader.load(os.path.join(DATA_DIR, tomo_id))
        ori_shape = tomo.shape
        target_spacings = TARGET_SPACINGS
        tomo = torch.from_numpy(tomo)
        assert tomo.dtype == torch.uint8

        # FIX BUG WRONG QUANTILE
        hist = torch.bincount(tomo.view(-1), minlength=256).cpu().numpy()
        assert hist.shape[0] == 256
        CUTOFF = 25
        is_wrong_quantile_dark = hist[:CUTOFF].sum() / hist[CUTOFF:].sum() > 0.08
        is_wrong_quantile_light = hist[-CUTOFF:].sum() / hist[:-CUTOFF].sum() > 0.08
        if is_wrong_quantile_dark or is_wrong_quantile_light:
            tomo = tomo + 127

        # Multiscale TTA
        logger.debug(
            "[TOMO PRODUCER %s] Read tomo_idx=%d tomo_id=%s tomo_shape=%s",
            worker_id,
            tomo_idx,
            tomo_id,
            tomo.shape,
        )
        proxy_tomos = []
        for spacing_idx, target_spacing in enumerate(target_spacings):
            spaced_tomo = spacing_torch(
                tomo, ori_spacing, target_spacing, device=device, mode=mode
            )
            gc.collect()
            logger.debug(
                "[TOMO PRODUCER %s] Spacing tomo_idx=%d tomo_id=%s shape %s -> %s",
                worker_id,
                tomo_idx,
                tomo_id,
                tomo.shape,
                spaced_tomo.shape,
            )
            proxy_tomos.append(
                ShmTensor.from_tensor(
                    spaced_tomo, f"{tomo_idx}_{tomo_id}_{spacing_idx}"
                )
            )
        del tomo
        gc.collect()
        end = time.time()
        logger.info(
            "[TOMO PRODUCER %s] Done tomo_idx=%d tomo_id=%s spacings %s -> %s with shape %s -> %s take %.2f sec",
            worker_id,
            tomo_idx,
            tomo_id,
            ori_spacing,
            target_spacings,
            ori_shape,
            tuple(e.shape for e in proxy_tomos),
            end - start,
        )
        out_queue.put((tomo_idx, tomo_id, ori_spacing, target_spacings, proxy_tomos))


@exception_printer
def patches_producer_worker_func(
    in_queue: mp.Queue,
    out_queue: queue.Queue,
    num_remain_workers: mp.Value,
    lock: threading.Lock,
    patch_size,
    border,
    overlap,
    worker_id="",
):
    """
    Should be lightweight, and expected to be run in thread level.

    Args:
        in_queue: Input queue contain item of (tomo_idx, tomo_id, ori_spacing, target_spacing, proxy_tomo)
        out_queue: Each item in output queue contains multiple patches belong to a same tomo
    """
    logger.info("[PATCHES PRODUCER %s] STARTING..", worker_id)
    while True:
        item = in_queue.get()
        if item is None:
            with lock:
                num_remain_workers.value -= 1
                if num_remain_workers.value == 0:
                    # put final None so out_queue's consumer should stop too
                    out_queue.put(None)
            # put None back to in_queue so other in_queue's consumer workers will stop too
            in_queue.put(None)
            logger.info(
                "\n\n[PATCHES PRODUCER %s] STOPPING..\n\n",
                worker_id,
            )
            return
        tomo_idx, tomo_id, ori_spacing, target_spacings, proxy_tomos = item
        start = time.time()
        assert len(proxy_tomos) == len(target_spacings) == len(TARGET_SPACINGS)

        items = []
        total_patches = 0
        cur_patch_idx = -1
        shm_tomos = []
        for target_spacing, proxy_tomo in zip(target_spacings, proxy_tomos):
            shm_tomo, tomo = proxy_tomo.to_tensor()
            tomo_shape = tomo.shape
            shm_tomos.append(shm_tomo)
            patch_positions = get_sliding_patch_positions(
                img_size=tomo.shape,
                patch_size=patch_size,
                border=border,
                overlap=overlap,
                validate=False,
            )
            # patch_positions = torch.from_numpy(patch_positions)
            patch_positions = patch_positions.tolist()
            total_patches += len(patch_positions)

            for patch_pos in patch_positions:
                # crop tomo patch
                crop = crop_tomo_patch(tomo, patch_pos)
                crop = crop[None]  # 1ZYX, uint8
                cur_patch_idx += 1
                item = {
                    "tomo_idx": torch.tensor(tomo_idx, dtype=torch.int16),
                    "patch_pos": torch.tensor(patch_pos, dtype=torch.float32),
                    "tomo_shape": torch.tensor(tomo_shape, dtype=torch.int16),
                    "ori_spacing": torch.tensor(ori_spacing, dtype=torch.float32),
                    "target_spacing": torch.tensor(target_spacing, dtype=torch.float32),
                    "cur_patch_idx": torch.tensor(cur_patch_idx, dtype=torch.int16),
                    "image": crop,
                }
                items.append(item)
        assert cur_patch_idx == total_patches - 1
        for item in items:
            item["total_patches"] = torch.tensor(total_patches, dtype=torch.int16)

        end = time.time()
        logger.debug(
            "[PATCHES PRODUCER %s] Done tomo_idx=%d tomo_id=%s take %.2f sec",
            worker_id,
            tomo_idx,
            tomo_id,
            end - start,
        )
        out_queue.put((tomo_idx, shm_tomos, items))


@exception_printer
def batch_producer_worker_func(
    in_queue: queue.Queue,
    out_queue: queue.Queue,
    batch_size,
    device="cpu",
):
    """Should be lightweight, and expected to be run in thread level."""
    count = 0
    items = []
    should_continue = True
    shm_dict = {}
    last_tomo_idx = -1
    while should_continue:
        item = in_queue.get()
        if item is None:
            # put None so in_queue's consumer should notice and stop too
            in_queue.put(None)
            should_continue = False
        else:
            tomo_idx, shm_tomos, new_items = item
            # @TODO - remove this line
            assert all([tomo_idx == e["tomo_idx"].item() for e in new_items])
            items.extend(new_items)
            shm_dict[tomo_idx] = shm_tomos

        while True:
            if len(items) == 0 or (len(items) < batch_size and should_continue):
                break
            start = time.time()
            cur_bs = min(batch_size, len(items))
            batch = items[:cur_bs]
            batch = default_collate(batch)
            if device != "cpu":
                batch = {k: v.to(device) for k, v in batch.items()}
            assert batch["image"].dtype == torch.uint8
            end = time.time()
            logger.debug(
                "[BATCH PRODUCER] new batch %d (tomo %d) take %.4f",
                count,
                tomo_idx,
                end - start,
            )
            out_queue.put(batch)
            count += 1
            last_tomo_idx = batch["tomo_idx"][-1].item()
            items = items[cur_bs:]
        for shm_tomo_idx in list(shm_dict.keys()):
            # un-collated batch should start with tomo_idx >= last_tomo_idx
            # so we're safe to unlink previous one
            # since default_collate() copy to a newly allocated tensor
            if shm_tomo_idx < last_tomo_idx:
                for shm in shm_dict[shm_tomo_idx]:
                    shm.unlink()
                del shm_dict[shm_tomo_idx]

    for shms in shm_dict.values():
        for shm in shms:
            shm.unlink()
    del shm_dict
    gc.collect()
    # put a final None to indicate `all done`, stop inference worker too..
    out_queue.put(None)
    logger.info("[BATCH PRODUCER] Stopped since tomo_idx=%d", tomo_idx)
    assert len(items) == 0


@exception_printer
def inference_worker(worker_id, input_queue, chunk_idx, gpu_id, model_cfgs):
    logger.info(
        "STARTING INFERENCE WORKER %d with device cuda:%d, current input queue size %d, %d models with configs:\n%s",
        worker_id,
        gpu_id,
        input_queue.qsize(),
        len(model_cfgs),
        model_cfgs,
    )
    device = torch.device(f"cuda:{gpu_id}")
    patch_size = PATCH_SIZE
    border = PATCH_BORDER
    overlap = PATCH_OVERLAP

    ############ START ALL WORKERS ###########
    ##########################################
    tomo_queue = mp.Queue(maxsize=TOMO_PRODUCER_PREFETCH)
    tomo_producer_num_remain_workers = mp.Value("i", TOMO_PRODUCER_NUM_WORKERS)
    tomo_producer_lock = mp.Lock()
    patches_queue = queue.Queue(maxsize=PATCHES_PRODUCER_PREFETCH)
    patches_producer_num_remain_workers = mp.Value("i", PATCHES_PRODUCER_NUM_WORKERS)
    patches_producer_lock = threading.Lock()
    batch_queue = queue.Queue(maxsize=DATALOADER_PREFETCH)

    # PRODUCE SPACED TOMO
    tomo_producer_workers = []
    for _worker_id in range(TOMO_PRODUCER_NUM_WORKERS):
        worker = mp.Process(
            group=None,
            target=tomo_producer_worker_func,
            name=f"tomo_producer_{worker_id}.{_worker_id}",
            args=(
                input_queue,
                tomo_queue,
                tomo_producer_num_remain_workers,
                tomo_producer_lock,
            ),
            kwargs={
                "device": TOMO_SPACING_DEVICE,
                "mode": TOMO_SPACING_MODE,
                "worker_id": f"{worker_id}.{_worker_id}",
            },
        )
        worker.start()
        tomo_producer_workers.append(worker)

    # PRODUCER TOMO PATCHES
    patches_producer_workers = []
    for _worker_id in range(PATCHES_PRODUCER_NUM_WORKERS):
        worker = threading.Thread(
            group=None,
            target=patches_producer_worker_func,
            name=f"patches_producer_{worker_id}.{_worker_id}",
            args=(
                tomo_queue,
                patches_queue,
                patches_producer_num_remain_workers,
                patches_producer_lock,
                patch_size,
                border,
                overlap,
            ),
            kwargs={"worker_id": f"{worker_id}.{_worker_id}"},
        )
        worker.start()
        patches_producer_workers.append(worker)

    # BATCH PRODUCER
    batch_producer_worker = threading.Thread(
        group=None,
        target=batch_producer_worker_func,
        name="batch_producer",
        args=(patches_queue, batch_queue, BATCH_SIZE),
        kwargs={"device": "cuda:0"},
    )
    batch_producer_worker.start()

    ############## LOAD MODEL ##################
    ############################################
    logger.info("LOADING MODEL..")
    models = []
    unique_tta_names = set()
    unique_agg_names = set()
    sessions = []
    for session_idx, model_cfg in enumerate(model_cfgs):
        model_tta_names = model_cfg["tta"]
        model_agg_weights = model_cfg["weight"]
        model_agg_names = model_cfg["agg_name"]
        assert len(model_tta_names) == len(model_agg_weights) == len(model_agg_names)
        assert len(model_tta_names) == len(set(model_tta_names))
        model = Model3dWrapper(
            model_cfg["sig"],
            gpu_id,
            act="identity" if HEATMAP_AGG_LOGITS else "sigmoid",
        )
        model.eval().to(device)
        models.append(model)
        unique_tta_names.update(model_tta_names)
        unique_agg_names.update(model_agg_names)
        logger.info(
            "Loaded model %d with TTA=%s, WEIGHTS=%s",
            session_idx,
            model_tta_names,
            model_agg_weights,
        )
        for tta_name, agg_weight, agg_name in zip(
            model_tta_names, model_agg_weights, model_agg_names
        ):
            assert tta_name in ALL_SUPPORTED_TTAS
            sessions.append((model, tta_name, agg_weight, agg_name))
    unique_tta_names = list(unique_tta_names)
    unique_agg_names = list(unique_agg_names)
    num_sessions = len(sessions)
    _session_agg_names = [e[-1] for e in sessions]
    agg_name_to_first_session_idx = {
        agg_name: _session_agg_names.index(agg_name) for agg_name in unique_agg_names
    }
    agg_name_to_last_session_idx = {
        agg_name: num_sessions - 1 - _session_agg_names[::-1].index(agg_name)
        for agg_name in unique_agg_names
    }
    logger.info(
        "FIRST/LAST SESSION IDX:\n%s\n%s",
        agg_name_to_first_session_idx,
        agg_name_to_last_session_idx,
    )

    logger.info("-----------\nLOADED %d MODELS:\n%s------------", len(models), models)
    logger.info(
        "TOTALLY ENABLE %d MODELS, %d TTA %s, %d AGG_NAMES %s => %d SESSIONS",
        len(models),
        len(unique_tta_names),
        unique_tta_names,
        len(unique_agg_names),
        unique_agg_names,
        len(sessions),
    )
    unique_tta_tfs = {
        tta_name: build_tta(tta_name, 1.0, ori_dims="zyx")
        for tta_name in unique_tta_names
    }

    # Postprocess: heatmap blurring
    if DECODE_NMS_BLUR_SIGMA is not None:
        from yagm.transforms import monai_custom as CT

        sigma_voxel = [
            1000 / e / HEATMAP_STRIDE * DECODE_NMS_BLUR_SIGMA
            for e in TARGET_SPACINGS[0]
        ]
        blur_operator = (
            CT.CustomGaussianFilter(
                spatial_dims=3,
                sigma=sigma_voxel,
                truncated=4,
                approx="erf",
                requires_grad=False,
            )
            .eval()
            .to(device)
        )
    else:
        blur_operator = None
    logger.info(
        "POST-PROCESSING BLUR OPERATOR:\n%s on device %s",
        blur_operator,
        getattr(blur_operator, "device", None),
    )

    cache_results = {
        agg_name: {
            "cur_pred_heatmap_sum": None,
            "cur_pred_heatmap_count": None,
            "cur_tomo_idx": None,
            "cur_tomo_id": None,
            "main_target_spacing": None,
            "submission": SubmissionDataFrame(),
        }
        for agg_name in unique_agg_names
    }
    logger.info("\n\n\nWORKER %d STARTING INFERENCE..\n", worker_id)
    infer_start = time.time()
    cur_batch_idx = 0
    pbar = tqdm(desc=f"===== INFERENCE {worker_id} =====")
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16
    ):
        while True:
            batch = batch_queue.get()
            if batch is None:
                break
            # print("GOT BATCH:", {(k, v.shape, v.dtype) for k, v in batch.items()})
            batch_image = batch["image"].to(device)
            assert batch_image.dtype == torch.uint8
            B = batch_image.shape[0]

            unique_tta_batch_images = {
                tta_name: tta_tf.transform(batch_image)[0].float().contiguous()
                for tta_name, tta_tf in unique_tta_tfs.items()
            }

            batch_outputs = []
            for model, tta_name, agg_weight, agg_name in sessions:
                tta_tf = unique_tta_tfs[tta_name]
                tta_batch_image = unique_tta_batch_images[tta_name]
                tta_batch_heatmap = model(tta_batch_image)
                batch_heatmap = tta_tf.invert(tta_batch_heatmap)[0]
                batch_outputs.append((batch_heatmap, agg_weight, agg_name))

            assert len(batch_outputs) == num_sessions

            for i in range(B):
                for session_idx, (batch_heatmap, agg_weight, agg_name) in enumerate(
                    batch_outputs
                ):
                    pred_patch_heatmap = batch_heatmap[i]
                    C = pred_patch_heatmap.shape[0]
                    assert C == 1

                    tomo_idx = batch["tomo_idx"][i]
                    tomo_id = ALL_TOMO_IDS[tomo_idx]
                    tomo_shape = batch["tomo_shape"][i].tolist()
                    ori_spacing = batch["ori_spacing"][i].tolist()
                    target_spacing = batch["target_spacing"][i].tolist()
                    cur_patch_idx = int(batch["cur_patch_idx"][i].item())
                    total_patches = int(batch["total_patches"][i].item())
                    patch_position = batch["patch_pos"][i].cpu()

                    is_first = (cur_patch_idx == 0) and (
                        session_idx == agg_name_to_first_session_idx[agg_name]
                    )
                    is_last = (cur_patch_idx == total_patches - 1) and (
                        session_idx == agg_name_to_last_session_idx[agg_name]
                    )

                    # load cache for this tomo+agg combination
                    agg_result = cache_results[agg_name]
                    ### start IS_FIRST
                    if is_first:
                        logger.info(
                            "[INFER %d] Model %s receive first patch of tomo %s (index %d)",
                            worker_id,
                            session_idx,
                            tomo_id,
                            tomo_idx,
                        )
                        assert all(
                            [
                                agg_result[k] is None
                                for k in [
                                    "cur_tomo_idx",
                                    "cur_tomo_id",
                                    "cur_pred_heatmap_sum",
                                    "cur_pred_heatmap_count",
                                    "main_target_spacing",
                                ]
                            ]
                        )

                        # allocate new heatmap_sum and heatmap_count
                        agg_result["cur_tomo_idx"] = tomo_idx
                        agg_result["cur_tomo_id"] = tomo_id
                        shared_agg_heatmap_shape = [
                            round(e / HEATMAP_STRIDE) for e in tomo_shape
                        ]
                        # float16 to save memory
                        agg_result["cur_pred_heatmap_sum"] = torch.zeros(
                            (C, *shared_agg_heatmap_shape),
                            dtype=torch.float16,
                            device=pred_patch_heatmap.device,
                        )
                        agg_result["cur_pred_heatmap_count"] = torch.zeros(
                            (C, *shared_agg_heatmap_shape),
                            dtype=torch.float16,
                            device=pred_patch_heatmap.device,
                        )
                        agg_result["main_target_spacing"] = target_spacing
                        assert all(
                            [
                                abs(a - b) < 1e-4
                                for a, b in zip(target_spacing, TARGET_SPACINGS[0])
                            ]
                        )
                    ### end IS_FIRST

                    assert tomo_idx == agg_result["cur_tomo_idx"]
                    assert all(
                        [
                            a <= b
                            for a, b in zip(
                                agg_result["main_target_spacing"], target_spacing
                            )
                        ]
                    )
                    cur_pred_heatmap_sum = agg_result["cur_pred_heatmap_sum"]
                    cur_pred_heatmap_count = agg_result["cur_pred_heatmap_count"]
                    main_target_spacing = agg_result["main_target_spacing"]
                    cur_submission: SubmissionDataFrame = agg_result["submission"]
                    shared_agg_heatmap_shape = cur_pred_heatmap_sum.shape[1:]

                    scale_zyx = [
                        a / b for a, b in zip(shared_agg_heatmap_shape, tomo_shape)
                    ]
                    scale_z, scale_y, scale_x = scale_zyx

                    # interpolate to shared coordinate space
                    C, Z2, Y2, X2 = pred_patch_heatmap.shape
                    assert C == 1
                    patch_heatmap_shape = [
                        round(e * s) for e, s in zip(patch_size, scale_zyx)
                    ]
                    if [Z2, Y2, X2] != patch_heatmap_shape:
                        logger.debug(
                            "PATCH HEATMAP INTERPOLATE: %s --> %s",
                            (Z2, Y2, X2),
                            patch_heatmap_shape,
                        )
                        pred_patch_heatmap = F.interpolate(
                            pred_patch_heatmap[None],  # (1, C, Z2, Y2, X2)
                            size=patch_heatmap_shape,
                            mode=HEATMAP_INTERPOLATION_MODE,
                            align_corners=False,
                        )[
                            0
                        ]  # (C, Z3, Y3, X3)
                    else:
                        logger.debug("PATCH HEATMAP SAME: %s", pred_patch_heatmap.shape)
                        pass

                    roi_start, roi_end, patch_start, patch_end = patch_position.tolist()
                    top_pad_z, top_pad_y, top_pad_x = [
                        max(-start, 0) for start in roi_start
                    ]
                    bot_pad_z, bot_pad_y, bot_pad_x = [
                        max(0, end - size) for end, size in zip(roi_end, tomo_shape)
                    ]
                    assert (
                        tuple(
                            round((pe - ps) * s)
                            for pe, ps, s in zip(patch_end, patch_start, scale_zyx)
                        )
                        == pred_patch_heatmap.shape[1:]
                    )
                    sz, sy, sx = tuple(
                        rs - ps for rs, ps in zip(roi_start, patch_start)
                    )
                    ez, ey, ex = tuple(
                        ps - ce + pe
                        for ps, ce, pe in zip(patch_size, patch_end, roi_end)
                    )

                    assert (roi_end[0] - bot_pad_z) - (roi_start[0] + top_pad_z) == (
                        ez - bot_pad_z
                    ) - (sz + top_pad_z)
                    assert (roi_end[1] - bot_pad_y) - (roi_start[1] + top_pad_y) == (
                        ey - bot_pad_y
                    ) - (sy + top_pad_y)
                    assert (roi_end[2] - bot_pad_x) - (roi_start[2] + top_pad_x) == (
                        ex - bot_pad_x
                    ) - (sx + top_pad_x)

                    dst_shape = shared_agg_heatmap_shape
                    dst_slices = [
                        slice(None),
                        slice(
                            *scale_range(
                                roi_start[0] + top_pad_z,
                                roi_end[0] - bot_pad_z,
                                scale_z,
                                0,
                                dst_shape[0],
                            )
                        ),
                        slice(
                            *scale_range(
                                roi_start[1] + top_pad_y,
                                roi_end[1] - bot_pad_y,
                                scale_y,
                                0,
                                dst_shape[1],
                            )
                        ),
                        slice(
                            *scale_range(
                                roi_start[2] + top_pad_x,
                                roi_end[2] - bot_pad_x,
                                scale_x,
                                0,
                                dst_shape[2],
                            )
                        ),
                    ]
                    src_shape = pred_patch_heatmap.shape[1:]
                    src_slices = [
                        slice(None),
                        slice(
                            *scale_range(
                                sz + top_pad_z, ez - bot_pad_z, scale_z, 0, src_shape[0]
                            )
                        ),
                        slice(
                            *scale_range(
                                sy + top_pad_y, ey - bot_pad_y, scale_y, 0, src_shape[1]
                            )
                        ),
                        slice(
                            *scale_range(
                                sx + top_pad_x, ex - bot_pad_x, scale_x, 0, src_shape[2]
                            )
                        ),
                    ]

                    if HEATMAP_AGG_MODE == "avg":
                        cur_pred_heatmap_sum[dst_slices] += (
                            pred_patch_heatmap[src_slices] * agg_weight
                        )
                        cur_pred_heatmap_count[dst_slices] += agg_weight
                    elif HEATMAP_AGG_MODE == "max":
                        torch.maximum(
                            cur_pred_heatmap_sum[dst_slices],
                            pred_patch_heatmap[src_slices],
                            out=cur_pred_heatmap_sum[dst_slices],
                        )
                    else:
                        raise ValueError

                    if is_last:
                        # DECODE HEATMAP TO COORDINATE
                        tomo_id = agg_result["cur_tomo_id"]
                        logger.info(
                            "[INFER %d] Receive last patch of tomo %s (index %d), decoding..",
                            worker_id,
                            tomo_id,
                            tomo_idx,
                        )

                        if HEATMAP_AGG_MODE == "avg":
                            # ensure prediction cover all the tomogram
                            # assert torch.all(cur_pred_heatmap_count > 0)
                            heatmap = torch.div(
                                cur_pred_heatmap_sum,
                                cur_pred_heatmap_count,
                                out=cur_pred_heatmap_sum,
                            )
                        elif HEATMAP_AGG_MODE == "max":
                            heatmap = cur_pred_heatmap_sum
                        else:
                            raise ValueError
                        if HEATMAP_AGG_LOGITS:
                            heatmap = F.sigmoid(heatmap)

                        if SAVE_HEATMAP_NPY:
                            heatmap_npy = heatmap.half().cpu().numpy()
                            assert heatmap_npy.dtype == np.float16
                            save_heatmap_path = os.path.join(
                                SAVE_HEATMAP_NPY_DIR, agg_name, f"{tomo_id}.npy"
                            )
                            os.makedirs(
                                os.path.dirname(save_heatmap_path), exist_ok=True
                            )
                            np.save(save_heatmap_path, heatmap_npy)

                        if DECODE_HEATMAP_TO_CSV:
                            # Decode heatmap
                            _start = time.time()
                            if DECODE_METHOD == "nms":
                                outputs = decode_heatmap(
                                    heatmap,
                                    main_target_spacing,
                                    HEATMAP_STRIDE,
                                    blur_operator=blur_operator,
                                )
                            elif DECODE_METHOD == "cc3d":
                                outputs = decode_segment_mask(
                                    heatmap,
                                    main_target_spacing,
                                    HEATMAP_STRIDE,
                                    prob_thres=DECODE_CC3D_CONF_THRES,
                                    radius_factor_thres=DECODE_CC3D_RADIUS_FACTOR,
                                    conf_mode=DECODE_CC3D_CONF_MODE,
                                    prob_mode=DECODE_CC3D_PROB_MODE,
                                )
                            else:
                                raise ValueError
                            _end = time.time()
                            logger.debug(
                                "Decode heatmap of shape %s take %.4f sec",
                                heatmap.shape,
                                _end - _start,
                            )
                            # Convert to submission coordinates
                            assert len(outputs) == 1
                            main_target_spacing_tensor = torch.tensor(
                                [main_target_spacing], dtype=torch.float32
                            )  # (1, 3)
                            ori_spacing_tensor = torch.tensor(
                                [ori_spacing], dtype=torch.float32
                            )  # (1, 3)
                            for _channel_idx, keypoints in enumerate(outputs):
                                # back to original coordinate space
                                keypoints[:, :3] = (keypoints[:, :3] + 0.5) * (
                                    HEATMAP_STRIDE
                                    * main_target_spacing_tensor
                                    / ori_spacing_tensor
                                ) - 0.5
                                keypoints = keypoints.tolist()

                                ###### VISUALIZATION ######
                                if VIZ_ENABLE and MODE != "KAGGLE":
                                    tomo_dir = os.path.join(DATA_DIR, tomo_id)
                                    row = GT_DF[GT_DF["tomo_id"] == tomo_id].iloc[0]
                                    ori_shape = (row["Z"], row["Y"], row["X"])
                                    gt_zyx = [
                                        row["motor_z"],
                                        row["motor_y"],
                                        row["motor_x"],
                                    ]
                                    viz = viz_byu(
                                        tomo_dir,
                                        ori_spacing,
                                        ori_shape,
                                        gt_zyx,
                                        heatmap[0],
                                        keypoints,
                                    )
                                    # get the kind/judge of current prediction, one of TP, TN, FP, FN
                                    if (
                                        len(keypoints) > 0
                                        and keypoints[0][3] >= DECODE_CONF_THRES
                                    ):
                                        # predicted as positive
                                        pred_zyx = keypoints[0][:3]
                                        if gt_zyx[0] == -1:
                                            # GT is negative
                                            judge = "FP"
                                        else:
                                            dist = (
                                                (gt_zyx[0] - pred_zyx[0]) ** 2
                                                + (gt_zyx[1] - pred_zyx[1]) ** 2
                                                + (gt_zyx[2] - pred_zyx[2]) ** 2
                                            ) ** 0.5
                                            if dist <= 1000 / ori_spacing:
                                                judge = "TP"
                                            else:
                                                judge = "FN"
                                    else:
                                        # predicted as negative
                                        if gt_zyx[0] == -1:
                                            # GT is negative
                                            judge = "TN"
                                        else:
                                            judge = "FN"

                                    save_name = f"{judge}-{tomo_idx}-{tomo_id}-{'x'.join([str(int(e)) for e in ori_shape])}-spacing{ori_spacing}.jpg"
                                    save_path = os.path.join(
                                        TMP_VIZ_DIR, judge, agg_name, save_name
                                    )
                                    os.makedirs(
                                        os.path.dirname(save_path), exist_ok=True
                                    )
                                    cv2.imwrite(
                                        save_path, cv2.cvtColor(viz, cv2.COLOR_RGB2BGR)
                                    )

                                if len(keypoints) > 0:
                                    for kpt in keypoints:
                                        z, y, x, conf = kpt
                                        cur_submission.add_row(tomo_id, x, y, z, conf)
                                else:
                                    # no motor detected
                                    cur_submission.add_row(tomo_id, -1, -1, -1, 0.0)

                        # clear the cache
                        for k in [
                            "cur_tomo_idx",
                            "cur_tomo_id",
                            "cur_pred_heatmap_sum",
                            "cur_pred_heatmap_count",
                            "main_target_spacing",
                        ]:
                            agg_result[k] = None
                        gc.collect()
                        # torch.cuda.empty_cache()
            cur_batch_idx += 1
            pbar.update(1)
    pbar.close()
    infer_end = time.time()

    logger.info(
        "\n\n\n INFER WORKER %d INFER ON %d BATCHES TAKE %.4f sec",
        worker_id,
        cur_batch_idx,
        infer_end - infer_start,
    )

    assert len(cache_results) == len(unique_agg_names)
    if DECODE_HEATMAP_TO_CSV:
        for agg_name, agg_result in cache_results.items():
            submision = agg_result["submission"]
            df = submision.to_pandas(submit=False)
            csv_path = os.path.join(TMP_CSV_DIR, f"{agg_name}_chunk{chunk_idx}.csv")
            df.to_csv(csv_path, index=False)

    # JOIN
    logger.info("Worker %d waiting for all worker to be finished..", worker_id)
    all_workers = [
        batch_producer_worker,
        *tomo_producer_workers,
        *patches_producer_workers,
    ]
    for worker in all_workers:
        worker.join()
    return None


########################### MAIN CODE #############################
###################################################################
logger.warning("CLEARING ALL CONTENTS IN: %s", [TMP_CSV_DIR, TMP_CSV_DIR, TMP_VIZ_DIR])
clear_dir_content("/dev/shm/")
# clear_dir_content(TMP_CSV_DIR)
clear_dir_content(TMP_VIZ_DIR)
if MODE == "LOCAL":
    clear_dir_content(WORKING_DIR)
os.makedirs(TMP_CSV_DIR, exist_ok=True)
os.makedirs(TMP_VIZ_DIR, exist_ok=True)


global_start = time.time()

all_unique_agg_names = set()
for job_idx, model_cfgs in enumerate(JOBS):
    # fill the input queue
    input_queue = mp.Queue()
    for tomo_idx, (tomo_id, ori_spacing) in enumerate(
        zip(ALL_TOMO_IDS, ALL_TOMO_SPACINGS)
    ):
        input_queue.put((tomo_idx, tomo_id, [ori_spacing] * 3))
    # None to indicate ending and workers should stop when receive this None
    input_queue.put(None)

    job_unique_agg_names = set()
    for model_cfg in model_cfgs:
        job_unique_agg_names.update(model_cfg["agg_name"])
    assert len(all_unique_agg_names.intersection(job_unique_agg_names)) == 0
    all_unique_agg_names.update(job_unique_agg_names)

    job_workers = []
    for chunk_idx, device_id in enumerate(DEVICES):
        # worker_id, input_queue, chunk_idx, gpu_id, model_cfgs
        worker = mp.Process(
            group=None,
            target=inference_worker,
            name=f"INFERENCE_WORKER_job={job_idx}_chunk={chunk_idx}",
            kwargs={
                "worker_id": chunk_idx,
                "input_queue": input_queue,
                "chunk_idx": chunk_idx,
                "gpu_id": device_id,
                "model_cfgs": model_cfgs,
            },
        )
        worker.start()
        job_workers.append(worker)

    for worker in job_workers:
        worker.join()

global_end = time.time()
logger.info(
    "\n\n\n>>>>>> FINISH ALL INFERENCE TASK WITHIN %.4f sec (%.4f min)",
    global_end - global_start,
    (global_end - global_start) / 60.0,
)
all_unique_agg_names = list(all_unique_agg_names)
print('ALL UNIQUE AGG NAMES:', all_unique_agg_names)


!python3 /kaggle/working/byu/submit_3d.py





%%writefile /kaggle/working/byu/submit_2d.py

import sys
sys.path.extend([
    '/kaggle/working/byu/yagm/src',
    '/kaggle/working/byu/src',
    '/kaggle/working/byu/',
    '/kaggle/working/byu/third_party/segmentation_models_pytorch_3d',
    '/kaggle/working/byu/third_party/slowfast',
    '/kaggle/working/byu/third_party/timm_3d'
])
print('SYS PATH:', sys.path, sep = '\n')


import warnings

warnings.simplefilter("ignore")
import gc
import json
import logging
import multiprocessing as mp
import os
import queue
import shutil
import sys
import threading
import time
from functools import partial

import cc3d
import cv2
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from torch.utils.data import default_collate
from tqdm import tqdm
from yagm.tasks.base_task import BaseTask
from yagm.transforms.keypoints.decode import decode_heatmap_3d, decode_segment_mask_3d
from yagm.transforms.sliding_window import get_sliding_patch_positions
from yagm.transforms.tta_3d import build_tta
from yagm.utils import hydra as hydra_utils
from yagm.utils import lightning as l_utils
from yagm.utils.concurrent import ShmTensor

# Setting up custom logging
from yagm.utils.logging import init_logging, setup_logging

from byu.data.io import MultithreadOpencvTomogramLoader
from byu.inference import zoo
from byu.inference.tensorrt_engine import ThreadsafeTRTEngine
from byu.utils.data import SubmissionDataFrame
from byu.utils.metrics import compute_metrics, kaggle_score
from byu.utils.viz import viz_byu
from byu.utils.wbf import weighted_boxes_fusion_3d

try:
    hydra_utils.init_hydra()
except:
    print("SKIP RE-INIT HYDRA")

### SETUP LOGGER FOR IPYTHON ###
# ref: https://github.com/ipython/ipykernel/issues/111
# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Create STDERR handler
handler = logging.StreamHandler(sys.stderr)
# Create formatter and add it to the handler
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
# Set STDERR handler as the only handler
logger.handlers = [handler]


##################################################
###### MODEL SIGNATURES OVERRIDES
##################################################
ALL_SUPPORTED_TTAS = ["yx", "yx_x", "yx_y", "yx_xy", "xy", "xy_x", "xy_y", "xy_xy"]

_SIG_TEMPLATE = {
    "backend": "torch",  # torch | trt
    "trt_path": "",
    "config_path": "",
    "torch_path": "",
    "ema": 0.99,
}


# MAXVIT, LB 85.4
# https://www.kaggle.com/code/dangnh0611/submit-heatmap-final?scriptVersionId=243128311
EXP27_2D_MAXVIT_TINY_ALLGTV3 = {
    "backend": "torch",  # torch | trt
    "trt_path": "",
    "config_path": "/kaggle/input/byu-final-dataset/EXP27_2D_MAXVIT_TINY_ALLGTV3_config.yaml",
    "torch_path": "/kaggle/input/byu-final-dataset/EXP27_2D_MAXVIT_TINY_ALLGTV3_ep5_step20000.ckpt",
    "ema": 0.99,
}


# COAT, LB 84.8
# https://www.kaggle.com/code/dangnh0611/submit-heatmap-final?scriptVersionId=242969781
EXP19_2D_COATLITEMEDIUM_ALLGTV2 = {
    "backend": "torch",  # torch | trt
    "trt_path": "",
    "config_path": "/kaggle/input/byu-final-dataset/EXP19_2D_COATLITEMEDIUM_ALLGTV2_config.yaml",
    "torch_path": "/kaggle/input/byu-final-dataset/EXP19_2D_COATLITEMEDIUM_ALLGTV2_ep2_step10000_val_Fbeta0.949087_val_PAP0.935705.ckpt",
    "ema": 0.99,
}


JOBS = [
    [
        {
            "sig": EXP27_2D_MAXVIT_TINY_ALLGTV3,
            "tta": ["yx", "xy"],
            "weight": [1.0, 1.0],
            "agg_name": ["MAXVIT-yx", "MAXVIT-xy"],
        }
    ]
]


##################################################
###### GLOBAL VARS
##################################################
_CV2_TOMO_LOADER_NUM_THREADS = 8

PATCH_SIZE = [3, 896, 896]
PATCH_BORDER = [0, 0, 0]
PATCH_OVERLAP = [0, 0, 0]

IO_BACKEND = "cv2"
BASE_SPACING = 13.1  # avg 15.6
# TARGET_SPACINGS = [13.1, 16.0, 19.7]
TARGET_SPACINGS = [[32.0, 16.0, 16.0]]
TOMO_SPACING_DEVICE = "cpu"
TOMO_SPACING_MODE = "trilinear"

HEATMAP_AGG_MODE = "avg"
HEATMAP_AGG_LOGITS = True
HEATMAP_INTERPOLATION_MODE = "bilinear"
FORWARD_SAVE_MEMORY = False
BATCH_SIZE = 8
HEATMAP_STRIDE = [4, 8, 8]

# MASK CC3D DECODING
DECODE_CC3D_CONF_THRES = 0.05
DECODE_CC3D_RADIUS_FACTOR = 0.1
DECODE_CC3D_CONF_MODE = "prob"  # volume | prob | fuse
DECODE_CC3D_PROB_MODE = "center"  # center | mean | max
# HEATMAP NMS DECODING
DECODE_NMS_BLUR_SIGMA = None
###### MAIN DECODE PARAMS ######
DECODE_METHOD = "nms"  # nms, cc3d
DECODE_CONF_THRES = 0.01
QUANTILE_THRES = 0.55

SAVE_HEATMAP_NPY = False
DECODE_HEATMAP_TO_CSV = True
CSV_ENSEMBLE_MODE = "wbf"  # max | wbf
WBF_CONF_TYPE = "avg"  # avg | max
WBF_CONF_THRES = 0.2

VIZ_ENABLE = False


# automatic determine the MODE
if os.path.isdir("/kaggle/input/byu-locating-bacterial-flagellar-motors-2025"):
    MODE = "KAGGLE"
else:
    MODE = "LOCAL"
# manual override
# MODE = "KAGGLE_VAL"

# LOCAL DEVELOPMENT ENV
if MODE == "LOCAL":
    DEVICES = [0]
    ASSETS_DIR = "assets/"
    WORKING_DIR = "outputs/submit/working_2d/"

    DATA_DIR = "/home/dangnh36/datasets/.comp/byu/raw/train"
    VAL_GT_CSV_PATH = "/home/dangnh36/datasets/.comp/byu/processed/gt_v3.csv"
    with open(
        "/home/dangnh36/datasets/.comp/byu/processed/cv/v3/skf4_rd42.json", "r"
    ) as f:
        cv_meta = json.load(f)
        ALL_TOMO_IDS = cv_meta["folds"][0]["val"][:]  # fold 0

    # DATA_DIR = "/home/dangnh36/datasets/.comp/byu/processed/pseudo_test/tomograms"
    # VAL_GT_CSV_PATH = "/home/dangnh36/datasets/.comp/byu/processed/pseudo_test/gt.csv"
    # ALL_TOMO_IDS = sorted(os.listdir(DATA_DIR)) * 10

    GT_DF = pd.read_csv(VAL_GT_CSV_PATH)
    GT_DF = GT_DF[GT_DF["tomo_id"].isin(ALL_TOMO_IDS)].reset_index(drop=True)
    _tomo2spacing = {}
    for i, row in GT_DF.iterrows():
        _tomo2spacing[row["tomo_id"]] = row["voxel_spacing"]
    ALL_TOMO_SPACINGS = [_tomo2spacing[tomo_id] for tomo_id in ALL_TOMO_IDS]

    # TOMO_PRODUCER_NUM_WORKERS = 8
    # TOMO_PRODUCER_PREFETCH = 8
    # PATCHES_PRODUCER_NUM_WORKERS = 2
    # PATCHES_PRODUCER_PREFETCH = 8
    # DATALOADER_PREFETCH = 8

    TOMO_PRODUCER_NUM_WORKERS = 1
    TOMO_PRODUCER_PREFETCH = 1
    PATCHES_PRODUCER_NUM_WORKERS = 1
    PATCHES_PRODUCER_PREFETCH = 2
    DATALOADER_PREFETCH = 8

# VALIDATION ON KAGGLE
elif MODE == "KAGGLE_VAL":
    DEVICES = [0, 1]
    ASSETS_DIR = "/kaggle/input/byu-final-dataset/"
    DATA_DIR = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train"
    WORKING_DIR = "/kaggle/working/"
    VAL_GT_CSV_PATH = "/kaggle/input/byu-checkpoints/gt_v2.csv"
    with open("/kaggle/input/byu-checkpoints/skf4_rd42.json", "r") as f:
        cv_meta = json.load(f)
        ALL_TOMO_IDS = cv_meta["folds"][0]["val"][:50]  # fold 0, top 10 first tomo
    GT_DF = pd.read_csv(VAL_GT_CSV_PATH)
    GT_DF = GT_DF[GT_DF["tomo_id"].isin(ALL_TOMO_IDS)].reset_index(drop=True)
    _tomo2spacing = {}
    for i, row in GT_DF.iterrows():
        _tomo2spacing[row["tomo_id"]] = row["voxel_spacing"]
    ALL_TOMO_SPACINGS = [_tomo2spacing[tomo_id] for tomo_id in ALL_TOMO_IDS]

    TOMO_PRODUCER_NUM_WORKERS = 1
    TOMO_PRODUCER_PREFETCH = 1
    PATCHES_PRODUCER_NUM_WORKERS = 1
    PATCHES_PRODUCER_PREFETCH = 2
    DATALOADER_PREFETCH = 8

# PRIVATE TEST SUBMISSION
elif MODE == "KAGGLE":
    DEVICES = [0, 1]
    ASSETS_DIR = "/kaggle/input/byu-final-dataset/"
    DATA_DIR = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/test"
    WORKING_DIR = "/kaggle/working/"
    ALL_TOMO_IDS = sorted(os.listdir(DATA_DIR))
    ALL_TOMO_SPACINGS = [BASE_SPACING] * len(ALL_TOMO_IDS)

    TOMO_PRODUCER_NUM_WORKERS = 1
    TOMO_PRODUCER_PREFETCH = 1
    PATCHES_PRODUCER_NUM_WORKERS = 1
    PATCHES_PRODUCER_PREFETCH = 2
    DATALOADER_PREFETCH = 8

else:
    raise ValueError

TMP_CSV_DIR = os.path.join(WORKING_DIR, "tmp_csv")
TMP_VIZ_DIR = os.path.join(WORKING_DIR, "tmp_viz")
SAVE_HEATMAP_NPY_DIR = os.path.join(WORKING_DIR, "tmp_heatmap")

# RECHECK SOME CONDITIONS
assert len(ALL_TOMO_IDS) == len(ALL_TOMO_SPACINGS)
# assert not (SAVE_HEATMAP_NPY and DECODE_HEATMAP_TO_CSV) and (SAVE_HEATMAP_NPY or DECODE_HEATMAP_TO_CSV)
# @TODO - refactor to support arbitary HEATMAP_STRIDE
# assert HEATMAP_STRIDE[0] == 1, f"Currently, heatmap stride along Z must be 1"


###################################################
###### FUNCTIONS/CLASSES DEFINITION
###################################################
def exception_printer(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception("Exception %s in %s: %s", type(e), func.__name__, e)
            import traceback

            traceback.print_exc()
            raise e

    return wrapper


def clear_dir_content(folder_path):
    if os.path.isdir(folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)


class Model2dWrapper(nn.Module):
    def __init__(self, sig, gpu_id, act="sigmoid"):
        super().__init__()
        if sig["backend"] == "trt":
            logger.info("Loading TRT model with config:\n%s", sig)
            self.model = ThreadsafeTRTEngine(sig["trt_path"], gpu_id)
        elif sig["backend"] == "torch":
            cfg = OmegaConf.load(sig["config_path"])
            cfg.misc.log_model = False
            cfg.ema.val_decays = [sig["ema"] if sig["ema"] is not None else 0]
            cfg.ckpt.strict = False

            ######### OVERWRITE SOME CONFIGS ##########
            # OVERWRITE SOME CONFIG
            try:
                cfg.model.encoder.pretrained = False
            except:
                pass

            # for backward-compatibility
            OmegaConf.set_readonly(cfg, False)
            # just keep outputing heatmap
            cfg.model.reg_head.enable = False
            cfg.model.dsnt = OmegaConf.create()
            cfg.model.dsnt.enable = False
            OmegaConf.set_readonly(cfg, True)
            ##########

            task: BaseTask = l_utils.build_task(cfg)
            l_utils.load_lightning_state_dict(
                model=task,
                ckpt_path=sig["torch_path"],
                cfg=cfg,
            )
            logger.info("Loaded Pytorch state dict from %s", sig["torch_path"])
            # self.model = TorchModelWrapper(task.model)
            self.model = task.model
            del task
            gc.collect()
        else:
            raise ValueError

        self.act_name = act
        if act == "sigmoid":
            self.act = F.sigmoid
            self.nan_to_num = partial(torch.nan_to_num, nan=0.0)
        elif act == "identity":
            self.act = lambda x: x
            self.nan_to_num = partial(
                torch.nan_to_num, nan=torch.finfo(torch.float32).min
            )
        else:
            raise ValueError

    def forward(self, x):
        heatmap = self.act(self.model(x)[2])
        heatmap = self.nan_to_num(heatmap)
        # if torch.isnan(heatmap).any():
        #     print('\n\nNAN!!!')
        #     raise Exception
        return heatmap


def decode_heatmap(pred_heatmap, target_spacing, stride, blur_operator=None):
    radius_voxel = [1000.0 / e / s for e, s in zip(target_spacing, stride)]
    radius_thres = max(radius_voxel)
    # @TODO - currently, using fixed pool_ksize=3 due to performance issue
    # when using large kernel size with Pytorch (>10 sec on 224x448x448)

    # # maximum ood number which <= radius_thres
    # # 0.7071067811865475 = 1 / sqrt(2)
    # pool_ksize = int((2 * radius_thres * 0.7071067811865475 - 1) // 2 * 2 + 1)
    # # at least 3, [1,1,1] == no pooling, which increase maxRecall but significantly reduce other metrics
    # pool_ksize = max(3, pool_ksize)
    # pool_ksize = [pool_ksize, pool_ksize, pool_ksize]

    pool_ksize = [3, 3, 3]
    logger.debug(
        "Heatmap decode with radius=%s, pool_ksize=%s, radius_thres=%s",
        radius_voxel,
        pool_ksize,
        radius_thres,
    )
    del target_spacing
    assert pred_heatmap.shape[0] == 1
    ret = []
    for channel_idx, heatmap in enumerate(pred_heatmap):
        outputs = decode_heatmap_3d(
            heatmap=heatmap,
            pool_ksize=pool_ksize,
            nms_radius_thres=radius_thres,
            blur_operator=blur_operator,
            conf_thres=DECODE_CONF_THRES,
            # max_dets=5 if (VIZ_ENABLE and MODE != "KAGGLE") else 1,
            max_dets=100,
            timeout=None,
        )
        outputs = outputs.cpu()
        ret.append(outputs)
    return ret


def decode_segment_mask(
    pred_heatmap,
    target_spacing,
    stride,
    prob_thres,
    radius_factor_thres,
    conf_mode="fuse",
    prob_mode="avg",
):
    radius_voxel_thres = [
        1000 / e / s * radius_factor_thres for e, s in zip(target_spacing, stride)
    ]
    volume_thres = (
        4
        / 3
        * np.pi
        * radius_voxel_thres[0]
        * radius_voxel_thres[1]
        * radius_voxel_thres[2]
    )
    assert len(pred_heatmap.shape) == 4 and pred_heatmap.shape[0] == 1
    ret = []
    for channel_idx, heatmap in enumerate(pred_heatmap):
        keypoints = decode_segment_mask_3d(
            heatmap,
            prob_thres,
            volume_thres,
            max_dets=100,
            conf_mode=conf_mode,
            prob_mode=prob_mode,
        )
        ret.append(keypoints)
    return ret


def crop_tomo_patch(tomo, patch_position):
    # start = time.time()
    tomo_shape = tomo.shape
    _roi_start, _roi_end, patch_start, patch_end = patch_position
    top_pad_z, top_pad_y, top_pad_x = [max(-start, 0) for start in patch_start]
    bot_pad_z, bot_pad_y, bot_pad_x = [
        max(0, end - size) for end, size in zip(patch_end, tomo_shape)
    ]
    actual_crop_start = [max(0, start) for start in patch_start]
    actual_crop_end = [min(end, size) for end, size in zip(patch_end, tomo_shape)]
    crop_slices = tuple(
        slice(start, end) for start, end in zip(actual_crop_start, actual_crop_end)
    )
    crop = tomo[crop_slices]
    # pad if needed
    pad = (top_pad_x, bot_pad_x, top_pad_y, bot_pad_y, top_pad_z, bot_pad_z)
    if any(pad):
        crop = F.pad(crop, pad, mode="constant", value=0)
    # end = time.time()
    # logger.debug("Crop take %f sec: %s", end - start, patch_position)
    return crop


@torch.inference_mode()
def spacing_torch(
    ori_tomo, ori_spacing, target_spacing, device="cuda", mode="trilinear"
):
    if tuple(ori_spacing) == tuple(target_spacing):
        return ori_tomo
    # NOTE: https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d
    # shrink -> AREA, enlarge -> CUBIC or LINEAR
    assert mode in ["trilinear", "nearest", "area", "nearest-exact"]
    assert len(ori_tomo.shape) == 3
    ori_tomo = ori_tomo[None, None].to(device)  # 11ZYX
    if mode != "nearest":
        ori_tomo = ori_tomo.float()

    scale_factor = tuple(ori / tgt for ori, tgt in zip(ori_spacing, target_spacing))
    ##### ISSUES #####
    ### with interpolation mode `nearest`
    # RuntimeError: upsample_nearest3d only supports output tensors with less than INT_MAX elements, but got [1, 1, 1178, 1367, 1367]
    # ref: https://github.com/pytorch/pytorch/issues/144855
    ### with interpolation mode `trilinear` on GPU
    # RuntimeError: CUDA error: invalid configuration argument
    ##### CURRENT SIMPLE FIX: if CUDA-interpolation-kernel fails, use CPU instead
    try:
        start = time.time()
        spaced_tomo = F.interpolate(
            ori_tomo,
            size=None,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=False if mode == "trilinear" else None,
            recompute_scale_factor=False,
        )[0, 0]
        end = time.time()
        logger.debug("INTERPOLATE WITH MODE=%s TAKE %.2f sec", mode, end - start)
    except RuntimeError as e:
        logger.warning(
            "EXCEPTION with torch.interpolate():\n%s\nAttempt CPU interpolation..", e
        )
        spaced_tomo = F.interpolate(
            ori_tomo.cpu(),
            size=None,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=False if mode == "trilinear" else None,
            recompute_scale_factor=False,
        )[0, 0]
    if mode != "nearest":
        spaced_tomo = spaced_tomo.to(torch.uint8)  # ZYX
    spaced_tomo = spaced_tomo.cpu()
    assert spaced_tomo.dtype == torch.uint8
    return spaced_tomo


def scale_range(start, end, scale, minv, maxv):
    n = max(1, round((end - start) * scale))
    if round(end * scale) - round(start * scale) == n:
        return round(start * scale), round(end * scale)
    start1 = int(start * scale)
    start2 = round(start * scale)
    candidate_starts = []
    if minv <= start1 < start1 + n <= maxv:
        candidate_starts.append(start1)
    if minv <= start2 < start2 + n <= maxv:
        candidate_starts.append(start2)
    if len(candidate_starts) == 1:
        return candidate_starts[0], candidate_starts[0] + n
    elif len(candidate_starts) == 2:
        if candidate_starts[0] == candidate_starts[1]:
            return candidate_starts[0], candidate_starts[0] + n
        diffs = [
            abs(s - start * scale) + abs(s + n - end * scale) for s in candidate_starts
        ]
        if diffs[0] < diffs[1]:
            return candidate_starts[0], candidate_starts[0] + n
        else:
            return candidate_starts[1], candidate_starts[1] + n
    else:
        raise ValueError


@exception_printer
def tomo_producer_worker_func(
    in_queue: mp.Queue,
    out_queue: queue.Queue,
    num_remain_workers: mp.Value,
    lock: mp.Lock,
    device: str = "cpu",
    mode: str = "trilinear",
    worker_id: str = "",
):
    """Read tomo, spacing, then save to shared memory.
    Should be run at process level (multiprocessing)
    """
    logger.info("[TOMO PRODUCER %s] STARTING..", worker_id)
    _old_num_threads = torch.get_num_threads()
    if _old_num_threads < 16:
        torch.set_num_threads(16)
    logger.info(
        "[TOMO PRODUCER %s] Number of threads change from %d to %d",
        worker_id,
        _old_num_threads,
        torch.get_num_threads(),
    )
    tomo_loader = MultithreadOpencvTomogramLoader(
        num_workers=_CV2_TOMO_LOADER_NUM_THREADS
    )
    while True:
        item = in_queue.get()
        if item is None:
            with lock:
                num_remain_workers.value -= 1
                if num_remain_workers.value == 0:
                    # all other workers are stopped, this worker is the last one alive
                    # put None to indicate stopping to out_queue's consumer
                    out_queue.put(None)
            # other workers should stop too..
            in_queue.put(None)
            logger.debug(
                "[TOMO PRODUCER %s] STOPPING..",
                worker_id,
            )
            return
        start = time.time()
        tomo_idx, tomo_id, ori_spacing = item
        # fetch new item
        tomo = tomo_loader.load(os.path.join(DATA_DIR, tomo_id))
        ori_shape = tomo.shape
        target_spacings = TARGET_SPACINGS
        tomo = torch.from_numpy(tomo)
        assert tomo.dtype == torch.uint8

        # FIX BUG WRONG QUANTILE
        hist = torch.bincount(tomo.view(-1), minlength=256).cpu().numpy()
        assert hist.shape[0] == 256
        CUTOFF = 25
        is_wrong_quantile_dark = hist[:CUTOFF].sum() / hist[CUTOFF:].sum() > 0.08
        is_wrong_quantile_light = hist[-CUTOFF:].sum() / hist[:-CUTOFF].sum() > 0.08
        if is_wrong_quantile_dark or is_wrong_quantile_light:
            tomo = tomo + 127

        # Multiscale TTA
        logger.debug(
            "[TOMO PRODUCER %s] Read tomo_idx=%d tomo_id=%s tomo_shape=%s",
            worker_id,
            tomo_idx,
            tomo_id,
            tomo.shape,
        )
        proxy_tomos = []
        for spacing_idx, target_spacing in enumerate(target_spacings):
            spaced_tomo = spacing_torch(
                tomo, ori_spacing, target_spacing, device=device, mode=mode
            )
            gc.collect()
            logger.debug(
                "[TOMO PRODUCER %s] Spacing tomo_idx=%d tomo_id=%s shape %s -> %s",
                worker_id,
                tomo_idx,
                tomo_id,
                tomo.shape,
                spaced_tomo.shape,
            )
            proxy_tomos.append(
                ShmTensor.from_tensor(
                    spaced_tomo, f"{tomo_idx}_{tomo_id}_{spacing_idx}"
                )
            )
        del tomo
        gc.collect()
        end = time.time()
        logger.info(
            "[TOMO PRODUCER %s] Done tomo_idx=%d tomo_id=%s spacings %s -> %s with shape %s -> %s take %.2f sec",
            worker_id,
            tomo_idx,
            tomo_id,
            ori_spacing,
            target_spacings,
            ori_shape,
            tuple(e.shape for e in proxy_tomos),
            end - start,
        )
        out_queue.put((tomo_idx, tomo_id, ori_spacing, target_spacings, proxy_tomos))


@exception_printer
def patches_producer_worker_func(
    in_queue: mp.Queue,
    out_queue: queue.Queue,
    num_remain_workers: mp.Value,
    lock: threading.Lock,
    patch_size,
    border,
    overlap,
    worker_id="",
):
    """
    Should be lightweight, and expected to be run in thread level.

    Args:
        in_queue: Input queue contain item of (tomo_idx, tomo_id, ori_spacing, target_spacing, proxy_tomo)
        out_queue: Each item in output queue contains multiple patches belong to a same tomo
    """
    logger.info("[PATCHES PRODUCER %s] STARTING..", worker_id)
    while True:
        item = in_queue.get()
        if item is None:
            with lock:
                num_remain_workers.value -= 1
                if num_remain_workers.value == 0:
                    # put final None so out_queue's consumer should stop too
                    out_queue.put(None)
            # put None back to in_queue so other in_queue's consumer workers will stop too
            in_queue.put(None)
            logger.info(
                "\n\n[PATCHES PRODUCER %s] STOPPING..\n\n",
                worker_id,
            )
            return
        tomo_idx, tomo_id, ori_spacing, target_spacings, proxy_tomos = item
        start = time.time()
        assert len(proxy_tomos) == len(target_spacings) == len(TARGET_SPACINGS)

        items = []
        total_patches = 0
        cur_patch_idx = -1
        shm_tomos = []
        for target_spacing, proxy_tomo in zip(target_spacings, proxy_tomos):
            shm_tomo, tomo = proxy_tomo.to_tensor()
            tomo_shape = tomo.shape
            shm_tomos.append(shm_tomo)
            patch_positions = get_sliding_patch_positions(
                img_size=tomo.shape,
                patch_size=patch_size,
                border=border,
                overlap=overlap,
                validate=False,
            )
            # patch_positions = torch.from_numpy(patch_positions)
            patch_positions = patch_positions.tolist()
            total_patches += len(patch_positions)

            for patch_pos in patch_positions:
                # crop tomo patch
                crop = crop_tomo_patch(tomo, patch_pos)  # ZYX, uint8
                cur_patch_idx += 1
                item = {
                    "tomo_idx": torch.tensor(tomo_idx, dtype=torch.int16),
                    "patch_pos": torch.tensor(patch_pos, dtype=torch.float32),
                    "tomo_shape": torch.tensor(tomo_shape, dtype=torch.int16),
                    "ori_spacing": torch.tensor(ori_spacing, dtype=torch.float32),
                    "target_spacing": torch.tensor(target_spacing, dtype=torch.float32),
                    "cur_patch_idx": torch.tensor(cur_patch_idx, dtype=torch.int16),
                    "image": crop,
                }
                items.append(item)
        assert cur_patch_idx == total_patches - 1
        for item in items:
            item["total_patches"] = torch.tensor(total_patches, dtype=torch.int16)

        end = time.time()
        logger.debug(
            "[PATCHES PRODUCER %s] Done tomo_idx=%d tomo_id=%s take %.2f sec",
            worker_id,
            tomo_idx,
            tomo_id,
            end - start,
        )
        out_queue.put((tomo_idx, shm_tomos, items))


@exception_printer
def batch_producer_worker_func(
    in_queue: queue.Queue,
    out_queue: queue.Queue,
    batch_size,
    device="cpu",
):
    """Should be lightweight, and expected to be run in thread level."""
    count = 0
    items = []
    should_continue = True
    shm_dict = {}
    last_tomo_idx = -1
    while should_continue:
        item = in_queue.get()
        if item is None:
            # put None so in_queue's consumer should notice and stop too
            in_queue.put(None)
            should_continue = False
        else:
            tomo_idx, shm_tomos, new_items = item
            # @TODO - remove this line
            assert all([tomo_idx == e["tomo_idx"].item() for e in new_items])
            items.extend(new_items)
            shm_dict[tomo_idx] = shm_tomos

        while True:
            if len(items) == 0 or (len(items) < batch_size and should_continue):
                break
            start = time.time()
            cur_bs = min(batch_size, len(items))
            batch = items[:cur_bs]
            batch = default_collate(batch)
            if device != "cpu":
                batch = {k: v.to(device) for k, v in batch.items()}
            assert batch["image"].dtype == torch.uint8
            end = time.time()
            logger.debug(
                "[BATCH PRODUCER] new batch %d (tomo %d) take %.4f",
                count,
                tomo_idx,
                end - start,
            )
            out_queue.put(batch)
            count += 1
            last_tomo_idx = batch["tomo_idx"][-1].item()
            items = items[cur_bs:]
        for shm_tomo_idx in list(shm_dict.keys()):
            # un-collated batch should start with tomo_idx >= last_tomo_idx
            # so we're safe to unlink previous one
            # since default_collate() copy to a newly allocated tensor
            if shm_tomo_idx < last_tomo_idx:
                for shm in shm_dict[shm_tomo_idx]:
                    shm.unlink()
                del shm_dict[shm_tomo_idx]

    for shms in shm_dict.values():
        for shm in shms:
            shm.unlink()
    del shm_dict
    gc.collect()
    # put a final None to indicate `all done`, stop inference worker too..
    out_queue.put(None)
    logger.info("[BATCH PRODUCER] Stopped since tomo_idx=%d", tomo_idx)
    assert len(items) == 0


@exception_printer
def inference_worker(worker_id, input_queue, chunk_idx, gpu_id, model_cfgs):
    logger.info(
        "STARTING INFERENCE WORKER %d with device cuda:%d, current input queue size %d, %d models with configs:\n%s",
        worker_id,
        gpu_id,
        input_queue.qsize(),
        len(model_cfgs),
        model_cfgs,
    )
    device = torch.device(f"cuda:{gpu_id}")
    patch_size = PATCH_SIZE
    border = PATCH_BORDER
    overlap = PATCH_OVERLAP

    ############ START ALL WORKERS ###########
    ##########################################
    tomo_queue = mp.Queue(maxsize=TOMO_PRODUCER_PREFETCH)
    tomo_producer_num_remain_workers = mp.Value("i", TOMO_PRODUCER_NUM_WORKERS)
    tomo_producer_lock = mp.Lock()
    patches_queue = queue.Queue(maxsize=PATCHES_PRODUCER_PREFETCH)
    patches_producer_num_remain_workers = mp.Value("i", PATCHES_PRODUCER_NUM_WORKERS)
    patches_producer_lock = threading.Lock()
    batch_queue = queue.Queue(maxsize=DATALOADER_PREFETCH)

    # PRODUCE SPACED TOMO
    tomo_producer_workers = []
    for _worker_id in range(TOMO_PRODUCER_NUM_WORKERS):
        worker = mp.Process(
            group=None,
            target=tomo_producer_worker_func,
            name=f"tomo_producer_{worker_id}.{_worker_id}",
            args=(
                input_queue,
                tomo_queue,
                tomo_producer_num_remain_workers,
                tomo_producer_lock,
            ),
            kwargs={
                "device": TOMO_SPACING_DEVICE,
                "mode": TOMO_SPACING_MODE,
                "worker_id": f"{worker_id}.{_worker_id}",
            },
        )
        worker.start()
        tomo_producer_workers.append(worker)

    # PRODUCER TOMO PATCHES
    patches_producer_workers = []
    for _worker_id in range(PATCHES_PRODUCER_NUM_WORKERS):
        worker = threading.Thread(
            group=None,
            target=patches_producer_worker_func,
            name=f"patches_producer_{worker_id}.{_worker_id}",
            args=(
                tomo_queue,
                patches_queue,
                patches_producer_num_remain_workers,
                patches_producer_lock,
                patch_size,
                border,
                overlap,
            ),
            kwargs={"worker_id": f"{worker_id}.{_worker_id}"},
        )
        worker.start()
        patches_producer_workers.append(worker)

    # BATCH PRODUCER
    batch_producer_worker = threading.Thread(
        group=None,
        target=batch_producer_worker_func,
        name="batch_producer",
        args=(patches_queue, batch_queue, BATCH_SIZE),
        kwargs={"device": "cuda:0"},
    )
    batch_producer_worker.start()

    ############## LOAD MODEL ##################
    ############################################
    logger.info("LOADING MODEL..")
    models = []
    unique_tta_names = set()
    unique_agg_names = set()
    sessions = []
    for session_idx, model_cfg in enumerate(model_cfgs):
        model_tta_names = model_cfg["tta"]
        model_agg_weights = model_cfg["weight"]
        model_agg_names = model_cfg["agg_name"]
        assert len(model_tta_names) == len(model_agg_weights) == len(model_agg_names)
        assert len(model_tta_names) == len(set(model_tta_names))
        model = Model2dWrapper(
            model_cfg["sig"],
            gpu_id,
            act="identity" if HEATMAP_AGG_LOGITS else "sigmoid",
        )
        model.eval().to(device)
        models.append(model)
        unique_tta_names.update(model_tta_names)
        unique_agg_names.update(model_agg_names)
        logger.info(
            "Loaded model %d with TTA=%s, WEIGHTS=%s",
            session_idx,
            model_tta_names,
            model_agg_weights,
        )
        for tta_name, agg_weight, agg_name in zip(
            model_tta_names, model_agg_weights, model_agg_names
        ):
            assert tta_name in ALL_SUPPORTED_TTAS
            sessions.append((model, tta_name, agg_weight, agg_name))
    unique_tta_names = list(unique_tta_names)
    unique_agg_names = list(unique_agg_names)
    num_sessions = len(sessions)
    _session_agg_names = [e[-1] for e in sessions]
    agg_name_to_first_session_idx = {
        agg_name: _session_agg_names.index(agg_name) for agg_name in unique_agg_names
    }
    agg_name_to_last_session_idx = {
        agg_name: num_sessions - 1 - _session_agg_names[::-1].index(agg_name)
        for agg_name in unique_agg_names
    }
    logger.info(
        "FIRST/LAST SESSION IDX:\n%s\n%s",
        agg_name_to_first_session_idx,
        agg_name_to_last_session_idx,
    )

    logger.info("-----------\nLOADED %d MODELS:\n%s------------", len(models), models)
    logger.info(
        "TOTALLY ENABLE %d MODELS, %d TTA %s, %d AGG_NAMES %s => %d SESSIONS",
        len(models),
        len(unique_tta_names),
        unique_tta_names,
        len(unique_agg_names),
        unique_agg_names,
        len(sessions),
    )
    unique_tta_tfs = {
        tta_name: build_tta(tta_name, 1.0, ori_dims="yx")
        for tta_name in unique_tta_names
    }

    # Postprocess: heatmap blurring
    if DECODE_NMS_BLUR_SIGMA is not None:
        from yagm.transforms import monai_custom as CT

        sigma_voxel = [
            1000 / e / s * DECODE_NMS_BLUR_SIGMA
            for e, s in zip(TARGET_SPACINGS[0], HEATMAP_STRIDE)
        ]
        blur_operator = (
            CT.CustomGaussianFilter(
                spatial_dims=3,
                sigma=sigma_voxel,
                truncated=4,
                approx="erf",
                requires_grad=False,
            )
            .eval()
            .to(device)
        )
    else:
        blur_operator = None
    logger.info(
        "POST-PROCESSING BLUR OPERATOR:\n%s on device %s",
        blur_operator,
        getattr(blur_operator, "device", None),
    )

    cache_results = {
        agg_name: {
            "cur_pred_heatmap_sum": None,
            "cur_pred_heatmap_count": None,
            "cur_tomo_idx": None,
            "cur_tomo_id": None,
            "main_target_spacing": None,
            "submission": SubmissionDataFrame(),
        }
        for agg_name in unique_agg_names
    }
    logger.info("\n\n\nWORKER %d STARTING INFERENCE..\n", worker_id)
    infer_start = time.time()
    cur_batch_idx = 0
    pbar = tqdm(desc=f"===== INFERENCE {worker_id} =====")
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16
    ):
        while True:
            batch = batch_queue.get()
            if batch is None:
                break
            # print("GOT BATCH:", {(k, v.shape, v.dtype) for k, v in batch.items()})
            batch_image = batch["image"].to(device)
            assert batch_image.dtype == torch.uint8
            B = batch_image.shape[0]

            unique_tta_batch_images = {
                tta_name: tta_tf.transform(batch_image)[0].float().contiguous()
                for tta_name, tta_tf in unique_tta_tfs.items()
            }

            batch_outputs = []
            for model, tta_name, agg_weight, agg_name in sessions:
                tta_tf = unique_tta_tfs[tta_name]
                tta_batch_image = unique_tta_batch_images[tta_name]
                tta_batch_heatmap = model(tta_batch_image)
                batch_heatmap = tta_tf.invert(tta_batch_heatmap)[0]
                batch_outputs.append((batch_heatmap, agg_weight, agg_name))

            assert len(batch_outputs) == num_sessions

            for i in range(B):
                for session_idx, (batch_heatmap, agg_weight, agg_name) in enumerate(
                    batch_outputs
                ):
                    pred_patch_heatmap = batch_heatmap[i]
                    C = pred_patch_heatmap.shape[0]
                    assert C == 1

                    tomo_idx = batch["tomo_idx"][i]
                    tomo_id = ALL_TOMO_IDS[tomo_idx]
                    tomo_shape = batch["tomo_shape"][i].tolist()
                    ori_spacing = batch["ori_spacing"][i].tolist()
                    target_spacing = batch["target_spacing"][i].tolist()
                    cur_patch_idx = int(batch["cur_patch_idx"][i].item())
                    total_patches = int(batch["total_patches"][i].item())
                    patch_position = batch["patch_pos"][i].cpu()

                    is_first = (cur_patch_idx == 0) and (
                        session_idx == agg_name_to_first_session_idx[agg_name]
                    )
                    is_last = (cur_patch_idx == total_patches - 1) and (
                        session_idx == agg_name_to_last_session_idx[agg_name]
                    )

                    # load cache for this tomo+agg combination
                    agg_result = cache_results[agg_name]
                    ### start IS_FIRST
                    if is_first:
                        logger.info(
                            "[INFER %d] Model %s receive first patch of tomo %s (index %d)",
                            worker_id,
                            session_idx,
                            tomo_id,
                            tomo_idx,
                        )
                        assert all(
                            [
                                agg_result[k] is None
                                for k in [
                                    "cur_tomo_idx",
                                    "cur_tomo_id",
                                    "cur_pred_heatmap_sum",
                                    "cur_pred_heatmap_count",
                                    "main_target_spacing",
                                ]
                            ]
                        )

                        # allocate new heatmap_sum and heatmap_count
                        agg_result["cur_tomo_idx"] = tomo_idx
                        agg_result["cur_tomo_id"] = tomo_id
                        shared_agg_heatmap_shape = [
                            round(e / stride)
                            for e, stride in zip(tomo_shape, HEATMAP_STRIDE)
                        ]
                        print(tomo_shape, '-->', shared_agg_heatmap_shape)
                        # float16 to save memory
                        agg_result["cur_pred_heatmap_sum"] = torch.zeros(
                            (C, *shared_agg_heatmap_shape),
                            dtype=torch.float16,
                            device=pred_patch_heatmap.device,
                        )
                        agg_result["cur_pred_heatmap_count"] = torch.zeros(
                            (C, *shared_agg_heatmap_shape),
                            dtype=torch.float16,
                            device=pred_patch_heatmap.device,
                        )
                        agg_result["main_target_spacing"] = target_spacing
                        assert all(
                            [
                                abs(a - b) < 1e-4
                                for a, b in zip(target_spacing, TARGET_SPACINGS[0])
                            ]
                        )
                    ### end IS_FIRST

                    assert tomo_idx == agg_result["cur_tomo_idx"]
                    assert all(
                        [
                            a <= b
                            for a, b in zip(
                                agg_result["main_target_spacing"], target_spacing
                            )
                        ]
                    )
                    cur_pred_heatmap_sum = agg_result["cur_pred_heatmap_sum"]
                    cur_pred_heatmap_count = agg_result["cur_pred_heatmap_count"]
                    main_target_spacing = agg_result["main_target_spacing"]
                    cur_submission: SubmissionDataFrame = agg_result["submission"]
                    shared_agg_heatmap_shape = cur_pred_heatmap_sum.shape[1:]

                    scale_zyx = [
                        a / b for a, b in zip(shared_agg_heatmap_shape, tomo_shape)
                    ]
                    scale_z, scale_y, scale_x = scale_zyx

                    # interpolate to shared coordinate space
                    C, Y2, X2 = pred_patch_heatmap.shape
                    assert C == 1
                    patch_heatmap_shape = [
                        round(e * s) for e, s in zip(patch_size, scale_zyx)
                    ]
                    if [Y2, X2] != patch_heatmap_shape[1:]:
                        logger.debug(
                            "PATCH HEATMAP INTERPOLATE: %s --> %s",
                            (Y2, X2),
                            patch_heatmap_shape[1:],
                        )
                        pred_patch_heatmap = F.interpolate(
                            pred_patch_heatmap[None],  # (1, C, Y2, X2)
                            size=patch_heatmap_shape[1:],
                            mode=HEATMAP_INTERPOLATION_MODE,
                            align_corners=False,
                        )[
                            0
                        ]  # (C, Y3, X3)
                    else:
                        logger.debug("PATCH HEATMAP SAME: %s", pred_patch_heatmap.shape)
                        pass

                    roi_start, roi_end, patch_start, patch_end = patch_position.tolist()
                    top_pad_z, top_pad_y, top_pad_x = [
                        max(-start, 0) for start in roi_start
                    ]
                    bot_pad_z, bot_pad_y, bot_pad_x = [
                        max(0, end - size) for end, size in zip(roi_end, tomo_shape)
                    ]
                    assert (
                        top_pad_z == bot_pad_z == 0
                        and roi_end[0] - roi_start[0] == patch_size[0]
                    )
                    assert (
                        tuple(
                            round((pe - ps) * s)
                            for pe, ps, s in zip(
                                patch_end[1:], patch_start[1:], scale_zyx[1:]
                            )
                        )
                        == pred_patch_heatmap.shape[1:]
                    )
                    sz, sy, sx = tuple(
                        rs - ps for rs, ps in zip(roi_start, patch_start)
                    )
                    ez, ey, ex = tuple(
                        ps - pe + re
                        for ps, pe, re in zip(
                            (1, patch_size[1], patch_size[2]), patch_end, roi_end
                        )
                    )
                    assert sz == 0 and ez == 1 and top_pad_z == bot_pad_z == 0
                    assert (roi_end[1] - bot_pad_y) - (roi_start[1] + top_pad_y) == (
                        ey - bot_pad_y
                    ) - (sy + top_pad_y)
                    assert (roi_end[2] - bot_pad_x) - (roi_start[2] + top_pad_x) == (
                        ex - bot_pad_x
                    ) - (sx + top_pad_x)

                    dst_shape = shared_agg_heatmap_shape
                    dst_slices = [
                        slice(None),
                        slice(
                            *scale_range(
                                roi_start[0] + top_pad_z,
                                roi_end[0] - bot_pad_z,
                                scale_z,
                                0,
                                dst_shape[0],
                            )
                        ),
                        slice(
                            *scale_range(
                                roi_start[1] + top_pad_y,
                                roi_end[1] - bot_pad_y,
                                scale_y,
                                0,
                                dst_shape[1],
                            )
                        ),
                        slice(
                            *scale_range(
                                roi_start[2] + top_pad_x,
                                roi_end[2] - bot_pad_x,
                                scale_x,
                                0,
                                dst_shape[2],
                            )
                        ),
                    ]
                    src_shape = pred_patch_heatmap.shape[1:]
                    src_slices = [
                        slice(None),
                        None,
                        slice(
                            *scale_range(
                                sy + top_pad_y, ey - bot_pad_y, scale_y, 0, src_shape[0]
                            )
                        ),
                        slice(
                            *scale_range(
                                sx + top_pad_x, ex - bot_pad_x, scale_x, 0, src_shape[1]
                            )
                        ),
                    ]

                    if HEATMAP_AGG_MODE == "avg":
                        cur_pred_heatmap_sum[dst_slices] += (
                            pred_patch_heatmap[src_slices] * agg_weight
                        )
                        cur_pred_heatmap_count[dst_slices] += agg_weight
                    elif HEATMAP_AGG_MODE == "max":
                        torch.maximum(
                            cur_pred_heatmap_sum[dst_slices],
                            pred_patch_heatmap[src_slices],
                            out=cur_pred_heatmap_sum[dst_slices],
                        )
                    else:
                        raise ValueError

                    if is_last:
                        # DECODE HEATMAP TO COORDINATE
                        tomo_id = agg_result["cur_tomo_id"]
                        logger.info(
                            "[INFER %d] Receive last patch of tomo %s (index %d), decoding..",
                            worker_id,
                            tomo_id,
                            tomo_idx,
                        )

                        if HEATMAP_AGG_MODE == "avg":
                            # ensure prediction cover all the tomogram
                            # assert torch.all(cur_pred_heatmap_count > 0)
                            heatmap = torch.div(
                                cur_pred_heatmap_sum,
                                cur_pred_heatmap_count,
                                out=cur_pred_heatmap_sum,
                            )
                        elif HEATMAP_AGG_MODE == "max":
                            heatmap = cur_pred_heatmap_sum
                        else:
                            raise ValueError
                        if HEATMAP_AGG_LOGITS:
                            heatmap = F.sigmoid(heatmap)

                        if SAVE_HEATMAP_NPY:
                            heatmap_npy = heatmap.half().cpu().numpy()
                            assert heatmap_npy.dtype == np.float16
                            save_heatmap_path = os.path.join(
                                SAVE_HEATMAP_NPY_DIR, agg_name, f"{tomo_id}.npy"
                            )
                            os.makedirs(
                                os.path.dirname(save_heatmap_path), exist_ok=True
                            )
                            np.save(save_heatmap_path, heatmap_npy)

                        if DECODE_HEATMAP_TO_CSV:
                            # Decode heatmap
                            _start = time.time()
                            if DECODE_METHOD == "nms":
                                outputs = decode_heatmap(
                                    heatmap,
                                    main_target_spacing,
                                    HEATMAP_STRIDE,
                                    blur_operator=blur_operator,
                                )
                            elif DECODE_METHOD == "cc3d":
                                outputs = decode_segment_mask(
                                    heatmap,
                                    main_target_spacing,
                                    HEATMAP_STRIDE,
                                    prob_thres=DECODE_CC3D_CONF_THRES,
                                    radius_factor_thres=DECODE_CC3D_RADIUS_FACTOR,
                                    conf_mode=DECODE_CC3D_CONF_MODE,
                                    prob_mode=DECODE_CC3D_PROB_MODE,
                                )
                            else:
                                raise ValueError
                            _end = time.time()
                            logger.debug(
                                "Decode heatmap of shape %s take %.4f sec",
                                heatmap.shape,
                                _end - _start,
                            )
                            # Convert to submission coordinates
                            assert len(outputs) == 1
                            heatmap_stride_tensor = torch.tensor(
                                [HEATMAP_STRIDE], dtype=torch.float32
                            )  # (1, 3)
                            main_target_spacing_tensor = torch.tensor(
                                [main_target_spacing], dtype=torch.float32
                            )  # (1, 3)
                            ori_spacing_tensor = torch.tensor(
                                [ori_spacing], dtype=torch.float32
                            )  # (1, 3)
                            for _channel_idx, keypoints in enumerate(outputs):
                                # back to original coordinate space
                                keypoints[:, :3] = (keypoints[:, :3] + 0.5) * (
                                    heatmap_stride_tensor
                                    * main_target_spacing_tensor
                                    / ori_spacing_tensor
                                ) - 0.5
                                keypoints = keypoints.tolist()

                                ###### VISUALIZATION ######
                                if VIZ_ENABLE and MODE != "KAGGLE":
                                    tomo_dir = os.path.join(DATA_DIR, tomo_id)
                                    row = GT_DF[GT_DF["tomo_id"] == tomo_id].iloc[0]
                                    ori_shape = (row["Z"], row["Y"], row["X"])
                                    gt_zyx = [
                                        row["motor_z"],
                                        row["motor_y"],
                                        row["motor_x"],
                                    ]
                                    viz = viz_byu(
                                        tomo_dir,
                                        ori_spacing,
                                        ori_shape,
                                        gt_zyx,
                                        heatmap[0],
                                        keypoints,
                                    )
                                    # get the kind/judge of current prediction, one of TP, TN, FP, FN
                                    if (
                                        len(keypoints) > 0
                                        and keypoints[0][3] >= DECODE_CONF_THRES
                                    ):
                                        # predicted as positive
                                        pred_zyx = keypoints[0][:3]
                                        if gt_zyx[0] == -1:
                                            # GT is negative
                                            judge = "FP"
                                        else:
                                            dist = (
                                                (gt_zyx[0] - pred_zyx[0]) ** 2
                                                + (gt_zyx[1] - pred_zyx[1]) ** 2
                                                + (gt_zyx[2] - pred_zyx[2]) ** 2
                                            ) ** 0.5
                                            if dist <= 1000 / ori_spacing:
                                                judge = "TP"
                                            else:
                                                judge = "FN"
                                    else:
                                        # predicted as negative
                                        if gt_zyx[0] == -1:
                                            # GT is negative
                                            judge = "TN"
                                        else:
                                            judge = "FN"

                                    save_name = f"{judge}-{tomo_idx}-{tomo_id}-{'x'.join([str(int(e)) for e in ori_shape])}-spacing{ori_spacing}.jpg"
                                    save_path = os.path.join(
                                        TMP_VIZ_DIR, judge, agg_name, save_name
                                    )
                                    os.makedirs(
                                        os.path.dirname(save_path), exist_ok=True
                                    )
                                    cv2.imwrite(
                                        save_path, cv2.cvtColor(viz, cv2.COLOR_RGB2BGR)
                                    )

                                if len(keypoints) > 0:
                                    for kpt in keypoints:
                                        z, y, x, conf = kpt
                                        cur_submission.add_row(tomo_id, x, y, z, conf)
                                else:
                                    # no motor detected
                                    cur_submission.add_row(tomo_id, -1, -1, -1, 0.0)

                        # clear the cache
                        for k in [
                            "cur_tomo_idx",
                            "cur_tomo_id",
                            "cur_pred_heatmap_sum",
                            "cur_pred_heatmap_count",
                            "main_target_spacing",
                        ]:
                            agg_result[k] = None
                        gc.collect()
                        # torch.cuda.empty_cache()
            cur_batch_idx += 1
            pbar.update(1)
    pbar.close()
    infer_end = time.time()

    logger.info(
        "\n\n\n INFER WORKER %d INFER ON %d BATCHES TAKE %.4f sec",
        worker_id,
        cur_batch_idx,
        infer_end - infer_start,
    )

    assert len(cache_results) == len(unique_agg_names)
    if DECODE_HEATMAP_TO_CSV:
        for agg_name, agg_result in cache_results.items():
            submision = agg_result["submission"]
            df = submision.to_pandas(submit=False)
            csv_path = os.path.join(TMP_CSV_DIR, f"{agg_name}_chunk{chunk_idx}.csv")
            df.to_csv(csv_path, index=False)

    # JOIN
    logger.info("Worker %d waiting for all worker to be finished..", worker_id)
    all_workers = [
        batch_producer_worker,
        *tomo_producer_workers,
        *patches_producer_workers,
    ]
    for worker in all_workers:
        worker.join()
    return None


########################### MAIN CODE #############################
###################################################################
logger.warning("CLEARING ALL CONTENTS IN: %s", [TMP_CSV_DIR, TMP_CSV_DIR, TMP_VIZ_DIR])
clear_dir_content("/dev/shm/")
# clear_dir_content(TMP_CSV_DIR)
clear_dir_content(TMP_VIZ_DIR)
if MODE == "LOCAL":
    clear_dir_content(WORKING_DIR)
os.makedirs(TMP_CSV_DIR, exist_ok=True)
os.makedirs(TMP_VIZ_DIR, exist_ok=True)


global_start = time.time()

all_unique_agg_names = set()
for job_idx, model_cfgs in enumerate(JOBS):
    # fill the input queue
    input_queue = mp.Queue()
    for tomo_idx, (tomo_id, ori_spacing) in enumerate(
        zip(ALL_TOMO_IDS, ALL_TOMO_SPACINGS)
    ):
        input_queue.put((tomo_idx, tomo_id, [ori_spacing] * 3))
    # None to indicate ending and workers should stop when receive this None
    input_queue.put(None)

    job_unique_agg_names = set()
    for model_cfg in model_cfgs:
        job_unique_agg_names.update(model_cfg["agg_name"])
    assert len(all_unique_agg_names.intersection(job_unique_agg_names)) == 0
    all_unique_agg_names.update(job_unique_agg_names)

    job_workers = []
    for chunk_idx, device_id in enumerate(DEVICES):
        # worker_id, input_queue, chunk_idx, gpu_id, model_cfgs
        worker = mp.Process(
            group=None,
            target=inference_worker,
            name=f"INFERENCE_WORKER_job={job_idx}_chunk={chunk_idx}",
            kwargs={
                "worker_id": chunk_idx,
                "input_queue": input_queue,
                "chunk_idx": chunk_idx,
                "gpu_id": device_id,
                "model_cfgs": model_cfgs,
            },
        )
        worker.start()
        job_workers.append(worker)

    for worker in job_workers:
        worker.join()

global_end = time.time()
logger.info(
    "\n\n\n>>>>>> FINISH ALL INFERENCE TASK WITHIN %.4f sec (%.4f min)",
    global_end - global_start,
    (global_end - global_start) / 60.0,
)
all_unique_agg_names = list(all_unique_agg_names)
print('ALL UNIQUE AGG NAMES:', all_unique_agg_names)


!python3 /kaggle/working/byu/submit_2d.py





import warnings

warnings.simplefilter("ignore")
import gc
import json
import logging
import multiprocessing as mp
import os
import queue
import shutil
import sys
import threading
import time
from functools import partial

import cc3d
import cv2
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from torch.utils.data import default_collate
from tqdm import tqdm
from yagm.tasks.base_task import BaseTask
from yagm.transforms.keypoints.decode import decode_heatmap_3d, decode_segment_mask_3d
from yagm.transforms.sliding_window import get_sliding_patch_positions
from yagm.transforms.tta_3d import build_tta
from yagm.utils import hydra as hydra_utils
from yagm.utils import lightning as l_utils
from yagm.utils.concurrent import ShmTensor

# Setting up custom logging
from yagm.utils.logging import init_logging, setup_logging

from byu.data.io import MultithreadOpencvTomogramLoader
from byu.inference import zoo
from byu.inference.tensorrt_engine import ThreadsafeTRTEngine
from byu.utils.data import SubmissionDataFrame
from byu.utils.metrics import compute_metrics, kaggle_score
from byu.utils.viz import viz_byu
from byu.utils.wbf import weighted_boxes_fusion_3d

try:
    hydra_utils.init_hydra()
except:
    print("SKIP RE-INIT HYDRA")

### SETUP LOGGER FOR IPYTHON ###
# ref: https://github.com/ipython/ipykernel/issues/111
# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Create STDERR handler
handler = logging.StreamHandler(sys.stderr)
# Create formatter and add it to the handler
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
# Set STDERR handler as the only handler
logger.handlers = [handler]


all_unique_agg_names = [
    'DENSENET-zyx', 'DENSENET-zxy', 'X3D-zyx_y', 'X3D-zyx_x', 'RESNEXT50-zyx_xy', 'CONVNEXT-zxy_xy', 'MAXVIT-yx', 'MAXVIT-xy'
]
all_unique_agg_weights = {agg_name: 1.0 for agg_name in all_unique_agg_names}


##################################################
###### GLOBAL VARS
##################################################
_CV2_TOMO_LOADER_NUM_THREADS = 8

PATCH_SIZE = [224, 448, 448]
PATCH_BORDER = [0, 0, 0]
PATCH_OVERLAP = [0, 0, 0]

IO_BACKEND = "cv2"
BASE_SPACING = 13.1  # avg 15.6
# TARGET_SPACINGS = [13.1, 16.0, 19.7]
TARGET_SPACINGS = [[16.0, 16.0, 16.0]]
TOMO_SPACING_DEVICE = "cpu"
TOMO_SPACING_MODE = "trilinear"

HEATMAP_AGG_MODE = "avg"
HEATMAP_AGG_LOGITS = True
HEATMAP_INTERPOLATION_MODE = "trilinear"
FORWARD_SAVE_MEMORY = False
BATCH_SIZE = 1
HEATMAP_STRIDE = 8

# MASK CC3D DECODING
DECODE_CC3D_CONF_THRES = 0.05
DECODE_CC3D_RADIUS_FACTOR = 0.1
DECODE_CC3D_CONF_MODE = "prob"  # volume | prob | fuse
DECODE_CC3D_PROB_MODE = "center"  # center | mean | max
# HEATMAP NMS DECODING
DECODE_NMS_BLUR_SIGMA = None
###### MAIN DECODE PARAMS ######
DECODE_METHOD = "nms"  # nms, cc3d
DECODE_CONF_THRES = 0.05
QUANTILE_THRES = 0.55

SAVE_HEATMAP_NPY = False
DECODE_HEATMAP_TO_CSV = True
CSV_ENSEMBLE_MODE = "max"  # max | wbf
WBF_CONF_TYPE = "avg"  # avg | max
WBF_CONF_THRES = 0.2

VIZ_ENABLE = False


# automatic determine the MODE
if os.path.isdir("/kaggle/input/byu-locating-bacterial-flagellar-motors-2025"):
    MODE = "KAGGLE"
else:
    MODE = "LOCAL"
# manual override
# MODE = "KAGGLE_VAL"

# LOCAL DEVELOPMENT ENV
if MODE == "LOCAL":
    DEVICES = [0]
    ASSETS_DIR = "assets/"
    WORKING_DIR = "outputs/submit/working/"

    DATA_DIR = "/home/dangnh36/datasets/.comp/byu/raw/train"
    VAL_GT_CSV_PATH = "/home/dangnh36/datasets/.comp/byu/processed/gt_v3.csv"
    with open(
        "/home/dangnh36/datasets/.comp/byu/processed/cv/v3/skf4_rd42.json", "r"
    ) as f:
        cv_meta = json.load(f)
        ALL_TOMO_IDS = cv_meta["folds"][0]["val"][:]  # fold 0

    # DATA_DIR = "/home/dangnh36/datasets/.comp/byu/processed/pseudo_test/tomograms"
    # VAL_GT_CSV_PATH = "/home/dangnh36/datasets/.comp/byu/processed/pseudo_test/gt.csv"
    # ALL_TOMO_IDS = sorted(os.listdir(DATA_DIR)) * 10

    GT_DF = pd.read_csv(VAL_GT_CSV_PATH)
    GT_DF = GT_DF[GT_DF["tomo_id"].isin(ALL_TOMO_IDS)].reset_index(drop=True)
    _tomo2spacing = {}
    for i, row in GT_DF.iterrows():
        _tomo2spacing[row["tomo_id"]] = row["voxel_spacing"]
    ALL_TOMO_SPACINGS = [_tomo2spacing[tomo_id] for tomo_id in ALL_TOMO_IDS]

    # TOMO_PRODUCER_NUM_WORKERS = 8
    # TOMO_PRODUCER_PREFETCH = 8
    # PATCHES_PRODUCER_NUM_WORKERS = 2
    # PATCHES_PRODUCER_PREFETCH = 8
    # DATALOADER_PREFETCH = 8

    TOMO_PRODUCER_NUM_WORKERS = 1
    TOMO_PRODUCER_PREFETCH = 1
    PATCHES_PRODUCER_NUM_WORKERS = 1
    PATCHES_PRODUCER_PREFETCH = 2
    DATALOADER_PREFETCH = 8

# VALIDATION ON KAGGLE
elif MODE == "KAGGLE_VAL":
    DEVICES = [0, 1]
    ASSETS_DIR = "/kaggle/input/byu-final-dataset/"
    DATA_DIR = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train"
    WORKING_DIR = "/kaggle/working/"
    VAL_GT_CSV_PATH = "/kaggle/input/byu-checkpoints/gt_v2.csv"
    with open("/kaggle/input/byu-checkpoints/skf4_rd42.json", "r") as f:
        cv_meta = json.load(f)
        ALL_TOMO_IDS = cv_meta["folds"][0]["val"][:50]  # fold 0, top 10 first tomo
    GT_DF = pd.read_csv(VAL_GT_CSV_PATH)
    GT_DF = GT_DF[GT_DF["tomo_id"].isin(ALL_TOMO_IDS)].reset_index(drop=True)
    _tomo2spacing = {}
    for i, row in GT_DF.iterrows():
        _tomo2spacing[row["tomo_id"]] = row["voxel_spacing"]
    ALL_TOMO_SPACINGS = [_tomo2spacing[tomo_id] for tomo_id in ALL_TOMO_IDS]

    TOMO_PRODUCER_NUM_WORKERS = 1
    TOMO_PRODUCER_PREFETCH = 1
    PATCHES_PRODUCER_NUM_WORKERS = 1
    PATCHES_PRODUCER_PREFETCH = 2
    DATALOADER_PREFETCH = 8

# PRIVATE TEST SUBMISSION
elif MODE == "KAGGLE":
    DEVICES = [0, 1]
    ASSETS_DIR = "/kaggle/input/byu-final-dataset/"
    DATA_DIR = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/test"
    WORKING_DIR = "/kaggle/working/"
    ALL_TOMO_IDS = sorted(os.listdir(DATA_DIR))
    ALL_TOMO_SPACINGS = [BASE_SPACING] * len(ALL_TOMO_IDS)

    TOMO_PRODUCER_NUM_WORKERS = 1
    TOMO_PRODUCER_PREFETCH = 1
    PATCHES_PRODUCER_NUM_WORKERS = 1
    PATCHES_PRODUCER_PREFETCH = 2
    DATALOADER_PREFETCH = 8

else:
    raise ValueError

TMP_CSV_DIR = os.path.join(WORKING_DIR, "tmp_csv")
TMP_VIZ_DIR = os.path.join(WORKING_DIR, "tmp_viz")
SAVE_HEATMAP_NPY_DIR = os.path.join(WORKING_DIR, "tmp_heatmap")

# RECHECK SOME CONDITIONS
assert len(ALL_TOMO_IDS) == len(ALL_TOMO_SPACINGS)
# assert not (SAVE_HEATMAP_NPY and DECODE_HEATMAP_TO_CSV) and (SAVE_HEATMAP_NPY or DECODE_HEATMAP_TO_CSV)


###################################################
###### FUNCTIONS/CLASSES DEFINITION
###################################################
def exception_printer(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception("Exception %s in %s: %s", type(e), func.__name__, e)
            import traceback

            traceback.print_exc()
            raise e

    return wrapper


def clear_dir_content(folder_path):
    if os.path.isdir(folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
                

if DECODE_HEATMAP_TO_CSV:
    all_agg_dfs = {}
    for agg_name in all_unique_agg_names:
        logger.info("PROCESSING AGG_NAME=%s", agg_name)
        chunk_agg_dfs = []
        for chunk_idx in range(len(DEVICES)):
            csv_name = f"{agg_name}_chunk{chunk_idx}.csv"
            csv_path = os.path.join(TMP_CSV_DIR, csv_name)
            try:
                df = pd.read_csv(csv_path)
                chunk_agg_dfs.append(df)
                logger.info(
                    "Append new chunk Dataframe with shape %s, columns=%s",
                    df.shape,
                    list(df.columns),
                )
            except Exception as e:
                logger.warning(
                    "EXCEPTION OCCUR when processing agg_name=%s:\n%s\nIGNORE CSV RESULT FILE AT %s",
                    agg_name,
                    e,
                    csv_path,
                )
        agg_df = pd.concat(chunk_agg_dfs, axis=0, ignore_index=True).reset_index(
            drop=True
        )
        logger.info(
            "CONCAT ALL TEMPORARY DATAFRAMES INTO A SINGLE DATAFRAME WITH SHAPE %s, COLUMNS=%s\n%s",
            df.shape,
            list(df.columns),
            df,
        )
        all_agg_dfs[agg_name] = agg_df
        print("-------------------------------------------\n\n")
    logger.info("TOTAL %d AGG DataFrame", len(all_agg_dfs))

if CSV_ENSEMBLE_MODE == "max":
    df = pd.concat(list(all_agg_dfs.values()), axis=0, ignore_index=True).reset_index(
        drop=True
    )
    df = (
        df.sort_values("conf", ascending=False)
        .groupby("tomo_id", as_index=False)
        .first()
    )
elif CSV_ENSEMBLE_MODE == "wbf":
    submission_df = SubmissionDataFrame()
    assert len(ALL_TOMO_IDS) == len(ALL_TOMO_SPACINGS)
    for tomo_idx, tomo_id in tqdm(enumerate(ALL_TOMO_IDS), desc="WBF"):
        tomo_zyxs = []
        tomo_confs = []
        tomo_labels = []
        weights = []
        for agg_name, agg_df in all_agg_dfs.items():
            preds = agg_df[agg_df["tomo_id"] == tomo_id][
                ["motor_z", "motor_y", "motor_x", "conf"]
            ].to_numpy()
            keep_idxs = np.all(preds[:, :3] != -1, axis=1) & (preds[:, -1] > 0.0)
            preds = preds[keep_idxs]
            tomo_zyxs.append(preds[:, :3])
            tomo_confs.append(preds[:, 3])
            tomo_labels.append([0] * len(preds))
            weights.append(all_unique_agg_weights[agg_name])
        # print(len(tomo_zyxs), len(tomo_confs), len(tomo_labels))
        # print(tomo_zyxs, tomo_confs, tomo_labels, sep = '\n\n')
        zyxs, confs, labels = weighted_boxes_fusion_3d(
            tomo_zyxs,
            tomo_confs,
            tomo_labels,
            weights=weights,
            dist_thr=1000.0 / ALL_TOMO_SPACINGS[tomo_idx],
            skip_box_thr=WBF_CONF_THRES,
            conf_type=WBF_CONF_TYPE,
            allows_overflow=False,
            rescale_conf=True,
        )
        if len(zyxs):
            # select the highest conf
            select_idx = np.argmax(confs)
            z, y, x = zyxs[select_idx]
            submission_df.add_row(tomo_id, x, y, z, confs[select_idx])
        else:
            submission_df.add_row(tomo_id, -1, -1, -1, 0.0)
    df = submission_df.to_pandas(submit=False)
else:
    raise ValueError

logger.info(
    "PREDICTION DATAFRAME WITH SHAPE %s COLUMNS=%s:\n%s",
    df.shape,
    list(df.columns),
    df,
)

if MODE in ["LOCAL", "KAGGLE_VAL"]:
    assert (
        len(GT_DF) == len(df) == len(ALL_TOMO_IDS)
    ), f"{GT_DF.shape} {df.shape} {len(ALL_TOMO_IDS)}"
    metrics = compute_metrics(GT_DF, df, mode="kaggle")
    logger.info("%s METRICS:\n%s", MODE, metrics)

# Filter based on conf
if QUANTILE_THRES is None:
    filter_thres = DECODE_CONF_THRES
else:
    assert 0 <= QUANTILE_THRES <= 1.0
    pred_confs = df["conf"].values
    filter_thres = np.quantile(pred_confs, QUANTILE_THRES)
    # assert 0 <= filter_thres <= 1.0


logger.info("USING FILTER THRESHOLD: %f", filter_thres)
df.loc[df["conf"] < filter_thres, ["motor_z", "motor_y", "motor_x"]] = -1
if MODE in ["LOCAL", "KAGGLE_VAL"]:
    kaggle_fbeta, kaggle_precision, kaggle_recall, (tn, fp, fn, tp) = kaggle_score(
        GT_DF, df
    )
    logger.info(
        "--------------\nKAGGLE METRICS AT THRESHOLD=%f:\n\nFBETA=%f PRECISION=%f RECALL=%f\nTN=%d FP=%d FN=%d TP=%d--------------\n\n",
        filter_thres,
        kaggle_fbeta,
        kaggle_precision,
        kaggle_recall,
        tn,
        fp,
        fn,
        tp,
    )

df.rename(
    columns={
        "motor_z": "Motor axis 0",
        "motor_y": "Motor axis 1",
        "motor_x": "Motor axis 2",
    },
    inplace=True,
)
df = df[["tomo_id", "Motor axis 0", "Motor axis 1", "Motor axis 2"]]
logger.info("SUBMISSION DATAFRAME:\n%s", df)

if MODE == "KAGGLE":
    logger.warning(f"CLEARING ALL CONTENTS IN %s", WORKING_DIR)
    clear_dir_content(WORKING_DIR)

submission_csv_path = os.path.join(WORKING_DIR, "submission.csv")
df.to_csv(submission_csv_path, index=False)





































