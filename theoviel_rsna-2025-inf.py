EVAL = False
DEBUG = False

import os
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    EVAL = False
    DEBUG = False


import os
import gc
import cv2
import sys
import glob
import json
import torch
import joblib
import shutil
import numpy as np
import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from scipy.special import expit
from collections import Counter, defaultdict

pd.set_option('max_colwidth', 400)
pd.set_option('display.max_columns', 400)


if os.path.exists("/kaggle/input/rsna-2025-pkgs/"):
    !pip install /kaggle/input/rsna-2025-pkgs/segmentation_models_pytorch-0.5.0-py3-none-any.whl --no-deps
    !pip install /kaggle/input/rsna-2025-pkgs/dicomsdl-0.109.4-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-deps


sys.path.append("/kaggle/input/rsna-intracranial-aneurysm-code/src")

from params import CLASSES_SEG, CLASSES

from data.processing import process, ct_windowing
from data.utils import get_largest_component_bbox, to_axial

from model_zoo.models_lvl2 import define_model as define_model_lvl2
from model_zoo.models_seg import define_model as define_model_seg
from model_zoo.models import define_model

from inference.lvl2 import preprocess_features, resize_fts_torch
from inference.utils import predict_modality, preprocessing_modality
from inference.extract_features import preprocess_images
from inference.processing import load_metadata, load_frame

from util.logger import Config
from util.torch import load_model_weights
from util.plots import plot_sample, show_label_colors, plot_features
from util.metrics import *


USE_FP16 = True
DEVICE = "cuda"
NUM_WORKERS = os.cpu_count()

FOLD = 0  # if DEBUG else "fullfit_0"


WINDOW = False
SEG_SIZE = (128, 128, 128)

EXP_FOLDER_SKULL = "/kaggle/input/rsna-2025-weights-1/2025-09-12_25/"  # TotalSegmentor r18
SKULL_TH = 0.5


config_skull = Config(json.load(open(EXP_FOLDER_SKULL + "config.json", "r")))

skull_model = define_model_seg(
    config_skull.decoder_name,
    config_skull.name,
    num_classes=config_skull.num_classes,
    num_classes_aux=config_skull.num_classes_aux,
    increase_stride=config_skull.increase_stride,
    use_cls=config_skull.use_cls,
    n_channels=config_skull.n_channels,
    use_3d=config_skull.use_3d,
    pretrained=False
).to(DEVICE)

fold = 0
skull_model = load_model_weights(skull_model, EXP_FOLDER_SKULL + f"{config_skull.name}_{fold}.pt")
skull_model = skull_model.to(DEVICE).eval()


# IMG_EXP_FOLDERS = [        
#     ("img", "/kaggle/input/rsna-2025-weights-1/2025-10-03_6/"),   # maxvit_rmlp_base coroT - 0.872
#     ("img_s", "/kaggle/input/rsna-2025-weights-1/2025-10-03_5/")  # coatnet2 384 ext  - 0.864
# ]

# IMG_EXP_FOLDERS = [  # 0.884 
#     ("img", "/kaggle/input/rsna-2025-weights-1/2025-10-07_5/"),   # maxvit_rmlp_base
#     ("img_s", "/kaggle/input/rsna-2025-weights-1/2025-10-07_6/")  # coatnet2 384
# ]

IMG_EXP_FOLDERS = [        
    ("img", "/kaggle/input/rsna-2025-weights-1/2025-10-10_2/"),   # maxvit_rmlp_base
    ("img", "/kaggle/input/rsna-2025-weights-1/2025-10-10_3/"),   # coatnet2 384
    ("img", "/kaggle/input/rsna-2025-weights-1/2025-10-11_10/"), 
]

FOLDS = [0]


img_models = defaultdict(list)

FOLDS = [0] if DEBUG or EVAL else ['fullfit_0']

for mode, exp_folder in IMG_EXP_FOLDERS:
    config = Config(json.load(open(exp_folder + "config.json", "r")))

    for fold in FOLDS:
        model = define_model(
            config.name,
            num_classes=config.num_classes,
            num_classes_aux=config.num_classes_aux,
            n_channels=config.n_channels,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
            pooling=config.pooling if hasattr(config, "pooling") else "avg",
            img_size=config.resize[0],
            head_3d=config.head_3d,
            delta=config.delta if hasattr(config, "delta") else 2,
            n_frames=config.n_frames,
            use_mask=config.use_mask,
            pretrained=False,
            verbose=0,
        )
        model = model.to(DEVICE).eval()

        weights = exp_folder + f"{config.name}_{fold}.pt"
        model = load_model_weights(model, weights, verbose=1)
        img_models[(mode, exp_folder)].append(model)


# LVL2_EXP_FOLDERS = [
#     "/kaggle/input/rsna-2025-weights-1/2025-10-10_5/"
# ]

# LVL2_EXP_FOLDERS = ["../logs/2025-10-10/5/"]

LVL2_EXP_FOLDERS = [
    # 2 models - 0.891
    # "/kaggle/input/rsna-2025-weights-1/2025-10-13_9/",  # CV 0.889 - RNN 2 models restrict 64 ax128
    # "/kaggle/input/rsna-2025-weights-1/2025-10-13_10/",  # CV 0.890  - Transfo 2 models ax128

    # 3 models - 0.895
    "/kaggle/input/rsna-2025-weights-1/2025-10-13_13/",  # CV 0.892 - RNN 3 models restrict 64 ax128
    "/kaggle/input/rsna-2025-weights-1/2025-10-13_14/",  # CV 0.894  - Transfo 3 models ax128
]

FOLDS = [0] if DEBUG or EVAL else [0, 1, 2, 3]


cp -r /kaggle/input/rsna-intracranial-aneurysm-code/src/configs ./


lvl2_models = []

for exp_folder in LVL2_EXP_FOLDERS:
    config_lvl2 = Config(json.load(open(exp_folder + "config.json", "r")))

    for fold in FOLDS:
        model = define_model_lvl2(
            config_lvl2.name,
            ft_dim=config_lvl2.ft_dim,
            layer_dim=config_lvl2.layer_dim,
            dense_dim=config_lvl2.dense_dim,
            p=config_lvl2.p,
            num_classes=config_lvl2.num_classes,
            num_classes_aux=config_lvl2.num_classes_aux,
            n_crop_fts=config_lvl2.n_crop_fts,
            layer=config_lvl2.layer,
            
        )
        model = model.to(DEVICE).eval()

        weights = exp_folder + f"{config_lvl2.name}_{fold}.pt"
        model = load_model_weights(model, weights, verbose=1)
        lvl2_models.append([config_lvl2, model])


def pipeline(series, data_path, plot=False, verbose=False):
    df_series = load_metadata(series, data_path)

    orientation = df_series['orientation'].values[0]
    spacing = df_series['slice_spacing'].values[0]
    modality = Counter(df_series['Modality']).most_common()[0][0]
    
    if verbose:
        print(f'Orientation: {orientation} - Spacing: {spacing} - Modality {modality}')

    # df_series = trim(df_series)
    # if spacing:
    #     if spacing < 0.5 and len(df_series) > 300:
    #         if verbose:
    #             print(f'Downsample series with spacing {spacing}')
    #         ids = np.arange(0, len(df_series), 2, dtype=int)
    #         spacing *= 2
    #         df_series = df_series.iloc[ids].reset_index(drop=True)

    images = joblib.Parallel(n_jobs=os.cpu_count())(
        joblib.delayed(load_frame)(
            path,
        ) for path in df_series['path'].values
    )
    images = np.array(images)
    if len(images.shape) == 4:
        images = images.squeeze(0)

    images = torch.from_numpy(images).to(DEVICE)  # Could move to GPU earlier ?

    if verbose:
        display(df_series.head(1))
        print('Image shape', images.shape)

    # Windowing
    try:
        if df_series["Modality"].values[0] == "CT":
            b = df_series["RescaleIntercept"].values[0]
            m = df_series["RescaleSlope"].values[0]
    
            if m is not None and b is not None:
                if m != 1 or b != 0:
                    min_ = images.min()
                    if min_ >= -100 or min_ == -2000:  # Ian fix
                        images = m * images + b
    
            # if WINDOW:  # Maybe later ??
            #     images = ct_windowing(images, mode="bone", normalize=False)    
    except:
        pass

    # if PLOT:
    #     plt.figure(figsize=(5, 5))
    #     plt.imshow(images.amax(0).cpu().numpy(), cmap="bone")
    #     plt.axis(False)
    #     plt.show()

    ###################
    # Skull inference #
    ###################

    # try:
    #     max_ = torch.quantile(images[::4, ::2, ::2].reshape(-1).float(), 0.99)
    # except RuntimeError:
    #     try:
    #         max_ = torch.quantile(images[::8, ::4, ::4].reshape(-1).float(), 0.99)
    #     except:
    #         max_ = 0
    # max_ = max_ if max_ > 0 else None

    images = torch.clamp(images, -1000, None)

    x = F.interpolate(
        images.unsqueeze(0).unsqueeze(0).float(),
        SEG_SIZE,
        mode="trilinear",
    )
    # x = torch.clamp(x, -1000, max_)  # NEW
    x = (x - x.min()) / (x.max() - x.min())

    with torch.inference_mode():
        with torch.amp.autocast("cuda", enabled=True):
            pred = skull_model(x)[0]
            mask_skull = F.interpolate(pred, images.shape, mode="trilinear")[0][0].sigmoid()
        
        th = SKULL_TH
        if mask_skull.max() < SKULL_TH:
            th = min(0.05, mask_skull.max() / 2)
        
        mask_skull = (mask_skull > th).cpu().numpy().astype(np.uint8)

        # TODO: criterion os mask_skull size to filter outliers

    (zmin, zmax), (ymin, ymax), (xmin, xmax) = get_largest_component_bbox(mask_skull)
    images_crop = images[zmin:zmax, ymin:ymax, xmin:xmax]

    if verbose:
        print(f'Skull crop : [{zmin}; {zmax}]')
        print('Crop size', images_crop.size())

    if plot:
        plt.figure(figsize=(5, 5))
        plt.imshow(images_crop[len(images_crop) // 2].cpu().numpy(), cmap="bone")
        plt.axis(False)
        plt.show()       
        
    # images_crop = torch.from_numpy(
    #     np.load(os.path.join('../input/processed/skull_crops/', series + ".npy"))
    # ).to(DEVICE)

    ################
    # Image models #
    ################
    images_crop = images_crop[:MAX_STACK_SIZE]

    if orientation == "sagittal":
        imgs = to_axial(
            images_crop.clone().float(),
            orientation,
            resize=None,
            fixed=True,
        )
        imgs = F.interpolate(
            imgs.unsqueeze(0).unsqueeze(0), (128, 512, 512), mode="trilinear"
        )[0, 0]  # should be enough
        spacing = 1
    else:
        imgs = images_crop.clone().float()

    x, masks = preprocess_images(
        imgs, modality, config, masks=None, spacing=spacing
    )
    
    x = x[:MAX_STACK_SIZES[orientation]]

    if verbose:
        print('Input size', x.size())

    if len(x) > RESTRICT_SIZE:
        x = x[::2].clone()

    if len(x) > RESTRICT_SIZE * 2:
        x = x[::2].clone()

    if verbose:
        print('Restricted input size', x.size())

    image_model_preds = defaultdict(list)
    for (mode, exp_folder) in img_models:
        batches = np.array_split(np.arange(x.size(0)), (x.size(0) // BATCH_SIZE) + 1)

        p = []
        for model in img_models[(mode, exp_folder)]:
            with torch.inference_mode():
                with torch.amp.autocast("cuda", enabled=USE_FP16):
                    preds = torch.cat(
                        [model(x[b], mask=None)[0].detach() for b in batches],
                        axis=0
                    )
                    p.append(preds)

        if "kaggle" in exp_folder:
            k = "../logs/" + '/'.join(exp_folder.split('/', 4)[-1].split('_'))
        else:
            k = exp_folder
        image_model_preds[(mode, k)] = torch.stack(p, 0).mean(0)
    
    if plot:
        for k in image_model_preds:
            plot_features(image_model_preds[k].cpu().sigmoid(), figsize=(10, 5))

    ###############
    # Lvl2 models #
    ###############

    lvl2_preds = []
    for cfg, model in lvl2_models:
        # create input
        fts_dict = preprocess_features(cfg, image_model_preds)

        # for k in fts_dict:
        #     try:
        #         plot_features(fts_dict[k].cpu().sigmoid()[0], figsize=(10, 5))
        #     except:
        #         continue

        # for k in fts_dict:
        #     print(k, fts_dict[k].mean(), fts_dict[k].max(), fts_dict[k].size())

        with torch.inference_mode():
            p = model(fts_dict)[0][0].sigmoid().cpu().numpy()
            lvl2_preds.append(p)
    lvl2_preds = np.mean(lvl2_preds, 0)

    if verbose:
        print('\n-> Preds:\n')
        for i, c in enumerate(CLASSES):
            print(f'{c} : {lvl2_preds[i] :.3f}')

    return lvl2_preds


WINDOW = False
USE_FP16 = True

MAX_STACK_SIZES = {"axial": 300, "sagittal": 100, "coronal": 75}
MAX_STACK_SIZE = 300
RESTRICT_SIZE = 64

BATCH_SIZE = 32


df = pd.read_csv(LVL2_EXP_FOLDERS[0] + 'df_val_0.csv')[['SeriesInstanceUID', "Modality"]]
y = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv')[["SeriesInstanceUID"] + CLASSES]
df = df.merge(y, on="SeriesInstanceUID")

df.head(1)


if DEBUG:
    DATA_PATH = "/kaggle/input/rsna-intracranial-aneurysm-detection/series/"
    
    idx = np.random.randint(len(df))
    # idx = 4
    series = df['SeriesInstanceUID'].values[idx]

    # series = '1.2.826.0.1.3680043.8.498.72399220582146674554688600955035038848'
    # series = '1.2.826.0.1.3680043.8.498.62690866817595040372272270948444876445'
    # idx = df['SeriesInstanceUID'].values.tolist().index(series)

    preds = pipeline(series, DATA_PATH, plot=False, verbose=True)

    gc.collect()
    torch.cuda.empty_cache()

    pred_val = np.mean([
        np.load(e + "pred_val_0.npy") for e in LVL2_EXP_FOLDERS
    ], 0)
    delta = np.max(np.abs(pred_val[idx] - preds))

    print()
    if delta > 0.01:
        for i, c in enumerate(CLASSES):
            d = np.abs(pred_val[idx][i] - preds[i])
            print(f'{"❌" if d > 0.01 else "✅"} {c} : {preds[i] :.3f}  (delta={d:.2f})')
    else:
        print('✅ Predictions ok !')


if EVAL:
    DATA_PATH = "/kaggle/input/rsna-intracranial-aneurysm-detection/series/"
    N = 200

    preds = []
    for series in tqdm(df['SeriesInstanceUID'].values[:N]):
        p = pipeline(series, DATA_PATH)
        preds.append(p)
    preds = np.array(preds)

    df.head(N).to_csv('df_val.csv', index=False)


if EVAL:
    try:
        try:
            aucs, auc_w = weighted_multilabel_auc(
                df[CLASSES].values[:N],
                preds,
                average=False
            )
    
            print()
            for i in range(len(aucs)):
                print(f"-  AUC:  {aucs[i]:.3f}  -  {CLASSES[i]}")
            print(f"\n -> CV AUC : {auc_w:.3f}\n")
        except:
            aucs, auc_w = weighted_multilabel_auc(
                df[CLASSES].values[:N][:, -1:],
                preds[:, -1:],
                average=False
            )
            print(f"\n -> Aneurysm Present AUC : {auc_w:.3f}\n")
    except:
        pass


if EVAL:
    # Reference Val score
    pred_val = np.mean([
        np.load(e + "pred_val_0.npy") for e in LVL2_EXP_FOLDERS
    ], 0)
    
    try:
        try:
            aucs, auc_w = weighted_multilabel_auc(
                df[CLASSES].values[:N],
                pred_val[:N],
                average=False
            )
    
            print()
            for i in range(len(aucs)):
                print(f"-  AUC:  {aucs[i]:.3f}  -  {CLASSES[i]}")
            print(f"\n -> CV AUC : {auc_w:.3f}\n")
        except:
            aucs, auc_w = weighted_multilabel_auc(
                df[CLASSES].values[:N][:, -1:],
                pred_val[:N, -1:],
                average=False
            )
            print(f"\n -> Aneurysm Present AUC : {auc_w:.3f}\n")
    except:
        pass


def predict(series_path: str) -> pl.DataFrame | pd.DataFrame:
    """
    Make a prediction.
    """
    series_id = os.path.basename(series_path)
    folder = series_path.rsplit('/', 1)[0] + "/"

    preds = pipeline(series_id, folder, plot=False, verbose=False)

    gc.collect()
    torch.cuda.empty_cache()

    predictions = pl.DataFrame(
        data=[list(preds)],
        schema=CLASSES,
        orient='row',
    )
    shutil.rmtree('/kaggle/shared', ignore_errors=True)
    return predictions


%%time
import kaggle_evaluation.rsna_inference_server

shutil.rmtree('/kaggle/shared', ignore_errors=True)
inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    if os.path.exists('df_val.csv'):
        data_paths=('df_val.csv', "/kaggle/input/rsna-intracranial-aneurysm-detection/series/")
    else:
        data_paths = None
    print(data_paths)
    inference_server.run_local_gateway(data_paths=data_paths)
    display(pd.read_parquet('/kaggle/working/submission.parquet'))

