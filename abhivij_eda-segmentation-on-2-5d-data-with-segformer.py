!pip install /kaggle/input/git-whl-files/segmentation_models_pytorch-0.5.0-py3-none-any.whl --no-deps
!pip install /kaggle/input/git-whl-files/monai-1.5.1-py3-none-any.whl --no-deps


import numpy as np
import pandas as pd

import os
import random
import re
import sys
import time
import gc
import threading, psutil, pynvml
import itertools
import warnings

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
import seaborn as sns

from glob import glob

from tqdm.notebook import tqdm
tqdm.pandas()

from sklearn.model_selection import StratifiedGroupKFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import sigmoid_focal_loss
from torchvision.transforms import ToTensor
from torch.amp import autocast, GradScaler

import albumentations as A

import segmentation_models_pytorch as smp

from monai.metrics.utils import get_mask_edges, get_surface_distance

from timm.utils import ModelEmaV2

from scipy.ndimage import distance_transform_edt


print(f"Number of available CPUs: {os.cpu_count()}")
print(f"Number of available GPUs: {torch.cuda.device_count()}")


DIR_PATH = '/kaggle/input/uw-madison-gi-tract-image-segmentation/'

pd.set_option('display.max_colwidth', 400) 

CMAP1 = ListedColormap([[0, 0, 0, 0], [1, 0, 0, 0.5]])  # black transparent, red semi-transparent
CMAP2 = ListedColormap([[0, 0, 0, 0], [0, 1, 0, 0.5]])  # black transparent, green semi-transparent
CMAP3 = ListedColormap([[0, 0, 0, 0], [0, 0, 1, 0.5]])  # black transparent, blue semi-transparent

RANDOM_SEED = 0

IMAGE_RESIZE = [384, 384]
D_MAX = (IMAGE_RESIZE[0]**2 + IMAGE_RESIZE[1]**2) ** 0.5

BATCH_SIZE_TRAIN = 16
BATCH_SIZE_VALID = 16
BATCH_SIZE_TEST = 64

GRAD_ACCUM_STEPS = 4

SLICE_STRIDE = 1
IMAGE_CHANNELS = 5   # num of slices combined to create the 2.5D image
IMAGE_OFFSETS = [-SLICE_STRIDE*2, -SLICE_STRIDE, 0, SLICE_STRIDE, SLICE_STRIDE*2]  # assign this in accordance with IMAGE_CHANNELS
NUM_CLASSES = 3
CLASS_NAMES = ['large_bowel', 'small_bowel', 'stomach']

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_SD = (0.229, 0.224, 0.225)

IMAGE_NORMALIZE_MEAN = [np.mean(IMAGENET_MEAN)] * IMAGE_CHANNELS
IMAGE_NORMALIZE_SD = [np.mean(IMAGENET_SD)] * IMAGE_CHANNELS

DATA_LOADER_NUM_WORKERS = 4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS = 15
LR_START = 1e-4
LR_END = 2e-4
LR_WARMUP_EPOCHS = 4

PRED_LABEL_CUTOFFS = [0.5, 0.5, 0.5]
PRED_MASK_CUTOFFS = [0.5, 0.5, 0.5]

TRAIN_VALID_SPLIT = False
SAVE_TRAIN_VALID_MODEL = False
MODEL_PARAMS_FILE_NAME = 'GIT-Seg-segformer-mitb4.pth'

EXPLORE_CUTOFFS = False
CUTOFFS_LIST = [
	(0.0, 0.0, 0.0),
	(0.25, 0.25, 0.25),
	(0.5, 0.5, 0.5),
	(0.75, 0.75, 0.75),
	(0.6, 0.7, 0.7),
	(0.6, 0.7, 0.8),
	(0.5, 0.75, 0.75)
]

TRAIN_ON_FULL_DATA = False
SAVE_FULL_DATA_MODEL = False
MODEL_PARAMS_FULL_DATA_FILE_NAME = 'GIT-Seg-fulldata-segformer-mitb4.pth'

TEST_PREDICT = True
LOAD_MODEL_FOR_TEST_PREDICT = True
MODEL_PARAMS_LOAD_FILE_PATH = '/kaggle/input/models/abhivij/git-seg/pytorch/segformer-mit-b4/38/EMA-Epoch12-GIT-Seg-segformer-mitb4.pth'

ENSEMBLE_MODEL_PREDICT = False
ENSEMBLE_PRED_LABEL_CUTOFFS = []
ENSEMBLE_MODEL_PARAMS_LOAD_FILE_PATH = []

SAVE_MASKS = False
LOAD_SAVED_MASKS = True

SAVE_RESIZED = False
LOAD_RESIZED = False

MASK_DATASET_ROOT = '/kaggle/input/git-seg-mask/'

RESIZED_MASK_DATASET_ROOT = '/kaggle/input/git-seg-resized-dataset/mask_256/'
RESIZED_IMAGE_DATASET_ROOT = '/kaggle/input/git-seg-resized-dataset/image_256/'


# ensure reproducibility(to some extent) across different runs
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


data = pd.read_csv(DIR_PATH + 'train.csv')
data.head()


data_nonnaseg = data.loc[data.segmentation.notna(), :]
data_nonnaseg.head()


data[['case', 'day', 'slice']] = data['id'].str.extract(r'case(\d+)_day(\d+)_slice_(\d+)')
data


# The image file corresponding to case123_day20_slice_0065 is train/case123/case123_day20/scans/slice_0065_266_266_1.50_1.50.png 
# 266, 266 are slice width, slice height and 1.5, 1.5 are pixel width, pixel height.

def get_path_df(train = True):
    if train:
        paths = glob(DIR_PATH + 'train/*/*/*/*')
    else:
        paths = glob(DIR_PATH + 'test/*/*/*/*')
    path_df = pd.DataFrame(paths, columns=['image_path'])
    path_df[['case', 'day', 'slice', 
             'slice_w', 'slice_h', 
             'px_w', 'px_h']] = \
            path_df.image_path.str.extract(r'.*/case(\d+)_day(\d+)/scans/slice_(\d+)_(\d+)_(\d+)_(\d+\.\d+)_(\d+\.\d+)\.png')
    
    return path_df

path_df = get_path_df()


data.info()


path_df.info()


data = data.merge(path_df, on = ['case', 'day', 'slice'])
data


data.info()


data.px_w.unique(), data.px_h.unique()


data.case.unique(), data.day.unique(), data.slice.unique(), data.slice_w.unique(), data.slice_h.unique()


int_cols = ['case', 'day', 'slice', 'slice_w', 'slice_h']
data[int_cols] = data[int_cols].astype(np.uint32)

float_cols = ['px_w', 'px_h']
data[float_cols] = data[float_cols].astype(np.float32)

data.info()


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = np.asarray(mask_rle.split(), dtype=int)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction


# ref: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def dict_size(d):
    size = sys.getsizeof(d)  # dict container itself
    for k, v in d.items():
        size += sys.getsizeof(k) + sys.getsizeof(v)
    return size

id_to_impath = dict(zip(data.id, data.image_path))
print('id_to_impath size:', dict_size(id_to_impath) / (1024*1024), 'MB')

id_dicts = {'impath': id_to_impath}

if not LOAD_SAVED_MASKS or SAVE_MASKS:
    id_to_shape = dict(zip(data['id'], zip(data['slice_h'], data['slice_w'])))
    idclass_to_rle = {
        (id_, class_): seg
        for id_, class_, seg in zip(data.id, data['class'], data.segmentation)
        if pd.notna(seg)
    }
    print('id_to_shape size:', dict_size(id_to_shape) / (1024*1024), 'MB')
    print('idclass_to_rle size:', dict_size(idclass_to_rle) / (1024*1024), 'MB')

    id_dicts['shape'] = id_to_shape
    id_dicts['rle'] = idclass_to_rle
    
# id_to_impath size: 9.673919677734375 MB
# id_to_shape size: 5.618408203125 MB
# idclass_to_rle size: 24.884278297424316 MB


def get_mask(id_, id_dicts, load_resized=False, load_saved_masks=LOAD_SAVED_MASKS):
    '''
    id_dicts : dict of id_mapping dicts - allowed keys : impath, shape, rle
    load_resized : if True, then precreated resized masks will be loaded - used while training the model
    load_saved_masks : if True, then precreated masks will be loaded
    '''
    if load_resized or load_saved_masks:
        mask_dataset_root = RESIZED_MASK_DATASET_ROOT if load_resized else MASK_DATASET_ROOT
        id_to_impath = id_dicts['impath']
        mask_path = mask_dataset_root + os.path.relpath(id_to_impath[id_], DIR_PATH)
        mask_path = os.path.splitext(mask_path)[0] + '.npy'
        mask = np.load(mask_path)
    else:
        id_to_shape, idclass_to_rle = id_dicts['shape'], id_dicts['rle']
        h, w = id_to_shape[id_]
        shape = (h, w, NUM_CLASSES)
        mask = np.zeros(shape, dtype=np.uint8)
        for i, class_ in enumerate(CLASS_NAMES):
            rle = idclass_to_rle.get((id_, class_))
            if rle:
                mask[..., i] = rle_decode(rle, shape[:2])
    return mask


full_image_file_path = DIR_PATH + 'train/case123/case123_day20/scans/slice_0065_266_266_1.50_1.50.png'

img = cv2.imread(full_image_file_path, cv2.IMREAD_UNCHANGED)
# default imread mode is IMREAD_COLOR which expects 8-bit 3 channel image, our input image is 16-bit grayscale which requires IMREAD_UNCHANGED

print(img.shape)
print(img)

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Gray')
plt.axis('off')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(img, cmap='bone')
plt.title('Bone')
plt.axis('off')
plt.colorbar()

# while gray cmap is technically most correct for 16-bit grayscale image, using bone cmap from now on to enhance contrast visually

plt.tight_layout()
plt.show()


img = cv2.imread(full_image_file_path, cv2.IMREAD_UNCHANGED).astype('float32')

img_norm = img
mx = np.max(img)
if mx > 0:
    img_norm /= mx

print(img_norm)
print(img_norm.shape)
print(max([max(r) for r in img_norm]))

img_norm = (img_norm*255).astype(np.uint8)
print(img_norm)
print(img_norm.shape)
print(max([max(r) for r in img_norm]))

clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
clahe3 = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2,2))

res1 = clahe1.apply(img_norm)
res2 = clahe2.apply(img_norm)
res3 = clahe3.apply(img_norm)

print(res1)
print(res1.shape)
print(max([max(r) for r in res1]))

print(res2)
print(res2.shape)
print(max([max(r) for r in res2]))

# Show results
plt.figure(figsize=(20, 4))
for i, (title, im) in enumerate(zip(['Original', 'Normalized', 'CLAHE clip=2 grid=8x8', 'CLAHE clip=2 grid=2x2', 'CLAHE clip=1 grid=2x2'], [img, img_norm, res1, res2, res3])):
    plt.subplot(1,5,i+1)
    plt.imshow(im, cmap='bone')
    plt.title(title)
    plt.colorbar()
    plt.axis('off')
plt.tight_layout()
plt.show()


def load_image(id_, id_to_impath, load_resized=False):
    '''
    load_resized : if True, then precreated resized images will be loaded - used while training the model
    '''
    if load_resized:    
        image_path = RESIZED_IMAGE_DATASET_ROOT + os.path.relpath(id_to_impath[id_], DIR_PATH)
        image_path = os.path.splitext(image_path)[0] + '.npy'
        image = np.load(image_path)
    else:
        image = cv2.imread(id_to_impath[id_], cv2.IMREAD_UNCHANGED).astype('float32')  # convert from original 16-bit
        #print(f'Raw image {id_} min : {np.min(img)} max : {np.max(img)}')
        mx = np.max(image)
        if mx > 0:
            image /= mx
    return image


def insert_padding(image_or_mask, border_value = 0):
	h, w = image_or_mask.shape[:2]
	post_pad_size = max(h, w)
	pad_h = post_pad_size - h
	pad_w = post_pad_size - w
	top = pad_h // 2
	bottom = pad_h - top
	left = pad_w // 2
	right = pad_w - left

	if image_or_mask.ndim == 3:
		pad_width = ((top, bottom), (left, right), (0, 0)) #no padding needed for channels
	else:
		pad_width = ((top, bottom), (left, right))

	padded_image_or_mask = np.pad(
		image_or_mask, 
		pad_width = pad_width,
		mode = 'constant',
		constant_values = border_value
	)

	return padded_image_or_mask, (top, bottom, left, right)


def undo_padding(image_or_mask, pad):
	top, bottom, left, right = pad
	h, w = image_or_mask.shape[:2]
	unpadded_image_or_mask = image_or_mask[top:h-bottom, left:w-right, ...]
	return unpadded_image_or_mask


#try 266x266 image/mask
img = load_image('case44_day20_slice_0101', id_to_impath)
print(img.shape)
p_img, pad = insert_padding(img)
print(p_img.shape, pad)
print(undo_padding(p_img, pad).shape)

mask = get_mask('case44_day20_slice_0101', id_dicts)
print(mask.shape)
p_mask, pad = insert_padding(mask)
print(p_mask.shape, pad)
print(undo_padding(p_mask, pad).shape)

print('----------------')

#try 360x310(wxh) image/mask 
# Note - numpy.shape returns (h, w)
img = load_image('case89_day0_slice_0079', id_to_impath)
print(img.shape)
p_img, pad = insert_padding(img)
print(p_img.shape, pad)
print(undo_padding(p_img, pad).shape)

mask = get_mask('case89_day0_slice_0079', id_dicts)
print(mask.shape)
p_mask, pad = insert_padding(mask)
print(p_mask.shape, pad)
print(undo_padding(p_mask, pad).shape)


def display_image(id_, id_dicts, pred_mask=None, apply_CLAHE=False,
                  show_orig_img=True, show_true_mask=True, show_pred_mask=False,
                  transforms=None, replay_transforms=None, caseday_replay_transform=None,
                  insert_pad=False, undo_pad=False, pad=None
                 ):
    
    img = load_image(id_, id_dicts['impath'])
    #print(f'load_image result {id_} min : {np.min(img)} max : {np.max(img)}')
    img = (img * 255).astype(np.uint8) # 0-255 range required for CLAHE. 
                                       # Using this in general to maintain consistency with the case where CLAHE is required
    #print(f'just before CLAHE {id_} min : {np.min(img)} max : {np.max(img)}')
    if apply_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2,2))
        img = clahe.apply(img)
    
    mask = get_mask(id_, id_dicts)

    if insert_pad:
        img, _ = insert_padding(img)
        mask, _ = insert_padding(mask)

    if undo_pad:
        img = undo_padding(img, pad)
        mask = undo_padding(mask, pad)
    
    if transforms is not None:
        augmented1 = transforms(image=img, mask=mask)
        if show_pred_mask and pred_mask is not None:
            augmented2 = transforms(image=img, mask=pred_mask)
            pred_mask = augmented2['mask']
        img, mask = augmented1['image'], augmented1['mask']

    if replay_transforms is not None and caseday_replay_transform is not None:
        caseday_str, _ = id_.rsplit('_slice_', 1)
        augmented1 = A.ReplayCompose.replay(caseday_replay_transform[caseday_str], image=img, mask=mask)
        if show_pred_mask and pred_mask is not None:
            augmented2 = A.ReplayCompose.replay(caseday_replay_transform[caseday_str], image=img, mask=pred_mask)
            pred_mask = augmented2['mask']
        img, mask = augmented1['image'], augmented1['mask']
            
    plt.figure(figsize=(9, 3))
    
    i = 1
    if show_orig_img:
        plt.subplot(1, 3, i)
        i += 1
        #print(f'just before imshow {id_} min : {np.min(img)} max : {np.max(img)}')
        plt.imshow(img, cmap='bone')
        plt.title(f'{id_} image')
        plt.axis('off')

    if show_true_mask:
        plt.subplot(1, 3, i)
        i += 1
        print(f'just before imshow {id_} min : {np.min(img)} max : {np.max(img)}')
        print(img.shape)
        plt.imshow(img, cmap='bone')
        if transforms is not None:
            plt.title(f"{'Augmented image with true mask':<30}")
        else:
            plt.title(f"{'Image with true mask':<30}")
        plt.imshow(mask[..., 0], cmap=CMAP1)
        plt.imshow(mask[..., 1], cmap=CMAP2)
        plt.imshow(mask[..., 2], cmap=CMAP3)
        
        handles = [
            Rectangle((0, 0), 1, 1, color=CMAP1(1.0)),
            Rectangle((0, 0), 1, 1, color=CMAP2(1.0)),
            Rectangle((0, 0), 1, 1, color=CMAP3(1.0))
        ]
        labels = ['Large Bowel', 'Small Bowel', 'Stomach']
        plt.axis('off')
        plt.legend(handles, labels, bbox_to_anchor=(1.0, -0.4), loc='lower right', borderaxespad=0.)
    
    if show_pred_mask and pred_mask is not None:
        plt.subplot(1, 3, i)
        plt.imshow(img, cmap='bone')
        plt.title(f"{'Image with predicted mask':<30}")
        plt.imshow(pred_mask[..., 0], cmap=CMAP1)
        plt.imshow(pred_mask[..., 1], cmap=CMAP2)
        plt.imshow(pred_mask[..., 2], cmap=CMAP3)
        
        handles = [
            Rectangle((0, 0), 1, 1, color=CMAP1(1.0)),
            Rectangle((0, 0), 1, 1, color=CMAP2(1.0)),
            Rectangle((0, 0), 1, 1, color=CMAP3(1.0))
        ]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
        plt.axis('off')
        plt.legend(handles, labels, bbox_to_anchor=(1.0, -0.4), loc='lower right', borderaxespad=0.)
    
    
    plt.tight_layout()
    plt.show()  


display_image('case131_day0_slice_0066', id_dicts)


display_image('case131_day0_slice_0066', id_dicts, apply_CLAHE=True)


# # testing pred_mask display using true mask
# display_image('case131_day0_slice_0066', id_dicts, pred_mask=get_mask('case131_day0_slice_0066', id_dicts),
#               show_pred_mask=True, apply_CLAHE=True)


# example image with only stomach segment
display_image('case123_day20_slice_0065', id_dicts, apply_CLAHE=True)


# example image without any segment
display_image('case123_day20_slice_0001', id_dicts, apply_CLAHE=True)


display_image('case89_day0_slice_0079', id_dicts)
display_image('case89_day0_slice_0079', id_dicts, insert_pad=True)
img = load_image('case89_day0_slice_0079', id_to_impath)
print(img.shape)
p_img, pad = insert_padding(img)
display_image('case89_day0_slice_0079', id_dicts, insert_pad=True, undo_pad=True, pad=pad)


display_image('case44_day20_slice_0101', id_dicts)
display_image('case44_day20_slice_0101', id_dicts, insert_pad=True)
img = load_image('case44_day20_slice_0101', id_to_impath)
print(img.shape)
p_img, pad = insert_padding(img)
display_image('case44_day20_slice_0101', id_dicts, insert_pad=True, undo_pad=True, pad=pad)


def display_multiple_slices(id_array, id_dicts, apply_CLAHE=False,
                            show_pred_mask=False, pred_mask_array=None):

    '''
    id_array : an array of ids like case123_day20_slice_0001
    id_dicts : dict of id_mapping dicts - allowed keys : impath, shape, rle
    apply_CLAHE : whether or not to apply CLAHE
    show_pred_mask : if this parameter is False, then true mask will be shown
                     if it is True, masks from pred_mask_array will be shown
    pred_mask_array : array of prediction masks
    '''

    l = len(id_array)
    rows = np.ceil(l/5).astype(int)
    max_cols = 5

    plt.figure(figsize=(max_cols*3, rows*3))

    for i in range(l):

        id_ = id_array[i]
        
        img = cv2.imread(id_dicts['impath'][id_], cv2.IMREAD_UNCHANGED).astype('float32')
        mx = np.max(img)
        if mx > 0:
            img /= mx

        if apply_CLAHE:
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2,2))
            img = (img * 255).astype(np.uint8)
            img = clahe3.apply(img)

        if show_pred_mask and pred_mask_array is not None:
            mask = pred_mask_array[i]
        else:
            mask = get_mask(id_, id_dicts)

        plt.subplot(rows, max_cols, i+1)
        plt.imshow(img, cmap='bone')
        plt.title(id_)
        plt.imshow(mask[..., 0], cmap=CMAP1)
        plt.imshow(mask[..., 1], cmap=CMAP2)
        plt.imshow(mask[..., 2], cmap=CMAP3)
        plt.axis('off')

        if i == 0:
            handles = [
                Rectangle((0, 0), 1, 1, color=CMAP1(1.0)),
                Rectangle((0, 0), 1, 1, color=CMAP2(1.0)),
                Rectangle((0, 0), 1, 1, color=CMAP3(1.0))
            ]
            labels = ['Large Bowel', 'Small Bowel', 'Stomach']
        
            plt.legend(handles, labels, bbox_to_anchor=(0.0, 1.5), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    plt.show()  


# using the below data to visualize change in segmentation mask across different slices
# data.query("case == 123 and day == 20 and slice >= 63 and slice <= 70")
display_multiple_slices(data.query("case == 123 and day == 20 and slice >= 63 and slice <= 70").id.unique(), 
                        id_dicts, apply_CLAHE=True)


# using the below data to visualize change in segmentation mask across different slices - for a case where all masks are present
# data.query("case == 131 and day == 0 and slice > 60 and slice <= 70")
display_multiple_slices(data.query("case == 131 and day == 0 and slice > 55 and slice <= 70").id.unique(), 
                        id_dicts, apply_CLAHE=True)


display_multiple_slices(data.query("case == 7 and day == 0 and slice >= 85 and slice < 95").id.unique(), 
                        id_dicts, apply_CLAHE=True)


display_multiple_slices(data.query("case == 81 and day == 30 and slice >= 85 and slice < 95").id.unique(), 
                        id_dicts, apply_CLAHE=True)
display_image('case81_day30_slice_0085', id_dicts, apply_CLAHE=True)


display_multiple_slices(data.query("case == 138 and day == 0 and slice >= 100 and slice < 115").id.unique(), 
                        id_dicts, apply_CLAHE=True)


display_multiple_slices(data.query("case == 43 and day == 18 and slice >= 60 and slice < 75").id.unique(), 
                        id_dicts, apply_CLAHE=True)


# faults = ['case7_day0', 'case81_day30']
faults = ['case7_day0', 'case81_day30', 'case138_day0']

pattern = '|'.join(faults)
data = data[~data['id'].str.contains(pattern, regex=True)].reset_index(drop=True)


data.loc[data.segmentation.isna(), :]


# How many missing values
data.isna().sum()


print(f'Num cases : {len(data.case.unique())} \
        Num unique days : {len(data.day.unique())}   \
        Num unique slices : {len(data.slice.unique())}')


# proportion of different slice sizes
def display_slice_size_prop(data):
    count_df = data[['id', 'slice_w', 'slice_h']].drop_duplicates()[['slice_w', 'slice_h']].value_counts().reset_index(name='count')
    count_df['percent'] = count_df['count']*100 / sum(count_df['count'])
    print(sum(count_df['count']))
    display(count_df)

display_slice_size_prop(data)


# proportion of different pixel sizes
count_df = data[['id', 'px_w', 'px_h']].drop_duplicates()[['px_w', 'px_h']].value_counts().reset_index(name='count')
count_df['percent'] = count_df['count']*100 / sum(count_df['count'])
print(sum(count_df['count']))
count_df


day_dist = data[['case', 'day']].drop_duplicates()['case'].value_counts().reset_index(name='num_days')

display(day_dist)

sns.histplot(data=day_dist, x='num_days', bins=range(1, day_dist['num_days'].max() + 1), discrete=True)
plt.xlabel('Number of Days per Case')
plt.ylabel('Number of Cases')
plt.title('Distribution of Days per Case')
plt.show()


slice_dist = data[['case', 'day', 'slice']].drop_duplicates()[['case', 'day']].value_counts().reset_index(name='num_slices')
display(slice_dist)

sns.histplot(data=slice_dist, x='num_slices', bins=range(1, slice_dist['num_slices'].max() + 1), discrete=True)
plt.xlabel('Number of slices per case-days')
plt.ylabel('Number of specific case-days')
plt.title('Distribution of slices per case-day')
plt.show()

display(slice_dist.num_slices.value_counts())


slice_dist.loc[slice_dist.num_slices == 80, :]


case_day_slice_df = data[['case', 'day', 'slice', 'slice_w', 'slice_h']].drop_duplicates()
case_day_slice_df.merge(case_day_slice_df, on=['case', 'day']).query("(slice_w_x != slice_w_y) | (slice_h_x != slice_h_y)")


case_day_slice_df = data[['case', 'day', 'slice', 'slice_w', 'slice_h']].drop_duplicates()
case_day_slice_df.merge(case_day_slice_df, on=['case']).query("(slice_w_x != slice_w_y) | (slice_h_x != slice_h_y)")


case_day_slice_df = data[['case', 'day', 'slice', 'px_w', 'px_h']].drop_duplicates()
case_day_slice_df.merge(case_day_slice_df, on=['case', 'day']).query("(px_w_x != px_w_y) | (px_h_x != px_h_y)")


case_day_slice_df = data[['case', 'day', 'slice', 'px_w', 'px_h']].drop_duplicates()
case_day_slice_df.merge(case_day_slice_df, on=['case']).query("(px_w_x != px_w_y) | (px_h_x != px_h_y)")


num_missing_seg_masks = data.segmentation.isna().sum() 
print(f'Missing Seg Mask \n count = {num_missing_seg_masks}\n percentage = {num_missing_seg_masks/len(data)*100}')


data['class'].value_counts()


def seg_na_prop(data):
    na_counts = (
        data.groupby('class')['segmentation']
        .apply(lambda s: s.isna().sum())
        .reset_index(name='count')
    )
    na_counts['percent'] = 100 * na_counts['count'] / data.groupby('class')['segmentation'].size().values
    return na_counts

na_counts = seg_na_prop(data)
display(na_counts)

sns.set_style("whitegrid") 
ax = sns.barplot(data=na_counts, x='class', y='percent', palette=[CMAP1(1.0), CMAP2(1.0), CMAP3(1.0)])

for i, row in na_counts.iterrows():
    ax.text(i, row['percent'] + 1,  # position just above the bar
            f"{row['percent']:.2f}% ({row['count']})",
            ha='center', va='bottom', fontsize=10)

plt.ylabel('Percentage')
plt.xlabel('Segmentation Class')
plt.title('Missing Segmentation Masks')
plt.yticks(range(0, 105, 10))
plt.show()


case_day_seg_missing = (
     data[['case', 'day', 'class', 'segmentation']]
     .groupby(['case', 'day', 'class'])['segmentation']
     .apply(lambda s: s.isna().sum())
     .reset_index(name='count').sort_values(by='count', ascending=False)
)
display(case_day_seg_missing)


# sns.set_style("whitegrid") 
# ax = sns.barplot(data=na_counts, x='class', y='percent', palette=[CMAP1(1.0), CMAP2(1.0), CMAP3(1.0)])

# for i, row in na_counts.iterrows():
#     ax.text(i, row['percent'] + 1,  # position just above the bar
#             f"{row['percent']:.2f}% ({row['count']})",
#             ha='center', va='bottom', fontsize=10)

# plt.ylabel('Percentage')
# plt.xlabel('Segmentation Class')
# plt.title('Missing Segmentation Masks')
# plt.yticks(range(0, 105, 10))
# plt.show()

sns.boxplot(data=case_day_seg_missing, x='class', y='count', palette=[CMAP1(1.0), CMAP2(1.0), CMAP3(1.0)])
sns.stripplot(data=case_day_seg_missing, x='class', y='count', color='black', size=3, jitter=True, alpha=0.4)
plt.ylabel('Missing Mask Count')
plt.xlabel('Segmentation Class')
plt.title('Distribution of Missing Masks per Class (by Case-Day)')
plt.show()


# display_multiple_slices(data.query("case == 43 and day == 26").id.unique(), 
#                         id_dicts, apply_CLAHE=True)


#visualizing the original image and image with true mask for border slices where segmentation classes just start appearing/disapperaing
display_image('case43_day26_slice_0057', id_dicts, apply_CLAHE=True)
display_image('case43_day26_slice_0058', id_dicts, apply_CLAHE=True)
display_image('case43_day26_slice_0121', id_dicts, apply_CLAHE=True)
display_image('case43_day26_slice_0122', id_dicts, apply_CLAHE=True)


# display_multiple_slices(data.query("case == 117 and day == 15").id.unique(), 
#                         id_dicts, apply_CLAHE=True)


#visualizing the original image and image with true mask for border slices where segmentation classes just start appearing/disapperaing
display_image('case117_day15_slice_0009', id_dicts, apply_CLAHE=True)
display_image('case117_day15_slice_0010', id_dicts, apply_CLAHE=True)
display_image('case117_day15_slice_0065', id_dicts, apply_CLAHE=True)
display_image('case117_day15_slice_0066', id_dicts, apply_CLAHE=True)


def nonna_masks_prop(data):
	nonna_seg_info = (
		data[['id', 'segmentation']]
			.groupby('id')['segmentation']
		 	.apply(lambda s: s.notna().sum())
		 	.value_counts()
		 	.sort_index()
		 	.rename_axis('nonna_masks_count')
		 	.reset_index(name='count')
	)
	nonna_seg_info['percent'] = 100 * nonna_seg_info['count'] / sum(nonna_seg_info['count']) 
	return nonna_seg_info

nonna_masks_prop(data)


# if SAVE_MASKS:
#     for i, (k, v) in enumerate(idclass_to_rle.items()):
#         if i == 5:   # stop after 5
#             break
#         print(k, v)


def save_mask(id_, id_dicts):
    mask = get_mask(id_, id_dicts)
    image_path = id_dicts['impath'][id_]
    rel_path = os.path.relpath(image_path, DIR_PATH)
    mask_path = os.path.splitext(rel_path)[0] + '.npy'
    #print(mask_path)
    mask_dir = mask_path.rsplit('/', 1)[0]
    os.makedirs(mask_dir, exist_ok = True)
    np.save(mask_path, mask)


# save_mask('case117_day15_slice_0065', id_dicts)
# mask = np.load('/kaggle/working/train/case117/case117_day15/scans/slice_0065_276_276_1.63_1.63.npy')
# mask.shape


if SAVE_MASKS:
    for id_ in tqdm(data.id.unique()):
        save_mask(id_, id_dicts)


resize_transform = A.Resize(IMAGE_RESIZE[0], IMAGE_RESIZE[1], 
                            interpolation = cv2.INTER_NEAREST,
                            mask_interpolation = cv2.INTER_NEAREST,)

def save_resized(id_, id_dicts):
    mask = get_mask(id_, id_dicts)
    image = load_image(id_, id_dicts['impath'])
    
    image, _ = insert_padding(image)
    mask, _ = insert_padding(mask)

    result = resize_transform(image = image, mask = mask)
    image, mask = result['image'], result['mask']

    image_orig_path = id_dicts['impath'][id_]
    rel_path = os.path.relpath(image_orig_path, DIR_PATH)
    
    mask_path = f'mask_{IMAGE_RESIZE[0]}/' + os.path.splitext(rel_path)[0] + '.npy'
    mask_dir = mask_path.rsplit('/', 1)[0]
    os.makedirs(mask_dir, exist_ok = True)
    np.save(mask_path, mask)

    image_path = f'image_{IMAGE_RESIZE[0]}/' + os.path.splitext(rel_path)[0] + '.npy'
    image_dir = image_path.rsplit('/', 1)[0]
    os.makedirs(image_dir, exist_ok = True)
    np.save(image_path, image)

if SAVE_RESIZED:
    for id_ in tqdm(data.id.unique()):
        save_resized(id_, id_dicts)


#visualize resized image and mask

#266x266
display_image('case90_day0_slice_0107', id_dicts, show_orig_img = False)
display_image('case90_day0_slice_0107', id_dicts, transforms = resize_transform, show_orig_img = False)

#360x310
display_image('case131_day0_slice_0066', id_dicts, show_orig_img = False)
display_image('case131_day0_slice_0066', id_dicts, transforms = resize_transform, show_orig_img = False)
display_image('case131_day0_slice_0066', id_dicts, transforms = resize_transform, show_orig_img = False, insert_pad=True)


sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
index_train, index_valid = list(sgkf.split(data.id, data.segmentation.isna(), data.case))[0]


len(index_train), len(index_valid)


data_train = data.iloc[index_train, :].reset_index(drop=True)
data_valid = data.iloc[index_valid, :].reset_index(drop=True)


data_train


data_valid


display_slice_size_prop(data_train)
display_slice_size_prop(data_valid)


display(seg_na_prop(data_train))
display(seg_na_prop(data_valid))


display(nonna_masks_prop(data_train))
display(nonna_masks_prop(data_valid))


print(len(data_train.case.unique()), len(data_valid.case.unique()))


data_train_sub = data_train.loc[data_train.case.isin(data_train.case.unique()[:11]), :]
data_valid_sub = data_valid.loc[data_valid.case.isin(data_valid.case.unique()[:2]), :]

print(len(data_train_sub), len(data_valid_sub), len(data_train_sub)/len(data_valid_sub))

# Note : 11, 2 numbers obtained by manually trying numbers 
#              with 17/10 ~ 2 for valid and such that train len/valid len ~ 4 similar to 4 splits for train and 1 split for valid


missing_masks_train = data_train_sub.segmentation.isna().sum() 
missing_masks_valid = data_valid_sub.segmentation.isna().sum() 
print(missing_masks_train, missing_masks_train*100/len(data_train_sub))
print(missing_masks_valid, missing_masks_valid*100/len(data_valid_sub))


display(seg_na_prop(data_train_sub))
display(seg_na_prop(data_valid_sub))


data_train_sub = data_train_sub.reset_index(drop=True)


data_valid_sub = data_valid_sub.reset_index(drop=True)


transform_train = A.ReplayCompose([
	A.Resize(IMAGE_RESIZE[0], IMAGE_RESIZE[1], interpolation = cv2.INTER_NEAREST, mask_interpolation = cv2.INTER_NEAREST),
	A.ShiftScaleRotate(shift_limit = 0.03, scale_limit = 0.05, rotate_limit = 5, p = 0.5),
	A.GridDistortion(num_steps = 5, distort_limit = 0.03, p = 0.25),
	A.CoarseDropout(num_holes_range = (9, 9), hole_height_range = (0.01,0.05), hole_width_range = (0.01,0.05), fill_mask = 0, p = 0.5),
	A.Normalize(mean = IMAGE_NORMALIZE_MEAN, std = IMAGE_NORMALIZE_SD, max_pixel_value = 1.0),
	A.ToTensorV2(transpose_mask = True),
])

transform_valid = A.Compose([
	A.Resize(IMAGE_RESIZE[0], IMAGE_RESIZE[1], interpolation = cv2.INTER_NEAREST, mask_interpolation = cv2.INTER_NEAREST),
	A.Normalize(mean = IMAGE_NORMALIZE_MEAN, std = IMAGE_NORMALIZE_SD, max_pixel_value = 1.0),
	A.ToTensorV2(transpose_mask = True),
])


warnings.filterwarnings('ignore', message = 'ShiftScaleRotate is a special case of Affine transform')


# transforms = A.ElasticTransform(alpha=100, sigma=50, p=1.0)
# transforms = A.ShiftScaleRotate(shift_limit=(0.3,0.3), scale_limit=0.0,
#                                 rotate_limit=0.0, p=1.0)
# transforms = A.RandomBrightnessContrast(brightness_limit = 0.1, contrast_limit = 0.1, p=1.0)
# transforms_test = A.CoarseDropout(num_holes_range = (6, 12), hole_height_range = (0.01,0.05),
#                                   hole_width_range = (0.01,0.05), fill_mask = 0, p = 0.5)
transforms_test = A.CoarseDropout(num_holes_range = (9, 9), hole_height_range = (0.01,0.05),
                                  hole_width_range = (0.01,0.05), fill_mask = 0, p = 1.0)


display_image('case131_day0_slice_0066', id_dicts, show_orig_img = True, show_true_mask=True)
display_image('case131_day0_slice_0066', id_dicts, show_orig_img = True,
              transforms = transforms_test,
              show_true_mask=True)



def create_caseday_replay_transform(df, transforms):
    caseday_replay_transform = {}
    h, w = IMAGE_RESIZE
    dummy_image = np.zeros((h, w, IMAGE_CHANNELS), dtype=np.float32)
    dummy_mask  = np.zeros((h, w), dtype=np.uint8)

    for (case, day), group in df.groupby(['case', 'day']):
        augmented = transforms(image=dummy_image, mask=dummy_mask)
        caseday_str = f'case{case}_day{day}'
        caseday_replay_transform[caseday_str] = augmented['replay']

    return caseday_replay_transform


transforms_test = A.ReplayCompose([A.CoarseDropout(num_holes_range = (9,9), hole_height_range = (0.01,0.05), 
                                                   hole_width_range = (0.01,0.05), fill_mask = 0, p = 0.5),
                                   A.Normalize(mean = IMAGE_NORMALIZE_MEAN, std = IMAGE_NORMALIZE_SD, max_pixel_value = 1.0),])
caseday_replay_transform = create_caseday_replay_transform(data, transforms_test)
display_image('case149_day15_slice_0108', id_dicts, show_orig_img = True,
              replay_transforms = transforms_test, caseday_replay_transform = caseday_replay_transform,
              show_true_mask = True)
display_image('case149_day15_slice_0109', id_dicts, show_orig_img = True,
              replay_transforms = transforms_test, caseday_replay_transform = caseday_replay_transform,
              show_true_mask = True)
display_image('case149_day15_slice_0110', id_dicts, show_orig_img = True,
              replay_transforms = transforms_test, caseday_replay_transform = caseday_replay_transform,
              show_true_mask = True)


# # intensity ranges of images are inconsistent as seen in this cell's output shown commented
# # so it is better to use per image scaling while loading images instead of globally normalizing by 65535(since image is 16 bit)

# def read_image(id_, id_to_impath):
#   img = cv2.imread(id_to_impath[id_], cv2.IMREAD_UNCHANGED)
#   return img

# sample_image = read_image('case123_day0_slice_0001', id_to_impath)
# print(np.min(sample_image), np.max(sample_image))

# sample_image = read_image('case123_day0_slice_0002', id_to_impath)
# print(np.min(sample_image), np.max(sample_image))

# sample_image = read_image('case123_day0_slice_0003', id_to_impath)
# print(np.min(sample_image), np.max(sample_image))

# sample_image = read_image('case123_day20_slice_0001', id_to_impath)
# print(np.min(sample_image), np.max(sample_image))

# sample_image = read_image('case123_day22_slice_0001', id_to_impath)
# print(np.min(sample_image), np.max(sample_image))

# sample_image = read_image('case42_day0_slice_0001', id_to_impath)
# print(np.min(sample_image), np.max(sample_image))

# sample_image = read_image('case42_day17_slice_0001', id_to_impath)
# print(np.min(sample_image), np.max(sample_image))

# sample_image = read_image('case42_day19_slice_0001', id_to_impath)
# print(np.min(sample_image), np.max(sample_image))

# sample_image = read_image('case129_day0_slice_0001', id_to_impath)
# print(np.min(sample_image), np.max(sample_image))

# sample_image = read_image('case129_day20_slice_0001', id_to_impath)
# print(np.min(sample_image), np.max(sample_image))

# sample_image = read_image('case129_day22_slice_0001', id_to_impath)
# print(np.min(sample_image), np.max(sample_image))

# # 0 3621
# # 0 3553
# # 0 2822
# # 0 2546
# # 0 6886
# # 0 2322
# # 0 1332
# # 0 4326
# # 0 521
# # 0 363
# # 0 200


class GITractDataset(Dataset):
	def __init__(self, df, dataset_type, transforms=None, load_saved_masks=LOAD_SAVED_MASKS, load_resized=LOAD_RESIZED):
		'''
		dataset_type : One of 'train, 'valid', 'test'
		'''
		self.id_ = df.id.unique()

		self.dataset_type = dataset_type
		self.transforms = transforms

		id_to_impath = dict(zip(df.id, df.image_path))
		self.id_dicts = {'impath': id_to_impath}

		id_to_shape = None
		idclass_to_rle = None
		
		if not (load_saved_masks or load_resized) or self.dataset_type == 'test':
			id_to_shape = dict(zip(df.id, zip(df.slice_h, df.slice_w)))
			
		if not (load_saved_masks or load_resized) and self.dataset_type != 'test':
			idclass_to_rle = {
				(id_, class_): seg
				for id_, class_, seg in zip(df.id, df['class'], df.segmentation)
				if pd.notna(seg)
			}

		self.id_dicts['shape'] = id_to_shape
		self.id_dicts['rle'] = idclass_to_rle

		self.load_resized = load_resized

		# note : can't use set of either 80 or 144 slice count casedays, since test data could have other slice count per caseday
		self.case_day_slice_count = {
			f'case{k[0]}_day{k[1]}': v
			for k, v in dict(
				df[['case', 'day', 'slice']].drop_duplicates().value_counts(['case', 'day'])
			).items()
		}
			

	def __len__(self):
		return len(self.id_)
	
	def __getitem__(self, idx):
		id_ = self.id_[idx]

		caseday_str, slice_str = id_.rsplit('_slice_', 1)
		slice_id = int(slice_str)

		# create a 2.5D image - by stacking <IMAGE_CHANNELS> nearby slices as <IMAGE_CHANNELS> channels of an image
		slices = []
		for offset in IMAGE_OFFSETS:
			i = np.clip(slice_id + offset, 1, self.case_day_slice_count[caseday_str])
			id_with_offset = f'{caseday_str}_slice_{i:04d}'
			img_slice = load_image(id_with_offset, self.id_dicts['impath'], load_resized=self.load_resized)
			slices.append(img_slice)
		img = np.stack(slices, axis=2) # shape (H, W, <IMAGE_CHANNELS>)
		
		if self.dataset_type != 'test':
			mask = get_mask(id_, self.id_dicts, load_resized=self.load_resized)
			if self.transforms:
				if not self.load_resized:
					img, _ = insert_padding(img)
					mask, _ = insert_padding(mask)
				if self.dataset_type == 'train':
					#using caseday_replay_transform global variable to avoid replicating this in different workers 
					#                                                           and using num_workers times memory
					augmented = A.ReplayCompose.replay(caseday_replay_transform[caseday_str], image=img, mask=mask)
				else:
					augmented = self.transforms(image=img, mask=mask)
				img, mask = augmented['image'], augmented['mask']
			return img, mask, id_
		else:
			h, w = self.id_dicts['shape'][id_]
			if self.transforms:
				img, padding = insert_padding(img)
				augmented = self.transforms(image=img)
				img = augmented['image']
			return img, id_, h, w, torch.tensor(padding, dtype=torch.int32) 


caseday_replay_transform = create_caseday_replay_transform(data_train, transform_train)

dataset_train = GITractDataset(data_train, dataset_type='train', transforms=transform_train)
dataset_valid = GITractDataset(data_valid, dataset_type='valid', transforms=transform_valid)

dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=DATA_LOADER_NUM_WORKERS, pin_memory=True)
dataloader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE_VALID, shuffle=False, num_workers=DATA_LOADER_NUM_WORKERS, pin_memory=True)


dataset = next(iter(dataloader_train))
img, mask, id_ = dataset
print(img.shape, mask.shape, len(id_))


idx = 15
np.max(img[idx].numpy()), np.min(img[idx].numpy())


type(img[idx].numpy()[0, 0, 0]), type(mask[idx].numpy()[0, 0, 0])


def display_dataset(dataset, display_orig=False, num_images=None, denormalize=False, apply_CLAHE=False, caseday_str=None):
	'''
	dataset : dataset to be displayed
	display_orig : Should the original images prior to augmentation be shown alongside images after augmentation.
				   In this case 1st 5 images before and after augmentation is shown and num_images parameter value is ignored
	num_images : Number of images to be shown. Defaults to the full dataset size i.e. the batch size
	denormalize : Set to True if A.normalize has been applied as part of augmentations and you wish to denormalize it
	'''
	img_arr, mask_arr, id_arr = dataset
	if num_images is None:
		num_images = len(img_arr)
	max_cols = 5
	
	if display_orig:
		num_images = 5
		rows = 2
		plt.figure(figsize=(max_cols*3, rows*3))
		ids_shown = list()
	else:
		rows = np.ceil(num_images/max_cols).astype(int)
		plt.figure(figsize=(max_cols*3, rows*3))

	idx, img_count = 0, 0
	while img_count < num_images and idx < len(img_arr):
		img, mask, id_ = img_arr[idx], mask_arr[idx], id_arr[idx]
		idx += 1
		if caseday_str is not None and caseday_str != id_.rsplit('_slice_', 1)[0]:
			continue
		else:
			img_count += 1
		img = img.permute(1,2,0)    #after the permute, img is in HxWxC format
		if denormalize:
			#print('denormalizing')
			#print(img.shape)
			img = img * torch.tensor(IMAGE_NORMALIZE_SD) + torch.tensor(IMAGE_NORMALIZE_MEAN)
			img = img.clamp(0, 1)
		img = img.cpu().numpy()
		img = (img * 255).astype(np.uint8)

		if apply_CLAHE:
			clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2,2))
			for ch in range(3):
				img[:,:,ch] = clahe.apply(img[:,:,ch])

		mask = mask.permute(1,2,0).cpu().numpy()

		plt.subplot(rows, max_cols, img_count)
		#print(f'just before imshow {id_} min : {np.min(img)} max : {np.max(img)}')
		print(img.shape)
		#print(f'just before imshow mask {id_} min : {np.min(mask)} max : {np.max(mask)}')

		#img is <IMAGE_CHANNELS> channel with different channels corresponding to nearby slices
		#for display, showing 1st, middle and last slice
		mid_slice_id = IMAGE_CHANNELS // 2
		plt.imshow(img[:, :, [0, mid_slice_id, -1]])  
		plt.title(f'{idx} : {id_}')
		
		plt.imshow(mask[..., 0], cmap=CMAP1)
		plt.imshow(mask[..., 1], cmap=CMAP2)
		plt.imshow(mask[..., 2], cmap=CMAP3)
		plt.axis('off')

		if idx == 0:
			handles = [
				Rectangle((0, 0), 1, 1, color=CMAP1(1.0)),
				Rectangle((0, 0), 1, 1, color=CMAP2(1.0)),
				Rectangle((0, 0), 1, 1, color=CMAP3(1.0))
			]
			labels = ['Large Bowel', 'Small Bowel', 'Stomach']
			plt.legend(handles, labels, bbox_to_anchor=(0.0, 1.5), loc='upper left', borderaxespad=0.)

		if display_orig:
			ids_shown.append(id_)

	if display_orig:
		idx = num_images - 1
		print(ids_shown)

		for id_ in ids_shown:
			idx += 1
			img = load_image(id_, id_to_impath)
			img = (img * 255).astype(np.uint8) # 0-255 range required for CLAHE. 
											   # Using this in general to maintain consistency with the case where CLAHE is required
			if apply_CLAHE:
				clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2,2))
				img = clahe.apply(img)
				
			mask = get_mask(id_, id_dicts)
			
			plt.subplot(rows, max_cols, idx+1)
			#print(f'just before imshow orig {id_} min : {np.min(img)} max : {np.max(img)}')
			print(img.shape)
			plt.imshow(img, cmap='bone')
			plt.title('Original')
			plt.imshow(mask[..., 0], cmap=CMAP1)
			plt.imshow(mask[..., 1], cmap=CMAP2)
			plt.imshow(mask[..., 2], cmap=CMAP3)
			plt.axis('off')

	plt.tight_layout()
	plt.show()


display_dataset(dataset, num_images=5, denormalize=True)


display_dataset(dataset, display_orig=True, denormalize=True, apply_CLAHE=True)


# display_dataset(dataset, display_orig=True, denormalize=True, apply_CLAHE=True, caseday_str="case74_day13")


def explore_mask_component_size(dataloader):
	min_area_list = [np.inf, np.inf, np.inf]
	for data in dataloader:
		_, masks, _ = data
		for mask in masks:
			for c in range(NUM_CLASSES):
				organ_mask = mask[c, ...].numpy()
				num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(organ_mask)
				if num_labels > 1:
					component_areas = stats[1:, cv2.CC_STAT_AREA] #ignore 0th that gives background area
					min_area_list[c] = min(min_area_list[c], component_areas.min())
	return min_area_list

# explore_mask_component_size(dataloader_train), explore_mask_component_size(dataloader_valid)
# # ([1, 1, 1], [42, 4, 65])
    
# dataset_train_dummy = GITractDataset(data_train, dataset_type='valid', transforms=transform_valid)
# dataloader_train_dummy = DataLoader(dataset_train_dummy, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=DATA_LOADER_NUM_WORKERS, pin_memory=True)
# # explore_mask_component_size(dataloader_train_dummy)
# # [2, 2, 1]


def identify_small_component_mask(dataloader, small_size):
    ids_of_interest = []
    for data in dataloader:
        _, masks, ids = data
        for mask, id_ in zip(masks, ids):
            for c in range(NUM_CLASSES):
                organ_mask = mask[c, ...].numpy()
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(organ_mask)
                if num_labels > 1:
                    component_areas = stats[1:, cv2.CC_STAT_AREA]
                    if component_areas.min() <= small_size:
                        ids_of_interest.append(id_)
                        break
    return list(set(ids_of_interest))

# print(identify_small_component_mask(dataloader_train_dummy, 2))
# print(identify_small_component_mask(dataloader_valid, 4))
# # ['case85_day0_slice_0102', 'case88_day36_slice_0101', 'case16_day0_slice_0036', 'case130_day20_slice_0077', 'case88_day0_slice_0099']
# # ['case124_day20_slice_0106']


def show_mask_component_size(id_):
    mask = get_mask(id_, id_dicts)
    for c in range(NUM_CLASSES):
        organ_mask = mask[..., c]
        print(f'Organ : {CLASS_NAMES[c]}')
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(organ_mask)
        for i in range(1, num_labels):
            print(f'LabelId: {i}   Area: {stats[i, cv2.CC_STAT_AREA]}')
    return 


print('Small component masks from train data')
display_image('case85_day0_slice_0102', id_dicts)
show_mask_component_size('case85_day0_slice_0102')
display_image('case88_day36_slice_0101', id_dicts)
show_mask_component_size('case88_day36_slice_0101')
display_image('case16_day0_slice_0036', id_dicts)
show_mask_component_size('case16_day0_slice_0036')
display_image('case130_day20_slice_0077', id_dicts)
show_mask_component_size('case130_day20_slice_0077')
display_image('case88_day0_slice_0099', id_dicts)
show_mask_component_size('case88_day0_slice_0099')

print('\nSmall component masks from valid data')
display_image('case124_day20_slice_0106', id_dicts)
show_mask_component_size('case124_day20_slice_0106')


if TRAIN_VALID_SPLIT or TRAIN_ON_FULL_DATA or TEST_PREDICT:
	# https://smp.readthedocs.io/en/latest/encoders_timm.html
	smp_encoder_weights = None if TEST_PREDICT and LOAD_MODEL_FOR_TEST_PREDICT else 'imagenet'
	model = smp.Segformer(
		encoder_name = 'mit_b4',
		encoder_weights = smp_encoder_weights,     
		in_channels = IMAGE_CHANNELS,                  
		classes = NUM_CLASSES,
		aux_params = dict(
			classes = NUM_CLASSES,
			pooling = 'max',
			dropout = 0.3,
			activation = None
		)
	)
	ema_model = ModelEmaV2(model, decay=0.999)
	model.to(DEVICE)
	ema_model.to(DEVICE)

	if ENSEMBLE_MODEL_PREDICT:
		ensemble_models = []
		model_tmp = smp.Unet(
				encoder_name = 'tu-tf_efficientnetv2_m',
				encoder_weights = None,     
				in_channels = IMAGE_CHANNELS,                  
				classes = NUM_CLASSES,
				aux_params = dict(
					classes = NUM_CLASSES,
					pooling = 'max',
					dropout = 0.3,
					activation = None
				)
			)
		model_tmp.to(DEVICE)
		ensemble_models.append(model_tmp)


#return optimizer param_groups with Layer-wise Learning Rate Decay(LLRD) and specific Weight Decay for different layers

def get_optimizer_param_groups(model, base_lr, weight_decay=1e-5):

	group_configs = [
	  {
		  'name': 'stage0', 
		  'lr_mult': 0.5, 
		  'keywords': ['encoder.patch_embed1']
	  },
	  {
		  'name': 'stage1', 
		  'lr_mult': 0.5,
		  'keywords': ['encoder.block1', 'encoder.norm1']
	  },
	  {
		  'name': 'stage2', 
		  'lr_mult': 0.5, 
		  'keywords': ['encoder.patch_embed2', 'encoder.block2', 'encoder.norm2']
	  },
	  {
		  'name': 'stage3', 
		  'lr_mult': 0.75, 
		  'keywords': ['encoder.patch_embed3', 'encoder.block3', 'encoder.norm3']
	  },
	  {
		  'name': 'stage4', 
		  'lr_mult': 1.0, 
		  'keywords': ['encoder.patch_embed4', 'encoder.block4', 'encoder.norm4']
	  },
	  {
		  'name': 'decoder_seghead', 
		  'lr_mult': 1.0, 
		  'keywords': ['decoder', 'segmentation_head']
	  },
	  {
		  'name': 'aux_head', 
		  'lr_mult': 0.5, 
		  'keywords': ['classification_head']
	  }
	]
	# group_configs = [
	#   {
	#       'name': 'stage1', 
	#       'lr_mult': 0.125, 
	#       'keywords': ['encoder.model.conv_stem', 'encoder.model.bn1', 'encoder.model.blocks.0']
	#   },
	#   {
	#       'name': 'stage2', 
	#       'lr_mult': 0.25, 
	#       'keywords': ['encoder.model.blocks.1', 'encoder.model.blocks.2']
	#   },
	#   {
	#       'name': 'stage3', 
	#       'lr_mult': 0.5, 
	#       'keywords': ['encoder.model.blocks.3', 'encoder.model.blocks.4']
	#   },
	#   {
	#       'name': 'stage4', 
	#       'lr_mult': 1.0, 
	#       'keywords': ['encoder.model.blocks.5', 'encoder.model.blocks.6']
	#   },
	#   {
	#       'name': 'decoder_seghead', 
	#       'lr_mult': 1.0, 
	#       'keywords': ['decoder', 'segmentation_head']
	#   },
	#   {
	#       'name': 'aux_head', 
	#       'lr_mult': 0.0625, 
	#       'keywords': ['classification_head']
	#   }
	# ]
	# group_configs = [
	#   {
	#       'name': 'stage1', 
	#       'lr_mult': 1.0, 
	#       'keywords': ['encoder.model.conv_stem', 'encoder.model.bn1', 'encoder.model.blocks.0']
	#   },
	#   {
	#       'name': 'stage2', 
	#       'lr_mult': 1.0, 
	#       'keywords': ['encoder.model.blocks.1', 'encoder.model.blocks.2']
	#   },
	#   {
	#       'name': 'stage3', 
	#       'lr_mult': 1.0, 
	#       'keywords': ['encoder.model.blocks.3', 'encoder.model.blocks.4']
	#   },
	#   {
	#       'name': 'stage4', 
	#       'lr_mult': 1.0, 
	#       'keywords': ['encoder.model.blocks.5', 'encoder.model.blocks.6']
	#   },
	#   {
	#       'name': 'decoder_seghead', 
	#       'lr_mult': 1.0, 
	#       'keywords': ['decoder', 'segmentation_head']
	#   },
	#   {
	#       'name': 'aux_head', 
	#       'lr_mult': 1.0, 
	#       'keywords': ['classification_head']
	#   }
	# ]

	# dict to hold param lists temporarily
	# dict key : (group_name, lr multiplier, requires_weight_decay)
	param_buckets = {} 

	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue
		
		assigned_group = None
		for config in group_configs:
			if any(k in name for k in config['keywords']):
				assigned_group = config
				break

		# Standard rule: No decay for Bias, Norms (bn, ln)
		no_decay_keywords = ['bias', 'bn', 'norm', 'ln']
		apply_wd = not any(nd in name.lower() for nd in no_decay_keywords)

		bucket_key = (assigned_group['name'], assigned_group['lr_mult'], apply_wd)
		if bucket_key not in param_buckets:
			param_buckets[bucket_key] = []
		param_buckets[bucket_key].append(param)

	final_param_groups = []
	for (name, mult, apply_wd), params in param_buckets.items():
		final_param_groups.append({
			'params': params,
			'lr': base_lr * mult,
			'weight_decay': weight_decay if apply_wd else 0.0,
			'name': f'{name}_wd_{apply_wd}'
		})

	for group in final_param_groups:
		print(group['name'], group['lr'], group['weight_decay'])

	return final_param_groups


if TRAIN_VALID_SPLIT or TRAIN_ON_FULL_DATA or TEST_PREDICT:
	
	optimizer = optim.AdamW(get_optimizer_param_groups(model, base_lr = LR_START), 
							lr = LR_START)

	#We use an LR scheduler that gives linear warmup in first few epochs and then decrease with cosine decay.
	#To assign a different lower LRs to various layers of the model, we save the regular LR in a list 
	#          and return the factor using LambdaLR scheduler
	
	dummy_opt = optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr = LR_START)
	warmup_scheduler = optim.lr_scheduler.LambdaLR(dummy_opt,
												   lr_lambda = lambda x: x)
	cosine_decay_scheduler = optim.lr_scheduler.CosineAnnealingLR(dummy_opt, 
																  T_max = EPOCHS - LR_WARMUP_EPOCHS, 
																  eta_min = LR_END)
	cosine_decay_scheduler.base_lrs = [LR_START * LR_WARMUP_EPOCHS]
	scheduler = optim.lr_scheduler.SequentialLR(dummy_opt, 
												schedulers = [warmup_scheduler, cosine_decay_scheduler], 
												milestones = [LR_WARMUP_EPOCHS])

	LR_LIST = []
	for epoch in range(EPOCHS):
		scheduler.step() #calling step at the start so that the LR_END value will be used in the last epoch 
		current_lr = scheduler.get_last_lr()[0]  
		LR_LIST.append(current_lr)
	print(LR_LIST)
	del dummy_opt, warmup_scheduler, cosine_decay_scheduler, scheduler

	scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda x: LR_LIST[x-1] / LR_START)

	
	scaler = GradScaler(DEVICE)


# scheduler.step()
# for group in optimizer.param_groups:
#     n_params = sum(p.numel() for p in group['params'])
#     group_param_ids = {id(p) for p in group['params']}
#     sample_names = [n for n, p in model.named_parameters()
#                     if id(p) in group_param_ids][:5]  # show up to 5 example names
#     print(f"Group {group['name']}: lr={group['lr']}, wd={group['weight_decay']}, n_params={n_params}")
#     print('Examples:', sample_names)


# for epoch in range(EPOCHS):
# 	scheduler.step() 
	
# 	for group in optimizer.param_groups:
# 		print(f"Epoch {epoch+1} - Group {group['name']} LR: {group['lr']:.4e}")
# 	print('\n')


dice_loss = smp.losses.DiceLoss(mode = 'multilabel') 
focal_loss = smp.losses.FocalLoss(mode = 'multilabel', alpha=0.25, gamma=2.0)

def loss_fn(seg_logits, presence_logits, seg_true, 
			loss_wt = 0.5, weight_presence = 0.3, weight_seg = 0.7):

	presence_true = seg_true.amax((2,3)).float()   #(B, C)  
	presence_loss = sigmoid_focal_loss(presence_logits, presence_true, alpha = 0.4, gamma = 2.0, reduction = 'mean')

	dice = dice_loss(seg_logits, seg_true)
	focal = focal_loss(seg_logits.contiguous(), seg_true.contiguous())

	seg_loss = dice * loss_wt + focal * (1 - loss_wt)

	total_loss = weight_presence * presence_loss + weight_seg * seg_loss
						
	loss_components = [
		presence_loss, seg_loss, dice, focal
	]
	loss_components = torch.stack(loss_components).detach().cpu().numpy()

	return total_loss, loss_components 


LOSS_NAMES = ['Pres', 'Seg', 'D', 'F']


class DiceScoreCustom:
	def __init__(self, num_classes, eps=1e-6):
		self.num_classes = num_classes
		self.eps = eps
		self.reset()

	def reset(self):
		self.dice_sum = 0.0
		self.image_count = 0

		# per-organ totals
		self.organ_dice_sum = torch.zeros(self.num_classes)
		self.organ_count = torch.zeros(self.num_classes)

	def update(self, preds, targets):
		"""
		preds, targets: (B, C, H, W) binary {0,1} tensors
		Implements host comment: skip organs where both pred & target are empty : https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/discussion/324934
		"""
		I = (targets & preds).sum((2, 3))
		U = (targets | preds).sum((2, 3))

		# Dice per organ (B, C)
		dice = (2 * I) / (U + I + self.eps)

		# Mask out empty organs (where both pred and gt are 0)
		non_empty = U > 0  # (B, C)

		# For each image (B), compute mean over valid organs only
		organ_counts = non_empty.sum(dim=1)  # (B)
		dice_per_image = dice.sum(dim=1) / organ_counts.clamp(min=1)

		#note : accumulators below are moved to CPU to reduce GPU memory usage

		# accumulate global
		valid_images = organ_counts > 0
		if valid_images.any():
			self.dice_sum += dice_per_image[valid_images].sum().item()
			self.image_count += dice_per_image[valid_images].numel()

		# accumulate per-organ
		self.organ_dice_sum += dice.sum(dim=0).detach().cpu()
		self.organ_count += non_empty.sum(dim=0).detach().cpu()

	def compute(self):
		"""
		Returns:
			overall dice: scalar tensor
			per_organ dice: (C,) tensor
		"""
		overall = (
			torch.tensor(self.dice_sum / self.image_count)
			if self.image_count > 0 
			else torch.tensor(0.0)
		)
		per_organ = torch.where(
			self.organ_count > 0,
			self.organ_dice_sum / self.organ_count,
			torch.tensor(0.0)
		)
		return overall, per_organ

	def compute_per_image(self, pred, target):
		"""
		standalone function to compute dice per image and not modify accumulators
		pred, target: (C, H, W) binary {0,1} tensors
		"""
		I = (target & pred).sum((1, 2))
		U = (target | pred).sum((1, 2))

		dice = (2 * I) / (U + I + self.eps)

		organ_count = (U > 0).sum()
		dice_per_image = dice.sum() / max(organ_count, 1)

		return dice_per_image.item()


class HausdorffDistanceCustom:
	def __init__(self, num_classes):
		self.num_classes = num_classes
		self.reset()

	def reset(self):
		self.h3d_sum = 0.0
		self.image3d_count = 0

		self.organ_h3d_sum = np.zeros(self.num_classes)
		self.organ_count_sum = np.zeros(self.num_classes)

	def _compute_hausdorff_per_organ(self, preds, targets):
		'''
		preds and targets : (Depth, Height, Width) binary {0,1} tensors
		'''
		if np.all(preds == targets):
			return 0.0
	
		(edges_preds, edges_targets) = get_mask_edges(preds, targets)
		surface_distance = get_surface_distance(edges_preds, edges_targets, distance_metric="euclidean")
	
		if surface_distance.shape == (0,):
			return 0.0
		dist = surface_distance.max()
		max_dist = np.sqrt(np.sum((np.array(preds.shape) - 1) ** 2))
	
		if dist > max_dist:
			return 1.0
	
		return dist / max_dist

	def update(self, preds, targets):
		'''
		preds and targets : (Channel, Depth, Height, Width) binary {0,1} tensors
		'''

		U = (targets | preds).sum((1, 2, 3))  # [C]

		hausdorff = np.array([self._compute_hausdorff_per_organ(preds[i, ...], targets[i, ...]) for i in range(self.num_classes)])  # [C]

		# Mask out empty organs (where both pred and gt are 0)
		non_empty = U > 0  # [C]

		organ_count = non_empty.sum()

		if organ_count != 0:
			hausdorff_per_3dimage = hausdorff.sum() / organ_count

			# accumulate global
			self.h3d_sum += hausdorff_per_3dimage
			self.image3d_count += 1

		# accumulate per-organ
		self.organ_h3d_sum += hausdorff
		self.organ_count_sum += non_empty

	def compute(self):
		"""
		Returns:
			overall hausdorff: scalar
			per_organ hausdorff: (C,)
		"""
		overall = self.h3d_sum / self.image3d_count

		per_organ = self.organ_h3d_sum / self.organ_count_sum

		return overall, per_organ

	def compute_per_caseday(self, preds, targets):
		'''
		standalone function to compute hausdorff per caseday slices and not modify accumulators
		preds and targets : (Channel, Depth, Height, Width) binary {0,1} tensors
		'''

		U = (targets | preds).sum((1, 2, 3))  # [C]

		hausdorff = np.array([self._compute_hausdorff_per_organ(preds[i, ...], targets[i, ...]) for i in range(self.num_classes)])  # [C]
		organ_count = (U > 0).sum()
		
		hausdorff_per_3dimage = hausdorff.sum() / max(organ_count, 1)

		return hausdorff_per_3dimage



dice_score_obj = DiceScoreCustom(num_classes=NUM_CLASSES)
hausdorff_obj = HausdorffDistanceCustom(num_classes=NUM_CLASSES)


def one_epoch_train(epoch, dataloader):

	epoch_start = time.time()
	
	model.train() #set model in training mode
	running_loss = 0.0
	running_loss_components = np.zeros(len(LOSS_NAMES), dtype=np.float32)

	data_time, gpu_times = 0.0, []
	data_end_time = time.time()  #used to compute data loading time

	optimizer.zero_grad()
	loop = tqdm(enumerate(dataloader), desc=f'Epoch {epoch+1}/{EPOCHS}')
	for step, data in loop:

		data_time += time.time() - data_end_time
		
		imgs, masks, ids = data
		imgs, masks = imgs.to(DEVICE, dtype=torch.float), masks.to(DEVICE, dtype=torch.float)

		start_event = torch.cuda.Event(enable_timing = True)
		end_event = torch.cuda.Event(enable_timing = True)
		start_event.record()
		
		with autocast('cuda'):
			pred_masks, pred_labels = model(imgs)
			loss, loss_components = loss_fn(pred_masks, pred_labels, masks)
			loss = loss / GRAD_ACCUM_STEPS
			
		scaler.scale(loss).backward()

		if ((step+1) % GRAD_ACCUM_STEPS == 0) or ((step+1) == len(dataloader)):
			scaler.step(optimizer)
			scaler.update()
			optimizer.zero_grad()
			ema_model.update(model)

		running_loss += loss.item() * GRAD_ACCUM_STEPS
		running_loss_components += loss_components

		end_event.record()
		gpu_times.append((start_event, end_event))
		
		loop.set_postfix(loss=loss.item()*GRAD_ACCUM_STEPS)

		data_end_time = time.time()

	torch.cuda.synchronize()
	gpu_time = sum(s.elapsed_time(e) for s,e in gpu_times) / 1000.0   #seconds

	avg_loss = running_loss / len(dataloader)
	avg_loss_components = running_loss_components / len(dataloader)

	epoch_time = time.time() - epoch_start

	time_log = {
		'epoch': epoch,
		't_epoch_time': epoch_time, #t for train
		't_data_time': data_time,
		't_gpu_time': gpu_time,
		't_data_perc': data_time * 100.0 / epoch_time,
		't_gpu_perc': gpu_time * 100.0 / epoch_time
	}
	
	return avg_loss, avg_loss_components, time_log


slices80_casedays = set(
    data_valid[['case', 'day', 'slice']]
    .drop_duplicates()
    .value_counts(['case', 'day'])
    .loc[lambda s: s == 80]
    .index
)
#slices80_casedays


def one_epoch_valid(epoch, dataloader):

	epoch_start = time.time()
	
	model.eval()

	data_time, gpu_times, hausdorff_time = 0.0, [], 0.0
	
	with torch.no_grad():
		running_loss = 0.0
		running_loss_components = np.zeros(len(LOSS_NAMES), dtype=np.float32)
		pred_masks_dict, masks_dict = {}, {}
		
		data_end_time = time.time()  #used to compute data loading time

		if EXPLORE_CUTOFFS:
			print(PRED_LABEL_CUTOFFS)
			print(PRED_MASK_CUTOFFS)
		pred_label_cutoffs = torch.tensor(PRED_LABEL_CUTOFFS, device=DEVICE, dtype=torch.float)
		pred_mask_cutoffs = torch.tensor(PRED_MASK_CUTOFFS, device=DEVICE, dtype=torch.float)
		for data in dataloader:
			data_time += time.time() - data_end_time
			
			imgs, masks, ids = data
			imgs, masks = imgs.to(DEVICE, dtype=torch.float), masks.to(DEVICE, dtype=torch.float)

			start_event = torch.cuda.Event(enable_timing = True)
			end_event = torch.cuda.Event(enable_timing = True)
			start_event.record()

			with autocast('cuda'):
				pred_masks, pred_labels = model(imgs)
				loss, loss_components = loss_fn(pred_masks, pred_labels, masks)
			running_loss += loss.item()
			running_loss_components += loss_components

			pred_labels = torch.sigmoid(pred_labels.float())
			pred_masks = (torch.sigmoid(pred_masks.float()) > pred_mask_cutoffs[:, None, None]).int()   
			# pred_masks.float() to convert from FP16 to FP32
			# pred_labels : (B,C)    pred_masks : (B,C,H,W)
			
			pred_masks = pred_masks * (pred_labels > pred_label_cutoffs)[...,None,None].int()

			masks = masks.int()
			dice_score_obj.update(pred_masks, masks)

			end_event.record()
			gpu_times.append((start_event, end_event))

			#loop through predictions and true masks to create 3D volume with all slices per caseday for Hausdorff computation
			hausdorff_start = time.time()
			for p, m, id_ in zip(pred_masks, masks, ids):
				match = re.match(r"case(\d+)_day(\d+)_slice_(\d+)", id_)
				if match:
					caseid, dayid, sliceid = map(int, match.groups())

				casedayid = (caseid, dayid) 

				pred_masks_dict.setdefault(casedayid, []).append((sliceid, p))
				masks_dict.setdefault(casedayid, []).append((sliceid, m))

				#in the data, casedays have either 144 slices or 80 slices
				if (len(pred_masks_dict[casedayid]) == 144) or (casedayid in slices80_casedays and len(pred_masks_dict[casedayid]) == 80):
					pred_masks_sorted = [p.cpu().numpy() for sid, p in sorted(pred_masks_dict[casedayid], key=lambda x: x[0])]
					masks_sorted = [m.cpu().numpy() for sid, m in sorted(masks_dict[casedayid], key=lambda x: x[0])]

					pred_masks_volume = np.stack(pred_masks_sorted, axis=1)
					masks_volume = np.stack(masks_sorted, axis=1)

					hausdorff_obj.update(pred_masks_volume, masks_volume)

					#free memory
					del pred_masks_dict[casedayid], masks_dict[casedayid]
			hausdorff_time += time.time() - hausdorff_start

			data_end_time = time.time()

			
		avg_loss = running_loss / len(dataloader)
		avg_loss_components = running_loss_components / len(dataloader)

		epoch_dice_score = dice_score_obj.compute()
		dice_score_obj.reset()
		
		epoch_hausdorff = hausdorff_obj.compute()
		hausdorff_obj.reset()   

	torch.cuda.synchronize()
	gpu_time = sum(s.elapsed_time(e) for s,e in gpu_times) / 1000.0   #seconds
	
	epoch_time = time.time() - epoch_start
	
	time_log = {
		'epoch': epoch,
		'v_epoch_time': epoch_time, #v for valid
		'v_data_time': data_time,
		'v_gpu_time': gpu_time,
		'v_hausdorff_time': hausdorff_time,
		'v_data_perc': data_time * 100.0 / epoch_time,
		'v_gpu_perc': gpu_time * 100.0 / epoch_time,
		'v_hausdorff_perc': hausdorff_time * 100.0 / epoch_time,
	}
		
	return avg_loss, avg_loss_components, epoch_dice_score, epoch_hausdorff, time_log
	


def evaluate_valid(dataloader):
	'''
	function meant to obtain per slice loss, dice score and per caseday hausdorff 
	at the end after training to evaluate the model on validation data
	'''
	
	model.eval()

	with torch.no_grad():

		per_slice_results, per_caseday_results = [], []
		pred_masks_dict, masks_dict = {}, {}
		total_confusion_matrix = torch.zeros((4, NUM_CLASSES), device=DEVICE, dtype=torch.int)

		pred_label_cutoffs = torch.tensor(PRED_LABEL_CUTOFFS, device=DEVICE, dtype=torch.float)
		pred_mask_cutoffs = torch.tensor(PRED_MASK_CUTOFFS, device=DEVICE, dtype=torch.float)
		for data in dataloader:
			
			imgs, masks, ids = data
			imgs, masks = imgs.to(DEVICE, dtype=torch.float), masks.to(DEVICE, dtype=torch.float)
			
			with autocast('cuda'):
				pred_masks, pred_labels = model(imgs)

			#loop through predictions and true masks to create 3D volume with all slices per caseday for Hausdorff computation
			# also use this loop to get slicewise loss and dice score

			for p, pl, m, id_ in zip(pred_masks, pred_labels, masks, ids):
				loss, _ = loss_fn(p.unsqueeze(0), pl.unsqueeze(0), m.unsqueeze(0))

				pl = torch.sigmoid(pl.float())
				p = (torch.sigmoid(p.float()) > pred_mask_cutoffs[:, None, None]).int()
				p = p * (pl > pred_label_cutoffs)[:,None,None].int()   #(C,H,W)
				m = m.int()                                       #(C,H,W)

				presence_true = m.amax((1,2))
				presence_pred = p.amax((1,2))

				TP = ((presence_pred == 1) & (presence_true == 1))
				FP = ((presence_pred == 1) & (presence_true == 0))
				FN = ((presence_pred == 0) & (presence_true == 1))
				TN = ((presence_pred == 0) & (presence_true == 0))

				total_confusion_matrix += torch.stack([TP,FP,FN,TN])
				
				dice_val = dice_score_obj.compute_per_image(p, m)

				per_slice_results.append((id_, loss.item(), dice_val))

				match = re.match(r"case(\d+)_day(\d+)_slice_(\d+)", id_)
				if match:
					caseid, dayid, sliceid = map(int, match.groups())

				casedayid = (caseid, dayid) 

				pred_masks_dict.setdefault(casedayid, []).append((sliceid, p))
				masks_dict.setdefault(casedayid, []).append((sliceid, m))

				#in the data, casedays have either 144 slices or 80 slices
				if (len(pred_masks_dict[casedayid]) == 144) or (casedayid in slices80_casedays and len(pred_masks_dict[casedayid]) == 80):
					pred_masks_sorted = [p.cpu().numpy() for sid, p in sorted(pred_masks_dict[casedayid], key=lambda x: x[0])]
					masks_sorted = [m.cpu().numpy() for sid, m in sorted(masks_dict[casedayid], key=lambda x: x[0])]

					pred_masks_volume = np.stack(pred_masks_sorted, axis=1)
					masks_volume = np.stack(masks_sorted, axis=1)

					hausdorff_val = hausdorff_obj.compute_per_caseday(pred_masks_volume, masks_volume)
					per_caseday_results.append((f'{caseid}_{dayid}', hausdorff_val))

					#free memory
					del pred_masks_dict[casedayid], masks_dict[casedayid]

	df_slice_results = pd.DataFrame(per_slice_results, columns=['id', 'loss', 'dice'])
	df_caseday_results = pd.DataFrame(per_caseday_results, columns=['caseday', 'hausdorff'])

	return df_slice_results, df_caseday_results, total_confusion_matrix.detach().cpu().numpy()



# Monitor resources
if TRAIN_VALID_SPLIT or TRAIN_ON_FULL_DATA:
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    stats_log = []
    stop_event = threading.Event()
    
    def monitor_resources(interval = 5):
        while not stop_event.is_set():
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
            gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used / 1024**2
            cpu_util = psutil.cpu_percent(interval = None)
            cpu_mem = psutil.virtual_memory().used / 1024**2
            stats_log.append({
                'time': time.time(),
                'gpu_util': gpu_util,
                'gpu_mem_MB': gpu_mem,
                'cpu_util': cpu_util,
                'cpu_mem_MB': cpu_mem
            })
            time.sleep(interval)
    
    # Start monitoring
    thread = threading.Thread(target = monitor_resources)
    thread.start()


if TRAIN_VALID_SPLIT:
	
	del caseday_replay_transform, dataset_train, dataloader_train
	gc.collect()

	time_logs_train, time_logs_valid, loss_metric_logs = [], [], []
	train_loss_cols = ' '.join(f'TLo-{name:<8}' for name in LOSS_NAMES)
	valid_loss_cols = ' '.join(f'VLo-{name:<8}' for name in LOSS_NAMES)
	header = (
		f"{'Epoch':<5} | {'LR':<9} | "
		f"{'Train Loss':<12} | {'Valid Loss':<12} | {'Combined Metric':<15} | "
		f"{'Dice Overall':<12} | {'Hausdorff Overall':<17} | "
		f"{train_loss_cols}| {valid_loss_cols}| "
		f"{'D-LB':<9} {'D-SB':<9} {'D-S':<9} | "
		f"{'H-LB':<9} {'H-SB':<9} {'H-S':<9}"
	)
	print(header)
	loss_metric_logs.append(header)
	loss_metric_logs.append('-' * len(header))
	max_combined_metric = 0.0

	for epoch in range(EPOCHS):
		scheduler.step() #calling step at the start so that the LR_END value will be used in the last epoch 

		#recreating caseday_replay_transform to get new frozen transforms for new epoch
		#recreating dataset and dataloader to start using the updated caseday_replay_transform
		caseday_replay_transform = create_caseday_replay_transform(data_train, transform_train)
		dataset_train = GITractDataset(data_train, dataset_type='train', transforms=transform_train)
		dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE_TRAIN, shuffle=True, 
									  num_workers=DATA_LOADER_NUM_WORKERS, pin_memory=True)
		
		loss_train, loss_components_train, time_log_train = one_epoch_train(epoch, dataloader_train)
		del caseday_replay_transform, dataset_train, dataloader_train
		gc.collect()
		
		loss_valid, loss_components_valid, dice_score, hausdorff, time_log_valid = one_epoch_valid(epoch, dataloader_valid)
		
		dice_overall, dice_per_organ = dice_score
		hausdorff_overall, hausdorff_per_organ = hausdorff
		combined_metric = 0.4*dice_overall + 0.6*(1-hausdorff_overall)

		train_loss_str = ' '.join(f'{x:<12.6f}' for x in loss_components_train)
		valid_loss_str = ' '.join(f'{x:<12.6f}' for x in loss_components_valid)
		loss_metric_log = (
			f"{epoch+1:<5} | {optimizer.param_groups[6]['lr']:<9.6f} | "
			f"{loss_train:<12.6f} | {loss_valid:<12.6f} | {combined_metric:<15.6f} | "
			f"{dice_overall:<12.6f} | {hausdorff_overall:<17.6f} | "
			f"{train_loss_str}| {valid_loss_str}| "
			f"{dice_per_organ[0]:<9.6f} {dice_per_organ[1]:<9.6f} {dice_per_organ[2]:<9.6f} | "
			f"{hausdorff_per_organ[0]:<9.6f} {hausdorff_per_organ[1]:<9.6f} {hausdorff_per_organ[2]:<9.6f}"
		)
		
		print(loss_metric_log)
		for group in optimizer.param_groups:
			print(f"Epoch {epoch+1} - Group {group['name']} LR: {group['lr']:.4e} WD: {group['weight_decay']:.0e}")

		time_logs_train.append(time_log_train)
		time_logs_valid.append(time_log_valid)
		loss_metric_logs.append(loss_metric_log)

		if SAVE_TRAIN_VALID_MODEL and (epoch+1 > EPOCHS//2):
			if epoch+1 == EPOCHS:
				torch.save(model.state_dict(), MODEL_PARAMS_FILE_NAME)
				torch.save(ema_model.module.state_dict(), f'EMA-{MODEL_PARAMS_FILE_NAME}')
			elif combined_metric >= max_combined_metric:
				max_combined_metric = combined_metric
				torch.save(model.state_dict(), f'Epoch{epoch+1}-{MODEL_PARAMS_FILE_NAME}')
				torch.save(ema_model.module.state_dict(), f'EMA-Epoch{epoch+1}-{MODEL_PARAMS_FILE_NAME}')

	print('\n'.join(loss_metric_logs))

	time_logs_df = pd.merge(
						pd.DataFrame(time_logs_train),
						pd.DataFrame(time_logs_valid),
						on = 'epoch', how = 'inner'
				   )[[
						'epoch', 
						't_epoch_time', 'v_epoch_time', 
						't_data_time', 'v_data_time', 
						't_gpu_time', 'v_gpu_time',
						'v_hausdorff_time',
						't_data_perc', 'v_data_perc', 
						't_gpu_perc', 'v_gpu_perc',
						'v_hausdorff_perc'
					]]
	display(time_logs_df)


# #uncomment this cell for debugging (assuming in Global variables TEST_PREDICT = True, LOAD_MODEL_FOR_TEST_PREDICT = True)
# TRAIN_VALID_SPLIT = True
# TEST_PREDICT = False
# model.load_state_dict(torch.load(MODEL_PARAMS_LOAD_FILE_PATH))


if TRAIN_VALID_SPLIT and EXPLORE_CUTOFFS:
	loss_metric_logs = []
	valid_loss_cols = ' '.join(f'VLo-{name:<8}' for name in LOSS_NAMES)
	header = (
		f"{'Label Cutoffs(LB,SB,S)':<22} | {'Mask Cutoffs(LB,SB,S)':<21} | "
		f"{'Valid Loss':<12} | {'Combined Metric':<15} | "
		f"{'Dice Overall':<12} | {'Hausdorff Overall':<17} | "
		f"{valid_loss_cols}| "
		f"{'D-LB':<9} {'D-SB':<9} {'D-S':<9} | "
		f"{'H-LB':<9} {'H-SB':<9} {'H-S':<9}"
	)
	loss_metric_logs.append(header)
	loss_metric_logs.append('-' * len(header))
	best_combined_metric = 0.0
	best_pred_label_cutoffs = (0.0, 0.0, 0.0)
	best_pred_mask_cutoffs = (0.0, 0.0, 0.0)
	for PRED_LABEL_CUTOFFS, PRED_MASK_CUTOFFS in itertools.product(CUTOFFS_LIST, repeat = 2):

		loss_valid, loss_components_valid, dice_score, hausdorff, _ = one_epoch_valid(EPOCHS, dataloader_valid)
		
		dice_overall, dice_per_organ = dice_score
		hausdorff_overall, hausdorff_per_organ = hausdorff
		combined_metric = 0.4*dice_overall + 0.6*(1-hausdorff_overall)

		if combined_metric > best_combined_metric:
			best_combined_metric = combined_metric
			best_pred_label_cutoffs = PRED_LABEL_CUTOFFS
			best_pred_mask_cutoffs = PRED_MASK_CUTOFFS

		l_LB, l_SB, l_S = PRED_LABEL_CUTOFFS
		pred_label_cutoffs_str = f'({l_LB:.2f},{l_SB:.2f},{l_S:.2f})'
		m_LB, m_SB, m_S = PRED_MASK_CUTOFFS
		pred_mask_cutoffs_str = f'({m_LB:.2f},{m_SB:.2f},{m_S:.2f})'
		valid_loss_str = ' '.join(f'{x:<12.6f}' for x in loss_components_valid)
		loss_metric_log = (
			f"{pred_label_cutoffs_str:<22} | {pred_mask_cutoffs_str:<21} |"
			f"{loss_valid:<12.6f} | {combined_metric:<15.6f} | "
			f"{dice_overall:<12.6f} | {hausdorff_overall:<17.6f} | "
			f"{valid_loss_str}| "
			f"{dice_per_organ[0]:<9.6f} {dice_per_organ[1]:<9.6f} {dice_per_organ[2]:<9.6f} | "
			f"{hausdorff_per_organ[0]:<9.6f} {hausdorff_per_organ[1]:<9.6f} {hausdorff_per_organ[2]:<9.6f}"
		)
		loss_metric_logs.append(loss_metric_log)

		gc.collect()
		torch.cuda.empty_cache()

	print('\n'.join(loss_metric_logs))

	print(best_pred_label_cutoffs, best_pred_mask_cutoffs, best_combined_metric)
	PRED_LABEL_CUTOFFS = best_pred_label_cutoffs
	PRED_MASK_CUTOFFS = best_pred_mask_cutoffs


#to fetch a slice from a caseday, width-height that has max nonna organs
nonna_seg_info = (
    data_valid[['id', 'case', 'day', 'slice', 'slice_w', 'slice_h', 'segmentation']]
     .groupby(['id', 'case', 'day', 'slice', 'slice_w', 'slice_h'])['segmentation']
     .apply(lambda s: s.notna().sum())
     .reset_index(name='count').sort_values(by='count', ascending=False)
)
nonna_seg_info



if TRAIN_VALID_SPLIT:

	df_slice_results, df_caseday_results, total_confusion_matrix = evaluate_valid(dataloader_valid)

	max3_loss = df_slice_results.nlargest(3, 'loss')
	min3_loss = df_slice_results.loc[df_slice_results.loss != 0.0, :].nsmallest(3, 'loss')
	max3_dice = df_slice_results.nlargest(3, 'dice')
	min3_dice = df_slice_results.loc[df_slice_results.dice != 0.0, :].nsmallest(3, 'dice')
	max_hausdorff = df_caseday_results.nlargest(1,'hausdorff') 
	min_hausdorff = df_caseday_results.nsmallest(1,'hausdorff')

	#choose 1 slice each with max number of organs from the different slice sizes in valid data 
	specific_slices_to_show_max = []
	for slice_w, slice_h in [(266, 266), (360, 310), (276, 276)]:
		nonna_seg_info_sub = nonna_seg_info.loc[(nonna_seg_info.slice_w == slice_w) & (nonna_seg_info.slice_h == slice_h), :]
		chosen_id = nonna_seg_info_sub.loc[nonna_seg_info_sub['count'].idxmax(), 'id']
		specific_slices_to_show_max.append(chosen_id)

	#choose 1 slice each with min number of organs from the different slice sizes in valid data 
	specific_slices_to_show_min = []
	for slice_w, slice_h in [(266, 266), (360, 310), (276, 276)]:
		nonna_seg_info_sub = nonna_seg_info.loc[(nonna_seg_info.slice_w == slice_w) & (nonna_seg_info.slice_h == slice_h), :]
		chosen_id = nonna_seg_info_sub.loc[nonna_seg_info_sub['count'].idxmin(), 'id']
		specific_slices_to_show_min.append(chosen_id)

	#choose 1 slice each with max number of organs from the different casedays with max/min hausdorff
	#and then include one slice prior and after 
	caseday_id_dict = {}
	hausdorff_slices = []
	for caseday in list(max_hausdorff['caseday']) + list(min_hausdorff['caseday']):
		case, day = caseday.split('_')
		nonna_seg_info_sub = nonna_seg_info.loc[(nonna_seg_info.case == int(case)) & (nonna_seg_info.day == int(day)), :]
		chosen_id_df = nonna_seg_info_sub.nlargest(1, 'count')[['id', 'slice']]
		chosen_slice = chosen_id_df.slice.values[0]
		chosen_id_df = (nonna_seg_info_sub.loc[nonna_seg_info_sub.slice.isin([chosen_slice-1, chosen_slice, chosen_slice+1]), ['id']]
                                          .sort_values(by='id'))
		
		caseday_id_dict[caseday] = list(chosen_id_df.id)
		hausdorff_slices.extend(list(chosen_id_df.id))

	id_list = (
		specific_slices_to_show_max + specific_slices_to_show_min + 
		hausdorff_slices +
		list(max3_loss['id']) + list(min3_loss['id']) +
		list(max3_dice['id']) + list(min3_dice['id'])
	)

	specific_slices_max_df = (
		df_slice_results.loc[df_slice_results['id'].isin(specific_slices_to_show_max), :]
		.set_index('id')
		.loc[specific_slices_to_show_max]
		.reset_index()
	)
	specific_slices_min_df = (
		df_slice_results.loc[df_slice_results['id'].isin(specific_slices_to_show_min), :]
		.set_index('id')
		.loc[specific_slices_to_show_min]
		.reset_index()
	)

	max_hausdorff = max_hausdorff.loc[max_hausdorff.index.repeat(3)].reset_index(drop=True)
	min_hausdorff = min_hausdorff.loc[min_hausdorff.index.repeat(3)].reset_index(drop=True)

	max_hausdorff.insert(1, 'id', [id_ for id_ in caseday_id_dict[max_hausdorff.caseday[0]]])
	min_hausdorff.insert(1, 'id', [id_ for id_ in caseday_id_dict[min_hausdorff.caseday[0]]])
	max_hausdorff = pd.merge(max_hausdorff, df_slice_results, on = 'id')
	min_hausdorff = pd.merge(min_hausdorff, df_slice_results, on = 'id')

	def insert_caseday_hausdorff(loss_dice_df):
		loss_dice_df.insert(1, 'caseday', loss_dice_df['id'].str.extract(r'case(\d+)_day(\d+)_slice_\d+').agg('_'.join, axis=1))
		loss_dice_df = pd.merge(loss_dice_df, df_caseday_results, on = 'caseday')
		return loss_dice_df

	specific_slices_max_df = insert_caseday_hausdorff(specific_slices_max_df)
	specific_slices_min_df = insert_caseday_hausdorff(specific_slices_min_df)
	max3_loss = insert_caseday_hausdorff(max3_loss)
	min3_loss = insert_caseday_hausdorff(min3_loss)
	max3_dice = insert_caseday_hausdorff(max3_dice)
	min3_dice = insert_caseday_hausdorff(min3_dice)

	print('Chosen slices with all organs')
	display(specific_slices_max_df)
	print('Chosen slices with no organs')
	display(specific_slices_min_df)
	print('Max Loss slices')
	display(max3_loss)
	print('Min Loss slices')
	display(min3_loss)
	print('Max Dice slices')
	display(max3_dice)
	print('Min Dice Slices')
	display(min3_dice)
	print('Slices with all organs from max hausdorff caseday slices')
	display(max_hausdorff)
	print('Slices with all organs from min hausdorff caseday slices')
	display(min_hausdorff)


def show_group(df_group, display_dict, valid_data_id_to_shape, title):
	n = len(df_group)
	fig, axes = plt.subplots(2, n, figsize=(5*n, 8))
	fig.suptitle(title, fontsize=18)
	
	if n == 1:
		axes = axes.reshape(2, 1)  # handle single-column case

	fig.text(0.02, 0.6, 'True Mask', ha='left', va='center',
			fontsize=14, rotation=90)
	fig.text(0.02, 0.25, 'Pred Mask', ha='left', va='center',
			fontsize=14, rotation=90)

	handles = [
		Rectangle((0, 0), 1, 1, color=CMAP1(1.0)),
		Rectangle((0, 0), 1, 1, color=CMAP2(1.0)),
		Rectangle((0, 0), 1, 1, color=CMAP3(1.0))
	]
	labels = ['Large Bowel', 'Small Bowel', 'Stomach']
	fig.legend(
		handles, labels,
		loc = 'upper center',            
		bbox_to_anchor = (0.5, 0.95),    
		ncol = 3,                        # single row for legend
		fontsize = 12
	)

	for col, (_, row) in enumerate(df_group.iterrows()):
		id_, loss, dice, hausdorff = row['id'], row['loss'], row['dice'], row['hausdorff']
		d = display_dict[id_]
		img, mask, pred, pred_label = d['i'], d['m'], d['p'], d['pl']
		h, w = valid_data_id_to_shape[id_]

		# Row 0: Image with TRUE mask
		ax = axes[0, col]
		ax.imshow(img, cmap='bone')
		ax.imshow(mask[..., 0], cmap=CMAP1)
		ax.imshow(mask[..., 1], cmap=CMAP2)
		ax.imshow(mask[..., 2], cmap=CMAP3)
		title_text = f'{id_}\n({w}x{h})\n\nLoss: {loss:.9f}\nDice: {dice:.6f}\n'
		title_text += f'Hausdorff (3d - all caseday slices) : {hausdorff:.6f}\n'
		title_text += f'Presence Label : {pred_label[0]:.3f} LB, {pred_label[1]:.3f} SB, {pred_label[2]:.3f} S'
		ax.set_title(title_text)
		ax.axis('off')

		# Row 1: Image with PRED mask
		ax = axes[1, col]
		ax.imshow(img, cmap='bone')
		ax.imshow(pred[..., 0], cmap=CMAP1)
		ax.imshow(pred[..., 1], cmap=CMAP2)
		ax.imshow(pred[..., 2], cmap=CMAP3)
		ax.axis('off')

	plt.tight_layout(rect=[0, 0, 1, 0.95])
	plt.show()



if TRAIN_VALID_SPLIT:

	display_dict = {}
	model.eval()
	with torch.no_grad():

		pred_label_cutoffs = torch.tensor(PRED_LABEL_CUTOFFS, device=DEVICE, dtype=torch.float)
		pred_mask_cutoffs = torch.tensor(PRED_MASK_CUTOFFS, device=DEVICE, dtype=torch.float)
		for data in dataloader_valid:
			
			imgs, masks, ids = data
			imgs, masks = imgs.to(DEVICE, dtype=torch.float), masks.to(DEVICE, dtype=torch.float)

			with autocast('cuda'):
				pred_masks, pred_labels = model(imgs)

			pred_labels = torch.sigmoid(pred_labels.float())
			pred_masks = (torch.sigmoid(pred_masks.float()) > pred_mask_cutoffs[:, None, None]).int()   #pred_masks.float() to convert from FP16 to FP32
			pred_masks = pred_masks * (pred_labels > pred_label_cutoffs)[...,None,None].int()
			masks = masks.int()

			for p, pl, m, i, id_ in zip(pred_masks, pred_labels, masks, imgs, ids): #HWC
				if id_ in id_list:
					i = i.permute(1,2,0)    #after the permute, img is in HxWxC format
					i = i * torch.tensor(IMAGE_NORMALIZE_SD).to(DEVICE, dtype=torch.float) + torch.tensor(IMAGE_NORMALIZE_MEAN).to(DEVICE, dtype=torch.float)
					i = i.clamp(0, 1).cpu().numpy()
					i = (i * 255).astype(np.uint8) 
					#i is in HxWxC format with C=IMAGE_CHANNELS. 
					#As part of 2.5D image creation <IMAGE_CHANNELS> neighbouring slices are kept as <IMAGE_CHANNELS> channels. 
					#We need the middle slice only for display
					i = i[:,:,IMAGE_CHANNELS//2].squeeze()
	
					m = m.permute(1,2,0).cpu().numpy()
					p = p.permute(1,2,0).cpu().numpy()
					display_dict[id_] = {'p': p, 'm': m, 'i': i, 'pl': pl}

	valid_data_id_to_shape = dict(zip(data_valid['id'], zip(data_valid['slice_h'], data_valid['slice_w'])))


if TRAIN_VALID_SPLIT:
	show_group(specific_slices_max_df, display_dict, valid_data_id_to_shape, 'Predictions on chosen slices with all organs')
	print('\n' * 3)
	show_group(specific_slices_min_df, display_dict, valid_data_id_to_shape, 'Predictions on chosen slices with no organs')
	print('\n' * 3)
	show_group(min3_loss, display_dict, valid_data_id_to_shape, 'Predictions with best(lowest) loss')
	print('\n' * 3)
	show_group(max3_loss, display_dict, valid_data_id_to_shape, 'Predictions with worst(highest) loss')
	print('\n' * 3)
	show_group(max3_dice, display_dict, valid_data_id_to_shape, 'Predictions with best dice')
	print('\n' * 3)
	show_group(min3_dice, display_dict, valid_data_id_to_shape, 'Predictions with worst dice')
	print('\n' * 3)
	show_group(min_hausdorff, display_dict, valid_data_id_to_shape, 'Predictions with best(lowest) hausdorff (on large organ count slices of the caseday)')
	print('\n' * 3)
	show_group(max_hausdorff, display_dict, valid_data_id_to_shape, 'Predictions with worst(highest) hausdorff (on large organ count slices of the caseday)')


if TRAIN_VALID_SPLIT:
	rows = ['TP', 'FP', 'FN', 'TN']
	cols = ['LB', 'SB', 'S']
	df_raw = pd.DataFrame(total_confusion_matrix, index=rows, columns=cols)
	df_norm = df_raw / df_raw.sum(axis=0)

	print('Confusion Matrix raw counts')
	print(df_raw.to_string())
	print('Confusion Matrix normalized per class')
	print(df_norm.round(3).to_string())


if TRAIN_VALID_SPLIT:
	empty_mask_slices = nonna_seg_info.loc[nonna_seg_info['count'] == 0, 'id']
	empty_mask_slices_pred_perf = (
		df_slice_results.loc[df_slice_results['id'].isin(empty_mask_slices), :]
		.set_index('id')
		.loc[empty_mask_slices]
		.reset_index()
	)
	display(empty_mask_slices_pred_perf.sort_values(by='loss', ascending=False))
		
	display(empty_mask_slices_pred_perf.loc[empty_mask_slices_pred_perf.loss > 1e-3, :].sort_values(by='loss', ascending=False))


#listing slices before&after specific slices thats empty masks but gave high prediction loss
# display_multiple_slices(data.query("case == 44 and day == 20 and slice > 115 and slice <= 120").id.unique(), 
#                         id_dicts, apply_CLAHE=True)
# display_multiple_slices(data.query("case == 149 and day == 15 and slice > 105 and slice <= 115").id.unique(), 
#                         id_dicts, apply_CLAHE=True)
# display_multiple_slices(data.query("case == 89 and day == 21 and slice > 60 and slice <= 70").id.unique(), 
#                         id_dicts, apply_CLAHE=True)


# we check the valid loss/metrics on different sized subsets of validation data
if TRAIN_VALID_SPLIT:
	loss_metric_logs = []
	valid_loss_cols = ' '.join(f'VLo-{name:<8}' for name in LOSS_NAMES)
	header = (
		f"{'Slice_W':<7} | {'Slice_H':<7} | "
		f"{'Valid Loss':<12} | {'Combined Metric':<15} | "
		f"{'Dice Overall':<12} | {'Hausdorff Overall':<17} | "
		f"{valid_loss_cols}| "
		f"{'D-LB':<9} {'D-SB':<9} {'D-S':<9} | "
		f"{'H-LB':<9} {'H-SB':<9} {'H-S':<9}"
	)
	loss_metric_logs.append(header)
	loss_metric_logs.append('-' * len(header))
	#obtain valid loss/metrics for whole data(-1, -1) and for specific sized slices
	for (slice_w, slice_h) in [(-1, -1), (266, 266), (360, 310), (276, 276)]:
		if slice_w != -1 and slice_h != -1:
			del dataset_valid, dataloader_valid
			gc.collect()
			data_valid_sub = data_valid.loc[(data_valid.slice_w == slice_w) & (data_valid.slice_h == slice_h), :]
			dataset_valid = GITractDataset(data_valid_sub, dataset_type='valid', transforms=transform_valid)
			dataloader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE_VALID, shuffle=False, num_workers=DATA_LOADER_NUM_WORKERS, pin_memory=True)

		loss_valid, loss_components_valid, dice_score, hausdorff, _ = one_epoch_valid(EPOCHS, dataloader_valid)
		
		dice_overall, dice_per_organ = dice_score
		hausdorff_overall, hausdorff_per_organ = hausdorff
		combined_metric = 0.4*dice_overall + 0.6*(1-hausdorff_overall)

		valid_loss_str = ' '.join(f'{x:<12.6f}' for x in loss_components_valid)
		loss_metric_log = (
			f"{slice_w:<7} | {slice_h:<7} | "
			f"{loss_valid:<12.6f} | {combined_metric:<15.6f} | "
			f"{dice_overall:<12.6f} | {hausdorff_overall:<17.6f} | "
			f"{valid_loss_str}| "
			f"{dice_per_organ[0]:<9.6f} {dice_per_organ[1]:<9.6f} {dice_per_organ[2]:<9.6f} | "
			f"{hausdorff_per_organ[0]:<9.6f} {hausdorff_per_organ[1]:<9.6f} {hausdorff_per_organ[2]:<9.6f}"
		)
		loss_metric_logs.append(loss_metric_log)

	print('\n'.join(loss_metric_logs))


if TRAIN_ON_FULL_DATA:

	del caseday_replay_transform, dataset_train, dataloader_train
	gc.collect()
	
	time_logs_train, loss_logs = [], []
	train_loss_cols = ' '.join(f'TLo-{name:<8}' for name in LOSS_NAMES)
	header = (
		f"{'Epoch':<5} | {'LR':<9} | {'Train Loss':<12} | "
		f"{train_loss_cols}"
	)
	print(header)
	loss_logs.append(header)
	loss_logs.append('-' * len(header))

	for epoch in range(EPOCHS):
		scheduler.step()
		
		caseday_replay_transform = create_caseday_replay_transform(data, transform_train)
		dataset_train_full = GITractDataset(data, dataset_type='train', transforms=transform_train)
		dataloader_train_full = DataLoader(dataset_train_full, batch_size=BATCH_SIZE_TRAIN, shuffle=True,
										   num_workers=DATA_LOADER_NUM_WORKERS, pin_memory=True)
		
		loss_train, loss_components_train, time_log_train = one_epoch_train(epoch, dataloader_train_full)
		del caseday_replay_transform, dataset_train_full, dataloader_train_full
		gc.collect()

		train_loss_str = ' '.join(f'{x:<12.6f}' for x in loss_components_train)
		loss_log = (
			f"{epoch+1:<5} | {optimizer.param_groups[6]['lr']:<9.6f} | {loss_train:<12.6f} | "
			f"{train_loss_str}"
		)
		print(loss_log)
		for group in optimizer.param_groups:
			print(f"Epoch {epoch+1} - Group {group['name']} LR: {group['lr']:.4e} WD: {group['weight_decay']:.0e}")

		time_logs_train.append(time_log_train)
		loss_logs.append(loss_log)

		if SAVE_FULL_DATA_MODEL and (epoch+1 > EPOCHS//2):
			if epoch+1 == EPOCHS:
				torch.save(model.state_dict(), MODEL_PARAMS_FULL_DATA_FILE_NAME)
				torch.save(ema_model.module.state_dict(), f'EMA-{MODEL_PARAMS_FULL_DATA_FILE_NAME}')
			else:
				torch.save(model.state_dict(), f'Epoch{epoch+1}-{MODEL_PARAMS_FULL_DATA_FILE_NAME}')
				torch.save(ema_model.module.state_dict(), f'EMA-Epoch{epoch+1}-{MODEL_PARAMS_FULL_DATA_FILE_NAME}')

	print('\n'.join(loss_logs))

	time_logs_df = pd.DataFrame(time_logs_train)
	display(time_logs_df)


if TRAIN_VALID_SPLIT or TRAIN_ON_FULL_DATA:
    stop_event.set()
    thread.join()

    df_stats = pd.DataFrame(stats_log)
    display(df_stats.describe())  # summary stats


if TEST_PREDICT: 
	if LOAD_MODEL_FOR_TEST_PREDICT:
		model.load_state_dict(torch.load(MODEL_PARAMS_LOAD_FILE_PATH))
	model.eval()

	if ENSEMBLE_MODEL_PREDICT:
		ensemble_models[0].load_state_dict(torch.load(ENSEMBLE_MODEL_PARAMS_LOAD_FILE_PATH[0]))    	
		ensemble_models[0].eval()

	data_test = pd.read_csv(DIR_PATH + 'sample_submission.csv')
	test_set_hidden = not bool(len(data_test))
	if test_set_hidden:
		data_test = data_valid  # Use validation data for testing the code prior to submission
	else:
		data_test[['case', 'day', 'slice']] = data_test['id'].str.extract(r'case(\d+)_day(\d+)_slice_(\d+)')
		path_df = get_path_df(train = False)

		data_test = data_test.merge(path_df, on = ['case', 'day', 'slice'])
	
		int_cols = ['case', 'day', 'slice', 'slice_w', 'slice_h']
		data_test[int_cols] = data_test[int_cols].astype(np.uint32)
	
		float_cols = ['px_w', 'px_h']
		data_test[float_cols] = data_test[float_cols].astype(np.float32)

	transform_test = A.Compose([
		A.Resize(IMAGE_RESIZE[0], IMAGE_RESIZE[1], interpolation=cv2.INTER_NEAREST),
		A.Normalize(mean=IMAGE_NORMALIZE_MEAN, std=IMAGE_NORMALIZE_SD, max_pixel_value=1.0),
		A.ToTensorV2(transpose_mask = False),
	])    
	dataset_test = GITractDataset(data_test, dataset_type='test', transforms=transform_test, load_resized=False)
	dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=DATA_LOADER_NUM_WORKERS)


if TEST_PREDICT:
	test_ids, test_class, test_pred_RLE = [], [], []   # data to be written to submission file
	
	with torch.no_grad():
		
		pred_label_cutoffs = torch.tensor(PRED_LABEL_CUTOFFS, device=DEVICE, dtype=torch.float)
		pred_mask_cutoffs = torch.tensor(PRED_MASK_CUTOFFS, device=DEVICE, dtype=torch.float)
		if ENSEMBLE_MODEL_PREDICT:
			ensemble_pred_label_cutoffs = torch.tensor(ENSEMBLE_PRED_LABEL_CUTOFFS, device=DEVICE, dtype=torch.float) 
		for imgs, ids, heights, widths, paddings in dataloader_test:
			imgs = imgs.to(DEVICE, dtype=torch.float)

			with autocast('cuda'):
				pred_masks, pred_labels = model(imgs)
				if ENSEMBLE_MODEL_PREDICT:
					pred_masks_tmp, pred_labels_tmp = ensemble_models[0](imgs)
			pred_labels = torch.sigmoid(pred_labels.float())
			pred_masks = torch.sigmoid(pred_masks.float())
			pred_masks = pred_masks * (pred_labels > pred_label_cutoffs)[...,None,None].int()

			if ENSEMBLE_MODEL_PREDICT:
				pred_labels_tmp = torch.sigmoid(pred_labels_tmp.float())
				pred_masks_tmp = torch.sigmoid(pred_masks_tmp.float())
				pred_masks_tmp = pred_masks_tmp * (pred_labels_tmp > ensemble_pred_label_cutoffs[0])[...,None,None].int()

				pred_masks = torch.stack([pred_masks, pred_masks_tmp], dim=0).mean(dim=0)

			pred_masks = (pred_masks > pred_mask_cutoffs[:, None, None]).int()
			pred_masks = pred_masks.permute(0, 2, 3, 1).cpu().numpy()   # shape after permute [B, H, W, C]

			for mask, id_, h, w, p in zip(pred_masks, ids, heights, widths, paddings):
				post_pad_size = max(h.item(), w.item())
				mask = cv2.resize(mask, dsize=(post_pad_size, post_pad_size), interpolation=cv2.INTER_NEAREST)
				mask = undo_padding(mask, p)
				rles = [rle_encode(mask[..., chid]) for chid in range(NUM_CLASSES)]
				
				test_ids.extend([id_] * NUM_CLASSES)
				test_class.extend(CLASS_NAMES)
				test_pred_RLE.extend(rles)

	submission_df = pd.DataFrame({
		'id': test_ids, 
		'class': test_class, 
		'predicted': test_pred_RLE
	})
	submission_df.to_csv('submission.csv', index=False)
	!head submission.csv
	display(submission_df.loc[submission_df.predicted != ''].head())

