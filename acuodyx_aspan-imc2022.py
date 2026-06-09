dry_run = False


import os 
import time
import numpy as np
import pandas as pd
import sys, os, csv
from PIL import Image
import cv2
import gc
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
from IPython import display
import PIL
import sys
from pathlib import Path
import matplotlib.cm as cm
from copy import deepcopy
import kornia as K
import kornia.feature as KF
data_path = "/kaggle/input/image-matching-challenge-2022/"
src_path = "/kaggle/input/aspan-main/ASpanFormer" 
!pip install /kaggle/input/loftr-utils/loguru-0.6.0-py3-none-any.whl
!pip install /kaggle/input/loftr-utils/kornia_moons-0.2.9-py3-none-any.whl
!pip install /kaggle/input/loftr-utils/einops-0.4.1-py3-none-any.whl
!pip install /kaggle/input/loftr-utils/yacs-0.1.8-py3-none-any.whl
from kornia_moons.feature import *
from kornia_moons.viz import draw_LAF_matches



test_samples = []
with open(f'{data_path}/test.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for i, row in enumerate(reader):
        # Skip header.
        if i == 0:
            continue
        test_samples += [row]


def FlattenMatrix(M, num_digits=8):
    '''Convenience function to write CSV files.'''
    
    return ' '.join([f'{v:.{num_digits}e}' for v in M.flatten()])


def load_torch_image(fname, device):
    img = cv2.imread(fname)
    img = K.image_to_tensor(img, False).float() /255.
    img = K.color.bgr_to_rgb(img)
    return img.to(device)



sys.path.append(src_path)

from aspan import ASpanFormer
from aspan import full_default_cfg
full_default_cfg['coarse']['coarsest_level']=[32,32]
full_default_cfg['coarse']['train_res']=[832,832]
full_default_cfg['coarse']['test_res']=[1152,1152]
matcher = ASpanFormer(pretrained="/kaggle/input/aspan-main/ASpanFormer/outdoor.ckpt", config=full_default_cfg)

matcher = matcher.eval().cuda()


F_dict = {}
import time
for i, row in enumerate(test_samples):
    sample_id, batch_id, image_1_id, image_2_id = row
    # Load the images.
    st = time.time()
    image_1_raw = load_torch_image(f'{data_path}/test_images/{batch_id}/{image_1_id}.png', "cuda")
    image_2_raw = load_torch_image(f'{data_path}/test_images/{batch_id}/{image_2_id}.png', "cuda")
    image_1_resize = TF.resize(image_1_raw,(1152, 1152))
    image_2_resize = TF.resize(image_2_raw,(1152, 1152))
    b,c,h1_raw,w1_raw=image_1_raw.shape
    b,c,h1_resize,w1_resize=image_1_resize.shape
    b,c,h2_raw,w2_raw=image_2_raw.shape
    b,c,h2_resize,w2_resize=image_2_resize.shape
    scale1 = torch.FloatTensor([w1_raw/w1_resize,h1_raw/h1_resize]).reshape(1,2).to(image_1_raw.device)
    scale2 = torch.FloatTensor([w2_raw/w2_resize,h2_raw/h2_resize]).reshape(1,2).to(image_1_raw.device)
    input_dict = {"image0": K.color.rgb_to_grayscale(image_1_resize), 
              "image1": K.color.rgb_to_grayscale(image_2_resize)}

    with torch.no_grad():
        correspondences = matcher(input_dict)
        
    mkpts0 = (input_dict['mkpts0_f']*scale1).cpu().numpy()
    mkpts1 = (input_dict['mkpts1_f']*scale2).cpu().numpy()
    
    if len(mkpts0) > 7:
        F, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.200, 0.999999, 250000)
        inliers = inliers > 0
        assert F.shape == (3, 3), 'Malformed F?'
        F_dict[sample_id] = F
    else:
        F_dict[sample_id] = np.zeros((3, 3))
        continue
    gc.collect()
    nd = time.time()    
    # if (i < 3):
    #     print("Running time: ", nd - st, " s")
    #     draw_LAF_matches(
    #     KF.laf_from_center_scale_ori(torch.from_numpy(mkpts0).view(1,-1, 2),
    #                                 torch.ones(mkpts0.shape[0]).view(1,-1, 1, 1),
    #                                 torch.ones(mkpts0.shape[0]).view(1,-1, 1)),

    #     KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1).view(1,-1, 2),
    #                                 torch.ones(mkpts1.shape[0]).view(1,-1, 1, 1),
    #                                 torch.ones(mkpts1.shape[0]).view(1,-1, 1)),
    #     torch.arange(mkpts0.shape[0]).view(-1,1).repeat(1,2),
    #     K.tensor_to_image(image_1_raw),
    #     K.tensor_to_image(image_2_raw),
    #     inliers,
    #     draw_dict={'inlier_color': (0.2, 1, 0.2),
    #                'tentative_color': None, 
    #                'feature_color': (0.2, 0.5, 1), 'vertical': False})
    
with open('submission.csv', 'w') as f:
    f.write('sample_id,fundamental_matrix\n')
    for sample_id, F in F_dict.items():
        f.write(f'{sample_id},{FlattenMatrix(F)}\n')

