# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/imc2024-packages-lightglue-rerun-kornia'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install --no-index /kaggle/input/imc2024-packages-lightglue-rerun-kornia/* --no-deps
!mkdir -p /root/.cache/torch/hub/checkpoints
!cp /kaggle/input/aliked/pytorch/aliked-n16/1/* /root/.cache/torch/hub/checkpoints/
!cp /kaggle/input/lightglue/pytorch/aliked/1/* /root/.cache/torch/hub/checkpoints/
!cp /kaggle/input/lightglue/pytorch/aliked/1/aliked_lightglue.pth /root/.cache/torch/hub/checkpoints/aliked_lightglue_v0-1_arxiv-pth



# General utilities
import matplotlib.pyplot as plt

import os
from tqdm import tqdm
from time import time, sleep
from fastprogress import progress_bar
import gc
import numpy as np
import h5py
from IPython.display import clear_output
from collections import defaultdict
from copy import deepcopy

# CV/MLe
import cv2
import torch
import torch.nn.functional as F
import kornia as K
import kornia.feature as KF
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# 3D reconstruction
import pycolmap

# Data importing into colmap
import sys
sys.path.append('/kaggle/input/colmap-db-import')
from database import *
from h5_to_db import *



def arr_to_str(a):
    return ';'.join([str(x) for x in a.reshape(-1)])


def load_torch_image(fname, device=torch.device('cpu')):
    img = K.io.load_image(fname, K.io.ImageLoadType.RGB32, device=device)[None, ...]
    return img

device = K.utils.get_cuda_device_if_available(0)
print (device)


# We will use efficientnet global descriptor to get matching shortlists.
def get_global_desc(fnames, device = torch.device('cpu')):
    processor = AutoImageProcessor.from_pretrained('/kaggle/input/dinov2/pytorch/base/1')
    model = AutoModel.from_pretrained('/kaggle/input/dinov2/pytorch/base/1')
    model = model.eval()
    model = model.to(device)
    global_descs_dinov2 = []
    for i, img_fname_full in tqdm(enumerate(fnames),total= len(fnames)):
        key = os.path.splitext(os.path.basename(img_fname_full))[0]
        timg = load_torch_image(img_fname_full)
        with torch.inference_mode():
            inputs = processor(images=timg, return_tensors="pt", do_rescale=False).to(device)
            outputs = model(**inputs)
            dino_mac = F.normalize(outputs.last_hidden_state[:,1:].max(dim=1)[0], dim=1, p=2)
        global_descs_dinov2.append(dino_mac.detach().cpu())
    global_descs_dinov2 = torch.cat(global_descs_dinov2, dim=0)
    return global_descs_dinov2


def get_img_pairs_exhaustive(img_fnames):
    index_pairs = []
    for i in range(len(img_fnames)):
        for j in range(i+1, len(img_fnames)):
            index_pairs.append((i,j))
    return index_pairs


def get_image_pairs_shortlist(fnames,
                              sim_th = 0.6, # should be strict
                              min_pairs = 20,
                              exhaustive_if_less = 20,
                              device=torch.device('cpu')):
    num_imgs = len(fnames)
    if num_imgs <= exhaustive_if_less:
        return get_img_pairs_exhaustive(fnames)
    descs = get_global_desc(fnames, device=device)
    dm = torch.cdist(descs, descs, p=2).detach().cpu().numpy()
    # removing half
    mask = dm <= sim_th
    total = 0
    matching_list = []
    ar = np.arange(num_imgs)
    already_there_set = []
    for st_idx in range(num_imgs-1):
        mask_idx = mask[st_idx]
        to_match = ar[mask_idx]
        if len(to_match) < min_pairs:
            to_match = np.argsort(dm[st_idx])[:min_pairs]  
        for idx in to_match:
            if st_idx == idx:
                continue
            if dm[st_idx, idx] < 1000:
                matching_list.append(tuple(sorted((st_idx, idx.item()))))
                total+=1
    matching_list = sorted(list(set(matching_list)))
    return matching_list


import torch
from lightglue import match_pair
from lightglue import ALIKED, LightGlue
from lightglue.utils import load_image, rbd


def detect_aliked(img_fnames,
                  feature_dir = '.featureout',
                  num_features = 4096,
                  resize_to = 1024,
                  device=torch.device('cpu')):
    dtype = torch.float32 # ALIKED has issues with float16
    extractor = ALIKED(max_num_keypoints=num_features, detection_threshold=0.01, resize=resize_to).eval().to(device, dtype)
    if not os.path.isdir(feature_dir):
        os.makedirs(feature_dir)
    with h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f_kp, \
         h5py.File(f'{feature_dir}/descriptors.h5', mode='w') as f_desc:
        for img_path in tqdm(img_fnames):
            img_fname = img_path.split('/')[-1]
            key = img_fname
            with torch.inference_mode():
                image0 = load_torch_image(img_path, device=device).to(dtype)
                feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
                kpts = feats0['keypoints'].reshape(-1, 2).detach().cpu().numpy()
                descs = feats0['descriptors'].reshape(len(kpts), -1).detach().cpu().numpy()
                f_kp[key] = kpts
                f_desc[key] = descs
    return
def match_with_lightglue(img_fnames,
                   index_pairs,
                   feature_dir = '.featureout',
                   device=torch.device('cpu'),
                   min_matches=15,verbose=True):
    lg_matcher = KF.LightGlueMatcher("aliked", {"width_confidence": -1,
                                                "depth_confidence": -1,
                                                 "mp": True if 'cuda' in str(device) else False}).eval().to(device)
    with h5py.File(f'{feature_dir}/keypoints.h5', mode='r') as f_kp, \
        h5py.File(f'{feature_dir}/descriptors.h5', mode='r') as f_desc, \
        h5py.File(f'{feature_dir}/matches.h5', mode='w') as f_match:
        for pair_idx in tqdm(index_pairs):
            idx1, idx2 = pair_idx
            fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
            key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]
            kp1 = torch.from_numpy(f_kp[key1][...]).to(device)
            kp2 = torch.from_numpy(f_kp[key2][...]).to(device)
            desc1 = torch.from_numpy(f_desc[key1][...]).to(device)
            desc2 = torch.from_numpy(f_desc[key2][...]).to(device)
            with torch.inference_mode():
                dists, idxs = lg_matcher(desc1,
                                         desc2,
                                         KF.laf_from_center_scale_ori(kp1[None]),
                                         KF.laf_from_center_scale_ori(kp2[None]))
            if len(idxs)  == 0:
                continue
            n_matches = len(idxs)
            if verbose:
                print (f'{key1}-{key2}: {n_matches} matches')
            group  = f_match.require_group(key1)
            if n_matches >= min_matches:
                 group.create_dataset(key2, data=idxs.detach().cpu().numpy().reshape(-1, 2))
    return

def import_into_colmap(img_dir,
                       feature_dir ='.featureout',
                       database_path = 'colmap.db'):
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    single_camera = False
    fname_to_id = add_keypoints(db, feature_dir, img_dir, '', 'simple-pinhole', single_camera)
    add_matches(
        db,
        feature_dir,
        fname_to_id,
    )
    db.commit()
    return


src = '/kaggle/input/image-matching-challenge-2024/'
# Get data from csv.

data_dict = {}
with open(f'{src}/sample_submission.csv', 'r') as f:
    for i, l in enumerate(f):
        if i== 0:
            print (l)
        # Skip header.
        if l and i > 0:
            image_path, dataset, scene, _, _ = l.strip().split(',')
            if dataset not in data_dict:
                data_dict[dataset] = {}
            if scene not in data_dict[dataset]:
                data_dict[dataset][scene] = []
            data_dict[dataset][scene].append(image_path)
for dataset in data_dict:
    for scene in data_dict[dataset]:
        print(f'{dataset} / {scene} -> {len(data_dict[dataset][scene])} images')

out_results = {}
timings = {"shortlisting":[],
           "feature_detection": [],
           "feature_matching":[],
           "RANSAC": [],
           "Reconstruction": []}


# Function to create a submission file.
def create_submission(out_results, data_dict):
    with open(f'submission.csv', 'w') as f:
        f.write('image_path,dataset,scene,rotation_matrix,translation_vector\n')
        for dataset in data_dict:
            if dataset in out_results:
                res = out_results[dataset]
            else:
                res = {}
            for scene in data_dict[dataset]:
                if scene in res:
                    scene_res = res[scene]
                else:
                    scene_res = {"R":{}, "t":{}}
                for image in data_dict[dataset][scene]:
                    if image in scene_res:
                        print (image)
                        R = scene_res[image]['R'].reshape(-1)
                        T = scene_res[image]['t'].reshape(-1)
                    else:
                        R = np.eye(3).reshape(-1)
                        T = np.zeros((3))
                    f.write(f'{image},{dataset},{scene},{arr_to_str(R)},{arr_to_str(T)}\n')


gc.collect()
from copy import deepcopy
datasets = []
for dataset in data_dict:
    datasets.append(dataset)
print (f"Extracting on device {device}")
for dataset in data_dict:
    print(dataset)
    if dataset not in out_results:
        out_results[dataset] = {}
    for scene in data_dict[dataset]:
        print(scene)
        img_dir =  os.path.join(src,  '/'.join(data_dict[dataset][scene][0].split('/')[:-1]))
        # Fail gently if the notebook has not been submitted and the test data is not populated.
        # You may want to run this on the training data in that case?
        # Wrap the meaty part in a try-except block.
        try:
            out_results[dataset][scene] = {}
            img_fnames = [os.path.join(src, x) for x in data_dict[dataset][scene] ] 
            print (f"Got {len(img_fnames)} images")
            feature_dir = f'featureout/{dataset}_{scene}'
            os.makedirs(feature_dir, exist_ok=True)
            t=time()
            #if len(img_fnames) > 60:
            #    continue
            index_pairs = get_image_pairs_shortlist(img_fnames,
                                  sim_th = 0.3, # should be strict
                                  min_pairs = 20, # we select at least min_pairs PER IMAGE with biggest similarity
                                  exhaustive_if_less = 20,
                                  device=device)
            t=time() -t 
            timings['shortlisting'].append(t)
            print (f'{len(index_pairs)}, pairs to match, {t:.4f} sec')
            gc.collect()
            t=time()
            detect_aliked(img_fnames, feature_dir, 4096, device=device)
            gc.collect()
            t=time() -t 
            timings['feature_detection'].append(t)
            print(f'Features detected in  {t:.4f} sec')
            t=time()
            match_with_lightglue(img_fnames, index_pairs, feature_dir=feature_dir,device=device)
            t=time() -t 
            timings['feature_matching'].append(t)
            print(f'Features matched in  {t:.4f} sec')
            database_path = f'{feature_dir}/colmap.db'
            if os.path.isfile(database_path):
                os.remove(database_path)
            gc.collect()
            sleep(1)
            import_into_colmap(img_dir, feature_dir=feature_dir, database_path=database_path)
            output_path = f'{feature_dir}/colmap_rec_aliked'
            t=time()
            pycolmap.match_exhaustive(database_path)
            t=time() - t 
            timings['RANSAC'].append(t)
            print(f'RANSAC in  {t:.4f} sec')
            t=time()
            # By default colmap does not generate a reconstruction if less than 10 images are registered. Lower it to 3.
            mapper_options = pycolmap.IncrementalPipelineOptions()
            mapper_options.min_model_size = 3
            mapper_options.max_num_models = 2
            os.makedirs(output_path, exist_ok=True)
            maps = pycolmap.incremental_mapping(database_path=database_path, 
                                                image_path=img_dir,
                                                output_path=output_path, options=mapper_options)
            sleep(1)
            print(maps)
            clear_output(wait=False)
            t=time() - t
            timings['Reconstruction'].append(t)
            print(f'Reconstruction done in  {t:.4f} sec')
            imgs_registered  = 0
            best_idx = None
            print ("Looking for the best reconstruction")
            if isinstance(maps, dict):
                for idx1, rec in maps.items():
                    print (idx1, rec.summary())
                    try:
                        if len(rec.images) > imgs_registered:
                            imgs_registered = len(rec.images)
                            best_idx = idx1
                    except:
                        continue
            if best_idx is not None:
                print (maps[best_idx].summary())
                for k, im in maps[best_idx].images.items():
                    key1 = f'test/{scene}/images/{im.name}'
                    print(key1)
                    out_results[dataset][scene][key1] = {}
                    out_results[dataset][scene][key1]["R"] = deepcopy(im.cam_from_world.rotation.matrix())
                    out_results[dataset][scene][key1]["t"] = deepcopy(np.array(im.cam_from_world.translation))
            print(f'Registered: {dataset} / {scene} -> {len(out_results[dataset][scene])} images')
            print(f'Total: {dataset} / {scene} -> {len(data_dict[dataset][scene])} images')
            create_submission(out_results, data_dict)
            gc.collect()
        except Exception as e:
            print (e)
            pass


!cat submission.csv

