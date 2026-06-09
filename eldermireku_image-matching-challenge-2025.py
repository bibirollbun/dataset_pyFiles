import sys
import os
from tqdm import tqdm
from time import time, sleep
import gc
import numpy as np
import h5py
import dataclasses
import pandas as pd
from IPython.display import clear_output
from collections import defaultdict
from copy import deepcopy
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
import glob

# Path to the offline packages
package_dir = '/kaggle/input/imc2025-pycolmap-kornia'

# Install all .whl files in the directory
whl_files = glob.glob(os.path.join(package_dir, '*.whl'))
for whl in whl_files:
    print(f"Installing {whl}")
    os.system(f"pip install --no-index --find-links={package_dir} {whl} --no-deps")

# Create cache directory if needed
!mkdir -p /root/.cache/torch/hub/checkpoints

# Copy all .pth files if they exist in the same folder
!cp /kaggle/input/imc2025-pycolmap-kornia/*.pth /root/.cache/torch/hub/checkpoints/ 2>/dev/null || true
!mkdir -p /root/.cache/torch/hub/checkpoints
!cp /kaggle/input/aliked/pytorch/default/1/aliked-n16.pth /root/.cache/torch/hub/checkpoints/
!cp /kaggle/input/lightglue/pytorch/default/1/aliked_lightglue.pth /root/.cache/torch/hub/checkpoints/
!cp /kaggle/input/lightglue/pytorch/default/1/aliked_lightglue.pth /root/.cache/torch/hub/checkpoints/aliked_lightglue_v0-1_arxiv-pth

import kornia as K
import kornia.feature as KF
from lightglue import match_pair
from lightglue import ALIKED, LightGlue
from lightglue.utils import load_image, rbd
from transformers import AutoImageProcessor, AutoModel


# IMPORTANT Utilities: importing data into colmap and competition metric

import pycolmap
sys.path.append('/kaggle/input/imc25-utils')
from database import *
from h5_to_db import *
import metric

# Do not forget to select an accelerator on the sidebar to the right.
device = K.utils.get_cuda_device_if_available(0)
print(f'{device=}')

def load_torch_image(fname, device=torch.device('cpu')):
    img = K.io.load_image(fname, K.io.ImageLoadType.RGB32, device=device)[None, ...]
    return img

# CONFIGURATION
BATCH_SIZE = 64  # Increased batch size
EPOCHS = 20  # Increased epochs
LR = 1e-3  # Increased learning rate
REPORT_EVERY = 20  # Increased reporting frequency
NUM_WORKERS = 8 # Increased num_workers
IMG_SIZE = 384 # Increased image size
EPS = 0.3

# Must Use efficientnet global descriptor to get matching shortlists.
def get_global_desc(fnames, device = torch.device('cpu')):
    processor = AutoImageProcessor.from_pretrained('/kaggle/input/dino-v2/pytorch/default/1')
    model = AutoModel.from_pretrained('/kaggle/input/dino-v2/pytorch/default/1')
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
                              min_pairs = 30,
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
                   min_matches=20,verbose=True):
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

def import_into_colmap(img_dir, feature_dir ='.featureout', database_path = 'colmap.db'):
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
    
# Collect vital info from the dataset

@dataclasses.dataclass
class Prediction:
    image_id: str | None  # A unique identifier for the row -- unused otherwise. Used only on the hidden test set.
    dataset: str
    filename: str
    cluster_index: int | None = None
    rotation: np.ndarray | None = None
    translation: np.ndarray | None = None

# Set is_train=True to run the notebook on the training data.
# Set is_train=False if submitting an entry to the competition (test data is hidden, and different from what you see on the "test" folder).
is_train = False
data_dir = '/kaggle/input/image-matching-challenge-2025'
workdir = '/kaggle/working/result/'
os.makedirs(workdir, exist_ok=True)

if is_train:
    sample_submission_csv = os.path.join(data_dir, 'train_labels.csv')
else:
    sample_submission_csv = os.path.join(data_dir, 'sample_submission.csv')

samples = {}
competition_data = pd.read_csv(sample_submission_csv)
for _, row in competition_data.iterrows():
    # Note: For the test data, the "scene" column has no meaning, and the rotation_matrix and translation_vector columns are random.
    if row.dataset not in samples:
        samples[row.dataset] = []
    samples[row.dataset].append(
        Prediction(
            image_id=None if is_train else row.image_id,
            dataset=row.dataset,
            filename=row.image
        )
    )

for dataset in samples:
    print(f'Dataset "{dataset}" -> num_images={len(samples[dataset])}')

gc.collect()

max_images = None 
datasets_to_process = None 

if is_train:
        # max_images = 5
    
    	# Data from IMC 2023 and 2024.
    	# 'imc2024_dioscuri_baalshamin',
    	# 'imc2023_theather_imc2024_church',
    	# 'imc2023_heritage',
    	# 'imc2023_haiper',
    	# 'imc2024_lizard_pond',
    	# Crowdsourced PhotoTourism data.
    	# 'pt_stpeters_stpauls',
    	# 'pt_brandenburg_british_buckingham',
    	# 'pt_piazzasanmarco_grandplace',
    	# 'pt_sacrecoeur_trevi_tajmahal',
    
        datasets_to_process = [
        	# New data.
        	'amy_gardens',
        	'ETs',
        	'fbk_vineyard',
        	'stairs',
        ]

timings = {
    "shortlisting":[],
    "feature_detection": [],
    "feature_matching":[],
    "RANSAC": [],
    "Reconstruction": [],
}
mapping_result_strs = []


for dataset, predictions in samples.items():
    if datasets_to_process and dataset not in datasets_to_process:
        print(f'Skipping "{dataset}"')
        continue
    
    images_dir = os.path.join(data_dir, 'train' if is_train else 'test', dataset)
    images = [os.path.join(images_dir, p.filename) for p in predictions]
    if max_images is not None:
        images = images[:max_images]

    print(f'\nProcessing dataset "{dataset}": {len(images)} images')

    filename_to_index = {p.filename: idx for idx, p in enumerate(predictions)}

    feature_dir = os.path.join(workdir, 'featureout', dataset)
    os.makedirs(feature_dir, exist_ok=True)

    # Wrap algos in try-except blocks so we can populate a submission even if one scene crashes.
    try:
        t = time()
        index_pairs = get_image_pairs_shortlist(
            images,
            sim_th = 0.59, # should be strict
            min_pairs = 20, # we should select at least min_pairs PER IMAGE with biggest similarity
            exhaustive_if_less = 20,
            device=device
        )
        timings['shortlisting'].append(time() - t)
        print (f'Shortlisting. Number of pairs to match: {len(index_pairs)}. Done in {time() - t:.4f} sec')
        gc.collect()
    
        t = time()

        detect_aliked(images, feature_dir, 4096, device=device)
        gc.collect()
        timings['feature_detection'].append(time() - t)
        print(f'Features detected in {time() - t:.4f} sec')
        
        t = time()
        match_with_lightglue(images, index_pairs, feature_dir=feature_dir, device=device, verbose=False)
        timings['feature_matching'].append(time() - t)
        print(f'Features matched in {time() - t:.4f} sec')

        database_path = os.path.join(feature_dir, 'colmap.db')
        if os.path.isfile(database_path):
            os.remove(database_path)
        gc.collect()
        sleep(1)
        import_into_colmap(images_dir, feature_dir=feature_dir, database_path=database_path)
        output_path = f'{feature_dir}/colmap_rec_aliked'
        
        t = time()
        pycolmap.match_exhaustive(database_path)
        timings['RANSAC'].append(time() - t)
        print(f'Ran RANSAC in {time() - t:.4f} sec')
        
        # By default colmap does not generate a reconstruction if less than 10 images are registered.
        # Lower it to 3.
        mapper_options = pycolmap.IncrementalPipelineOptions()
        mapper_options.min_model_size = 4
        mapper_options.max_num_models = 25
        os.makedirs(output_path, exist_ok=True)
        t = time()
        maps = pycolmap.incremental_mapping(
            database_path=database_path, 
            image_path=images_dir,
            output_path=output_path,
            options=mapper_options)
        sleep(1)
        timings['Reconstruction'].append(time() - t)
        print(f'Reconstruction done in  {time() - t:.4f} sec')
        print(maps)

        clear_output(wait=False)
    
        registered = 0
        for map_index, cur_map in maps.items():
            for index, image in cur_map.images.items():
                prediction_index = filename_to_index[image.name]
                predictions[prediction_index].cluster_index = map_index
                predictions[prediction_index].rotation = deepcopy(image.cam_from_world.rotation.matrix())
                predictions[prediction_index].translation = deepcopy(image.cam_from_world.translation)
                registered += 1
        mapping_result_str = f'Dataset "{dataset}" -> Registered {registered} / {len(images)} images with {len(maps)} clusters'
        mapping_result_strs.append(mapping_result_str)
        print(mapping_result_str)
        gc.collect()
    except Exception as e:
        print(e)
        # raise e
        mapping_result_str = f'Dataset "{dataset}" -> Failed!'
        mapping_result_strs.append(mapping_result_str)
        print(mapping_result_str)

print('\nResults')
for s in mapping_result_strs:
    print(s)

print('\nTimings')
for k, v in timings.items():
    print(f'{k} -> total={sum(v):.02f} sec.')

# Submission file
array_to_str = lambda array: ';'.join([f"{x:.09f}" for x in array])
none_to_str = lambda n: ';'.join(['nan'] * n)

submission_file = '/kaggle/working/submission.csv'
with open(submission_file, 'w') as f:
    if is_train:
        f.write('dataset,scene,image,rotation_matrix,translation_vector\n')
        for dataset in samples:
            for prediction in samples[dataset]:
                cluster_name = 'outliers' if prediction.cluster_index is None else f'cluster{prediction.cluster_index}'
                rotation = none_to_str(9) if prediction.rotation is None else array_to_str(prediction.rotation.flatten())
                translation = none_to_str(3) if prediction.translation is None else array_to_str(prediction.translation)
                f.write(f'{prediction.dataset},{cluster_name},{prediction.filename},{rotation},{translation}\n')
    else:
        f.write('image_id,dataset,scene,image,rotation_matrix,translation_vector\n')
        for dataset in samples:
            for prediction in samples[dataset]:
                cluster_name = 'outliers' if prediction.cluster_index is None else f'cluster{prediction.cluster_index}'
                rotation = none_to_str(9) if prediction.rotation is None else array_to_str(prediction.rotation.flatten())
                translation = none_to_str(3) if prediction.translation is None else array_to_str(prediction.translation)
                f.write(f'{prediction.image_id},{prediction.dataset},{cluster_name},{prediction.filename},{rotation},{translation}\n')

!head {submission_file}

