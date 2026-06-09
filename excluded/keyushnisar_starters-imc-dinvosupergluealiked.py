# Install dependencies and copy model weights to run the notebook without internet access when submitting to the competition.

!pip install --no-index /kaggle/input/imc2024-packages-lightglue-rerun-kornia/* --no-deps
!mkdir -p /root/.cache/torch/hub/checkpoints
!cp /kaggle/input/aliked/pytorch/aliked-n16/1/aliked-n16.pth /root/.cache/torch/hub/checkpoints/
!cp /kaggle/input/lightglue/pytorch/aliked/1/aliked_lightglue.pth /root/.cache/torch/hub/checkpoints/
!cp /kaggle/input/lightglue/pytorch/aliked/1/aliked_lightglue.pth /root/.cache/torch/hub/checkpoints/aliked_lightglue_v0-1_arxiv-pth


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
import kornia as K
import kornia.feature as KF

import torch
from lightglue import match_pair
from lightglue import ALIKED, LightGlue
from lightglue.utils import load_image, rbd
from transformers import AutoImageProcessor, AutoModel

import pycolmap
sys.path.append('/kaggle/input/imc25-utils')
from database import *
from h5_to_db import *
import metric


device = K.utils.get_cuda_device_if_available(0)
print(f'{device=}')


def load_torch_image(fname, device):
    img = K.io.load_image(fname, K.io.ImageLoadType.RGB32, device=device)[None, ...]
    return img

def get_global_desc(fnames, device):
    processor = AutoImageProcessor.from_pretrained('/kaggle/input/dinov2/pytorch/base/1')
    model = AutoModel.from_pretrained('/kaggle/input/dinov2/pytorch/base/1').eval().to(device)
    descs = []
    for img_path in tqdm(fnames, desc='Computing global descriptors'):
        img = load_torch_image(img_path, device)
        with torch.no_grad():
            inputs = processor(images=img, return_tensors="pt", do_rescale=False).to(device)
            outputs = model(**inputs)
            desc = F.normalize(outputs.last_hidden_state[:,1:].max(dim=1)[0], dim=1)
        descs.append(desc.detach().cpu())
    return torch.cat(descs, dim=0)

def get_image_pairs(fnames, sim_th=0.5, min_pairs=20, exhaustive_if_less=20):
    if len(fnames) <= exhaustive_if_less:
        return [(i, j) for i in range(len(fnames)) for j in range(i+1, len(fnames))]
    descs = get_global_desc(fnames, device)
    dists = torch.cdist(descs, descs).cpu().numpy()
    pairs = []
    for i in range(len(fnames)):
        matches = np.where(dists[i] < sim_th)[0]
        if len(matches) < min_pairs:
            matches = np.argsort(dists[i])[:min_pairs]
        for j in matches:
            if i < j:
                pairs.append((i, j))
    return sorted(list(set(pairs)))

def detect_disk(fnames, feature_dir, num_features=5000):
    os.makedirs(feature_dir, exist_ok=True)
    disk = KF.DISK(n=num_features, score_threshold=0.1).eval().to(device)
    with h5py.File(f'{feature_dir}/keypoints.h5', 'w') as f_kp, \
         h5py.File(f'{feature_dir}/descriptors.h5', 'w') as f_desc:
        for img_path in tqdm(fnames, desc='Detecting keypoints with DISK'):
            img = load_torch_image(img_path, device)
            with torch.no_grad():
                features = disk(img)
                kpts = features.keypoints.cpu().numpy()
                descs = features.descriptors.cpu().numpy()
            key = os.path.basename(img_path)
            f_kp[key] = kpts
            f_desc[key] = descs

def match_superglue(fnames, pairs, feature_dir):
    config = {
        'weights': 'outdoor',
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }
    matcher = SuperGlue(config).eval().to(device)
    with h5py.File(f'{feature_dir}/keypoints.h5', 'r') as f_kp, \
         h5py.File(f'{feature_dir}/descriptors.h5', 'r') as f_desc, \
         h5py.File(f'{feature_dir}/matches.h5', 'w') as f_match:
        for i, j in tqdm(pairs, desc='Matching with SuperGlue'):
            fname1, fname2 = fnames[i], fnames[j]
            key1, key2 = os.path.basename(fname1), os.path.basename(fname2)
            kp1 = torch.from_numpy(f_kp[key1][...]).to(device)
            desc1 = torch.from_numpy(f_kp[key1][...]).to(device)
            kp2 = torch.from_numpy(f_kp[key2][...]).to(device)
            desc2 = torch.from_numpy(f_kp[key2][...]).to(device)
            data = {
                'keypoints0': kp1[None],
                'descriptors0': desc1[None],
                'keypoints1': kp2[None],
                'descriptors1': desc2[None]
            }
            with torch.no_grad():
                pred = matcher(data)
                matches = pred['matches0'][0].cpu().numpy()
                valid = matches > -1
                matches = np.stack([np.where(valid)[0], matches[valid]], axis=1)
            if len(matches) >= 15:
                group = f_match.require_group(key1)
                group.create_dataset(key2, data=matches)

def geometric_verification(fnames, pairs, feature_dir):
    with h5py.File(f'{feature_dir}/keypoints.h5', 'r') as f_kp, \
         h5py.File(f'{feature_dir}/matches.h5', 'r+') as f_match:
        for i, j in tqdm(pairs, desc='Geometric verification'):
            fname1, fname2 = fnames[i], fnames[j]
            key1, key2 = os.path.basename(fname1), os.path.basename(fname2)
            if key1 in f_match and key2 in f_match[key1]:
                kp1 = f_kp[key1][...]
                kp2 = f_kp[key2][...]
                matches = f_match[key1][key2][...]
                if len(matches) >= 8:
                    _, inliers = cv2.findFundamentalMat(
                        kp1[matches[:, 0]],
                        kp2[matches[:, 1]],
                        cv2.FM_RANSAC,
                        1.0,
                        0.999
                    )
                    if inliers is not None:
                        valid = inliers.ravel() > 0
                        if np.sum(valid) >= 15:
                            f_match[key1][key2][...] = matches[valid]
                        else:
                            del f_match[key1][key2]

def import_into_colmap(img_dir, feature_dir, database_path):
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    fname_to_id = add_keypoints(db, feature_dir, img_dir, '', 'simple-pinhole', False)
    add_matches(db, feature_dir, fname_to_id)
    db.commit()



import dataclasses

@dataclasses.dataclass
class Prediction:
    image_id: str
    dataset: str
    filename: str
    cluster_index: int = None
    rotation: np.ndarray = None
    translation: np.ndarray = None

is_train = False
data_dir = '/kaggle/input/image-matching-challenge-2025'
work_dir = '/kaggle/working/result'
os.makedirs(work_dir, exist_ok=True)

submission_csv = os.path.join(data_dir, 'train_labels.csv' if is_train else 'sample_submission.csv')
samples = {}
for _, row in pd.read_csv(submission_csv).iterrows():
    if row.dataset not in samples:
        samples[row.dataset] = []
    samples[row.dataset].append(Prediction(
        image_id=row.image_id if not is_train else None,
        dataset=row.dataset,
        filename=row.image
    ))

print('Datasets loaded:')
for dataset in samples:
    print(f'  - {dataset}: {len(samples[dataset])} images')


from time import time

timings = {
    'shortlisting': [],
    'feature_detection': [],
    'feature_matching': [],
    'geometric_verification': [],
    'reconstruction': []
}
results = []

print('Starting processing...')
for dataset, predictions in samples.items():
    print(f'\nProcessing dataset: {dataset}')
    images_dir = os.path.join(data_dir, 'train' if is_train else 'test', dataset)
    images = [os.path.join(images_dir, p.filename) for p in predictions]
    
    feature_dir = os.path.join(work_dir, 'featureout', dataset)
    database_path = os.path.join(feature_dir, 'colmap.db')
    output_path = os.path.join(feature_dir, 'colmap_rec')
    
    try:
        # Find similar image pairs
        t = time()
        pairs = get_image_pairs(images, sim_th=0.5, min_pairs=20)
        timings['shortlisting'].append(time() - t)
        print(f'  - Shortlisted {len(pairs)} pairs in {time()-t:.2f}s')
        
        # Detect features
        t = time()
        detect_disk(images, feature_dir)
        timings['feature_detection'].append(time() - t)
        print(f'  - Detected features in {time()-t:.2f}s')
        
        # Match features
        t = time()
        match_superglue(images, pairs, feature_dir)
        timings['feature_matching'].append(time() - t)
        print(f'  - Matched features in {time()-t:.2f}s')
        
        # Geometric verification
        t = time()
        geometric_verification(images, pairs, feature_dir)
        timings['geometric_verification'].append(time() - t)
        print(f'  - Verified matches in {time()-t:.2f}s')
        
        # Create COLMAP database
        if os.path.exists(database_path):
            os.remove(database_path)
        import_into_colmap(images_dir, feature_dir, database_path)
        
        # Run reconstruction
        t = time()
        pycolmap.match_exhaustive(database_path)
        mapper_options = pycolmap.IncrementalPipelineOptions()
        mapper_options.min_model_size = 3
        mapper_options.max_num_models = 25
        os.makedirs(output_path, exist_ok=True)
        maps = pycolmap.incremental_mapping(database_path, images_dir, output_path, mapper_options)
        timings['reconstruction'].append(time() - t)
        
        # Store results
        registered = 0
        filename_to_index = {p.filename: idx for idx, p in enumerate(predictions)}
        for map_idx, cur_map in maps.items():
            for _, image in cur_map.images.items():
                idx = filename_to_index[image.name]
                predictions[idx].cluster_index = map_idx
                predictions[idx].rotation = image.cam_from_world.rotation.matrix()
                predictions[idx].translation = image.cam_from_world.translation
                registered += 1
        
        result = f'Dataset "{dataset}": Registered {registered}/{len(images)} images in {len(maps)} clusters'
        results.append(result)
        print(result)
    except Exception as e:
        result = f'Dataset "{dataset}": Failed - {str(e)}'
        results.append(result)
        print(result)

print('\nSummary:')
for result in results:
    print(result)
print('\nTimings:')
for k, v in timings.items():
    print(f'  - {k}: {sum(v):.2f}s')


def array_to_str(array):
    return ';'.join([f'{x:.9f}' for x in array])

def none_to_str(n):
    return ';'.join(['nan'] * n)

submission_file = '/kaggle/working/submission.csv'
with open(submission_file, 'w') as f:
    header = 'image_id,dataset,scene,image,rotation_matrix,translation_vector\n' if not is_train else \
             'dataset,scene,image,rotation_matrix,translation_vector\n'
    f.write(header)
    
    for dataset in samples:
        for pred in samples[dataset]:
            cluster = 'outliers' if pred.cluster_index is None else f'cluster{pred.cluster_index}'
            rot = none_to_str(9) if pred.rotation is None else array_to_str(pred.rotation.flatten())
            trans = none_to_str(3) if pred.translation is None else array_to_str(pred.translation)
            
            if is_train:
                f.write(f'{pred.dataset},{cluster},{pred.filename},{rot},{trans}\n')
            else:
                f.write(f'{pred.image_id},{pred.dataset},{cluster},{pred.filename},{rot},{trans}\n')

print(f'ðŸ“„ Submission file created: {submission_file}')
!head {submission_file}


if is_train:
    t = time()
    final_score, dataset_scores = metric.score(
        gt_csv='/kaggle/input/image-matching-challenge-2025/train_labels.csv',
        user_csv=submission_file,
        thresholds_csv='/kaggle/input/image-matching-challenge-2025/train_thresholds.csv',
        mask_csv=None,
        inl_cf=0,
        strict_cf=-1,
        verbose=True
    )
    print(f'Evaluation score: {final_score:.3f}')
    print(f'Evaluation time: {time()-t:.2f}s')




