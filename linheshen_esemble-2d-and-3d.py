from IPython.display import clear_output
!tar xfvz /kaggle/input/ultralytics-for-offline-install/archive.tar.gz
!pip install --no-index --find-links=./packages ultralytics
!rm -rf ./packages
try:
    import zarr
except: 
    !cp -r '/kaggle/input/hengck-czii-cryo-et-01/wheel_file' '/kaggle/working/'
    !pip install /kaggle/working/wheel_file/asciitree-0.3.3/asciitree-0.3.3
    !pip install --no-index --find-links=/kaggle/working/wheel_file zarr
    !pip install --no-index --find-links=/kaggle/working/wheel_file connected-components-3d
from typing import List, Tuple, Union
deps_path = '/kaggle/input/czii-cryoet-dependencies'
! pip install -q --no-index --find-links {deps_path} --requirement {deps_path}/requirements.txt
import lightning.pytorch as pl
from datetime import datetime
import pytz
import sys
sys.path.append('/kaggle/input/hengck-czii-cryo-et-01')
from czii_helper import *
from dataset import *
from model2 import *
clear_output()


import os
import glob
import time
import sys
import warnings
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO
import zarr
from scipy.spatial import cKDTree
from collections import defaultdict


model_path = '/kaggle/input/czii-yolo-l-trained-with-synthetic-data/best_synthetic.pt'
model = YOLO(model_path)


runs_path = '/kaggle/input/czii-cryo-et-object-identification/test/static/ExperimentRuns/*'
runs = sorted(glob.glob(runs_path))
runs = [os.path.basename(run) for run in runs]
sp = len(runs)//2
runs1 = runs[:sp]
runs1[:5]

#add by @minfuka
runs2 = runs[sp:]
runs2[:5]

#add by @minfuka
assert torch.cuda.device_count() == 2


particle_names = [
    'apo-ferritin',
    'beta-amylase',
    'beta-galactosidase',
    'ribosome',
    'thyroglobulin',
    'virus-like-particle'
]

particle_to_index = {
    'apo-ferritin': 0,
    'beta-amylase': 1,
    'beta-galactosidase': 2,
    'ribosome': 3,
    'thyroglobulin': 4,
    'virus-like-particle': 5
}

index_to_particle = {index: name for name, index in particle_to_index.items()}

particle_radius = {
    'apo-ferritin': 60,
    'beta-amylase': 65,
    'beta-galactosidase': 90,
    'ribosome': 150,
    'thyroglobulin': 130,
    'virus-like-particle': 135,
}



# add by @sesasj
class UnionFind:
    def __init__(self, size):
        self.parent = np.arange(size)
        self.rank = np.zeros(size, dtype=int)

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])  
        return self.parent[u]

    def union(self, u, v):
        u_root = self.find(u)
        v_root = self.find(v)
        if u_root == v_root:
            return
            
        if self.rank[u_root] < self.rank[v_root]:
            self.parent[u_root] = v_root
        else:
            self.parent[v_root] = u_root
            if self.rank[u_root] == self.rank[v_root]:
                self.rank[u_root] += 1

class PredictionAggregator:
    def __init__(self, first_conf=0.2, conf_coef=0.75):
        self.first_conf = first_conf
        self.conf_coef = conf_coef
        self.particle_confs = np.array([0.5, 0.0, 0.2, 0.5, 0.2, 0.5])
        
    def convert_to_8bit(self, volume):
        lower, upper = np.percentile(volume, (0.5, 99.5))
        clipped = np.clip(volume, lower, upper)
        scaled = ((clipped - lower) / (upper - lower + 1e-12) * 255).astype(np.uint8)
        return scaled

    def make_predictions(self, run_id, model, device_no):
        volume_path = f'/kaggle/input/czii-cryo-et-object-identification/test/static/ExperimentRuns/{run_id}/VoxelSpacing10.000/denoised.zarr'
        volume = zarr.open(volume_path, mode='r')[0]
        volume_8bit = self.convert_to_8bit(volume)
        num_slices = volume_8bit.shape[0]

        detections = {
            'particle_type': [],
            'confidence': [],
            'x': [],
            'y': [],
            'z': []
        }

        for slice_idx in range(num_slices):
            
            img = volume_8bit[slice_idx]
            input_image = cv2.resize(np.stack([img]*3, axis=-1), (640, 640))

            results = model.predict(
                input_image,
                save=False,
                imgsz=640,
                conf=self.first_conf,
                device=device_no,
                batch=1,
                verbose=False,
            )

            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                cls = boxes.cls.cpu().numpy().astype(int)
                conf = boxes.conf.cpu().numpy()
                xyxy = boxes.xyxy.cpu().numpy()

                xc = ((xyxy[:, 0] + xyxy[:, 2]) / 2.0) * 10 * (63/64) # 63/64 because of the resize
                yc = ((xyxy[:, 1] + xyxy[:, 3]) / 2.0) * 10 * (63/64)
                zc = np.full(xc.shape, slice_idx * 10 + 5)

                particle_types = [index_to_particle[c] for c in cls]

                detections['particle_type'].extend(particle_types)
                detections['confidence'].extend(conf)
                detections['x'].extend(xc)
                detections['y'].extend(yc)
                detections['z'].extend(zc)

        if not detections['particle_type']:
            return pd.DataFrame()  

        particle_types = np.array(detections['particle_type'])
        confidences = np.array(detections['confidence'])
        xs = np.array(detections['x'])
        ys = np.array(detections['y'])
        zs = np.array(detections['z'])

        aggregated_data = []

        for idx, particle in enumerate(particle_names):
            if particle == 'beta-amylase':
                continue 

            mask = (particle_types == particle)
            if not np.any(mask):
                continue  
                
            particle_confidences = confidences[mask]
            particle_xs = xs[mask]
            particle_ys = ys[mask]
            particle_zs = zs[mask]
            # -------------modified by @sersasj ------------------------
            coords = np.vstack((particle_xs, particle_ys, particle_zs)).T

           
            z_distance = 30 # How many slices can you "jump" to aggregate predictions 10 = 1, 20 = 2...
            xy_distance = 20 # xy_tol_p2 in original code by ITK8191
            
            max_distance = math.sqrt(z_distance**2 + xy_distance**2)
            tree = cKDTree(coords)            
            pairs = tree.query_pairs(r=max_distance, p=2)

            
            uf = UnionFind(len(coords))
            
            coords_xy = coords[:, :2]
            coords_z = coords[:, 2]
            for u, v in pairs:
                z_diff = abs(coords_z[u] - coords_z[v])
                if z_diff > z_distance:
                    continue  

                xy_diff = np.linalg.norm(coords_xy[u] - coords_xy[v])
                if xy_diff > xy_distance:
                    continue  

                uf.union(u, v)

            roots = np.array([uf.find(i) for i in range(len(coords))])
            unique_roots, inverse_indices, counts = np.unique(roots, return_inverse=True, return_counts=True)
            conf_sums = np.bincount(inverse_indices, weights=particle_confidences)
            
            aggregated_confidences = conf_sums / (counts ** self.conf_coef)
            cluster_per_particle = [4,1,2,9,4,8] # Update
            valid_clusters = (counts >= cluster_per_particle[idx]) & (aggregated_confidences > self.particle_confs[idx])

            if not np.any(valid_clusters):
                continue  

            cluster_ids = unique_roots[valid_clusters]

            centers_x = np.bincount(inverse_indices, weights=particle_xs) / counts
            centers_y = np.bincount(inverse_indices, weights=particle_ys) / counts
            centers_z = np.bincount(inverse_indices, weights=particle_zs) / counts

            centers_x = centers_x[valid_clusters]
            centers_y = centers_y[valid_clusters]
            centers_z = centers_z[valid_clusters]

            aggregated_df = pd.DataFrame({
                'experiment': [run_id] * len(centers_x),
                'particle_type': [particle] * len(centers_x),
                'x': centers_x,
                'y': centers_y,
                'z': centers_z
            })

            aggregated_data.append(aggregated_df)

        if aggregated_data:
            return pd.concat(aggregated_data, axis=0)
        else:
            return pd.DataFrame()  


# instance main class
aggregator = PredictionAggregator(first_conf=0.19,  conf_coef=0.34) #Update
aggregated_results = []
#add by @minfuka
from concurrent.futures import ProcessPoolExecutor #add by @minfuka

#add by @minfuka
def inference(runs, model, device_no):
    subs = []
    for r in tqdm(runs, total=len(runs)):
        df = aggregator.make_predictions(r, model, device_no)
        subs.append(df)
    
    return subs
start_time = time.time()

with ProcessPoolExecutor(max_workers=2) as executor:
    results = list(executor.map(inference, (runs1, runs2), (model, model), ("0", "1")))


end_time = time.time()

estimated_total_time = (end_time - start_time) / len(runs) * 500  
print(f'estimated total prediction time for 500 runs: {estimated_total_time:.4f} seconds')


#change by @minfuka
submission0 = pd.concat(results[0])
submission1 = pd.concat(results[1])
submission_ = pd.concat([submission0, submission1]).reset_index(drop=True)


submission_.insert(0, 'id', range(len(submission_)))


class Model(pl.LightningModule):
    def __init__(
        self, 
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 7,
        channels: Union[Tuple[int, ...], List[int]] = (48, 64, 80, 80),
        strides: Union[Tuple[int, ...], List[int]] = (2, 2, 1),
        num_res_units: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(
            spatial_dims=self.hparams.spatial_dims,
            in_channels=self.hparams.in_channels,
            out_channels=self.hparams.out_channels,
            channels=self.hparams.channels,
            strides=self.hparams.strides,
            num_res_units=self.hparams.num_res_units,
        )
    def forward(self, x):
        return self.model(x)
# 设置训练超参数
channels = (48, 64, 80, 80)
strides_pattern = (2, 2, 1)
num_res_units = 1
def extract_3d_patches_minimal_overlap(arrays: List[np.ndarray], patch_size: int) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]]]:
    if not arrays or not isinstance(arrays, list):
        raise ValueError("Input must be a non-empty list of arrays")
    
    # Verify all arrays have the same shape
    shape = arrays[0].shape
    if not all(arr.shape == shape for arr in arrays):
        raise ValueError("All input arrays must have the same shape")
    
    if patch_size > min(shape):
        raise ValueError(f"patch_size ({patch_size}) must be smaller than smallest dimension {min(shape)}")
    
    m, n, l = shape
    patches = []
    coordinates = []
    
    # Calculate starting positions for each dimension
    x_starts = calculate_patch_starts(m, patch_size)
    y_starts = calculate_patch_starts(n, patch_size)
    z_starts = calculate_patch_starts(l, patch_size)
    
    # Extract patches from each array
    for arr in arrays:
        for x in x_starts:
            for y in y_starts:
                for z in z_starts:
                    patch = arr[
                        x:x + patch_size,
                        y:y + patch_size,
                        z:z + patch_size
                    ]
                    patches.append(patch)
                    coordinates.append((x, y, z))
    
    return patches, coordinates
def reconstruct_array(patches: List[np.ndarray], 
                     coordinates: List[Tuple[int, int, int]], 
                     original_shape: Tuple[int, int, int]) -> np.ndarray:
    reconstructed = np.zeros(original_shape, dtype=np.int64)  # To track overlapping regions
    
    patch_size = patches[0].shape[0]
    
    for patch, (x, y, z) in zip(patches, coordinates):
        reconstructed[
            x:x + patch_size,
            y:y + patch_size,
            z:z + patch_size
        ] = patch
        
    
    return reconstructed
def calculate_patch_starts(dimension_size: int, patch_size: int) -> List[int]:
    if dimension_size <= patch_size:
        return [0]
        
    # Calculate number of patches needed
    n_patches = np.ceil(dimension_size / patch_size)
    
    if n_patches == 1:
        return [0]
    
    # Calculate overlap
    total_overlap = (n_patches * patch_size - dimension_size) / (n_patches - 1)
    
    # Generate starting positions
    positions = []
    for i in range(int(n_patches)):
        pos = int(i * (patch_size - total_overlap))
        if pos + patch_size > dimension_size:
            pos = dimension_size - patch_size
        if pos not in positions:  # Avoid duplicates
            positions.append(pos)
    
    return positions
import pandas as pd

def dict_to_df(coord_dict, experiment_name):
    # Create lists to store data
    all_coords = []
    all_labels = []
    
    # Process each label and its coordinates
    for label, coords in coord_dict.items():
        all_coords.append(coords)
        all_labels.extend([label] * len(coords))
    
    # Concatenate all coordinates
    all_coords = np.vstack(all_coords)
    
    df = pd.DataFrame({
        'experiment': experiment_name,
        'particle_type': all_labels,
        'x': all_coords[:, 0],
        'y': all_coords[:, 1],
        'z': all_coords[:, 2]
    })

    
    return df
from typing import List, Tuple, Union
import numpy as np
import torch
from monai.data import DataLoader, Dataset, CacheDataset, decollate_batch
from monai.transforms import (
    Compose, 
    EnsureChannelFirstd, 
    Orientationd,  
    AsDiscrete,  
    RandFlipd, 
    RandRotate90d, 
    NormalizeIntensityd,
    RandCropByLabelClassesd,
)
TRAIN_DATA_DIR = "/kaggle/input/create-numpy-dataset-exp-name"
import json
copick_config_path = TRAIN_DATA_DIR + "/copick.config"

with open(copick_config_path) as f:
    copick_config = json.load(f)

copick_config['static_root'] = '/kaggle/input/czii-cryo-et-object-identification/test/static'

copick_test_config_path = 'copick_test.config'

with open(copick_test_config_path, 'w') as outfile:
    json.dump(copick_config, outfile)
import copick

root = copick.from_file(copick_test_config_path)

copick_user_name = "copickUtils"
copick_segmentation_name = "paintedPicks"
voxel_size = 10
tomo_type = "denoised"
inference_transforms = Compose([
    EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
    NormalizeIntensityd(keys="image"),
    Orientationd(keys=["image"], axcodes="RAS")
])
import cc3d

id_to_name = {1: "apo-ferritin", 
              2: "beta-amylase",
              3: "beta-galactosidase", 
              4: "ribosome", 
              5: "thyroglobulin", 
              6: "virus-like-particle"}
BLOB_THRESHOLD = 200
CERTAINTY_THRESHOLD = 0.05

classes = [1, 2, 3, 4, 5, 6]
import torch
import numpy as np
import pandas as pd
import cc3d
from monai.data import CacheDataset
from monai.transforms import Compose, EnsureType
from torch import nn
from tqdm import tqdm

# 假设你有一个函数来加载模型权重
def load_models(model_paths):
    models = []
    for model_path in model_paths:
        model = torch.load(model_path)
        model.to('cuda')
        model.eval()
        models.append(model)
    return models

# 假设你有3个模型的路径
model_paths = [
    '/kaggle/input/weight-all/weight/val_on_TS_6_6_best_model.pt'
]

# 加载模型
models = load_models(model_paths)
def ensemble_prediction_tta(models, input_tensor, threshold=0.5):
    probs_list = []
    data_copy0 = input_tensor.clone()
    data_copy0=torch.flip(data_copy0, dims=[2])
    data_copy1 = input_tensor.clone()
    data_copy1=torch.flip(data_copy1, dims=[3])
    data_copy2 = input_tensor.clone()
    data_copy2=torch.flip(data_copy2, dims=[4])
    data_copy3 = input_tensor.clone()
    data_copy3 = data_copy3.rot90(1, dims=[3, 4])
    with torch.no_grad():
        model_output0 = model(input_tensor)
        model_output1 = model(data_copy0)
        model_output1=torch.flip(model_output1, dims=[2])
        model_output2 = model(data_copy1)
        model_output2=torch.flip(model_output2, dims=[3])
        model_output3 = model(data_copy2)
        model_output3=torch.flip(model_output3, dims=[4])
        probs0 = torch.softmax(model_output0[0], dim=0)
        probs1 = torch.softmax(model_output1[0], dim=0)
        probs2 = torch.softmax(model_output2[0], dim=0)
        probs3 = torch.softmax(model_output3[0], dim=0)
        probs_list.append(probs0)
        probs_list.append(probs1)
        probs_list.append(probs2)
        probs_list.append(probs3)
    avg_probs = torch.mean(torch.stack(probs_list), dim=0)
    thresh_probs = avg_probs > threshold
    _, max_classes = thresh_probs.max(dim=0)
    return max_classes
sub=[]
for model in models:
    with torch.no_grad():
        location_df = []
        for run in root.runs:
            tomo = run.get_voxel_spacing(10)
            tomo = tomo.get_tomogram(tomo_type).numpy()
            tomo_patches, coordinates = extract_3d_patches_minimal_overlap([tomo], 96)
            tomo_patched_data = [{"image": img} for img in tomo_patches]
            tomo_ds = CacheDataset(data=tomo_patched_data, transform=inference_transforms, cache_rate=1.0)
            pred_masks = []
            for i in tqdm(range(len(tomo_ds))):
                input_tensor = tomo_ds[i]['image'].unsqueeze(0).to("cuda")
                max_classes = ensemble_prediction_tta(models, input_tensor, threshold=CERTAINTY_THRESHOLD)
                pred_masks.append(max_classes.cpu().numpy())
            reconstructed_mask = reconstruct_array(pred_masks, coordinates, tomo.shape)
            location = {}
            for c in classes:
                cc = cc3d.connected_components(reconstructed_mask == c)
                stats = cc3d.statistics(cc)
                zyx = stats['centroids'][1:] * 10.012444  # 转换单位
                zyx_large = zyx[stats['voxel_counts'][1:] > BLOB_THRESHOLD]
                xyz = np.ascontiguousarray(zyx_large[:, ::-1])
                location[id_to_name[c]] = xyz
            df = dict_to_df(location, run.name)
            location_df.append(df)
        location_df = pd.concat(location_df)
        location_df.insert(loc=0, column='id', value=np.arange(len(location_df)))


import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

# 假设sub已经给定，拼接DataFrame
df = pd.concat([submission_,location_df], ignore_index=True)
# 粒子半径映射
particle_names = ['apo-ferritin', 'beta-amylase', 'beta-galactosidase', 'ribosome', 'thyroglobulin', 'virus-like-particle']
particle_radius = {
    'apo-ferritin': 60,
    'beta-amylase': 65,
    'beta-galactosidase': 90,
    'ribosome': 150,
    'thyroglobulin': 130,
    'virus-like-particle': 135,
}

final = []  # 用于存储最终的结果
for pidx, p in enumerate(particle_names):
    # 筛选出该粒子类型的所有点
    pdf = df[df['particle_type'] == p].reset_index(drop=True)
    p_rad = particle_radius[p]
    
    # 根据 experiment 分组
    grouped = pdf.groupby(['experiment'])
    
    for exp, group in grouped:
        group = group.reset_index(drop=True)
        
        # 使用DBSCAN进行聚类
        coords = group[['x', 'y', 'z']].values
        db = DBSCAN(eps=p_rad, min_samples=2, metric='euclidean').fit(coords)
        labels = db.labels_
        
        # 将聚类结果添加到DataFrame中
        group['cluster'] = labels
        
        # 对每个簇进行处理
        for cluster_id in np.unique(labels):
            if cluster_id == -1:
                continue  # 跳过噪声点
            
            cluster_points = group[group['cluster'] == cluster_id]
            
            # 计算簇的中心（平均位置）
            avg_x = cluster_points['x'].mean()
            avg_y = cluster_points['y'].mean()
            avg_z = cluster_points['z'].mean()
            
            # 更新簇内点的位置
            group.loc[group['cluster'] == cluster_id, ['x', 'y', 'z']] = avg_x, avg_y, avg_z
            group = group.drop_duplicates(subset=['x', 'y', 'z'])
        # 将处理后的数据添加到 final 列表
        final.append(group)

# 合并处理后的数据
df_save = pd.concat(final, ignore_index=True)
df_save = df_save.drop(columns=['cluster'])

# 排序按 'experiment' 和 'particle_type' 两列
df_save = df_save.sort_values(by=['experiment', 'particle_type']).reset_index(drop=True)

# 重新生成 'id' 列，从 1 开始
df_save['id'] = np.arange(0, len(df_save))

# 输出结果到 CSV
df_save.to_csv('submission.csv', index=False)

