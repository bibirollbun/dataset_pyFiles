!tar xfvz /kaggle/input/ultralytics-for-offline-install/archive.tar.gz
!pip install --no-index --find-links=./packages ultralytics
!rm -rf ./packages

!cp -r '/kaggle/input/hengck-czii-cryo-et-01/wheel_file' '/kaggle/working/'
!pip install /kaggle/working/wheel_file/asciitree-0.3.3/asciitree-0.3.3
!pip install --no-index --find-links=/kaggle/working/wheel_file zarr


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
submission = pd.concat([submission0, submission1]).reset_index(drop=True)


submission.insert(0, 'id', range(len(submission)))
submission.to_csv("submission.csv", index=False)
submission.head()

