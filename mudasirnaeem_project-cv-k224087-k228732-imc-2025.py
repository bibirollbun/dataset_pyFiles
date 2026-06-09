# Installing LightGlue and other dependencies (offline installation)
!pip install --no-index /kaggle/input/imc2024-packages-lightglue-rerun-kornia/lightglue* --no-deps

# Installing IMC 2025 related wheels (also offline)
!pip install --no-index /kaggle/input/icm2025-packages/*.whl --no-deps


# Creating directory for torch checkpoints
!mkdir -p /root/.cache/torch/hub/checkpoints

# Copying official weights into the torch checkpoint folder
!cp /kaggle/input/imc2024-official-weights/* /root/.cache/torch/hub/checkpoints


import os
import json
from glob import glob

# Define paths for input dataset and output JSON directory
INPUT_ROOT = "/kaggle/input/image-matching-challenge-2025"
OUTPUT_ROOT = "/kaggle/working/fnames"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_ROOT, exist_ok=True)


# collecting scene wise images path

# Dictionary to store scenes with their image paths
scenes = {}

# Iterate through all files inside test scenes
for x in glob(os.path.join(INPUT_ROOT, "test/*/*.*")):
    if "LICENSE" not in x:  # Skip LICENSE files
        folder = os.path.basename(os.path.dirname(x))  # Extract folder/scene name
        if folder not in scenes:
            scenes[folder] = []  # Create list for new scene
        scenes[folder].append(x)  # Add image path


# Sort keys based on number of images (small → large)
sorted_keys = sorted(scenes.keys(), key=lambda k: len(scenes[k]))

# Prepare tasks for 2 workers
tasks = [[] for _ in range(2)]
for idx, key in enumerate(sorted_keys):
    worker_id = idx % 2  # Alternate assignment
    tasks[worker_id].append(key)

# Remove empty lists
tasks = [t for t in tasks if len(t) > 0]


# Save scenes list for each worker into JSON files
for i, task in enumerate(tasks):
    output_path = os.path.join(OUTPUT_ROOT, f"worker_{i}_scenes.json")
    with open(output_path, "w") as f:
        json.dump({s: scenes[s] for s in task}, f, indent=2)

# Final confirmation
print(f"Saved scene splits to {OUTPUT_ROOT}")


%%writefile inference.py
import os
import json
import time
import sys
import networkx as nx
import pandas as pd
import numpy as np
import gc
import hdbscan
import argparse
from math import ceil
sys.path.append("/kaggle/input/image-matching-2025-code/image-matching-3D-main")

from images_to_3d.tasks.get_pair.get_image_pair_exhaustive import task_get_image_pair_exhaustive
from images_to_3d.tasks.get_pair.get_image_pair_DINO import task_get_image_pair_DINO
from images_to_3d.tasks.get_pair.get_image_pair_kNN import task_get_image_pair_kNN
from images_to_3d.tasks.get_pair.get_transparent_pair import task_get_transparent_pair
from images_to_3d.tasks.matching.matching import task_matching
from images_to_3d.tasks.matching.matching_find_best import task_matching_find_best
from images_to_3d.tasks.matching.rotate_matching_find_best import task_rotate_matching_find_best
from images_to_3d.tasks.crop.sfm_mkpc import task_sfm_mkpc
from images_to_3d.tasks.crop.pair_mkpc import task_pair_mkpc
from images_to_3d.tasks.crop.transparent_crop import task_transparent_crop
from images_to_3d.tasks.utils.ransac import task_ransac
from images_to_3d.tasks.utils.concat import task_concat
from images_to_3d.tasks.utils.rem_less_match_pair import task_rem_less_match_pair
from images_to_3d.tasks.utils.count_matching_num import task_count_matching_num
from images_to_3d.tasks.utils.extract_inliner_matching_points import task_extract_inliner_matching_points
from images_to_3d.tasks.utils.extract_csv_pair import task_extract_csv_pair
from images_to_3d.tasks.utils.estimate_rot import task_estimate_rot
from images_to_3d.tasks.utils.get_exif import task_get_exif
from images_to_3d.tasks.global_feat.get_image_feats import task_generate_image_feats
from images_to_3d.models.utils import read_image, numpy_image_to_torch
from images_to_3d.models import ALIKED
from glob import glob
from tqdm  import tqdm
from pathlib import Path
import kornia as K

INPUT_ROOT = "/kaggle/input/image-matching-challenge-2025"
OUTPUT_ROOT = "/kaggle/working/outputs"
task_map = {
    "get_image_pair_exhaustive": task_get_image_pair_exhaustive,
    "get_image_pair_DINO": task_get_image_pair_DINO,
    "get_image_pair_kNN": task_get_image_pair_kNN,
    "get_transparent_pair": task_get_transparent_pair,

    "matching": task_matching,
    "matching_find_best": task_matching_find_best,
    "rotate_matching_find_best": task_rotate_matching_find_best,

    "sfm_mkpc": task_sfm_mkpc,
    "pair_mkpc": task_pair_mkpc,
    "transparent_crop": task_transparent_crop,
    
    "ransac": task_ransac,
    "concat": task_concat,
    "rem_less_match_pair": task_rem_less_match_pair,
    "count_matching_num": task_count_matching_num,
    "extract_inliner_matching_points": task_extract_inliner_matching_points,
    "extract_csv_pair": task_extract_csv_pair,
    "estimate_rot": task_estimate_rot,
    "get_exif": task_get_exif,
    "generate_image_feats": task_generate_image_feats,
}

import json
pipleline = json.load(open("/kaggle/input/image-matching-2025-code/image-matching-3D-main/notebooks/pipeline.json", "r"))

filter_threshold1 = 0.531049
pipleline[2]["params"]["keypoint_matching_args"]["matcher_params"]["filter_threshold"] = filter_threshold1
filter_threshold2 = 0.485205
pipleline[3]["params"]["keypoint_matching_args"]["matcher_params"]["filter_threshold"] = filter_threshold2
filter_threshold3 = 0.758147
pipleline[5]["params"]["keypoint_matching_args"]["matcher_params"]["filter_threshold"] = filter_threshold3
filter_threshold4 = 0.944721
pipleline[10]["params"]["keypoint_matching_args"]["matcher_params"]["filter_threshold"] = filter_threshold4

th_matching_num = 33
pipleline[7]['params']["th_matching_num"] = th_matching_num * 10

ransac_min_matches = 7
pipleline[-2]['params']['min_matches'] = ransac_min_matches * 10

pipleline[-2]['params']['ransac_params']["param1"] = 2/2
pipleline[-2]['params']['ransac_params']["param2"] = 0.767528

class Pipeline():
    def __init__(self, data_dict, work_dir, input_dir_root, pipeline_config, device_id, pdb = False):
        self.device = K.utils.get_cuda_device_if_available(device_id)
        print(f"device: {self.device}")
        self.data_dict = data_dict
        self.work_dir = work_dir
        self.input_dir_root = input_dir_root
        os.makedirs(self.work_dir, exist_ok=True)
        self.pipeline_config = pipeline_config
        self.processing_times = {
            "task": [],
            "comment": [],
            "processing_time": []
        }
        self.pdb = pdb


    def exec(self):
        all_processing_time = 0
        for p in self.pipeline_config:
            task = p["task"]
            comment = p["comment"]
            p["params"]["device"] = self.device
            p["params"]["data_dict"] = self.data_dict
            p["params"]["work_dir"] = self.work_dir
            p["params"]["input_dir_root"] = self.input_dir_root
            p["params"]["pdb"] = self.pdb
            
            start = time.time()
            print(f"===== [{task}] {comment} =====")
            task_map[task](p["params"])
            gc.collect()
            print("====================")
            end = time.time()

            self.processing_times["task"].append(task)
            self.processing_times["comment"].append(comment)
            self.processing_times["processing_time"].append(end-start)
            all_processing_time += end-start
        
        self.processing_times["task"].append("All")
        self.processing_times["comment"].append("")
        self.processing_times["processing_time"].append(all_processing_time)
        processing_times_df = pd.DataFrame.from_dict(self.processing_times)
        processing_times_df.to_csv(os.path.join(self.work_dir, "processing_time.csv"), index=False)

from collections import defaultdict
import shutil
from images_to_3d.tasks.reconstruction.reconstruct import reconstruction
def find_connected_components(edges, minimum_size=9):
    # Build the adjacency list
    graph = defaultdict(set)
    nodes = set()
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)
        nodes.update([u, v])
    
    visited = set()
    components = []

    def dfs(node, component):
        visited.add(node)
        component.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, component)

    for node in nodes:
        if node not in visited:
            component = []
            dfs(node, component)
            components.append(component)
    #post processing
    components = [c for c in components if len(c)>=minimum_size]
    return len(components), components

def get_components(edges,minimum_size=5):
    G = nx.Graph()
    G.add_edges_from(edges)
    num_degrees = list(dict(G.degree()).values())
    
    pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42)
    model = hdbscan.HDBSCAN(min_cluster_size=5,)
    labels = model.fit_predict(np.stack(list(pos.values())))   

    degree_counts = {x: [] for x in np.unique(labels)}
    for d,l in zip(num_degrees,labels):
        degree_counts[l].append(d)
    
    connectedness = np.mean([np.mean(v)/len(v) for k,v in degree_counts.items() if k!=-1])
    connectedness = np.nan_to_num(connectedness,posinf=0,neginf=0,nan=0)
    if model.probabilities_.mean()>0.8 and connectedness>0.5:
        print("using hdbscan")
        c = np.unique(labels[labels!=-1])
        connected_components = {idx:[] for idx in c}
        for node_label, position in zip(labels, pos.keys()):
            if node_label in connected_components:
                connected_components[node_label].append(position)
        connected_components = [v for v in connected_components.values() if len(v)>0]
        return len(connected_components),connected_components
    else :
        print("using spanning tree")
        c,connected_components = find_connected_components(edges, minimum_size=minimum_size)
    
    return c,connected_components
        
class Config: 
    min_size=5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, required=True)
    parser.add_argument('--rank', type=int, required=True)
    args = parser.parse_args()

    scenes = json.load(open(os.path.join("/kaggle/working/fnames", f"worker_{args.rank}_scenes.json")))
    scenes = {k:[Path(p) for p in v] for k,v in scenes.items()}
    gpu_id = os.environ["CUDA_VISIBLE_DEVICES"]
    print(f"running {args.rank}/ {args.world_size} with gpu {gpu_id} : {scenes.keys()}")
    for dset_name in scenes:
        try:
            work_dir = os.path.join(OUTPUT_ROOT, dset_name)
            pipeline = Pipeline(scenes[dset_name], Path(work_dir), Path(INPUT_ROOT), pipleline, 0, pdb = False)
            pipeline.exec()
            print(f"Processing {work_dir}")
            print("="*100)
            image_pairs = pd.read_csv(os.path.join(work_dir, "image_pair.csv"))
            counts, connected_components = get_components([(a,b) for a,b in zip(image_pairs.key1, image_pairs.key2)], minimum_size=Config.min_size)
            print(f"Found {counts} connected components of length {[len(c) for c in connected_components]}")
            
            colmap_mapper_options = {
                "min_model_size": 3, # By default colmap does not generate a reconstruction if less than 10 images are registered. Lower it to 3.
                "max_num_models": 25,
                #"num_threads": 1,
            }
            
            for i,image_paths in enumerate(connected_components):
                output_dir = os.path.join(work_dir, f"minsize_{Config.min_size}_scene_{i}")
                os.makedirs(output_dir, exist_ok=True)
                image_root = str(scenes[os.path.basename(work_dir)][0].parents[0])
                reconstruction(image_root, image_paths, os.path.basename(work_dir), f"minsize_{Config.min_size}_scene_{i}", work_dir, output_dir,colmap_mapper_options,image_model="radial")
                print(f"Scene {i} done")    
                print("="*100)
        except:
            pass


import subprocess
import os

# Launch process for rank 0 on GPU 0
p0 = subprocess.Popen(
    ['python3', 'inference.py', '--world_size=2', '--rank=0'],
    env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'}
)

# Launch process for rank 1 on GPU 1
p1 = subprocess.Popen(
    ['python3', 'inference.py', '--world_size=2', '--rank=1'],
    env={**os.environ, 'CUDA_VISIBLE_DEVICES': '1'}
)

# (Optional) Wait for both processes to complete
p0.wait()
p1.wait()

print("Both inference processes have completed.")


import os
import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path

INPUT_ROOT = "/kaggle/input/image-matching-challenge-2025"
OUTPUT_ROOT = "/kaggle/working/outputs"

# --- Load scenes --------------------------------------------------------
scenes = {}
for x in glob(os.path.join(INPUT_ROOT, "test/*/*.*")):
    if "LICENSE" in x:
        continue
    scene = os.path.basename(os.path.dirname(x))
    scenes.setdefault(scene, []).append(Path(x))

USED_COLS = ["image_id", "dataset", "scene", "image", "rotation_matrix", "translation_vector"]

# --- Load all worker submission csvs ------------------------------------
submission_files = glob(os.path.join(OUTPUT_ROOT, "*", "minsize_5_scene_*", "submission.csv"))

submission_images = pd.concat([pd.read_csv(fn) for fn in submission_files], ignore_index=True)

# --- Build proper columns -----------------------------------------------
submission_images["dataset"] = submission_images["image_path"].apply(lambda x: x.split("/")[5])
submission_images["image"] = submission_images["image_path"].apply(lambda x: os.path.basename(x))
submission_images["image_id"] = submission_images["dataset"] + "_" + submission_images["image"] + "_public"

submission_images = submission_images[USED_COLS]

# --- Determine processed image IDs --------------------------------------
non_outlier_ids = set(submission_images["image_id"])

# --- Find missing images and create outlier rows -------------------------
outliers = []
for dset_name, images in scenes.items():
    for img_path in images:
        filename = img_path.name
        img_id = f"{dset_name}_{filename}_public"

        if img_id in non_outlier_ids:
            continue

        outliers.append({
            "image_id": img_id,
            "dataset": dset_name,
            "scene": "outliers",
            "image": filename,
            "rotation_matrix": "nan;nan;nan;nan;nan;nan;nan;nan;nan",
            "translation_vector": "nan;nan;nan"
        })

# --- Merge final submission ----------------------------------------------
submission_images = pd.concat(
    [submission_images, pd.DataFrame(outliers)],
    ignore_index=True
)

submission_images.sort_values(["dataset", "scene"]).to_csv("submission.csv", index=False)





