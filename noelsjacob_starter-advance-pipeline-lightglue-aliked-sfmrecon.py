# !git clone https://github.com/cvg/LightGlue.git 
# !python -m pip install -e LightGlue/


# !cp -r /kaggle/input/lightglue /kaggle/working/LightGlue
# !python -m pip install -e /kaggle/working/LightGlue/


!cp -r /kaggle/input/lightglue_model_v2 /kaggle/working/

# Add the copied repository to sys.path so Python can find it.
import sys
sys.path.insert(0, "/kaggle/input/lightglue_model_v2/pytorch/v2/1")
from lightglue import ALIKED, LightGlue


# Install dependencies and copy model weights to run the notebook without internet access when submitting to the competition.

# !pip install --no-index /kaggle/input/imc2024-packages-lightglue-rerun-kornia/* --no-deps
!mkdir -p /root/.cache/torch/hub/checkpoints
!cp /kaggle/input/aliked/pytorch/aliked-n16/1/aliked-n16.pth /root/.cache/torch/hub/checkpoints/
!cp /kaggle/input/lightglue/pytorch/aliked/1/aliked_lightglue.pth /root/.cache/torch/hub/checkpoints/
!cp /kaggle/input/lightglue/pytorch/aliked/1/aliked_lightglue.pth /root/.cache/torch/hub/checkpoints/aliked_lightglue_v0-1_arxiv-pth


import os
import gc
import sys
import cv2
import h5py
import torch
import dataclasses
import kornia as K
import numpy as np
import pandas as pd
import networkx as nx
import kornia.feature as KF
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from copy import deepcopy
from time import time, sleep
from collections import defaultdict
# from lightglue import ALIKED, LightGlue
from IPython.display import clear_output
from transformers import AutoImageProcessor, AutoModel



DATA_DIR = "/kaggle/input/image-matching-challenge-2025/train/"


# train_thresholds = pd.read_csv('/kaggle/input/image-matching-challenge-2025/train_thresholds.csv')
# data_dir = '/kaggle/input/image-matching-challenge-2025'
# is_train=False
# if is_train:
#     train_labels_path = os.path.join(data_dir, 'train_labels.csv')
# else:
#     train_labels_path = os.path.join(data_dir, 'sample_submission.csv')
# train_labels = pd.read_csv(train_labels_path)


# # Parse rotation matrix and translation vector
# def parse_matrix(matrix_str):
#     return np.array([float(x) for x in matrix_str.split(';')]).reshape(3, 3)

# def parse_vector(vector_str):
#     return np.array([float(x) for x in vector_str.split(';')])

# train_labels["rotation_matrix"] = train_labels["rotation_matrix"].apply(parse_matrix)
# train_labels["translation_vector"] = train_labels["translation_vector"].apply(parse_vector)


# Collect info from the dataset

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


def load_torch_image(fname, device=torch.device('cpu')):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return img.unsqueeze(0).to(device)


device = K.utils.get_cuda_device_if_available(0)
print(f'{device=}')


processor = AutoImageProcessor.from_pretrained('/kaggle/input/dinov2/pytorch/base/1')
model = AutoModel.from_pretrained('/kaggle/input/dinov2/pytorch/base/1')
model = model.eval()
model = model.to(device)


# ===========================
#  Extract Global Descriptors
# ===========================
def extract_dino_descriptor(fnames):
    global_descs = []
    for i, img_fname_full in tqdm(enumerate(fnames), total=len(fnames)):
        timg = load_torch_image(img_fname_full, device=device)
        with torch.inference_mode():
            inputs = processor(images=timg, return_tensors="pt", do_rescale=False).to(device)
            outputs = model(**inputs)
            # Here we take the [CLS] token embedding; adjust if needed
            cls_feature = F.normalize(outputs.last_hidden_state[:, 1:].max(dim=1)[0], dim=1, p=2) #L2 Normalization
        global_descs.append(cls_feature.detach().cpu())# CLS with max pooling
    global_descs = torch.cat(global_descs, dim=0)
    return global_descs
    


sim_th=0.6
min_pairs=20
topN=20


timings = {
    "shortlisting":[],
    "feature_detection": [],
    "feature_matching":[],
    "RANSAC": [],
    "Reconstruction": [],
}
t = time()


# Initialize LightGlue matcher
# matcher = LightGlue()

# Feature extraction and storage
def detect_aliked(img_fnames, feature_dir='.featureout', num_features=4096, resize_to=1024, device=torch.device('cpu')):
    dtype = torch.float32  # ALIKED has issues with float16
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


# Image pair matching using stored features
def match_with_lightglue(img_fnames, index_pairs, feature_dir='.featureout', device=torch.device('cpu'), min_matches=15, verbose=True):
    lg_matcher = KF.LightGlueMatcher("aliked", {
        "width_confidence": -1,
        "depth_confidence": -1,
        "mp": True if 'cuda' in str(device) else False
    }).eval().to(device)
    
    with h5py.File(f'{feature_dir}/keypoints.h5', mode='r') as f_kp, \
        h5py.File(f'{feature_dir}/descriptors.h5', mode='r') as f_desc, \
        h5py.File(f'{feature_dir}/matches.h5', mode='w') as f_match:
        
        for pair_idx in tqdm(index_pairs):
            fname1, fname2, _ = pair_idx  # Ignore the third element (the score)
            key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]
            kp1 = torch.from_numpy(f_kp[key1][...]).to(device)
            kp2 = torch.from_numpy(f_kp[key2][...]).to(device)
            desc1 = torch.from_numpy(f_desc[key1][...]).to(device)
            desc2 = torch.from_numpy(f_desc[key2][...]).to(device)
            
            with torch.inference_mode():
                dists, idxs = lg_matcher(
                    desc1, desc2,
                    KF.laf_from_center_scale_ori(kp1[None]),
                    KF.laf_from_center_scale_ori(kp2[None])
                )
            
            if len(idxs) == 0:
                continue
            
            n_matches = len(idxs)
            if verbose:
                print(f'{key1}-{key2}: {n_matches} matches')
            
            group = f_match.require_group(key1)
            if n_matches >= min_matches:
                group.create_dataset(key2, data=idxs.detach().cpu().numpy().reshape(-1, 2))
    return


# Example intrinsic matrix K (adjust with your real values)
fx = fy = 1200  # focal lengths
cx = cy = 640   # principal point (image center)
intrinsics = np.array([
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
], dtype=np.float32)



workdir = '/kaggle/working/result/'
os.makedirs(workdir, exist_ok=True)



def reconstruct_scene(image_paths, intrinsics, feature_dir='.featureout', match_file='matches.h5', output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load keypoints
    with h5py.File(os.path.join(feature_dir, 'keypoints.h5'), 'r') as f_kp, \
         h5py.File(os.path.join(feature_dir, match_file), 'r') as f_match:
        
        # Choose first pair
        img1, img2 = image_paths[0], image_paths[1]
        name1, name2 = os.path.basename(img1), os.path.basename(img2)

        kpts1 = f_kp[name1][()]
        kpts2 = f_kp[name2][()]
        
        matches = f_match[name1][name2][()]
        valid = matches >= 0
        idx1 = np.where(valid)[0]
        idx2 = matches[valid]
        
        pts1 = kpts1[idx1]
        pts2 = kpts2[idx2]

        # Estimate Essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, intrinsics, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        print(f"[INFO] Essential matrix shape: {E.shape}")

        # Recover relative pose
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, intrinsics)
        print("[INFO] Recovered pose")

        # Projection matrices
        P1 = intrinsics @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Camera 1
        P2 = intrinsics @ np.hstack((R, t))  # Camera 2

        # Triangulate
        pts1_h = cv2.convertPointsToHomogeneous(pts1[mask_pose.ravel() == 1])[:, 0, :]
        pts2_h = cv2.convertPointsToHomogeneous(pts2[mask_pose.ravel() == 1])[:, 0, :]

        pts4d_hom = cv2.triangulatePoints(P1, P2, pts1[mask_pose.ravel() == 1].T, pts2[mask_pose.ravel() == 1].T)
        pts3d = (pts4d_hom[:3] / pts4d_hom[3]).T  # Convert to 3D

        print(f"[INFO] Triangulated {pts3d.shape[0]} 3D points")

        # Save to PLY
        save_ply(os.path.join(output_dir, 'reconstruction.ply'), pts3d)

def save_ply(filename, points, colors=None):
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        if colors is not None:
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
        f.write('end_header\n')
        for i, pt in enumerate(points):
            line = f"{pt[0]} {pt[1]} {pt[2]}"
            if colors is not None:
                c = colors[i]
                line += f" {int(c[0])} {int(c[1])} {int(c[2])}"
            f.write(line + '\n')



def reconstruct_scene_sfm(image_paths, intrinsics, feature_dir='.featureout', match_file='matches.h5', output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load keypoints and matches
    with h5py.File(os.path.join(feature_dir, 'keypoints.h5'), 'r') as f_kp, \
         h5py.File(os.path.join(feature_dir, match_file), 'r') as f_match:
        
        # Load keypoints from the first image pair
        img1, img2 = image_paths[0], image_paths[1]
        name1, name2 = os.path.basename(img1), os.path.basename(img2)
             

        kpts1 = f_kp[name1][()]
        kpts2 = f_kp[name2][()]
        matches = f_match[name1][name2][()]
        

        valid = matches >= 0
        idx1 = np.where(valid)[0]
        idx2 = matches[valid]
        
        pts1 = kpts1[idx1]
        pts2 = kpts2[idx2]
             
        # Estimate the Fundamental matrix (for camera pose)
        F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_RANSAC)
        print(f"[INFO] Fundamental matrix shape: {F.shape}")

        # Recover relative pose (rotation and translation)
        _, R, t, mask_pose = cv2.recoverPose(F, pts1, pts2, intrinsics)
        print("[INFO] Recovered relative pose")
             
        if mask_pose is None:
            raise ValueError("[ERROR] pose recovery failed, mask is None")
        

        # Create Projection matrices
        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # Camera 1
        P2 = np.hstack((R, t))  # Camera 2

        # Triangulate 3D points from matching points in the two images
        pts1_h = cv2.convertPointsToHomogeneous(pts1[mask_pose.ravel() == 1])[:, 0, :]
        pts2_h = cv2.convertPointsToHomogeneous(pts2[mask_pose.ravel() == 1])[:, 0, :]

        pts4d_hom = cv2.triangulatePoints(P1, P2, pts1[mask_pose.ravel() == 1].T, pts2[mask_pose.ravel() == 1].T)
        pts3d = (pts4d_hom[:3] / pts4d_hom[3]).T  # Convert to 3D

        print(f"[INFO] Triangulated {pts3d.shape[0]} 3D points")

        # Save to PLY
        save_ply(os.path.join(output_dir, 'reconstruction_sfm.ply'), pts3d)

def save_ply(filename, points, colors=None):
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        if colors is not None:
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
        f.write('end_header\n')
        for i, pt in enumerate(points):
            line = f"{pt[0]} {pt[1]} {pt[2]}"
            if colors is not None:
                c = colors[i]
                line += f" {int(c[0])} {int(c[1])} {int(c[2])}"
            f.write(line + '\n')


# ------------------------------------------------------------------------------
# STEP 1: Relative Pose Estimation from Deep Matches
# ------------------------------------------------------------------------------
def compute_relative_pose_from_deep_matches(kp1, kp2, matches, K):
    """
    Given keypoints from two images and deep matching index pairs, estimate the
    relative pose (rotation and translation) using classical geometry.
    
    Parameters:
      kp1: np.array of shape (N1, 2) – keypoints from image 1
      kp2: np.array of shape (N2, 2) – keypoints from image 2
      matches: np.array of shape (M, 2) – each row contains indices [i, j] where
               kp1[i] matches kp2[j]
      K: 3x3 intrinsic camera matrix
      
    Returns:
      R: 3x3 relative rotation matrix (from image1 to image2)
      t: 3x1 relative translation vector
      mask: inlier mask from cv2.recoverPose (optional for debugging)
    """
    # Select matching keypoint coordinates
    pts1 = kp1[matches[:, 0]]
    pts2 = kp2[matches[:, 1]]
    
    # Compute the Essential matrix robustly using RANSAC.
    # The function returns the Essential matrix and an inlier mask.
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    # Recover the pose (rotation and translation) from the Essential matrix.
    # cv2.recoverPose uses the inlier keypoints to return the best estimate.
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    return R, t, mask


# ------------------------------------------------------------------------------
# STEP 2: Build a Relative Pose Graph from Stored Matches
# ------------------------------------------------------------------------------
def build_relative_pose_graph(feature_dir, K):
    """
    Reads in the saved keypoints and matches from HDF5 files, and computes a graph
    where each node is an image (identified by its filename) and each edge contains
    the relative pose between the two images.
    
    Parameters:
      feature_dir: Directory containing 'keypoints.h5' and 'matches.h5'
      K: Intrinsic camera matrix
    
    Returns:
      G: NetworkX undirected graph with each edge storing keys 'R' and 't'
    """
    G = nx.Graph()
    
    with h5py.File(os.path.join(feature_dir, 'keypoints.h5'), 'r') as f_kp, \
         h5py.File(os.path.join(feature_dir, 'matches.h5'), 'r') as f_match:
        
        # The matches file is organized as f_match[img1][img2]
        for key1 in f_match.keys():
            for key2 in f_match[key1].keys():
                # Load the matches (each row has two indices)
                matches = f_match[key1][key2][...]
                # Load keypoints for the two images (assume shape: (num_points, 2))
                kp1 = np.array(f_kp[key1][...])
                kp2 = np.array(f_kp[key2][...])
                
                try:
                    # Compute the relative pose using deep matching and geometry.
                    R, t, mask = compute_relative_pose_from_deep_matches(kp1, kp2, matches, K)
                    # Add this relative transformation as an edge attribute in the graph.
                    G.add_edge(key1, key2, R=R, t=t)
                except Exception as e:
                    print(f"Failed to compute relative pose for pair ({key1}, {key2}): {e}")
    
    return G


# ------------------------------------------------------------------------------
# STEP 3: Propagate a Global Pose via a Spanning Tree
# ------------------------------------------------------------------------------
def compute_global_poses(G):
    """
    Using the relative pose graph, compute a simplified set of global poses. The root
    image (first in the graph) is fixed as identity. Then, using a BFS traversal,
    we propagate the relative poses to obtain an initial guess for each image’s global
    orientation and position.
    
    Parameters:
      G: NetworkX graph with edge attributes 'R' and 't'
    
    Returns:
      global_poses: dict mapping image filename (node) to tuple (R, t) as global pose.
    """
    global_poses = {}
    # Choose an arbitrary root node and set its pose as identity.
    root = list(G.nodes())[0]
    global_poses[root] = (np.eye(3), np.zeros((3, 1)))
    
    # BFS traversal to propagate the poses.
    visited = set([root])
    queue = [root]
    
    while queue:
        current = queue.pop(0)
        current_R, current_t = global_poses[current]
        
        # Iterate over neighbors (images that have a relative transformation with current)
        for neighbor in G.neighbors(current):
            if neighbor not in visited:
                edge_data = G.get_edge_data(current, neighbor)
                R_edge = edge_data['R']
                t_edge = edge_data['t']
                
                # Propagate the pose.
                # If the relative pose from current to neighbor is (R_edge, t_edge),
                # then the global pose of the neighbor is:
                #   R_neighbor = R_edge @ current_R
                #   t_neighbor = R_edge @ current_t + t_edge
                neighbor_R = R_edge @ current_R
                neighbor_t = R_edge @ current_t + t_edge
                global_poses[neighbor] = (neighbor_R, neighbor_t)
                
                visited.add(neighbor)
                queue.append(neighbor)
    
    return global_poses


# A list to store prediction output for every processed image.
predictions = []
id_dict = {}

# Loop over each dataset/scene group from your train_labels.
# grouped = train_labels.groupby(['dataset', 'scene'])
for dataset, predictions in samples.items():
    print(f"Processing: dataset = {dataset}, scene = {predictions}")
    
    # Define a feature directory per dataset.
    feature_dir = os.path.join(workdir, 'featureout', dataset)
    os.makedirs(feature_dir, exist_ok=True)
    
    # Form full image file paths (adjust DATA_DIR according to your project)
    image_paths = group["image"].apply(lambda img: os.path.join(DATA_DIR, dataset, img)).tolist()
    
    # Step 1: Global Descriptor Extraction using DINOv2.
    descriptors = extract_dino_descriptor(image_paths)
    
    # Compute cosine similarity between descriptors.
    sim_matrix = descriptors @ descriptors.T  # shape: (num_images x num_images)
    num_images = len(image_paths)
    
    # Mask self-similarity entries.
    sim_matrix.fill_diagonal_(-1.0)
    
    # Shortlist image pairs based on a similarity threshold.
    shortlisted_pairs = set()
    for i in range(num_images):
        sim_scores = sim_matrix[i]
        top_similar = (sim_scores >= sim_th).nonzero(as_tuple=True)[0].tolist()
        
        # If the number of similar images is below a set minimum, take top-k.
        if len(top_similar) < min_pairs:
            num_valid = sim_scores.numel()
            k = min(min_pairs, num_valid)
            top_similar = torch.topk(sim_scores, k).indices.tolist()
            
        for j in top_similar:
            if i == j:
                continue
            pair = tuple(sorted((i, j)))
            shortlisted_pairs.add(pair)
    
    # Convert shortlisted pairs to a list containing tuples of (img_path1, img_path2, similarity)
    shortlisted_images_result = [
        (image_paths[i], image_paths[j], sim_matrix[i, j].item())
        for i, j in shortlisted_pairs
    ]
    
    # Optionally, limit the number of pairs globally if topN is set.
    if topN is not None:
        shortlisted_images_result = sorted(shortlisted_images_result, key=lambda x: -x[2])[:topN]
    
    # Step 2: Local Feature Extraction using ALIKED.
    t = time()
    detect_aliked(image_paths, feature_dir, num_features=4096, resize_to=1024, device=device)
    gc.collect()
    timings['feature_detection'].append(time() - t)
    print(f'Features detected in {time() - t:.4f} sec')
    
    # Step 3: Feature Matching using LightGlue.
    t = time()
    match_with_lightglue(image_paths, shortlisted_images_result, feature_dir=feature_dir, device=device, verbose=False)
    gc.collect()
    timings['feature_matching'].append(time() - t)
    print(f'Features matched in {time() - t:.4f} sec')
    
    # Step 4: Build the deep-enhanced relative pose graph and compute global poses.
    pose_graph = build_relative_pose_graph(feature_dir, intrinsics)
    global_poses = compute_global_poses(pose_graph)
    
    # Step 5: Store the predictions from this group.
    # Here each item in global_poses has the filename as key, and (R, t) as value.
    for image_file, (R, t_vec) in global_poses.items():
        image_id = id_dict.get(image_file, None)
        print(image_id)
        # We store the dataset, scene, image filename, and corresponding global pose.
        prediction = {
            'dataset': dataset,
            'scene': scene,
            'image_id': image_id,
            'image': image_file,  # image_file should be a filename, e.g., "img001.jpg"
            'rotation_matrix': R,  # 3x3 rotation matrix
            'translation_vector': t_vec.flatten()  # flatten in case it's a column vector
        }
        predictions.append(prediction)
    
    # reconstruct_scene_sfm(
    #     image_paths=image_paths,
    #     intrinsics=intrinsics,
    #     feature_dir=feature_dir,
    #     match_file='matches.h5',
    #     output_dir=os.path.join('output', f'{dataset}_{scene}')
    # )


# ------------------------------------------------------------------------------
# Generate the Submission CSV File
# ------------------------------------------------------------------------------

# Define formatting helper functions.
array_to_str = lambda array: ';'.join([f"{x:.09f}" for x in array])
none_to_str = lambda n: ';'.join(['nan'] * n)

# Set submission file path (e.g., for Kaggle working directory)
submission_file = '/kaggle/working/submission.csv'
with open(submission_file, 'w') as f:
    # Write header line.
    f.write('image_id, dataset,scene,image,rotation_matrix,translation_vector\n')
    # Write one line per prediction.
    for prediction in predictions:
        rotation = none_to_str(9) if prediction['rotation_matrix'] is None else array_to_str(prediction['rotation_matrix'].flatten())
        translation = none_to_str(3) if prediction['translation_vector'] is None else array_to_str(prediction['translation_vector'])
        f.write(f"{prediction['image_id']},{prediction['dataset']},{prediction['scene']},{prediction['image']},{rotation},{translation}\n")
        
# Optionally, if using Kaggle Notebook, preview the file.
!head {submission_file}


print(descriptors)


# import matplotlib.pyplot as plt
# from PIL import Image

# def visualize_similar_image_pairs(pairs, max_pairs=10, title="DINO Shortlisted Similar Images"):
#     plt.figure(figsize=(12, 3 * max_pairs))

#     for idx, (img1_path, img2_path, score) in enumerate(pairs[:max_pairs]):
#         img1 = Image.open(img1_path).convert("RGB")
#         img2 = Image.open(img2_path).convert("RGB")

#         # Show first image
#         ax1 = plt.subplot(max_pairs, 2, 2 * idx + 1)
#         ax1.imshow(img1)
#         ax1.set_title(f"Image 1\n{os.path.basename(img1_path)}")
#         ax1.axis("off")

#         # Show second image
#         ax2 = plt.subplot(max_pairs, 2, 2 * idx + 2)
#         ax2.imshow(img2)
#         ax2.set_title(f"Image 2\n{os.path.basename(img2_path)}\nSimilarity: {score:.3f}")
#         ax2.axis("off")

#     plt.suptitle(title, fontsize=16)
#     plt.tight_layout()
#     plt.show()



# visualize_similar_image_pairs(shortlisted_iamges_result, max_pairs=10)


# def visualize_matches(img_fnames, index_pairs, feature_dir=feature_dir, device=torch.device('cpu')):
#     with h5py.File(f'{feature_dir}/keypoints.h5', mode='r') as f_kp, \
#          h5py.File(f'{feature_dir}/descriptors.h5', mode='r') as f_desc, \
#          h5py.File(f'{feature_dir}/matches.h5', mode='r') as f_match:
        
#         for pair_idx in index_pairs:
#             fname1, fname2, _ = pair_idx  # Ignore the third element (the score)
#             key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]
#             kp1 = f_kp[key1][...]
#             kp2 = f_kp[key2][...]
#             matches = f_match[key1][key2][...]  # Matches for the pair

#             # Load the images
#             img1 = cv2.imread(fname1)
#             img2 = cv2.imread(fname2)

#             # Convert images to RGB (OpenCV loads them in BGR)
#             img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#             img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

#             # Draw keypoints on the images
#             img1_kp = img1_rgb.copy()
#             img2_kp = img2_rgb.copy()

#             for kp in kp1:
#                 cv2.circle(img1_kp, tuple(kp.astype(int)), 5, (255, 0, 0), -1)  # Red keypoints
#             for kp in kp2:
#                 cv2.circle(img2_kp, tuple(kp.astype(int)), 5, (0, 0, 255), -1)  # Blue keypoints

#             # Create a new image to concatenate both images side by side
#             concat_img = np.concatenate([img1_rgb, img2_rgb], axis=1)

#             # Draw lines between matched keypoints
#             for match in matches:
#                 pt1 = tuple(kp1[match[0]].astype(int))
#                 pt2 = tuple(kp2[match[1]].astype(int))
#                 pt2 = (pt2[0] + img1.shape[1], pt2[1])  # Adjust second image point position

#                 # Draw a line between the points on the concatenated image
#                 cv2.line(concat_img, pt1, pt2, (0, 255, 0), 2)  # Green lines for matches

#             # Show the final image
#             plt.figure(figsize=(12, 6))
#             plt.imshow(concat_img)
#             plt.axis('off')  # Turn off axis
#             plt.show()


# visualize_matches(image_paths, shortlisted_iamges_result, device=torch.device('cpu'))

