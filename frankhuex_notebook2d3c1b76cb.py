# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
# DBSCAN is no longer the primary method, but we keep it for context if needed
from sklearn.cluster import DBSCAN 
from sklearn.metrics import adjusted_rand_score
from tqdm.notebook import tqdm
from pathlib import Path
import timm
import subprocess
import os
import shutil

# --- NEW IMPORTS for the Coarse-to-Fine Strategy ---
import cv2
import networkx as nx
from sklearn.neighbors import NearestNeighbors


# --- Configuration ---
ROOT_DIR = Path("/kaggle/input/image-matching-challenge-2025")
TRAIN_DIR = ROOT_DIR / "train"
TEST_DIR = ROOT_DIR / "test"
TRAIN_LABELS_PATH = ROOT_DIR / "train_labels.csv"


# 将缓存和输出目录放在可写的路径下（当前工作目录）
CACHE_DIR = Path("/kaggle/working/cache")  # 修改为可写路径
COLMAP_OUTPUT_DIR = Path("/kaggle/working/colmap_output")  # 修改为可写路径
SUBMISSION_PATH = "/kaggle/working/submission.csv"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
COLMAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model configuration
DINO_MODEL_NAME = "dinov2_vits14"

# --- NEW Hyperparameters for Coarse-to-Fine Strategy ---
# 1. For Coarse Search: How many initial candidates to check for each image
K_NEIGHBORS = 50 

# 2. For Fine Matching: Minimum number of geometrically consistent matches (inliers)
#    to consider two images as part of the same scene.
MIN_INLIERS = 15


# ### Part 2: Core Helper Functions ###
# This cell defines all necessary helper functions for device handling,
# model loading, and feature extraction.

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    # if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
print(f"Using device: {DEVICE}")

def load_models(device):
    """Loads the DINOv2 model. SigLIP is no longer needed for clustering."""
    print("Loading DINOv2 model...")
    dino_model = torch.hub.load('facebookresearch/dinov2', DINO_MODEL_NAME, verbose=False)
    dino_model.to(device).eval()
    
    dino_transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    # We return placeholders for the siglip model to keep the function signature consistent
    # if other parts of the code were to use it, but it's not loaded to save memory.
    return dino_model, None, dino_transform, None

def extract_features(image_paths, model, transform, device, cache_path):
    """Extracts features for a list of images using a given model."""
    if cache_path.exists():
        print(f"Loading features from cache: {cache_path}")
        return np.load(cache_path)
        
    print(f"Extracting features for {cache_path.stem}...")
    features = []
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc=f"Extracting {cache_path.stem}"):
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)
                feature = model(img_tensor)
                features.append(feature.cpu().numpy().flatten())
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")
                # DINOv2 small model has an embedding dimension of 384
                features.append(np.zeros(384))
                
    features_np = np.array(features, dtype=np.float32)
    features_np = features_np / np.linalg.norm(features_np, axis=1, keepdims=True)
    
    print(f"Saving features to cache: {cache_path}")
    np.save(cache_path, features_np)
    return features_np


# ### Part 3: New Helper Function for Geometric Verification
# This cell replaces the old tuning logic with the core function for our new strategy.

def geometric_verification(img_path1, img_path2, min_inliers=MIN_INLIERS):
    """
    Performs geometric verification between two images using SIFT and RANSAC.
    
    Returns:
        True if the number of inlier matches is above the threshold, False otherwise.
    """
    try:
        # 1. Read images and convert to grayscale
        img1 = cv2.imread(str(img_path1), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(img_path2), cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None: return False

        # 2. Detect and compute SIFT features
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            return False

        # 3. Match features using a Brute-Force Matcher with Lowe's Ratio Test
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # 4. Perform RANSAC to find homography and count inliers
        if len(good_matches) > min_inliers:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # findHomography is a robust way to check for geometric consistency
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if mask is None: return False
            
            num_inliers = np.sum(mask)
            return num_inliers >= min_inliers
        else:
            return False
            
    except Exception as e:
        # print(f"Error during geometric verification between {img_path1.name} and {img_path2.name}: {e}")
        return False


# ### Part 4: New Clustering logic for the `test` set (Coarse-to-Fine)

all_test_clusters = {}

# We only need the DINO model for the coarse search phase
dino_model, _, dino_transform, _ = load_models(DEVICE)
dino_model.eval()

for dataset_path in sorted(TEST_DIR.iterdir()):
    if not dataset_path.is_dir():
        continue
    dataset_name = dataset_path.name
    print(f"\n--- Processing Test Dataset: {dataset_name} with Coarse-to-Fine Strategy ---")
    
    test_image_paths = sorted(list(dataset_path.glob("*.png")))
    num_images = len(test_image_paths)
    if num_images == 0:
        print(f"No images found in {dataset_name}, skipping.")
        continue

    # --- 1. Coarse Search: Use DINO features to find candidate pairs ---
    print(f"Step 1/3: Coarse search using DINOv2 features for {num_images} images.")
    # Use only DINO features, as they are better for instance-level geometry
    dino_features_test = extract_features(test_image_paths, dino_model, dino_transform, DEVICE, CACHE_DIR / f"test_{dataset_name}_features_dino.npy")
    
    # Use NearestNeighbors to find top K candidates for each image
    nn_model = NearestNeighbors(n_neighbors=min(K_NEIGHBORS, num_images), metric='cosine', algorithm='brute')
    nn_model.fit(dino_features_test)
    distances, indices = nn_model.kneighbors(dino_features_test)
    
    # --- 2. Fine Matching: Build a graph using geometric verification ---
    print(f"Step 2/3: Fine matching on candidate pairs using geometric verification.")
    G = nx.Graph()
    G.add_nodes_from(range(num_images)) # Each image is a node

    # A set to keep track of pairs we've already checked
    verified_pairs = set()

    for i in tqdm(range(num_images), desc="Verifying pairs"):
        for j in indices[i, 1:]: # The first neighbor is always the image itself
            # Ensure we check each pair only once (e.g., (1, 5) but not (5, 1))
            if (min(i, j), max(i, j)) in verified_pairs:
                continue
            
            if geometric_verification(test_image_paths[i], test_image_paths[j]):
                G.add_edge(i, j)
            
            verified_pairs.add((min(i, j), max(i, j)))

    # --- 3. Final Clustering: Find connected components in the graph ---
    print(f"Step 3/3: Finding clusters from the verified graph.")
    components = list(nx.connected_components(G))
    
    # Format the results into the required dictionary structure
    dataset_clusters = {}
    used_indices = set()
    for cluster_id, component in enumerate(components):
        # Only consider components with at least 2 images as a valid cluster
        if len(component) >= 2:
            scene_name = f"cluster_{cluster_id}"
            dataset_clusters[scene_name] = [test_image_paths[i] for i in component]
            used_indices.update(component)

    # Images not in any valid cluster are outliers
    dataset_clusters["outliers"] = [test_image_paths[i] for i in range(num_images) if i not in used_indices]
    
    all_test_clusters[dataset_name] = dataset_clusters
    
    num_clusters = len(components) - (1 if 'outliers' in dataset_clusters and not dataset_clusters['outliers'] else 0)
    num_outliers = len(dataset_clusters.get("outliers", []))
    print(f"Clustering for {dataset_name} complete: Found {num_clusters} clusters and {num_outliers} outliers.")


!pip install pycolmap


import pycolmap
from pathlib import Path
import shutil

print(f"pycolmap版本: 0.4.0")

def load_reconstruction_from_output(output_path):
    """从输出目录加载重建结果 - 独立函数"""
    poses = {}
    try:
        # 查找重建目录（通常是0, 1, 2等数字命名的目录）
        for item in output_path.iterdir():
            if item.is_dir() and item.name.isdigit():
                try:
                    reconstruction = pycolmap.Reconstruction(str(item))
                    for image_id, image in reconstruction.images.items():
                        try:
                            rotation_matrix = pycolmap.qvec_to_rotmat(image.qvec)
                            translation_vector = image.tvec
                            
                            rot_str = ";".join([f"{x:.8f}" for x in rotation_matrix.flatten()])
                            trans_str = ";".join([f"{x:.8f}" for x in translation_vector.flatten()])
                            
                            poses[image.name] = (rot_str, trans_str)
                        except Exception as e:
                            print(f"处理图片 {image.name} 时出错: {e}")
                except Exception as e:
                    print(f"加载重建目录 {item} 时出错: {e}")
    except Exception as e:
        print(f"从输出目录加载重建时出错: {e}")
    
    return poses

def estimate_poses_pycolmap_0_4_0(image_paths_in_cluster, work_dir):
    """
    兼容pycolmap 0.4.0版本的姿态估计
    """
    print(f"Starting pose estimation for cluster: {work_dir.name}")
    
    image_dir = work_dir / "images"
    
    if work_dir.exists():
        shutil.rmtree(work_dir)
    image_dir.mkdir(parents=True)
    
    # 拷贝图片
    for img_path in image_paths_in_cluster:
        shutil.copy2(img_path, image_dir / img_path.name)
    
    try:
        database_path = work_dir / "database.db"
        output_path = work_dir / "sparse"
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("Step 1: Extracting features...")
        pycolmap.extract_features(str(database_path), str(image_dir))
        
        print("Step 2: Matching features...")
        pycolmap.match_exhaustive(str(database_path))
        
        print("Step 3: Running reconstruction...")
        result = pycolmap.incremental_mapping(str(database_path), str(image_dir), str(output_path))
        
        print(f"Reconstruction result type: {type(result)}")
        
        poses = {}
        
        # 处理0.4.0版本的返回值
        if result is not None:
            if isinstance(result, list) and len(result) > 0:
                # 可能是重建对象列表
                for i, recon in enumerate(result):
                    if hasattr(recon, 'images'):
                        print(f"处理重建 {i}, 图片数量: {len(recon.images)}")
                        for image_id, image in recon.images.items():
                            try:
                                rotation_matrix = pycolmap.qvec_to_rotmat(image.qvec)
                                translation_vector = image.tvec
                                
                                rot_str = ";".join([f"{x:.8f}" for x in rotation_matrix.flatten()])
                                trans_str = ";".join([f"{x:.8f}" for x in translation_vector.flatten()])
                                
                                poses[image.name] = (rot_str, trans_str)
                            except Exception as e:
                                print(f"处理图片 {image.name} 时出错: {e}")
            else:
                # 尝试从输出目录加载
                poses = load_reconstruction_from_output(output_path)
        else:
            # 如果返回None，从输出目录加载
            poses = load_reconstruction_from_output(output_path)
        
        if poses:
            print(f"Pose estimation successful for {work_dir.name}. Registered {len(poses)} images.")
        else:
            print(f"No images registered for {work_dir.name}")
            
        return poses if poses else None
        
    except Exception as e:
        print(f"Pose estimation failed for {work_dir.name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def estimate_poses_with_options(image_paths_in_cluster, work_dir):
    """
    带选项的版本，尝试不同的参数组合
    """
    print(f"Using options method for {work_dir.name}")
    
    image_dir = work_dir / "images"
    
    if work_dir.exists():
        shutil.rmtree(work_dir)
    image_dir.mkdir(parents=True)
    
    # 拷贝图片
    for img_path in image_paths_in_cluster:
        shutil.copy2(img_path, image_dir / img_path.name)
    
    try:
        database_path = work_dir / "database.db"
        output_path = work_dir / "sparse"
        
        # 尝试不同的相机模型
        camera_models = ["SIMPLE_RADIAL", "SIMPLE_PINHOLE", "RADIAL"]
        
        for camera_model in camera_models:
            try:
                print(f"尝试相机模型: {camera_model}")
                
                # 清理之前的输出
                if output_path.exists():
                    shutil.rmtree(output_path)
                output_path.mkdir(parents=True)
                
                if database_path.exists():
                    database_path.unlink()
                
                # 运行COLMAP流程
                pycolmap.extract_features(str(database_path), str(image_dir))
                pycolmap.match_exhaustive(str(database_path))
                result = pycolmap.incremental_mapping(str(database_path), str(image_dir), str(output_path))
                
                # 检查结果
                poses = load_reconstruction_from_output(output_path)
                
                if len(poses) > len(image_paths_in_cluster) * 0.3:  # 如果成功注册了30%以上的图片
                    print(f"使用相机模型 {camera_model} 成功注册 {len(poses)} 张图片")
                    return poses
                else:
                    print(f"相机模型 {camera_model} 只注册了 {len(poses)} 张图片，继续尝试...")
                    
            except Exception as e:
                print(f"相机模型 {camera_model} 失败: {e}")
                continue
        
        return None
        
    except Exception as e:
        print(f"Options method failed: {e}")
        return None

# --- 主循环 ---
all_poses = {}
nan_rotation = ";".join(["nan"] * 9)
nan_translation = ";".join(["nan"] * 3)

for dataset_name, clusters in all_test_clusters.items():
    dataset_poses = {}
    print(f"\n--- Estimating Poses for Dataset: {dataset_name} ---")
    
    for scene_name, image_paths in tqdm(clusters.items(), desc=f"COLMAP for {dataset_name}"):
        if scene_name == "outliers" or len(image_paths) < 3:
            for img_path in image_paths: 
                dataset_poses[img_path.name] = (nan_rotation, nan_translation)
            continue
            
        work_dir = COLMAP_OUTPUT_DIR / dataset_name / scene_name
        
        # 先尝试基础方法
        poses = estimate_poses_pycolmap_0_4_0(image_paths, work_dir)
        
        # 如果失败，尝试带选项的方法
        if not poses:
            print(f"基础方法失败，尝试带选项的方法 for {scene_name}")
            poses = estimate_poses_with_options(image_paths, work_dir)
        
        # 记录结果
        for img_path in image_paths:
            if poses and img_path.name in poses:
                rot, trans = poses[img_path.name]
                dataset_poses[img_path.name] = (rot, trans)
            else:
                dataset_poses[img_path.name] = (nan_rotation, nan_translation)
        
        # 打印调试信息
        registered_count = sum(1 for v in dataset_poses.values() if "nan" not in v[0])
        print(f"{scene_name}: {registered_count}/{len(image_paths)} images registered")
                
    all_poses[dataset_name] = dataset_poses


# !pip install pycolmap
# !apt install colmap


# import subprocess
# import shutil
# from pathlib import Path
# import numpy as np

# def qvec_to_rotmat(qvec):
#     """手动实现四元数到旋转矩阵的转换"""
#     w, x, y, z = qvec
#     return np.array([
#         [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
#         [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
#         [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
#     ])

# def parse_colmap_images_file(images_file_path):
#     """解析COLMAP的images.txt文件来获取姿态"""
#     poses = {}
#     try:
#         with open(images_file_path, 'r') as f:
#             lines = f.readlines()
        
#         i = 0
#         while i < len(lines):
#             line = lines[i].strip()
#             if line and not line.startswith('#'):
#                 parts = line.split()
#                 if len(parts) >= 10:
#                     image_name = parts[-1]
#                     qw, qx, qy, qz = map(float, parts[1:5])
#                     tx, ty, tz = map(float, parts[5:8])
                    
#                     qvec = [qw, qx, qy, qz]
#                     tvec = [tx, ty, tz]
                    
#                     rotation_matrix = qvec_to_rotmat(qvec)
                    
#                     rot_str = ";".join([f"{x:.8f}" for x in rotation_matrix.flatten()])
#                     trans_str = ";".join([f"{x:.8f}" for x in tvec])
                    
#                     poses[image_name] = (rot_str, trans_str)
                
#                 i += 4
#             else:
#                 i += 1
                
#     except Exception as e:
#         print(f"解析images.txt文件出错: {e}")
    
#     return poses

# def convert_colmap_model_to_txt(model_path, output_path):
#     """使用model_converter将二进制模型转换为文本格式"""
#     cmd_converter = [
#         "colmap", "model_converter",
#         "--input_path", str(model_path),
#         "--output_path", str(output_path),
#         "--output_type", "TXT"
#     ]
#     try:
#         result = subprocess.run(cmd_converter, check=True, capture_output=True, text=True)
#         print("Model conversion successful")
#         return True
#     except subprocess.CalledProcessError as e:
#         print(f"Model conversion failed: {e.stderr}")
#         return False

# def estimate_poses_with_colmap(image_paths_in_cluster, work_dir):
#     """使用命令行COLMAP进行姿态估计"""
#     print(f"Starting pose estimation for cluster: {work_dir.name}")
    
#     # 设置工作目录
#     image_dir = work_dir / "images"
#     database_path = work_dir / "database.db"
    
#     if work_dir.exists():
#         shutil.rmtree(work_dir)
#     image_dir.mkdir(parents=True)
    
#     # 拷贝图片
#     for img_path in image_paths_in_cluster:
#         shutil.copy2(img_path, image_dir / img_path.name)

#     # 1. 特征提取
#     print("Step 1/4: Feature Extraction")
#     cmd_feature_extractor = [
#         "colmap", "feature_extractor",
#         "--database_path", str(database_path),
#         "--image_path", str(image_dir),
#         "--ImageReader.camera_model", "SIMPLE_RADIAL",
#         "--SiftExtraction.use_gpu", "false",
#     ]
#     try:
#         subprocess.run(cmd_feature_extractor, check=True, capture_output=True, text=True)
#         print("Feature extraction successful")
#     except subprocess.CalledProcessError as e:
#         print(f"Feature Extraction FAILED: {e.stderr}")
#         return None
#     except FileNotFoundError:
#         print("COLMAP not found. Please ensure COLMAP is installed.")
#         return None

#     # 2. 特征匹配
#     print("Step 2/4: Feature Matching")
#     cmd_exhaustive_matcher = [
#         "colmap", "exhaustive_matcher",
#         "--database_path", str(database_path),
#         "--SiftMatching.use_gpu", "false",
#     ]
#     try:
#         subprocess.run(cmd_exhaustive_matcher, check=True, capture_output=True, text=True)
#         print("Feature matching successful")
#     except subprocess.CalledProcessError as e:
#         print(f"Feature Matching FAILED: {e.stderr}")
#         return None

#     # 3. 场景重建
#     print("Step 3/4: Scene Reconstruction")
#     sparse_model_dir = work_dir / "sparse"
#     sparse_model_dir.mkdir(parents=True, exist_ok=True)
    
#     cmd_mapper = [
#         "colmap", "mapper",
#         "--database_path", str(database_path),
#         "--image_path", str(image_dir),
#         "--output_path", str(sparse_model_dir),
#     ]
#     try:
#         subprocess.run(cmd_mapper, check=True, capture_output=True, text=True)
#         print("Scene reconstruction successful")
#     except subprocess.CalledProcessError as e:
#         print(f"Mapper FAILED: {e.stderr}")
#         return None

#     # 4. 查找重建目录并转换为文本格式
#     print("Step 4/4: Converting and parsing reconstruction")
    
#     # 查找重建目录
#     reconstruction_dirs = []
#     for item in sparse_model_dir.iterdir():
#         if item.is_dir() and item.name.isdigit():
#             reconstruction_dirs.append(item)
    
#     if not reconstruction_dirs:
#         print("No reconstruction directories found")
#         return None
    
#     reconstruction_path = reconstruction_dirs[0]
#     print(f"Found reconstruction in {reconstruction_path}")
    
#     # 转换为文本格式
#     txt_output_dir = work_dir / "txt_output"
#     txt_output_dir.mkdir(parents=True, exist_ok=True)
    
#     if not convert_colmap_model_to_txt(reconstruction_path, txt_output_dir):
#         return None
    
#     # 解析images.txt文件
#     images_file_path = txt_output_dir / "images.txt"
#     if not images_file_path.exists():
#         print("images.txt not found after conversion")
#         return None
    
#     poses = parse_colmap_images_file(images_file_path)
    
#     if poses:
#         print(f"Pose estimation successful. Registered {len(poses)} images.")
#     else:
#         print("No poses extracted from images.txt")
    
#     return poses

# # --- 主循环 ---
# all_poses = {}
# nan_rotation = ";".join(["nan"] * 9)
# nan_translation = ";".join(["nan"] * 3)

# for dataset_name, clusters in all_test_clusters.items():
#     dataset_poses = {}
#     print(f"\n--- Estimating Poses for Dataset: {dataset_name} ---")
    
#     for scene_name, image_paths in tqdm(clusters.items(), desc=f"COLMAP for {dataset_name}"):
#         if scene_name == "outliers" or len(image_paths) < 2:
#             for img_path in image_paths: 
#                 dataset_poses[img_path.name] = (nan_rotation, nan_translation)
#             continue
            
#         work_dir = COLMAP_OUTPUT_DIR / dataset_name / scene_name
        
#         poses = estimate_poses_with_colmap(image_paths, work_dir)
        
#         # 记录结果
#         for img_path in image_paths:
#             if poses and img_path.name in poses:
#                 rot, trans = poses[img_path.name]
#                 dataset_poses[img_path.name] = (rot, trans)
#             else:
#                 dataset_poses[img_path.name] = (nan_rotation, nan_translation)
        
#         registered_count = sum(1 for v in dataset_poses.values() if "nan" not in v[0])
#         print(f"{scene_name}: {registered_count}/{len(image_paths)} images registered")
                
#     all_poses[dataset_name] = dataset_poses


submission_data = []
for dataset_name, clusters in all_test_clusters.items():
    for scene_name, image_paths in clusters.items():
        for img_path in image_paths:
            img_name = img_path.name
            rotation, translation = all_poses[dataset_name].get(img_name, (nan_rotation, nan_translation))
            submission_data.append({
                "dataset": dataset_name,
                "scene": scene_name,
                "image": img_name,
                "rotation_matrix": rotation,
                "translation_vector": translation,
            })
            
submission_df = pd.DataFrame(submission_data)
submission_df.to_csv(SUBMISSION_PATH, index=False)

print(f"\nSubmission file created at: {SUBMISSION_PATH}")
print("--- Submission File Head ---")
print(submission_df)

