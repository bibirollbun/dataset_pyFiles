# Import necessary libraries
import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tqdm import tqdm
from collections import defaultdict

# Set up paths
TEST_DIR = '/kaggle/input/image-matching-challenge-2025/test'
SUBMISSION_PATH = '/kaggle/working/submission.csv'

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Scan all test images and organize by dataset
def scan_test_images():
    images_by_dataset = defaultdict(list)
    for root, _, files in os.walk(TEST_DIR):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, TEST_DIR)
                parts = rel_path.split(os.sep)
                if parts:
                    dataset_name = parts[0]
                    images_by_dataset[dataset_name].append({
                        'full_path': full_path,
                        'rel_path': rel_path,
                        'filename': filename
                    })
    return images_by_dataset

# SIFT feature extractor
def extract_sift_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(img, None)
    if descriptors is None:
        return None
    if descriptors.shape[0] > 100:
        descriptors = descriptors[np.random.choice(descriptors.shape[0], 100, replace=False)]
    elif descriptors.shape[0] < 100:
        padding = np.zeros((100 - descriptors.shape[0], descriptors.shape[1]))
        descriptors = np.vstack([descriptors, padding])
    return descriptors.flatten()

# Feature extraction pipeline
def extract_features(image_paths):
    features = []
    valid_indices = []
    for i, path in enumerate(tqdm(image_paths)):
        feat = extract_sift_features(path)
        if feat is not None:
            features.append(feat)
            valid_indices.append(i)
    return np.array(features), valid_indices

# Pose estimation using essential matrix
def estimate_relative_pose(img1_path, img2_path, focal=1000, pp=(512, 512)):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        return None, None
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    if des1 is None or des2 is None or len(des1) < 8 or len(des2) < 8:
        return None, None
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return None, None
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    E, _ = cv2.findEssentialMat(pts1, pts2, focal=focal, pp=pp, method=cv2.RANSAC)
    if E is None:
        return None, None
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, focal=focal, pp=pp)
    return R, t

# Structure from Motion on cluster

def perform_sfm(images):
    if len(images) <= 1:
        return [None]*len(images), [None]*len(images)
    R_list = [np.eye(3)]
    t_list = [np.zeros(3)]
    ref_path = images[0]['full_path']
    for i in range(1, len(images)):
        R, t = estimate_relative_pose(ref_path, images[i]['full_path'])
        R_list.append(R if R is not None else None)
        t_list.append(t.flatten() if t is not None else None)
    return R_list, t_list

# Format R and t for submission
def format_pose(R, t):
    try:
        R_str = ';'.join([f"{x:.6f}" for x in R.flatten()])
        t_str = ';'.join([f"{x:.6f}" for x in t])
    except:
        R_str = 'nan;nan;nan;nan;nan;nan;nan;nan;nan'
        t_str = 'nan;nan;nan'
    return R_str, t_str

# Validate R and t string lengths and content
def validate_pose(pose_str, length):
    try:
        parts = [float(x) for x in pose_str.split(';')]
        return len(parts) == length
    except:
        return False

# Main execution function
def main():
    print("Starting pipeline...")
    images_by_dataset = scan_test_images()
    all_rows = []
    for dataset, images in images_by_dataset.items():
        print(f"Processing dataset: {dataset} with {len(images)} images")
        image_paths = [img['full_path'] for img in images]
        features, valid_idx = extract_features(image_paths)
        if features.shape[0] == 0:
            continue
        n_components = min(16, features.shape[0], features.shape[1])
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(features)
        k = max(2, min(10, len(reduced)//20))
        kmeans = KMeans(n_clusters=k, random_state=42).fit(reduced)
        labels = kmeans.labels_
        for cid in range(k):
            indices = [i for i, label in zip(valid_idx, labels) if label == cid]
            if not indices:
                continue
            cluster_images = [images[i] for i in indices]
            R_list, t_list = perform_sfm(cluster_images)
            for img, R, t in zip(cluster_images, R_list, t_list):
                R_str, t_str = format_pose(R, t)
                all_rows.append({
                    'dataset': dataset,
                    'scene': f"cluster{cid}",
                    'image': img['filename'],
                    'rotation_matrix': R_str,
                    'translation_vector': t_str
                })

    # Final DataFrame
    df = pd.DataFrame(all_rows)
    df = df[['dataset', 'scene', 'image', 'rotation_matrix', 'translation_vector']]
    df = df[df['rotation_matrix'].apply(lambda x: validate_pose(x, 9))]
    df = df[df['translation_vector'].apply(lambda x: validate_pose(x, 3))]
    df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved: {SUBMISSION_PATH} with {len(df)} rows")

if __name__ == "__main__":
    main()


