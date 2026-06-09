import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from pathlib import Path
from collections import defaultdict



# Constants
INPUT_DIR = "/kaggle/input/image-matching-challenge-2025"
OUTPUT_CSV = "submission.csv"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")


# Feature extractor placeholder
def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors


# Match descriptors between two images
def match_descriptors(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    return sorted(matches, key=lambda x: x.distance)


# Estimate pose placeholder (mocked for now)
def estimate_pose(img_path1, img_path2):
    R = np.eye(3).flatten()
    t = np.zeros(3)
    return R, t

# Build image graph for clustering
def build_similarity_graph(image_paths):
    descriptors = []
    for path in tqdm(image_paths, desc="Extracting features"):
        _, desc = extract_features(str(path))
        if desc is None:
            desc = np.zeros((1, 128))
        desc = desc.mean(axis=0)  # Very rough global descriptor
        descriptors.append(desc)

    descriptors = np.array(descriptors)
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=300.0, linkage="average")
    labels = clustering.fit_predict(descriptors)
    return labels


# Pose estimation per cluster
def estimate_cluster_poses(image_paths, labels):
    poses = defaultdict(dict)
    clusters = defaultdict(list)
    for path, label in zip(image_paths, labels):
        clusters[label].append(path)

    for label, imgs in clusters.items():
        if len(imgs) < 2:
            poses[label][imgs[0].name] = (np.full(9, np.nan), np.full(3, np.nan))
            continue

        base = imgs[0]
        R = np.eye(3).flatten()
        t = np.zeros(3)
        poses[label][base.name] = (R, t)
        for img in imgs[1:]:
            R_est, t_est = estimate_pose(str(base), str(img))
            poses[label][img.name] = (R_est, t_est)

    return poses


# Write submission file
def write_submission(cluster_labels, poses_by_cluster, image_paths, dataset_name="test"):
    rows = []
    for label, path in zip(cluster_labels, image_paths):
        image_name = path.name
        R, t = poses_by_cluster[label].get(image_name, (np.full(9, np.nan), np.full(3, np.nan)))
        R_flat = ";".join(map(str, R))
        t_flat = ";".join(map(str, t))
        rows.append([dataset_name, f"cluster{label}", image_name, R_flat, t_flat])

    df = pd.DataFrame(rows, columns=["dataset", "scene", "image", "rotation_matrix", "translation_vector"])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"âœ… Submission written to {OUTPUT_CSV}")


# Main pipeline
def main():
    print("ğŸš€ Starting Image Matching Challenge 2025 pipeline")
    dataset_folder = os.path.join(TEST_DIR, os.listdir(TEST_DIR)[0])
    image_paths = sorted(list(Path(dataset_folder).glob("*.png")))

    print("ğŸ”� Clustering images...")
    cluster_labels = build_similarity_graph(image_paths)

    print("ğŸ“� Estimating poses for each cluster...")
    poses_by_cluster = estimate_cluster_poses(image_paths, cluster_labels)

    print("ğŸ“� Writing submission file...")
    write_submission(cluster_labels, poses_by_cluster, image_paths)
    print("ğŸ�� Done!")

if __name__ == '__main__':
    main()

