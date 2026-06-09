# Import pre-installed libraries
import os
import numpy as np
import pandas as pd
import cv2
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


# Define paths
DATASET_PATH = "/kaggle/input/image-matching-challenge-2025"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH = os.path.join(DATASET_PATH, "test")

# List datasets and images
datasets = [d for d in os.listdir(TRAIN_PATH) if os.path.isdir(os.path.join(TRAIN_PATH, d))]
print(f"Datasets: {datasets}")

# Example: Load one dataset
dataset_name = datasets[0]
dataset_path = os.path.join(TRAIN_PATH, dataset_name)
image_paths = [
    os.path.join(dataset_path, img)
    for img in os.listdir(dataset_path)
    if img.endswith(".png") and not img.startswith("LICENSE")
]
print(f"Images in {dataset_name}: {len(image_paths)}")


# Function to extract ORB features
def extract_orb_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return descriptors

# Extract features for all images in a dataset
def extract_dataset_features(image_paths):
    features = []
    valid_image_paths = []
    for img_path in tqdm(image_paths, desc="Extracting Features"):
        desc = extract_orb_features(img_path)
        if desc is not None:
            features.append(desc)
            valid_image_paths.append(img_path)
    return features, valid_image_paths

# Extract features
features, image_paths = extract_dataset_features(image_paths)
print(f"Extracted features for {len(features)} images.")


# Flatten descriptors into a single array
def flatten_descriptors(features):
    all_desc = np.vstack(features)
    return all_desc.astype('float32')

# Flatten features
flat_features = flatten_descriptors(features)
print(f"Flattened features to shape: {flat_features.shape}")


# Cluster images using approximate nearest neighbors
def cluster_images(features, eps=50, min_samples=2):
    print("Running DBSCAN with approximate nearest neighbors...")
    
    # Compute nearest neighbors
    nbrs = NearestNeighbors(metric='euclidean').fit(features)
    distances, indices = nbrs.kneighbors(features, n_neighbors=min_samples)
    
    # Assign cluster labels based on neighbor indices
    clusters = np.full(len(features), -1)  # Initialize all as outliers
    current_cluster = 0
    
    for i in range(len(features)):
        if clusters[i] != -1:  # Already assigned
            continue
        
        # Find neighbors within eps
        neighbors = [j for j in indices[i] if distances[i][np.where(indices[i] == j)] <= eps]
        if len(neighbors) >= min_samples:
            clusters[i] = current_cluster
            stack = list(neighbors)
            
            while stack:
                node = stack.pop()
                if clusters[node] == -1:
                    clusters[node] = current_cluster
                    new_neighbors = [j for j in indices[node] if distances[node][np.where(indices[node] == j)] <= eps]
                    stack.extend(new_neighbors)
            
            current_cluster += 1
    
    return clusters

# Cluster images
clusters = cluster_images(flat_features)
print(f"Cluster labels: {np.unique(clusters)}")


# Ensure alignment between clusters and image_paths
valid_image_paths = []
valid_clusters = []

for i, img_path in enumerate(image_paths):
    if i < len(clusters):  # Check if cluster label exists for this image
        valid_image_paths.append(img_path)
        valid_clusters.append(clusters[i])

# Replace image_paths and clusters with the aligned versions
image_paths = valid_image_paths
clusters = np.array(valid_clusters)

print(f"Aligned {len(image_paths)} images with {len(clusters)} cluster labels.")


# Perform basic SfM using OpenCV
def reconstruct_scene(image_paths, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Match features between pairs of images
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    for i in range(len(image_paths) - 1):
        img1 = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image_paths[i + 1], cv2.IMREAD_GRAYSCALE)
        
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None:
            continue  # Skip if descriptors are missing
        
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        
        # Compute Essential Matrix and recover pose
        focal = 1.0  # Assume normalized coordinates
        pp = (0, 0)  # Principal point
        E, mask = cv2.findEssentialMat(pts1, pts2, focal, pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, focal=focal, pp=pp)
        
        # Save results (rotation matrix and translation vector)
        np.save(os.path.join(output_dir, f"pose_{i}.npy"), {"R": R, "t": t})

# Reconstruct scenes for each cluster
for cluster_id in np.unique(clusters):
    if cluster_id == -1:
        continue  # Skip outliers
    
    cluster_images = [image_paths[i] for i, label in enumerate(clusters) if label == cluster_id]
    output_dir = f"/kaggle/working/scene_{cluster_id}"
    reconstruct_scene(cluster_images, output_dir)
    print(f"Reconstructed scene {cluster_id} with {len(cluster_images)} images.")


def generate_submission(clusters, image_paths, output_path):
    rows = []
    for i, label in enumerate(clusters):
        dataset = dataset_name  # Use the current dataset name
        scene = f"cluster{label}" if label != -1 else "outliers"
        image = os.path.basename(image_paths[i])
        
        # Placeholder for rotation matrix and translation vector
        rotation_matrix = "nan;nan;nan;nan;nan;nan;nan;nan;nan" if label == -1 else "0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9"
        translation_vector = "nan;nan;nan" if label == -1 else "0.1;0.2;0.3"
        
        rows.append([dataset, scene, image, rotation_matrix, translation_vector])
    
    df = pd.DataFrame(rows, columns=["dataset", "scene", "image", "rotation_matrix", "translation_vector"])
    df.to_csv(output_path, index=False)
    print(f"Submission file saved to {output_path}")

# Save submission file
generate_submission(clusters, image_paths, "/kaggle/working/submission.csv")




