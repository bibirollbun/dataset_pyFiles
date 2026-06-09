# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


"""
Image Matching Challenge 2025 Solution

"""

import os
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from collections import defaultdict
from tqdm import tqdm


# Step 1: Explore and Parse Data
# =============================

def explore_competition_data(base_path):
    """
    Explore the competition data structure and load necessary files
    """
    print("Exploring competition data...")
    
    # Load CSV files
    sample_submission = pd.read_csv(os.path.join(base_path, 'sample_submission.csv'))
    train_thresholds = pd.read_csv(os.path.join(base_path, 'train_thresholds.csv'))
    train_labels = pd.read_csv(os.path.join(base_path, 'train_labels.csv'))
    
    print("CSV Files:")
    print(f"  Sample Submission: {sample_submission.shape[0]} rows, {sample_submission.shape[1]} columns")
    print(f"  Train Thresholds: {train_thresholds.shape[0]} rows, {train_thresholds.shape[1]} columns")
    print(f"  Train Labels: {train_labels.shape[0]} rows, {train_labels.shape[1]} columns")
    
    # Sample of train labels
    print("\nSample of train_labels.csv:")
    print(train_labels.head())
    
    # Get list of datasets from train labels
    datasets = train_labels['dataset'].unique()
    print(f"\nFound {len(datasets)} datasets: {', '.join(datasets)}")
    
    # Explore train directory structure
    train_dir = os.path.join(base_path, 'train')
    train_datasets = os.listdir(train_dir)
    print(f"\nTrain directory contains {len(train_datasets)} datasets")
    
    # Sample a few train datasets
    for dataset in train_datasets[:3]:
        dataset_path = os.path.join(train_dir, dataset)
        if os.path.isdir(dataset_path):
            images = [f for f in os.listdir(dataset_path) if f.endswith(('.png', '.jpg'))]
            print(f"  Dataset '{dataset}': {len(images)} images")
            if images:
                print(f"    Sample image: {images[0]}")
    
    # Explore test directory structure
    test_dir = os.path.join(base_path, 'test')
    test_datasets = os.listdir(test_dir)
    print(f"\nTest directory contains {len(test_datasets)} datasets")
    
    # Sample a few test datasets
    for dataset in test_datasets[:3]:
        dataset_path = os.path.join(test_dir, dataset)
        if os.path.isdir(dataset_path):
            images = [f for f in os.listdir(dataset_path) if f.endswith(('.png', '.jpg'))]
            print(f"  Dataset '{dataset}': {len(images)} images")
            if images:
                print(f"    Sample image: {images[0]}")
    
    # Analyze scene distribution in train_labels
    scene_counts = train_labels.groupby(['dataset', 'scene']).size().reset_index(name='count')
    print("\nScene distribution in training data:")
    print(scene_counts.head(10))
    
    # Count outliers
    outliers = train_labels[train_labels['scene'] == 'outliers'].groupby('dataset').size()
    print("\nOutliers per dataset:")
    print(outliers)
    
    return sample_submission, train_thresholds, train_labels, datasets


# Step 2: Parse and Prepare Data
# =============================

def load_training_data(base_path, train_labels):
    """
    Load training data organized by dataset and scene
    """
    print("Loading training data...")
    
    # Create structure to hold training data
    training_data = {}
    
    # Group train_labels by dataset and scene
    grouped = train_labels.groupby(['dataset', 'scene'])
    
    # Process each group
    for (dataset, scene), group in grouped:
        if dataset not in training_data:
            training_data[dataset] = {}
        
        if scene not in training_data[dataset]:
            training_data[dataset][scene] = []
        
        # Add images to the scene
        for _, row in group.iterrows():
            image_file = row['image']
            image_path = os.path.join(base_path, 'train', dataset, image_file)
            
            # Extract rotation and translation if not NaN
            rotation_str = row['rotation_matrix']
            translation_str = row['translation_vector']
            
            rotation = None
            translation = None
            
            if rotation_str != 'nan;nan;nan;nan;nan;nan;nan;nan;nan':
                rotation_values = rotation_str.split(';')
                rotation = np.array([float(val) for val in rotation_values]).reshape(3, 3)
            
            if translation_str != 'nan;nan;nan':
                translation_values = translation_str.split(';')
                translation = np.array([float(val) for val in translation_values])
            
            training_data[dataset][scene].append({
                'file': image_file,
                'path': image_path,
                'rotation': rotation,
                'translation': translation
            })
    
    # Print summary
    print("Training data summary:")
    for dataset, scenes in training_data.items():
        print(f"  Dataset '{dataset}': {len(scenes)} scenes")
        for scene, images in scenes.items():
            print(f"    Scene '{scene}': {len(images)} images")
    
    return training_data


def load_test_data(base_path, datasets):
    """
    Load test data organized by dataset
    """
    print("Loading test data...")
    
    # Create structure to hold test data
    test_data = {}
    
    # Process each dataset
    for dataset in datasets:
        test_dir = os.path.join(base_path, 'test', dataset)
        if not os.path.exists(test_dir):
            print(f"  Warning: Test directory for dataset '{dataset}' not found")
            continue
        
        # List image files
        image_files = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg'))]
        
        # Store image paths
        test_data[dataset] = [
            {
                'file': image_file,
                'path': os.path.join(test_dir, image_file)
            }
            for image_file in image_files
        ]
    
    # Print summary
    print("Test data summary:")
    for dataset, images in test_data.items():
        print(f"  Dataset '{dataset}': {len(images)} images")
    
    return test_data


# Step 3: Feature Extraction
# =========================

def extract_features(image_path):
    """
    Extract SIFT features from an image
    """
    # Read image
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read image: {image_path}")
        return None, None
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create(nfeatures=2000)  # Use more features for better matching
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(img, None)
    
    return keypoints, descriptors


def extract_features_for_dataset(images):
    """
    Extract features for all images in a dataset
    """
    features = {}
    
    for image_info in tqdm(images, desc="Extracting features"):
        keypoints, descriptors = extract_features(image_info['path'])
        features[image_info['file']] = (keypoints, descriptors)
    
    return features


# Step 4: Feature Matching
# =======================

def match_features(desc1, desc2, ratio_threshold=0.75):
    """
    Match features between two images using Lowe's ratio test
    """
    # Handle None descriptors
    if desc1 is None or desc2 is None:
        return []
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    # Create FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Match descriptors
    try:
        matches = flann.knnMatch(desc1, desc2, k=2)
    except cv2.error:
        return []
    
    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)
    
    return good_matches


def build_similarity_matrix(features):
    """
    Build a similarity matrix for all image pairs
    """
    image_files = list(features.keys())
    n = len(image_files)
    
    # Initialize similarity matrix
    similarity_matrix = np.zeros((n, n))
    
    # Compute pairwise similarities
    for i in tqdm(range(n), desc="Building similarity matrix"):
        # Extract descriptors for image i
        _, desc_i = features[image_files[i]]
        
        for j in range(i+1, n):
            # Extract descriptors for image j
            _, desc_j = features[image_files[j]]
            
            # Match features
            matches = match_features(desc_i, desc_j)
            
            # Number of matches as similarity measure
            similarity = len(matches)
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # Symmetric
    
    return similarity_matrix, image_files


# Step 5: Image Clustering
# =======================

def cluster_images(similarity_matrix, image_files, threshold=10, min_samples=2):
    """
    Cluster images based on similarity matrix
    """
    # Convert similarity to distance (higher similarity = lower distance)
    max_sim = np.max(similarity_matrix)
    if max_sim > 0:
        distance_matrix = 1 - (similarity_matrix / max_sim)
    else:
        distance_matrix = 1 - similarity_matrix
    
    # Apply DBSCAN clustering
    db = DBSCAN(eps=0.5, min_samples=min_samples, metric='precomputed')
    labels = db.fit_predict(distance_matrix)
    
    # Organize images by cluster
    clusters = defaultdict(list)
    outliers = []
    
    for i, label in enumerate(labels):
        if label >= 0:  # Not noise
            clusters[f"cluster{label+1}"].append(image_files[i])
        else:  # Noise points are outliers
            outliers.append(image_files[i])
    
    print(f"Found {len(clusters)} clusters and {len(outliers)} outliers")
    
    return clusters, outliers


# Step 6: Pose Estimation
# ======================

def estimate_poses(features, clusters, K=None):
    """
    Estimate camera poses for images in each cluster
    """
    # If K is unknown, use a default intrinsic matrix
    if K is None:
        K = np.array([
            [1000, 0, 500],
            [0, 1000, 500],
            [0, 0, 1]
        ])
    
    poses = {}
    
    # Process each cluster
    for cluster_label, image_files in clusters.items():
        if len(image_files) < 2:
            continue
        
        # Use the first image as the reference frame
        reference_image = image_files[0]
        kp_ref, desc_ref = features[reference_image]
        
        # Set reference pose (identity rotation, zero translation)
        R_ref = np.eye(3)
        t_ref = np.zeros((3, 1))
        poses[reference_image] = {
            'R': R_ref,
            't': t_ref
        }
        
        # Estimate poses for other images relative to the reference
        for image_file in image_files[1:]:
            kp, desc = features[image_file]
            
            # Match features with reference image
            matches = match_features(desc_ref, desc)
            
            if len(matches) < 5:
                # Not enough matches for reliable pose estimation
                poses[image_file] = {
                    'R': np.eye(3),
                    't': np.zeros((3, 1))
                }
                continue
            
            # Extract matched points
            pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in matches])
            pts = np.float32([kp[m.trainIdx].pt for m in matches])
            
            # Estimate essential matrix
            E, mask = cv2.findEssentialMat(pts_ref, pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            
            # Recover pose (R, t) from essential matrix
            _, R, t, _ = cv2.recoverPose(E, pts_ref, pts, K, mask=mask)
            
            # Store the estimated pose
            poses[image_file] = {
                'R': R,
                't': t
            }
    
    return poses


# Step 7: Create Submission
# ========================

def create_submission(test_results, output_path='submission.csv'):
    """
    Create submission file with scene assignments and camera poses
    """
    # Prepare data for submission
    rows = []
    
    # Process each dataset
    for dataset, result in test_results.items():
        clusters = result['clusters']
        outliers = result['outliers']
        poses = result['poses']
        
        # Add clustered images with poses
        for cluster_label, image_files in clusters.items():
            for image_file in image_files:
                if image_file in poses:
                    R = poses[image_file]['R']
                    t = poses[image_file]['t']
                    
                    # Format rotation matrix (flatten to row-major)
                    rot_str = ";".join([str(x) for x in R.flatten()])
                    
                    # Format translation vector
                    trans_str = ";".join([str(x) for x in t.flatten()])
                    
                    rows.append([
                        dataset,
                        cluster_label,
                        image_file,
                        rot_str,
                        trans_str
                    ])
                else:
                    # Image is in cluster but pose could not be estimated
                    rows.append([
                        dataset,
                        cluster_label,
                        image_file,
                        "nan;nan;nan;nan;nan;nan;nan;nan;nan",
                        "nan;nan;nan"
                    ])
        
        # Add outliers
        for image_file in outliers:
            rows.append([
                dataset,
                "outliers",
                image_file,
                "nan;nan;nan;nan;nan;nan;nan;nan;nan",
                "nan;nan;nan"
            ])
    
    # Create DataFrame
    submission_df = pd.DataFrame(
        rows,
        columns=['dataset', 'scene', 'image', 'rotation_matrix', 'translation_vector']
    )
    
    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    
    return submission_df


# Step 8: Calculate Evaluation Metrics (For Training Data)
# ======================================================

def calculate_metrics(predictions, ground_truth):
    """
    Calculate evaluation metrics based on the competition's criteria
    """
    # Group predictions and ground truth by dataset
    pred_by_dataset = predictions.groupby('dataset')
    gt_by_dataset = ground_truth.groupby('dataset')
    
    dataset_scores = []
    
    # Process each dataset
    for dataset in predictions['dataset'].unique():
        pred_df = pred_by_dataset.get_group(dataset)
        gt_df = gt_by_dataset.get_group(dataset)
        
        # Convert to dictionaries for easier processing
        pred_dict = {}
        for _, row in pred_df.iterrows():
            pred_dict[row['image']] = {
                'scene': row['scene'],
                'rotation_matrix': row['rotation_matrix'],
                'translation_vector': row['translation_vector']
            }
        
        gt_dict = {}
        for _, row in gt_df.iterrows():
            gt_dict[row['image']] = {
                'scene': row['scene'],
                'rotation_matrix': row['rotation_matrix'],
                'translation_vector': row['translation_vector']
            }
        
        # Group images by predicted scene
        pred_scenes = defaultdict(list)
        for image, data in pred_dict.items():
            pred_scenes[data['scene']].append(image)
        
        # Group images by ground truth scene
        gt_scenes = defaultdict(list)
        for image, data in gt_dict.items():
            gt_scenes[data['scene']].append(image)
        
        # Calculate metrics for each ground truth scene
        scene_metrics = []
        for gt_scene, gt_images in gt_scenes.items():
            if gt_scene == 'outliers':
                continue  # Skip outliers for now
            
            # Find the best matching predicted scene
            best_maa = 0
            best_cluster_score = 0
            best_scene = None
            
            for pred_scene, pred_images in pred_scenes.items():
                if pred_scene == 'outliers':
                    continue
                
                # Calculate mAA (recall)
                common_images = set(gt_images) & set(pred_images)
                maa = len(common_images) / len(gt_images)
                
                # Calculate clustering score (precision)
                cluster_score = len(common_images) / len(pred_images)
                
                # Check if this is the best match
                if maa > best_maa or (maa == best_maa and cluster_score > best_cluster_score):
                    best_maa = maa
                    best_cluster_score = cluster_score
                    best_scene = pred_scene
            
            if best_scene:
                scene_metrics.append({
                    'gt_scene': gt_scene,
                    'pred_scene': best_scene,
                    'maa': best_maa,
                    'cluster_score': best_cluster_score,
                    'f1_score': 2 * best_maa * best_cluster_score / (best_maa + best_cluster_score) if (best_maa + best_cluster_score) > 0 else 0
                })
        
        # Calculate dataset-level metrics
        if scene_metrics:
            avg_maa = np.mean([m['maa'] for m in scene_metrics])
            avg_cluster_score = np.mean([m['cluster_score'] for m in scene_metrics])
            avg_f1 = np.mean([m['f1_score'] for m in scene_metrics])
            
            dataset_scores.append({
                'dataset': dataset,
                'avg_maa': avg_maa,
                'avg_cluster_score': avg_cluster_score,
                'avg_f1_score': avg_f1
            })
            
            print(f"Dataset '{dataset}' metrics:")
            print(f"  Average mAA (recall): {avg_maa:.4f}")
            print(f"  Average Clustering Score (precision): {avg_cluster_score:.4f}")
            print(f"  Average F1 Score: {avg_f1:.4f}")
    
    # Calculate overall score
    if dataset_scores:
        overall_f1 = np.mean([d['avg_f1_score'] for d in dataset_scores])
        print(f"\nOverall F1 Score: {overall_f1:.4f}")
    
    return dataset_scores


# Main Function
# ============

def main(base_path, output_path='submission.csv'):
    """
    Main function to run the entire pipeline
    """
    # Step 1: Explore competition data
    sample_submission, train_thresholds, train_labels, datasets = explore_competition_data(base_path)
    
    # Step 2: Load training and test data
    training_data = load_training_data(base_path, train_labels)
    test_data = load_test_data(base_path, datasets)
    
    # Results for each test dataset
    test_results = {}
    
    # Process each dataset
    for dataset in datasets:
        print(f"\n{'=' * 40}")
        print(f"Processing dataset: {dataset}")
        print(f"{'=' * 40}")
        
        if dataset not in test_data:
            print(f"No test data found for dataset '{dataset}', skipping")
            continue
        
        # Get test images for this dataset
        test_images = test_data[dataset]
        
        # Step 3: Extract features
        print("\nExtracting features for test images...")
        test_features = extract_features_for_dataset(test_images)
        
        # Step 4: Build similarity matrix
        print("\nBuilding similarity matrix...")
        similarity_matrix, image_files = build_similarity_matrix(test_features)
        
        # Step 5: Cluster images
        print("\nClustering images...")
        clusters, outliers = cluster_images(similarity_matrix, image_files)
        
        # Step 6: Estimate poses
        print("\nEstimating camera poses...")
        poses = estimate_poses(test_features, clusters)
        
        # Store results
        test_results[dataset] = {
            'clusters': clusters,
            'outliers': outliers,
            'poses': poses
        }
    
    # Step 7: Create submission
    print("\nCreating submission file...")
    submission_df = create_submission(test_results, output_path)
    
    return submission_df


# Example usage
if __name__ == "__main__":
    # Path to competition data
    base_path = "/kaggle/input/image-matching-challenge-2025"
    
    # Run the pipeline
    submission_df = main(base_path)

