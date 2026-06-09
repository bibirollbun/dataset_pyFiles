import pandas as pd

df_submission = pd.read_csv('sample_submission.csv')
df_labels = pd.read_csv('train_labels.csv')
df_thresholds = pd.read_csv('train_thresholds.csv')

print(df_submission.shape)
print(df_labels.shape)
print(df_thresholds.shape)


# Examine the shape, data types, and descriptive statistics of each DataFrame.
print("df_submission:")
display(df_submission.info())
display(df_submission.describe(include='all'))
print("\n")

print("df_labels:")
display(df_labels.info())
display(df_labels.describe(include='all'))
print("\n")

print("df_thresholds:")
display(df_thresholds.info())
display(df_thresholds.describe(include='all'))
print("\n")

# Analyze df_labels in detail and visualize key variables.
import matplotlib.pyplot as plt

# Visualize the distribution of 'dataset' and 'scene'
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
df_labels['dataset'].value_counts().plot(kind='bar')
plt.title('Distribution of Datasets in df_labels')
plt.xlabel('Dataset')
plt.ylabel('Count')
plt.subplot(1, 2, 2)
df_labels['scene'].value_counts().plot(kind='bar')
plt.title('Distribution of Scenes in df_labels')
plt.xlabel('Scene')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Investigate the relationship between df_labels and df_thresholds.
# Identify a potential key for merging.
print("Potential merge keys:")
print("  - 'dataset' and 'scene'")
print("  Further analysis needed to confirm this and how to handle the 'thresholds' column.")


# Merge df_labels and df_thresholds
merged_df = pd.merge(df_labels, df_thresholds, on=['dataset', 'scene'], how='left')

# Display the first few rows of the merged DataFrame
display(merged_df.head())

# Check for missing values after the merge
print(merged_df.info())

# Investigate the 'thresholds' column in the merged DataFrame
print("\nUnique values in 'thresholds' column:")
print(merged_df['thresholds'].unique())


# Analyze the missing values in the 'thresholds' column
print(merged_df['thresholds'].isnull().sum())
missing_thresholds = merged_df[merged_df['thresholds'].isnull()]
display(missing_thresholds.head())

# Investigate the 'thresholds' column further
print("\nValue counts for 'thresholds':")
display(merged_df['thresholds'].value_counts())

print("\nUnique values of 'dataset' and 'scene' for missing thresholds")
display(missing_thresholds[['dataset', 'scene']].value_counts())


# Fill missing 'thresholds' with -1
merged_df['thresholds'] = merged_df['thresholds'].fillna(-1)

# Merge the dataframes to create a submission file
submission_df = pd.merge(df_submission, merged_df[['dataset', 'scene', 'image', 'thresholds']],
                        on=['dataset', 'scene', 'image'], how='left')

# Update the prediction column in the submission file based on thresholds
submission_df['prediction'] = submission_df['thresholds']

# Ensure the prediction column has the correct data type
submission_df['prediction'] = submission_df['prediction'].astype(str)

# Display the first few rows of the submission dataframe
display(submission_df.head())

# Check for missing values in the prediction column
print(submission_df['prediction'].isnull().sum())

# Display the info of the submission dataframe
print(submission_df.info())


# Fill missing 'thresholds' with -1
merged_df['thresholds'] = merged_df['thresholds'].fillna(-1)

# Merge the dataframes to create a submission file. Use 'image_id' instead of 'image'
submission_df = pd.merge(df_submission, merged_df[['dataset', 'scene', 'thresholds']],
                        on=['dataset', 'scene'], how='left')

# Update the prediction column in the submission file based on thresholds
submission_df['prediction'] = submission_df['thresholds']

# Ensure the prediction column has the correct data type
submission_df['prediction'] = submission_df['prediction'].astype(str)

# Display the first few rows of the submission dataframe
display(submission_df.head())

# Check for missing values in the prediction column
print(submission_df['prediction'].isnull().sum())

# Display the info of the submission dataframe
print(submission_df.info())


# Fill missing 'thresholds' with -1
merged_df['thresholds'] = merged_df['thresholds'].fillna(-1)

# Try an inner join to see if we get better results.
submission_df = pd.merge(df_submission, merged_df[['dataset', 'scene', 'thresholds']], on=['dataset', 'scene'], how='inner')

# Update the prediction column
submission_df['prediction'] = submission_df['thresholds']

# Convert prediction to string type
submission_df['prediction'] = submission_df['prediction'].astype(str)

# Display first few rows
display(submission_df.head())

# Check for missing values
print(submission_df['prediction'].isnull().sum())

# Display info
print(submission_df.info())


# Fill missing 'thresholds' with -1 in merged_df
merged_df['thresholds'] = merged_df['thresholds'].fillna(-1)

# Merge with a left join
submission_df = pd.merge(df_submission, merged_df[['dataset', 'scene', 'thresholds']], on=['dataset', 'scene'], how='left')

# Handle missing 'thresholds' after the merge. Fill with -1
submission_df['thresholds'].fillna(-1, inplace=True)

# Update prediction column
submission_df['prediction'] = submission_df['thresholds']

# Ensure 'prediction' is of the same type as in original submission file.
submission_df['prediction'] = submission_df['prediction'].astype(str)

# Display first few rows
display(submission_df.head())

# Check for missing values in 'prediction' column
print(submission_df['prediction'].isnull().sum())

# Display info
print(submission_df.info())


import matplotlib.pyplot as plt

# Analyze the relationship between df_thresholds and df_labels based on 'dataset' and 'scene'
# Check if the 'dataset' and 'scene' columns in both DataFrames contain the same unique values.
# If not, the datasets might be mismatched.

# Visualize the distribution of threshold values
# Since the 'thresholds' column contains multiple values separated by semicolons, we need to preprocess it
# to extract individual threshold values.

def extract_thresholds(thresholds_str):
    return [float(x) for x in thresholds_str.split(';')]

# Create a list to store all the extracted thresholds
all_thresholds = []
for index, row in df_thresholds.iterrows():
    all_thresholds.extend(extract_thresholds(row['thresholds']))

plt.figure(figsize=(10, 6))
plt.hist(all_thresholds, bins=50, color='skyblue', edgecolor='black')
plt.xlabel('Threshold Value')
plt.ylabel('Frequency')
plt.title('Distribution of Threshold Values')
plt.show()


# Further investigation is needed to see if the thresholds are related to specific images or labels
# This would involve looking at the rotation_matrix and translation_vector information in df_labels
# and then trying to understand how these variables relate to the threshold value.



print(df_labels.columns)


# Install required libraries if not already installed
# Ensure pycolmap is included here
!pip install opencv-python numpy pandas pycolmap scikit-learn pathlib

import os
import cv2
import numpy as np
import pandas as pd
import pycolmap
from pathlib import Path
from sklearn.cluster import DBSCAN

# Configuration
# Update with your actual test dataset path
# You need to replace "YOUR_DATASET_FOLDER" with the actual name of your dataset folder
DATA_ROOT = "/kaggle/input/kaggle competitions download -c image-matching-challenge-2025"  # <---- REPLACE THIS WITH YOUR ACTUAL DATASET PATH
TEST_DIR = os.path.join(DATA_ROOT, "test")
OUTPUT_SUBMISSION = "submission.csv"
IMAGE_EXT = ".png"

# Helper functions
def parse_matrix(vector_str):
    """Convert a semicolon-separated string to a 3x3 matrix."""
    values = list(map(float, vector_str.split(";")))
    return np.array(values).reshape(3, 3)

def parse_vector(vector_str):
    """Convert a semicolon-separated string to a 3D vector."""
    return np.array(list(map(float, vector_str.split(";"))))

def load_images(dataset_path):
    """Load all images from a dataset folder."""
    # Check if the dataset_path exists before listing files
    if not os.path.isdir(dataset_path):
        print(f"Error: Dataset path not found: {dataset_path}")
        return {}

    image_files = list(Path(dataset_path).glob(f"*{IMAGE_EXT}"))
    images = {img.name: cv2.imread(str(img)) for img in image_files}
    return images

def extract_features(image):
    """Extract SIFT features from an image."""
    # Ensure image is not None before processing
    if image is None:
        return [], None
    sift = cv2.SIFT_create()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def cluster_images(dataset_path, images):
    """Cluster images into scenes using feature matching and DBSCAN."""
    features = {}
    for name, img in images.items():
        kp, desc = extract_features(img)
        if desc is not None: # Only store features if extraction was successful
            features[name] = (kp, desc)

    image_names = list(features.keys()) # Use keys from features with valid descriptors
    n_images = len(image_names)
    similarity_matrix = np.zeros((n_images, n_images))

    # Ensure there are enough images for clustering
    if n_images < 2:
        print(f"Not enough images with valid features for clustering in {dataset_path}. Skipping clustering.")
        return {f"scene_0": image_names} # Return a single scene with all images if clustering is not possible


    for i in range(n_images):
        for j in range(i + 1, n_images):
            # Ensure descriptors exist for both images before matching
            if image_names[i] in features and image_names[j] in features:
                kp1, desc1 = features[image_names[i]]
                kp2, desc2 = features[image_names[j]]
                # Ensure descriptors are not empty or None
                if desc1 is not None and desc2 is not None and len(desc1) > 0 and len(desc2) > 0:
                    flann = cv2.FlannBasedMatcher({"algorithm": 0, "trees": 5}, {"checks": 50})
                    try:
                        matches = flann.knnMatch(desc1, desc2, k=2)
                        # Filter out cases where knnMatch returns None or empty list
                        if matches is not None:
                             good_matches = [m for m, n in matches if len(n) > 0 and m.distance < 0.7 * n.distance] # Added check for n
                             similarity_matrix[i, j] = len(good_matches)
                             similarity_matrix[j, i] = len(good_matches)
                    except cv2.error as e:
                        print(f"Error during feature matching between {image_names[i]} and {image_names[j]}: {e}")
                        # Continue to the next pair if matching fails


    distance_matrix = 1 / (1 + similarity_matrix)
    # Handle cases where distance_matrix might contain inf or NaN due to zero similarity
    distance_matrix[np.isinf(distance_matrix)] = np.max(distance_matrix[~np.isinf(distance_matrix)]) if np.any(~np.isinf(distance_matrix)) else 1.0
    distance_matrix[np.isnan(distance_matrix)] = 1.0


    # Adjust eps or min_samples if clustering fails with current parameters
    try:
        clustering = DBSCAN(eps=0.5, min_samples=2, metric="precomputed").fit(distance_matrix)
        labels = clustering.labels_
    except Exception as e:
         print(f"Error during DBSCAN clustering in {dataset_path}: {e}")
         # Fallback: treat all images as a single scene
         return {f"scene_0": image_names}


    scene_clusters = {}
    for idx, label in enumerate(labels):
        if label != -1:  # Ignore outliers
            scene_name = f"scene_{label}"
            if scene_name not in scene_clusters:
                scene_clusters[scene_name] = []
            scene_clusters[scene_name].append(image_names[idx])
        else: # Handle outliers by putting them in their own scene
            scene_name = f"scene_outlier_{idx}"
            scene_clusters[scene_name] = [image_names[idx]]


    return scene_clusters

def run_sfm(dataset_path, image_names):
    """Run Structure-from-Motion using COLMAP to estimate camera poses."""
    colmap_db = os.path.join(dataset_path, "database.db")
    colmap_output = os.path.join(dataset_path, "sfm_output")

    # Clean up previous runs if necessary
    if os.path.exists(colmap_db):
        os.remove(colmap_db)
    if os.path.exists(colmap_output):
        import shutil
        shutil.rmtree(colmap_output)

    os.makedirs(colmap_output, exist_ok=True)

    try:
        # Filter image_names to only include files that exist in the dataset_path
        existing_image_names = [img_name for img_name in image_names if os.path.exists(os.path.join(dataset_path, img_name))]
        if not existing_image_names:
            print(f"No existing images found in {dataset_path} from the provided list. Skipping SFM.")
            return {}

        # Create an image list file for COLMAP
        image_list_path = os.path.join(dataset_path, "image_list.txt")
        with open(image_list_path, "w") as f:
            for img_name in existing_image_names:
                f.write(f"{img_name}\n")


        print(f"Running feature extraction for {dataset_path}")
        # Added checks for successful COLMAP execution
        try:
             pycolmap.feature_extraction(database_path=colmap_db, image_path=dataset_path, image_list=image_list_path)
        except Exception as e:
            print(f"COLMAP feature extraction failed for {dataset_path}: {e}")
            return {name: (np.eye(3).flatten(), np.zeros(3)) for name in image_names} # Return default poses on failure

        print(f"Running feature matching for {dataset_path}")
        try:
            pycolmap.feature_matching(database_path=colmap_db, image_list=image_list_path)
        except Exception as e:
             print(f"COLMAP feature matching failed for {dataset_path}: {e}")
             return {name: (np.eye(3).flatten(), np.zeros(3)) for name in image_names} # Return default poses on failure


        print(f"Running incremental SFM for {dataset_path}")
        try:
            reconstruction = pycolmap.incremental_sfm(
                database_path=colmap_db,
                image_path=dataset_path,
                output_path=colmap_output,
                image_list=image_list_path
            )
        except Exception as e:
             print(f"COLMAP incremental SFM failed for {dataset_path}: {e}")
             return {name: (np.eye(3).flatten(), np.zeros(3)) for name in image_names} # Return default poses on failure


        poses = {}
        # Check if reconstruction is valid before accessing images
        if reconstruction:
            for image_name in image_names:
                if image_name in reconstruction.images:
                    image = reconstruction.images[image_name]
                    # Ensure rotation matrix is a numpy array before flattening
                    rotation_matrix = np.asarray(image.rotation_matrix()).flatten()  # Row-major
                    translation_vector = np.asarray(image.translation_vector())
                    poses[image_name] = (rotation_matrix, translation_vector)
                else:
                    # Use default poses if image is not in reconstruction
                    poses[image_name] = (
                        np.eye(3).flatten(),  # Default rotation
                        np.zeros(3)  # Default translation
                    )
        else:
            # Return default poses if reconstruction is None
             print(f"COLMAP reconstruction failed for {dataset_path}. Returning default poses.")
             poses = {name: (np.eye(3).flatten(), np.zeros(3)) for name in image_names}


    except Exception as e:
        print(f"An error occurred during COLMAP processing for {dataset_path}: {e}")
        # Return default poses in case of any unexpected error
        poses = {name: (np.eye(3).flatten(), np.zeros(3)) for name in image_names}

    return poses


def generate_submission():
    """Generate submission.csv for the test set."""
    submission_data = []

    # Check if the TEST_DIR exists before listing directories
    if not os.path.isdir(TEST_DIR):
        print(f"Error: Test directory not found: {TEST_DIR}")
        # Optionally, you could exit or raise an error here if the test directory is essential
        return


    # Iterate over test datasets
    for dataset_name in os.listdir(TEST_DIR):
        dataset_path = os.path.join(TEST_DIR, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        print(f"Processing dataset: {dataset_name}")

        # Load images
        images = load_images(dataset_path)
        if not images:
            print(f"No images found in {dataset_path}. Skipping.")
            continue

        # Cluster images into scenes
        scene_clusters = cluster_images(dataset_path, images)
        print(f"Found {len(scene_clusters)} scenes in {dataset_name}")

        # Process each scene
        for scene_name, image_names in scene_clusters.items():
            print(f"Processing scene: {scene_name} with {len(image_names)} images")
            # Ensure there are images in the scene before running SFM
            if not image_names:
                 print(f"No images in scene {scene_name}. Skipping SFM for this scene.")
                 continue

            poses = run_sfm(dataset_path, image_names)

            for image_name in image_names:
                rotation_matrix, translation_vector = poses.get(
                    image_name,
                    (np.eye(3).flatten(), np.zeros(3)) # Default poses if not found
                )

                submission_data.append({
                    "image_id": f"{dataset_name}/{image_name}",
                    "dataset": dataset_name,
                    "scene": scene_name,
                    "image": image_name,
                    "rotation_matrix": ";".join(map(str, rotation_matrix)),
                    "translation_vector": ";".join(map(str, translation_vector))
                })

    # Create and save submission DataFrame
    if submission_data: # Only create DataFrame if there is data
        submission_df = pd.DataFrame(submission_data)
        submission_df.to_csv(OUTPUT_SUBMISSION, index=False)
        print(f"Submission file saved to {OUTPUT_SUBMISSION}")
    else:
        print("No data generated for submission.")


if __name__ == "__main__":
    # This block will only run if the script is executed directly.
    # In a notebook, each cell runs independently.
    # The installation command should ideally be in a cell before the import.
    # We will keep it here for completeness, but the fix above is more direct.
    pass

generate_submission()

