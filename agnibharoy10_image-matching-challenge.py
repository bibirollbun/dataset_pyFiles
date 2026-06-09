# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN  # Used for scene grouping based on feature similarity
from collections import defaultdict
from sklearn.metrics import pairwise_distances
from scipy.spatial.transform import Rotation as R  # For pose estimation


# Load the data
train_path = "../input/image-matching-challenge-2025/train"
train_labels = pd.read_csv('/kaggle/input/image-matching-challenge-2025/train_labels.csv')
train_thresholds = pd.read_csv("/kaggle/input/image-matching-challenge-2025/train_thresholds.csv")


def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()  # can be switch to SuperPoint or DELF for better results
    kp, des = sift.detectAndCompute(image, None)
    return kp, des


# Loading images for a specific scene
scene_name = "fountain"
scene_images = train_labels[train_labels["scene"] == scene_name]["image"].values

# Feature extraction and clustering images by similarity
features = []
image_paths = []

for img_name in scene_images:
    img_path = os.path.join(train_path, train_labels[train_labels["scene"] == scene_name]["dataset"].values[0], img_name)
    kp, des = extract_features(img_path)
    features.append(des)
    image_paths.append(img_path)


# Matching images using pairwise distances between descriptors (can be optimize with FLANN or nearest neighbor search)
dist_matrix = pairwise_distances([np.mean(f, axis=0) for f in features], metric="cosine")

# Grouping images into scenes using DBSCAN
clustering = DBSCAN(eps=0.5, min_samples=3, metric="precomputed").fit(dist_matrix)


# Visualizing clusters
scene_groups = defaultdict(list)
for idx, label in enumerate(clustering.labels_):
    scene_groups[label].append(image_paths[idx])


# Seeing clustered images from the same scene
fig, axes = plt.subplots(1, len(scene_groups[0]), figsize=(15, 10))
for i, img_path in enumerate(scene_groups[0]):  # Display first group
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    axes[i].imshow(img, cmap='gray')
    axes[i].axis('off')
plt.show()


# For camera pose estimation using essential matrix(RANSAC)
def estimate_camera_pose(kp1, kp2, des1, des2):
    # Matcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Essential matrix and pose recovery
    E, mask = cv2.findEssentialMat(src_pts, dst_pts, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R_mat, t_vec, mask = cv2.recoverPose(E, src_pts, dst_pts)
    
    return R_mat, t_vec


# matching two images from the same scene and estimate pose
img1_path = scene_groups[0][0]  # Choose first image in the group
img2_path = scene_groups[0][1]  # Choose another image in the group

# Extracting keypoints and descriptors
kp1, des1 = extract_features(img1_path)
kp2, des2 = extract_features(img2_path)

# Estimating camera pose between two images
R_mat, t_vec = estimate_camera_pose(kp1, kp2, des1, des2)

# Visualizing rotation and translation
print(f"Rotation Matrix:\n{R_mat}")
print(f"Translation Vector:\n{t_vec}")


sample_submission = pd.read_csv('/kaggle/input/image-matching-challenge-2025/sample_submission.csv')
sample_submission["rotation_matrix"] = "1;0;0;0;1;0;0;0;1"  # Placeholder
sample_submission["translation_vector"] = "0;0;0"  # Placeholder


sample_submission.to_csv("submission.csv", index=False)
print("Submission file created!")




