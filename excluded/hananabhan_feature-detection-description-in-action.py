import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from PIL import Image



train_labels = pd.read_csv("/kaggle/input/image-matching-challenge-2025/train_labels.csv")
train_labels.head()


train_labels.info()


scene_number =train_labels["scene"].unique()
scecne_count=train_labels["scene"].value_counts()

scene_number,scecne_count


dataset_counts = train_labels["dataset"].value_counts()
num_datasets = train_labels["dataset"].nunique()
num_datasets,dataset_counts


train_thresholds_path = "../input/image-matching-challenge-2025/train_thresholds.csv"
train_thresholds = pd.read_csv(train_thresholds_path)
train_thresholds.head()


train_thresholds.describe()



threshold_lists = train_thresholds["thresholds"].apply(lambda x: list(map(float, x.split(";"))))
all_thresholds = np.concatenate(threshold_lists.values)
all_thresholds


plt.figure(figsize=(8, 5))
plt.hist(all_thresholds, bins=30, edgecolor="black")
plt.xlabel("Score Threshold")
plt.ylabel("Count")
plt.title("Distribution of Similarity Thresholds")
plt.show()


img1 = cv2.imread("/kaggle/input/image-matching-challenge-2025/train/imc2024_dioscuri_baalshamin/baalshamin_19577300988_4e4ff423a7_o.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("/kaggle/input/image-matching-challenge-2025/train/imc2024_dioscuri_baalshamin/baalshamin_194d.png", cv2.IMREAD_GRAYSCALE)

plt.subplot(121), plt.imshow(img1, cmap='gray'), plt.title("Image 1")
plt.subplot(122), plt.imshow(img2, cmap='gray'), plt.title("Image 2")
plt.show()



orb = cv2.ORB_create()

# Detect keypoints
kp1_ORB = orb.detect(img1, None)
kp2_ORB = orb.detect(img2, None)

img_kp1 = cv2.drawKeypoints(img1, kp1_ORB, None, color=(0,255,0))
plt.imshow(img_kp1)
plt.title("Keypoints Scene 1")
plt.show()



# SIFT Detector
sift = cv2.SIFT_create()

# Detect keypoints + descriptors directly
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Draw keypoints
img_kp1 = cv2.drawKeypoints(img1, kp1, None, color=(255,0,0))
plt.imshow(img_kp1)
plt.title("SIFT Keypoints - Image 1")
plt.show()


# calculate key points like (size and rotation) and Descriptors, which is numpy array, each row describes 1 kp
kp1_ORB, des1_ORB = orb.compute(img1, kp1_ORB)
kp2_ORB, des2_ORB = orb.compute(img2, kp2_ORB)



# Brute Force Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(des1_ORB, des2_ORB)

# Sort by distance (good → bad)
matches = sorted(matches, key=lambda x:x.distance)

# Draw top matches
img_matches = cv2.drawMatches(img1, kp1_ORB, img2, kp2_ORB, matches[:20], None, flags=2)
plt.imshow(img_matches)
plt.title("Feature Matching")
plt.show()



# Brute Force Matcher for SIFT
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors
matches = bf.match(des1, des2)

# Sort by distance (good → bad)
matches = sorted(matches, key=lambda x: x.distance)

# Draw top matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)
plt.imshow(img_matches)
plt.title("SIFT Feature Matching")
plt.show()



# FLANN parameters
FLANN_INDEX_KDTREE = 1  
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # higher = more accurate but slower

flann = cv2.FlannBasedMatcher(index_params, search_params)

# KNN Match (k=2 → best 2 matches for each descriptor)
matches = flann.knnMatch(des1, des2, k=2)

# Lowe’s ratio test to filter good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:  # 0.7 is the common threshold
        good_matches.append(m)

# Draw matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)
plt.imshow(img_matches)
plt.title("SIFT + FLANN Matching with Ratio Test")
plt.show()



des1_ORB = np.float32(des1_ORB)
des2_ORB = np.float32(des2_ORB)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# KNN Match (k=2 → best 2 matches for each descriptor)
matches = flann.knnMatch(des1_ORB, des2_ORB, k=2)

# Lowe’s ratio test to filter good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:  # 0.7 is the common threshold
        good_matches.append(m)

# Draw matches
img_matches = cv2.drawMatches(img1, kp1_ORB, img2, kp2_ORB, good_matches, None, flags=2)
plt.imshow(img_matches)
plt.title("ORB + FLANN Matching with Ratio Test")
plt.show()


