!pip install -q mediapy


%cd /kaggle/working/
!rm -rf /kaggle/working/Hierarchical-Localization
!git clone --quiet --recursive https://github.com/cvg/Hierarchical-Localization/
%cd /kaggle/working/Hierarchical-Localization
!pip install -e .

from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d

%cd /kaggle/working/


# Standard Libraries
import os
import random
from pathlib import Path

# Data Handling
import pandas as pd
import numpy as np

# Image Processing
import cv2
from PIL import Image

# Visualization
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib import cm

# Media & Display
import mediapy

# 3D Reconstruction / SfM
import pycolmap


# Loading train_path file
train_path = "/kaggle/input/image-matching-challenge-2025/train" 


train_labels = pd.read_csv("/kaggle/input/image-matching-challenge-2025/train_labels.csv")
train_labels


train_labels.head()


train_labels.info


train_labels.shape


train_labels.describe().round(2).T


train_labels.groupby("dataset")["scene"].nunique()


# Assuming train_labels is already loaded as a DataFrame
dataset_counts = train_labels["dataset"].value_counts()

# Generate rainbow colors based on the number of datasets
colors = cm.rainbow(np.linspace(0, 1, len(dataset_counts)))

# Plot with rainbow colors
plt.figure(figsize=(12, 7))
dataset_counts.plot(kind='bar', color=colors, edgecolor='black')

# Plot formatting
plt.title('Distribution of Images Across Datasets', fontsize=16, fontweight='bold')
plt.xlabel('Dataset', fontsize=14)
plt.ylabel('Number of Images', fontsize=14)
plt.xticks(rotation=60, ha='right', fontsize=12)  
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display plot
plt.tight_layout()
plt.show()


def plot_random_images(scene_name, base_path="/kaggle/input/image-matching-challenge-2025/train"):
    scene_path = os.path.join(base_path, scene_name)
    image_filenames = [f for f in os.listdir(scene_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    random_images = random.sample(image_filenames, min(5, len(image_filenames)))
    fig, axes = plt.subplots(1, len(random_images), figsize=(15, 5))
    if len(random_images) == 1:
        axes = [axes]

    for ax, img_filename in zip(axes, random_images):
        img_path = os.path.join(scene_path, img_filename)
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(img_filename, fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


plot_random_images("pt_sacrecoeur_trevi_tajmahal")


plot_random_images("amy_gardens")


plot_random_images("imc2023_heritage")


plot_random_images("fbk_vineyard")


plot_random_images("pt_piazzasanmarco_grandplace")


plot_random_images("imc2024_lizard_pond")


train_thresholds_path = "../input/image-matching-challenge-2025/train_thresholds.csv"
train_thresholds = pd.read_csv(train_thresholds_path)
train_thresholds.head() 


# Describe thresholds data
train_thresholds.describe().round(2).T


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

# Step 1: Convert the "thresholds" column into a list of lists (split by ";")
threshold_lists = train_thresholds["thresholds"].apply(lambda x: list(map(float, x.split(";"))))

# Step 2: Flatten the list
all_thresholds = np.concatenate(threshold_lists.values)

# Step 3: Create histogram data
counts, bins = np.histogram(all_thresholds, bins=30)

# Normalize the bin centers for color mapping
norm = Normalize(vmin=min(bins), vmax=max(bins))
colors = cm.rainbow(norm((bins[:-1] + bins[1:]) / 2))

# Step 4: Plot the colored histogram manually using bar
plt.figure(figsize=(10, 6))
for i in range(len(bins) - 1):
    plt.bar(bins[i], counts[i], width=bins[i+1]-bins[i], color=colors[i], edgecolor='black', align='edge')

# Step 5: Formatting
plt.xlabel("Score Threshold", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Distribution of Similarity Thresholds", fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



# Scene and image selection
scene_name = "fountain"
scene_df = train_labels[train_labels["scene"] == scene_name]
dataset_name = scene_df["dataset"].values[0]
scene_images = scene_df["image"].sample(n=2, random_state=42).values  # Random 2 images

# Set image directory
scene_path = os.path.join(train_path, dataset_name)

# Initialize the figure
fig, axes = plt.subplots(1, len(scene_images), figsize=(12, 6))

for i, img_name in enumerate(scene_images):
    img_path = os.path.join(scene_path, img_name)

    if not os.path.exists(img_path):
        print(f"Warning: Image {img_path} not found.")
        continue

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error reading image: {img_path}")
        continue

    axes[i].imshow(img, cmap="gray")
    axes[i].set_title(f"{img_name}", fontsize=12)
    axes[i].axis("off")

# Styling
plt.suptitle(f"Scene: {scene_name.capitalize()} (Random 2 Images)", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()


# Parameters
scene_name = "fountain"
num_matches_to_display = 50
random_state = 42

# Get image filenames and dataset
scene_df = train_labels[train_labels["scene"] == scene_name]
dataset_name = scene_df["dataset"].values[0]
scene_images = scene_df["image"].sample(n=2, random_state=random_state).values

# Build image paths
img1_path = os.path.join(train_path, dataset_name, scene_images[0])
img2_path = os.path.join(train_path, dataset_name, scene_images[1])

# Load images in grayscale
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

# Validate loading
if img1 is None or img2 is None:
    raise FileNotFoundError("One or both images could not be loaded. Check paths.")

# ORB Detector
orb = cv2.ORB_create(nfeatures=1000)

# Detect keypoints and descriptors
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Brute Force Matcher (Hamming for ORB)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort by distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw top N matches
matched_img = cv2.drawMatches(
    img1, kp1, img2, kp2, matches[:num_matches_to_display], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# Plot
plt.figure(figsize=(14, 7))
plt.imshow(matched_img, cmap='gray')
plt.title(f"ORB Feature Matching (Top {num_matches_to_display} Matches)\nScene: {scene_name.capitalize()}")
plt.axis("off")
plt.tight_layout()
plt.show()


# Load sample submission template
submission_path = "/kaggle/input/image-matching-challenge-2025/sample_submission.csv"
sample_submission = pd.read_csv(submission_path)

# Define dummy values
identity_matrix = "1;0;0;0;1;0;0;0;1"   # 3x3 identity matrix (row-major)
zero_vector = "0;0;0"                  # zero translation vector

# Fill the columns with dummy values
sample_submission["rotation_matrix"] = identity_matrix
sample_submission["translation_vector"] = zero_vector

# Save the submission file
output_file = "submission.csv"
sample_submission.to_csv(output_file, index=False)

print(f"Dummy submission file saved as: {output_file}")
print(f"Entries: {len(sample_submission)} | Format: Identity rotation, zero translation")




