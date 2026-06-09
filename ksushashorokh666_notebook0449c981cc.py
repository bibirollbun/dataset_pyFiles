#LOADING THE DATASET

import os #For file and directory operations (ex: checking dataset structure)

# List all files in dataset directory
dataset_path = "/kaggle/input/"
os.listdir(dataset_path)
#Just to ensure that the dataset is already there! 
#(Kaggle usually mounts the competition dataset automatically in /kaggle/input/)


#EXPLORING THE DATASET STRUCTURE

dataset_path = "/kaggle/input/image-matching-challenge-2025"

# Check files inside (list the available folders)
os.listdir(dataset_path)




# Importing necessary libraries 

import numpy as np  # For numerical operations (arrays, math, etc.)
import pandas as pd  # For handling CSV files (reading, writing, manipulating tables)
import matplotlib.pyplot as plt  # For visualizing images
from PIL import Image  # To open and inspect images


# Loading train_labels.csv file
train_path = "../input/image-matching-challenge-2025/train" 
train_labels = pd.read_csv('/kaggle/input/image-matching-challenge-2025/train_labels.csv')

# Displaying first few rows of it
train_labels.head()


train_labels.info()


num_scenes = train_labels["scene"].nunique()
num_scenes


scene_counts = train_labels["scene"].value_counts()
scene_counts


num_datasets = train_labels["dataset"].nunique()
num_datasets


dataset_counts = train_labels["dataset"].value_counts()
dataset_counts


import seaborn as sns

# Counting images per dataset
dataset_counts = train_labels["dataset"].value_counts()

# Plotting the bar chart
plt.figure(figsize=(12, 6))  # Set figure size
sns.barplot(x=dataset_counts.index, y=dataset_counts.values, palette="viridis")  # Use seaborn for better styling

# Adding labels and title
plt.xlabel("Dataset", fontsize=14)
plt.ylabel("Number of Images", fontsize=14)
plt.title("Number of Images per Dataset", fontsize=16)
plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability
plt.grid(axis="y", linestyle="--", alpha=0.7)  # Add a light grid for clarity

# Show the plot
plt.show()


train_thresholds_path = "../input/image-matching-challenge-2025/train_thresholds.csv"
train_thresholds = pd.read_csv(train_thresholds_path)
train_thresholds.head() 


train_thresholds.describe()


import matplotlib.pyplot as plt
import numpy as np

# Step 1: Convert the "thresholds" column into a list of lists (split by ";")
threshold_lists = train_thresholds["thresholds"].apply(lambda x: list(map(float, x.split(";"))))

# Step 2: Flatten the list (convert list of lists into a single list)
all_thresholds = np.concatenate(threshold_lists.values)

# Step 3: Plot the histogram
plt.figure(figsize=(8, 5))
plt.hist(all_thresholds, bins=30, edgecolor="black")
plt.xlabel("Score Threshold")
plt.ylabel("Count")
plt.title("Distribution of Similarity Thresholds")
plt.show()


import cv2  # OpenCV library for image processing

# Pick a scene from the dataset
scene_name = "fountain"  
scene_images = train_labels[train_labels["scene"] == scene_name]["image"].values[:2]  # Picked two images

# Load and display images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for i, img_name in enumerate(scene_images):
    img_path = os.path.join(train_path, train_labels[train_labels["scene"] == scene_name]["dataset"].values[0], img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Loaded image in grayscale (easier for feature matching)
    
    axes[i].imshow(img, cmap="gray")
    axes[i].set_title(f"Image: {img_name}")
    axes[i].axis("off")

plt.show()



img1 = cv2.imread(os.path.join(train_path, train_labels[train_labels["scene"] == scene_name]["dataset"].values[0], scene_images[0]), cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(os.path.join(train_path, train_labels[train_labels["scene"] == scene_name]["dataset"].values[0], scene_images[1]), cv2.IMREAD_GRAYSCALE)

# Initialize ORB detector
orb = cv2.ORB_create()

# Detect keypoints and descriptors
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Initialize Brute-Force Matcher and match descriptors
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches by distance (lower distance = better match)
matches = sorted(matches, key=lambda x: x.distance)

# Draw matches
match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(12, 6))
plt.imshow(match_img)
plt.title("Feature Matching using ORB")
plt.axis("off")
plt.show()


sample_submission = pd.read_csv('/kaggle/input/image-matching-challenge-2025/sample_submission.csv')
#sample_submission.head()
#checking column named to have in dummy submission

# Create dummy values
sample_submission["rotation_matrix"] = "1;0;0;0;1;0;0;0;1"  # Identity matrix as a placeholder
sample_submission["translation_vector"] = "0;0;0"  # Zero translation

# Save the dummy submission file
sample_submission.to_csv("submission.csv", index=False)

print("Submission file created successfully!")

