!pip install ultralytics
#!pip install torchxrayvision
!pip install pydicom Pillow
!pip install scikit-image
!pip install tqdm --upgrade
!pip install scikit-learn
!pip install -q ensemble-boxes


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# âœ… Standard libraries
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import os
import gc
import ast
import zipfile
import shutil
import random
import pprint
import warnings
from glob import glob
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# âœ… Data handling
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import numpy as np
import pandas as pd
import yaml
from tqdm.autonotebook import tqdm

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# âœ… Image handling & visualization
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from IPython.display import display, FileLink
import pydicom

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# âœ… Scientific image processing
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import skimage.io
import skimage.transform
import albumentations as A

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# âœ… Machine learning & utilities
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.manifold import TSNE

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# âœ… Deep learning
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
#import torchxrayvision as xrv  # For X-ray-specific processing

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# âœ… Object Detection (YOLO & WBF)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion


# âœ… Load dataset
label_data_file = "/kaggle/input/vinbigdata-1024-image-dataset/vinbigdata/train.csv"
train_df = pd.read_csv(label_data_file)

# âœ… Add image_path column
train_df['image_path'] = '/kaggle/input/vinbigdata-1024-image-dataset/vinbigdata/train/' + train_df.image_id + '.png'

# âœ… Remove class 14 (No Finding) and class 2 (Calcification) completely
train_df = train_df[~train_df.class_id.isin([14, 2])].reset_index(drop=True)

# âœ… Print remaining images to confirm
print(f"âœ… Number of images remaining: {train_df['image_id'].nunique()}")

# âœ… Convert VinBigData bbox format to YOLO format
train_df['x_mid'] = (train_df['x_min'] + train_df['x_max']) / (2 * train_df['width'])
train_df['y_mid'] = (train_df['y_min'] + train_df['y_max']) / (2 * train_df['height'])
train_df['w'] = (train_df['x_max'] - train_df['x_min']) / train_df['width']
train_df['h'] = (train_df['y_max'] - train_df['y_min']) / train_df['height']

train_df['source_dataset'] = 'vinbig'

train_df.head()


# âœ… Load the new NIH dataset
new_nih_file = "/kaggle/input/nih-chest-xray-dataset-bbox-for-vinbigdata/nih.csv"
new_nih_df = pd.read_csv(new_nih_file)

# âœ… Add image path for new NIH dataset
new_nih_df['image_path'] = '/kaggle/input/nih-chest-xray-dataset-bbox-for-vinbigdata/nih/' + new_nih_df['image_id'] + '.png'

# âœ… Remove rows with unmapped class names (NaN)
new_nih_df = new_nih_df.dropna(subset=['class_name'])

# âœ… Define the class_name_to_id mapping
class_name_to_id = {
    "Aortic enlargement": 0,
    "Cardiomegaly": 2,  
    "Consolidation": 3,
    "ILD": 4,
    "Infiltration": 5,
    "Lung Opacity": 6,
    "Nodule/Mass": 7,
    "Other lesion": 8,
    "Pleural effusion": 9,
    "Pleural thickening": 10,
    "Pneumothorax": 11,
    "Pulmonary fibrosis": 12,
    "Atelectasis": 1
}

# âœ… Assign class_id based on class_name
new_nih_df["class_id"] = new_nih_df["class_name"].map(class_name_to_id)

# âœ… Calculate actual image width and height
image_widths = []
image_heights = []

for path in new_nih_df["image_path"]:
    image = cv2.imread(path)
    if image is not None:
        height, width = image.shape[:2]
    else:
        height, width = -1, -1  # Handle missing/corrupted image case
    image_widths.append(width)
    image_heights.append(height)

# âœ… Store as 'width' and 'height' (actual image dimensions, not bbox)
new_nih_df["width"] = image_widths
new_nih_df["height"] = image_heights

# âœ… Convert bbox to YOLO format (1-step calculation + normalization)
new_nih_df['x_mid'] = (new_nih_df['x_min'] + new_nih_df['x_max']) / (2 * new_nih_df['width'])
new_nih_df['y_mid'] = (new_nih_df['y_min'] + new_nih_df['y_max']) / (2 * new_nih_df['height'])
new_nih_df['w'] = (new_nih_df['x_max'] - new_nih_df['x_min']) / new_nih_df['width']
new_nih_df['h'] = (new_nih_df['y_max'] - new_nih_df['y_min']) / new_nih_df['height']

new_nih_df['source_dataset'] = 'nih_for_vin'

# âœ… Select relevant columns (now width/height = image size)
new_nih_df = new_nih_df[[
    'image_id', 'class_name', 'class_id', 'rad_id',
    'x_mid', 'y_mid', 'w', 'h',
    'x_min', 'y_min', 'x_max', 'y_max',
    'width', 'height', 'image_path', 'source_dataset'
]]

# âœ… Filter for only the classes you want to add: Atelectasis, Pneumothorax, and Nodule/Mass
relevant_classes = ["Atelectasis", "Pneumothorax", "Nodule/Mass"]
filtered_nih_df = new_nih_df[new_nih_df["class_name"].isin(relevant_classes)]

# âœ… Merge the filtered DataFrame with the existing train_df
train_df = pd.concat([train_df, filtered_nih_df], ignore_index=True)

# âœ… Check for NaN class IDs after merging
print(f"Number of NaN class IDs after merge: {train_df['class_id'].isna().sum()}")  # Should be 0

# âœ… Print final dataset details
print(f"âœ… Number of unique images after final merge: {train_df['image_id'].nunique()}")
train_df.head()


def visualize_bboxes(image_paths, bboxes_list, labels_list, image_sizes, num_images=5):
    num_images = min(num_images, len(image_paths))  
    fig, axes = plt.subplots(1, num_images, figsize=(50, 30))

    if num_images == 1:
        axes = [axes]

    for idx in range(num_images):
        image_path, bboxes, labels, (orig_width, orig_height) = (
            image_paths[idx], bboxes_list[idx], labels_list[idx], image_sizes[idx])

        image = cv2.imread(image_path)
        if image is None:
            print(f"âš ï¸� Error: Could not read {image_path}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        height, width, _ = image.shape  

        # âœ… Rescale bounding boxes
        for i in range(len(bboxes)):
            x_min, y_min, x_max, y_max = bboxes[i]
            bboxes[i] = [
                int((x_min / orig_width) * width),
                int((y_min / orig_height) * height),
                int((x_max / orig_width) * width),
                int((y_max / orig_height) * height),
            ]
            cv2.rectangle(image, (bboxes[i][0], bboxes[i][1]), (bboxes[i][2], bboxes[i][3]), (0, 255, 0), 2)
            cv2.putText(image, labels[i], (bboxes[i][0], bboxes[i][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        axes[idx].imshow(image)
        axes[idx].axis("off")

    plt.show()
    return bboxes_list  # Return updated bounding boxes


# âœ… Select a random sample of unique images
num_samples = 5
sampled_images = train_df["image_id"].drop_duplicates().sample(n=min(num_samples, train_df["image_id"].nunique())).tolist()

# âœ… Prepare lists for visualization
image_paths, bboxes_list, labels_list, image_sizes = [], [], [], []

for image_id in sampled_images:
    sample_df = train_df[train_df["image_id"] == image_id]
    image_path = sample_df["image_path"].iloc[0]
    print(f"ğŸ”� Visualizing image: {image_path}")  # <-- âœ… Print the image path here
    image_paths.append(image_path)
    bboxes_list.append(sample_df[["x_min", "y_min", "x_max", "y_max"]].values.tolist())
    labels_list.append(sample_df["class_name"].tolist())
    image_sizes.append(sample_df[["width", "height"]].iloc[0].tolist())

# âœ… Visualize & Update Bounding Boxes
bboxes_list_updated = visualize_bboxes(image_paths, bboxes_list, labels_list, image_sizes, num_images=num_samples)

# âœ… Update train_df with new bounding boxes
for i, image_id in enumerate(sampled_images):
    sample_df = train_df[train_df["image_id"] == image_id].copy()
    updated_bboxes = bboxes_list_updated[i]
    
    for j, bbox in enumerate(updated_bboxes):
        train_df.loc[sample_df.index[j], ['x_min', 'y_min', 'x_max', 'y_max']] = bbox

print("âœ… Bounding boxes updated in train_df!")


# âœ… Load NIH dataset
nih_bbox_file = "/kaggle/input/nih-chest-x-rays-bbox-version/BBox_List_2017.csv"
nih_df = pd.read_csv(nih_bbox_file)

# âœ… Add image path
nih_df['image_path'] = '/kaggle/input/nih-chest-x-rays-bbox-version/bbox_img/' + nih_df['Image Index']

# âœ… Map class names
nih_class_mapping = {
    "Infiltrate": "Infiltration",
    "Atelectasis": "Atelectasis",
    "Pneumonia": "Pneumonia",
    "Cardiomegaly": "Cardiomegaly",
    "Effusion": "Pleural effusion",
    "Pneumothorax": "Pneumothorax",
    "Mass": "Nodule/Mass",
    "Nodule": "Nodule/Mass"
}
nih_df['class_name'] = nih_df['Finding Label'].map(nih_class_mapping)
nih_df = nih_df.dropna(subset=['class_name'])

# âœ… Rename bbox columns
nih_df = nih_df.rename(columns={
    "Image Index": "image_id",
    "Bbox [x": "x_min",
    "y": "y_min",
    "w": "w",
    "h]": "h"
})

# âœ… Compute x_max, y_max
nih_df['x_max'] = nih_df['x_min'] + nih_df['w']
nih_df['y_max'] = nih_df['y_min'] + nih_df['h']

# âœ… Assume fixed image size if actual dimensions are not available (e.g., 1024x1024)
nih_df['width'] = 1024
nih_df['height'] = 1024

# âœ… Compute YOLO format in one step
nih_df['x_mid'] = (nih_df['x_min'] + nih_df['x_max']) / (2 * nih_df['width'])
nih_df['y_mid'] = (nih_df['y_min'] + nih_df['y_max']) / (2 * nih_df['height'])
nih_df['w'] = (nih_df['x_max'] - nih_df['x_min']) / nih_df['width']
nih_df['h'] = (nih_df['y_max'] - nih_df['y_min']) / nih_df['height']

nih_df['source_dataset'] = 'nih'

# âœ… Final column selection
nih_df = nih_df[['image_id', 'class_name', 'x_mid', 'y_mid', 'w', 'h', 'x_min', 'y_min', 'x_max', 'y_max', 'width', 'height', 'image_path','source_dataset']]

# âœ… Filter for only the classes you want to add: Atelectasis, Pneumothorax, and Nodule/Mass
relevant_classes = ["Atelectasis", "Pneumothorax", "Nodule/Mass"]
filtered_nih_df = nih_df[nih_df["class_name"].isin(relevant_classes)]

# âœ… Merge the filtered DataFrame with the existing train_df
train_df = pd.concat([train_df, filtered_nih_df], ignore_index=True)

# âœ… Assign class_id to the merged dataset
class_name_to_id = {
    "Aortic enlargement": 0,
    "Cardiomegaly": 2,  
    "Consolidation": 3,
    "ILD": 4,
    "Infiltration": 5,
    "Lung Opacity": 6,
    "Nodule/Mass": 7,
    "Other lesion": 8,
    "Pleural effusion": 9,
    "Pleural thickening": 10,
    "Pneumothorax": 11,
    "Pulmonary fibrosis": 12,
    "Atelectasis": 1,
    "Pneumonia": 13 
}
train_df["class_id"] = train_df["class_name"].map(class_name_to_id)

# âœ… Check for NaN class IDs after merging
print(f"Number of NaN class IDs: {train_df['class_id'].isna().sum()}")  # Should be 0

# âœ… Print final dataset details
print(f"âœ… Number of unique images after merging: {train_df['image_id'].nunique()}")
train_df.head()


import os
import cv2
import pandas as pd

# Define the directories where the images and their corresponding label files are stored
images_dir = "/kaggle/input/chestxrayabnormalities/train/images"  # Correct directory
labels_dir = "/kaggle/input/chestxrayabnormalities/train/labels"  # Correct directory

# Class mapping (ID to class name) based on the ChestX-ray dataset
class_mapping = {
    0: "Aortic_enlargement",
    1: "Atelectasis",
    2: "Calcification",
    3: "Cardiomegaly",
    4: "Consolidation",
    5: "ILD",
    6: "Infiltration",
    7: "Lung_Opacity",
    8: "Nodule-Mass",  # This is mapped to 'Nodule/Mass'
    9: "Other_lesion",
    10: "Pleural_effusion",
    11: "Pleural_thickening",
    12: "Pneumothorax",
    13: "Pulmonary_fibrosis"
}

# Define the relevant classes you want to keep
relevant_classes = ["Atelectasis", "Pneumothorax", "Nodule-Mass"]

# Create a list to hold the processed data
processed_data = []

# List all image files in the images directory
image_files = os.listdir(images_dir)

# Loop through all the label files in the directory
for label_file in os.listdir(labels_dir):
    if label_file.endswith('.txt'):
        # Extract the image ID (without extension)
        image_id = label_file.split('.')[0]
        
        # Search for the image file in the images directory that matches the image_id (contains it as a substring)
        matching_image_files = [img for img in image_files if image_id in img]
        
        if matching_image_files:
            # If there is a match, take the first one (in case there are multiple matches)
            image_path = os.path.join(images_dir, matching_image_files[0])
        else:
            print(f"â�Œ No matching image found for {image_id}")
            continue
        
        # Read the image to get its dimensions (height and width)
        image = cv2.imread(image_path)
        if image is not None:
            image_height, image_width = image.shape[:2]
        else:
            image_height, image_width = -1, -1  # Handle missing or corrupted image case
        
        # Read the corresponding label file
        label_file_path = os.path.join(labels_dir, label_file)
        with open(label_file_path, 'r') as f:
            lines = f.readlines()
        
        # Process each line in the label file
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])  # Original class ID

            # Only process the class_ids that are in the relevant_classes set
            if class_mapping.get(class_id) in relevant_classes:
                class_name = class_mapping[class_id]
                
                # YOLO format is already in normalized form
                x_mid = float(parts[1])
                y_mid = float(parts[2])
                bbox_width = float(parts[3])
                bbox_height = float(parts[4])

                # Append the data for this image and label (including class_id and class_name)
                processed_data.append([image_id, class_name, class_id, x_mid, y_mid, bbox_width, bbox_height])

# Convert the processed data into a DataFrame
processed_df = pd.DataFrame(processed_data, columns=['image_id', 'class_name', 'class_id', 'x_mid', 'y_mid', 'w', 'h'])

# Construct the image path correctly by mapping to the correct image file in the directory
processed_df['image_path'] = processed_df['image_id'].apply(
    lambda x: os.path.join(images_dir, next((img for img in image_files if x in img), None))
)

# Add the source dataset name (in this case, 'chestxrayabnormalities')
processed_df['source_dataset'] = 'chestxrayabnormalities'

# Standardize class names to match the format in train_df (e.g., 'Nodule-Mass' to 'Nodule/Mass')
processed_df['class_name'] = processed_df['class_name'].replace("Nodule-Mass", "Nodule/Mass")

# Define the class_name to class_id mapping for the final step
class_name_to_id = {
    "Aortic_enlargement": 0,
    "Atelectasis": 1,
    "Calcification": 2,
    "Cardiomegaly": 3,
    "Consolidation": 4,
    "ILD": 5,
    "Infiltration": 6,
    "Lung_Opacity": 7,
    "Nodule/Mass": 8,  # Ensure it matches with 'Nodule/Mass' for consistency
    "Other_lesion": 9,
    "Pleural_effusion": 10,
    "Pleural_thickening": 11,
    "Pneumothorax": 12,
    "Pulmonary_fibrosis": 13
}

# Assign class_id based on class_name
processed_df['class_id'] = processed_df['class_name'].map(class_name_to_id)

# Check for any NaN values in class_id (should be 0 if no issue)
print(f"âœ… Number of NaN class IDs: {processed_df['class_id'].isna().sum()}")  # Should be 0

# Print the first few rows of the final DataFrame
print(f"âœ… Final dataset preview:")
print(processed_df.head())

# âœ… Now, merge with train_df (if exists) or create a new train_df
train_df = pd.concat([train_df, processed_df], ignore_index=True)

# âœ… Print final dataset details
print(f"âœ… Number of unique images after merging: {train_df['image_id'].nunique()}")
train_df.head()


# âœ… Group by image_id and get unique classes per image
unique_class_per_image = train_df.groupby("image_id")["class_name"].unique()

# âœ… Count how many images each class appears in
class_counts = unique_class_per_image.explode().value_counts()

# âœ… Map class_name to class_id
class_name_to_id = train_df.drop_duplicates("class_name")[["class_name", "class_id"]].set_index("class_name")["class_id"].to_dict()

# âœ… Add class_id to the labels
class_labels_with_ids = [f"{cls} (ID {class_name_to_id.get(cls, 'Unknown')})" for cls in class_counts.index]

# âœ… Sort class counts
class_counts = class_counts.sort_values(ascending=False)
class_labels_with_ids = [label for _, label in sorted(zip(class_counts.values, class_labels_with_ids), reverse=True)]

# âœ… Create color palette
colors = sns.color_palette("tab20", len(class_counts))

# âœ… Plot the class distribution
plt.figure(figsize=(14, 6))
bars = plt.bar(class_labels_with_ids, class_counts.values, color=colors)
plt.title('Class Distribution Across Images (Unique Occurrences)', fontsize=16)
plt.xlabel('Class Name (with Class ID)', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
plt.xticks(rotation=45, ha='right')

# âœ… Annotate counts on top of bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, str(height), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()


# âœ… Group by class_name and count total occurrences (across all bounding boxes)
class_counts = train_df['class_name'].value_counts()

# âœ… Map class_name to class_id
class_name_to_id = train_df.drop_duplicates("class_name")[["class_name", "class_id"]].set_index("class_name")["class_id"].to_dict()

# âœ… Add class_id to the labels
class_labels_with_ids = [f"{cls} (ID {class_name_to_id.get(cls, 'Unknown')})" for cls in class_counts.index]

# âœ… Sort class counts
class_counts = class_counts.sort_values(ascending=False)
class_labels_with_ids = [label for _, label in sorted(zip(class_counts.values, class_labels_with_ids), reverse=True)]

# âœ… Create color palette
colors = sns.color_palette("tab20", len(class_counts))

# âœ… Plot the class distribution
plt.figure(figsize=(14, 6))
bars = plt.bar(class_labels_with_ids, class_counts.values, color=colors)
plt.title('Class Distribution Across All Bounding Boxes (Total Occurrences)', fontsize=16)
plt.xlabel('Class Name (with Class ID)', fontsize=12)
plt.ylabel('Total Occurrences (Bounding Boxes)', fontsize=12)
plt.xticks(rotation=45, ha='right')

# âœ… Annotate counts on top of bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, str(height), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()


# âœ… Define the class mapping dictionary with old IDs (0 to 13) and their corresponding new groupings
class_mapping = {
    0: "Aortic Enlargement",    # Aortic enlargement (ID 0)
    1: "Lung Collapse",         # Atelectasis (ID 1)
    2: "Cardiomegaly",          # Cardiomegaly (ID 2)
    3: "Opacities/Infiltration",  # Consolidation (ID 3)
    4: "Opacities/Infiltration",  # ILD (ID 4)
    5: "Opacities/Infiltration",  # Infiltration (ID 5)
    6: "Opacities/Infiltration",  # Lung Opacity (ID 6)
    7: "Nodule/Mass",           # Nodule/Mass (ID 7)
    8: "Nodule/Mass",           # Nodule/Mass (ID 8)
    9: "Pleural Conditions",    # Pleural Effusion (ID 9)
    10: "Pleural Conditions",   # Pleural Thickening (ID 10)
    11: "Lung Collapse",        # Pneumothorax (ID 11)
    12: "Opacities/Infiltration",  # Pulmonary fibrosis (ID 12)
    13: "Opacities/Infiltration"   # Lung Opacity (ID 13)
}

# âœ… Apply the class mapping to create `mapped_class_name`
train_df['mapped_class_name'] = train_df['class_id'].map(class_mapping)

# âœ… Debugging: Check for unmapped class IDs
unmapped_classes = train_df[train_df['mapped_class_name'].isna()]['class_id'].unique()
if len(unmapped_classes) > 0:
    print(f"âš ï¸� Warning: Some class IDs are not mapped! Unmapped class IDs: {unmapped_classes}")

# âœ… Remove any NaN values before creating unique class names
train_df = train_df.dropna(subset=['mapped_class_name'])

# âœ… Get unique class names (ensuring correct count)
class_names = sorted(train_df['mapped_class_name'].unique())

# âœ… Explicitly map class names to new indices
new_class_ids = {name: idx for idx, name in enumerate(class_names)}

# âœ… Apply the new mapping to create a new class ID
train_df['new_class_id'] = train_df['mapped_class_name'].map(new_class_ids)

# âœ… Create the final new class mapping (new class IDs)
new_class_mapping = {idx: name for name, idx in new_class_ids.items()}

# âœ… Verify the new class mapping
print(f"âœ… New class mapping (new class IDs): {new_class_mapping}")
train_df.head()


import matplotlib.pyplot as plt
import cv2

# âœ… Function to visualize images with bounding boxes and show image path
def visualize_bboxes(image_paths, bboxes_list, labels_list, image_sizes, num_images=5):
    num_images = min(num_images, len(image_paths))  
    fig, axes = plt.subplots(1, num_images, figsize=(50, 30))

    if num_images == 1:
        axes = [axes]

    for idx in range(num_images):
        image_path, bboxes, labels, (orig_width, orig_height) = (
            image_paths[idx], bboxes_list[idx], labels_list[idx], image_sizes[idx])
        
        # âœ… Print image path to console/log
        print(f"\nğŸ–¼ Visualizing image {idx + 1}/{num_images}: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"âš ï¸� Error: Could not read {image_path}")
            continue
    
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        height, width, _ = image.shape  
    
        # âœ… Rescale and draw bounding boxes
        for i in range(len(bboxes)):
            x_min, y_min, x_max, y_max = bboxes[i]
            x_min = int((x_min / orig_width) * width)
            y_min = int((y_min / orig_height) * height)
            x_max = int((x_max / orig_width) * width)
            y_max = int((y_max / orig_height) * height)
            bboxes[i] = [x_min, y_min, x_max, y_max]
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, labels[i], (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
        axes[idx].imshow(image)
        axes[idx].axis("off")
        axes[idx].set_title(image_path.split('/')[-1], fontsize=14, color='blue')  # Optional shorter title

    plt.tight_layout()
    plt.show()

# âœ… Select a random sample of unique images for visualization
num_samples = 5
sampled_images = train_df["image_id"].drop_duplicates().sample(n=min(num_samples, train_df["image_id"].nunique())).tolist()

# âœ… Prepare lists for visualization
image_paths, bboxes_list, labels_list, image_sizes = [], [], [], []

for image_id in sampled_images:
    sample_df = train_df[train_df["image_id"] == image_id]
    image_paths.append(sample_df["image_path"].iloc[0])
    bboxes_list.append(sample_df[["x_min", "y_min", "x_max", "y_max"]].values.tolist())
    labels_list.append(sample_df["mapped_class_name"].tolist())  # Use mapped class names
    image_sizes.append(sample_df[["width", "height"]].iloc[0].tolist())

# âœ… Visualize images with bounding boxes!
visualize_bboxes(image_paths, bboxes_list, labels_list, image_sizes, num_images=num_samples)



# âœ… Function to update the bounding boxes in the DataFrame
def update_bboxes_in_df(sampled_images, bboxes_list_updated):
    for i, image_id in enumerate(sampled_images):
        sample_df = train_df[train_df["image_id"] == image_id].copy()
        updated_bboxes = bboxes_list_updated[i]
        
        for j, bbox in enumerate(updated_bboxes):
            train_df.loc[sample_df.index[j], ['x_min', 'y_min', 'x_max', 'y_max']] = bbox

    print("âœ… Bounding boxes updated in train_df!")

# âœ… Assuming bboxes_list_updated has been returned from the visualization function
# Call the update function after visualizing
update_bboxes_in_df(sampled_images, bboxes_list_updated)

# âœ… (Optional) Save the updated DataFrame to a CSV or other formats
# train_df.to_csv('updated_train_df.csv', index=False)



import seaborn as sns
import matplotlib.pyplot as plt

# âœ… Group by image_id and mapped_class_name, counting unique classes per image
unique_classes_per_image = train_df.groupby("image_id")["mapped_class_name"].nunique()

# âœ… Count how many images have each unique class (counting each class once per image)
class_counts = train_df.groupby("mapped_class_name")["image_id"].nunique()
class_counts = class_counts.sort_values(ascending=False)

# âœ… Create a color palette for the plot
colors = sns.color_palette("Set2", len(class_counts))

# âœ… Plot the class distribution showing how many images each class appeared in
plt.figure(figsize=(12, 6))
class_counts.plot(kind='bar', color=colors)

# âœ… Add class count labels on top of bars
for i, value in enumerate(class_counts.values):
    plt.text(i, value + 1, str(value), ha='center', va='bottom', fontsize=10)

# âœ… Set plot title and labels
plt.title('Mapped Class Distribution Across Images (Unique Occurrences)', fontsize=16)
plt.xlabel('Mapped Class Name', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
plt.xticks(rotation=45, ha='right')

# âœ… Add new class ID labels next to the class name on the x-axis
# Here, we get the new class ID directly from train_df's 'new_class_id'
new_class_labels = [f"{name} (ID: {train_df[train_df['mapped_class_name'] == name]['new_class_id'].iloc[0]})" 
                    for name in class_counts.index]

plt.xticks(ticks=range(len(class_counts)), labels=new_class_labels, rotation=45, ha='right')

# âœ… Adjust layout and show plot
plt.tight_layout()
plt.show()

# âœ… Display the class distribution with new class IDs
print(f"Mapped class distribution (counting each class once per image):\n{class_counts}")

# âœ… Create the new class ID mapping and display it
new_class_id_mapping = {name: train_df[train_df['mapped_class_name'] == name]['new_class_id'].iloc[0] 
                        for name in class_counts.index}
print(f"\nNew Class IDs Mapping:\n{new_class_id_mapping}")


import seaborn as sns
import matplotlib.pyplot as plt

# âœ… Count how many times each class appears in each image (including duplicates for multiple bboxes)
class_occurrences = train_df.groupby(["image_id", "mapped_class_name"]).size().reset_index(name="count")

# âœ… Sum the counts of each class across all images
class_counts = class_occurrences.groupby("mapped_class_name")["count"].sum()

# âœ… Sort by count (optional, for better visualization)
class_counts = class_counts.sort_values(ascending=False)

# âœ… Create a color palette for the plot
colors = sns.color_palette("Set2", len(class_counts))

# âœ… Plot the class distribution with annotations
plt.figure(figsize=(12, 6))
barplot = class_counts.plot(kind='bar', color=colors)

# âœ… Add value annotations above each bar
for i, value in enumerate(class_counts.values):
    plt.text(i, value + max(class_counts.values) * 0.01, str(value), ha='center', va='bottom', fontsize=10)

# âœ… Set plot title and labels
plt.title('Mapped Class Distribution Across All Images (Total Bounding Box Occurrences)', fontsize=16)
plt.xlabel('Mapped Class Name', fontsize=12)
plt.ylabel('Total Number of Bounding Boxes', fontsize=12)
plt.xticks(rotation=45, ha='right')

# âœ… Add new class ID labels next to the class name on the x-axis
# We are using the `new_class_id` from `train_df` to get the correct IDs for the classes
new_class_labels = [f"{name} (ID: {train_df[train_df['mapped_class_name'] == name]['new_class_id'].iloc[0]})" for name in class_counts.index]

plt.xticks(ticks=range(len(class_counts)), labels=new_class_labels, rotation=45, ha='right')

# âœ… Grid lines for better readability
plt.grid(axis='y', linestyle='--', alpha=0.5)

# âœ… Adjust layout and show plot
plt.tight_layout()
plt.show()

# âœ… Print the full distribution as a summary
print("\nğŸ“Š Mapped class distribution (counting total bounding box occurrences across images):")
print(class_counts)

# âœ… Print the new class ID mapping
new_class_id_mapping = {name: train_df[train_df['mapped_class_name'] == name]['new_class_id'].iloc[0] for name in class_counts.index}
print(f"\nNew Class IDs Mapping:\n{new_class_id_mapping}")



# âœ… Apply WBF Function
def apply_wbf(train_df, iou_thr=0.1, skip_box_thr=0.0001, min_box_size=0.001):
    output = []

    for image_id, group in tqdm(train_df.groupby("image_id"), desc="Applying WBF"):
        w, h = group['width'].iloc[0], group['height'].iloc[0]

        boxes_list = []
        scores_list = []
        labels_list = []

        boxes_single = []
        labels_single = []

        count_dict = Counter(group['new_class_id'].tolist())
        class_ids = group['new_class_id'].unique().tolist()

        for cid in class_ids:
            class_group = group[group.new_class_id == cid]

            if count_dict[cid] == 1:
                row = class_group.iloc[0]
                # Use YOLO-normalized box and convert to (x_min, y_min, x_max, y_max)
                x_mid, y_mid, box_w, box_h = row['x_mid'], row['y_mid'], row['w'], row['h']
                x_min = x_mid - box_w / 2
                y_min = y_mid - box_h / 2
                x_max = x_mid + box_w / 2
                y_max = y_mid + box_h / 2

                box = [x_min, y_min, x_max, y_max]
                boxes_single.append(box)
                labels_single.append(cid)
            else:
                # Same as above, use YOLO-normalized coords
                x_mid = class_group['x_mid'].to_numpy()
                y_mid = class_group['y_mid'].to_numpy()
                box_w = class_group['w'].to_numpy()
                box_h = class_group['h'].to_numpy()

                x_min = x_mid - box_w / 2
                y_min = y_mid - box_h / 2
                x_max = x_mid + box_w / 2
                y_max = y_mid + box_h / 2

                bboxes = np.stack([x_min, y_min, x_max, y_max], axis=1)
                bboxes = np.clip(bboxes, 0, 1)  # Ensure in [0,1]

                boxes_list.append(bboxes.tolist())
                scores_list.append([1.0] * len(class_group))
                labels_list.append([cid] * len(class_group))

        # Apply WBF
        if boxes_list:
            fused_boxes, _, fused_labels = weighted_boxes_fusion(
                boxes_list, scores_list, labels_list,
                weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr
            )
        else:
            fused_boxes, fused_labels = np.empty((0, 4)), np.empty((0,))

        # Combine with singles
        if len(boxes_single) > 0:
            all_boxes = np.vstack([fused_boxes, boxes_single])
            all_labels = np.hstack([fused_labels, labels_single])
        else:
            all_boxes = fused_boxes
            all_labels = fused_labels

        # Convert back to YOLO format and append
        for box, label in zip(all_boxes, all_labels):
            x_min, y_min, x_max, y_max = box
            box_w = x_max - x_min
            box_h = y_max - y_min
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2

            # Filter out tiny boxes
            if box_w > min_box_size and box_h > min_box_size:
                output.append({
                    "image_id": image_id,
                    "x_mid": x_center,
                    "y_mid": y_center,
                    "w": box_w,
                    "h": box_h,
                    "new_class_id": int(label) if isinstance(label, (int, float)) else label
                })

    return pd.DataFrame(output)

# âœ… Display summary BEFORE WBF
print("ğŸ“Š Before WBF:")
print(f"ğŸ”¹ Total images: {train_df['image_id'].nunique()}")
print(f"ğŸ”¹ Total labels: {len(train_df)}\n")

# âœ… Apply WBF once
train_df_wbf = apply_wbf(train_df)

# âœ… Merge additional image metadata
train_df_wbf = train_df_wbf.merge(
    train_df[['image_id', 'image_path', 'width', 'height', 'source_dataset']].drop_duplicates(),
    on='image_id',
    how='left'
)

# âœ… Display summary AFTER WBF
print("ğŸ“Š After WBF:")
print(f"âœ… Total images: {train_df_wbf['image_id'].nunique()}")
print(f"âœ… Total labels: {len(train_df_wbf)}")

print("\nğŸ”� Sample of processed DataFrame:")
print(train_df_wbf.head(5))

print("\nğŸ“Œ Columns in WBF output:")
print(train_df_wbf.columns.tolist())


import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# âœ… Visualization function
# âœ… Compare visualization before vs after WBF
def visualize_before_after(image_id, df_before, df_after, class_names=None):
    # Get the image path from df_before
    img_path = df_before[df_before["image_id"] == image_id].iloc[0]["image_path"]
    
    # Check if the image exists
    if not os.path.exists(img_path):
        print(f"â�Œ Image not found at {img_path}")
        return
    
    # Read and process the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"â�Œ Failed to load image at {img_path}")
        return
    
    # Convert the image to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Set up the visualization with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    titles = ["Before WBF", "After WBF"]
    dfs = [df_before, df_after]

    # Loop through each subplot (before and after WBF)
    for i, (ax, title, df) in enumerate(zip(axs, titles, dfs)):
        bboxes = df[df["image_id"] == image_id]

        ax.imshow(img)
        ax.set_title(f"{title}", fontsize=16)

        # Loop through the bounding boxes and draw them on the image
        for _, row in bboxes.iterrows():
            x_mid = row["x_mid"] * w
            y_mid = row["y_mid"] * h
            box_w = row["w"] * w
            box_h = row["h"] * h

            x_min = x_mid - box_w / 2
            y_min = y_mid - box_h / 2

            # Draw rectangle for bounding box
            rect = patches.Rectangle(
                (x_min, y_min),
                box_w,
                box_h,
                linewidth=2,
                edgecolor='lime',
                facecolor='none'
            )
            ax.add_patch(rect)

            # Get class ID and label
            class_id = int(row["new_class_id"])
            label = class_names[class_id] if class_names else str(class_id)
            ax.text(
                x_min, y_min - 5, label,
                color='white',
                fontsize=12,
                bbox=dict(facecolor='green', alpha=0.6, edgecolor='none', pad=1)
            )

        ax.axis('off')

    plt.tight_layout()
    plt.show()

# âœ… Define the class names based on your class_mapping
class_names = [
    "Aortic Enlargement",    # ID 0
    "Cardiomegaly",     # ID 1 
    "Lung Collapse", #ID 2
    "Nodule/Mass",   # ID 3 
    "Opacities/Infiltration", # ID 4 
    "Pleural Conditions"     # ID 5 
]

# âœ… Pick 5 random image_ids
sample_ids = random.sample(list(train_df["image_id"].unique()), 5)

# âœ… Visualize each image before and after WBF
for img_id in sample_ids:
    visualize_before_after(img_id, train_df, train_df_wbf, class_names)



# Map class IDs to class names
class_id_to_name = {
    0: "Aortic Enlargement",    # ID 0
    1: "Cardiomegaly",          # ID 1
    2: "Lung Collapse",         # ID 2
    3: "Nodule/Mass",           # ID 3
    4: "Opacities/Infiltration",# ID 4
    5: "Pleural Conditions"     # ID 5
}

# âœ… Unique Class Distribution (Images Count) After WBF using class names
unique_class_distribution = train_df_wbf.groupby('new_class_id')['image_id'].nunique().sort_index()

# Map class IDs to class names
unique_class_distribution = unique_class_distribution.rename(index=class_id_to_name)

# Plot the distribution
plt.figure(figsize=(12, 6))
bars = unique_class_distribution.plot(kind='bar', color='lightgreen')
plt.title('Unique Class Distribution (Images Count) After WBF', fontsize=16)
plt.xlabel('Class Name', fontsize=12)
plt.ylabel('Number of Unique Images', fontsize=12)
plt.xticks(rotation=45)

# Annotate the count on top of each bar
for bar in bars.patches:
    height = bar.get_height()
    bars.text(
        bar.get_x() + bar.get_width() / 2, height + 50,  # Positioning the text above the bar
        f'{height:.0f}', ha='center', va='bottom', fontsize=10
    )

plt.tight_layout()
plt.show()


# âœ… Total Class Distribution (Bounding Boxes Count) After WBF using class names
total_class_distribution = train_df_wbf['new_class_id'].value_counts().sort_index()

# Map class IDs to class names
total_class_distribution = total_class_distribution.rename(index=class_id_to_name)

# Plot the distribution
plt.figure(figsize=(12, 6))
bars = total_class_distribution.plot(kind='bar', color='skyblue')
plt.title('Total Class Distribution (Bounding Boxes Count) After WBF', fontsize=16)
plt.xlabel('Class Name', fontsize=12)
plt.ylabel('Number of Bounding Boxes', fontsize=12)
plt.xticks(rotation=45)

# Annotate the count on top of each bar
for bar in bars.patches:
    height = bar.get_height()
    bars.text(
        bar.get_x() + bar.get_width() / 2, height + 50,  # Positioning the text above the bar
        f'{height:.0f}', ha='center', va='bottom', fontsize=10
    )

plt.tight_layout()
plt.show()


from pandas import DataFrame, read_csv
import cv2
import numpy as np
import random
import shutil
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# Before augmentation
print("Class distribution before augmentation:")
print(train_df_wbf['new_class_id'].value_counts())

# -------------------- Step 1: Define Augmentation Functions --------------------
def apply_clahe(image):
    """ Apply CLAHE contrast enhancement """
    if len(image.shape) == 3:  
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image_gray)

def apply_augmentation(image, class_id, strong_freq_classes, moderate_freq_classes):
    """ Augment the image based on the class_id """
    if class_id in strong_freq_classes:
        if random.random() < 0.7:
            image = cv2.flip(image, 1)
        scale = random.uniform(0.5, 1.5)
        image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
    elif class_id in moderate_freq_classes:
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
        scale = random.uniform(0.8, 1.3)
        image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
    else:
        if random.random() < 0.3:
            image = cv2.flip(image, 1)
        scale = random.uniform(0.9, 1.1)
        image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))

    if random.random() < 0.5:
        image = apply_clahe(image)
    
    return image

# -------------------- Step 2: Class Frequency Analyzer --------------------
def identify_low_freq_classes(df, moderate_threshold=0.165, strong_threshold=0.1):
    class_counts = df['new_class_id'].value_counts(normalize=True)
    strong_freq_classes = class_counts[class_counts < strong_threshold].index.tolist()
    moderate_freq_classes = class_counts[(class_counts >= strong_threshold) & (class_counts < moderate_threshold)].index.tolist()

    print(f"Strong frequency classes (<{strong_threshold}): {strong_freq_classes}")
    print(f"Moderate frequency classes (<{moderate_threshold}): {moderate_freq_classes}")
    
    return moderate_freq_classes, strong_freq_classes

# -------------------- Step 3: Augment Data Based on Class with Most Images --------------------
def augment_and_balance_data(train_df_wbf, max_iter=10):
    augmented_data = []
    augmented_image_ids = set()

    # Step 1: Initial class distribution
    class_counts = train_df_wbf['new_class_id'].value_counts()

    # Find the class with the most images
    max_class_id = class_counts.idxmax()
    max_class_count = class_counts.max()
    print(f"[INFO] Class with most images is {max_class_id} with {max_class_count} images.")

    # Step 2: Augment classes to match the class with the most images
    for iteration in range(max_iter):
        print(f"[INFO] Iteration {iteration + 1}...")

        # Combine base and augmented data for up-to-date class counts
        temp_df = pd.concat([train_df_wbf, pd.DataFrame(augmented_data)], ignore_index=True)
        class_counts = temp_df['new_class_id'].value_counts()

        for class_id in class_counts.index:
            current_class_count = len(temp_df[temp_df['new_class_id'] == class_id])
            print(f"[INFO] Class {class_id} currently has {current_class_count} images.")

            if current_class_count < max_class_count:
                augment_needed = max_class_count - current_class_count
                print(f"[INFO] Augmenting class {class_id} by {augment_needed} samples.")

                class_group = train_df_wbf[train_df_wbf['new_class_id'] == class_id]

                # Ensure enough samples by repeating class_group
                repeat_factor = (augment_needed // len(class_group)) + 1
                extended_group = pd.concat([class_group] * repeat_factor, ignore_index=True)

                for i in range(augment_needed):
                    row = extended_group.iloc[i]
                    image_path = row['image_path']
                    image = cv2.imread(image_path)

                    if image is None:
                        print(f"â�Œ Failed to load image: {image_path}")
                        continue

                    augmented_image = apply_augmentation(image.copy(), class_id, [], [])

                    augmented_image_id = f"{row['image_id']}_aug_{iteration}_{i}"
                    if augmented_image_id not in augmented_image_ids:
                        augmented_image_ids.add(augmented_image_id)

                        # Recalculate bounding boxes (same as original for now)
                        x_min, y_min = row['x_mid'] - row['w'] / 2, row['y_mid'] - row['h'] / 2
                        x_max, y_max = row['x_mid'] + row['w'] / 2, row['y_mid'] + row['h'] / 2
                        new_x_mid = (x_min + x_max) / 2
                        new_y_mid = (y_min + y_max) / 2
                        new_w = x_max - x_min
                        new_h = y_max - y_min

                        augmented_data.append({
                            'image_id': augmented_image_id,
                            'image_path': image_path,
                            'new_class_id': row['new_class_id'],
                            'x_mid': new_x_mid,
                            'y_mid': new_y_mid,
                            'w': new_w,
                            'h': new_h
                        })

        # Recalculate combined class counts and check if weâ€™re done
        temp_df = pd.concat([train_df_wbf, pd.DataFrame(augmented_data)], ignore_index=True)
        class_counts = temp_df['new_class_id'].value_counts()
        if all(count >= max_class_count for count in class_counts):
            print(f"[âœ…] All classes have reached {max_class_count}. Stopping early.")
            break

    # Final balanced dataset
    augmented_df = pd.DataFrame(augmented_data)
    balanced_df = pd.concat([train_df_wbf, augmented_df], ignore_index=True)

    return balanced_df
# -------------------- Step 4: Final Prep & Label Generation --------------------
def prepare_yolo_labels_for_augmented(df, image_dest_dir, label_dest_dir):
    df = df.drop_duplicates(subset=['image_id', 'new_class_id', 'x_mid', 'y_mid', 'w', 'h'])
    os.makedirs(image_dest_dir, exist_ok=True)
    os.makedirs(label_dest_dir, exist_ok=True)

    for image_id, group in tqdm(df.groupby('image_id'), desc=f"Processing {image_dest_dir}"):
        image_path = group.iloc[0]['image_path']
        label_file = os.path.join(label_dest_dir, os.path.basename(image_path).replace('.png', '.txt'))

        augmented_image_path = os.path.join(image_dest_dir, f"{image_id}.png")
        shutil.copy(image_path, augmented_image_path)

        with open(label_file, 'w') as f:
            for _, row in group.iterrows():
                f.write(f"{row['new_class_id']} {row['x_mid']} {row['y_mid']} {row['w']} {row['h']}\n")

# -------------------- Run the Full Pipeline --------------------
moderate_freq_classes, strong_freq_classes = identify_low_freq_classes(train_df_wbf)

# Augmentation and balancing
balanced_df = augment_and_balance_data(train_df_wbf)

# Save to YOLO format
prepare_yolo_labels_for_augmented(balanced_df, 'data/images/train', 'data/labels/train')

# Summary Stats
print(f"\nğŸ“Š Total Images in Augmented Training Set: {balanced_df['image_id'].nunique()}")
print(f"âœ… Total Labels in Augmented Training Set: {len(balanced_df)}")

# Plot Distribution
class_id_to_name = {
    0: "Aortic Enlargement",    
    1: "Cardiomegaly",
    2: "Lung Collapse",
    3: "Nodule/Mass",
    4: "Opacities/Infiltration",
    5: "Pleural Conditions"
}

augmented_class_distribution = balanced_df['new_class_id'].value_counts(normalize=True).sort_index()
augmented_class_distribution = augmented_class_distribution.rename(index=class_id_to_name)

# After augmentation
print("Class distribution after augmentation:")
print(balanced_df['new_class_id'].value_counts())

# Plot the class distribution
augmented_class_distribution.plot(kind='bar', title="Class Distribution After Augmentation")
plt.xlabel('Class')
plt.ylabel('Proportion')
plt.show()


# Pretty plot of the class distribution
plt.figure(figsize=(12, 6))
bars = augmented_class_distribution.plot(
    kind='bar', 
    color='skyblue', 
    edgecolor='black'
)

# Add title and labels
plt.title("Class Distribution After Data Augmentation", fontsize=16, fontweight='bold')
plt.xlabel("Condition", fontsize=12)
plt.ylabel("Proportion of Images", fontsize=12)
plt.xticks(rotation=45)

# Annotate each bar with its value
for bar in bars.patches:
    height = bar.get_height()
    bars.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.005,
        f"{height:.2f}",
        ha='center',
        va='bottom',
        fontsize=10
    )

plt.tight_layout()
plt.show()


import os
import shutil
from tqdm import tqdm

# âœ… Function to prepare YOLO labels and move the images to appropriate directories
def prepare_yolo_labels(df, image_dest_dir, label_dest_dir):
    # âœ… Remove duplicate bounding boxes before writing
    df = df.drop_duplicates(subset=['image_id', 'new_class_id', 'x_mid', 'y_mid', 'w', 'h'])

    # Create directories if they don't exist
    os.makedirs(image_dest_dir, exist_ok=True)
    os.makedirs(label_dest_dir, exist_ok=True)

    for image_id, group in tqdm(df.groupby('image_id'), desc=f"Processing {image_dest_dir}"):
        image_path = group.iloc[0]['image_path']
        label_file = os.path.join(label_dest_dir, os.path.basename(image_path).replace('.png', '.txt'))

        # âœ… Copy image
        shutil.copy(image_path, os.path.join(image_dest_dir, os.path.basename(image_path)))

        # âœ… Write all labels in one go
        with open(label_file, 'w') as f:
            for _, row in group.iterrows():
                f.write(f"{row['new_class_id']} {row['x_mid']} {row['y_mid']} {row['w']} {row['h']}\n")

# Your data split code (StratifiedGroupKFold)
from sklearn.model_selection import StratifiedGroupKFold

# StratifiedGroupKFold split
balanced_df_multi = balanced_df.groupby('image_id')['new_class_id'].agg(lambda x: list(set(x))).reset_index()
balanced_df = balanced_df.merge(balanced_df_multi, on='image_id', suffixes=("", "_multi"))
# Drop duplicated column â€” keep only one of them
balanced_df = balanced_df.loc[:, ~balanced_df.columns.duplicated()]

# Now safely assign the stratification string
balanced_df['multi_class_str'] = balanced_df['new_class_id_multi'].apply(lambda x: str(sorted(x)))


sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)

for train_idx, val_idx in sgkf.split(balanced_df, balanced_df['multi_class_str'], groups=balanced_df['image_id']):
    train_df_split = balanced_df.iloc[train_idx].reset_index(drop=True)
    val_df_split = balanced_df.iloc[val_idx].reset_index(drop=True)
    break

train_df_split.drop(columns=['new_class_id_multi', 'multi_class_str'], inplace=True)
val_df_split.drop(columns=['new_class_id_multi', 'multi_class_str'], inplace=True)

# Prepare YOLO labels
prepare_yolo_labels(train_df_split, 'data/images/train', 'data/labels/train')
prepare_yolo_labels(val_df_split, 'data/images/val', 'data/labels/val')

# Function to remove exact duplicate lines from YOLO label files
def deduplicate_yolo_labels(label_dir):
    label_paths = glob(os.path.join(label_dir, "*.txt"))
    total_files = len(label_paths)
    deduplicated_count = 0
    duplicate_files = []

    for path in label_paths:
        with open(path, 'r') as f:
            lines = f.readlines()
        original_count = len(lines)
        deduped = list(set([line.strip() for line in lines]))
        deduped_count = len(deduped)

        if deduped_count < original_count:
            deduplicated_count += 1
            duplicate_files.append((os.path.basename(path), original_count - deduped_count))
            with open(path, 'w') as f:
                f.write('\n'.join(deduped) + '\n')

    print(f"âœ… Deduplicated {deduplicated_count}/{total_files} files in: {label_dir}")
    if duplicate_files:
        print("ğŸ”� Files with duplicates removed:")
        for fname, dup_count in duplicate_files:
            print(f"{fname}: {dup_count} duplicates removed")

# Deduplicate labels
deduplicate_yolo_labels('data/labels/train')
deduplicate_yolo_labels('data/labels/val')



import matplotlib.pyplot as plt
import random
from PIL import Image

# Define the path to your augmented image directory
augmented_image_dir = 'data/images/train'

# Get a list of all the image filenames in the directory
augmented_image_files = [f for f in os.listdir(augmented_image_dir) if f.endswith('.png')]

# Randomly sample a few images (for example, 5 images)
sample_images = random.sample(augmented_image_files, 5)

# Create a plot to display the images
fig, axes = plt.subplots(1, 5, figsize=(15, 10))

for ax, img_file in zip(axes, sample_images):
    # Open the image
    img = Image.open(os.path.join(augmented_image_dir, img_file))
    
    # Display the image
    ax.imshow(img)
    ax.axis('off')  # Remove axis labels
    
    # Set title as the image filename (or any other info)
    ax.set_title(f"Sample: {img_file}")

# Show the plot
plt.tight_layout()
plt.show()



import cv2
import os

def convert_images_to_grayscale_in_place(image_dir):
    """
    Convert all images in the directory to grayscale and overwrite them in place.
    Displays progress as it works.
    """
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total = len(image_files)
    converted = 0

    print(f"ğŸ•µï¸� Processing {total} images in: {image_dir}")

    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(image_dir, image_file)
        img = cv2.imread(image_path)

        if img is None:
            print(f"â�Œ Failed to read: {image_file}")
            continue

        # Convert the image to grayscale without checking if it's already grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(image_path, img_gray)  # Overwrite the original image
        converted += 1

        # Show progress every 100 images or at end
        if idx % 100 == 0 or idx == total:
            print(f"âœ… [{idx}/{total}] processed, {converted} converted", flush=True)

    print(f"ğŸ�‰ Done! {converted} images converted in: {image_dir}\n")

# Directories for train and validation images
train_dir = '/kaggle/working/data/images/train'
val_dir = '/kaggle/working/data/images/val'

# Convert both folders
convert_images_to_grayscale_in_place(train_dir)
convert_images_to_grayscale_in_place(val_dir)

# Save updated YAML file for YOLOv5
data_yaml = f"""
train: {train_dir}
val: {val_dir}

nc: 6
names:
  - Aortic Enlargement
  - Cardiomegaly
  - Lung Collapse
  - Nodule/Mass
  - Opacities/Infiltration
  - Pleural Conditions
"""

# Save YAML file
yaml_path = '/kaggle/working/data.yaml'
with open(yaml_path, 'w') as f:
    f.write(data_yaml)

print(f"ğŸ“„ data.yaml file updated at {yaml_path} for grayscale images.")


# âœ… Check available GPUs
num_gpus = torch.cuda.device_count()
print(f"Available GPUs: {num_gpus}")
for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# âœ… CUDA memory optimization
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# âœ… Load YOLOv12-M model (ensure yolo12m.pt is available)
model = YOLO("yolo12m.pt")  # Adjust the model path if necessary

# âœ… Train the model with the grayscale images
train_results = model.train(
    data="/kaggle/working/data.yaml",  # Path to the updated data.yaml file
    epochs=80,  # Number of epochs
    batch=16,  # Batch size
    imgsz=640,  # Image size (should match your model's input requirements)
    device="auto",  # Let it choose the best available GPU
    half=True,  # Use half precision if available for faster training
    workers=4,  # Number of workers for loading the data
    project="yolov12-training",  # Path where the training outputs will be saved
    name="yolo12m-vinbigdata",  # Name for the training run
    exist_ok=True,  # Overwrite existing project with the same name if necessary
    save=True,  # Save the best model
    save_period=10,  # Save every 10 epochs
)

# âœ… Print the final results
print("âœ… Training complete!")
print("The training results have been saved in the 'yolov12-training' folder.")





# âœ… Check training results
print("âœ… Training results:")
print(train_results)

# âœ… Validate the model
metrics = model.val()

# âœ… Print evaluation metrics
print("Evaluation metrics:")
print(metrics)


!zip -r yolov12-training.zip /kaggle/working/yolov12-training/


# Provide a clickable download link
FileLink(r'yolov12-training.zip')


!pip install ultralytics
from ultralytics import YOLO


model_path = "/kaggle/input/mymodel/pytorch/yolo12/1/best.pt"
model = YOLO(model_path)


import torch
import numpy as np
import glob
from collections import defaultdict, Counter

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Get all image file paths
image_dir = "/kaggle/input/vinbigdata-1024-image-dataset/vinbigdata/test/"
image_paths = sorted(glob.glob(image_dir + "*.png"))  # Adjust extension if needed

batch_size = 16  # Adjust based on GPU memory
all_results = []

# Run inference in batches
for i in range(0, len(image_paths), batch_size):
    batch = image_paths[i : i + batch_size]  # Get batch of image paths
    batch_results = model(batch, conf=0.1)  # Run inference
    all_results.extend(batch_results)

# Collect confidence scores by class
conf_scores = defaultdict(list)

for result in all_results:
    if result.boxes is not None:  # Ensure detections exist
        for det in result.boxes.to(device):  # Keep tensors on GPU
            cls = int(det.cls.item())  # Get class ID
            conf = float(det.conf.item())  # Get confidence score
            class_name = model.names[cls] if hasattr(model, "names") else str(cls)
            conf_scores[class_name].append(conf)

# Calculate the mode (most common confidence score) and 90th percentile for each class
class_conf_stats = {}

for cls, scores in conf_scores.items():
    # Calculate the mode
    mode_conf = Counter(scores).most_common(1)[0][0]
    
    # Calculate the 90th percentile
    percentile_90 = np.percentile(scores, 90)
    
    class_conf_stats[cls] = {
        "mode": mode_conf,
        "90th_percentile": percentile_90
    }

print(class_conf_stats)


# modes & 90th percentile for each class
class_conf_data = {
    'Cardiac & Vascular': {'mode': 0.131, '90th_percentile': 0.705},
    'Pleural Abnormalities': {'mode': 0.115, '90th_percentile': 0.389},
    'Lung Collapse': {'mode': 0.111, '90th_percentile': 0.507},
    'Fibrosis & ILD': {'mode': 0.337, '90th_percentile': 0.635},
    'Nodule/Mass or Other Lesion': {'mode': 0.208, '90th_percentile': 0.524},
    'Lung Opacity': {'mode': 0.195, '90th_percentile': 0.515}
}

adjusted_thresholds = {}

for cls, values in class_conf_data.items():
    mode = values['mode']
    perc90 = values['90th_percentile']

    if mode < 0.2:
        # Use the 75th percentile if mode is too low
        new_threshold = np.percentile([mode, perc90], 75)
    elif mode >= 0.3:
        # Use the mode directly if reasonable
        new_threshold = mode
    else:
        # Use a weighted average if 90th percentile is much higher
        new_threshold = (0.7 * mode) + (0.3 * perc90)

    adjusted_thresholds[cls] = round(new_threshold, 3)

print(adjusted_thresholds)


import os
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO  

# âœ… Force inline display in Kaggle
%matplotlib inline  

# Load YOLO model
model_path = "/kaggle/input/mymodel/pytorch/yolo12/1/best.pt"
model = YOLO(model_path)

# Confidence thresholds
adjusted_thresholds = {
    'Cardiac & Vascular': 0.562,
    'Pleural Abnormalities': 0.32,
    'Lung Collapse': 0.408,
    'Fibrosis & ILD': 0.337,
    'Nodule/Mass or Other Lesion': 0.303,
    'Lung Opacity': 0.435
}

# Unique colors per class
class_colors = {
    'Cardiac & Vascular': (255, 0, 0),
    'Pleural Abnormalities': (0, 255, 0),
    'Lung Collapse': (0, 0, 255),
    'Fibrosis & ILD': (255, 255, 0),
    'Nodule/Mass or Other Lesion': (255, 165, 0),
    'Lung Opacity': (128, 0, 128)
}

# Dataset path
dataset_path = Path("/kaggle/input/testing/")

# Function to preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return img

# Process images
for image_path in dataset_path.glob("*.*"):
    if image_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
        processed_img = preprocess_image(image_path)
        img_array = np.array(processed_img)  
        img_height, img_width = img_array.shape[:2]  

        # Run inference
        results = model(processed_img)[0]  

        # Extract boxes, confidences, and classes
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        filtered_boxes, filtered_classes, filtered_confidences = [], [], []

        for i in range(len(classes)):
            class_idx = classes[i]
            class_name = model.names[class_idx]
            conf = confidences[i]

            class_threshold = adjusted_thresholds.get(class_name, 0.1)
            if conf >= class_threshold:
                filtered_boxes.append(boxes[i])
                filtered_classes.append(class_name)
                filtered_confidences.append(conf)

        # If valid detections exist
        if filtered_boxes:
            image = np.array(processed_img)  

            for i in range(len(filtered_boxes)):
                x1, y1, x2, y2 = map(int, filtered_boxes[i])

                # Assign unique color
                color = class_colors.get(filtered_classes[i], (255, 255, 255))  

                # Dynamic font scaling
                font_scale = max(0.5, min(img_width, img_height) / 600)  # Adjusted for image size
                thickness = max(2, int(font_scale * 2))  # Bold text by increasing thickness

                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)  

                # Text label (No background)
                label = f"{filtered_classes[i]} ({filtered_confidences[i]:.2f})"
                
                # Draw text
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

            # âœ… Convert BGR to RGB for proper display
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # âœ… Ensure images appear properly
            plt.figure(figsize=(8, 8))
            plt.imshow(image_rgb)  # Show the image in RGB format
            plt.axis("off")
            plt.title(f"Detections for {image_path.name}")
            plt.show()



%matplotlib inline





