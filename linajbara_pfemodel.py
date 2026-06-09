!pip install ultralytics
!pip install torchxrayvision
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
import torchxrayvision as xrv  # For X-ray-specific processing

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

# âœ… Merge with existing train_df
train_df = pd.concat([train_df, new_nih_df], ignore_index=True)

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

# âœ… Merge with train_df
train_df = pd.concat([train_df, nih_df], ignore_index=True)

# âœ… Assign class_id
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

# âœ… Check + summary
print(f"Number of NaN class IDs: {train_df['class_id'].isna().sum()}")
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


# âœ… Define the class mapping dictionary
class_mapping = {
    0: "Cardiac & Vascular", 
    1: "Lung Collapse",  
    2: "Cardiac & Vascular",
    3: "Lung Opacity", 
    4: "Fibrosis & ILD", 
    5: "Lung Opacity",  
    6: "Lung Opacity",  
    7: "Nodule/Mass or Other Lesion", 
    8: "Nodule/Mass or Other Lesion",  
    9: "Pleural Abnormalities",
    10: "Pleural Abnormalities",  
    11: "Lung Collapse",  
    12: "Fibrosis & ILD",
    13: "Lung Opacity"
}

# âœ… Apply class mapping to create `mapped_class_name`
train_df['mapped_class_name'] = train_df['class_id'].map(class_mapping)

# âœ… Debugging: Check for unmapped class IDs
unmapped_classes = train_df[train_df['mapped_class_name'].isna()]['class_id'].unique()
if len(unmapped_classes) > 0:
    print(f"âš ï¸� Warning: Some class IDs are not mapped! Unmapped class IDs: {unmapped_classes}")

# âœ… Remove any NaN values before creating unique class names
train_df = train_df.dropna(subset=['mapped_class_name'])

# âœ… Get unique class names (ensuring correct count)
class_names = sorted(train_df['mapped_class_name'].unique())

# âœ… Explicitly map class names to correct indices
new_class_ids = {name: idx for idx, name in enumerate(class_names)}

# âœ… Apply the new mapping
train_df['new_class_id'] = train_df['mapped_class_name'].map(new_class_ids)

# âœ… Create the final new class mapping
new_class_mapping = {idx: name for name, idx in new_class_ids.items()}

# âœ… Verify the new class mapping
print(f"âœ… New class mapping (new class IDs): {new_class_mapping}")
train_df.head()


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
    return bboxes_list  # Return updated bounding boxes

# âœ… Select a random sample of unique images
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

# âœ… Visualize and update bounding boxes
bboxes_list_updated = visualize_bboxes(image_paths, bboxes_list, labels_list, image_sizes, num_images=num_samples)

# âœ… Update train_df with new bounding boxes
for i, image_id in enumerate(sampled_images):
    sample_df = train_df[train_df["image_id"] == image_id].copy()
    updated_bboxes = bboxes_list_updated[i]
    
    for j, bbox in enumerate(updated_bboxes):
        train_df.loc[sample_df.index[j], ['x_min', 'y_min', 'x_max', 'y_max']] = bbox

print("âœ… Bounding boxes updated in train_df!")


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
for i, value in enumerate(class_counts.values):
    plt.text(i, value + 1, str(value), ha='center', va='bottom', fontsize=10)
plt.title('Mapped Class Distribution Across Images (Unique Occurrences)', fontsize=16)
plt.xlabel('Mapped Class Name', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# âœ… Display the class distribution
print(f"Mapped class distribution (counting each class once per image):\n{class_counts}")


# âœ… Count how many times each class appears in each image (including duplicates for multiple bboxes)
class_occurrences = train_df.groupby(["image_id", "mapped_class_name"]).size().reset_index(name="count")

# âœ… Sum the counts of each class across all images
class_counts = class_occurrences.groupby("mapped_class_name")["count"].sum()

# âœ… Sort by count (optional, for better visualization)
class_counts = class_counts.sort_values(ascending=False)

# âœ… Create a color palette
colors = sns.color_palette("Set2", len(class_counts))

# âœ… Plot the class distribution with annotations
plt.figure(figsize=(12, 6))
barplot = class_counts.plot(kind='bar', color=colors)

# âœ… Add value annotations above each bar
for i, value in enumerate(class_counts.values):
    plt.text(i, value + max(class_counts.values) * 0.01, str(value), ha='center', va='bottom', fontsize=10)

plt.title('Mapped Class Distribution Across All Images (Total Bounding Box Occurrences)', fontsize=16)
plt.xlabel('Mapped Class Name', fontsize=12)
plt.ylabel('Total Number of Bounding Boxes', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# âœ… Print the full distribution as a summary
print("\nğŸ“Š Mapped class distribution (counting total bounding box occurrences across images):")
print(class_counts)


# âœ… Apply WBF Function
def apply_wbf(train_df, iou_thr=0.1, skip_box_thr=0.0001, min_box_size=0.01):
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


# âœ… Visualization function
# âœ… Compare visualization before vs after WBF
def visualize_before_after(image_id, df_before, df_after, class_names=None):
    img_path = df_before[df_before["image_id"] == image_id].iloc[0]["image_path"]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    titles = ["Before WBF", "After WBF"]
    dfs = [df_before, df_after]

    for i, (ax, title, df) in enumerate(zip(axs, titles, dfs)):
        bboxes = df[df["image_id"] == image_id]

        ax.imshow(img)
        ax.set_title(f"{title}", fontsize=16)

        for _, row in bboxes.iterrows():
            x_mid = row["x_mid"] * w
            y_mid = row["y_mid"] * h
            box_w = row["w"] * w
            box_h = row["h"] * h

            x_min = x_mid - box_w / 2
            y_min = y_mid - box_h / 2

            rect = patches.Rectangle(
                (x_min, y_min),
                box_w,
                box_h,
                linewidth=2,
                edgecolor='lime',
                facecolor='none'
            )
            ax.add_patch(rect)

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

# âœ… Define class names (if needed)
class_names = [
    "Cardiac & Vascular",
    "Lung Collapse",
    "Lung Opacity",
    "Fibrosis & ILD",
    "Nodule/Mass or Other Lesion",
    "Pleural Abnormalities"
]

# âœ… Pick 5 random image_ids
sample_ids = random.sample(list(train_df_wbf["image_id"].unique()), 5)

# âœ… Visualize each image before and after WBF
for img_id in sample_ids:
    visualize_before_after(img_id, train_df, train_df_wbf, class_names)


# Step 3: Multi-label per image for stratification
train_df_multi = train_df_wbf.groupby('image_id')['new_class_id'].agg(lambda x: list(set(x))).reset_index()
train_df_wbf = train_df_wbf.merge(train_df_multi, on='image_id', suffixes=("", "_multi"))
train_df_wbf['multi_class_str'] = train_df_wbf['new_class_id_multi'].astype(str)

# Step 4: Perform the stratified group split
sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)

for train_idx, val_idx in sgkf.split(train_df_wbf, train_df_wbf['multi_class_str'], groups=train_df_wbf['image_id']):
    train_df_split = train_df_wbf.iloc[train_idx].reset_index(drop=True)
    val_df_split = train_df_wbf.iloc[val_idx].reset_index(drop=True)
    break

# âœ… Display summary
print(f"âœ… Train Images: {train_df_split['image_id'].nunique()}")
print(f"âœ… Val Images: {val_df_split['image_id'].nunique()}")
print("\nğŸ“Š Source Dataset Distribution:")
print("Train:")
print(train_df_split['source_dataset'].value_counts(normalize=True))
print("Val:")
print(val_df_split['source_dataset'].value_counts(normalize=True))

print("\nğŸ“š Class Distribution:")
print("Train:")
print(train_df_split['new_class_id'].value_counts(normalize=True).sort_index())
print("Val:")
print(val_df_split['new_class_id'].value_counts(normalize=True).sort_index())

# Step 5: Drop temp stratification columns
train_df_split.drop(columns=['new_class_id_multi', 'multi_class_str'], inplace=True)
val_df_split.drop(columns=['new_class_id_multi', 'multi_class_str'], inplace=True)


# âœ… Create necessary directories for YOLO
os.makedirs('data/images/train', exist_ok=True)
os.makedirs('data/images/val', exist_ok=True)
os.makedirs('data/labels/train', exist_ok=True)
os.makedirs('data/labels/val', exist_ok=True)

# âœ… Function to prepare YOLO labels and move the images to appropriate directories
def prepare_yolo_labels(df, image_dest_dir, label_dest_dir):
    # âœ… Remove duplicate bounding boxes before writing
    df = df.drop_duplicates(subset=['image_id', 'new_class_id', 'x_mid', 'y_mid', 'w', 'h'])

    for image_id, group in tqdm(df.groupby('image_id'), desc=f"Processing {image_dest_dir}"):
        image_path = group.iloc[0]['image_path']
        label_file = os.path.join(label_dest_dir, os.path.basename(image_path).replace('.png', '.txt'))

        # âœ… Copy image
        shutil.copy(image_path, os.path.join(image_dest_dir, os.path.basename(image_path)))

        # âœ… Write all labels in one go
        with open(label_file, 'w') as f:
            for _, row in group.iterrows():
                f.write(f"{row['new_class_id']} {row['x_mid']} {row['y_mid']} {row['w']} {row['h']}\n")


# âœ… Prepare and move train and validation data (images and labels) with progress bars
prepare_yolo_labels(train_df_split, 'data/images/train', 'data/labels/train')
prepare_yolo_labels(val_df_split, 'data/images/val', 'data/labels/val')

# âœ… Function to remove exact duplicate lines from YOLO label files
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

# âœ… Deduplicate labels for both train and val
deduplicate_yolo_labels('data/labels/train')
deduplicate_yolo_labels('data/labels/val')



# âœ… Create the data.yaml file for YOLOv11
data_yaml = """  
train: /kaggle/working/data/images/train  
val: /kaggle/working/data/images/val  

nc: 6 
names: [  
  "Cardiac & Vascular",  
  "Lung Collapse",  
  "Lung Opacity",  
  "Fibrosis & ILD",  
  "Nodule/Mass or Other Lesion",  
  "Pleural Abnormalities"  
]
"""

# âœ… Save the YAML file to the working directory
with open('/kaggle/working/data.yaml', 'w') as f:  
    f.write(data_yaml)  

print("âœ… data.yaml file has been created!")


# âœ… Check available GPUs
num_gpus = torch.cuda.device_count()
print(f"Available GPUs: {num_gpus}")
for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# âœ… Set device for training (use the first GPU if available, otherwise use CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} for training")

# âœ… Load YOLOv12-M model
model = YOLO("yolo12m.pt")  # Load pre-trained YOLOv12-M weights

# âœ… Move model to the appropriate device (single GPU or CPU)
model = model.to(device)

# âœ… Optimize CUDA memory allocation
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# âœ… Train the model using the correct training method (from YOLOv12 docs)
train_results = model.train(
    data="/kaggle/working/data.yaml",  # Path to dataset YAML
    epochs=40,
    batch=16,  
    imgsz=640,
    device=device,
    half=True,
    workers=4,  # Increase workers to match GPUs
    project="yolov12-training",
    name="yolo12m-vinbigdata",
    exist_ok=True,
    save=True,
    save_period=10,
)



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





