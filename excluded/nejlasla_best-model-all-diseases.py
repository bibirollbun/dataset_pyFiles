!pip install ultralytics
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
import shutil
import random
import warnings
from glob import glob

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# âœ… Data handling
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import numpy as np
import pandas as pd
import yaml
from tqdm.autonotebook import tqdm

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# âœ… Image handling
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import cv2
import albumentations as A
import skimage.io
import skimage.transform

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# âœ… Machine learning & utilities
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from sklearn.model_selection import StratifiedGroupKFold

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# âœ… Deep learning
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset

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

# Loop through all the label files in the directory
for label_file in os.listdir(labels_dir):
    if label_file.endswith('.txt'):
        # Extract the image ID (without extension and remove _jpg.rf.* part from the filename)
        image_id = label_file.split('.')[0]
        
        # Remove the unwanted parts like _jpg.rf.XXX from the image_id
        image_id_cleaned = image_id.split('_jpg')[0]
        
        # Construct the image path correctly (append .jpg)
        image_path = os.path.join(images_dir, f"{image_id_cleaned}.jpg")
        
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

# Construct the image path correctly (remove unwanted parts of image_id and append .jpg)
processed_df['image_path'] = processed_df['image_id'].apply(lambda x: os.path.join(images_dir, f"{x.split('_jpg')[0]}.jpg"))

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


# Unique classes per image
unique_class_per_image = train_df.groupby("image_id")["class_name"].unique()

# Count how many images each class appears in
class_image_counts = unique_class_per_image.explode().value_counts().sort_values(ascending=False)

# Convert to DataFrame with class_id
class_name_to_id = train_df.drop_duplicates("class_name")[["class_name", "class_id"]].set_index("class_name")["class_id"].to_dict()

# Final DataFrame
class_image_stats = pd.DataFrame({
    "class_name": class_image_counts.index,
    "class_id": [class_name_to_id[c] for c in class_image_counts.index],
    "num_images": class_image_counts.values
}).reset_index(drop=True)



# Total bounding boxes per class
class_counts = train_df['class_name'].value_counts().sort_values(ascending=False)

# Map class_name to class_id
class_name_to_id = train_df.drop_duplicates("class_name")[["class_name", "class_id"]].set_index("class_name")["class_id"].to_dict()

# Create final DataFrame
class_box_distribution = pd.DataFrame({
    "class_name": class_counts.index,
    "class_id": [class_name_to_id[cls] for cls in class_counts.index],
    "num_boxes": class_counts.values
}).reset_index(drop=True)



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


from collections import Counter  # Add this import
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


# ----------------------------------------------
# ğŸ“¦ Import necessary libraries
# ----------------------------------------------
import os
import cv2
import random
import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from glob import glob
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

# ----------------------------------------------
# ğŸ“¦ Split original data BEFORE augmentation
# ----------------------------------------------

# Assuming your original dataframe is `train_df_wbf`
balanced_df_multi = train_df_wbf.groupby('image_id')['new_class_id'].agg(lambda x: list(set(x))).reset_index()
train_df_wbf = train_df_wbf.merge(balanced_df_multi, on='image_id', suffixes=("", "_multi"))
train_df_wbf = train_df_wbf.loc[:, ~train_df_wbf.columns.duplicated()]
train_df_wbf['multi_class_str'] = train_df_wbf['new_class_id_multi'].apply(lambda x: str(sorted(x)))

sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)

for train_idx, val_idx in sgkf.split(train_df_wbf, train_df_wbf['multi_class_str'], groups=train_df_wbf['image_id']):
    train_df_split = train_df_wbf.iloc[train_idx].reset_index(drop=True)
    val_df_split = train_df_wbf.iloc[val_idx].reset_index(drop=True)
    break

# Drop extra columns not needed after split
train_df_split.drop(columns=['new_class_id_multi', 'multi_class_str'], inplace=True)
val_df_split.drop(columns=['new_class_id_multi', 'multi_class_str'], inplace=True)



def check_bboxes_validity(df):
    valid_bbox_count = 0
    invalid_bbox_count = 0

    valid_yolo_count = 0
    invalid_yolo_count = 0

    total = len(df)

    for _, row in df.iterrows():
        # YOLO-format components
        x_center, y_center = row['x_mid'], row['y_mid']
        box_width, box_height = row['w'], row['h']
        
        # Convert YOLO to corner format
        x_min = x_center - box_width / 2
        y_min = y_center - box_height / 2
        x_max = x_center + box_width / 2
        y_max = y_center + box_height / 2

        # Check corner format validity and normalized range
        if 0 <= x_min < x_max <= 1 and 0 <= y_min < y_max <= 1:
            valid_bbox_count += 1
        else:
            invalid_bbox_count += 1

        # Check YOLO format validity (size > 0)
        if box_width > 0 and box_height > 0:
            valid_yolo_count += 1
        else:
            invalid_yolo_count += 1

    # Print summary
    print(f"\nğŸ”� Total bounding boxes: {total}")
    print(f"âœ… Valid (corner format): {valid_bbox_count} ({valid_bbox_count / total:.2%})")
    print(f"â�Œ Invalid (corner format): {invalid_bbox_count} ({invalid_bbox_count / total:.2%})")
    print(f"âœ… Valid (YOLO format): {valid_yolo_count} ({valid_yolo_count / total:.2%})")
    print(f"â�Œ Invalid (YOLO format): {invalid_yolo_count} ({invalid_yolo_count / total:.2%})")

    # Optional warning
    if invalid_bbox_count > 0 or invalid_yolo_count > 0:
        print("âš ï¸� Warning: Some bounding boxes are invalid. Please review the data.")

# âœ… Use it
print("Checking bounding box validity...")
check_bboxes_validity(train_df)



def count_bboxes_formats(df):
    # Convert YOLO to corner format
    df = df.copy()
    df['x_min'] = df['x_mid'] - df['w'] / 2
    df['y_min'] = df['y_mid'] - df['h'] / 2
    df['x_max'] = df['x_mid'] + df['w'] / 2
    df['y_max'] = df['y_mid'] + df['h'] / 2

    # Check YOLO validity: all components must be finite and w, h > 0
    yolo_invalid = (
        df[['x_mid', 'y_mid', 'w', 'h']].isna().any(axis=1) |
        (df['w'] <= 0) | (df['h'] <= 0)
    )

    # Check Corner validity: x_max > x_min, y_max > y_min
    corner_invalid = (
        df[['x_min', 'y_min', 'x_max', 'y_max']].isna().any(axis=1) |
        (df['x_max'] <= df['x_min']) | (df['y_max'] <= df['y_min'])
    )

    # Count
    total = len(df)
    valid_yolo = (~yolo_invalid).sum()
    valid_corner = (~corner_invalid).sum()

    print(f"ğŸ”� Total bounding boxes: {total}")
    print(f"âœ… Valid YOLO format: {valid_yolo} ({valid_yolo / total:.2%})")
    print(f"â�Œ Invalid YOLO format: {total - valid_yolo} ({(total - valid_yolo) / total:.2%})")
    print(f"âœ… Valid Corner format: {valid_corner} ({valid_corner / total:.2%})")
    print(f"â�Œ Invalid Corner format: {total - valid_corner} ({(total - valid_corner) / total:.2%})")

# âœ… Run the check
count_bboxes_formats(train_df)



import os
import cv2
import shutil
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from glob import glob
import albumentations as A

# ----------------------------------------------
# ğŸ“¦ Fix paths for 'chestxrayabnormalities' images
# ----------------------------------------------
def fix_chestxrayabnormalities_paths(df, base_dir):
    df = df.copy()
    mask = df['source_dataset'] == 'chestxrayabnormalities'

    image_files = glob(os.path.join(base_dir, '*.jpg'))
    stem_to_path = {Path(f).stem.split('_jpg')[0]: f for f in image_files}

    def correct_path(row):
        if row['source_dataset'] != 'chestxrayabnormalities':
            return row['image_path']
        image_id = Path(row['image_path']).stem
        return stem_to_path.get(image_id, row['image_path'])  # fallback

    df.loc[mask, 'image_path'] = df[mask].apply(correct_path, axis=1)
    return df

# ----------------------------------------------
# ğŸ“¦ Function to define augmentations
# ----------------------------------------------
def get_chest_xray_augmentations():
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.RandomGamma(gamma_limit=(95, 105), p=0.5),
        A.Rotate(limit=5, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.05, rotate_limit=3, p=0.5, border_mode=0),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.GaussNoise(var_limit=(5.0, 10.0), p=0.2),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# ----------------------------------------------
# ğŸ“¦ Apply augmentation
# ----------------------------------------------
def apply_augmentation(image, bboxes, labels):
    transform = get_chest_xray_augmentations()
    augmented = transform(image=image, bboxes=bboxes, labels=labels)
    return augmented['image'], augmented['bboxes']

# ----------------------------------------------
# ğŸ“¦ Identify low-frequency classes
# ----------------------------------------------
def identify_low_freq_classes(df, moderate_threshold=0.165, strong_threshold=0.1):
    class_counts = df['new_class_id'].value_counts(normalize=True)
    strong_freq_classes = class_counts[class_counts < strong_threshold].index.tolist()
    moderate_freq_classes = class_counts[(class_counts >= strong_threshold) & (class_counts < moderate_threshold)].index.tolist()

    print(f"Strong frequency classes (<{strong_threshold}): {strong_freq_classes}")
    print(f"Moderate frequency classes (<{moderate_threshold}): {moderate_freq_classes}")
    return moderate_freq_classes, strong_freq_classes

# ----------------------------------------------
# ğŸ“¦ Save one image
# ----------------------------------------------
def save_image(save_path, image):
    cv2.imwrite(save_path, image)

# ----------------------------------------------
# ğŸ“¦ Augment and balance training data
# ----------------------------------------------
def augment_and_balance_data(train_df, save_augmented_dir='augmented_images', target_image_count=3000):
    train_df = train_df[train_df['source_dataset'] != 'chestxrayabnormalities'].copy()
    augmented_data = []
    augmented_image_ids = set()

    os.makedirs(save_augmented_dir, exist_ok=True)

    class_counts = train_df['new_class_id'].value_counts()
    print(f"[INFO] Class distribution before augmentation:")
    print(class_counts)

    temp_df = train_df.copy()
    print(f"[INFO] Targeting {target_image_count} images per class.")

    for class_id in tqdm(class_counts.index, desc="Augmenting Classes"):
        current_count = (temp_df['new_class_id'] == class_id).sum()
        if current_count >= target_image_count:
            continue

        augment_needed = target_image_count - current_count
        class_group = train_df[train_df['new_class_id'] == class_id]
        sampled_group = class_group.sample(n=augment_needed, replace=True, random_state=42)

        for idx, (image_id, group) in enumerate(sampled_group.groupby('image_id')):
            image_path = group.iloc[0]['image_path']
            image = cv2.imread(image_path)
            if image is None:
                print(f"[WARN] Failed to load image {image_path}. Skipping.")
                continue

            h_img, w_img = image.shape[:2]
            bboxes, labels = [], []

            for _, row in group.iterrows():
                box_width, box_height = row['w'] * w_img, row['h'] * h_img
                x_center, y_center = row['x_mid'] * w_img, row['y_mid'] * h_img
                x_min = x_center - box_width / 2
                y_min = y_center - box_height / 2
                x_max = x_center + box_width / 2
                y_max = y_center + box_height / 2

                if x_min >= x_max or y_min >= y_max or box_width <= 0 or box_height <= 0:
                    continue

                bboxes.append((x_min, y_min, x_max, y_max))
                labels.append(row['new_class_id'])

            if not bboxes:
                continue

            try:
                augmented_image, augmented_bboxes = apply_augmentation(image.copy(), bboxes, labels)
            except Exception as e:
                print(f"[ERROR] Augmentation failed for {image_id}: {e}")
                continue

            h_aug, w_aug = augmented_image.shape[:2]
            augmented_image_id = f"{image_id}_aug_{len(augmented_image_ids)}"

            if augmented_image_id not in augmented_image_ids:
                augmented_image_ids.add(augmented_image_id)
                save_path = os.path.join(save_augmented_dir, f"{augmented_image_id}.jpg")
                save_image(save_path, augmented_image)

                for (x_min, y_min, x_max, y_max), class_id_aug in zip(augmented_bboxes, labels):
                    x_min = max(0, min(x_min, w_aug))
                    x_max = max(0, min(x_max, w_aug))
                    y_min = max(0, min(y_min, h_aug))
                    y_max = max(0, min(y_max, h_aug))

                    if x_max <= x_min or y_max <= y_min:
                        continue

                    x_center = ((x_min + x_max) / 2) / w_aug
                    y_center = ((y_min + y_max) / 2) / h_aug
                    width = (x_max - x_min) / w_aug
                    height = (y_max - y_min) / h_aug

                    augmented_data.append({
                        'image_id': augmented_image_id,
                        'image_path': save_path,
                        'new_class_id': class_id_aug,
                        'x_mid': x_center,
                        'y_mid': y_center,
                        'w': width,
                        'h': height
                    })

    print("[âœ…] Augmented images saved!")

    augmented_df = pd.DataFrame(augmented_data)
    balanced_df = pd.concat([train_df, augmented_df], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"[âœ…] Balancing complete. Class distribution:")
    print(balanced_df['new_class_id'].value_counts())

    return balanced_df

# ----------------------------------------------
# ğŸ“¦ Prepare YOLO labels
# ----------------------------------------------
def prepare_yolo_labels(df, image_dest_dir, label_dest_dir):
    df = df.drop_duplicates(subset=['image_id', 'new_class_id', 'x_mid', 'y_mid', 'w', 'h'])
    os.makedirs(image_dest_dir, exist_ok=True)
    os.makedirs(label_dest_dir, exist_ok=True)

    for image_id, group in tqdm(df.groupby('image_id'), desc=f"Processing {image_dest_dir}"):
        image_path = group.iloc[0]['image_path']
        image_name = os.path.basename(image_path)
        label_file = Path(label_dest_dir) / f"{Path(image_name).stem}.txt"
        image_target = Path(image_dest_dir) / image_name

        if not image_target.exists():
            shutil.copy(image_path, image_target)

        group_sorted = group.sort_values(by='new_class_id')

        with open(label_file, 'w') as f:
            for _, row in group_sorted.iterrows():
                f.write(f"{row['new_class_id']} {row['x_mid']:.6f} {row['y_mid']:.6f} {row['w']:.6f} {row['h']:.6f}\n")

# ----------------------------------------------
# ğŸ“¦ Deduplicate YOLO labels
# ----------------------------------------------
def deduplicate_yolo_labels(label_dir):
    label_paths = glob(os.path.join(label_dir, "*.txt"))
    total_files = len(label_paths)
    deduplicated_count = 0

    for path in label_paths:
        with open(path, 'r') as f:
            lines = f.readlines()
        original_count = len(lines)
        deduped = list(set([line.strip() for line in lines]))

        if len(deduped) < original_count:
            deduplicated_count += 1
            with open(path, 'w') as f:
                f.write('\n'.join(deduped) + '\n')

    print(f"âœ… Deduplicated {deduplicated_count}/{total_files} files in: {label_dir}")

# ----------------------------------------------
# ğŸ“¦ Full Processing Flow
# ----------------------------------------------
# Assuming train_df_split and val_df_split are already defined

# 1. ğŸ©º Fix validation image paths (for chestxrayabnormalities)
val_df_split = fix_chestxrayabnormalities_paths(val_df_split, '/kaggle/input/chestxrayabnormalities/train/images')

# 2. ğŸ“ˆ Augment training set only
balanced_train_df = augment_and_balance_data(train_df_split)

# 3. ğŸ“„ Prepare YOLO labels
prepare_yolo_labels(balanced_train_df, 'data/images/train', 'data/labels/train')
prepare_yolo_labels(val_df_split, 'data/images/val', 'data/labels/val')

# 4. ğŸ§¹ Deduplicate labels
deduplicate_yolo_labels('data/labels/train')
deduplicate_yolo_labels('data/labels/val')

# 5. ğŸ§¼ Clean up temporary files
shutil.rmtree('augmented_images')


class_id_to_name = {
    0: "Aortic Enlargement",
    1: "Cardiomegaly",
    2: "Lung Collapse",
    3: "Nodule/Mass",
    4: "Opacities/Infiltration",
    5: "Pleural Conditions"
}

# Bounding box count per class
bbox_counts = balanced_train_df['new_class_id'].value_counts().sort_index()

# Image count per class
image_counts = balanced_train_df.groupby('new_class_id')['image_id'].nunique().sort_index()

# Combine both into a DataFrame
summary_df = pd.DataFrame({
    'class_id': bbox_counts.index,
    'class_name': [class_id_to_name[i] for i in bbox_counts.index],
    'bbox_count': bbox_counts.values,
    'unique_image_count': image_counts.values
})

print(summary_df)



import matplotlib.pyplot as plt
import random
from PIL import Image
from pathlib import Path

# Define the path to your augmented image directory
augmented_image_dir = Path('data/images/train')

# Get list of all .png image files
augmented_image_files = sorted([f for f in augmented_image_dir.glob('*') if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])

# Limit to available number of files
num_samples = min(5, len(augmented_image_files))
if num_samples == 0:
    print("No images found.")
else:
    # Optional: for reproducibility
    random.seed(42)
    sample_images = random.sample(augmented_image_files, num_samples)

    # Plot sampled images
    fig, axes = plt.subplots(1, num_samples, figsize=(3 * num_samples, 6))

    if num_samples == 1:
        axes = [axes]

    for ax, img_path in zip(axes, sample_images):
        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(img_path.name)

    plt.tight_layout()
    plt.show()



import cv2
import os

def check_channels(image_dir):
    for f in os.listdir(image_dir):
        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
            img = cv2.imread(os.path.join(image_dir, f))
            if img is not None and img.shape[2] != 3:
                print(f"{f} is not RGB")

check_channels('/kaggle/working/data/images/train')
check_channels('/kaggle/working/data/images/val')


import cv2
import os

def convert_grayscale_to_rgb_in_place(image_dir):
    """
    Convert grayscale or single-channel images to 3-channel RGB for YOLO compatibility.
    Overwrites the images in place.
    """
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total = len(image_files)
    converted = 0

    print(f"Processing {total} images in: {image_dir}")

    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(image_dir, image_file)
        img = cv2.imread(image_path)

        if img is None:
            print(f"âš ï¸� Failed to read: {image_file}")
            continue

        # Skip if image is already RGB (3 channels)
        if len(img.shape) == 3 and img.shape[2] == 3:
            continue

        # Convert grayscale to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(image_path, img_rgb)
        converted += 1

        if idx % 100 == 0 or idx == total:
            print(f"[{idx}/{total}] processed, {converted} converted", flush=True)

    print(f"âœ… Done. {converted} images converted in: {image_dir}\n")


# Directories
train_dir = '/kaggle/working/data/images/train'
val_dir = '/kaggle/working/data/images/val'

# Convert grayscale images to 3-channel RGB
convert_grayscale_to_rgb_in_place(train_dir)
convert_grayscale_to_rgb_in_place(val_dir)

# Class names for YOLO
class_names = [
    'Aortic Enlargement',
    'Cardiomegaly',
    'Lung Collapse',
    'Nodule/Mass',
    'Opacities/Infiltration',
    'Pleural Conditions'
]

# Correctly format the names
names_yaml = '\n'.join([f"  - {name}" for name in class_names])

# Create YOLO data.yaml dynamically
data_yaml = f"""train: {train_dir}
val: {val_dir}

nc: {len(class_names)}
names:
{names_yaml}
"""

# Save data.yaml
yaml_path = 'data.yaml'
try:
    with open(yaml_path, 'w') as f:
        f.write(data_yaml)
    print(f"âœ… data.yaml file created at {yaml_path}")
except Exception as e:
    print(f"â�Œ Failed to create data.yaml: {e}")



# âœ… Check available GPUs
num_gpus = torch.cuda.device_count()
print(f"Available GPUs: {num_gpus}")
for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# âœ… Set device for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} for training")

# âœ… Optimize CUDA memory allocation
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# âœ… Load YOLOv12-M model
model = YOLO("yolo12m.pt").to(device)

# âœ… Train the model
train_results = model.train(
    data="/kaggle/working/data.yaml",
    epochs=110,
    batch=22,
    imgsz=640,
    device="auto",     # This will still use GPU automatically
    half=True,
    amp=True,
    workers=5,
    cache=False,
    save=True,
    save_period=10,
    patience=15,
    project="yolov12-training",
    name="yolo12m-vinbigdata",
    exist_ok=True,
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


from IPython.display import FileLink
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





