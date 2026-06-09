import os
import re
import time
import datetime
from tqdm import tqdm
import seaborn as sns
import numpy as np
import glob
import json
import random
import yaml
from collections import defaultdict
from tqdm import tqdm
from typing import Any, Dict, List, Tuple, Union, Optional
from concurrent.futures import ThreadPoolExecutor

import timm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset
from torch.utils.data import Subset
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

import pandas as pd
import cv2
import csv

from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter

import pydicom as dicom
import matplotlib.patches as patches

import matplotlib.pyplot as plt
from matplotlib import animation, rc
from pathlib import Path
import pandas as pd

import pydicom as dicom 
import pydicom
from pydicom.pixel_data_handlers.util import (
    apply_voi_lut,
    apply_color_lut,
    apply_modality_lut,
)


# Reads the  data
data_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/'
train_data  = pd.read_csv(data_path + 'train.csv')
train_label = pd.read_csv(data_path + 'train_label_coordinates.csv')
train_description  = pd.read_csv(data_path + 'train_series_descriptions.csv')
test_description   = pd.read_csv(data_path + 'test_series_descriptions.csv')
submission         = pd.read_csv(data_path + 'sample_submission.csv')




train_data.head(5)


train_description.head(5)


train_label.head(5)


# Types of modalities
modalities = list(train_description.iloc[:,-1].unique())
modality_count = train_description['series_description'].value_counts()
plt.figure(figsize = (10,5))
sns.barplot(x=modality_count.index, y=modality_count.values, palette="Set1")
plt.xlabel("Modalities")
plt.ylabel("Count")
plt.title("Distribution of Modalities")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()



fig, ax = plt.subplots()
condition_count = train_label['condition'].value_counts()
ax.pie(condition_count, labels=condition_count.index, autopct='%1.1f%%')
plt.axis('equal') 
plt.suptitle('Distribution of Lumbar Spine Condition') 
plt.tight_layout()
plt.show()


# Identify all condition columns (excluding 'study_id' or other non-relevant columns)
condition_columns = [col for col in train_data.columns if col not in ['study_id']]
severity_counts = train_data[condition_columns].stack().value_counts()
plt.figure(figsize=(10,5))
sns.barplot(x=severity_counts.index, y=severity_counts.values, palette="Set1")
plt.xlabel("Severity Levels")
plt.ylabel("Count")
plt.title("Distribution of Severity Levels Across All Lumbar Spine Conditions")
plt.xticks(rotation=90)


# Count occurrences of severity levels per condition
severity_counts = train_data[condition_columns].apply(pd.Series.value_counts)

# Reshape for plotting
severity_counts = severity_counts.T  # Transpose for better readability
severity_counts = severity_counts.reset_index().melt(id_vars='index', var_name='Severity Level', value_name='Count')
severity_counts.rename(columns={'index': 'Condition'}, inplace=True)
severity_counts.sort_values('Condition')
severity_order = ["Normal/Mild", "Moderate", "Severe"]

# Convert the 'Severity Level' column to a categorical type with the defined order
severity_counts["Severity Level"] = pd.Categorical(severity_counts["Severity Level"], 
                                                   categories=severity_order, 
                                                   ordered=True)
# Sort the dataframe based on this order
severity_counts = severity_counts.sort_values(by=["Severity Level", "Condition"])


plt.figure(figsize=(15,7))

# Create grouped bar plot
sns.barplot(data=severity_counts, x="Condition", y="Count", hue="Severity Level", palette="Set1")

# Rotate x-axis labels for better readability
plt.xticks(rotation=90, ha='right')
# Add title and labels
plt.title("Severity Level Distribution for Each Lumbar Spine Condition")
plt.xlabel("Condition")
plt.ylabel("Count")
plt.legend(title="Severity Level")



# Easier alternative to plot the Severity Level Distribution
figure, axis = plt.subplots(1,3, figsize=(20,5))
for idx, condition in enumerate(['foraminal', 'subarticular', 'canal']):
    diagnosis = list(filter(lambda x: x.find(condition)!= -1, train_data.columns))
    filtered_dataframe_by_diagnosis = train_data[diagnosis]
    value_counts = filtered_dataframe_by_diagnosis.apply(filtered_dataframe_by_diagnosis.value_counts).T
    value_counts.plot(kind='bar',stacked=True, ax=axis[idx], cmap='Set1')


grouped_counts = train_data[condition_columns]

# Function to extract disease category (splitting by the second last '_')
def extract_disease_category(col_name):
    parts = col_name.split('_')
    if len(parts) > 2:
        return ' '.join(parts[:-2]).title() 
    return col_name  

# Create a mapping of condition columns to disease categories
column_category_mapping ={col: extract_disease_category(col) for col in condition_columns}

# Group columns by disease category
grouped_severity_counts = {}
for category in column_category_mapping.values():
    category_columns = [col for col in condition_columns if column_category_mapping[col] == category]
    grouped_data = grouped_counts[category_columns].apply(pd.Series.value_counts).apply(pd.Series.sum, axis=1)
    grouped_severity_counts[category] =  grouped_data
 
grouped_severity_counts_df = pd.DataFrame(grouped_severity_counts)
grouped_severity_counts_df = grouped_severity_counts_df.reset_index().melt(id_vars='index', var_name='Disease Category', value_name='Count')
grouped_severity_counts_df.rename(columns={'index': 'Severity Level'}, inplace=True)

grouped_severity_counts_df["Severity Level"] = pd.Categorical(grouped_severity_counts_df["Severity Level"], 
                                                   categories=severity_order, 
                                                   ordered=True)
# Sort the dataframe based on this order
grouped_severity_counts_df = grouped_severity_counts_df.sort_values(by=["Severity Level"])

# Plot grouped dataframe
sns.barplot(data=grouped_severity_counts_df, x="Disease Category", y="Count", hue="Severity Level", palette="Set1")

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
# Add title and labels
plt.title("Severity Level Distribution for Each Lumbar Spine Condition")
plt.xlabel("Condition")
plt.ylabel("Count")
plt.legend(title="Severity Level") 


# Create the count plot
disk_level_counts = train_label['level'].value_counts()

sns.barplot(x=disk_level_counts.index, y=disk_level_counts.values, palette="Set1")
plt.xlabel("Disk Levels")
plt.ylabel("Count")
plt.title("Distribution of Disk Levels Across All Lumbar Spine Conditions")


def visualize_condition_counts(df, title, path=None):
    
    """
    Visualize condition counts on training set
    
    Parameters
    ----------
    df: pandas.DataFrame
        Counts and percentages of conditions
        
    title: str
        Title of the plot
        
    path: str, pathlib.Path or None
        Path of the output file (if path is None, plot is displayed with selected backend)
    """
    
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(
        y=np.arange(df.shape[0] // 3) - 0.2,
        width=df['count'].values[0::3],
        height=0.2,
        align='center',
        label='Normal/Mild'
    )
    ax.barh(
        y=np.arange(df.shape[0] // 3),
        width=df['count'].values[1::3],
        height=0.2,
        align='center',
        label='Moderate'
    )
    ax.barh(
        y=np.arange(df.shape[0] // 3) + 0.2,
        width=df['count'].values[2::3],
        height=0.2,
        align='center',
        label='Severe'
    )
    ax.set_yticks(np.arange(df.shape[0] // 3))
    ax.set_yticklabels([
        f'{level}\nNormal Count: {normal_count} ({normal_percentage:.2f}%)\nModerate Count: {moderate_count} ({moderate_percentage:.2f}%)\nSevere Count: {severe_count} ({severe_percentage:.2f}%)' for level, normal_count, normal_percentage, moderate_count, moderate_percentage, severe_count, severe_percentage, in zip(
            df['Spine Level'].values[0::3],
            df['count'].values[0::3],
            df['percentage'].values[0::3],
            df['count'].values[1::3],
            df['percentage'].values[1::3],
            df['count'].values[2::3],
            df['percentage'].values[2::3],
        )
    ])
    ax.set_xlabel('')
    ax.tick_params(axis='x', pad=10)
    ax.tick_params(axis='y')
    ax.set_title(title, pad=15)
    ax.legend(loc='best')
    plt.gca().invert_yaxis()

    plt.show()

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


# Get unique disease categories
disease_categories = sorted(set(column_category_mapping.values()))

# Dictionary to store processed severity data
all_conditions_data = {}

for category in disease_categories:
    # Select columns belonging to this disease category
    category_columns = [col for col in condition_columns if column_category_mapping[col] == category]

    # Create a DataFrame with severity levels mapped to spine levels
    df_condition = train_data[category_columns].copy()

    # Convert to long format (stacked)
    df_condition = df_condition.stack().reset_index()
   
    df_condition = df_condition.rename(columns={0: 'Severity', 'level_1': 'Spine Level'})
    # Extract spine level (last two parts of column name)
    df_condition['Spine Level'] = df_condition['Spine Level'].apply(lambda x: ' '.join(x.split('_')[-2:]).title())

    # Count occurrences
    df_condition_counts = df_condition.groupby(['Spine Level', 'Severity']).size().reset_index(name='count')

    # Map severity to numerical values
    severity_map = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
    df_condition_counts['Severity Level'] = df_condition_counts['Severity'].map(severity_map)
    
    # Sort by Spine Level and Severity Level
    df_condition_counts = df_condition_counts.sort_values(by=['Spine Level', 'Severity Level'], ascending=True)

    # Compute percentages per level
    df_condition_counts['percentage'] = df_condition_counts['count'] / df_condition_counts.groupby('Spine Level')['count'].transform('sum') * 100

    # Store processed data
    all_conditions_data[category] = df_condition_counts



# Visualize each condition separately
for category, df_counts in all_conditions_data.items():
    visualize_condition_counts(
        df=df_counts,
        title=f"{category} Counts By Spine Level"
    )




def generate_image_paths(data_frame, data_dir):
    image_paths = []
    for study_id, series_id in zip(data_frame['study_id'], data_frame['series_id']):
        study_dir = os.path.join(data_dir, str(study_id))
        series_dir = os.path.join(study_dir, str(series_id))
        image_paths.extend([os.path.join(series_dir, img) for img in os.listdir(series_dir)])
    return image_paths

train_image_paths = generate_image_paths(train_description, os.path.join(data_path, "train_images"))


def display_dicom_images(image_paths):
    plt.figure(figsize=(15, 5))  # Adjust figure size if needed
    for i, path in enumerate(image_paths[:3]):
        ds = pydicom.dcmread(path)
        plt.subplot(1, 3, i+1)
        plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
        plt.title(f"Image {i+1}")
        plt.axis('off')
    plt.show()
display_dicom_images(train_image_paths)


def load_dicom_files(path_to_folder):
    # Filter and sort DICOM files based on numeric part extracted from filename.
    dicom_filenames = sorted(
        [f for f in os.listdir(path_to_folder) if f.endswith('.dcm')],
        key=lambda f: int(os.path.splitext(f)[0].split('-')[-1])
    )
    return [os.path.join(path_to_folder, f) for f in dicom_filenames]

def load_first_dicom(series_folder_path):
    dicom_files = load_dicom_files(series_folder_path)
    return dicom_files[0] if dicom_files else None


# ---------------------------
# Helper Functions
# ---------------------------

def iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two boxes.
    Each box is defined as a tuple: (x1, y1, x2, y2)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

def cluster_boxes_representative(centers, box_size, iou_threshold=0.3):
    """
    Cluster bounding boxes (given by their center coordinates) that are close together.
    Instead of averaging, choose one representative center (the first in the cluster).
    
    Parameters:
      centers: list of tuples (center_x, center_y)
      box_size: tuple (width, height) for the boxes
      iou_threshold: threshold for clustering boxes
    
    Returns:
      A list of representative center coordinates (from the original annotations).
    """
    if not centers:
        return []
    
    w, h = box_size
    # Create boxes for each center.
    boxes = [(cx - w/2, cy - h/2, cx + w/2, cy + h/2) for cx, cy in centers]
    
    clusters = []  # each cluster is a list of indices
    used = set()
    
    for i in range(len(boxes)):
        if i in used:
            continue
        cluster = [i]
        used.add(i)
        for j in range(i+1, len(boxes)):
            if j in used:
                continue
            if iou(boxes[i], boxes[j]) >= iou_threshold:
                cluster.append(j)
                used.add(j)
        clusters.append(cluster)
    
    # Choose the first center from each cluster as the representative.
    representative_centers = [centers[cluster[0]] for cluster in clusters]
    return representative_centers

# ---------------------------
# Display Functions
# ---------------------------

def display_dicom_for_condition(image, study_id, series_id, cond, rows, box_size=(50,50), iou_threshold=0.3, color='red'):
    """
    Plot a single image with bounding boxes for a given condition.
    Uses the representative annotation from each cluster.
    """
    # Extract centers for the current condition and cluster them.
    centers = [(row['x'], row['y']) for row in rows]
    rep_centers = cluster_boxes_representative(centers, box_size, iou_threshold=iou_threshold)
    
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(image, cmap='gray')
    ax.set_title(f"Study: {study_id}, Series: {series_id} - {cond}")
    ax.axis('off')
    
    for center_x, center_y in rep_centers:
        # Compute top-left of the bounding box from the representative center.
        top_left = (center_x - box_size[0]/2, center_y - box_size[1]/2)
        rect = patches.Rectangle(top_left, box_size[0], box_size[1],
                                 linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(center_x, top_left[1] - 5, cond, color=color, fontsize=8, ha='center')
    
    plt.tight_layout()
    plt.show()

def display_dicom_with_conditions(image_paths, label_df, box_size=(50,50), iou_threshold=0.3):
    """
    For each image, if multiple conditions are present, duplicate the image and plot one condition per subplot.
    """
    # Build mapping from (study_id, series_id) to annotations.
    label_dict = {}
    for _, row in label_df.iterrows():
        key = (int(row['study_id']), int(row['series_id']))
        label_dict.setdefault(key, []).append(row)
    
    # Define colors for conditions.
    available_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    condition_colors = {}
    for cond in label_df['condition'].unique():
        condition_colors[cond] = available_colors[len(condition_colors) % len(available_colors)]
    
    def read_dicom(path):
        ds = pydicom.dcmread(path)
        return ds.pixel_array

    # Load DICOM images concurrently.
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_path = {executor.submit(read_dicom, path): path for path in image_paths}
        images = {}
        for future in future_to_path:
            path = future_to_path[future]
            try:
                images[path] = future.result()
            except Exception as exc:
                print(f"Error reading {path}: {exc}")
                images[path] = None

    # Process each image.
    for path in image_paths:
        parts = path.split(os.sep)
        try:
            study_id = int(parts[-3])
            series_id = int(parts[-2])
        except (IndexError, ValueError):
            print(f"Unable to parse IDs from path: {path}")
            continue
        
        image = images.get(path)
        if image is None:
            continue
        
        # Get all annotations for this image.
        annotations = label_dict.get((study_id, series_id), [])
        if annotations:
            # Group annotations by condition.
            condition_groups = {}
            for row in annotations:
                cond = row['condition']
                condition_groups.setdefault(cond, []).append(row)
            
            # For each condition, duplicate the image and plot only that condition.
            for cond, rows in condition_groups.items():
                color = condition_colors.get(cond, 'red')
                display_dicom_for_condition(image, study_id, series_id, cond, rows,
                                            box_size=box_size, iou_threshold=iou_threshold, color=color)
        else:
            # If no annotations, just show the image.
            fig, ax = plt.subplots(figsize=(6,6))
            ax.imshow(image, cmap='gray')
            ax.set_title(f"Study: {study_id}, Series: {series_id}")
            ax.axis('off')
            plt.tight_layout()
            plt.show()



train_images_dir = os.path.join(data_path, 'train_images')
study_ids = [folder for folder in os.listdir(train_images_dir) if os.path.isdir(os.path.join(train_images_dir, folder))]
selected_study_id = random.choice(study_ids)
print(f"Selected study_id: {selected_study_id}")

# Build study folder path for the selected study.
study_folder = os.path.join(train_images_dir, selected_study_id)

# List all series folders within the selected study folder.
series_folders = [os.path.join(study_folder, sf) for sf in os.listdir(study_folder)
                  if os.path.isdir(os.path.join(study_folder, sf))]

# Use ThreadPoolExecutor to get the first DICOM file per series in parallel.
with ThreadPoolExecutor(max_workers=4) as executor:
    first_dicom_futures = [executor.submit(load_first_dicom, series_folder) for series_folder in series_folders]
    image_paths = [f.result() for f in first_dicom_futures if f.result() is not None]

# Now display the images with separate plots per condition.
display_dicom_with_conditions(image_paths, train_label, box_size=(25,25), iou_threshold=0.3)



# Merge the two dataframes on 'study_id' and 'series_id'
merged_csv = pd.merge(train_label, train_description[['study_id', 'series_id', 'series_description']], 
                      on=['study_id', 'series_id'], how='left')
merged_csv


# We want to obtain three different csv files for each condition
cleaned_data = merged_csv.copy()
# Define conditions for each group
condition_groups = {
    'Spinal Canal Stenosis': ['Spinal Canal Stenosis'],
    'Neural Foraminal Narrowing': ['Right Neural Foraminal Narrowing', 'Left Neural Foraminal Narrowing'],
    'Subarticular Stenosis': ['Right Subarticular Stenosis', 'Left Subarticular Stenosis']
}

# Split and save to separate CSV files
for group_name, conditions in condition_groups.items():
    # Filter rows based on condition
    filtered_df = cleaned_data[cleaned_data['condition'].isin(conditions)]
    # Save to new CSV file
    group_name_save=group_name.replace(' ','_')
    filtered_df.to_csv(f'{group_name_save}.csv', index=False)

print("CSV files have been split and saved.")


# -----------------------------------------------------------
# 1. Helper function to perform an 80/20 stratified split
# -----------------------------------------------------------
def train_val_split(csv_file, results_name):
    # Load CSV
    df = pd.read_csv(csv_file)
    
    # Create a unique label from condition and level and assign numeric codes
    df['condition_level'] = df['condition'] + '_' + df['level']
    df['class_id'] = df['condition_level'].astype('category').cat.codes

    # Perform an 80/20 split (stratified by class_id)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['class_id'], random_state=42)
    
    # Save the splits (if desired)
    train_df.to_csv(f'{results_name}_train.csv', index=False)
    val_df.to_csv(f'{results_name}_val.csv', index=False)
    
    print(f"Data split: train ({len(train_df)}) and val ({len(val_df)})")
    return train_df, val_df

# -----------------------------------------------------------
# 2. Updated Data Preparation Class for ML pipelines
# -----------------------------------------------------------
class DetectorDataPreparation:
    def __init__(
        self,
        dataset_directory='../Data/train_images',
        csv_directory='',
        condition_level_classes={},
        condition_name='',
        width_box=16
    ):
        self.dataset_directory = dataset_directory
        self.csv_directory = csv_directory
        self.condition_level_classes = condition_level_classes
        self.condition_name = condition_name
        self.width_box = width_box
        
        # Define a directory to save processed data
        self.save_directory = f'./{self.condition_name}'
        
        # Create folder structure
        self.create_folder()
        
        # Read CSV and perform an 80/20 trainâ€“validation split
        self.read_train_val_split()
        
        # Process training data: convert DICOM to PNG, save HW info, merge, then create YOLO labels
        self.dicom_to_png(self.training_data, self.train_image_path)
        self.training_data = self.save_height_width_to_csv(self.training_data, data_type='train')
        self.create_label_for_yolo(self.training_data, self.train_labels_path)
        
        # Process validation data
        self.dicom_to_png(self.validation_data, self.val_images_path)
        self.validation_data = self.save_height_width_to_csv(self.validation_data, data_type='val')
        self.create_label_for_yolo(self.validation_data, self.val_labels_path)
        
        # Create a YAML config file for YOLO training
        self.create_yaml_file()

    def create_folder(self):
        # Base folder
        base_path = Path(self.save_directory)
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Dataset folder and subfolders for training and validation (images and labels)
        self.dataset_path = base_path / 'datasets'
        (self.dataset_path / 'train/images').mkdir(parents=True, exist_ok=True)
        (self.dataset_path / 'train/labels').mkdir(parents=True, exist_ok=True)
        (self.dataset_path / 'val/images').mkdir(parents=True, exist_ok=True)
        (self.dataset_path / 'val/labels').mkdir(parents=True, exist_ok=True)
        
        self.train_image_path = self.dataset_path / 'train/images'
        self.train_labels_path = self.dataset_path / 'train/labels'
        self.val_images_path = self.dataset_path / 'val/images'
        self.val_labels_path = self.dataset_path / 'val/labels'
    
    def read_train_val_split(self):
        df = pd.read_csv(self.csv_directory)
        # Create a unique class from condition and level
        df['condition_level'] = df['condition'] + '_' + df['level']
        df['class_id'] = df['condition_level'].astype('category').cat.codes

        # 80/20 stratified split based on the numeric class labels
        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['class_id'], random_state=42)
        train_df['split'] = 'train'
        val_df['split'] = 'val'
        
        self.training_data = train_df.copy()
        self.validation_data = val_df.copy()
        
        # Save split files (optional)
        train_df.to_csv(f'{self.save_directory}/train_split.csv', index=False)
        val_df.to_csv(f'{self.save_directory}/val_split.csv', index=False)
        print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")
    
    def read_dicom(self, dicom_path):
        # Read a DICOM file and convert to a 3-channel image (normalized to 0-255)
        ds = pydicom.dcmread(dicom_path)
        image = ds.pixel_array.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255
        image = np.stack([image] * 3, axis=-1).astype('uint8')
        return image
    
    def dicom_to_png(self, df, image_directory):
        # Reset the list for height and width info for each split
        self.height_width_info = []
        for _, row in df.iterrows():
            study_id = row['study_id']
            series_id = row['series_id']
            instance = row['instance_number']
            dcm_path = f'{self.dataset_directory}/{study_id}/{series_id}/{instance}.dcm'
            try:
                dcm_image = self.read_dicom(dcm_path)
            except Exception as e:
                print(f"Error reading DICOM {dcm_path}: {e}")
                continue
            height, width, _ = dcm_image.shape
            self.height_width_info.append({
                'study_id': study_id,
                'series_id': series_id,
                'instance_number': instance,
                'height': height,
                'width': width
            })
            png_filename = f'{study_id}_{series_id}_{instance}.png'
            cv2.imwrite(str(Path(image_directory) / png_filename), dcm_image)
    
    def save_height_width_to_csv(self, df, data_type='train'):
        # Save the height/width info extracted during DICOM conversion
        csv_file = f'{self.save_directory}/{self.condition_name}_height_width_{data_type}.csv'
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.height_width_info[0].keys())
            writer.writeheader()
            writer.writerows(self.height_width_info)
        print(f"Height/width data saved to {csv_file}")
        
        # Merge the height/width info with the original dataframe on study_id, series_id, and instance_number
        hw_df = pd.read_csv(csv_file)
        merged_data = pd.merge(df, hw_df, on=['study_id', 'series_id', 'instance_number'], how='left')
        merged_csv = f'{self.save_directory}/{self.condition_name}_merged_{data_type}.csv'
        merged_data.to_csv(merged_csv, index=False)
        print(f"Merged data saved to {merged_csv}")
        
        # Return the merged DataFrame for later use
        return merged_data
    
    def find_class_label(self, condition, level):
        # Construct the key exactly as defined in condition_level_classes
        key = f"{condition.replace(' ', '_')}_{level.replace('/', '_')}"
        return self.condition_level_classes.get(key, -1)
    
    def create_label_for_yolo(self, df, labels_directory):
        # For each row (i.e. each instance), write a YOLO label file
        for _, row in df.iterrows():
            # Ensure that the merged data has height/width information
            if 'height' not in row or 'width' not in row:
                print(f"Missing height/width for row: {row}")
                continue
            study_id = row['study_id']
            series_id = row['series_id']
            instance = row['instance_number']
            height = row['height']
            width = row['width']
            condition = row['condition']
            level = row['level']
            # Expecting x and y columns to be present in your CSV
            x = row['x']
            y = row['y']
            class_id = self.find_class_label(condition, level)
            
            # Normalize the x, y, width and height for YOLO (assuming width_box defines a fixed box size)
            norm_x = float(x) / width
            norm_y = float(y) / height
            norm_w = float(self.width_box) / width
            norm_h = float(self.width_box) / height
            
            label_filename = f'{study_id}_{series_id}_{instance}.txt'
            with open(str(Path(labels_directory) / label_filename), 'w') as f:
                f.write(f"{class_id} {norm_x} {norm_y} {norm_w} {norm_h}\n")
    
    def create_yaml_file(self):
        # Create a YAML file for YOLO that contains paths, number of classes, and class names
        yaml_file_path = f'{self.dataset_path}/yolo_config.yaml'
        num_classes = len(self.condition_level_classes)
        yaml_data = {
            'train': './train',
            'val': './val',
            'nc': num_classes,
            'names': list(self.condition_level_classes.keys())
        }
        with open(yaml_file_path, 'w') as file:
            yaml.dump(yaml_data, file, default_flow_style=False)
        print(f"YAML config saved to {yaml_file_path}")




condition_level_classes_spinal_canal = {
    'Spinal_Canal_Stenosis_L1_L2': 0,
    'Spinal_Canal_Stenosis_L2_L3': 1,
    'Spinal_Canal_Stenosis_L3_L4': 2,
    'Spinal_Canal_Stenosis_L4_L5': 3,
    'Spinal_Canal_Stenosis_L5_S1': 4,
}

dataset_dir = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images'
csv_path = './Spinal_Canal_Stenosis.csv' 

# Instantiate the data preparation class.
prep = DetectorDataPreparation(
    dataset_directory=dataset_dir,
    csv_directory=csv_path,
    condition_level_classes=condition_level_classes_spinal_canal,
    condition_name='Spinal_Canal_Stenosis',
    width_box=16
)



!pip install ultralytics


from ultralytics import YOLO  # âœ… required import


class YOLOTraining:
    def __init__(self,
                 data_directory,
                 condition,
                 results_directory,
                 epochs=500,
                 patience=20,
                 batch=4):
        self.condition = condition
        self.results_directory = results_directory
        self.result_condition_directory = f'{self.results_directory}/{self.condition}'
        self.data_directory = data_directory
        self.epochs = epochs
        self.patience = patience
        self.batch = batch

        # The YAML configuration is now under the datasets folder (80/20 split)
        self.yolo_config_yaml = f'{self.data_directory}/{self.condition}/datasets/yolo_config.yaml'
        # Load a pretrained YOLO model
        model = self.load_pretrain_model()

        # Start training
        self.results = self.training(model)

    def load_pretrain_model(self):
        # Build a new model from the model configuration YAML and load pretrained weights
        # model = YOLO('yolov8n.yaml')  # build a new model from YAML
        model = YOLO('yolo11n.pt')    # load pretrained weights
        return model

    def training(self, model):
        # Ensure the results folder exists
        result_condition_path = Path(self.result_condition_directory)
        if not result_condition_path.exists():
            result_condition_path.mkdir(parents=True, exist_ok=True)

        # Define a name for this training run
        name = f"epochs-{self.epochs}_batch-{self.batch}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"


        # Train the model using the YAML configuration for the 80/20 split data
        results = model.train(data=self.yolo_config_yaml,
                              project=result_condition_path,
                              name=name,
                              epochs=self.epochs,
                              patience=self.patience,
                              batch=self.batch)
        print("Training complete!")
        return results





# Train Yolo Model for each condition separately
condition = 'Spinal_Canal_Stenosis'       
results_directory = './results_spinal_canal_stenosis'          
epochs = 1
patience = 20
batch = 4


# Instantiate the YOLO training class with the provided arguments
result = YOLOTraining(data_directory=" /kaggle/working",
             condition=condition,
             results_directory=results_directory,
             epochs=epochs,
             patience=patience,
             batch=batch)


# Access specific metrics
metrics = result.metrics  # .metrics is a dict!

print(metrics)


metrics = result.results  # .metrics is a dict!

print(metrics)


print(result.yolo_config_yaml)


# Subarticular Stenosis
condition_level_classes_subarticular = {
    'Left_Subarticular_Stenosis_L1_L2': 0,
    'Left_Subarticular_Stenosis_L2_L3': 1,
    'Left_Subarticular_Stenosis_L3_L4': 2,
    'Left_Subarticular_Stenosis_L4_L5': 3,
    'Left_Subarticular_Stenosis_L5_S1': 4,
    'Right_Subarticular_Stenosis_L1_L2': 5,
    'Right_Subarticular_Stenosis_L2_L3': 6,
    'Right_Subarticular_Stenosis_L3_L4': 7,
    'Right_Subarticular_Stenosis_L4_L5': 8,
    'Right_Subarticular_Stenosis_L5_S1': 9,
}
csv_path = './Subarticular_Stenosis.csv' 

DetectorDataPreparation(
    dataset_directory=dataset_dir,
    csv_directory=csv_path,
    condition_level_classes=condition_level_classes_spinal_canal,
    condition_name='Subarticular_Stenosis',
    width_box=16
)


condition = 'Subarticular_Stenosis'       
results_directory = './results_subarticular_stenosis'          
epochs = 1
patience = 20
batch = 4

results = YOLOTraining(data_directory=" /kaggle/working",
             condition=condition,
             results_directory=results_directory,
             epochs=epochs,
             patience=patience,
             batch=batch)

# 'results' now holds the metrics!
print(results)  # Will show basic info

# Access specific metrics
metrics = results.metrics  # .metrics is a dict!

print("mAP50-95:", metrics['mAP50-95'])
print("mAP50:", metrics['mAP50'])
print("Precision:", metrics['precision'])
print("Recall:", metrics['recall'])
print('Subarticular Stenosis DONE')



!git clone https://github.com/ultralytics/yolov5


%cd ..



!ls 


import os
os.environ['WANDB_DISABLED'] = 'true'


!python yolov5/train.py --img 640 --batch 16 --epochs 50 --data Spinal_Canal_Stenosis/datasets/yolo_config.yaml --weights yolov5s.pt --name lumbar_yolov5



# Neural Foraminal Narrowing
condition_level_classes_neural = {
    'Left_Neural_Foraminal_Narrowing_L1_L2': 0,
    'Left_Neural_Foraminal_Narrowing_L2_L3': 1,
    'Left_Neural_Foraminal_Narrowing_L3_L4': 2,
    'Left_Neural_Foraminal_Narrowing_L4_L5': 3,
    'Left_Neural_Foraminal_Narrowing_L5_S1': 4,
    'Right_Neural_Foraminal_Narrowing_L1_L2': 5,
    'Right_Neural_Foraminal_Narrowing_L2_L3': 6,
    'Right_Neural_Foraminal_Narrowing_L3_L4': 7,
    'Right_Neural_Foraminal_Narrowing_L4_L5': 8,
    'Right_Neural_Foraminal_Narrowing_L5_S1': 9,
}

DetectorDataPreparation(
    dataset_directory=dataset_dir,
    csv_directory=csv_path,
    condition_level_classes=condition_level_classes_spinal_canal,
    condition_name='Neural_Foraminal_Narrowing',
    width_box=16
)


condition = 'Neural_Foraminal_Narrowing'       
results_directory = './results_neural_foraminal_narrowing.csv'          
epochs = 1
patience = 20
batch = 4

YOLOTraining(data_directory=" /kaggle/working",
             condition=condition,
             results_directory=results_directory,
             epochs=epochs,
             patience=patience,
             batch=batch)

print('Neural Foraminal Narrowing Stenosis DONE')




# ðŸ“Œ 1. IMPORTS
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import pandas as pd
import numpy as np
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# ðŸ“Œ 2. DATASET CLASS
class LumbarSpineDataset(Dataset):
    def __init__(self, images_dir, labels_csv, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms
        
        self.labels = pd.read_csv(labels_csv)
        self.image_ids = self.labels['image_id'].unique()
        
        # Group labels by image for fast access
        self.image_boxes = self.labels.groupby('image_id')
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.images_dir, f"{image_id}.png")  # Assuming .png format
        img = Image.open(img_path).convert("RGB")
        
        records = self.image_boxes.get_group(image_id)
        
        boxes = records[['x_min', 'y_min', 'x_max', 'y_max']].values
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        labels = torch.ones((records.shape[0],), dtype=torch.int64)  # Assuming one class (e.g., "spine")
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])

        if self.transforms:
            img = self.transforms(img)
        
        else:
            img = F.to_tensor(img)

        return img, target

    def __len__(self):
        return len(self.image_ids)

# ðŸ“Œ 3. COLLATE FUNCTION
def collate_fn(batch):
    return tuple(zip(*batch))

# ðŸ“Œ 4. MODEL SETUP
def get_model(num_classes):
    # Load Faster R-CNN pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    
    # Get number of input features for classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

# ðŸ“Œ 5. HYPERPARAMETERS
BATCH_SIZE = 4
NUM_CLASSES = 2  # 1 class (spine) + background
NUM_EPOCHS = 10
LEARNING_RATE = 0.005

# ðŸ“Œ 6. PREPARE DATA
train_dataset = LumbarSpineDataset(
    images_dir='/kaggle/working',  # <<--- Change this!
    labels_csv='/path/to/train_label_coordinates.csv'
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn
)

# ðŸ“Œ 7. PREPARE MODEL
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model(NUM_CLASSES)
model.to(device)

# ðŸ“Œ 8. OPTIMIZER
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# ðŸ“Œ 9. TRAIN LOOP
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    
    for images, targets in train_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        running_loss += losses.item()
    
    lr_scheduler.step()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

print("Training completed!")



# Paths to each condition's best weights
spinal_stenosis_weights = (
    "results_spinal_canal_stenosis/Spinal_Canal_Stenosis/"
    "epochs-1_batch-4_20250311_1303/weights/best.pt"
)
neural_foraminal_weights = (
    "results_neural_foraminal_narrowing/Neural_Foraminal_Narrowing/"
    "epochs-1_batch-4_20250311_1303/weights/best.pt"
)
subarticular_weights = (
    "results_subarticular_stenosis/Subarticular_Stenosis/"
    "epochs-1_batch-4_20250311_1303/weights/best.pt"
)

# Load each YOLO model
model_spinal_stenosis = YOLO(spinal_stenosis_weights)
model_neural_foraminal = YOLO(neural_foraminal_weights)
model_subarticular = YOLO(subarticular_weights)



from ultralytics import YOLO


RSNA_ROOT_DIR = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification"

LEVELS = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']
CLASSES = ['Normal_Mild', 'Moderate', 'Severe']

DATASET_TYPE = "test"
TEST_SERIES_DESCRIPTIONS_CSV = f"{DATASET_TYPE}_series_descriptions.csv"
TEST_IMAGES_ROOT_DIR = os.path.join(RSNA_ROOT_DIR, f"{DATASET_TYPE}_images")
TEST_DF = pd.read_csv(os.path.join(RSNA_ROOT_DIR, TEST_SERIES_DESCRIPTIONS_CSV))
DEBUG = len(TEST_DF.study_id.unique()) == 1

YOLO_PT_AXIAL_T2_PATH = "/kaggle/input/rsna24-pt-yolo-axial-t2/best.pt"    
SIAMESE_AXIAL_T2_REFIMG_ROOT_DIR = '/kaggle/input/rsna24-refimages-axial-t2-saimese/refimages'
SIAMESE_AXIAL_T2_PT_LIST = sorted(glob.glob('/kaggle/input/rsna24-pt-axial-t2-siamese/to_upload/*.pth'))

PATCH_IMAGES_DIR = "/kaggle/working/patches"
os.makedirs(PATCH_IMAGES_DIR, exist_ok=True)
AXIAL_T2_DIR = os.path.join(PATCH_IMAGES_DIR, "Axial_T2")
os.makedirs(AXIAL_T2_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Helper functions

def convert_to_dict(d: Any) -> Any:
    """
    Recursively convert a defaultdict to a regular dict.

    Args:
        d: A dictionary or defaultdict.

    Returns:
        A regular dictionary with the same keys and values.
    """
    if isinstance(d, defaultdict):
        return {k: convert_to_dict(v) for k, v in d.items()}
    return d


def atoi(text: str) -> Union[int, str]:
    """
    Convert a string to an integer if it is numeric; otherwise, return the string.

    Args:
        text: The input string.

    Returns:
        The integer conversion or the original string.
    """
    return int(text) if text.isdigit() else text


def natural_keys(text: str) -> List[Union[int, str]]:
    """
    Generate keys to sort strings in human order.

    Splits the text into integers and non-numeric parts.

    Args:
        text: The text to be split.

    Returns:
        A list of integers and strings for natural sorting.
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def convert_dicom_to_image(dcm_path: str) -> np.ndarray:
    """
    Convert a DICOM file to a normalized 8-bit grayscale image.

    The function applies modality and VOI LUTs, and adjusts for photometric interpretation.

    Args:
        dcm_path: Path to the DICOM file.

    Returns:
        The converted image as a NumPy array with dtype uint8.
    """
    dicom = pydicom.dcmread(dcm_path)
    arr = dicom.pixel_array

    # Handle palette color images if necessary
    if dicom.PhotometricInterpretation == "PALETTE COLOR":
        arr = pydicom.pixel_data_handlers.apply_color_lut(arr, dicom)

    arr = pydicom.pixel_data_handlers.apply_modality_lut(arr, dicom)
    arr = pydicom.pixel_data_handlers.apply_voi_lut(arr, dicom, index=0)

    if dicom.PhotometricInterpretation == "MONOCHROME1":
        arr = np.amax(arr) - arr

    lower, upper = np.percentile(arr, (1, 99))
    arr = np.clip(arr, lower, upper)
    arr = arr - np.min(arr)
    arr = arr / np.max(arr)
    arr = (arr * 255).astype(np.uint8)
    return arr


def visualize_detections(image: np.ndarray, 
                         p0: Tuple[int, int], 
                         p1: Tuple[int, int], 
                         level: str, 
                         side: str) -> None:
    """
    Visualize detection patches by drawing a rectangle and overlaying text on the image.

    Args:
        image: The input grayscale image.
        p0: Top-left corner coordinates (x0, y0) of the rectangle.
        p1: Bottom-right corner coordinates (x1, y1) of the rectangle.
        level: The lumbar spine level (e.g. 'l1_l2').
        side: Side of the detection (e.g. 'left' or 'right').
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image_rgb = cv2.rectangle(image_rgb, p0, p1, color=(10, 10, 200), thickness=2)
    plt.imshow(image_rgb)
    text = f"{level}_{side}"
    x_position, y_position = 50, 50  # Coordinates for text
    plt.text(x_position, y_position, text, color='red', fontsize=15, 
             fontweight='bold', backgroundcolor='white')
    plt.axis('off')
    plt.show()


def save_patch(image: np.ndarray, 
               center: Tuple[int, int], 
               patch_size: int, 
               save_path: str) -> None:
    """
    Extract a patch centered at the provided coordinates from the image and save it as a PNG.

    Args:
        image: The full image from which to extract the patch.
        center: A tuple (xc, yc) indicating the center of the patch.
        patch_size: The full width/height of the square patch.
        save_path: The file path to save the extracted patch.
    """
    xc, yc = center
    half_patch = patch_size // 2
    x0, y0 = int(xc) - half_patch, int(yc) - half_patch
    x1, y1 = int(xc) + half_patch, int(yc) + half_patch
    patch = image[y0:y1, x0:x1]
    cv2.imwrite(save_path, patch)


def process_detection_results(dcm_conf_per_class: Dict[int, List[Tuple[float, float, float, float, float, np.ndarray, int]]]
                             ) -> Dict[str, Dict[str, Tuple[float, float, float, float, float, np.ndarray, int]]]:
    """
    Process raw detection outputs to select the best detections for each level and side.

    Detections for each class are sorted by confidence and then partitioned by side (left/right).
    Further logic compares left and right detections based on confidence scores for possible adjustments.

    Args:
        dcm_conf_per_class: A dictionary keyed by class_id that maps to a list of tuples containing
            bounding box coordinates, confidence, image and DICOM number.

    Returns:
        A nested dictionary organized as best_dcm[level][side] containing the best detection tuple.
    """
    sorted_dcm: Dict[int, Any] = defaultdict(dict)
    for class_id, detections in dcm_conf_per_class.items():
        if detections:
            detections = sorted(detections, key=lambda x: x[4])
            side: str = "left" if class_id < 5 else "right"
            level: str = LEVELS[class_id % 5]
            sorted_dcm[level][side] = detections

    best_dcm: Dict[str, Dict[str, Tuple[float, float, float, float, float, np.ndarray, int]]] = defaultdict(dict)
    for level in sorted_dcm:
        if 'left' in sorted_dcm[level]:
            best_dcm[level]['left'] = sorted_dcm[level]['left'][-1]
        if 'right' in sorted_dcm[level]:
            best_dcm[level]['right'] = sorted_dcm[level]['right'][-1]

        # Adjust detection if there is high confidence on one side and low on the other
        if 'left' in sorted_dcm[level] and 'right' in sorted_dcm[level]:
            left_best = sorted_dcm[level]['left'][-1]
            right_best = sorted_dcm[level]['right'][-1]
            if left_best[4] > 0.5 and right_best[4] < 0.5:
                best_dcm_number = left_best[-1]
                for det in sorted_dcm[level]['right']:
                    if det[-1] == best_dcm_number:
                        best_dcm[level]['right'] = det
                        break
                else:
                    # Use left image and dcm number if matching detection not found
                    best_dcm[level]['right'] = (right_best[0], right_best[1], 
                                                right_best[2], right_best[3], 
                                                0, left_best[-2], left_best[-1])
            elif left_best[4] < 0.5 and right_best[4] > 0.5:
                best_dcm_number = right_best[-1]
                for det in sorted_dcm[level]['left']:
                    if det[-1] == best_dcm_number:
                        best_dcm[level]['left'] = det
                        break
                else:
                    best_dcm[level]['left'] = (left_best[0], left_best[1], 
                                               left_best[2], left_best[3], 
                                               0, right_best[-2], right_best[-1])
    return convert_to_dict(best_dcm)


def save_patch_with_metadata(image: np.ndarray, 
                             center: Tuple[int, int], 
                             patch_size: int, 
                             save_path: str) -> None:
    """
    Extract a patch from the full image, save it, and save its bounding box as metadata.
    """
    xc, yc = center
    half_patch = patch_size // 2
    x0, y0 = int(xc) - half_patch, int(yc) - half_patch
    x1, y1 = int(xc) + half_patch, int(yc) + half_patch
    patch = image[y0:y1, x0:x1]
    cv2.imwrite(save_path, patch)
    
    # Save metadata.
    metadata = {
        "center": [xc, yc],
        "patch_size": patch_size,
        "bbox": [x0, y0, x1, y1]
    }
    metadata_path = Path(save_path).with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)


def extract_axial_t2_patches() -> None:
    """
    Extract and save axial T2 patches from DICOM images.

    For each study and series having the "Axial T2" description, the function:
      - Reads DICOM files.
      - Converts DICOM images to an 8-bit format.
      - Uses a YOLO detector (loaded externally) to detect regions of interest.
      - Selects the best detections for each spine level and side.
      - Extracts and saves a patch around the detected region.
      - Optionally visualizes detections if DEBUG is True.
    """
    # Load the YOLO detector (assumes YOLO class is available in the environment)
    detector = YOLO(YOLO_PT_AXIAL_T2_PATH)

    study_ids = TEST_DF.study_id.unique()
    for study_id in tqdm(study_ids, desc="Processing studies"):
        df_study = TEST_DF[(TEST_DF.study_id == study_id) & (TEST_DF.series_description == "Axial T2")]
        if not df_study.empty:
            axial_t2_series_ids = df_study['series_id'].unique()
            for series_id in axial_t2_series_ids:
                read_dir = os.path.join(TEST_IMAGES_ROOT_DIR, str(study_id), str(series_id))
                dcm_paths = sorted(glob.glob(os.path.join(read_dir, "*.dcm")), key=natural_keys)
                
                # Dictionary to store detections per class
                dcm_conf_per_class: Dict[int, List[Tuple[float, float, float, float, float, np.ndarray, int]]] = defaultdict(list)
                
                # Process each DICOM file in the series
                for dcm_path in dcm_paths:
                    image = convert_dicom_to_image(dcm_path)
                    # Convert grayscale image to RGB for detector
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    detection = detector.predict(source=image_rgb, verbose=False)
                    
                    # Process detections
                    for row in detection[0].boxes.data:
                        row_np = row.cpu().numpy()
                        class_id: int = int(row_np[-1])
                        # Append tuple: (x0, y0, x1, y1, confidence, image, dcm_number)
                        dcm_conf_per_class[class_id].append((
                            row_np[0], row_np[1], row_np[2], row_np[3], row_np[4], image, int(Path(dcm_path).stem)
                        ))
                
                best_detections = process_detection_results(dcm_conf_per_class)
                
                # Save patches for each level and side
                for level, sides in best_detections.items():
                    for side, data in sides.items():
                        if DEBUG:
                            print(level, side, data[0], data[1], data[2], data[3], data[4], data[-1])
                        x0, y0, x1, y1, _, image_ref, dcm_number = data
                        xc = int(np.round((x0 + x1) / 2.0))
                        yc = int(np.round((y0 + y1) / 2.0))
                        width = image_ref.shape[1]
                        # Define patch size as 10% of image width (patch half size computed accordingly)
                        patch_size_half = int(np.round(0.1 * width / 2))
                        patch_size = patch_size_half * 2
                        
                        target_dir = os.path.join(AXIAL_T2_DIR, str(study_id), level)
                        os.makedirs(target_dir, exist_ok=True)
                        png_path = os.path.join(target_dir, f"{study_id}_{series_id}_{side}_{dcm_number:04}.png")
                        save_patch_with_metadata(image_ref, (xc, yc), patch_size, png_path)
                        # save_patch(image_ref, (xc, yc), patch_size, png_path)
                        
                        if DEBUG:
                            visualize_detections(image_ref, (xc - patch_size_half, yc - patch_size_half),
                                                 (xc + patch_size_half, yc + patch_size_half), level, side)


extract_axial_t2_patches()

