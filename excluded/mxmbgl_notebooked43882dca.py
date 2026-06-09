!tar xfvz /kaggle/input/ultralytics-for-offline-install/archive.tar.gz
!pip install --no-index --find-links=./packages ultralytics
!rm -rf ./packages


import plotly.express as px 
from PIL import Image, ImageDraw
import random
import seaborn as sns
from matplotlib.patches import Rectangle 
from ultralytics import YOLO
import yaml 
import json 
import os
import glob 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau 
import cv2
import threading 
import time
from contextlib import nullcontext 
from concurrent.futures import ThreadPoolExecutor 
import math 


DATA = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025'
LABELS = os.path.join(DATA, 'train_labels.csv')
TRAIN = os.path.join(DATA, 'train')
TEST = os.path.join(DATA, 'test')
OUTPUT = './'
MODEL = './models'

os.makedirs(OUTPUT, exist_ok=True)
os.makedirs(MODEL, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True


train_labels = pd.read_csv(LABELS)

print('Dataset shape: ', train_labels.shape)
print('\nColumns in the dataset: ')
print(train_labels.columns.tolist())

print('\nBasic statistic: ')
display(train_labels.describe)


unique_tomo_count = train_labels['tomo_id'].nunique()
print(f"\nNumber of unique tomograms: {unique_tomo_count}")

motors_per_tomo = train_labels.groupby('tomo_id')['Number of motors'].first().value_counts().sort_index()
print("\nDistribution of motors per tomogram:")
print(motors_per_tomo)

plt.figure(figsize=(8, 5))
motors_per_tomo.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribution of Motors per Tomogram')
plt.xlabel('Number of Motors')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


print('\nSample rows from training labels: ')
display(train_labels.head())

print('\nMissing values per column: ')
display(train_labels.isnull().sum())

print("\nTomogram size ranges:")
print("Z-axis (slices):", train_labels['Array shape (axis 0)'].min(), "to", train_labels['Array shape (axis 0)'].max())
print("X-axis (width):", train_labels['Array shape (axis 1)'].min(), "to", train_labels['Array shape (axis 1)'].max())
print("Y-axis (height):", train_labels['Array shape (axis 2)'].min(), "to", train_labels['Array shape (axis 2)'].max())
print("\nVoxel spacing distribution:")
voxel_spacing_counts = train_labels['Voxel spacing'].value_counts().sort_index()
display(voxel_spacing_counts)


fig_motor = px.scatter_3d(
    train_labels,
    x = 'Motor axis 0',
    y = 'Motor axis 1',
    z = 'Motor axis 2',
    color = 'Number of motors',
    size_max=8,
    width=900,
    height=600,
    opacity=0.85,
    template='plotly_dark',
    title='Motor Axes'
)

fig_motor.update_layout(
    font_size = 12,
    legend_font_size = 14,
    margin=dict(l=10, r=10, b=10, t=40)
)

fig_motor.show()


fig_shape = px.scatter_3d(
    train_labels, 
    x='Array shape (axis 0)', 
    y='Array shape (axis 1)', 
    z='Array shape (axis 2)',
    color='Number of motors', 
    color_continuous_scale="magma",  
    size_max=8, 
    width=900, 
    height=600, 
    opacity=0.85, 
    template="seaborn",  
    title="ğŸ§¬ 3D Scatter Plot: Tomogram Shapes"
)

fig_shape.update_layout(
    font_size=10,
    legend_font_size=14,
    margin=dict(l=10, r=10, b=10, t=40)
)

fig_shape.show()


display(train_labels.describe().loc[['mean', 'min', 'max']].T)

train_labels.hist(
    bins=30,
    figsize=(14 ,8),
    layout=(3,4),
    edgecolor='black',
    color='#4CAF50'
)
plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold', color='darkblue')
plt.tight_layout()


plt.figure(figsize=(9,5), facecolor='white')
sns.heatmap(
    data=train_labels.corr(numeric_only=True),
    cmap='coolwarm',
    vmin=-1, vmax=1,
    linecolor='white', linewidth=0.6,
    annot=True,
    fmt='.2f'
)
plt.title('Correlation Heatmap', fontsize=14, fontweight='bold', color='black')
plt.show()


def plotImages(title, directory, images=16, img_size=(128,128)):
    print(f'{title}')
    image_files = glob.glob(directory)

    if not image_files:
        print('No images found.')
        return

    plt.figure(figsize=(12,12))
    plt.subplots_adjust(wspace=0.1, hspace=0.1) 
    print(f'Loaded {len(image_files)} images.')
    
    for i, file_path in enumerate(image_files[:images]):
        img = cv2.imread(file_path)
        if img is None:
            continue
        img = cv2.resize(img, img_size)
        plt.subplot(4, 4, i+1) 
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')

    plt.suptitle(title, fontsize=14, fontweight='bold', color='darkred')
    plt.show()

plotImages("Bacterial Flagellar Motors - Train Images", "../input/byu-locating-bacterial-flagellar-motors-2025/train/***/**")


def visualize_images(path, n_images=12, is_random=True, figsize=(14, 14)):
   
    plt.figure(figsize=figsize)  
    image_names = os.listdir(path)  
    if is_random:
         image_names = random.sample(image_names, min(len(image_names), n_images))
    else:
        image_names = image_names[:n_images]

    w = int(math.sqrt(n_images))  
    h = math.ceil(n_images / w)  
    for ind, image_name in enumerate(image_names):
        img_path = os.path.join(path, image_name)  
        img = cv2.imread(img_path)  
        if img is None:
            print(f"Warning: Could not read {img_path}")  
            continue  
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        plt.subplot(h, w, ind + 1)  
        plt.imshow(img)  
        plt.xticks([]) 
        plt.yticks([]) 

    plt.suptitle("Sample Tomogram Images", fontsize=14, fontweight='bold',
                 color="darkblue") 
    plt.show() 

visualize_images("/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train/tomo_098751",
                 n_images=9) 


sample_tomo_id = train_labels['tomo_id'].iloc[0] 
print(f"\nVisualizing sample tomogram: {sample_tomo_id}") 

TRAIN = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train" 
sample_folder = os.path.join(TRAIN, sample_tomo_id)  

if os.path.exists(sample_folder):  
    slice_files = sorted(glob.glob(os.path.join(sample_folder, '*.jpg')))  
    print(f"Number of slice files in tomogram '{sample_tomo_id}': {len(slice_files)}")  
    if slice_files: 
        sample_slice = Image.open(slice_files[0]) 
        print(f"Dimensions of a sample slice: {sample_slice.size}")  

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))  
        slice_indices = [0, len(slice_files) // 2,len(slice_files) - 1]  
        for i, idx in enumerate(slice_indices):
            img = Image.open(slice_files[idx])  
            axes[i].imshow(img,cmap='gray')  
            axes[i].set_title(f"Slice {idx}") 
            axes[i].axis('off')  
        plt.tight_layout()  
        plt.show()  
    else:
        print("No slice files found in the folder.")  
else:
    print(f"Folder '{sample_folder}' does not exist. Please check the dataset directory.") 


yolo_dataset_dir = '/kaggle/working/yolo_dataset'
yolo_images_train = os.path.join(yolo_dataset_dir, "images", "train")
yolo_images_val = os.path.join(yolo_dataset_dir, "images", "val")
yolo_labels_train = os.path.join(yolo_dataset_dir, "labels", "train")
yolo_labels_val = os.path.join(yolo_dataset_dir, "labels", "val")

for dir_path in [yolo_images_train, yolo_images_val, yolo_labels_train, yolo_labels_val]:
    os.makedirs(dir_path, exist_ok=True)


TRUST = 4 
BOX_SIZE = 24
TRAIN_SPLIT = 0.8


def normalize_slice(slice_data):
    p2 = np.percentile(slice_data, 2)
    p98 = np.percentile(slice_data, 98)
    clipped_data = np.clip(slice_data, p2, p98)
    normalized = 255 * (clipped_data - p2) / (p98 - p2)
    return np.uint8(normalized)


images_train_dir = os.path.join(yolo_dataset_dir, 'images','train')
labels_train_dir = os.path.join(yolo_dataset_dir, 'labels','train')

{images_train_dir}



class YoloDatasetPreprocessor:
   
    def __init__(self, data_path, yolo_dataset_dir, trust=4, train_split=0.8, box_size=24):
        
        self.data_path = data_path  
        self.yolo_dataset_dir = yolo_dataset_dir  
        self.trust = trust  
        self.train_split = train_split  
        self.box_size = box_size  
        
        self.train_dir = os.path.join(data_path, "train")  
        self.yolo_images_train = os.path.join(yolo_dataset_dir, "images","train")  
        self.yolo_images_val = os.path.join(yolo_dataset_dir, "images","val")  
        self.yolo_labels_train = os.path.join(yolo_dataset_dir, "labels","train")  
        self.yolo_labels_val = os.path.join(yolo_dataset_dir, "labels","val")  
        
        os.makedirs(self.yolo_images_train,exist_ok=True)  
        os.makedirs(self.yolo_images_val,exist_ok=True)  
        os.makedirs(self.yolo_labels_train,exist_ok=True)  
        os.makedirs(self.yolo_labels_val,exist_ok=True) 
        
        self.labels_df = pd.read_csv(os.path.join(data_path,"train_labels.csv"))  
        
    def normalize_slice(self, slice_data):
       
        p2 = np.percentile(slice_data, 2)
        p98 = np.percentile(slice_data, 98)
        clipped_data = np.clip(slice_data, p2, p98)
        normalized = 255 * (clipped_data - p2) / (p98 - p2)
        return np.uint8(normalized)  
        
    def process_tomogram_set(self, tomogram_ids, images_dir, labels_dir, set_name):
        
        motor_counts = []  
        for tomo_id in tomogram_ids:
            tomo_motors = self.labels_df[self.labels_df['tomo_id'] == tomo_id]  
            for _, motor in tomo_motors.iterrows():  
                if pd.isna(motor['Motor axis 0']):  
                    continue
                motor_counts.append(
                    (tomo_id,
                     int(motor['Motor axis 0']),
                     int(motor['Motor axis 1']),
                     int(motor['Motor axis 2']),
                     int(motor['Array shape (axis 0)']))
                )  
                
        print(f"Ğ�Ğ±Ñ€Ğ°Ğ±Ğ°Ñ‚Ñ‹Ğ²Ğ°ĞµÑ‚Ñ�Ñ� Ğ¿Ñ€Ğ¸Ğ¼ĞµÑ€Ğ½Ğ¾ {len(motor_counts) * (2 * self.trust + 1)} Ñ�Ñ€ĞµĞ·Ğ¾Ğ² Ğ´Ğ»Ñ� {set_name}")  # Ğ’Ñ‹Ğ²Ğ¾Ğ´Ğ¸Ñ‚ Ğ¾Ñ€Ğ¸ĞµĞ½Ñ‚Ğ¸Ñ€Ğ¾Ğ²Ğ¾Ñ‡Ğ½Ğ¾Ğµ ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ñ�Ñ€ĞµĞ·Ğ¾Ğ², ĞºĞ¾Ñ‚Ğ¾Ñ€Ñ‹Ğµ Ğ±ÑƒĞ´ÑƒÑ‚ Ğ¾Ğ±Ñ€Ğ°Ğ±Ğ¾Ñ‚Ğ°Ğ½Ñ‹.
        
        processed_slices = 0 
        
        for tomo_id, z_center, y_center, x_center, z_max in tqdm(motor_counts, desc=f"Processing {set_name} motors"):  # Ğ˜Ñ‚ĞµÑ€Ğ¸Ñ€ÑƒĞµÑ‚Ñ�Ñ� Ğ¿Ğ¾ Ñ�Ğ¿Ğ¸Ñ�ĞºÑƒ Ğ¼Ğ¾Ñ‚Ğ¾Ñ€Ğ¾Ğ² Ñ� Ğ¾Ñ‚Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸ĞµĞ¼ Ğ¿Ñ€Ğ¾Ğ³Ñ€ĞµÑ�Ñ�Ğ°.
            z_min = max(0, z_center - self.trust)  
            z_max_bound = min(z_max - 1, z_center + self.trust)  
            for z in range(z_min,z_max_bound + 1):  
                slice_filename = f"slice_{z:04d}.jpg"  
                src_path = os.path.join(self.train_dir, tomo_id,slice_filename)  
                if not os.path.exists(src_path):  
                    print(
                        f"Warning: {src_path} does not exist, skipping.")  
                    continue  
                    
                img = Image.open(src_path)  
                img_array = np.array(img)  
                normalized_img = self.normalize_slice(img_array)  
                dest_filename = f"{tomo_id}_z{z:04d}_y{y_center:04d}_x{x_center:04d}.jpg"  
                dest_path = os.path.join(images_dir,dest_filename)  
                Image.fromarray(normalized_img).save(dest_path)  
                
                img_width, img_height = img.size  
                x_center_norm = x_center / img_width  
                y_center_norm = y_center / img_height  
                box_width_norm = self.box_size / img_width  
                box_height_norm = self.box_size / img_height  
                label_path = os.path.join(labels_dir,dest_filename.replace('.jpg','.txt'))  
                with open(label_path, 'w') as f:
                    f.write(f"0 {x_center_norm} {y_center_norm} {box_width_norm} {box_height_norm}\n")  # Ğ—Ğ°Ğ¿Ğ¸Ñ�Ñ‹Ğ²Ğ°ĞµÑ‚ Ğ°Ğ½Ğ½Ğ¾Ñ‚Ğ°Ñ†Ğ¸Ñ� Ğ² Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ‚Ğµ YOLO:
                processed_slices += 1  
        return processed_slices, len(motor_counts)  
        
    def prepare_yolo_dataset(self):
        """
        ĞŸĞ¾Ğ´Ğ³Ğ¾Ñ‚Ğ°Ğ²Ğ»Ğ¸Ğ²Ğ°ĞµÑ‚ Ğ´Ğ°Ñ‚Ğ°Ñ�ĞµÑ‚ Ğ² Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ‚Ğµ YOLO.
        """
        labels_df = self.labels_df  
        total_motors = labels_df['Number of motors'].sum()  
        print(f"Ğ’Ñ�ĞµĞ³Ğ¾ Ğ¼Ğ¾Ñ‚Ğ¾Ñ€Ğ¾Ğ² Ğ² Ğ´Ğ°Ñ‚Ğ°Ñ�ĞµÑ‚Ğµ: {total_motors}")

        tomo_df = labels_df[labels_df['Number of motors'] > 0].copy()  
        unique_tomos = tomo_df['tomo_id'].unique()  
        print(f"Ğ�Ğ°Ğ¹Ğ´ĞµĞ½Ğ¾ {len(unique_tomos)} ÑƒĞ½Ğ¸ĞºĞ°Ğ»ÑŒĞ½Ñ‹Ñ… Ñ‚Ğ¾Ğ¼Ğ¾Ğ³Ñ€Ğ°Ğ¼Ğ¼ Ñ� Ğ¼Ğ¾Ñ‚Ğ¾Ñ€Ğ°Ğ¼Ğ¸")

        np.random.shuffle(unique_tomos)  
        split_idx = int(len(unique_tomos) * self.train_split)  
        train_tomos = unique_tomos[:split_idx]  
        val_tomos = unique_tomos[split_idx:]  
        print(f"Ğ Ğ°Ğ·Ğ´ĞµĞ»ĞµĞ½Ğ¸Ğµ: {len(train_tomos)} Ñ‚Ğ¾Ğ¼Ğ¾Ğ³Ñ€Ğ°Ğ¼Ğ¼ Ğ´Ğ»Ñ� Ñ‚Ñ€ĞµĞ½Ğ¸Ñ€Ğ¾Ğ²ĞºĞ¸, {len(val_tomos)} Ñ‚Ğ¾Ğ¼Ğ¾Ğ³Ñ€Ğ°Ğ¼Ğ¼ Ğ´Ğ»Ñ� Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¸")

        train_slices, train_motors = self.process_tomogram_set(train_tomos, self.yolo_images_train, self.yolo_labels_train,"training")  # Ğ�Ğ±Ñ€Ğ°Ğ±Ğ°Ñ‚Ñ‹Ğ²Ğ°ĞµÑ‚ Ğ¾Ğ±ÑƒÑ‡Ğ°Ñ�Ñ‰Ğ¸Ğµ Ñ‚Ğ¾Ğ¼Ğ¾Ğ³Ñ€Ğ°Ğ¼Ğ¼Ñ‹.
        val_slices, val_motors = self.process_tomogram_set(val_tomos, self.yolo_images_val,self.yolo_labels_val,"validation")  # Ğ�Ğ±Ñ€Ğ°Ğ±Ğ°Ñ‚Ñ‹Ğ²Ğ°ĞµÑ‚ Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¾Ğ½Ğ½Ñ‹Ğµ Ñ‚Ğ¾Ğ¼Ğ¾Ğ³Ñ€Ğ°Ğ¼Ğ¼Ñ‹.

        yaml_content = {
            'path': self.yolo_dataset_dir,  
            'train': 'images/train',  
            'val': 'images/val',  
            'names': { 0: 'motor' }  
            }
        with open(os.path.join(self.yolo_dataset_dir,'dataset.yaml'), 'w') as f:  
            yaml.dump(yaml_content, f,default_flow_style=False)  
            
        print(f"\nĞ ĞµĞ·Ñ�Ğ¼Ğµ Ğ¾Ğ±Ñ€Ğ°Ğ±Ğ¾Ñ‚ĞºĞ¸:")  
        print(f"- Ğ¢Ñ€ĞµĞ½Ğ¸Ñ€Ğ¾Ğ²Ğ¾Ñ‡Ğ½Ñ‹Ğµ Ğ´Ğ°Ğ½Ğ½Ñ‹Ğµ: {len(train_tomos)} Ñ‚Ğ¾Ğ¼Ğ¾Ğ³Ñ€Ğ°Ğ¼Ğ¼, {train_motors} Ğ¼Ğ¾Ñ‚Ğ¾Ñ€Ğ¾Ğ², {train_slices} Ñ�Ñ€ĞµĞ·Ğ¾Ğ²")
        print(f"- Ğ’Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¾Ğ½Ğ½Ñ‹Ğµ Ğ´Ğ°Ğ½Ğ½Ñ‹Ğµ: {len(val_tomos)} Ñ‚Ğ¾Ğ¼Ğ¾Ğ³Ñ€Ğ°Ğ¼Ğ¼, {val_motors} Ğ¼Ğ¾Ñ‚Ğ¾Ñ€Ğ¾Ğ², {val_slices} Ñ�Ñ€ĞµĞ·Ğ¾Ğ²")
        print(f"- Ğ’Ñ�ĞµĞ³Ğ¾: {len(train_tomos) + len(val_tomos)} Ñ‚Ğ¾Ğ¼Ğ¾Ğ³Ñ€Ğ°Ğ¼Ğ¼, {train_motors + val_motors} Ğ¼Ğ¾Ñ‚Ğ¾Ñ€Ğ¾Ğ², {train_slices + val_slices} Ñ�Ñ€ĞµĞ·Ğ¾Ğ²")

        return {  
            "dataset_dir": self.yolo_dataset_dir,
            "yaml_path": os.path.join(self.yolo_dataset_dir, 'dataset.yaml'),
            "train_tomograms": len(train_tomos),
            "val_tomograms": len(val_tomos),
            "train_motors": train_motors,
            "val_motors": val_motors,
            "train_slices": train_slices,
            "val_slices": val_slices
        }


DATA = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025" 
yolo_dataset_dir = '/kaggle/working/yolo_dataset'  
processor = YoloDatasetPreprocessor(DATA, yolo_dataset_dir)  
summary = processor.prepare_yolo_dataset()  

print(f"\nĞ�Ğ±Ñ€Ğ°Ğ±Ğ¾Ñ‚ĞºĞ° Ğ·Ğ°Ğ²ĞµÑ€ÑˆĞµĞ½Ğ°:")  
print(f"- Ğ¢Ñ€ĞµĞ½Ğ¸Ñ€Ğ¾Ğ²Ğ¾Ñ‡Ğ½Ñ‹Ğµ Ğ´Ğ°Ğ½Ğ½Ñ‹Ğµ: {summary['train_tomograms']} Ñ‚Ğ¾Ğ¼Ğ¾Ğ³Ñ€Ğ°Ğ¼Ğ¼, {summary['train_motors']} Ğ¼Ğ¾Ñ‚Ğ¾Ñ€Ğ¾Ğ², {summary['train_slices']} Ñ�Ñ€ĞµĞ·Ğ¾Ğ²")
print(f"- Ğ’Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¾Ğ½Ğ½Ñ‹Ğµ Ğ´Ğ°Ğ½Ğ½Ñ‹Ğµ: {summary['val_tomograms']} Ñ‚Ğ¾Ğ¼Ğ¾Ğ³Ñ€Ğ°Ğ¼Ğ¼, {summary['val_motors']} Ğ¼Ğ¾Ñ‚Ğ¾Ñ€Ğ¾Ğ², {summary['val_slices']} Ñ�Ñ€ĞµĞ·Ğ¾Ğ²")
print(f"- Ğ”Ğ¸Ñ€ĞµĞºÑ‚Ğ¾Ñ€Ğ¸Ñ� Ğ´Ğ°Ñ‚Ğ°Ñ�ĞµÑ‚Ğ°: {summary['dataset_dir']}")  
print(f"- YAML ĞºĞ¾Ğ½Ñ„Ğ¸Ğ³ÑƒÑ€Ğ°Ñ†Ğ¸Ñ�: {summary['yaml_path']}")  
print("\nĞ“Ğ¾Ñ‚Ğ¾Ğ²Ğ¾ Ğ´Ğ»Ñ� Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ñ� YOLO!")  



def visualize_random_training_samples(num_samples=4):
    image_files = []
    for ext in ['*.jpg','*.jpeg','*.png']: 
        image_files.extend(glob.glob(os.path.join(images_train_dir, '**', ext), recursive=True))

    if len(image_files) == 0:
        print('No image files found in the train directory')
        return 

    num_samples = min(num_samples, len(image_files))
    random_images = random.sample(image_files, num_samples)

    rows = int(np.ceil(num_samples/2))
    cols = min(num_samples, 2)
    fig,axes = plt.subplots(rows, cols, figsize=(14, 5*rows))

    if num_samples == 1:
        axes = np.array([axes]) 
    axes = axes.flatten()

    for i, img_path in enumerate(random_images):
        try:
            relative_path = os.path.relpath(img_path, images_train_dir) # Ğ²Ñ‹Ñ‡Ğ¸Ñ�Ğ»Ñ�ĞµÑ‚ Ğ¾Ñ‚Ğ½Ğ¾Ñ�Ğ¸Ñ‚ĞµĞ»ÑŒĞ½Ñ‹Ğ¹ Ğ¿ÑƒÑ‚ÑŒ Ğ¾Ñ‚ images_train_dir Ğ´Ğ¾ img_path.
            label_path = os.path.join(labels_train_dir, os.path.splitext(relative_path)[0] + '.txt')
            img = Image.open(img_path)
            img_width, img_height = img.size
            img_array = np.array(img)
            p2 = np.percentile(img_array, 2) 
            p98 = np.percentile(img_array, 98)
            normalized = np.clip(img_array, p2, p98)
            normalized = 255*(normalized - p2) / (p98-p2)
            img_normalized = Image.fromarray(np.uint8(normalized))
            img_rgb = img_normalized.convert('RGB')
            overlay = Image.new('RGBA', img_rgb.size, (0,0,0,0))
            draw = ImageDraw.Draw(overlay) 
            
            annotations = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        values = line.strip().split() #line.strip() ÑƒĞ´Ğ°Ğ»Ñ�ĞµÑ‚ Ğ»Ğ¸ÑˆĞ½Ğ¸Ğµ Ğ¿Ñ€Ğ¾Ğ±ĞµĞ»Ñ‹ Ğ¸ Ñ�Ğ¸Ğ¼Ğ²Ğ¾Ğ»Ñ‹ Ğ½Ğ¾Ğ²Ğ¾Ğ¹ Ñ�Ñ‚Ñ€Ğ¾ĞºĞ¸ (Ğ¿ĞµÑ€ĞµĞ²Ğ¾Ğ´Ñ‹ Ñ�Ñ‚Ñ€Ğ¾ĞºĞ¸) Ñ� Ğ½Ğ°Ñ‡Ğ°Ğ»Ğ° Ğ¸ ĞºĞ¾Ğ½Ñ†Ğ° Ñ�Ñ‚Ñ€Ğ¾ĞºĞ¸.
                        class_id = int(values[0])
                        x_center = float(values[1])*img_width
                        y_center = float(values[2])*img_height
                        width = float(values[3])*img_width
                        height = float(values[4])*img_height
                        annotations.append({
                            'class_id':class_id,
                            'x_center':x_center,
                            'y_center':y_center,
                            'width':width,
                            'height':height
                        })

            for ann in annotations:
                x_center = ann['x_center']
                y_center = ann['y_center']
                width = ann['width']
                height = ann['height']
                x1 = max(0, int(x_center - width/2))
                y1 = max(0, int(y_center - height/2))
                x2 = min(img_width, int(x_center+width/2))
                y2 = min(img_height, int(y_center+height/2))
                draw.rectangle([x1,y1,x2,y2], fill=(255,0,0,64), outline=(255,0,0,200))
                draw.text((x1, y1-10), f"Class {ann['class_id']}", fill=(255,0,0,255))

            if not annotations:
                draw.text((10,10), "No annotations found", fill=(255,0,0,255))

            img_rgb = Image.alpha_composite(img_rgb.convert('RGBA'), overlay).convert('RGB')
            axes[i].imshow(np.array(img_rgb)) 
            img_name = os.path.basename(img_path)
            axes[i].set_title(f"Image: {img_name}\nAnnotations: {len(annotations)}") 
            axes[i].axis('on')

        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            axes[i].text(0.5, 0.5, f"Error loading image: {os.path.basename(img_path)}",
                         horizontalalignment='center', verticalalignment='center')
            axes[i].axis('off')


    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout() 
    plt.show()
    print(f"Displayed {num_samples} random images with YOLO annotations")

visualize_random_training_samples(4)


np.random.seed(42) 
random.seed(42) 
torch.manual_seed(42) 
yolo_weights_dir = '/kaggle/working/yolo_weights'
yolo_pretrained_weights = '/kaggle/input/ultralytics-for-offline-install/yolov8-weights/yolov8n.pt'
os.makedirs(yolo_weights_dir, exist_ok=True)


def fix_yaml_paths(yaml_path):
   
    print(f'Fixing YAML paths in {yaml_path}')
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)   
    if 'path' in yaml_data:
        yaml_data['path'] = yolo_dataset_dir   
    fixed_yaml_path = "/kaggle/working/fixed_dataset.yaml"

    with open(fixed_yaml_path, 'w') as f:
        yaml.dump(yaml_data,f)  
    print(f"Created fixed YAML at {fixed_yaml_path} with path: {yaml_data.get('path')}")    
    return fixed_yaml_path


def plot_dfl_loss_curve(run_dir):
    
    results_csv = os.path.join(run_dir, 'results.csv')

    if not os.path.exists(results_csv):
        print(f'Results file not found at {results_csv}')
        return 
    results_df = pd.read_csv(results_csv)
    train_dfl_col = [col for col in results_df.columns if 'train/dfl_loss' in col]
    val_dfl_col = [col for col in results_df.columns if 'val/dfl_loss' in col]

    if not train_dfl_col or not val_dfl_col:
        print('DFL loss columns not founds in results CSV')
        print(f'Available columns: {results_df.columns.tolist()}') 
        return 
    train_dfl_col = train_dfl_col[0]
    val_dfl_col = val_dfl_col[0]
    
    best_epoch = results_df[val_dfl_col].idxmin()
    best_val_loss = results_df.loc[best_epoch, val_dfl_col] 
    plt.figure(figsize=(10,6))
    plt.plot(results_df['epoch'], results_df[train_dfl_col], label='Train DFL Loss')
    plt.plot(results_df['epoch'], results_df[val_dfl_col], label='Validation DFL Loss')
    plt.axvline(x=results_df.loc[best_epoch, 'epoch'], color='r', linestyle='--', 
                label=f'Best Model (Epoch {int(results_df.loc[best_epoch, "epoch"])}, Val Loss: {best_val_loss:.4f})')
    
    plt.xlabel('Epoch')
    plt.ylabel('DFL Loss')
    plt.title('Training and Validation DFL Loss')    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)    
    plot_path = os.path.join(run_dir, 'dfl_loss_curve.png')
    plt.savefig(plot_path)    
    plt.savefig(os.path.join('/kaggle/working', 'dfl_loss_curve.png'))
    
    print(f"Loss curve saved to {plot_path}")    
    plt.close()    
    return best_epoch, best_val_loss



# ĞŸÑ€ĞµĞ´Ğ¿Ğ¾Ğ»Ğ°Ğ³Ğ°ĞµÑ‚Ñ�Ñ�, Ñ‡Ñ‚Ğ¾ Ğ¿ĞµÑ€ĞµĞ¼ĞµĞ½Ğ½Ñ‹Ğµ yolo_weights_dir Ğ¸ plot_dfl_loss_curve
# Ğ¾Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ñ‹ Ğ³Ğ´Ğµ-Ñ‚Ğ¾ Ğ²Ñ‹ÑˆĞµ Ğ² ĞºĞ¾Ğ´Ğµ.
# Ğ�Ğ°Ğ¿Ñ€Ğ¸Ğ¼ĞµÑ€:
# yolo_weights_dir = "/kaggle/working/yolo_weights"
# def plot_dfl_loss_curve(run_dir): ... (Ğ²Ğ°ÑˆĞ° Ñ„ÑƒĞ½ĞºÑ†Ğ¸Ñ� Ğ´Ğ»Ñ� Ğ¿Ğ¾Ñ�Ñ‚Ñ€Ğ¾ĞµĞ½Ğ¸Ñ� Ğ³Ñ€Ğ°Ñ„Ğ¸ĞºĞ° Ğ¿Ğ¾Ñ‚ĞµÑ€ÑŒ)

def train_yolo_model(yaml_path, pretrained_weights_path, epochs=30, batch_size=16, img_size=640):
    """
    Ğ�Ğ±ÑƒÑ‡Ğ°ĞµÑ‚ Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ YOLO Ğ½Ğ° Ğ¿Ğ¾Ğ´Ğ³Ğ¾Ñ‚Ğ¾Ğ²Ğ»ĞµĞ½Ğ½Ğ¾Ğ¼ Ğ½Ğ°Ğ±Ğ¾Ñ€Ğµ Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ….

    Args:
        yaml_path (str): ĞŸÑƒÑ‚ÑŒ Ğº YAML-Ñ„Ğ°Ğ¹Ğ»Ñƒ ĞºĞ¾Ğ½Ñ„Ğ¸Ğ³ÑƒÑ€Ğ°Ñ†Ğ¸Ğ¸ Ğ½Ğ°Ğ±Ğ¾Ñ€Ğ° Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ… (Ğ½Ğ°Ğ¿Ñ€Ğ¸Ğ¼ĞµÑ€, "fixed_dataset.yaml").
        pretrained_weights_path (str): ĞŸÑƒÑ‚ÑŒ Ğº Ğ¿Ñ€ĞµĞ´Ğ²Ğ°Ñ€Ğ¸Ñ‚ĞµĞ»ÑŒĞ½Ğ¾ Ğ·Ğ°Ğ³Ñ€ÑƒĞ¶ĞµĞ½Ğ½Ñ‹Ğ¼ Ğ²ĞµÑ�Ğ°Ğ¼ Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸ (Ğ½Ğ°Ğ¿Ñ€Ğ¸Ğ¼ĞµÑ€, "yolov8n.pt").
        epochs (int): ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ñ�Ğ¿Ğ¾Ñ… Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ñ�. ĞŸĞ¾ ÑƒĞ¼Ğ¾Ğ»Ñ‡Ğ°Ğ½Ğ¸Ñ� 30.
        batch_size (int): Ğ Ğ°Ğ·Ğ¼ĞµÑ€ Ğ±Ğ°Ñ‚Ñ‡Ğ° (Ğ¿Ğ°ĞºĞµÑ‚Ğ°) Ğ´Ğ»Ñ� Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ñ�. ĞŸĞ¾ ÑƒĞ¼Ğ¾Ğ»Ñ‡Ğ°Ğ½Ğ¸Ñ� 16.
        img_size (int): Ğ Ğ°Ğ·Ğ¼ĞµÑ€ Ğ¸Ğ·Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ñ�, Ğ´Ğ¾ ĞºĞ¾Ñ‚Ğ¾Ñ€Ğ¾Ğ³Ğ¾ Ğ±ÑƒĞ´ÑƒÑ‚ Ğ¼Ğ°Ñ�ÑˆÑ‚Ğ°Ğ±Ğ¸Ñ€Ğ¾Ğ²Ğ°Ñ‚ÑŒÑ�Ñ� Ğ²Ñ…Ğ¾Ğ´Ğ½Ñ‹Ğµ Ğ¸Ğ·Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ñ�. ĞŸĞ¾ ÑƒĞ¼Ğ¾Ğ»Ñ‡Ğ°Ğ½Ğ¸Ñ� 640.

    Returns:
        tuple: ĞšĞ¾Ñ€Ñ‚ĞµĞ¶, Ñ�Ğ¾Ğ´ĞµÑ€Ğ¶Ğ°Ñ‰Ğ¸Ğ¹ Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ½ÑƒÑ� Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ YOLO Ğ¸ Ğ¾Ğ±ÑŠĞµĞºÑ‚ Ñ€ĞµĞ·ÑƒĞ»ÑŒÑ‚Ğ°Ñ‚Ğ¾Ğ² Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ñ�.
    """
    # Ğ’Ñ‹Ğ²Ğ¾Ğ´Ğ¸Ğ¼ Ñ�Ğ¾Ğ¾Ğ±Ñ‰ĞµĞ½Ğ¸Ğµ Ğ¾ Ñ‚Ğ¾Ğ¼, Ğ¾Ñ‚ĞºÑƒĞ´Ğ° Ğ·Ğ°Ğ³Ñ€ÑƒĞ¶Ğ°Ñ�Ñ‚Ñ�Ñ� Ğ¿Ñ€ĞµĞ´Ğ²Ğ°Ñ€Ğ¸Ñ‚ĞµĞ»ÑŒĞ½Ğ¾ Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ½Ñ‹Ğµ Ğ²ĞµÑ�Ğ°.
    print(f"Loading pre-trained weights from: {pretrained_weights_path}")
    # Ğ˜Ğ½Ğ¸Ñ†Ğ¸Ğ°Ğ»Ğ¸Ğ·Ğ¸Ñ€ÑƒĞµĞ¼ Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ YOLO, Ğ·Ğ°Ğ³Ñ€ÑƒĞ¶Ğ°Ñ� Ğ¿Ñ€ĞµĞ´Ğ²Ğ°Ñ€Ğ¸Ñ‚ĞµĞ»ÑŒĞ½Ğ¾ Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ½Ñ‹Ğµ Ğ²ĞµÑ�Ğ°.
    # Ğ­Ñ‚Ğ¾ Ğ¿Ğ¾Ğ·Ğ²Ğ¾Ğ»Ñ�ĞµÑ‚ Ğ¸Ñ�Ğ¿Ğ¾Ğ»ÑŒĞ·Ğ¾Ğ²Ğ°Ñ‚ÑŒ Ñ‚Ñ€Ğ°Ğ½Ñ�Ñ„ĞµÑ€Ğ½Ğ¾Ğµ Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ğµ, Ñ‡Ñ‚Ğ¾ Ğ·Ğ½Ğ°Ñ‡Ğ¸Ñ‚ĞµĞ»ÑŒĞ½Ğ¾ ÑƒÑ�ĞºĞ¾Ñ€Ñ�ĞµÑ‚
    # Ñ�Ñ…Ğ¾Ğ´Ğ¸Ğ¼Ğ¾Ñ�Ñ‚ÑŒ Ğ¸ ÑƒĞ»ÑƒÑ‡ÑˆĞ°ĞµÑ‚ ĞºĞ°Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸ Ğ½Ğ° Ğ½Ğ¾Ğ²Ğ¾Ğ¼ Ğ´Ğ°Ñ‚Ğ°Ñ�ĞµÑ‚Ğµ.
    model = YOLO(pretrained_weights_path)
    model.to(device)
    # Ğ—Ğ°Ğ¿ÑƒÑ�ĞºĞ°ĞµĞ¼ Ğ¿Ñ€Ğ¾Ñ†ĞµÑ�Ñ� Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ñ� Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸.
    # ĞœĞµÑ‚Ğ¾Ğ´ .train() Ğ¿Ñ€Ğ¸Ğ½Ğ¸Ğ¼Ğ°ĞµÑ‚ Ğ¼Ğ½Ğ¾Ğ¶ĞµÑ�Ñ‚Ğ²Ğ¾ Ğ°Ñ€Ğ³ÑƒĞ¼ĞµĞ½Ñ‚Ğ¾Ğ² Ğ´Ğ»Ñ� Ğ½Ğ°Ñ�Ñ‚Ñ€Ğ¾Ğ¹ĞºĞ¸ Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ñ�.
    results = model.train(
        data=yaml_path,        # ĞŸÑƒÑ‚ÑŒ Ğº YAML-Ñ„Ğ°Ğ¹Ğ»Ñƒ Ğ´Ğ°Ñ‚Ğ°Ñ�ĞµÑ‚Ğ°, ĞºĞ¾Ñ‚Ğ¾Ñ€Ñ‹Ğ¹ Ñ�Ğ¾Ğ´ĞµÑ€Ğ¶Ğ¸Ñ‚ Ğ¿ÑƒÑ‚Ğ¸ Ğº train/val/test Ğ´Ğ°Ğ½Ğ½Ñ‹Ğ¼
                               # Ğ¸ Ğ¸Ğ½Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ†Ğ¸Ñ� Ğ¾ ĞºĞ»Ğ°Ñ�Ñ�Ğ°Ñ….
        epochs=epochs,         # ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ñ�Ğ¿Ğ¾Ñ…, Ğ² Ñ‚ĞµÑ‡ĞµĞ½Ğ¸Ğµ ĞºĞ¾Ñ‚Ğ¾Ñ€Ñ‹Ñ… Ğ±ÑƒĞ´ĞµÑ‚ Ğ¿Ñ€Ğ¾Ğ¸Ñ�Ñ…Ğ¾Ğ´Ğ¸Ñ‚ÑŒ Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ğµ.
        batch=batch_size,      # ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ğ¸Ğ·Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ğ¹, Ğ¾Ğ±Ñ€Ğ°Ğ±Ğ°Ñ‚Ñ‹Ğ²Ğ°ĞµĞ¼Ñ‹Ñ… Ğ·Ğ° Ğ¾Ğ´Ğ½Ñƒ Ğ¸Ñ‚ĞµÑ€Ğ°Ñ†Ğ¸Ñ� Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ñ�.
                               # Ğ‘Ğ¾Ğ»ÑŒÑˆĞ¸Ğ¹ Ğ±Ğ°Ñ‚Ñ‡ Ñ‚Ñ€ĞµĞ±ÑƒĞµÑ‚ Ğ±Ğ¾Ğ»ÑŒÑˆĞµ Ğ¿Ğ°Ğ¼Ñ�Ñ‚Ğ¸ GPU, Ğ½Ğ¾ Ğ¼Ğ¾Ğ¶ĞµÑ‚ Ğ±Ñ‹Ñ‚ÑŒ Ñ�Ñ‚Ğ°Ğ±Ğ¸Ğ»ÑŒĞ½ĞµĞµ.
        imgsz=img_size,        # Ğ Ğ°Ğ·Ğ¼ĞµÑ€ Ğ¸Ğ·Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ñ�, Ğ´Ğ¾ ĞºĞ¾Ñ‚Ğ¾Ñ€Ğ¾Ğ³Ğ¾ Ğ±ÑƒĞ´ÑƒÑ‚ Ğ¼Ğ°Ñ�ÑˆÑ‚Ğ°Ğ±Ğ¸Ñ€Ğ¾Ğ²Ğ°Ñ‚ÑŒÑ�Ñ� Ğ²Ñ�Ğµ Ğ²Ñ…Ğ¾Ğ´Ğ½Ñ‹Ğµ Ğ¸Ğ·Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ñ�
                               # Ğ¿ĞµÑ€ĞµĞ´ Ğ¿Ğ¾Ğ´Ğ°Ñ‡ĞµĞ¹ Ğ² Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ. Ğ�Ğ±Ñ‹Ñ‡Ğ½Ğ¾ 640x640 Ğ¸Ğ»Ğ¸ 1280x1280.
        project=yolo_weights_dir, # Ğ�Ñ�Ğ½Ğ¾Ğ²Ğ½Ğ°Ñ� Ğ´Ğ¸Ñ€ĞµĞºÑ‚Ğ¾Ñ€Ğ¸Ñ�, ĞºÑƒĞ´Ğ° Ğ±ÑƒĞ´ÑƒÑ‚ Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�Ñ‚ÑŒÑ�Ñ� Ñ€ĞµĞ·ÑƒĞ»ÑŒÑ‚Ğ°Ñ‚Ñ‹ Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ñ�
                               # (Ğ»Ğ¾Ğ³Ğ¸, Ğ³Ñ€Ğ°Ñ„Ğ¸ĞºĞ¸, Ğ²ĞµÑ�Ğ° Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸).
        name='motor_detector', # Ğ˜Ğ¼Ñ� Ğ¿Ğ¾Ğ´Ğ´Ğ¸Ñ€ĞµĞºÑ‚Ğ¾Ñ€Ğ¸Ğ¸ Ğ²Ğ½ÑƒÑ‚Ñ€Ğ¸ 'project', Ğ³Ğ´Ğµ Ğ±ÑƒĞ´ÑƒÑ‚ Ñ…Ñ€Ğ°Ğ½Ğ¸Ñ‚ÑŒÑ�Ñ� Ñ€ĞµĞ·ÑƒĞ»ÑŒÑ‚Ğ°Ñ‚Ñ‹
                               # ĞºĞ¾Ğ½ĞºÑ€ĞµÑ‚Ğ½Ğ¾Ğ³Ğ¾ Ğ·Ğ°Ğ¿ÑƒÑ�ĞºĞ° Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ñ�.
        exist_ok=True,         # Ğ•Ñ�Ğ»Ğ¸ True, Ñ‚Ğ¾ Ğ´Ğ¸Ñ€ĞµĞºÑ‚Ğ¾Ñ€Ğ¸Ñ� 'motor_detector' Ğ±ÑƒĞ´ĞµÑ‚ Ğ¿ĞµÑ€ĞµĞ·Ğ°Ğ¿Ğ¸Ñ�Ğ°Ğ½Ğ°,
                               # ĞµÑ�Ğ»Ğ¸ Ğ¾Ğ½Ğ° ÑƒĞ¶Ğµ Ñ�ÑƒÑ‰ĞµÑ�Ñ‚Ğ²ÑƒĞµÑ‚. Ğ•Ñ�Ğ»Ğ¸ False, Ğ²Ğ¾Ğ·Ğ½Ğ¸ĞºĞ½ĞµÑ‚ Ğ¾ÑˆĞ¸Ğ±ĞºĞ°.
        patience=5,            # ĞŸĞ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€ "Ñ€Ğ°Ğ½Ğ½ĞµĞ¹ Ğ¾Ñ�Ñ‚Ğ°Ğ½Ğ¾Ğ²ĞºĞ¸". Ğ�Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ğµ Ğ±ÑƒĞ´ĞµÑ‚ Ğ¾Ñ�Ñ‚Ğ°Ğ½Ğ¾Ğ²Ğ»ĞµĞ½Ğ¾,
                               # ĞµÑ�Ğ»Ğ¸ Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¾Ğ½Ğ½Ğ°Ñ� Ğ¼ĞµÑ‚Ñ€Ğ¸ĞºĞ° (Ğ½Ğ°Ğ¿Ñ€Ğ¸Ğ¼ĞµÑ€, mAP) Ğ½Ğµ ÑƒĞ»ÑƒÑ‡ÑˆĞ°ĞµÑ‚Ñ�Ñ�
                               # Ğ² Ñ‚ĞµÑ‡ĞµĞ½Ğ¸Ğµ ÑƒĞºĞ°Ğ·Ğ°Ğ½Ğ½Ğ¾Ğ³Ğ¾ ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ° Ñ�Ğ¿Ğ¾Ñ….
        save_period=5,         # Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�Ñ‚ÑŒ Ğ²ĞµÑ�Ğ° Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸ ĞºĞ°Ğ¶Ğ´Ñ‹Ğµ 5 Ñ�Ğ¿Ğ¾Ñ….
        val=True,              # Ğ’ĞºĞ»Ñ�Ñ‡Ğ¸Ñ‚ÑŒ Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ñ� Ğ½Ğ° Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¾Ğ½Ğ½Ğ¾Ğ¼ Ğ½Ğ°Ğ±Ğ¾Ñ€Ğµ Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ… Ğ¿Ğ¾Ñ�Ğ»Ğµ ĞºĞ°Ğ¶Ğ´Ğ¾Ğ¹ Ñ�Ğ¿Ğ¾Ñ…Ğ¸.
        verbose=True           # Ğ’Ñ‹Ğ²Ğ¾Ğ´Ğ¸Ñ‚ÑŒ Ğ¿Ğ¾Ğ´Ñ€Ğ¾Ğ±Ğ½ÑƒÑ� Ğ¸Ğ½Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ†Ğ¸Ñ� Ğ¾ Ğ¿Ñ€Ğ¾Ñ†ĞµÑ�Ñ�Ğµ Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ñ� Ğ² ĞºĞ¾Ğ½Ñ�Ğ¾Ğ»ÑŒ.
    )

    # --- Ğ�Ğ½Ğ°Ğ»Ğ¸Ğ· Ñ€ĞµĞ·ÑƒĞ»ÑŒÑ‚Ğ°Ñ‚Ğ¾Ğ² Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ñ� ---
    # Ğ¤Ğ¾Ñ€Ğ¼Ğ¸Ñ€ÑƒĞµĞ¼ Ğ¿ÑƒÑ‚ÑŒ Ğº Ğ´Ğ¸Ñ€ĞµĞºÑ‚Ğ¾Ñ€Ğ¸Ğ¸, Ğ³Ğ´Ğµ Ğ±Ñ‹Ğ»Ğ¸ Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½ĞµĞ½Ñ‹ Ñ€ĞµĞ·ÑƒĞ»ÑŒÑ‚Ğ°Ñ‚Ñ‹ Ñ‚ĞµĞºÑƒÑ‰ĞµĞ³Ğ¾ Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ñ�.
    # Ğ­Ñ‚Ğ¾ Ğ²Ğ°Ğ¶Ğ½Ğ¾, Ñ‚Ğ°Ğº ĞºĞ°Ğº plot_dfl_loss_curve Ğ¾Ğ¶Ğ¸Ğ´Ğ°ĞµÑ‚ Ğ¿ÑƒÑ‚ÑŒ Ğº Ñ�Ñ‚Ğ¾Ğ¹ Ğ´Ğ¸Ñ€ĞµĞºÑ‚Ğ¾Ñ€Ğ¸Ğ¸.
    run_dir = os.path.join(yolo_weights_dir, 'motor_detector')
    # Ğ’Ñ‹Ğ·Ñ‹Ğ²Ğ°ĞµĞ¼ Ñ„ÑƒĞ½ĞºÑ†Ğ¸Ñ� Ğ´Ğ»Ñ� Ğ¿Ğ¾Ñ�Ñ‚Ñ€Ğ¾ĞµĞ½Ğ¸Ñ� Ğ³Ñ€Ğ°Ñ„Ğ¸ĞºĞ° Ğ¿Ğ¾Ñ‚ĞµÑ€ÑŒ DFL.
    # Ğ­Ñ‚Ğ° Ñ„ÑƒĞ½ĞºÑ†Ğ¸Ñ� Ñ‚Ğ°ĞºĞ¶Ğµ Ğ²Ğ¾Ğ·Ğ²Ñ€Ğ°Ñ‰Ğ°ĞµÑ‚ Ğ¸Ğ½Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ†Ğ¸Ñ� Ğ¾ Ğ»ÑƒÑ‡ÑˆĞµĞ¹ Ñ�Ğ¿Ğ¾Ñ…Ğµ Ğ¸ Ñ�Ğ¾Ğ¾Ñ‚Ğ²ĞµÑ‚Ñ�Ñ‚Ğ²ÑƒÑ�Ñ‰ĞµĞ¹ Ğ¿Ğ¾Ñ‚ĞµÑ€Ğµ.
    best_epoch_info = plot_dfl_loss_curve(run_dir)

    # Ğ•Ñ�Ğ»Ğ¸ Ğ¸Ğ½Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ†Ğ¸Ñ� Ğ¾ Ğ»ÑƒÑ‡ÑˆĞµĞ¹ Ñ�Ğ¿Ğ¾Ñ…Ğµ Ğ±Ñ‹Ğ»Ğ° ÑƒÑ�Ğ¿ĞµÑˆĞ½Ğ¾ Ğ¿Ğ¾Ğ»ÑƒÑ‡ĞµĞ½Ğ° (Ñ„ÑƒĞ½ĞºÑ†Ğ¸Ñ� Ğ½Ğµ Ğ²ĞµÑ€Ğ½ÑƒĞ»Ğ° None).
    if best_epoch_info:
        # Ğ Ğ°Ñ�Ğ¿Ğ°ĞºĞ¾Ğ²Ñ‹Ğ²Ğ°ĞµĞ¼ Ğ¿Ğ¾Ğ»ÑƒÑ‡ĞµĞ½Ğ½Ñ‹Ğµ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ñ�.
        best_epoch, best_val_loss = best_epoch_info
        # Ğ’Ñ‹Ğ²Ğ¾Ğ´Ğ¸Ğ¼ Ğ¸Ğ½Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ†Ğ¸Ñ� Ğ¾ Ğ»ÑƒÑ‡ÑˆĞµĞ¹ Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸.
        print(f"\nBest model found at epoch {best_epoch} with validation DFL loss: {best_val_loss:.4f}")

    # Ğ’Ğ¾Ğ·Ğ²Ñ€Ğ°Ñ‰Ğ°ĞµĞ¼ Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ½ÑƒÑ� Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ Ğ¸ Ğ¾Ğ±ÑŠĞµĞºÑ‚ Ñ€ĞµĞ·ÑƒĞ»ÑŒÑ‚Ğ°Ñ‚Ğ¾Ğ².
    # Ğ�Ğ±ÑƒÑ‡ĞµĞ½Ğ½Ğ°Ñ� Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ Ğ¼Ğ¾Ğ¶ĞµÑ‚ Ğ±Ñ‹Ñ‚ÑŒ Ğ¸Ñ�Ğ¿Ğ¾Ğ»ÑŒĞ·Ğ¾Ğ²Ğ°Ğ½Ğ° Ğ´Ğ»Ñ� Ğ¸Ğ½Ñ„ĞµÑ€ĞµĞ½Ñ�Ğ°.
    return model, results



def predict_on_samples(model, num_samples=4):
   
    val_dir = os.path.join(yolo_dataset_dir, 'images', 'val')

    if not os.path.exists(val_dir):
        print('No images directory found for predictions')
        return 
    val_images = os.listdir(val_dir)
    if len(val_images) == 0:
        print("No images found for prediction")
        return 
    num_samples = min(num_samples, len(val_images))
    samples = random.sample(val_images, num_samples)

    fig, axes = plt.subplots(2, 2, figsize=(12,12))
    axes = axes.flatten()
    for i, img_file in enumerate(samples):
        if i >= len(axes):
            break 
        img_path = os.path.join(val_dir, img_file)
        results = model.predict(img_path, conf=0.25)[0]

        img = Image.open(img_path)       
        axes[i].imshow(np.array(img), cmap='gray')

        try:
            parts = img_file.split('_')
            y_part = [p for p in parts if p.startswith('y')]
            x_part = [p for p in parts if p.startswith('x')]

            if y_part and x_part: 
                y_gt = int(y_part[0][1:])
                x_gt = int(x_part[0][1:].split('.')[0])
                box_size = 24 
                rect_gt = Rectangle((x_gt - box_size//2, y_gt - box_size//2), box_size, box_size,
                                   linewidth=1, edgecolor='g', gacecolor='none')
                axes[i].add_patch(rect_gt)
        except:
            pass
        if len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = box 
                rect_pred = Rectangle((x1,y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
                axes[i].add_patch(rect_pred)
                axes[i].text(x1, y1-5, f'{conf:.2f}', color='red')
        axes[i].set_title(f'Image:{img_file}\nGT (green) vs Pred (red)')
    plt.tight_layout()
    plt.savefig(os.path.join('/kaggle/working', 'predictions.png'))
    plt.show()


def prepare_dataset():

    train_images_dir = os.path.join(yolo_dataset_dir, 'images', 'train')
    val_images_dir = os.path.join(yolo_dataset_dir, 'images', 'val')
    train_labels_dir = os.path.join(yolo_dataset_dir, 'labels', 'train')
    val_labels_dir = os.path.join(yolo_dataset_dir, 'labels', 'val')

    print(f"Directory status:")
    print(f"- Train images exists: {os.path.exists(train_images_dir)}")
    print(f"- Val images exists: {os.path.exists(val_images_dir)}")
    print(f"- Train labels exists: {os.path.exists(train_labels_dir)}")
    print(f"- Val labels exists: {os.path.exists(val_labels_dir)}")


    original_yaml_path = os.path.join(yolo_dataset_dir, 'dataset.yaml')

    if os.path.exists(original_yaml_path):
        print(f"Found original dataset.yaml at {original_yaml_path}")
        return fix_yaml_paths(original_yaml_path)
    else:
        print("Original dataset.yaml not found, creating a new one")
        yaml_data = {
            'path': yolo_dataset_dir, 
            'train': 'images/train',   
            'val': 'images/train' if not os.path.exists(val_images_dir) else 'images/val',
            'names': {0: 'motor'}      
        }
        new_yaml_path = "/kaggle/working/dataset.yaml"
        with open(new_yaml_path, 'w') as f:
            yaml.dump(yaml_data, f)
        print(f"Created new YAML at {new_yaml_path}")
        return new_yaml_path

def main():
    
    print("Starting YOLO training process...")

    yaml_path = prepare_dataset()
    print(f"Using YAML file: {yaml_path}")

    with open(yaml_path, 'r') as f:
        print(f"YAML contents:\n{f.read()}")

    print("\nStarting YOLO training...")
    model, results = train_yolo_model(
        yaml_path,
        pretrained_weights_path=yolo_pretrained_weights,
        epochs=30  
    )

    print("\nTraining complete!")
    print("\nRunning predictions on sample images...")
    predict_on_samples(model, num_samples=4)

if __name__ == "__main__":
    main()



np.random.seed(42)          
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42) 

data_path = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/"
test_dir = os.path.join(data_path, "test")
submission_path = "/kaggle/working/submission.csv"
model_path = "/kaggle/working/yolo_weights/motor_detector/weights/best.pt"
CONFIDENCE_THRESHOLD = 0.45
MAX_DETECTIONS_PER_TOMO = 3
NMS_IOU_THRESHOLD = 0.2
CONCENTRATION = 1  
class GPUProfiler:
    def __init__(self, name):
        self.name = name            
        self.start_time = None      
        
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize() 
        self.start_time = time.time() 
        return self                

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize() 
        elapsed = time.time() - self.start_time 
        print(f"[PROFILE] {self.name}: {elapsed:.3f}s") 

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8
if device.startswith('cuda'):
    torch.backends.cudnn.benchmark = True        
    torch.backends.cudnn.deterministic = False    
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True      
    gpu_name = torch.cuda.get_device_name(0)      
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9 
    print(f"Using GPU: {gpu_name} with {gpu_mem:.2f} GB memory")
    free_mem = gpu_mem - torch.cuda.memory_allocated(0) / 1e9 
    BATCH_SIZE = max(8, min(32, int(free_mem * 4)))
    print(f"Dynamic batch size set to {BATCH_SIZE} based on {free_mem:.2f}GB free memory")
else:
    print("GPU not available, using CPU")
    BATCH_SIZE = 4


def normalize_slice(slice_data):    
    p2 = np.percentile(slice_data, 2)
    p98 = np.percentile(slice_data, 98)
    clipped_data = np.clip(slice_data, p2, p98)
    normalized = 255 * (clipped_data - p2) / (p98 - p2)
    return np.uint8(normalized)

def preload_image_batch(file_paths):
    images = []
    for path in file_paths:
        img = cv2.imread(path)
        if img is None:
            img = np.array(Image.open(path))
        images.append(img)
    return images

def perform_3d_nms(detections, iou_threshold):
   
    if not detections:
        return []

    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    final_detections = []

    def distance_3d(d1, d2):
        return np.sqrt((d1['z'] - d2['z'])**2 + (d1['y'] - d2['y'])**2 + (d1['x'] - d2['x'])**2)

    box_size = 24
    distance_threshold = box_size * iou_threshold

    while detections:
        best_detection = detections.pop(0)
        final_detections.append(best_detection)
        detections = [d for d in detections if distance_3d(d, best_detection) > distance_threshold]

    return final_detections

def process_tomogram(tomo_id, model, index=0, total=1):
    
    print(f"Processing tomogram {tomo_id} ({index}/{total})")
    tomo_dir = os.path.join(test_dir, tomo_id)
    slice_files = sorted([f for f in os.listdir(tomo_dir) if f.endswith('.jpg')])

    selected_indices = np.linspace(0, len(slice_files)-1, int(len(slice_files) * CONCENTRATION))
    selected_indices = np.round(selected_indices).astype(int)
    slice_files = [slice_files[i] for i in selected_indices]

    print(f"Processing {len(slice_files)} out of {len(os.listdir(tomo_dir))} slices (CONCENTRATION={CONCENTRATION})")
    all_detections = []

    if device.startswith('cuda'):
        streams = [torch.cuda.Stream() for _ in range(min(4, BATCH_SIZE))]
    else:
        streams = [None]

    next_batch_thread = None
    next_batch_images = None

    for batch_start in range(0, len(slice_files), BATCH_SIZE):
        if next_batch_thread is not None:
            next_batch_thread.join()
            next_batch_images = None

        batch_end = min(batch_start + BATCH_SIZE, len(slice_files))
        batch_files = slice_files[batch_start:batch_end]

        next_batch_start = batch_end
        next_batch_end = min(next_batch_start + BATCH_SIZE, len(slice_files))
        next_batch_files = slice_files[next_batch_start:next_batch_end] if next_batch_start < len(slice_files) else []
        if next_batch_files:
            next_batch_paths = [os.path.join(tomo_dir, f) for f in next_batch_files]
            next_batch_thread = threading.Thread(target=preload_image_batch, args=(next_batch_paths,))
            next_batch_thread.start()
        else:
            next_batch_thread = None

        sub_batches = np.array_split(batch_files, len(streams))
        for i, sub_batch in enumerate(sub_batches):
            if len(sub_batch) == 0:
                continue
            stream = streams[i % len(streams)]
            with torch.cuda.stream(stream) if stream and device.startswith('cuda') else nullcontext():
                sub_batch_paths = [os.path.join(tomo_dir, slice_file) for slice_file in sub_batch]
                sub_batch_slice_nums = [int(slice_file.split('_')[1].split('.')[0]) for slice_file in sub_batch]
                with GPUProfiler(f"Inference batch {i+1}/{len(sub_batches)}"):
                    sub_results = model(sub_batch_paths, verbose=False)
                for j, result in enumerate(sub_results):
                    if len(result.boxes) > 0:
                        for box_idx, confidence in enumerate(result.boxes.conf):
                            if confidence >= CONFIDENCE_THRESHOLD:
                                x1, y1, x2, y2 = result.boxes.xyxy[box_idx].cpu().numpy()
                                x_center = (x1 + x2) / 2
                                y_center = (y1 + y2) / 2
                                all_detections.append({
                                    'z': round(sub_batch_slice_nums[j]), 
                                    'y': round(y_center),               
                                    'x': round(x_center),               
                                    'confidence': float(confidence)     
                                })
        if device.startswith('cuda'):
            torch.cuda.synchronize()

    if next_batch_thread is not None:
        next_batch_thread.join()

    final_detections = perform_3d_nms(all_detections, NMS_IOU_THRESHOLD)
    final_detections.sort(key=lambda x: x['confidence'], reverse=True)

    if not final_detections:
        return {'tomo_id': tomo_id, 'Motor axis 0': -1, 'Motor axis 1': -1, 'Motor axis 2': -1}

    best_detection = final_detections[0]
    return {
        'tomo_id': tomo_id,
        'Motor axis 0': round(best_detection['z']), 
        'Motor axis 1': round(best_detection['y']), 
        'Motor axis 2': round(best_detection['x'])  
    }

def debug_image_loading(tomo_id):
    
    tomo_dir = os.path.join(test_dir, tomo_id)
    slice_files = sorted([f for f in os.listdir(tomo_dir) if f.endswith('.jpg')])
    if not slice_files:
        print(f"No image files found in {tomo_dir}")
        return

    print(f"Found {len(slice_files)} image files in {tomo_dir}")
    sample_file = slice_files[len(slice_files)//2]
    img_path = os.path.join(tomo_dir, sample_file)

    try:
        img_pil = Image.open(img_path)
        print(f"PIL Image shape: {np.array(img_pil).shape}, dtype: {np.array(img_pil).dtype}")
        img_cv2 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        print(f"OpenCV Image shape: {img_cv2.shape}, dtype: {img_cv2.dtype}")
        img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        print(f"OpenCV RGB Image shape: {img_rgb.shape}, dtype: {img_rgb.dtype}")
        print("Image loading successful!")
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")

    try:
        test_model = YOLO(model_path)
        test_results = test_model([img_path], verbose=False)
        print("YOLO model successfully processed the test image")
    except Exception as e:
        print(f"Error with YOLO processing: {e}")


def generate_submission():
    
    test_tomos = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    total_tomos = len(test_tomos) # Ğ�Ğ±Ñ‰ĞµĞµ ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ñ‚ĞµÑ�Ñ‚Ğ¾Ğ²Ñ‹Ñ… Ñ‚Ğ¾Ğ¼Ğ¾Ğ³Ñ€Ğ°Ğ¼Ğ¼.
    print(f"Found {total_tomos} tomograms in test directory")
  
    if test_tomos:
        debug_image_loading(test_tomos[0])
    
   
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"Loading YOLO model from {model_path}")
    model = YOLO(model_path) 
    model.to(device)        
    
    if device.startswith('cuda'):
        model.fuse() 
        
        if torch.cuda.get_device_capability(0)[0] >= 7:
            model.model.half() 
            print("Using half precision (FP16) for inference")
    
    results = []       
    motors_found = 0   
  
    with ThreadPoolExecutor(max_workers=1) as executor:
        future_to_tomo = {}
        for i, tomo_id in enumerate(test_tomos, 1):
            future = executor.submit(process_tomogram, tomo_id, model, i, total_tomos)
            future_to_tomo[future] = tomo_id # Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµĞ¼ Future Ğ¸ ID Ñ‚Ğ¾Ğ¼Ğ¾Ğ³Ñ€Ğ°Ğ¼Ğ¼Ñ‹.
        
        for future in future_to_tomo:
            tomo_id = future_to_tomo[future] 
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                result = future.result() 
                results.append(result)   
                
                has_motor = not pd.isna(result['Motor axis 0'])
                if has_motor:
                    motors_found += 1 
                    print(f"Motor found in {tomo_id} at position: z={result['Motor axis 0']}, y={result['Motor axis 1']}, x={result['Motor axis 2']}")
                else:
                    print(f"No motor detected in {tomo_id}")
                print(f"Current detection rate: {motors_found}/{len(results)} ({motors_found/len(results)*100:.1f}%)")
            except Exception as e:
                print(f"Error processing {tomo_id}: {e}")
                results.append({'tomo_id': tomo_id, 'Motor axis 0': -1, 'Motor axis 1': -1, 'Motor axis 2': -1})
    
    submission_df = pd.DataFrame(results)
    submission_df = submission_df[['tomo_id', 'Motor axis 0', 'Motor axis 1', 'Motor axis 2']]
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\nSubmission complete!")
    print(f"Motors detected: {motors_found}/{total_tomos} ({motors_found/total_tomos*100:.1f}%)")
    print(f"Submission saved to: {submission_path}")
    print("\nSubmission preview:")
    print(submission_df.head()) 
    return submission_df 


if __name__ == "__main__":
    
    start_time = time.time()
    submission = generate_submission()
    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")    




