import os
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def clean_working(directory_path: str = "/kaggle/working/"):
    """
    Clean kaggle output directory.
    """
    if os.path.exists(directory_path):
        for item in os.listdir(directory_path):
            if item == "submission.csv":
                continue
            item_path = os.path.join(directory_path, item)
            os.remove(item_path) if os.path.isfile(item_path) else shutil.rmtree(item_path)
        print(f"All items in '{directory_path}' have been removed.")
    else:
        print(f"'{directory_path}' does not exist.")
        
clean_working()


!pip install zarr cryoet_data_portal -q


import pandas as pd 
pd.read_csv("/kaggle/input/cryoet-flagellar-motors-dataset/labels.csv")


import os
import numpy as np
import pandas as pd
from PIL import Image
import yaml
from pathlib import Path
from tqdm.notebook import tqdm

# Set paths
data_path = "/kaggle/input/cryoet-flagellar-motors-dataset/"
volumes_dir = os.path.join(data_path, "volumes")
labels_path = os.path.join(data_path, "labels.csv")

# YOLO dataset structure
yolo_dataset_dir = "/kaggle/working/yolo_dataset"
yolo_images_train = os.path.join(yolo_dataset_dir, "images", "train")
yolo_images_val = os.path.join(yolo_dataset_dir, "images", "val")
yolo_labels_train = os.path.join(yolo_dataset_dir, "labels", "train")
yolo_labels_val = os.path.join(yolo_dataset_dir, "labels", "val")

# Create directories
for dir_path in [yolo_images_train, yolo_images_val, yolo_labels_train, yolo_labels_val]:
    os.makedirs(dir_path, exist_ok=True)

# Constants
TRUST = 4  # Number of slices above and below the center slice
BOX_SIZE = 24  # Bounding box size in pixels
TRAIN_SPLIT = 0.8  # 80% training, 20% validation

def normalize_slice(slice_data):
    """Normalize slice data using the 2nd and 98th percentiles."""
    p2, p98 = np.percentile(slice_data, [2, 98])
    clipped = np.clip(slice_data, p2, p98)
    normalized = 255 * (clipped - p2) / (p98 - p2)
    return np.uint8(normalized)

def prepare_yolo_dataset():
    """Prepare dataset for YOLO training."""
    labels_df = pd.read_csv(labels_path)
    unique_tomos = labels_df["tomo_id"].unique()
    np.random.shuffle(unique_tomos)
    split_idx = int(len(unique_tomos) * TRAIN_SPLIT)
    train_tomos, val_tomos = unique_tomos[:split_idx], unique_tomos[split_idx:]

    def process_tomograms(tomogram_ids, images_dir, labels_dir, set_name):
        processed_slices = 0
        for tomo_id in tqdm(tomogram_ids, desc=f"Processing {set_name} set"):
            volume_path = os.path.join(volumes_dir, f"{tomo_id}.npy")
            if not os.path.exists(volume_path):
                print(f"Warning: {volume_path} not found, skipping.")
                continue
            volume = np.load(volume_path)  # Load 3D volume
            tomo_motors = labels_df[labels_df["tomo_id"] == tomo_id]
            for _, motor in tomo_motors.iterrows():
                z_center, y_center, x_center = int(motor["z"]), int(motor["y"]), int(motor["x"])
                z_min, z_max = max(0, z_center - TRUST), min(volume.shape[0] - 1, z_center + TRUST)
                for z in range(z_min, z_max + 1):
                    slice_data = volume[z]
                    normalized_img = normalize_slice(slice_data)
                    dest_filename = f"{tomo_id}_z{z:04d}_y{y_center:04d}_x{x_center:04d}.jpg"
                    Image.fromarray(normalized_img).save(os.path.join(images_dir, dest_filename))
                    img_height, img_width = slice_data.shape
                    x_norm, y_norm = x_center / img_width, y_center / img_height
                    box_w_norm, box_h_norm = BOX_SIZE / img_width, BOX_SIZE / img_height
                    with open(os.path.join(labels_dir, dest_filename.replace('.jpg', '.txt')), 'w') as f:
                        f.write(f"0 {x_norm} {y_norm} {box_w_norm} {box_h_norm}\n")
                    processed_slices += 1
        return processed_slices

    train_slices = process_tomograms(train_tomos, yolo_images_train, yolo_labels_train, "training")
    val_slices = process_tomograms(val_tomos, yolo_images_val, yolo_labels_val, "validation")
    yaml_content = {
        'path': yolo_dataset_dir,
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'motor'}
    }
    with open(os.path.join(yolo_dataset_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    print(f"\nDataset ready: {train_slices} training slices, {val_slices} validation slices.")

prepare_yolo_dataset()


