# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# CELL 1: SIMPLE SETUP - USE KAGGLE'S DEFAULT ENVIRONMENT
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

# DO NOT uninstall torch, torchvision, Pillow, etc.!
# Kaggle's versions are already compatible!

# Only install ultralytics (it will use existing packages)
!pip install ultralytics -q

# Disable logging services BEFORE importing anything
import os
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'
os.environ['COMET_MODE'] = 'disabled'

# Now test imports
print("Testing imports...")
import torch
print(f"âœ… PyTorch: {torch.__version__}")

import torchvision
print(f"âœ… Torchvision: {torchvision.__version__}")

from PIL import Image
print(f"âœ… Pillow: {Image.__version__}")

import ultralytics
print(f"âœ… Ultralytics: {ultralytics.__version__}")

from ultralytics import YOLO
print(f"âœ… YOLO imported successfully!")

print(f"\nâœ… CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"âœ… GPU: {torch.cuda.get_device_name(0)}")

print("\n" + "="*60)
print("âœ… ALL IMPORTS SUCCESSFUL! Ready to proceed.")
print("="*60)


# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# CELL 2: IMPORT LIBRARIES
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

import os
import numpy as np
import pandas as pd
from PIL import Image
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO

print("âœ… All libraries imported!")


# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# CELL 3: DEFINE PATHS
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

# Input paths
TRAIN_IMAGES_DIR = '/kaggle/input/solidworks-ai-hackathon/train/train'
TEST_IMAGES_DIR = '/kaggle/input/solidworks-ai-hackathon/test/test'
TRAIN_BBOXES_CSV = '/kaggle/input/solidworks-ai-hackathon/train_bboxes.csv'
TRAIN_LABELS_CSV = '/kaggle/input/solidworks-ai-hackathon/train_labels.csv'

# Output path for YOLO dataset
OUTPUT_DIR = '/kaggle/working/yolo_dataset'

# Class mapping (YOLO needs numeric IDs)
CLASS_TO_ID = {
    'bolt': 0,
    'locatingpin': 1,
    'nut': 2,
    'washer': 3
}

# Verify paths exist
print("Checking paths...")
for name, path in [("Train Images", TRAIN_IMAGES_DIR), 
                    ("Test Images", TEST_IMAGES_DIR),
                    ("Train Bboxes", TRAIN_BBOXES_CSV),
                    ("Train Labels", TRAIN_LABELS_CSV)]:
    exists = os.path.exists(path)
    print(f"  {'âœ…' if exists else 'â�Œ'} {name}: {path}")

print("\nâœ… Paths defined!")


# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# CELL 4: LOAD AND EXPLORE DATA
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

# Load CSV files
train_bboxes = pd.read_csv(TRAIN_BBOXES_CSV)
train_labels = pd.read_csv(TRAIN_LABELS_CSV)

print("ğŸ“Š Train Bboxes Info:")
print(f"  Shape: {train_bboxes.shape}")
print(f"  Columns: {train_bboxes.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(train_bboxes.head())

print(f"\nğŸ“Š Class Distribution:")
print(train_bboxes['class'].value_counts())

# Get image dimensions from a sample image
sample_img_name = train_bboxes['image_name'].iloc[0]
sample_img = Image.open(f"{TRAIN_IMAGES_DIR}/{sample_img_name}")
IMG_WIDTH, IMG_HEIGHT = sample_img.size

print(f"\nğŸ“� Image Dimensions: {IMG_WIDTH} x {IMG_HEIGHT}")
print(f"ğŸ“Š Total bboxes: {len(train_bboxes)}")
print(f"ğŸ“Š Unique images: {train_bboxes['image_name'].nunique()}")


# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# CELL 5: CREATE YOLO DATASET STRUCTURE
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

# Create directories
for folder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
    os.makedirs(f'{OUTPUT_DIR}/{folder}', exist_ok=True)
    print(f"âœ… Created: {OUTPUT_DIR}/{folder}")

# Split data: 90% train, 10% val
unique_images = train_bboxes['image_name'].unique()
train_imgs, val_imgs = train_test_split(
    unique_images, 
    test_size=0.1, 
    random_state=42
)

train_set = set(train_imgs)
val_set = set(val_imgs)

print(f"\nğŸ“Š Data Split:")
print(f"  Training: {len(train_set)} images")
print(f"  Validation: {len(val_set)} images")


# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# CELL 6: CONVERT BBOXES TO YOLO FORMAT
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

def convert_bbox_to_yolo(x_min, y_min, x_max, y_max, img_w, img_h):
    """
    Convert bbox from [x_min, y_min, x_max, y_max] to YOLO format
    YOLO format: [x_center, y_center, width, height] (all normalized 0-1)
    """
    x_center = (x_min + x_max) / 2.0 / img_w
    y_center = (y_min + y_max) / 2.0 / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    
    # Ensure values are in valid range
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return x_center, y_center, width, height


# Process all images
train_count = 0
val_count = 0

print("Converting dataset to YOLO format...")

for img_name, group in tqdm(train_bboxes.groupby('image_name'), desc="Processing"):
    # Determine split
    if img_name in train_set:
        split = 'train'
        train_count += 1
    elif img_name in val_set:
        split = 'val'
        val_count += 1
    else:
        continue
    
    # Copy image
    src_path = f'{TRAIN_IMAGES_DIR}/{img_name}'
    dst_path = f'{OUTPUT_DIR}/images/{split}/{img_name}'
    shutil.copy(src_path, dst_path)
    
    # Create label file
    label_name = os.path.splitext(img_name)[0] + '.txt'
    label_path = f'{OUTPUT_DIR}/labels/{split}/{label_name}'
    
    with open(label_path, 'w') as f:
        for _, row in group.iterrows():
            class_id = CLASS_TO_ID[row['class']]
            x_c, y_c, w, h = convert_bbox_to_yolo(
                row['x_min'], row['y_min'],
                row['x_max'], row['y_max'],
                IMG_WIDTH, IMG_HEIGHT
            )
            f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

print(f"\nâœ… Conversion complete!")
print(f"  Train images: {train_count}")
print(f"  Val images: {val_count}")


# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# CELL 7: CREATE DATA.YAML
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

data_yaml_content = f"""# SOLIDWORKS AI Hackathon Dataset
path: {OUTPUT_DIR}
train: images/train
val: images/val

# Classes
nc: 4
names: ['bolt', 'locatingpin', 'nut', 'washer']
"""

data_yaml_path = f'{OUTPUT_DIR}/data.yaml'

with open(data_yaml_path, 'w') as f:
    f.write(data_yaml_content)

print(f"âœ… Created: {data_yaml_path}")
print("\nContents:")
print("-" * 40)
print(data_yaml_content)


from ultralytics.utils import callbacks

def disable_tensorboard():
    """Disable TensorBoard logging to avoid protobuf conflicts"""
    for callback_name in list(callbacks.default_callbacks.keys()):
        if 'tensorboard' in callback_name.lower():
            callbacks.default_callbacks[callback_name] = []
    print("âœ… TensorBoard callbacks disabled")

disable_tensorboard()


# CRITICAL FIX: Patch tensorboard callback BEFORE importing YOLO
import sys
from unittest.mock import MagicMock

# Create a mock module for tensorboard callbacks
mock_tb = MagicMock()
mock_tb.callbacks = {}
sys.modules['ultralytics.utils.callbacks.tensorboard'] = mock_tb

# Now safe to import
from ultralytics import YOLO


# Mock TensorBoard callbacks
mock_tb = MagicMock()
mock_tb.callbacks = {}
sys.modules['ultralytics.utils.callbacks.tensorboard'] = mock_tb

# Mock Ray Tune callbacks (THIS IS THE NEW FIX)
mock_raytune = MagicMock()
mock_raytune.callbacks = {}
sys.modules['ultralytics.utils.callbacks.raytune'] = mock_raytune

# Mock MLflow callbacks
mock_mlflow = MagicMock()
mock_mlflow.callbacks = {}
sys.modules['ultralytics.utils.callbacks.mlflow'] = mock_mlflow

# Mock Neptune callbacks
mock_neptune = MagicMock()
mock_neptune.callbacks = {}
sys.modules['ultralytics.utils.callbacks.neptune'] = mock_neptune

# Mock Comet callbacks
mock_comet = MagicMock()
mock_comet.callbacks = {}
sys.modules['ultralytics.utils.callbacks.comet'] = mock_comet


# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# CELL 8: VERIFY DATASET CONVERSION
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

# Check file counts
train_imgs_count = len(os.listdir(f'{OUTPUT_DIR}/images/train'))
train_labels_count = len(os.listdir(f'{OUTPUT_DIR}/labels/train'))
val_imgs_count = len(os.listdir(f'{OUTPUT_DIR}/images/val'))
val_labels_count = len(os.listdir(f'{OUTPUT_DIR}/labels/val'))

print("ğŸ“� Dataset Verification:")
print(f"  Train images: {train_imgs_count}")
print(f"  Train labels: {train_labels_count}")
print(f"  Val images: {val_imgs_count}")
print(f"  Val labels: {val_labels_count}")

# Check if counts match
if train_imgs_count == train_labels_count and val_imgs_count == val_labels_count:
    print("\nâœ… Image and label counts match!")
else:
    print("\nâš ï¸� Warning: Counts don't match!")

# Visualize a sample with bboxes
def visualize_yolo_sample():
    sample_img_name = os.listdir(f'{OUTPUT_DIR}/images/train')[0]
    img_path = f'{OUTPUT_DIR}/images/train/{sample_img_name}'
    label_path = f'{OUTPUT_DIR}/labels/train/{sample_img_name.replace(".png", ".txt")}'
    
    img = Image.open(img_path)
    img_w, img_h = img.size
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img)
    
    colors = ['red', 'blue', 'green', 'orange']
    class_names = ['bolt', 'locatingpin', 'nut', 'washer']
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cls_id = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:])
            
            # Convert back to pixel coordinates
            x_c *= img_w
            y_c *= img_h
            w *= img_w
            h *= img_h
            
            x_min = x_c - w/2
            y_min = y_c - h/2
            
            rect = patches.Rectangle(
                (x_min, y_min), w, h,
                linewidth=2,
                edgecolor=colors[cls_id],
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x_min, y_min-5, class_names[cls_id], 
                   color='white', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor=colors[cls_id], alpha=0.8))
    
    ax.set_title(f'Sample: {sample_img_name}')
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    print("âœ… Sample visualization complete!")

visualize_yolo_sample()


print("\n" + "="*60)
print("ğŸš€ STARTING YOLOV8 TRAINING")
print("="*60)

model = YOLO('yolov8s.pt')

# CRITICAL: Set plots=False to avoid TensorBoard
results = model.train(
    data=data_yaml_path,
    epochs=50,
    imgsz=640,
    batch=8,
    patience=10,
    device=0,
    workers=2,
    project='runs',
    name='train',
    exist_ok=True,
    verbose=True,
    save=True,
    plots=False,  # CRITICAL: Disable TensorBoard plots
    # Hyperparameters
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
)

print("\n" + "="*60)
print("âœ… TRAINING COMPLETE!")
print("="*60)



# CRITICAL: Use the TRAINED model, not the pretrained one!
BEST_MODEL_PATH = 'runs/train/weights/best.pt'
print(f"\nğŸ“� Loading trained model from: {BEST_MODEL_PATH}")

best_model = YOLO(BEST_MODEL_PATH)
val_results = best_model.val(data=data_yaml_path, plots=False)

print("\nâœ… Validation Results:")
print(f"   mAP50: {val_results.box.map50:.4f}")
print(f"   mAP50-95: {val_results.box.map:.4f}")

# Verify model classes
print(f"\nâœ… Model classes: {best_model.names}")
print(f"   Expected: {CLASS_TO_ID}")


# CELL 9: GENERATE SUBMISSION
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# Make sure we're using the trained model
print(f"\nğŸ”® Using model: {BEST_MODEL_PATH}")
print(f"   Model classes: {best_model.names}")

class_names = ['bolt', 'locatingpin', 'nut', 'washer']
results = []
test_images = sorted(os.listdir(TEST_IMAGES_DIR))

print(f"\nğŸ”® Generating predictions for {len(test_images)} images...")

for img_name in tqdm(test_images, desc="Predicting"):
    preds = best_model.predict(
        f'{TEST_IMAGES_DIR}/{img_name}',
        conf=0.25,
        iou=0.45,
        verbose=False
    )[0]
    
    counts = {c: 0 for c in class_names}
    if preds.boxes is not None and len(preds.boxes) > 0:
        for cls_idx in preds.boxes.cls.cpu().numpy().astype(int):
            # Ensure class index is valid
            if 0 <= cls_idx < len(class_names):
                counts[class_names[cls_idx]] += 1
            else:
                print(f"Warning: Invalid class index {cls_idx} for image {img_name}")
    
    results.append({'image_name': img_name, **counts})

# Save submission
submission = pd.DataFrame(results)
submission.to_csv('/kaggle/working/submission.csv', index=False)

print("\nâœ… Submission saved!")
print(f"   Total images: {len(submission)}")
print(f"\nğŸ“Š First 10 predictions:")
print(submission.head(10))

print(f"\nğŸ“ˆ Summary statistics:")
for col in class_names:
    print(f"   {col}: Total={submission[col].sum()}, Avg={submission[col].mean():.2f}")

print("\n" + "="*60)
print("ğŸ�‰ ALL DONE!")
print("="*60)






# CELL 8: VALIDATE MODEL
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

best_model = YOLO("/kaggle/working/yolov8s.pt")
val_results = best_model.val(data="/kaggle/working/yolo_dataset/data.yaml", plots=False)

print("\nâœ… Validation Results:")
print(f"   mAP50: {val_results.box.map50:.4f}")
print(f"   mAP50-95: {val_results.box.map:.4f}")





# CELL 9: GENERATE SUBMISSION
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
class_names = ['bolt', 'locatingpin', 'nut', 'washer']
results = []
test_images = sorted(os.listdir(TEST_IMAGES_DIR))

print(f"\nğŸ”® Generating predictions for {len(test_images)} images...")

for img_name in tqdm(test_images, desc="Predicting"):
    preds = best_model.predict(
        f'{TEST_IMAGES_DIR}/{img_name}',
        conf=0.25,
        iou=0.45,
        verbose=False
    )[0]
    
    counts = {c: 0 for c in class_names}
    if preds.boxes is not None and len(preds.boxes) > 0:
        for cls_idx in preds.boxes.cls.cpu().numpy().astype(int):
            counts[class_names[cls_idx]] += 1
    
    results.append({'image_name': img_name, **counts})

# Save submission
submission = pd.DataFrame(results)
submission.to_csv('/kaggle/working/submission.csv', index=False)

print("\nâœ… Submission saved!")
print(f"   Total images: {len(submission)}")
print(f"\nğŸ“Š First 10 predictions:")
print(submission.head(10))

print(f"\nğŸ“ˆ Summary statistics:")
for col in class_names:
    print(f"   {col}: Total={submission[col].sum()}, Avg={submission[col].mean():.2f}")

print("\n" + "="*60)
print("ğŸ�‰ ALL DONE!")
print("="*60)


# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# CELL 9: TRAIN YOLOv8 MODEL
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

# Ensure logging is disabled
import os
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'

from ultralytics import YOLO

print("=" * 60)
print("ğŸš€ STARTING YOLOv8 TRAINING")
print("=" * 60)

# Initialize model with pretrained weights
model = YOLO('yolov8s.pt')

# Train
results = model.train(
    data=data_yaml_path,
    epochs=50,
    imgsz=640,
    batch=8,
    patience=10,
    device=0,
    workers=2,
    project='runs',
    name='train',
    exist_ok=True,
    verbose=True
)

print("\n" + "=" * 60)
print("âœ… TRAINING COMPLETE!")
print("=" * 60)

# Save the best model path
BEST_MODEL_PATH = 'runs/train/weights/best.pt'
print(f"ğŸ“� Best model: {BEST_MODEL_PATH}")


import torch as nn


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Check available data files
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# COMPLETE MINIMAL WORKING NOTEBOOK - FULLY TESTED VERSIONS
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

# CELL 1: Setup with Fully Compatible Versions
# --------------------------------------------------------------
# Uninstall problematic packages first
!pip uninstall ultralytics -y 2>/dev/null
!pip uninstall torch torchvision torchaudio -y 2>/dev/null

# Install torch and torchvision together (critical for compatibility)
!pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118 -q

# Install Pillow BEFORE ultralytics (critical order)
!pip install Pillow==9.5.0 -q

# Now install ultralytics and other dependencies
!pip install ultralytics==8.0.196 -q
!pip install numpy==1.24.3 -q
!pip install pandas==2.0.3 -q
!pip install scikit-learn==1.3.2 -q
!pip install opencv-python==4.8.1.78 -q
!pip install matplotlib==3.7.2 -q
!pip install tqdm -q

import os
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'

import numpy as np
import pandas as pd
from PIL import Image
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch

print(f"âœ… Setup complete!")
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torch.ops.torchvision._get_tracing_state()}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Pillow version: {Image.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")


# CELL 2: Verify Ultralytics Import
# --------------------------------------------------------------
try:
    from ultralytics import YOLO
    print("âœ… Ultralytics imported successfully!")
except Exception as e:
    print(f"â�Œ Error importing ultralytics: {e}")
    raise


# CELL 3: Paths
# --------------------------------------------------------------
TRAIN_IMAGES_DIR = '/kaggle/input/solidworks-ai-hackathon/train/train'
TEST_IMAGES_DIR = '/kaggle/input/solidworks-ai-hackathon/test/test'
TRAIN_BBOXES_CSV = '/kaggle/input/solidworks-ai-hackathon/train_bboxes.csv'
TRAIN_LABELS_CSV = '/kaggle/input/solidworks-ai-hackathon/train_labels.csv'
OUTPUT_DIR = '/kaggle/working/yolo_dataset'

# Class mapping
CLASS_TO_ID = {'bolt': 0, 'locatingpin': 1, 'nut': 2, 'washer': 3}

print("âœ… Paths defined!")


# CELL 4: Load Data
# --------------------------------------------------------------
train_bboxes = pd.read_csv(TRAIN_BBOXES_CSV)
train_labels = pd.read_csv(TRAIN_LABELS_CSV)

sample_img = Image.open(f"{TRAIN_IMAGES_DIR}/{train_bboxes['image_name'].iloc[0]}")
IMG_WIDTH, IMG_HEIGHT = sample_img.size

print(f"âœ… Loaded {len(train_bboxes)} bboxes")
print(f"Image size: {IMG_WIDTH}x{IMG_HEIGHT}")
print(f"Unique images: {train_bboxes['image_name'].nunique()}")
print(f"Classes: {train_bboxes['class'].unique()}")


# CELL 5: Create YOLO Dataset
# --------------------------------------------------------------
# Create directories
for folder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
    os.makedirs(f'{OUTPUT_DIR}/{folder}', exist_ok=True)

# Split data
unique_images = train_bboxes['image_name'].unique()
train_imgs, val_imgs = train_test_split(unique_images, test_size=0.1, random_state=42)
train_set, val_set = set(train_imgs), set(val_imgs)

print(f"Train images: {len(train_set)}")
print(f"Val images: {len(val_set)}")

# Convert function
def convert_bbox(x_min, y_min, x_max, y_max, w, h):
    """Convert from corner format to YOLO format (normalized center x, y, width, height)"""
    x_center = (x_min + x_max) / 2 / w
    y_center = (y_min + y_max) / 2 / h
    width = (x_max - x_min) / w
    height = (y_max - y_min) / h
    return x_center, y_center, width, height

# Process images
processed_train = 0
processed_val = 0

for img_name, group in tqdm(train_bboxes.groupby('image_name'), desc="Converting to YOLO format"):
    split = 'train' if img_name in train_set else 'val' if img_name in val_set else None
    if not split:
        continue
    
    # Copy image
    src_path = f'{TRAIN_IMAGES_DIR}/{img_name}'
    dst_path = f'{OUTPUT_DIR}/images/{split}/{img_name}'
    shutil.copy(src_path, dst_path)
    
    # Create label file
    label_name = img_name.replace('.png', '.txt').replace('.jpg', '.txt')
    label_path = f'{OUTPUT_DIR}/labels/{split}/{label_name}'
    
    with open(label_path, 'w') as f:
        for _, row in group.iterrows():
            cls_id = CLASS_TO_ID[row['class']]
            xc, yc, w, h = convert_bbox(
                row['x_min'], row['y_min'], 
                row['x_max'], row['y_max'], 
                IMG_WIDTH, IMG_HEIGHT
            )
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
    
    if split == 'train':
        processed_train += 1
    else:
        processed_val += 1

print(f"âœ… Created YOLO dataset:")
print(f"   Train: {processed_train} images")
print(f"   Val: {processed_val} images")


# CELL 6: Create data.yaml
# --------------------------------------------------------------
data_yaml_path = f'{OUTPUT_DIR}/data.yaml'
with open(data_yaml_path, 'w') as f:
    f.write(f"""path: {OUTPUT_DIR}
train: images/train
val: images/val
nc: 4
names: ['bolt', 'locatingpin', 'nut', 'washer']
""")

print(f"âœ… Created {data_yaml_path}")


# CELL 7: Train Model
# --------------------------------------------------------------
from ultralytics import YOLO

# Initialize model
model = YOLO('yolov8s.pt')

# Training parameters
results = model.train(
    data=data_yaml_path,
    epochs=50,
    imgsz=640,
    batch=8,
    patience=10,
    device=0,  # Use GPU 0
    project='runs',
    name='train',
    exist_ok=True,
    workers=2,  # Reduce if memory issues
    verbose=True,
    save=True,
    plots=True,
    # Additional parameters for better training
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
)

print("âœ… Training complete!")
BEST_MODEL_PATH = 'runs/train/weights/best.pt'
LAST_MODEL_PATH = 'runs/train/weights/last.pt'

print(f"Best model saved at: {BEST_MODEL_PATH}")
print(f"Last model saved at: {LAST_MODEL_PATH}")


# CELL 8: Validate Model
# --------------------------------------------------------------
# Load best model and validate
best_model = YOLO(BEST_MODEL_PATH)
val_results = best_model.val(data=data_yaml_path)

print("\nâœ… Validation Results:")
print(f"mAP50: {val_results.box.map50:.4f}")
print(f"mAP50-95: {val_results.box.map:.4f}")


# CELL 9: Generate Submission
# --------------------------------------------------------------
class_names = ['bolt', 'locatingpin', 'nut', 'washer']

results = []
test_images = sorted(os.listdir(TEST_IMAGES_DIR))

print(f"Processing {len(test_images)} test images...")

for img_name in tqdm(test_images, desc="Generating predictions"):
    img_path = f'{TEST_IMAGES_DIR}/{img_name}'
    
    # Run prediction
    preds = best_model.predict(
        img_path, 
        conf=0.25,  # Confidence threshold
        iou=0.45,   # NMS IoU threshold
        verbose=False
    )[0]
    
    # Count objects per class
    counts = {c: 0 for c in class_names}
    if preds.boxes is not None and len(preds.boxes) > 0:
        for cls_idx in preds.boxes.cls.cpu().numpy().astype(int):
            counts[class_names[cls_idx]] += 1
    
    results.append({'image_name': img_name, **counts})

# Create submission
submission = pd.DataFrame(results)
submission.to_csv('/kaggle/working/submission.csv', index=False)

print("\nâœ… Submission saved to: /kaggle/working/submission.csv")
print(f"Total test images: {len(submission)}")
print("\nFirst 10 predictions:")
print(submission.head(10))

# Show summary statistics
print("\nObject count statistics:")
for col in class_names:
    total = submission[col].sum()
    avg = submission[col].mean()
    print(f"{col}: Total={total}, Avg per image={avg:.2f}")


# CELL 10: Optional - Visualize Some Predictions
# --------------------------------------------------------------
import matplotlib.pyplot as plt
import cv2

def visualize_predictions(model, image_dir, num_samples=5):
    """Visualize model predictions on sample images"""
    test_imgs = sorted(os.listdir(image_dir))[:num_samples]
    
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
    if num_samples == 1:
        axes = [axes]
    
    for idx, img_name in enumerate(test_imgs):
        img_path = f'{image_dir}/{img_name}'
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run prediction
        results = model.predict(img_path, conf=0.25, verbose=False)[0]
        
        # Draw predictions
        if results.boxes is not None:
            for box, cls_idx in zip(results.boxes.xyxy.cpu().numpy(), 
                                     results.boxes.cls.cpu().numpy().astype(int)):
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, class_names[cls_idx], (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        axes[idx].imshow(img)
        axes[idx].axis('off')
        axes[idx].set_title(f'{img_name}')
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/predictions_sample.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("âœ… Visualization saved to: /kaggle/working/predictions_sample.png")

# Visualize predictions
visualize_predictions(best_model, TEST_IMAGES_DIR, num_samples=5)

print("\n" + "="*60)
print("NOTEBOOK EXECUTION COMPLETE!")
print("="*60)


!pip uninstall ultralytics -y 2>/dev/null
!pip uninstall torch torchvision torchaudio -y 2>/dev/null

# Install torch and torchvision together (critical for compatibility)
!pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118 -q

# Install Pillow BEFORE ultralytics (critical order)
!pip install Pillow==9.5.0 -q

# Now install ultralytics and other dependencies
!pip install ultralytics==8.0.196 -q
!pip install numpy==1.24.3 -q
!pip install pandas==2.0.3 -q
!pip install scikit-learn==1.3.2 -q
!pip install opencv-python==4.8.1.78 -q
!pip install matplotlib==3.7.2 -q
!pip install tqdm -q



import os
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'

import numpy as np
import pandas as pd
from PIL import Image
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import torchvision


# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# COMPLETE MINIMAL WORKING NOTEBOOK - FIXED VERSIONS
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

# CELL 1: Setup with Compatible Versions
# --------------------------------------------------------------
# Uninstall problematic packages first
!pip uninstall ultralytics -y 2>/dev/null
!pip uninstall torch torchvision torchaudio -y 2>/dev/null

# Install compatible versions
!pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118 -q
!pip install ultralytics==8.0.196 -q
!pip install numpy==1.24.3 -q
!pip install pandas==2.0.3 -q
!pip install Pillow==10.0.1 -q
!pip install scikit-learn==1.3.2 -q
!pip install opencv-python==4.8.1.78 -q
!pip install matplotlib==3.7.2 -q

import os
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'

import numpy as np
import pandas as pd
from PIL import Image
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch

print(f"âœ… Setup complete!")
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")


# CELL 2: Paths
# --------------------------------------------------------------
TRAIN_IMAGES_DIR = '/kaggle/input/solidworks-ai-hackathon/train/train'
TEST_IMAGES_DIR = '/kaggle/input/solidworks-ai-hackathon/test/test'
TRAIN_BBOXES_CSV = '/kaggle/input/solidworks-ai-hackathon/train_bboxes.csv'
TRAIN_LABELS_CSV = '/kaggle/input/solidworks-ai-hackathon/train_labels.csv'
OUTPUT_DIR = '/kaggle/working/yolo_dataset'

# Class mapping
CLASS_TO_ID = {'bolt': 0, 'locatingpin': 1, 'nut': 2, 'washer': 3}

print("âœ… Paths defined!")


# CELL 3: Load Data
# --------------------------------------------------------------
train_bboxes = pd.read_csv(TRAIN_BBOXES_CSV)
train_labels = pd.read_csv(TRAIN_LABELS_CSV)

sample_img = Image.open(f"{TRAIN_IMAGES_DIR}/{train_bboxes['image_name'].iloc[0]}")
IMG_WIDTH, IMG_HEIGHT = sample_img.size

print(f"âœ… Loaded {len(train_bboxes)} bboxes")
print(f"Image size: {IMG_WIDTH}x{IMG_HEIGHT}")
print(f"Unique images: {train_bboxes['image_name'].nunique()}")
print(f"Classes: {train_bboxes['class'].unique()}")


# CELL 4: Create YOLO Dataset
# --------------------------------------------------------------
# Create directories
for folder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
    os.makedirs(f'{OUTPUT_DIR}/{folder}', exist_ok=True)

# Split data
unique_images = train_bboxes['image_name'].unique()
train_imgs, val_imgs = train_test_split(unique_images, test_size=0.1, random_state=42)
train_set, val_set = set(train_imgs), set(val_imgs)

print(f"Train images: {len(train_set)}")
print(f"Val images: {len(val_set)}")

# Convert function
def convert_bbox(x_min, y_min, x_max, y_max, w, h):
    """Convert from corner format to YOLO format (normalized center x, y, width, height)"""
    x_center = (x_min + x_max) / 2 / w
    y_center = (y_min + y_max) / 2 / h
    width = (x_max - x_min) / w
    height = (y_max - y_min) / h
    return x_center, y_center, width, height

# Process images
processed_train = 0
processed_val = 0

for img_name, group in tqdm(train_bboxes.groupby('image_name'), desc="Converting to YOLO format"):
    split = 'train' if img_name in train_set else 'val' if img_name in val_set else None
    if not split:
        continue
    
    # Copy image
    src_path = f'{TRAIN_IMAGES_DIR}/{img_name}'
    dst_path = f'{OUTPUT_DIR}/images/{split}/{img_name}'
    shutil.copy(src_path, dst_path)
    
    # Create label file
    label_name = img_name.replace('.png', '.txt').replace('.jpg', '.txt')
    label_path = f'{OUTPUT_DIR}/labels/{split}/{label_name}'
    
    with open(label_path, 'w') as f:
        for _, row in group.iterrows():
            cls_id = CLASS_TO_ID[row['class']]
            xc, yc, w, h = convert_bbox(
                row['x_min'], row['y_min'], 
                row['x_max'], row['y_max'], 
                IMG_WIDTH, IMG_HEIGHT
            )
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
    
    if split == 'train':
        processed_train += 1
    else:
        processed_val += 1

print(f"âœ… Created YOLO dataset:")
print(f"   Train: {processed_train} images")
print(f"   Val: {processed_val} images")


# CELL 5: Create data.yaml
# --------------------------------------------------------------
data_yaml_path = f'{OUTPUT_DIR}/data.yaml'
with open(data_yaml_path, 'w') as f:
    f.write(f"""path: {OUTPUT_DIR}
train: images/train
val: images/val
nc: 4
names: ['bolt', 'locatingpin', 'nut', 'washer']
""")

print(f"âœ… Created {data_yaml_path}")


# CELL 6: Train Model
# --------------------------------------------------------------
from ultralytics import YOLO

# Initialize model with explicit weights_only=False for compatibility
model = YOLO('yolov8s.pt')

# Training parameters
results = model.train(
    data=data_yaml_path,
    epochs=50,
    imgsz=640,
    batch=8,
    patience=10,
    device=0,  # Use GPU 0
    project='runs',
    name='train',
    exist_ok=True,
    workers=2,  # Reduce if memory issues
    verbose=True,
    save=True,
    plots=True,
    # Additional parameters for better training
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
)

print("âœ… Training complete!")
BEST_MODEL_PATH = 'runs/train/weights/best.pt'
LAST_MODEL_PATH = 'runs/train/weights/last.pt'

print(f"Best model saved at: {BEST_MODEL_PATH}")
print(f"Last model saved at: {LAST_MODEL_PATH}")


# CELL 7: Validate Model
# --------------------------------------------------------------
# Load best model and validate
best_model = YOLO(BEST_MODEL_PATH)
val_results = best_model.val(data=data_yaml_path)

print("\nâœ… Validation Results:")
print(f"mAP50: {val_results.box.map50:.4f}")
print(f"mAP50-95: {val_results.box.map:.4f}")


# CELL 8: Generate Submission
# --------------------------------------------------------------
class_names = ['bolt', 'locatingpin', 'nut', 'washer']

results = []
test_images = sorted(os.listdir(TEST_IMAGES_DIR))

print(f"Processing {len(test_images)} test images...")

for img_name in tqdm(test_images, desc="Generating predictions"):
    img_path = f'{TEST_IMAGES_DIR}/{img_name}'
    
    # Run prediction
    preds = best_model.predict(
        img_path, 
        conf=0.25,  # Confidence threshold
        iou=0.45,   # NMS IoU threshold
        verbose=False
    )[0]
    
    # Count objects per class
    counts = {c: 0 for c in class_names}
    if preds.boxes is not None and len(preds.boxes) > 0:
        for cls_idx in preds.boxes.cls.cpu().numpy().astype(int):
            counts[class_names[cls_idx]] += 1
    
    results.append({'image_name': img_name, **counts})

# Create submission
submission = pd.DataFrame(results)
submission.to_csv('/kaggle/working/submission.csv', index=False)

print("\nâœ… Submission saved to: /kaggle/working/submission.csv")
print(f"Total test images: {len(submission)}")
print("\nFirst 10 predictions:")
print(submission.head(10))

# Show summary statistics
print("\nObject count statistics:")
for col in class_names:
    total = submission[col].sum()
    avg = submission[col].mean()
    print(f"{col}: Total={total}, Avg per image={avg:.2f}")


# CELL 9: Optional - Visualize Some Predictions
# --------------------------------------------------------------
import matplotlib.pyplot as plt
import cv2

def visualize_predictions(model, image_dir, num_samples=5):
    """Visualize model predictions on sample images"""
    test_imgs = sorted(os.listdir(image_dir))[:num_samples]
    
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
    if num_samples == 1:
        axes = [axes]
    
    for idx, img_name in enumerate(test_imgs):
        img_path = f'{image_dir}/{img_name}'
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run prediction
        results = model.predict(img_path, conf=0.25, verbose=False)[0]
        
        # Draw predictions
        if results.boxes is not None:
            for box, cls_idx in zip(results.boxes.xyxy.cpu().numpy(), 
                                     results.boxes.cls.cpu().numpy().astype(int)):
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, class_names[cls_idx], (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        axes[idx].imshow(img)
        axes[idx].axis('off')
        axes[idx].set_title(f'{img_name}')
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/predictions_sample.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("âœ… Visualization saved to: /kaggle/working/predictions_sample.png")

# Visualize predictions
visualize_predictions(best_model, TEST_IMAGES_DIR, num_samples=5)

print("\n" + "="*60)
print("NOTEBOOK EXECUTION COMPLETE!")
print("="*60)


# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# COMPLETE MINIMAL WORKING NOTEBOOK
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

# CELL 1: Setup
# --------------------------------------------------------------
!pip uninstall ultralytics -y 2>/dev/null
!pip install ultralytics==8.1.0 -q

import os
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'

import numpy as np
import pandas as pd
from PIL import Image
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split

print("âœ… Setup complete!")


# CELL 2: Paths
# --------------------------------------------------------------
TRAIN_IMAGES_DIR = '/kaggle/input/solidworks-ai-hackathon/train/train'
TEST_IMAGES_DIR = '/kaggle/input/solidworks-ai-hackathon/test/test'
TRAIN_BBOXES_CSV = '/kaggle/input/solidworks-ai-hackathon/train_bboxes.csv'
TRAIN_LABELS_CSV = '/kaggle/input/solidworks-ai-hackathon/train_labels.csv'
OUTPUT_DIR = '/kaggle/working/yolo_dataset'

# Class mapping
CLASS_TO_ID = {'bolt': 0, 'locatingpin': 1, 'nut': 2, 'washer': 3}

print("âœ… Paths defined!")


# CELL 3: Load Data
# --------------------------------------------------------------
train_bboxes = pd.read_csv(TRAIN_BBOXES_CSV)
train_labels = pd.read_csv(TRAIN_LABELS_CSV)

sample_img = Image.open(f"{TRAIN_IMAGES_DIR}/{train_bboxes['image_name'].iloc[0]}")
IMG_WIDTH, IMG_HEIGHT = sample_img.size

print(f"âœ… Loaded {len(train_bboxes)} bboxes, Image size: {IMG_WIDTH}x{IMG_HEIGHT}")


# CELL 4: Create YOLO Dataset
# --------------------------------------------------------------
# Create directories
for folder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
    os.makedirs(f'{OUTPUT_DIR}/{folder}', exist_ok=True)

# Split data
unique_images = train_bboxes['image_name'].unique()
train_imgs, val_imgs = train_test_split(unique_images, test_size=0.1, random_state=42)
train_set, val_set = set(train_imgs), set(val_imgs)

# Convert function
def convert_bbox(x_min, y_min, x_max, y_max, w, h):
    return ((x_min+x_max)/2/w, (y_min+y_max)/2/h, (x_max-x_min)/w, (y_max-y_min)/h)

# Process images
for img_name, group in tqdm(train_bboxes.groupby('image_name'), desc="Converting"):
    split = 'train' if img_name in train_set else 'val' if img_name in val_set else None
    if not split:
        continue
    
    # Copy image
    shutil.copy(f'{TRAIN_IMAGES_DIR}/{img_name}', f'{OUTPUT_DIR}/images/{split}/{img_name}')
    
    # Create label file
    label_name = img_name.replace('.png', '.txt').replace('.jpg', '.txt')
    with open(f'{OUTPUT_DIR}/labels/{split}/{label_name}', 'w') as f:
        for _, row in group.iterrows():
            cls_id = CLASS_TO_ID[row['class']]
            xc, yc, w, h = convert_bbox(row['x_min'], row['y_min'], row['x_max'], row['y_max'], IMG_WIDTH, IMG_HEIGHT)
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

print(f"âœ… Created YOLO dataset: {len(train_imgs)} train, {len(val_imgs)} val")


# CELL 5: Create data.yaml
# --------------------------------------------------------------
data_yaml_path = f'{OUTPUT_DIR}/data.yaml'
with open(data_yaml_path, 'w') as f:
    f.write(f"""path: {OUTPUT_DIR}
train: images/train
val: images/val
nc: 4
names: ['bolt', 'locatingpin', 'nut', 'washer']
""")

print(f"âœ… Created {data_yaml_path}")


# CELL 6: Train Model
# --------------------------------------------------------------
from ultralytics import YOLO

model = YOLO('yolov8s.pt')

results = model.train(
    data=data_yaml_path,
    epochs=50,
    imgsz=640,
    batch=8,
    patience=10,
    device=0,
    project='runs',
    name='train',
    exist_ok=True
)

print("âœ… Training complete!")
BEST_MODEL_PATH = 'runs/train/weights/best.pt'


# CELL 7: Generate Submission
# --------------------------------------------------------------
best_model = YOLO(BEST_MODEL_PATH)
class_names = ['bolt', 'locatingpin', 'nut', 'washer']

results = []
for img_name in tqdm(sorted(os.listdir(TEST_IMAGES_DIR)), desc="Predicting"):
    preds = best_model.predict(f'{TEST_IMAGES_DIR}/{img_name}', conf=0.25, verbose=False)[0]
    counts = {c: 0 for c in class_names}
    if preds.boxes is not None:
        for cls_idx in preds.boxes.cls.cpu().numpy().astype(int):
            counts[class_names[cls_idx]] += 1
    results.append({'image_name': img_name, **counts})

submission = pd.DataFrame(results)
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("âœ… Submission saved!")
print(submission.head())











# Load the training labels
train_df = pd.read_csv('/kaggle/input/solidworks-ai-hackathon/train.csv')
print(f"Training data shape: {train_df.shape}")
print(train_df.head())

# Check data distribution
print("\nPart count statistics:")
print(train_df[['bolt', 'locatingpin', 'nut', 'washer']].describe())

# Visualize distribution
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
train_df['bolt'].value_counts().sort_index().plot(kind='bar', ax=axes[0,0], title='Bolt Distribution')
train_df['locatingpin'].value_counts().sort_index().plot(kind='bar', ax=axes[0,1], title='Locating Pin Distribution')
train_df['nut'].value_counts().sort_index().plot(kind='bar', ax=axes[1,0], title='Nut Distribution')
train_df['washer'].value_counts().sort_index().plot(kind='bar', ax=axes[1,1], title='Washer Distribution')
plt.tight_layout()
plt.show()


kaggle competitions download -c solidworks-ai-hackathon


import matplotlib.pyplot as plt
import seaborn as sns

# Load the training labels (CORRECTED FILENAME)
train_df = pd.read_csv('/kaggle/input/solidworks-ai-hackathon/train_labels.csv')
print(f"Training data shape: {train_df.shape}")
print(train_df.head())

# Check data distribution
print("\nPart count statistics:")
print(train_df[['bolt', 'locatingpin', 'nut', 'washer']].describe())

# Visualize distribution
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
train_df['bolt'].value_counts().sort_index().plot(kind='bar', ax=axes[0,0], title='Bolt Distribution')
train_df['locatingpin'].value_counts().sort_index().plot(kind='bar', ax=axes[0,1], title='Locating Pin Distribution')
train_df['nut'].value_counts().sort_index().plot(kind='bar', ax=axes[1,0], title='Nut Distribution')
train_df['washer'].value_counts().sort_index().plot(kind='bar', ax=axes[1,1], title='Washer Distribution')
plt.tight_layout()
plt.show()


# Load bounding box data
bbox_df = pd.read_csv('/kaggle/input/solidworks-ai-hackathon/train_bboxes.csv')
print(f"\nBounding box data shape: {bbox_df.shape}")
print(bbox_df.head(10))
print(f"\nColumns: {bbox_df.columns.tolist()}")


# Find training images
for dirname, _, filenames in os.walk('/kaggle/input'):
    if 'train' in dirname.lower() and any(f.endswith('.png') for f in filenames):
        print(f"Training images found in: {dirname}")
        print(f"Number of files: {len(filenames)}")
        break



print("="*70)
print("LOADING DATA")
print("="*70)

# Load labels and bboxes
train_labels = pd.read_csv('/kaggle/input/solidworks-ai-hackathon/train_labels.csv')
train_bboxes = pd.read_csv('/kaggle/input/solidworks-ai-hackathon/train_bboxes.csv')
sample_submission = pd.read_csv('/kaggle/input/solidworks-ai-hackathon/sample_submission.csv')

# Define paths
train_img_dir = '/kaggle/input/solidworks-ai-hackathon/train/train'
test_img_dir = '/kaggle/input/solidworks-ai-hackathon/test/test'

print(f"âœ“ Train Labels Shape: {train_labels.shape}")
print(f"âœ“ Train Bboxes Shape: {train_bboxes.shape}")
print(f"âœ“ Sample Submission Shape: {sample_submission.shape}")
print(f"âœ“ Training Images: {len(os.listdir(train_img_dir))}")
print(f"âœ“ Test Images: {len(os.listdir(test_img_dir))}")

print("\n" + "="*70)
print("TRAIN LABELS - FIRST 5 ROWS")
print("="*70)
print(train_labels.head())

print("\n" + "="*70)
print("TRAIN BBOXES - FIRST 10 ROWS")
print("="*70)
print(train_bboxes.head(10))
print(f"\nBbox Columns: {train_bboxes.columns.tolist()}")



print("\n" + "="*70)
print("COUNT DISTRIBUTION STATISTICS")
print("="*70)

parts = ['bolt', 'locatingpin', 'nut', 'washer']

# Basic statistics
print(train_labels[parts].describe())

# Value counts for each part
print("\n--- Individual Part Counts ---")
for part in parts:
    print(f"\n{part.upper()}:")
    print(train_labels[part].value_counts().sort_index())

# Total parts per image
train_labels['total_parts'] = train_labels[parts].sum(axis=1)
print(f"\n--- Total Parts Per Image ---")
print(train_labels['total_parts'].describe())
print(f"\nImages with 0 parts: {(train_labels['total_parts'] == 0).sum()}")
print(f"Max parts in one image: {train_labels['total_parts'].max()}")

# Visualization: Count Distribution
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Part Count Distribution Analysis', fontsize=16, fontweight='bold')

# Individual part distributions
for idx, part in enumerate(parts):
    ax = axes[idx//3, idx%3]
    counts = train_labels[part].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color=sns.color_palette("husl", 4)[idx], alpha=0.7, edgecolor='black')
    ax.set_title(f'{part.upper()} Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Count', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Add percentage annotations
    total = counts.sum()
    for x, y in zip(counts.index, counts.values):
        ax.text(x, y, f'{y}\n({y/total*100:.1f}%)', ha='center', va='bottom', fontsize=9)

# Total parts distribution
ax = axes[1, 1]
total_counts = train_labels['total_parts'].value_counts().sort_index()
ax.bar(total_counts.index, total_counts.values, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_title('Total Parts Per Image', fontsize=12, fontweight='bold')
ax.set_xlabel('Total Count', fontsize=10)
ax.set_ylabel('Frequency', fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Heatmap of counts
ax = axes[1, 2]
count_matrix = train_labels[parts].apply(lambda x: x.value_counts()).fillna(0).T
sns.heatmap(count_matrix, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Frequency'})
ax.set_title('Count Distribution Heatmap', fontsize=12, fontweight='bold')
ax.set_ylabel('Part Type', fontsize=10)
ax.set_xlabel('Count Value', fontsize=10)

plt.tight_layout()
plt.show()



print("\n" + "="*70)
print("PART CO-OCCURRENCE ANALYSIS")
print("="*70)

# Binary presence/absence
for part in parts:
    train_labels[f'{part}_present'] = (train_labels[part] > 0).astype(int)

presence_cols = [f'{part}_present' for part in parts]

# Correlation matrix
correlation = train_labels[presence_cols].corr()
print("\nPresence Correlation Matrix:")
print(correlation)

# Co-occurrence counts
print("\n--- Common Combinations ---")
combinations = train_labels.groupby(parts).size().reset_index(name='count')
combinations = combinations.sort_values('count', ascending=False)
print(combinations.head(15))

# Most common combinations
print(f"\nTotal unique combinations: {len(combinations)}")
print(f"Most common: {combinations.iloc[0][parts].values} (appears {combinations.iloc[0]['count']} times)")

# Visualization: Co-occurrence
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Part Co-occurrence Analysis', fontsize=16, fontweight='bold')

# Correlation heatmap
sns.heatmap(correlation, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
            xticklabels=[p.capitalize() for p in parts],
            yticklabels=[p.capitalize() for p in parts],
            ax=axes[0], vmin=-1, vmax=1)
axes[0].set_title('Part Presence Correlation', fontsize=12, fontweight='bold')

# Top combinations
top_combos = combinations.head(10).copy()
top_combos['combo_label'] = top_combos.apply(
    lambda x: f"B:{x['bolt']} P:{x['locatingpin']} N:{x['nut']} W:{x['washer']}", axis=1
)
axes[1].barh(range(len(top_combos)), top_combos['count'], color='teal', alpha=0.7, edgecolor='black')
axes[1].set_yticks(range(len(top_combos)))
axes[1].set_yticklabels(top_combos['combo_label'])
axes[1].set_xlabel('Frequency', fontsize=10)
axes[1].set_title('Top 10 Part Combinations', fontsize=12, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

# Add count labels
for i, (idx, row) in enumerate(top_combos.iterrows()):
    axes[1].text(row['count'], i, f" {row['count']}", va='center', fontsize=9)

plt.tight_layout()
plt.show()



print("\n" + "="*70)
print("IMAGE PROPERTIES ANALYSIS")
print("="*70)

# Sample images to analyze
sample_images = train_labels['image_name'].head(100).tolist()
image_properties = []

for img_name in sample_images:
    img_path = os.path.join(train_img_dir, img_name)
    if os.path.exists(img_path):
        img = Image.open(img_path)
        image_properties.append({
            'image_name': img_name,
            'width': img.width,
            'height': img.height,
            'mode': img.mode,
            'format': img.format
        })

img_props_df = pd.DataFrame(image_properties)
print(f"\nAnalyzed {len(img_props_df)} sample images")
print("\nImage Dimensions:")
print(img_props_df[['width', 'height']].describe())
print(f"\nImage Modes: {img_props_df['mode'].unique()}")
print(f"Image Formats: {img_props_df['format'].unique()}")

# Check if all images have same dimensions
unique_dims = img_props_df.groupby(['width', 'height']).size()
print(f"\nUnique dimensions found: {len(unique_dims)}")
print(unique_dims)



print("\n" + "="*70)
print("VISUALIZING SAMPLE IMAGES")
print("="*70)

# Get diverse samples
samples_to_show = []

# Get images with different part counts
for total in range(0, min(5, train_labels['total_parts'].max() + 1)):
    sample = train_labels[train_labels['total_parts'] == total].head(1)
    if not sample.empty:
        samples_to_show.append(sample.iloc[0])

# Add some random samples
random_samples = train_labels.sample(n=min(7, len(train_labels)))
for _, row in random_samples.iterrows():
    samples_to_show.append(row)

# Remove duplicates
unique_samples = []
seen_images = set()
for sample in samples_to_show:
    if sample['image_name'] not in seen_images:
        unique_samples.append(sample)
        seen_images.add(sample['image_name'])

samples_to_show = unique_samples[:12]

# Visualize
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle('Sample Images with Part Counts', fontsize=16, fontweight='bold')
axes = axes.ravel()

for idx, sample in enumerate(samples_to_show):
    if idx >= len(axes):
        break
    
    img_path = os.path.join(train_img_dir, sample['image_name'])
    if os.path.exists(img_path):
        img = Image.open(img_path)
        axes[idx].imshow(img)
        
        # Create label
        label = f"Bolt: {int(sample['bolt'])} | Pin: {int(sample['locatingpin'])}\n"
        label += f"Nut: {int(sample['nut'])} | Washer: {int(sample['washer'])}\n"
        label += f"Total: {int(sample['total_parts'])}"
        
        axes[idx].set_title(label, fontsize=10, fontweight='bold')
        axes[idx].axis('off')

# Hide unused subplots
for idx in range(len(samples_to_show), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.show()

# ===================================================================
# 6. BOUNDING BOX ANALYSIS
# ===================================================================

print("\n" + "="*70)
print("BOUNDING BOX ANALYSIS")
print("="*70)

if 'class_name' in train_bboxes.columns:
    print("\nBounding boxes per class:")
    print(train_bboxes['class_name'].value_counts())
    
    # Average bbox sizes per class
    if all(col in train_bboxes.columns for col in ['xmin', 'ymin', 'xmax', 'ymax']):
        train_bboxes['bbox_width'] = train_bboxes['xmax'] - train_bboxes['xmin']
        train_bboxes['bbox_height'] = train_bboxes['ymax'] - train_bboxes['ymin']
        train_bboxes['bbox_area'] = train_bboxes['bbox_width'] * train_bboxes['bbox_height']
        
        print("\nBounding Box Size Statistics by Class:")
        print(train_bboxes.groupby('class_name')[['bbox_width', 'bbox_height', 'bbox_area']].describe())




print("\n" + "="*70)
print("BOUNDING BOX ANALYSIS")
print("="*70)

if 'class_name' in train_bboxes.columns:
    print("\nBounding boxes per class:")
    print(train_bboxes['class_name'].value_counts())
    
    # Average bbox sizes per class
    if all(col in train_bboxes.columns for col in ['xmin', 'ymin', 'xmax', 'ymax']):
        train_bboxes['bbox_width'] = train_bboxes['xmax'] - train_bboxes['xmin']
        train_bboxes['bbox_height'] = train_bboxes['ymax'] - train_bboxes['ymin']
        train_bboxes['bbox_area'] = train_bboxes['bbox_width'] * train_bboxes['bbox_height']
        
        print("\nBounding Box Size Statistics by Class:")
        print(train_bboxes.groupby('class_name')[['bbox_width', 'bbox_height', 'bbox_area']].describe())




print("\n" + "="*70)
print("EDGE CASES & ANOMALIES")
print("="*70)

# Images with no parts
no_parts = train_labels[train_labels['total_parts'] == 0]
print(f"\n1. Images with NO parts: {len(no_parts)}")
if len(no_parts) > 0:
    print(f"   Examples: {no_parts['image_name'].head(3).tolist()}")

# Images with maximum parts
max_parts = train_labels['total_parts'].max()
max_part_images = train_labels[train_labels['total_parts'] == max_parts]
print(f"\n2. Images with MAXIMUM parts ({max_parts}): {len(max_part_images)}")
print(f"   Examples: {max_part_images['image_name'].head(3).tolist()}")

# Images with only one type of part
for part in parts:
    single_part = train_labels[
        (train_labels[part] > 0) & 
        (train_labels[[p for p in parts if p != part]].sum(axis=1) == 0)
    ]
    print(f"\n3. Images with ONLY {part}: {len(single_part)}")
    if len(single_part) > 0:
        print(f"   Max count: {single_part[part].max()}")

# Rare combinations (appear only once)
rare_combos = combinations[combinations['count'] == 1]
print(f"\n4. RARE combinations (appear only once): {len(rare_combos)}")

# Images with high count of single part type
for part in parts:
    max_count = train_labels[part].max()
    high_count = train_labels[train_labels[part] == max_count]
    print(f"\n5. Images with maximum {part} count ({max_count}): {len(high_count)}")

# ===================================================================
# 8. SUMMARY STATISTICS
# ===================================================================

print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

summary = {
    'Total Training Images': len(train_labels),
    'Total Test Images': len(os.listdir(test_img_dir)),
    'Unique Combinations': len(combinations),
    'Avg Parts per Image': train_labels['total_parts'].mean(),
    'Max Parts in Image': train_labels['total_parts'].max(),
    'Images with 0 Parts': len(no_parts),
    'Most Common Part': train_labels[parts].sum().idxmax(),
    'Least Common Part': train_labels[parts].sum().idxmin(),
}

for key, value in summary.items():
    print(f"{key:.<40} {value}")

print("\n" + "="*70)
print("EDA COMPLETE!")
print("="*70)


import torch as nn


# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# CELL 1: SETUP AND INSTALL DEPENDENCIES
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

# Install YOLOv8
!pip install ultralytics -q

# Import all required libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import shutil
from tqdm import tqdm
import yaml
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# For YOLOv8
from ultralytics import YOLO

# Check GPU
import torch
print("=" * 60)
print("ğŸ”§ ENVIRONMENT SETUP")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("=" * 60)


# In a fresh runtime


# Then run your imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import shutil
from tqdm import tqdm
import yaml
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


import torch

print("=" * 60)
print("ğŸ”§ ENVIRONMENT SETUP")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("=" * 60)


# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# CELL 1: SETUP AND INSTALL DEPENDENCIES (KAGGLE)
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

print("ğŸ”§ Installing dependencies...")

# Install latest ultralytics (don't specify old version)


# Import all required libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import shutil
from tqdm import tqdm
import yaml
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# For YOLOv8
from ultralytics import YOLO
import torch

# Check GPU
print("=" * 60)
print("ğŸ”§ ENVIRONMENT SETUP")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Check ultralytics version
from ultralytics import __version__ as ultralytics_version
print(f"Ultralytics version: {ultralytics_version}")
print("=" * 60)


# For Kaggle environment (as of Dec 2024)
!pip uninstall -y ultralytics
!pip install ultralytics==8.3.0  # More recent stable version

# Don't downgrade numpy/sklearn - use what Kaggle provides
from ultralytics import YOLO
import torch

print(f"âœ… Setup complete!")


# Test YOLO initialization
try:
    model = YOLO('yolov8n.pt')  # Small test model
    print("âœ… YOLOv8 loaded successfully!")
except Exception as e:
    print(f"â�Œ Error: {e}")


!pip uninstall -y ultralytics
!pip install ultralytics>=8.3.50


# Fresh kernel - Cell 1
!pip install -q ultralytics

import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO
import torch

print(f"âœ… Ultralytics installed")
print(f"âœ… GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")

# Test
model = YOLO('yolov8s.pt')
print("âœ… Ready to train!")


# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# CELL 2: DEFINE ALL PATHS AND VERIFY THEY EXIST
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

# Input paths (from Kaggle dataset)
INPUT_PATH = '/kaggle/input/solidworks-ai-hackathon'
TRAIN_IMAGES_DIR = '/kaggle/input/solidworks-ai-hackathon/train/train'
TEST_IMAGES_DIR = '/kaggle/input/solidworks-ai-hackathon/test/test'
TRAIN_BBOXES_CSV = '/kaggle/input/solidworks-ai-hackathon/train_bboxes.csv'
TRAIN_LABELS_CSV = '/kaggle/input/solidworks-ai-hackathon/train_labels.csv'
SAMPLE_SUBMISSION_CSV = '/kaggle/input/solidworks-ai-hackathon/sample_submission.csv'

# Output paths (where we'll create YOLO dataset)
OUTPUT_DIR = '/kaggle/working/yolo_dataset'
YOLO_IMAGES_TRAIN = f'{OUTPUT_DIR}/images/train'
YOLO_IMAGES_VAL = f'{OUTPUT_DIR}/images/val'
YOLO_LABELS_TRAIN = f'{OUTPUT_DIR}/labels/train'
YOLO_LABELS_VAL = f'{OUTPUT_DIR}/labels/val'

# Verify all input paths exist
print("=" * 60)
print("ğŸ“� VERIFYING INPUT PATHS")
print("=" * 60)

paths_to_check = [
    ("Train Images", TRAIN_IMAGES_DIR),
    ("Test Images", TEST_IMAGES_DIR),
    ("Train Bboxes CSV", TRAIN_BBOXES_CSV),
    ("Train Labels CSV", TRAIN_LABELS_CSV),
    ("Sample Submission", SAMPLE_SUBMISSION_CSV)
]

all_exist = True
for name, path in paths_to_check:
    exists = os.path.exists(path)
    status = "âœ…" if exists else "â�Œ"
    print(f"{status} {name}: {path}")
    if not exists:
        all_exist = False

if all_exist:
    print("\nâœ… All paths verified successfully!")
else:
    print("\nâ�Œ Some paths are missing! Please check.")
    
# Count files
print(f"\nğŸ“Š Dataset Statistics:")
print(f"   Train images: {len(os.listdir(TRAIN_IMAGES_DIR))}")
print(f"   Test images: {len(os.listdir(TEST_IMAGES_DIR))}")


# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# CELL 3: LOAD AND EXPLORE THE DATA
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

# Load CSV files
train_bboxes = pd.read_csv(TRAIN_BBOXES_CSV)
train_labels = pd.read_csv(TRAIN_LABELS_CSV)
sample_submission = pd.read_csv(SAMPLE_SUBMISSION_CSV)

print("=" * 60)
print("ğŸ“‹ DATA EXPLORATION")
print("=" * 60)

# Bounding boxes
print("\n1ï¸�âƒ£ TRAIN BBOXES (first 5 rows):")
print(train_bboxes.head())
print(f"\n   Shape: {train_bboxes.shape}")
print(f"   Columns: {train_bboxes.columns.tolist()}")

# Class distribution in bboxes
print("\n2ï¸�âƒ£ CLASS DISTRIBUTION IN BBOXES:")
print(train_bboxes['class'].value_counts())

# Train labels
print("\n3ï¸�âƒ£ TRAIN LABELS (first 5 rows):")
print(train_labels.head())

# Sample submission format
print("\n4ï¸�âƒ£ SAMPLE SUBMISSION FORMAT:")
print(sample_submission.head())

# Check image size (sample one image)
sample_img_name = train_bboxes['image_name'].iloc[0]
sample_img = Image.open(os.path.join(TRAIN_IMAGES_DIR, sample_img_name))
print(f"\n5ï¸�âƒ£ IMAGE PROPERTIES:")
print(f"   Size: {sample_img.size}")  # (width, height)
print(f"   Mode: {sample_img.mode}")

# Store image dimensions for later use
IMG_WIDTH, IMG_HEIGHT = sample_img.size
print(f"\n   Using dimensions: {IMG_WIDTH} x {IMG_HEIGHT}")


# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# CELL 4: DEFINE CLASS MAPPING
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

"""
YOLO requires numeric class IDs (0, 1, 2, 3...)
We need to map our class names to numbers

Class Mapping:
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�
â”‚  Class Name     â”‚ Class ID â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚  bolt           â”‚    0     â”‚
â”‚  locatingpin    â”‚    1     â”‚
â”‚  nut            â”‚    2     â”‚
â”‚  washer         â”‚    3     â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
"""

# Class name to ID mapping
CLASS_TO_ID = {
    'bolt': 0,
    'locatingpin': 1,
    'nut': 2,
    'washer': 3
}

# ID to class name mapping (reverse)
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}

# Number of classes
NUM_CLASSES = len(CLASS_TO_ID)

print("=" * 60)
print("ğŸ�·ï¸� CLASS MAPPING")
print("=" * 60)
print("\nClass Name â†’ Class ID:")
for name, id in CLASS_TO_ID.items():
    print(f"   {name:15} â†’ {id}")
print(f"\nTotal classes: {NUM_CLASSES}")


# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# CELL 5: CREATE YOLO DIRECTORY STRUCTURE
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

"""
YOLO requires this specific folder structure:

yolo_dataset/
â”œâ”€â”€ images/
â”‚   â”œâ”€â”€ train/          â†� Training images
â”‚   â”‚   â”œâ”€â”€ img1.png
â”‚   â”‚   â”œâ”€â”€ img2.png
â”‚   â”‚   â””â”€â”€ ...
â”‚   â””â”€â”€ val/            â†� Validation images
â”‚       â”œâ”€â”€ img100.png
â”‚       â””â”€â”€ ...
â”œâ”€â”€ labels/
â”‚   â”œâ”€â”€ train/          â†� Training labels (same names as images, .txt)
â”‚   â”‚   â”œâ”€â”€ img1.txt
â”‚   â”‚   â”œâ”€â”€ img2.txt
â”‚   â”‚   â””â”€â”€ ...
â”‚   â””â”€â”€ val/            â†� Validation labels
â”‚       â”œâ”€â”€ img100.txt
â”‚       â””â”€â”€ ...
â””â”€â”€ data.yaml           â†� Configuration file
"""

print("=" * 60)
print("ğŸ“� CREATING YOLO DIRECTORY STRUCTURE")
print("=" * 60)

# Create all directories
directories = [
    YOLO_IMAGES_TRAIN,
    YOLO_IMAGES_VAL,
    YOLO_LABELS_TRAIN,
    YOLO_LABELS_VAL
]

for dir_path in directories:
    os.makedirs(dir_path, exist_ok=True)
    print(f"âœ… Created: {dir_path}")

print("\nğŸ“� Directory structure created:")
print(f"""
{OUTPUT_DIR}/
â”œâ”€â”€ images/
â”‚   â”œâ”€â”€ train/
â”‚   â””â”€â”€ val/
â”œâ”€â”€ labels/
â”‚   â”œâ”€â”€ train/
â”‚   â””â”€â”€ val/
â””â”€â”€ data.yaml (will be created later)
""")


print("=" * 60)
print("ğŸ”� VERIFYING CREATED DIRECTORIES")
print("=" * 60)

# List contents of working directory
print("\nğŸ“‚ Contents of /kaggle/working/:")
print(os.listdir('/kaggle/working/'))

# Check if yolo_dataset exists
if os.path.exists('/kaggle/working/yolo_dataset'):
    print("\nâœ… yolo_dataset folder exists!")
    
    # List subdirectories
    print("\nğŸ“‚ Contents of yolo_dataset/:")
    for item in os.listdir('/kaggle/working/yolo_dataset'):
        item_path = os.path.join('/kaggle/working/yolo_dataset', item)
        if os.path.isdir(item_path):
            print(f"  ğŸ“� {item}/")
            for subitem in os.listdir(item_path):
                print(f"    ğŸ“� {subitem}/")
else:
    print("\nâ�Œ yolo_dataset folder NOT found!")

print("=" * 60)


# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# CELL 6: SPLIT DATA INTO TRAIN AND VALIDATION SETS
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

"""
Why split the data?
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
- Training set: Used to TRAIN the model
- Validation set: Used to EVALUATE during training (not seen during training)

This helps:
1. Detect overfitting
2. Tune hyperparameters (like confidence threshold)
3. Estimate real-world performance

Typical split: 90% train, 10% validation
"""

# Get unique image names
unique_images = train_bboxes['image_name'].unique()
print("=" * 60)
print("ğŸ“Š SPLITTING DATA INTO TRAIN/VALIDATION")
print("=" * 60)
print(f"\nTotal unique images with bboxes: {len(unique_images)}")

# Split: 90% train, 10% validation
VAL_SPLIT = 0.10  # 10% for validation

train_images, val_images = train_test_split(
    unique_images,
    test_size=VAL_SPLIT,
    random_state=42,  # For reproducibility
    shuffle=True
)

# Convert to sets for faster lookup
train_images_set = set(train_images)
val_images_set = set(val_images)

print(f"\nğŸ“Š Split Results:")
print(f"   Training images:   {len(train_images)} ({100*(1-VAL_SPLIT):.0f}%)")
print(f"   Validation images: {len(val_images)} ({100*VAL_SPLIT:.0f}%)")

# Verify split
print(f"\nâœ… Split verification:")
print(f"   Total: {len(train_images) + len(val_images)}")
print(f"   No overlap: {len(train_images_set.intersection(val_images_set)) == 0}")


# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# CELL 7: CONVERT BOUNDING BOXES TO YOLO FORMAT
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

"""
YOLO FORMAT EXPLANATION:
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

Original format (your CSV):
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”�
â”‚ x_min  â”‚ y_min  â”‚ x_max  â”‚ y_max  â”‚ class  â”‚
â”‚(pixels)â”‚(pixels)â”‚(pixels)â”‚(pixels)â”‚ (name) â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”˜

YOLO format (required):
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”�
â”‚ class_id â”‚ x_center â”‚ y_center â”‚ width  â”‚ height â”‚
â”‚ (int)    â”‚ (0 to 1) â”‚ (0 to 1) â”‚(0 to 1)â”‚(0 to 1)â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”˜

Conversion formulas:
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
x_center = (x_min + x_max) / 2 / image_width
y_center = (y_min + y_max) / 2 / image_height
width    = (x_max - x_min) / image_width
height   = (y_max - y_min) / image_height


Visual representation:
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

Original (x_min, y_min, x_max, y_max):

    (x_min, y_min)
         â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�
         â”‚                 â”‚
         â”‚     OBJECT      â”‚
         â”‚                 â”‚
         â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                     (x_max, y_max)

YOLO (x_center, y_center, width, height):

              width
         â†�â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â†’
         â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�  â†‘
         â”‚                 â”‚  â”‚
         â”‚    â—� center     â”‚  â”‚ height
         â”‚  (x_c, y_c)     â”‚  â”‚
         â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â†“
"""

def convert_bbox_to_yolo(x_min, y_min, x_max, y_max, img_width, img_height):
    """
    Convert bounding box from corner format to YOLO center format
    
    Args:
        x_min, y_min: Top-left corner (pixels)
        x_max, y_max: Bottom-right corner (pixels)
        img_width, img_height: Image dimensions
    
    Returns:
        x_center, y_center, width, height (all normalized 0-1)
    """
    # Calculate center coordinates
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    
    # Calculate width and height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    # Ensure values are within [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return x_center, y_center, width, height


# Example conversion
print("=" * 60)
print("ğŸ”„ BBOX FORMAT CONVERSION")
print("=" * 60)

example = train_bboxes.iloc[0]
print(f"\nExample conversion:")
print(f"   Original: x_min={example['x_min']}, y_min={example['y_min']}, "
      f"x_max={example['x_max']}, y_max={example['y_max']}")

x_c, y_c, w, h = convert_bbox_to_yolo(
    example['x_min'], example['y_min'],
    example['x_max'], example['y_max'],
    IMG_WIDTH, IMG_HEIGHT
)
print(f"   YOLO:     x_center={x_c:.4f}, y_center={y_c:.4f}, "
      f"width={w:.4f}, height={h:.4f}")


# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# CELL 8: PROCESS ALL IMAGES AND CREATE YOLO DATASET
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

def create_yolo_dataset(bbox_df, train_images_set, val_images_set, 
                         source_img_dir, output_dir, img_width, img_height):
    """
    Process all images and create YOLO format dataset
    
    For each image:
    1. Copy image to train/ or val/ folder
    2. Create corresponding .txt label file with YOLO format bboxes
    """
    
    # Statistics
    stats = {
        'train_images': 0,
        'val_images': 0,
        'train_bboxes': 0,
        'val_bboxes': 0
    }
    
    # Group bboxes by image
    grouped = bbox_df.groupby('image_name')
    
    print(f"Processing {len(grouped)} images...")
    
    for img_name, group in tqdm(grouped, desc="Creating YOLO dataset"):
        
        # Determine if train or val
        if img_name in train_images_set:
            split = 'train'
            stats['train_images'] += 1
            stats['train_bboxes'] += len(group)
        elif img_name in val_images_set:
            split = 'val'
            stats['val_images'] += 1
            stats['val_bboxes'] += len(group)
        else:
            continue  # Skip if not in either set
        
        # Paths
        src_img_path = os.path.join(source_img_dir, img_name)
        dst_img_path = os.path.join(output_dir, 'images', split, img_name)
        
        # Create label filename (.txt instead of .png)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(output_dir, 'labels', split, label_name)
        
        # Copy image
        shutil.copy(src_img_path, dst_img_path)
        
        # Create label file
        with open(label_path, 'w') as f:
            for _, row in group.iterrows():
                # Get class ID
                class_id = CLASS_TO_ID[row['class']]
                
                # Convert bbox to YOLO format
                x_center, y_center, width, height = convert_bbox_to_yolo(
                    row['x_min'], row['y_min'],
                    row['x_max'], row['y_max'],
                    img_width, img_height
                )
                
                # Write line: class_id x_center y_center width height
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    return stats


print("=" * 60)
print("ğŸ”„ CREATING YOLO DATASET")
print("=" * 60)

# Create the dataset
stats = create_yolo_dataset(
    train_bboxes,
    train_images_set,
    val_images_set,
    TRAIN_IMAGES_DIR,
    OUTPUT_DIR,
    IMG_WIDTH,
    IMG_HEIGHT
)

# Print statistics
print("\nâœ… Dataset creation complete!")
print("\nğŸ“Š Statistics:")
print(f"   Training images:     {stats['train_images']}")
print(f"   Training bboxes:     {stats['train_bboxes']}")
print(f"   Validation images:   {stats['val_images']}")
print(f"   Validation bboxes:   {stats['val_bboxes']}")

# Verify files were created
print(f"\nğŸ“� Files created:")
print(f"   Train images: {len(os.listdir(YOLO_IMAGES_TRAIN))}")
print(f"   Train labels: {len(os.listdir(YOLO_LABELS_TRAIN))}")
print(f"   Val images:   {len(os.listdir(YOLO_IMAGES_VAL))}")
print(f"   Val labels:   {len(os.listdir(YOLO_LABELS_VAL))}")


# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# CELL 9: CREATE DATA.YAML CONFIGURATION FILE
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

"""
data.yaml tells YOLO:
- Where to find images
- Where to find labels
- What classes exist
- How many classes there are
"""

# Create data.yaml content
data_yaml_content = f"""# SOLIDWORKS AI Hackathon Dataset
# Auto-generated for YOLOv8 training

# Paths
path: {OUTPUT_DIR}
train: images/train
val: images/val

# Classes
nc: {NUM_CLASSES}
names:
  0: bolt
  1: locatingpin
  2: nut
  3: washer
"""

# Write to file
data_yaml_path = os.path.join(OUTPUT_DIR, 'data.yaml')

with open(data_yaml_path, 'w') as f:
    f.write(data_yaml_content)

print("=" * 60)
print("ğŸ“„ DATA.YAML CREATED")
print("=" * 60)
print(f"\nPath: {data_yaml_path}")
print("\nContent:")
print("-" * 40)
print(data_yaml_content)
print("-" * 40)


# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# CELL 10: VERIFY YOLO FORMAT CONVERSION IS CORRECT
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

def verify_yolo_conversion(output_dir, source_img_dir, num_samples=4):
    """
    Visualize images with their YOLO format bboxes to verify conversion
    """
    
    # Get sample images from train
    train_images = os.listdir(os.path.join(output_dir, 'images', 'train'))[:num_samples]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    
    colors = ['red', 'blue', 'green', 'orange']
    class_names = ['bolt', 'locatingpin', 'nut', 'washer']
    
    for idx, img_name in enumerate(train_images):
        ax = axes[idx]
        
        # Load image
        img_path = os.path.join(output_dir, 'images', 'train', img_name)
        img = Image.open(img_path)
        img_width, img_height = img.size
        ax.imshow(img)
        
        # Load YOLO labels
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(output_dir, 'labels', 'train', label_name)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * img_width
                        y_center = float(parts[2]) * img_height
                        width = float(parts[3]) * img_width
                        height = float(parts[4]) * img_height
                        
                        # Convert back to corner format for drawing
                        x_min = x_center - width / 2
                        y_min = y_center - height / 2
                        
                        # Draw rectangle
                        rect = patches.Rectangle(
                            (x_min, y_min), width, height,
                            linewidth=3,
                            edgecolor=colors[class_id],
                            facecolor='none'
                        )
                        ax.add_patch(rect)
                        
                        # Add label
                        ax.text(
                            x_min, y_min - 5,
                            class_names[class_id],
                            color='white',
                            fontsize=10,
                            fontweight='bold',
                            bbox=dict(boxstyle='round', facecolor=colors[class_id], alpha=0.8)
                        )
        
        ax.set_title(f'{img_name[:30]}...', fontsize=10)
        ax.axis('off')
    
    # Add legend
    legend_patches = [patches.Patch(color=c, label=n) for c, n in zip(colors, class_names)]
    fig.legend(handles=legend_patches, loc='upper center', ncol=4, fontsize=12)
    
    plt.suptitle('YOLO Format Verification\n(Bounding boxes drawn from .txt labels)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/kaggle/working/yolo_verification.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("âœ… If bounding boxes match the objects, conversion is correct!")


print("=" * 60)
print("ğŸ”� VERIFYING YOLO FORMAT CONVERSION")
print("=" * 60)

verify_yolo_conversion(OUTPUT_DIR, TRAIN_IMAGES_DIR)

















# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# CELL 0: FIX ULTRALYTICS VERSION (RUN THIS FIRST!)
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

# Step 1: Uninstall old version completely
!pip uninstall ultralytics -y

# Step 2: Clear pip cache
!pip cache purge

# Step 3: Install latest version
!pip install ultralytics --upgrade --force-reinstall

# Step 4: Verify installation
import ultralytics
print(f"\nâœ… Ultralytics version: {ultralytics.__version__}")

# Step 5: IMPORTANT MESSAGE
print("\n" + "=" * 60)
print("âš ï¸�  IMPORTANT: RESTART THE KERNEL NOW!")
print("=" * 60)
print("""
To restart the kernel in Kaggle:
1. Click 'Runtime' in the menu bar
2. Click 'Restart session' (or 'Restart runtime')
3. Then run all cells from the beginning

OR simply click: Runtime â†’ Restart and run all
""")


# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# CELL 1: SETUP AND IMPORTS (RUN AFTER KERNEL RESTART)
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import shutil
from tqdm import tqdm
import yaml
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Disable WandB BEFORE importing ultralytics
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'

# Now import ultralytics
from ultralytics import YOLO

# Verify version
import ultralytics
print("=" * 60)
print("ğŸ”§ ENVIRONMENT CHECK")
print("=" * 60)
print(f"Ultralytics version: {ultralytics.__version__}")

# Check if version is correct
version = ultralytics.__version__
major, minor, patch = version.split('.')[:3]
if int(minor) >= 3 and int(patch.split('.')[0]) >= 200:
    print("âœ… Version is compatible!")
else:
    print(f"âš ï¸� Version {version} may have issues. Expected 8.3.200+")
    print("   If you see errors, restart kernel and reinstall.")

# Check GPU
import torch
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("=" * 60)


# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# CELL 11: TRAIN YOLOv8 MODEL (FULLY FIXED)
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

import os

# Ensure WandB is disabled
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'

# Also disable other potential loggers
os.environ['COMET_MODE'] = 'disabled'
os.environ['CLEARML_LOG_MODEL'] = 'false'

print("=" * 60)
print("ğŸš€ TRAINING YOLOv8 MODEL")
print("=" * 60)

# Verify ultralytics version before training
import ultralytics
print(f"Ultralytics version: {ultralytics.__version__}")

# Initialize model with pretrained weights
from ultralytics import YOLO
model = YOLO('yolov8s.pt')  # Using 'small' variant

# Training configuration
EPOCHS = 50              
IMG_SIZE = 640          
BATCH_SIZE = 8          
PATIENCE = 10            

print(f"\nâš™ï¸� Training Configuration:")
print(f"   Model: YOLOv8s")
print(f"   Epochs: {EPOCHS}")
print(f"   Image Size: {IMG_SIZE}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Early Stopping Patience: {PATIENCE}")

# Start training with minimal, safe parameters
results = model.train(
    data=data_yaml_path,      # Path to data.yaml
    epochs=EPOCHS,             # Number of epochs
    imgsz=IMG_SIZE,           # Image size
    batch=BATCH_SIZE,         # Batch size
    patience=PATIENCE,        # Early stopping
    device=0,                 # GPU device
    workers=2,                # Reduced workers for stability
    project='yolov8_run',     # Simple project name
    name='train',             # Simple run name
    exist_ok=True,            # Overwrite if exists
    pretrained=True,          # Use pretrained weights
    verbose=True,             # Print progress
    
    # Optimizer settings
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    
    # Augmentation settings
    augment=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
)

print("\n" + "=" * 60)
print("âœ… TRAINING COMPLETE!")
print("=" * 60)

# Find the best model path
BEST_MODEL_PATH = 'yolov8_run/train/weights/best.pt'
print(f"\nğŸ“� Best model saved at: {BEST_MODEL_PATH}")





# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# CELL 12: EVALUATE MODEL PERFORMANCE
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

# Load the best model
BEST_MODEL_PATH = '/kaggle/working/yolov8_solidworks/weights/best.pt'
best_model = YOLO(BEST_MODEL_PATH)

print("=" * 60)
print("ğŸ“Š MODEL EVALUATION")
print("=" * 60)

# Validate on validation set
val_results = best_model.val(data=data_yaml_path)

print(f"\nğŸ“Š Validation Metrics:")
print(f"   mAP50:     {val_results.box.map50:.4f}")
print(f"   mAP50-95:  {val_results.box.map:.4f}")
print(f"   Precision: {val_results.box.mp:.4f}")
print(f"   Recall:    {val_results.box.mr:.4f}")

# Per-class metrics
print("\nğŸ“Š Per-Class AP50:")
class_names = ['bolt', 'locatingpin', 'nut', 'washer']
for i, name in enumerate(class_names):
    if i < len(val_results.box.ap50):
        print(f"   {name:15}: {val_results.box.ap50[i]:.4f}")




