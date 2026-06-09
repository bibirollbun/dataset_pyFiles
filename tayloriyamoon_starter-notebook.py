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


# Complete YOLOv8 Object Detection Pipeline for DU AI Challenge Demo
# This notebook provides an end-to-end solution from data loading to submission

import os
import shutil
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# PART 1: ENVIRONMENT SETUP
# =====================================================

print("Setting up environment...")

# Install required packages (run this in Kaggle)
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

# Install ultralytics for YOLOv8
try:
    import ultralytics
    print(f"Ultralytics version: {ultralytics.__version__}")
except:
    print("Installing ultralytics...")
    install('ultralytics')
    import ultralytics
    print(f"Ultralytics version installed: {ultralytics.__version__}")

from ultralytics import YOLO

# =====================================================
# PART 2: DATA PATHS AND CONFIGURATION
# =====================================================

# Define paths (adjust if running locally vs Kaggle)
BASE_PATH = '/kaggle/input/du-ai-challenge-demo'
WORKING_PATH = '/kaggle/working'

# Check if running on Kaggle or locally
if not os.path.exists(BASE_PATH):
    print("Not on Kaggle, adjusting paths...")
    BASE_PATH = './du-ai-challenge-demo'  # Adjust this to your local path
    WORKING_PATH = './working'
    os.makedirs(WORKING_PATH, exist_ok=True)

# Define specific paths
TRAIN_IMAGES = os.path.join(BASE_PATH, 'train/images')
TRAIN_LABELS = os.path.join(BASE_PATH, 'train/labels')
TEST_IMAGES = os.path.join(BASE_PATH, 'test/images')
SAMPLE_SUBMISSION = os.path.join(BASE_PATH, 'sample_submission.csv')

print(f"Base path: {BASE_PATH}")
print(f"Working path: {WORKING_PATH}")

# =====================================================
# PART 3: DATA EXPLORATION
# =====================================================

print("\n" + "="*50)
print("DATA EXPLORATION")
print("="*50)

# Count files
train_images = os.listdir(TRAIN_IMAGES) if os.path.exists(TRAIN_IMAGES) else []
train_labels = os.listdir(TRAIN_LABELS) if os.path.exists(TRAIN_LABELS) else []
test_images = os.listdir(TEST_IMAGES) if os.path.exists(TEST_IMAGES) else []

print(f"Training images: {len(train_images)}")
print(f"Training labels: {len(train_labels)}")
print(f"Test images: {len(test_images)}")

# Load class names
class_names = ['bicycle', 'bus', 'car', 'motorbike', 'person']
num_classes = len(class_names)
print(f"\nClasses ({num_classes}): {class_names}")

# Create class mapping
class_id_to_name = {i: name for i, name in enumerate(class_names)}
print(f"Class mapping: {class_id_to_name}")

# =====================================================
# PART 4: PREPARE DATASET FOR YOLOV8
# =====================================================

print("\n" + "="*50)
print("PREPARING DATASET FOR YOLOV8")
print("="*50)

# Create directory structure for YOLOv8
dataset_path = os.path.join(WORKING_PATH, 'yolo_dataset')
os.makedirs(dataset_path, exist_ok=True)

# Create train and validation splits
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')
test_path = os.path.join(dataset_path, 'test')

for path in [train_path, val_path, test_path]:
    os.makedirs(os.path.join(path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(path, 'labels'), exist_ok=True)

# Copy training data (80% train, 20% validation)
train_image_list = sorted([f for f in os.listdir(TRAIN_IMAGES) if f.endswith('.jpg')])
np.random.seed(42)
np.random.shuffle(train_image_list)

split_idx = int(0.8 * len(train_image_list))
train_split = train_image_list[:split_idx]
val_split = train_image_list[split_idx:]

print(f"Training split: {len(train_split)} images")
print(f"Validation split: {len(val_split)} images")

# Copy files to appropriate directories
def copy_files(image_list, source_img_dir, source_lbl_dir, dest_img_dir, dest_lbl_dir):
    for img_name in image_list:
        # Copy image
        src_img = os.path.join(source_img_dir, img_name)
        dst_img = os.path.join(dest_img_dir, img_name)
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
        
        # Copy corresponding label
        label_name = img_name.replace('.jpg', '.txt')
        src_lbl = os.path.join(source_lbl_dir, label_name)
        dst_lbl = os.path.join(dest_lbl_dir, label_name)
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)

print("Copying training files...")
copy_files(train_split, TRAIN_IMAGES, TRAIN_LABELS, 
           os.path.join(train_path, 'images'), os.path.join(train_path, 'labels'))

print("Copying validation files...")
copy_files(val_split, TRAIN_IMAGES, TRAIN_LABELS, 
           os.path.join(val_path, 'images'), os.path.join(val_path, 'labels'))

# Copy test images
print("Copying test files...")
test_image_list = [f for f in os.listdir(TEST_IMAGES) if f.endswith('.jpg')]
for img_name in test_image_list:
    src = os.path.join(TEST_IMAGES, img_name)
    dst = os.path.join(test_path, 'images', img_name)
    if os.path.exists(src):
        shutil.copy2(src, dst)

# Create data.yaml configuration file
data_yaml = {
    'path': dataset_path,
    'train': 'train/images',
    'val': 'val/images',
    'test': 'test/images',
    'nc': num_classes,
    'names': class_names
}

yaml_path = os.path.join(dataset_path, 'data.yaml')
with open(yaml_path, 'w') as f:
    yaml.dump(data_yaml, f)

print(f"\nDataset configuration saved to: {yaml_path}")

# =====================================================
# PART 5: TRAIN YOLOV8 MODEL
# =====================================================

print("\n" + "="*50)
print("TRAINING YOLOV8 MODEL")
print("="*50)

# Initialize YOLOv8 model
# Using YOLOv8n (nano) for faster training. You can use yolov8s, yolov8m, yolov8l for better accuracy
model = YOLO('yolov8n.pt')  # Use pretrained YOLOv8 nano model

# Training parameters
EPOCHS = 30  # Increase for better results
BATCH_SIZE = 16
IMG_SIZE = 640
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

print(f"Training parameters:")
print(f"  - Model: YOLOv8n")
print(f"  - Epochs: {EPOCHS}")
print(f"  - Batch size: {BATCH_SIZE}")
print(f"  - Image size: {IMG_SIZE}")

# Train the model
results = model.train(
    data=yaml_path,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    patience=10,
    save=True,
    device='cpu',  # Use 'cuda' or 0 if GPU is available
    workers=2,
    project=WORKING_PATH,
    name='yolov8_du_challenge',
    exist_ok=True,
    pretrained=True,
    optimizer='SGD',
    verbose=True,
    seed=42,
    deterministic=True,
    single_cls=False,
    rect=False,
    cos_lr=False,
    close_mosaic=10,
    amp=False,
    fraction=1.0,
    profile=False,
    overlap_mask=True,
    mask_ratio=4,
    dropout=0.0,
    val=True,
    plots=True
)

print("\nTraining completed!")

# Get the best model path
best_model_path = os.path.join(WORKING_PATH, 'yolov8_du_challenge/weights/best.pt')
if not os.path.exists(best_model_path):
    best_model_path = os.path.join(WORKING_PATH, 'yolov8_du_challenge/weights/last.pt')

print(f"Best model saved at: {best_model_path}")

# =====================================================
# PART 6: LOAD BEST MODEL AND MAKE PREDICTIONS
# =====================================================

print("\n" + "="*50)
print("MAKING PREDICTIONS ON TEST SET")
print("="*50)

# Load the best model
best_model = YOLO(best_model_path)

# Function to convert YOLO predictions to submission format
def yolo_to_submission_format(predictions, image_name, conf_threshold=0.25):
    """
    Convert YOLO predictions to competition submission format
    Format: class_id confidence x_min y_min x_max y_max
    """
    pred_string = []
    
    if len(predictions) > 0 and predictions[0].boxes is not None:
        boxes = predictions[0].boxes
        
        for i in range(len(boxes)):
            # Get box coordinates (xyxy format)
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            
            # Get confidence score
            conf = boxes.conf[i].cpu().numpy()
            
            # Get class id
            cls_id = int(boxes.cls[i].cpu().numpy())
            
            # Filter by confidence threshold
            if conf >= conf_threshold:
                # Format: class_id confidence x_min y_min x_max y_max
                pred_string.append(f"{cls_id} {conf:.4f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}")
    
    return ' '.join(pred_string)

# Make predictions on test set
test_predictions = []
test_image_dir = os.path.join(test_path, 'images')
test_images_list = sorted([f for f in os.listdir(test_image_dir) if f.endswith('.jpg')])

print(f"Predicting on {len(test_images_list)} test images...")

for idx, image_name in enumerate(test_images_list):
    if idx % 20 == 0:
        print(f"Processing image {idx + 1}/{len(test_images_list)}")
    
    image_path = os.path.join(test_image_dir, image_name)
    
    # Run inference
    predictions = best_model.predict(
        image_path,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMG_SIZE,
        verbose=False
    )
    
    # Convert predictions to submission format
    image_id = image_name.replace('.jpg', '')
    pred_string = yolo_to_submission_format(predictions, image_name, CONFIDENCE_THRESHOLD)
    
    test_predictions.append({
        'image_id': image_id,
        'PredictionString': pred_string
    })

print("Predictions completed!")

# =====================================================
# PART 7: CREATE SUBMISSION FILE
# =====================================================

print("\n" + "="*50)
print("CREATING SUBMISSION FILE")
print("="*50)

# Create submission dataframe
submission_df = pd.DataFrame(test_predictions)

# Load sample submission to ensure correct format
sample_sub = pd.read_csv(SAMPLE_SUBMISSION)
print(f"Sample submission shape: {sample_sub.shape}")
print(f"Our submission shape: {submission_df.shape}")

# Ensure all test images are included
if len(submission_df) != len(sample_sub):
    print(f"Warning: Number of predictions ({len(submission_df)}) doesn't match sample ({len(sample_sub)})")
    
    # Get missing image IDs
    sample_ids = set(sample_sub['image_id'].values)
    pred_ids = set(submission_df['image_id'].values)
    missing_ids = sample_ids - pred_ids
    
    if missing_ids:
        print(f"Adding {len(missing_ids)} missing images with empty predictions")
        missing_df = pd.DataFrame({
            'image_id': list(missing_ids),
            'PredictionString': [''] * len(missing_ids)
        })
        submission_df = pd.concat([submission_df, missing_df], ignore_index=True)

# Sort by image_id to match sample submission order
submission_df = submission_df.sort_values('image_id').reset_index(drop=True)

# Save submission file
submission_path = os.path.join(WORKING_PATH, 'submission.csv')
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to: {submission_path}")

# Display first few rows
print("\nFirst 5 rows of submission:")
print(submission_df.head())

# =====================================================
# PART 8: VALIDATION AND STATISTICS
# =====================================================

print("\n" + "="*50)
print("SUBMISSION STATISTICS")
print("="*50)

# Calculate statistics
total_images = len(submission_df)
images_with_detections = sum(submission_df['PredictionString'] != '')
empty_predictions = sum(submission_df['PredictionString'] == '')

print(f"Total images: {total_images}")
print(f"Images with detections: {images_with_detections} ({100*images_with_detections/total_images:.1f}%)")
print(f"Images without detections: {empty_predictions} ({100*empty_predictions/total_images:.1f}%)")

# Count detections per class
class_counts = {name: 0 for name in class_names}
total_detections = 0

for pred_string in submission_df['PredictionString']:
    if pred_string and pred_string != '':
        predictions = pred_string.split()
        # Each detection has 6 values: class_id, confidence, x1, y1, x2, y2
        num_detections = len(predictions) // 6
        total_detections += num_detections
        
        for i in range(0, len(predictions), 6):
            if i < len(predictions):
                try:
                    class_id = int(predictions[i])
                    if class_id in class_id_to_name:
                        class_counts[class_id_to_name[class_id]] += 1
                except:
                    pass

print(f"\nTotal detections: {total_detections}")
print(f"Average detections per image: {total_detections/total_images:.2f}")

print("\nDetections per class:")
for class_name, count in class_counts.items():
    print(f"  {class_name}: {count}")

print("\n" + "="*50)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("="*50)
print(f"\nSubmission file ready at: {submission_path}")
print("You can now submit this file to the competition!")

# =====================================================
# OPTIONAL: VISUALIZE SOME PREDICTIONS
# =====================================================

def visualize_predictions(model, image_path, conf_threshold=0.25):
    """Visualize predictions on a single image"""
    # Run prediction
    results = model.predict(image_path, conf=conf_threshold, imgsz=IMG_SIZE)
    
    # Plot results
    if len(results) > 0:
        # Get the annotated image
        annotated_img = results[0].plot()
        
        # Display
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"Predictions for: {os.path.basename(image_path)}")
        plt.show()
        
        # Print detection details
        if results[0].boxes is not None:
            boxes = results[0].boxes
            print(f"Found {len(boxes)} objects:")
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].cpu().numpy())
                conf = boxes.conf[i].cpu().numpy()
                print(f"  - {class_id_to_name[cls_id]}: {conf:.3f}")

# Visualize predictions on a few test images
print("\n" + "="*50)
print("VISUALIZING SAMPLE PREDICTIONS")
print("="*50)

sample_test_images = test_images_list[:3]  # Visualize first 3 images
for img_name in sample_test_images:
    img_path = os.path.join(test_image_dir, img_name)
    print(f"\nImage: {img_name}")
    visualize_predictions(best_model, img_path, CONFIDENCE_THRESHOLD)

