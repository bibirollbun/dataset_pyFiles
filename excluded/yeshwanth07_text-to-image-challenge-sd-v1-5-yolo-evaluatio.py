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


# ============================================================================
# COMPETITION SUBMISSION: Text-to-Image Generation Challenge
# PARTICIPANT: Yeshwanth Vemula
# APPROACH: Stable Diffusion v1.5 with YOLO object detection evaluation
# ============================================================================

# Install YOLOv8 for object detection evaluation
!pip install -q ultralytics

print("âœ… YOLO installed successfully!")


# Import libraries for YOLO object detection and text processing
from ultralytics import YOLO
import re

print("âœ… All libraries imported!")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")


# Download and load YOLOv8 nano model for object detection
# This model detects objects in generated images for F1 score calculation

def download_yolo_model():
    """Download YOLOv8 model from Ultralytics"""
    model = YOLO('yolov8n.pt')
    return model

yolo_model = download_yolo_model()

print(f"âœ… YOLO model loaded: yolov8n.pt")


# Load generated images and results from Kaggle dataset
# Dataset path is automatically set by Kaggle when dataset is added

dataset_path = '/kaggle/input/text-to-image-competition-submission'

# Verify dataset exists
if not os.path.exists(dataset_path):
    print("â�Œ Dataset not found!")
    print("Available paths:")
    for dirname, _, filenames in os.walk('/kaggle/input'):
        print(f"  {dirname}")
else:
    print(f"âœ… Dataset found: {dataset_path}")
    
# Load results.csv which maps prompts to generated image filenames
results_csv_path = f'{dataset_path}/results.csv'
results_df = pd.read_csv(results_csv_path)
results_df = results_df[['run_id', 'prompt', 'filenames']].rename(columns={'filenames': 'generated_images'})

print(f"\nâœ… Loaded {len(results_df)} prompts from results.csv")
print(f"\nFirst 5 entries:")
print(results_df.head())


# Extract expected objects from prompts using keyword matching
# Keeping it simple to avoid over-expecting objects

common_objects = {
    'man', 'woman', 'person', 'people', 'child', 'boy', 'girl',
    'dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'car', 'truck', 'bus', 'train', 'airplane', 'boat', 'bicycle', 'motorcycle',
    'food', 'pizza', 'cake', 'donut', 'doughnut', 'sandwich', 'apple', 'banana', 'orange',
    'chair', 'table', 'bed', 'couch', 'desk', 'toilet', 'sink', 'mirror',
    'tree', 'flower', 'grass', 'sky', 'mountain', 'ocean', 'beach', 'field',
    'phone', 'laptop', 'computer', 'keyboard', 'mouse', 'book', 'pen',
    'clock', 'umbrella', 'bag', 'suitcase', 'backpack', 'hat', 'shirt',
    'building', 'house', 'shop', 'store', 'restaurant', 'window', 'door',
    'bench', 'vase', 'bottle', 'cup', 'glass', 'plate', 'knife', 'fork'
}

def extract_expected_objects(prompt):
    """Extract expected objects from prompt"""
    words = re.findall(r'\b\w+\b', prompt.lower())
    expected = set(word for word in words if word in common_objects)
    return expected

results_df['expected_objects'] = results_df['prompt'].apply(extract_expected_objects)

print("âœ… Expected objects extracted")
print(f"\nExample - Prompt 1:")
print(f"Prompt: {results_df.iloc[0]['prompt']}")
print(f"Expected: {results_df.iloc[0]['expected_objects']}")


# Run YOLO object detection on all generated images
# Detects actual objects present in images for comparison with expected objects
# 
# Process:
# 1. Load each generated image from dataset
# 2. Run YOLOv8 inference to detect objects
# 3. Extract detected object class names
# 4. Store as set for F1 calculation

def detect_objects_in_image(image_path):
    """
    Run YOLO detection on image and return set of detected object names
    """
    results = yolo_model(image_path, verbose=False)
    
    detected_objects = set()
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            detected_objects.add(class_name)
    
    return detected_objects

# Run detection on all generated images
print("ğŸ”� Running YOLO object detection on all generated images...")
print("Estimated time: 2-3 minutes\n")

detected_objects_list = []
for idx, row in results_df.iterrows():
    image_path = f"{dataset_path}/{row['generated_images']}"
    
    if os.path.exists(image_path):
        detected = detect_objects_in_image(image_path)
        detected_objects_list.append(detected)
        
        # Progress indicator every 10 images
        if (idx + 1) % 10 == 0:
            print(f"âœ“ Processed {idx + 1}/{len(results_df)} images")
    else:
        print(f"âš ï¸� Image not found: {image_path}")
        detected_objects_list.append(set())

results_df['detected_objects'] = detected_objects_list

print(f"\nâœ… Object detection complete on {len(results_df)} images!")
print(f"\nExample - Image 1:")
print(f"Detected objects: {results_df.iloc[0]['detected_objects']}")


# Calculate F1 score for each image
# F1 = 2 * (Precision * Recall) / (Precision + Recall)
# 
# Where:
# - Precision = True Positives / Total Detected
# - Recall = True Positives / Total Expected
# - True Positives = Objects that were both expected AND detected
# 
# This measures how well the generated image matches the prompt

def calculate_f1_score(expected, detected):
    """
    Calculate F1 score comparing expected vs detected object sets
    """
    # Edge cases
    if len(expected) == 0 and len(detected) == 0:
        return 1.0  # Perfect match if both empty
    if len(expected) == 0 or len(detected) == 0:
        return 0.0  # No match if one is empty
    
    # Calculate overlap (true positives)
    true_positives = len(expected.intersection(detected))
    
    # Calculate precision and recall
    precision = true_positives / len(detected) if len(detected) > 0 else 0
    recall = true_positives / len(expected) if len(expected) > 0 else 0
    
    # Calculate F1 score (harmonic mean of precision and recall)
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# Calculate F1 score for all images
results_df['f1_score'] = results_df.apply(
    lambda row: calculate_f1_score(row['expected_objects'], row['detected_objects']),
    axis=1
)

print("âœ… F1 scores calculated for all images!")
print(f"\nğŸ“Š F1 Score Statistics:")
print(f"{'='*50}")
print(f"Mean F1:   {results_df['f1_score'].mean():.4f}")
print(f"Median F1: {results_df['f1_score'].median():.4f}")
print(f"Min F1:    {results_df['f1_score'].min():.4f}")
print(f"Max F1:    {results_df['f1_score'].max():.4f}")
print(f"{'='*50}")


# Create submission.csv with zero-based IDs (0, 1, 2, ...)
# Some competitions expect IDs starting from 0 instead of 1

# Create submission with zero-based IDs
submission_df = results_df[['f1_score']].copy()
submission_df['ID'] = range(len(submission_df))  # 0, 1, 2, 3, ..., 48

# Reorder columns
submission_df = submission_df[['ID', 'f1_score']]

# Save to CSV
submission_df.to_csv('submission.csv', index=False)

print(f"âœ… submission.csv created with zero-based IDs!")
print(f"\nSubmission preview (first 10 rows):")
print(submission_df.head(10))
print(f"\nLast 3 rows:")
print(submission_df.tail(3))

print(f"\n{'='*60}")
print(f"ğŸ“Š FINAL COMPETITION RESULTS")
print(f"{'='*60}")
print(f"Total images evaluated: {len(submission_df)}")
print(f"Average F1 Score: {results_df['f1_score'].mean():.4f}")
print(f"{'='*60}")
print(f"\nColumns: {list(submission_df.columns)}")
print(f"ID range: {submission_df['ID'].min()} to {submission_df['ID'].max()}")


# Display detailed results for first 10 images for manual review
# Shows prompt, expected objects, detected objects, and F1 score

print("ğŸ“‹ Detailed Results (First 10 images):\n")
for idx, row in results_df.head(10).iterrows():
    print(f"Image {row['run_id']:04d}:")
    print(f"  Prompt: {row['prompt']}")
    print(f"  Expected: {row['expected_objects']}")
    print(f"  Detected: {row['detected_objects']}")
    print(f"  F1 Score: {row['f1_score']:.4f}")
    print()

