import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report


# Paths to train CSV and images
train_csv_path = '/kaggle/input/ai-vs-human-generated-dataset/train.csv'  
images_dir = '/kaggle/input/ai-vs-human-generated-dataset'

# Output directory for dataset structure (working directory)
output_dir = '/kaggle/working/dataset'
train_out_dir = os.path.join(output_dir, "train")
val_out_dir   = os.path.join(output_dir, "val")

label_mapping = {0: "Non-AI", 1: "AI"}


# Remove existing dataset directory if it exists
#shutil.rmtree("/kaggle/working/dataset") optional


# Create base directories
os.makedirs(train_out_dir, exist_ok=True)
os.makedirs(val_out_dir, exist_ok=True)


# Load and split the CSV into training and validation sets
df = pd.read_csv(train_csv_path)
print("Total number of images:", len(df))

# Split into 85% training and 15% validation
df_train, df_val = train_test_split(
    df, test_size=0.05, stratify=df['label'], random_state=42
)

print(f"Images after split -> Train: {len(df_train)}, Val: {len(df_val)}")


# Create subdirectories for each class in training and validation
for split_dir in [train_out_dir, val_out_dir]:
    for class_name in label_mapping.values():
        os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)


# Function to copy an image to the structured dataset directory based on its class
def copy_image(src_directory, dst_directory, relative_path, label):
    src_path = os.path.normpath(os.path.join(src_directory, relative_path))
    filename = os.path.basename(relative_path)
    dst_path = os.path.join(dst_directory, label, filename)
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
    else:
        print("File not found:", src_path)


# Copy images for training
print("Copying images for training...")
for _, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Train"):
    file_path = row['file_name']
    numeric_label = row['label']
    class_name = label_mapping.get(numeric_label, str(numeric_label))
    copy_image(images_dir, train_out_dir, file_path, class_name)

# Copy images for validation
print("Copying images for validation...")
for _, row in tqdm(df_val.iterrows(), total=len(df_val), desc="Val"):
    file_path = row['file_name']
    numeric_label = row['label']
    class_name = label_mapping.get(numeric_label, str(numeric_label))
    copy_image(images_dir, val_out_dir, file_path, class_name)

print("\nDataset structure created successfully in:", output_dir)
# Expected structure:
# /kaggle/working/dataset/
# ├── train/
# │   ├── AI/
# │   └── Non-AI/
# └── val/
#     ├── AI/
#     └── Non-AI/


# Import dependencies
import random
import numpy as np
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt


# Set fixed seeds for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("Seeds set for Python, NumPy, and PyTorch.")


# Load YOLO model for classification
#model = YOLO("yolo11m-cls.pt", task="classify") #To start with a new model
model = YOLO("/kaggle/working/runs/train/yolo-finetune5/weights/best.pt", task="classify")
print("YOLO classification model loaded.")


# Train the model
model.train(
    data="/kaggle/working/dataset",  
    epochs=6,                        
    #imgsz=224,
    imgsz=640,
    batch=64,
    seed=SEED,
    pretrained=True,
    project="runs/train",
    name="yolo-finetune",
    device=0,                  
    auto_augment="autoaugment", #Data Augmentation
    cos_lr=True,
    freeze=3,
    resume=True,
)


# Evaluate on validation dataset
results_val = model.val()
print("Validation results:", results_val)


# Make predictions on test dataset
results_test = model.predict(source="/kaggle/input/ai-vs-human-generated-dataset/test_data_v2", imgsz=640,device=0,stream=True)


# Load test CSV file
test_df = pd.read_csv('/kaggle/input/ai-vs-human-generated-dataset/test.csv')
print(test_df.head())

predicted_labels = [int(res.probs.top1) for res in results_test]
test_df['label'] = predicted_labels
submission_df = test_df[['id', 'label']]

# Save the submission file
submission_df.to_csv('submission.csv', index=False)
print("Submission file submission.csv generated successfully!")

