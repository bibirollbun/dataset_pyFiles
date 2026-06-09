import os
import json
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
import random

# --- Paths & Parameters ---
INPUT_DIR = "/kaggle/input/deepfake-detection-challenge/train_sample_videos/"
METADATA_PATH = os.path.join(INPUT_DIR, 'metadata.json')
OUTPUT_DIR = "/kaggle/working/processed_dfdc_images/"
FRAMES_PER_VIDEO = 20
VALIDATION_SPLIT = 0.2
MAX_VIDEOS_TO_PROCESS = 400

print("Configuration set.")


# Load the metadata using the path defined in Cell 1
metadata_df = pd.read_json(METADATA_PATH).T
print(f"Loaded metadata for {len(metadata_df)} videos.")

# Create the output directories using the path defined in Cell 1
for split in ['train', 'validation']:
    os.makedirs(os.path.join(OUTPUT_DIR, split, 'real'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, 'fake'), exist_ok=True)
print(f"Output folders created at {OUTPUT_DIR}")


def extract_frames(video_path, output_folder, max_frames):
    """Extracts a set number of evenly spaced frames from a single video."""
    if not os.path.exists(video_path):
        return
        
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < max_frames:
        return
        
    video_name = os.path.basename(video_path).split('.')[0]
    frame_indices = np.linspace(0, total_frames - 1, num=max_frames, dtype=int)
    
    for i, frame_index in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_filename = os.path.join(output_folder, f"{video_name}_frame_{i}.jpg")
        cv2.imwrite(frame_filename, frame)
        
    cap.release()

print("Frame extraction function is ready.")


# Get a list of videos to process and shuffle them
video_files = list(metadata_df.index)[:MAX_VIDEOS_TO_PROCESS]
random.shuffle(video_files)

# Split the list into training and validation sets
split_index = int(len(video_files) * VALIDATION_SPLIT)
validation_videos = video_files[:split_index]
train_videos = video_files[split_index:]

print(f"Splitting data: {len(train_videos)} for training, {len(validation_videos)} for validation.")

# --- Process Training Videos ---
for video_file in tqdm(train_videos, desc="Processing TRAIN set"):
    video_path = os.path.join(INPUT_DIR, video_file)
    label = metadata_df.loc[video_file, 'label']
    
    if label == 'REAL':
        output_folder = os.path.join(OUTPUT_DIR, 'train/real')
    else: # FAKE
        output_folder = os.path.join(OUTPUT_DIR, 'train/fake')
    
    extract_frames(video_path, output_folder, FRAMES_PER_VIDEO)

# --- Process Validation Videos ---
for video_file in tqdm(validation_videos, desc="Processing VALIDATION set"):
    video_path = os.path.join(INPUT_DIR, video_file)
    label = metadata_df.loc[video_file, 'label']
    
    if label == 'REAL':
        output_folder = os.path.join(OUTPUT_DIR, 'validation/real')
    else: # FAKE
        output_folder = os.path.join(OUTPUT_DIR, 'validation/fake')
        
    extract_frames(video_path, output_folder, FRAMES_PER_VIDEO)

print("\n--- Data Preparation Complete! ---")

