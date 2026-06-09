# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# Example of listing files (can be removed if not needed for final notebook)
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Install Ultralytics and its dependency thop from wheels
!pip install /kaggle/input/ultralytics-whl-pkg/ultralytics_thop-2.0.14-py3-none-any.whl -q --no-deps
!pip install /kaggle/input/ultralytics-whl-pkg/ultralytics-8.3.133-py3-none-any.whl -q --no-deps

import glob
from ultralytics import YOLO
import matplotlib.pyplot as plt # For displaying images
import cv2 # For explicit image loading and resizing


# --- Configuration for Kaggle Environment ---
KAGGLE_INPUT_DIR = '/kaggle/input/global-wheat-detection'
KAGGLE_TEST_IMAGE_DIR = os.path.join(KAGGLE_INPUT_DIR, 'test')
KAGGLE_MODEL_PATH = '/kaggle/input/gwd-yolov8s-best-weights/best.pt' # User provided path

KAGGLE_WORKING_DIR = '/kaggle/working'
KAGGLE_PREDICT_RUN_DIR = os.path.join(KAGGLE_WORKING_DIR, 'runs_predict') # Temp dir for predict outputs
KAGGLE_PREDICT_LABEL_DIR = os.path.join(KAGGLE_PREDICT_RUN_DIR, 'predict/labels') # YOLO saves labels here by default
KAGGLE_OUTPUT_SUBMISSION_FILE = os.path.join(KAGGLE_WORKING_DIR, 'submission.csv')

IMAGE_WIDTH = 1024.0
IMAGE_HEIGHT = 1024.0
CONFIDENCE_THRESHOLD = 0.35

print(f"Kaggle Test Image Dir: {KAGGLE_TEST_IMAGE_DIR}")
print(f"Kaggle Model Path: {KAGGLE_MODEL_PATH}")
print(f"Kaggle Submission File Path: {KAGGLE_OUTPUT_SUBMISSION_FILE}")


# --- 1. Perform Prediction ---
print("\nLoading YOLOv8 model...")
model = YOLO(KAGGLE_MODEL_PATH)
print("Model loaded.")

print(f"\nStarting prediction on images in: {KAGGLE_TEST_IMAGE_DIR}")
# os.makedirs(KAGGLE_PREDICT_RUN_DIR, exist_ok=True) # Not strictly needed if not saving txt files via project/name
# KAGGLE_PREDICT_LABEL_DIR is also not used if we process results directly

# The main 'results' variable will not be a list of all results at once anymore.
# We will process image by image. Visualization will be handled differently if re-added.
print("Predictions will be processed image by image with explicit resizing.")

# Visualization section removed for now to focus on core submission logic with resize fix.
# If this works, visualization can be re-added by collecting plotted images from per-image predictions.


# --- 2. Format Predictions for Submission ---
# This section will now include the prediction loop.
print("\nStarting prediction and formatting for submission...")

def yolo_to_voc_abs_kaggle(center_x_norm, center_y_norm, width_norm, height_norm, img_w, img_h):
    w_abs = width_norm * img_w
    h_abs = height_norm * img_h
    x_min_abs = (center_x_norm * img_w) - (w_abs / 2.0)
    y_min_abs = (center_y_norm * img_h) - (h_abs / 2.0)
    
    # Ensure results are integers and clamped to image boundaries
    x_min_abs = max(0, int(round(x_min_abs)))
    y_min_abs = max(0, int(round(y_min_abs)))
    w_abs = max(1, int(round(w_abs))) # Ensure width is at least 1
    h_abs = max(1, int(round(h_abs))) # Ensure height is at least 1

    # Clip to image boundaries
    if x_min_abs + w_abs > img_w:
        w_abs = int(img_w - x_min_abs)
    if y_min_abs + h_abs > img_h:
        h_abs = int(img_h - y_min_abs)
    
    # Ensure width and height are still positive after clipping
    w_abs = max(1, w_abs)
    h_abs = max(1, h_abs)
            
    return x_min_abs, y_min_abs, w_abs, h_abs

# Initialize submission_data here, before the try block
submission_data = []

# It's safer to iterate over images found in the test directory by the notebook.
test_image_files = []
if os.path.exists(KAGGLE_TEST_IMAGE_DIR):
    test_image_files = [f for f in os.listdir(KAGGLE_TEST_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
else:
    print(f"ERROR: Test image directory not found at {KAGGLE_TEST_IMAGE_DIR}")

print(f"Found {len(test_image_files)} images in {KAGGLE_TEST_IMAGE_DIR}")

try:
    for image_filename in test_image_files:
        image_id = os.path.splitext(image_filename)[0]
        img_path = os.path.join(KAGGLE_TEST_IMAGE_DIR, image_filename)
        prediction_string_parts = []
        # print(f"Processing image: {image_id}") # Optional: for verbose logging

        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Failed to load image {img_path}. Appending empty prediction.")
                submission_data.append({'image_id': image_id, 'PredictionString': ""})
                continue

            # Explicitly resize to 1024x1024 as per Kaggle discussion advice
            img_resized = cv2.resize(img, (int(IMAGE_WIDTH), int(IMAGE_HEIGHT)))
            
            # Predict on the single resized image
            # Set verbose=False to reduce log spam per image if many test images
            individual_results = model.predict(source=img_resized, imgsz=int(IMAGE_WIDTH), conf=CONFIDENCE_THRESHOLD, verbose=False) 
            
            if individual_results and individual_results[0].boxes and len(individual_results[0].boxes.xywhn) > 0:
                # .boxes.data gives [x1, y1, x2, y2, conf, cls]
                # We need normalized [cx, cy, w, h] and conf
                # results[0].boxes.xywhn gives normalized [cx, cy, w, h]
                # results[0].boxes.conf gives confidences
                
                norm_boxes = individual_results[0].boxes.xywhn.cpu().numpy()
                confs = individual_results[0].boxes.conf.cpu().numpy()

                for i in range(len(norm_boxes)):
                    center_x_norm, center_y_norm, width_norm, height_norm = norm_boxes[i]
                    confidence = confs[i]
                    
                    x_abs, y_abs, w_abs, h_abs = yolo_to_voc_abs_kaggle(
                        center_x_norm, center_y_norm, width_norm, height_norm,
                        IMAGE_WIDTH, IMAGE_HEIGHT # De-normalize based on the size fed to model (1024x1024)
                    )
                    prediction_string_parts.append(f"{confidence:.4f} {x_abs} {y_abs} {w_abs} {h_abs}")
            # else:
                # print(f"No boxes found for {image_id} above threshold {CONFIDENCE_THRESHOLD}")
                                
        except Exception as e_img_process:
            print(f"Error processing image {image_filename}: {e_img_process}")
            # Fallback to empty prediction string for this image if an error occurs during its processing
        
        full_prediction_string = " ".join(prediction_string_parts)
        submission_data.append({'image_id': image_id, 'PredictionString': full_prediction_string})

except Exception as e: # This outer try-except catches errors in the loop itself or file listing
    print(f"An error occurred during prediction processing: {e}")
    print("Will attempt to create a submission file based on sample_submission.csv or headers only.")
    # Ensure submission_data is empty if an error occurred mid-processing,
    # so it falls through to the logic that uses sample_submission.csv
    submission_data = []

# The yolo_to_voc_abs_kaggle function definition has been moved up.

# The loop for populating submission_data is now inside the try-except block above.
# This 'if' condition below handles the case where the loop completed but found no predictions for any existing test files.
if not submission_data and test_image_files: # Check if submission_data is empty AND test_image_files is NOT empty
     print("Warning: No predictions were made or processed, but test images exist. Submission file will have empty PredictionStrings.")
     # Ensure all test images are in the submission file, even if with empty predictions
     for image_filename in test_image_files:
        image_id = os.path.splitext(image_filename)[0]
        # Check if already added (should not happen if this block is reached)
        if not any(d['image_id'] == image_id for d in submission_data):
            submission_data.append({'image_id': image_id, 'PredictionString': ""})


submission_df = pd.DataFrame(submission_data)

# Ensure the DataFrame is not empty before saving, especially if test_image_files was empty
if not submission_df.empty:
    submission_df.to_csv(KAGGLE_OUTPUT_SUBMISSION_FILE, index=False)
    print(f"\nSubmission file created: {KAGGLE_OUTPUT_SUBMISSION_FILE}")
    print(f"Total images in submission: {len(submission_df)}")
    print("\nSample of submission (first 5 rows):")
    print(submission_df.head())
else:
    # Create an empty submission file with header if no test images were found at all
    # or if submission_data remained empty for other reasons.
    # Kaggle might expect a file even if it's just the header.
    print("Warning: No data to write to submission file. Creating a file with header only or based on sample_submission if available.")
    sample_submission_path = os.path.join(KAGGLE_INPUT_DIR, 'sample_submission.csv')
    if os.path.exists(sample_submission_path):
        try:
            sample_df = pd.read_csv(sample_submission_path)
            # Create an empty prediction string for all image_ids in sample_submission
            # This ensures the submission has the right image_ids if our processing failed to find any.
            empty_predictions = []
            for img_id in sample_df['image_id']:
                empty_predictions.append({'image_id': img_id, 'PredictionString': ""})
            final_empty_df = pd.DataFrame(empty_predictions)
            final_empty_df.to_csv(KAGGLE_OUTPUT_SUBMISSION_FILE, index=False)
            print(f"Created submission file from sample_submission.csv with empty predictions: {KAGGLE_OUTPUT_SUBMISSION_FILE}")
        except Exception as e:
            print(f"Error reading sample_submission.csv: {e}. Creating a header-only submission file.")
            pd.DataFrame(columns=['image_id', 'PredictionString']).to_csv(KAGGLE_OUTPUT_SUBMISSION_FILE, index=False)
            print(f"Created header-only submission file: {KAGGLE_OUTPUT_SUBMISSION_FILE}")
    else:
        pd.DataFrame(columns=['image_id', 'PredictionString']).to_csv(KAGGLE_OUTPUT_SUBMISSION_FILE, index=False)
        print(f"Created header-only submission file: {KAGGLE_OUTPUT_SUBMISSION_FILE}")


print("\nNotebook execution finished.")

