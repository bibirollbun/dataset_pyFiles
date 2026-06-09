# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import numpy as np 
import pandas as pd
import os
import cv2 # For image loading, resizing, and flipping
import glob
from tqdm import tqdm # For progress bars, optional but nice

# --- Package Installation ---
KAGGLE_WHEELS_DIR = '/kaggle/input/ultralytics-whl-pkg'
ULTRALYTICS_THOP_WHL = os.path.join(KAGGLE_WHEELS_DIR, 'ultralytics_thop-2.0.14-py3-none-any.whl') 
ULTRALYTICS_WHL = os.path.join(KAGGLE_WHEELS_DIR, 'ultralytics-8.3.133-py3-none-any.whl') 
ENSEMBLE_BOXES_WHL = os.path.join(KAGGLE_WHEELS_DIR, 'ensemble_boxes-1.0.9-py3-none-any.whl')

print(f"Installing {ULTRALYTICS_THOP_WHL}...")
!pip install {ULTRALYTICS_THOP_WHL} -q --no-deps
print(f"Installing {ULTRALYTICS_WHL}...")
!pip install {ULTRALYTICS_WHL} -q --no-deps
print(f"Installing {ENSEMBLE_BOXES_WHL}...")
!pip install {ENSEMBLE_BOXES_WHL} -q --no-deps

from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion




# --- Configuration for Kaggle Environment & Model ---
KAGGLE_INPUT_DIR = '/kaggle/input/global-wheat-detection'
KAGGLE_TEST_IMAGE_DIR = os.path.join(KAGGLE_INPUT_DIR, 'test')
KAGGLE_MODEL_PATH = '/kaggle/input/gwhd-2020-best-weights-ever/FinalBestExtraData.pt' 

KAGGLE_WORKING_DIR = '/kaggle/working'
KAGGLE_OUTPUT_SUBMISSION_FILE = os.path.join(KAGGLE_WORKING_DIR, 'submission.csv')

IMAGE_WIDTH = 1024.0
IMAGE_HEIGHT = 1024.0


BASE_PREDICT_CONF = 0.01 
WBF_IOU_THR = 0.55
WBF_SKIP_BOX_THR = 0.25
FINAL_SCORE_THR = 0.12

print(f"Kaggle Test Image Dir: {KAGGLE_TEST_IMAGE_DIR}")
print(f"Kaggle Model Path: {KAGGLE_MODEL_PATH}")
print(f"Kaggle Submission File Path: {KAGGLE_OUTPUT_SUBMISSION_FILE}")
print(f"Using WBF IoU Thr: {WBF_IOU_THR}, WBF Skip Box Thr: {WBF_SKIP_BOX_THR}, Final Score Thr: {FINAL_SCORE_THR}")


# --- Helper Function for Box Conversion (same as before) ---
def yolo_normalized_xyxyn_to_submission_xywh(norm_boxes_xyxy, scores, img_w, img_h):
    prediction_strings = []
    for i in range(len(norm_boxes_xyxy)):
        nx1, ny1, nx2, ny2 = norm_boxes_xyxy[i]
        conf = scores[i]
        x1_abs = nx1 * img_w; y1_abs = ny1 * img_h
        x2_abs = nx2 * img_w; y2_abs = ny2 * img_h
        abs_w = x2_abs - x1_abs; abs_h = y2_abs - y1_abs
        x_min_final = max(0, int(round(x1_abs))); y_min_final = max(0, int(round(y1_abs)))
        w_final = int(round(abs_w)); h_final = int(round(abs_h))
        if x_min_final + w_final > img_w: w_final = int(img_w - x_min_final)
        if y_min_final + h_final > img_h: h_final = int(img_h - y_min_final)
        w_final = max(1, w_final); h_final = max(1, h_final)
        prediction_strings.append(f"{conf:.4f} {x_min_final} {y_min_final} {w_final} {h_final}")
    return prediction_strings

# --- Main Logic (with TTA and WBF, same structure as before) ---
submission_data = []
test_image_files = []

try:
    model = YOLO(KAGGLE_MODEL_PATH)
    print("Model loaded.")

    if os.path.exists(KAGGLE_TEST_IMAGE_DIR):
        test_image_files = [f for f in os.listdir(KAGGLE_TEST_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        print(f"ERROR: Test image directory not found at {KAGGLE_TEST_IMAGE_DIR}")

    print(f"Found {len(test_image_files)} images to process for submission.")

    for image_filename in tqdm(test_image_files, desc="Processing test images"):
        image_id = os.path.splitext(image_filename)[0]
        img_path = os.path.join(KAGGLE_TEST_IMAGE_DIR, image_filename)
        
        tta_boxes_list_norm = [] 
        tta_scores_list = []   
        tta_labels_list = []   

        try:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"Warning: Failed to load image {img_path}. Appending empty prediction.")
                submission_data.append({'image_id': image_id, 'PredictionString': ""})
                continue

            img_resized_bgr = cv2.resize(img_bgr, (int(IMAGE_WIDTH), int(IMAGE_HEIGHT)))

            for tta_idx in range(2): # 0: original, 1: hflip
                img_to_predict = img_resized_bgr.copy()
                if tta_idx == 1: 
                    img_to_predict = cv2.flip(img_to_predict, 1)

                individual_results = model.predict(source=img_to_predict, imgsz=int(IMAGE_WIDTH), conf=BASE_PREDICT_CONF, verbose=False)
                
                current_tta_boxes_norm = []
                current_tta_scores = []
                
                if individual_results and individual_results[0].boxes:
                    norm_boxes_xyxy_tta_space = individual_results[0].boxes.xyxyn.cpu().numpy() 
                    confs = individual_results[0].boxes.conf.cpu().numpy()

                    for i in range(len(norm_boxes_xyxy_tta_space)):
                        nx1, ny1, nx2, ny2 = norm_boxes_xyxy_tta_space[i]
                        if tta_idx == 1: 
                            nx1_orig = 1.0 - nx2; nx2_orig = 1.0 - nx1
                            current_tta_boxes_norm.append([nx1_orig, ny1, nx2_orig, ny2])
                        else: 
                            current_tta_boxes_norm.append([nx1, ny1, nx2, ny2])
                        current_tta_scores.append(confs[i])
                
                if current_tta_boxes_norm: 
                    tta_boxes_list_norm.append(current_tta_boxes_norm)
                    tta_scores_list.append(current_tta_scores)
                    tta_labels_list.append([0] * len(current_tta_scores))
            
            final_prediction_string_parts = []
            if tta_boxes_list_norm: 
                fused_boxes_norm, fused_scores, _ = weighted_boxes_fusion(
                    tta_boxes_list_norm, tta_scores_list, tta_labels_list,
                    iou_thr=WBF_IOU_THR, skip_box_thr=WBF_SKIP_BOX_THR
                )
                indices = fused_scores >= FINAL_SCORE_THR
                final_fused_boxes_norm_filtered = fused_boxes_norm[indices]
                final_fused_scores_filtered = fused_scores[indices]
                
                if len(final_fused_boxes_norm_filtered) > 0:
                    final_prediction_string_parts = yolo_normalized_xyxyn_to_submission_xywh(
                        final_fused_boxes_norm_filtered, final_fused_scores_filtered, IMAGE_WIDTH, IMAGE_HEIGHT
                    )
            
            full_prediction_string = " ".join(final_prediction_string_parts)
            submission_data.append({'image_id': image_id, 'PredictionString': full_prediction_string})

        except Exception as e_img_process:
            print(f"Error processing image {image_filename}: {e_img_process}")
            submission_data.append({'image_id': image_id, 'PredictionString': ""})

except Exception as e_main:
    print(f"A critical error occurred in the main processing block: {e_main}")
    submission_data = [] 

finally:
    print("\nExecuting finally block to ensure submission file is written.")
    if not submission_data and test_image_files:
        print("INFO (finally): submission_data is empty but test_image_files were found. Populating with empty predictions.")
        for image_filename in test_image_files:
            image_id = os.path.splitext(image_filename)[0]
            if not any(d['image_id'] == image_id for d in submission_data):
                submission_data.append({'image_id': image_id, 'PredictionString': ""})
    
    print(f"INFO (finally): Number of records in submission_data before creating DataFrame: {len(submission_data)}")
    submission_df = pd.DataFrame(submission_data)

    if not submission_df.empty:
        print(f"INFO (finally): Attempting to write submission_df (shape: {submission_df.shape}) to {KAGGLE_OUTPUT_SUBMISSION_FILE}")
        try:
            submission_df.to_csv(KAGGLE_OUTPUT_SUBMISSION_FILE, index=False)
            print(f"INFO (finally): Successfully wrote submission_df to {KAGGLE_OUTPUT_SUBMISSION_FILE}")
            if os.path.exists(KAGGLE_OUTPUT_SUBMISSION_FILE):
                print(f"INFO (finally): File {KAGGLE_OUTPUT_SUBMISSION_FILE} confirmed to exist after writing.")
            else:
                print(f"CRITICAL_ERROR (finally): File {KAGGLE_OUTPUT_SUBMISSION_FILE} NOT FOUND after attempting to write submission_df!")
        except Exception as e_csv:
            print(f"CRITICAL_ERROR (finally): Failed to write submission_df to CSV: {e_csv}")
    else:
        print("INFO (finally): submission_df is empty. Attempting to create fallback submission file.")
        sample_submission_path = os.path.join(KAGGLE_INPUT_DIR, 'sample_submission.csv')
        if os.path.exists(sample_submission_path):
            try:
                sample_df = pd.read_csv(sample_submission_path)
                empty_predictions = [{'image_id': img_id, 'PredictionString': ""} for img_id in sample_df['image_id']]
                final_empty_df = pd.DataFrame(empty_predictions)
                print(f"INFO (finally): Attempting to write fallback (from sample) to {KAGGLE_OUTPUT_SUBMISSION_FILE}")
                final_empty_df.to_csv(KAGGLE_OUTPUT_SUBMISSION_FILE, index=False)
                print(f"INFO (finally): Successfully wrote fallback (from sample) to {KAGGLE_OUTPUT_SUBMISSION_FILE}")
            except Exception as e_sample:
                print(f"CRITICAL_ERROR (finally): Error with sample_submission fallback: {e_sample}. Creating header-only file.")
                pd.DataFrame(columns=['image_id', 'PredictionString']).to_csv(KAGGLE_OUTPUT_SUBMISSION_FILE, index=False)
        else:
            print("INFO (finally): sample_submission.csv not found. Creating header-only submission file.")
            pd.DataFrame(columns=['image_id', 'PredictionString']).to_csv(KAGGLE_OUTPUT_SUBMISSION_FILE, index=False)
            print(f"INFO (finally): Successfully wrote header-only file to {KAGGLE_OUTPUT_SUBMISSION_FILE}")

    if os.path.exists(KAGGLE_OUTPUT_SUBMISSION_FILE):
        print(f"FINAL_CONFIRMATION: {KAGGLE_OUTPUT_SUBMISSION_FILE} exists at the end of the finally block.")
    else:
        print(f"FINAL_CRITICAL_ERROR: {KAGGLE_OUTPUT_SUBMISSION_FILE} DOES NOT EXIST at the end of the finally block.")
        
    print("\nNotebook execution finished (from finally block).")


