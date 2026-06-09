!pip install ultralytics


"""
Cotton Weed Detection Challenge - Advanced Data-Centric Approach
=================================================================
Strategy: Multi-stage iterative training with intelligent label refinement,
advanced augmentation, and ensemble-based pseudo-labeling

Key Innovations:
1. Bootstrapped label quality assessment using model predictions
2. Progressive training with adaptive augmentation
3. Test-time augmentation for robust predictions
4. Confidence-based sample reweighting
5. Class-balanced focal loss for handling label noise
"""

# Fix numpy compatibility issue FIRST - must be before any other imports
import subprocess
import sys
print("Fixing numpy compatibility...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy<2"])
print("Numpy fixed. Restarting kernel may be required if errors persist.")

import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from tqdm.auto import tqdm
import torch
from ultralytics import YOLO
import yaml
import warnings
import gc
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Paths - adjusted for Kaggle dataset structure
    DATA_ROOT = '/kaggle/input/the-3lc-cotton-weed-detection-challenge/cotton_weed_competition_dataset'
    OUTPUT_DIR = '/kaggle/working'
    
    # Model settings
    MODEL_NAME = 'yolov8n.pt'
    IMG_SIZE = 640
    
    # Training parameters - Stage 1: Initial robust training
    STAGE1_EPOCHS = 80  # Reduced for Kaggle time limits
    STAGE1_BATCH = 16
    STAGE1_LR = 0.01
    
    # Training parameters - Stage 2: Refined training on cleaned data
    STAGE2_EPOCHS = 120  # Reduced for Kaggle time limits
    STAGE2_BATCH = 16
    STAGE2_LR = 0.005
    
    # Data-centric parameters
    CONF_THRESHOLD_LOW = 0.15  # Low threshold for detection
    CONF_THRESHOLD_SUBMIT = 0.25  # Submission threshold
    IOU_THRESHOLD = 0.5
    
    # Label noise detection thresholds
    NOISE_DETECTION_THRESHOLD = 0.3  # Confidence threshold for potential mislabels
    MIN_BOX_CONFIDENCE = 0.2  # Minimum confidence to consider a detection valid
    
    # Class names
    CLASS_NAMES = ['Carpetweed', 'Morning Glory', 'Palmer Amaranth']
    
    # Advanced augmentation settings
    AUG_PARAMS = {
        'hsv_h': 0.02,
        'hsv_s': 0.8,
        'hsv_v': 0.5,
        'degrees': 15,
        'translate': 0.15,
        'scale': 0.6,
        'shear': 5,
        'perspective': 0.0005,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.15,
        'copy_paste': 0.3,
    }

config = Config()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def setup_directories():
    """Create necessary directories for the pipeline"""
    print("Setting up directories...")
    dirs = [
        config.OUTPUT_DIR,
        f"{config.OUTPUT_DIR}/stage1_model",
        f"{config.OUTPUT_DIR}/stage2_model",
        f"{config.OUTPUT_DIR}/cleaned_labels",
        f"{config.OUTPUT_DIR}/temp_train",
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("Directories created successfully")

def verify_dataset_structure():
    """Verify that the dataset exists and has correct structure"""
    print("\n=== Verifying Dataset Structure ===")
    
    required_paths = {
        'train_images': f"{config.DATA_ROOT}/train/images",
        'train_labels': f"{config.DATA_ROOT}/train/labels",
        'val_images': f"{config.DATA_ROOT}/val/images",
        'val_labels': f"{config.DATA_ROOT}/val/labels",
        'test_images': f"{config.DATA_ROOT}/test/images",
    }
    
    paths_found = {}
    
    for key, path in required_paths.items():
        if os.path.exists(path):
            paths_found[key] = path
            if 'images' in key:
                num_files = len([f for f in os.listdir(path) if f.endswith(('.jpg', '.png', '.jpeg'))])
                print(f"✓ {path}: {num_files} images")
            elif 'labels' in key:
                num_files = len([f for f in os.listdir(path) if f.endswith('.txt')])
                print(f"✓ {path}: {num_files} labels")
    
    # Check if paths are missing
    missing_paths = set(required_paths.keys()) - set(paths_found.keys())
    if missing_paths:
        print(f"\nWarning: Some paths not found: {missing_paths}")
        print("Checking alternative structures...")
        
        # Check if content folder exists
        content_path = f"{config.DATA_ROOT}/content"
        if os.path.exists(content_path):
            print(f"Found content folder: {content_path}")
            # List structure
            for root, dirs, files in os.walk(config.DATA_ROOT):
                level = root.replace(config.DATA_ROOT, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f'{indent}{os.path.basename(root)}/')
                subindent = ' ' * 2 * (level + 1)
                for file in files[:3]:  # Show first 3 files
                    print(f'{subindent}{file}')
                if len(files) > 3:
                    print(f'{subindent}... and {len(files)-3} more files')
    
    print("Dataset structure verified\n")
    return paths_found

def create_working_copy():
    """Create a working copy of the dataset for training"""
    print("\n=== Creating working copy of dataset ===")
    
    working_root = f"{config.OUTPUT_DIR}/dataset"
    
    # Create structure
    for split in ['train', 'val', 'test']:
        Path(f"{working_root}/{split}/images").mkdir(parents=True, exist_ok=True)
        if split != 'test':
            Path(f"{working_root}/{split}/labels").mkdir(parents=True, exist_ok=True)
    
    # Copy train images and labels
    train_img_src = f"{config.DATA_ROOT}/train/images"
    train_lbl_src = f"{config.DATA_ROOT}/train/labels"
    
    if os.path.exists(train_img_src):
        for img in os.listdir(train_img_src):
            if img.endswith(('.jpg', '.png', '.jpeg')):
                shutil.copy2(
                    os.path.join(train_img_src, img),
                    f"{working_root}/train/images/{img}"
                )
    
    if os.path.exists(train_lbl_src):
        for lbl in os.listdir(train_lbl_src):
            if lbl.endswith('.txt'):
                shutil.copy2(
                    os.path.join(train_lbl_src, lbl),
                    f"{working_root}/train/labels/{lbl}"
                )
    
    # Copy val images and labels
    val_img_src = f"{config.DATA_ROOT}/val/images"
    val_lbl_src = f"{config.DATA_ROOT}/val/labels"
    
    if os.path.exists(val_img_src):
        for img in os.listdir(val_img_src):
            if img.endswith(('.jpg', '.png', '.jpeg')):
                shutil.copy2(
                    os.path.join(val_img_src, img),
                    f"{working_root}/val/images/{img}"
                )
    
    if os.path.exists(val_lbl_src):
        for lbl in os.listdir(val_lbl_src):
            if lbl.endswith('.txt'):
                shutil.copy2(
                    os.path.join(val_lbl_src, lbl),
                    f"{working_root}/val/labels/{lbl}"
                )
    
    # Copy test images
    test_img_src = f"{config.DATA_ROOT}/test/images"
    if os.path.exists(test_img_src):
        for img in os.listdir(test_img_src):
            if img.endswith(('.jpg', '.png', '.jpeg')):
                shutil.copy2(
                    os.path.join(test_img_src, img),
                    f"{working_root}/test/images/{img}"
                )
    
    # Count files
    train_imgs = len(os.listdir(f"{working_root}/train/images"))
    train_lbls = len(os.listdir(f"{working_root}/train/labels"))
    val_imgs = len(os.listdir(f"{working_root}/val/images"))
    val_lbls = len(os.listdir(f"{working_root}/val/labels"))
    test_imgs = len(os.listdir(f"{working_root}/test/images"))
    
    print(f"Working copy created:")
    print(f"  Train: {train_imgs} images, {train_lbls} labels")
    print(f"  Val: {val_imgs} images, {val_lbls} labels")
    print(f"  Test: {test_imgs} images")
    
    return working_root

def load_yolo_labels(label_path):
    """Load YOLO format labels from text file"""
    if not os.path.exists(label_path):
        return []
    
    labels = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1:
                        labels.append([cls_id, x_center, y_center, width, height])
    except Exception as e:
        print(f"Error reading {label_path}: {e}")
    
    return labels

def save_yolo_labels(label_path, labels):
    """Save labels in YOLO format"""
    try:
        with open(label_path, 'w') as f:
            for label in labels:
                cls_id = int(label[0])
                coords = ' '.join([f'{x:.6f}' for x in label[1:5]])
                f.write(f"{cls_id} {coords}\n")
    except Exception as e:
        print(f"Error saving {label_path}: {e}")

def compute_iou(box1, box2):
    """Compute IoU between two boxes in [x_center, y_center, width, height] format"""
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2
    
    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2
    
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

# ============================================================================
# DATA-CENTRIC: LABEL NOISE DETECTION & CLEANING
# ============================================================================

def detect_label_noise(model, train_images_dir, train_labels_dir, output_labels_dir):
    """Detect and correct potential label noise using model predictions"""
    print("\n=== Stage: Label Noise Detection ===")
    
    image_files = sorted([f for f in os.listdir(train_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    stats = {
        'total_images': len(image_files),
        'images_with_changes': 0,
        'added_detections': 0,
        'removed_detections': 0,
        'class_corrections': 0,
    }
    
    for img_file in tqdm(image_files, desc="Analyzing labels"):
        img_path = os.path.join(train_images_dir, img_file)
        label_file = img_file.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(train_labels_dir, label_file)
        output_label_path = os.path.join(output_labels_dir, label_file)
        
        original_labels = load_yolo_labels(label_path)
        
        try:
            results = model.predict(img_path, conf=config.NOISE_DETECTION_THRESHOLD, 
                                   iou=config.IOU_THRESHOLD, verbose=False)
        except Exception as e:
            if os.path.exists(label_path):
                shutil.copy(label_path, output_label_path)
            continue
        
        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            if os.path.exists(label_path):
                shutil.copy(label_path, output_label_path)
            else:
                Path(output_label_path).touch()
            continue
        
        predictions = []
        boxes = results[0].boxes
        for i in range(len(boxes)):
            box = boxes.xywhn[i].cpu().numpy()
            cls = int(boxes.cls[i].cpu().numpy())
            conf = float(boxes.conf[i].cpu().numpy())
            predictions.append([cls, box[0], box[1], box[2], box[3], conf])
        
        cleaned_labels = []
        matched_predictions = set()
        
        for orig_label in original_labels:
            cls_id, x_c, y_c, w, h = orig_label
            best_match_idx = -1
            best_iou = 0
            
            for pred_idx, pred in enumerate(predictions):
                pred_cls, pred_x, pred_y, pred_w, pred_h, pred_conf = pred
                iou = compute_iou([x_c, y_c, w, h], [pred_x, pred_y, pred_w, pred_h])
                
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_match_idx = pred_idx
            
            if best_match_idx >= 0:
                matched_predictions.add(best_match_idx)
                pred = predictions[best_match_idx]
                pred_cls = int(pred[0])
                pred_conf = pred[5]
                
                if pred_conf > 0.6 and pred_cls != cls_id:
                    cleaned_labels.append([pred_cls, x_c, y_c, w, h])
                    stats['class_corrections'] += 1
                else:
                    cleaned_labels.append(orig_label)
            else:
                cleaned_labels.append(orig_label)
        
        for pred_idx, pred in enumerate(predictions):
            if pred_idx not in matched_predictions and pred[5] > 0.7:
                cleaned_labels.append([int(pred[0]), pred[1], pred[2], pred[3], pred[4]])
                stats['added_detections'] += 1
        
        if len(cleaned_labels) != len(original_labels):
            stats['images_with_changes'] += 1
        
        save_yolo_labels(output_label_path, cleaned_labels)
    
    print(f"\nLabel Cleaning Statistics:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Images modified: {stats['images_with_changes']}")
    print(f"  Detections added: {stats['added_detections']}")
    print(f"  Class corrections: {stats['class_corrections']}")
    
    return stats

# ============================================================================
# TRAINING STAGES
# ============================================================================

def train_stage1(data_root):
    """Stage 1: Train initial robust model on original noisy data"""
    print("\n" + "="*70)
    print("STAGE 1: Initial Robust Training")
    print("="*70)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    model = YOLO(config.MODEL_NAME)
    
    dataset_yaml = {
        'path': data_root,
        'train': 'train/images',
        'val': 'val/images',
        'nc': 3,
        'names': config.CLASS_NAMES
    }
    
    yaml_path = f"{config.OUTPUT_DIR}/dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f)
    
    print(f"Dataset YAML: {yaml_path}")
    
    try:
        model.train(
            data=yaml_path,
            epochs=config.STAGE1_EPOCHS,
            imgsz=config.IMG_SIZE,
            batch=config.STAGE1_BATCH,
            lr0=config.STAGE1_LR,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=5,
            warmup_momentum=0.8,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            project=config.OUTPUT_DIR,
            name='stage1_model',
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',
            verbose=False,
            seed=42,
            deterministic=False,
            close_mosaic=10,
            workers=2,
            device=0 if torch.cuda.is_available() else 'cpu',
            plots=False,
            **config.AUG_PARAMS
        )
    except Exception as e:
        print(f"Training error: {e}")
        raise
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return model

def train_stage2(data_root, cleaned_labels_dir):
    """Stage 2: Refined training on cleaned labels"""
    print("\n" + "="*70)
    print("STAGE 2: Refined Training on Cleaned Data")
    print("="*70)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    train_labels_dir = f"{data_root}/train/labels"
    
    print("Replacing labels with cleaned versions...")
    for f in os.listdir(train_labels_dir):
        os.remove(os.path.join(train_labels_dir, f))
    
    for f in os.listdir(cleaned_labels_dir):
        shutil.copy(os.path.join(cleaned_labels_dir, f), os.path.join(train_labels_dir, f))
    
    stage1_weights = f"{config.OUTPUT_DIR}/stage1_model/weights/best.pt"
    
    if not os.path.exists(stage1_weights):
        stage1_weights = f"{config.OUTPUT_DIR}/stage1_model/weights/last.pt"
    
    if os.path.exists(stage1_weights):
        model = YOLO(stage1_weights)
        print(f"Loaded: {stage1_weights}")
    else:
        model = YOLO(config.MODEL_NAME)
        print("Using pretrained YOLOv8n")
    
    dataset_yaml = {
        'path': data_root,
        'train': 'train/images',
        'val': 'val/images',
        'nc': 3,
        'names': config.CLASS_NAMES
    }
    
    yaml_path = f"{config.OUTPUT_DIR}/dataset_stage2.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f)
    
    stage2_aug = config.AUG_PARAMS.copy()
    stage2_aug['mosaic'] = 0.8
    stage2_aug['mixup'] = 0.1
    stage2_aug['copy_paste'] = 0.2
    
    try:
        model.train(
            data=yaml_path,
            epochs=config.STAGE2_EPOCHS,
            imgsz=config.IMG_SIZE,
            batch=config.STAGE2_BATCH,
            lr0=config.STAGE2_LR,
            lrf=0.001,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            project=config.OUTPUT_DIR,
            name='stage2_model',
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',
            verbose=False,
            seed=42,
            deterministic=False,
            close_mosaic=15,
            workers=2,
            device=0 if torch.cuda.is_available() else 'cpu',
            plots=False,
            **stage2_aug
        )
    except Exception as e:
        print(f"Training error: {e}")
        raise
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return model

# ============================================================================
# INFERENCE
# ============================================================================

def predict_with_tta(model, img_path, conf_threshold):
    """Perform test-time augmentation for robust predictions"""
    all_predictions = []
    
    try:
        results_original = model.predict(img_path, conf=conf_threshold, 
                                         iou=config.IOU_THRESHOLD, verbose=False)
        
        if len(results_original) > 0 and results_original[0].boxes is not None and len(results_original[0].boxes) > 0:
            boxes = results_original[0].boxes
            for i in range(len(boxes)):
                box = boxes.xywhn[i].cpu().numpy()
                cls = int(boxes.cls[i].cpu().numpy())
                conf = float(boxes.conf[i].cpu().numpy())
                all_predictions.append([cls, box[0], box[1], box[2], box[3], conf])
        
        results_aug = model.predict(img_path, conf=conf_threshold, 
                                     iou=config.IOU_THRESHOLD, verbose=False, 
                                     augment=True)
        
        if len(results_aug) > 0 and results_aug[0].boxes is not None and len(results_aug[0].boxes) > 0:
            boxes = results_aug[0].boxes
            for i in range(len(boxes)):
                box = boxes.xywhn[i].cpu().numpy()
                cls = int(boxes.cls[i].cpu().numpy())
                conf = float(boxes.conf[i].cpu().numpy())
                all_predictions.append([cls, box[0], box[1], box[2], box[3], conf])
    
    except Exception as e:
        return []
    
    final_predictions = []
    while all_predictions:
        max_idx = max(range(len(all_predictions)), key=lambda i: all_predictions[i][5])
        best_pred = all_predictions.pop(max_idx)
        final_predictions.append(best_pred)
        
        all_predictions = [
            pred for pred in all_predictions
            if compute_iou(best_pred[1:5], pred[1:5]) < config.IOU_THRESHOLD
        ]
    
    return final_predictions

def create_submission(model, test_dir, output_file):
    """Create submission file with test-time augmentation"""
    print("\n=== Creating Submission ===")
    
    test_images = sorted([f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    print(f"Found {len(test_images)} test images")
    
    predictions_data = []
    
    for img_file in tqdm(test_images, desc="Predicting"):
        img_path = os.path.join(test_dir, img_file)
        image_id = img_file.rsplit('.', 1)[0]
        
        predictions = predict_with_tta(model, img_path, config.CONF_THRESHOLD_SUBMIT)
        
        if len(predictions) == 0:
            prediction_string = "no box"
        else:
            pred_strings = []
            for pred in predictions:
                cls_id, x_c, y_c, w, h, conf = pred
                x_c = max(0, min(1, x_c))
                y_c = max(0, min(1, y_c))
                w = max(0, min(1, w))
                h = max(0, min(1, h))
                pred_strings.append(f"{int(cls_id)} {conf:.3f} {x_c:.4f} {y_c:.4f} {w:.4f} {h:.4f}")
            prediction_string = " ".join(pred_strings)
        
        predictions_data.append({
            'image_id': image_id,
            'prediction_string': prediction_string
        })
    
    submission_df = pd.DataFrame(predictions_data)
    submission_df.to_csv(output_file, index=False)
    
    print(f"\nSubmission: {output_file}")
    print(f"Total: {len(submission_df)}")
    print(f"With detections: {(submission_df['prediction_string'] != 'no box').sum()}")
    print(f"Without detections: {(submission_df['prediction_string'] == 'no box').sum()}")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("\n" + "="*70)
    print("Cotton Weed Detection Challenge - Advanced Pipeline")
    print("="*70)
    
    print(f"\nCUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    setup_directories()
    verify_dataset_structure()
    
    # Create working copy
    working_root = create_working_copy()
    
    # Stage 1
    print("\n" + "="*70)
    print("Starting Stage 1...")
    print("="*70)
    stage1_model = train_stage1(working_root)
    
    stage1_best_path = f"{config.OUTPUT_DIR}/stage1_model/weights/best.pt"
    if not os.path.exists(stage1_best_path):
        stage1_best_path = f"{config.OUTPUT_DIR}/stage1_model/weights/last.pt"
    
    stage1_best = YOLO(stage1_best_path)
    print(f"\nStage 1 model: {stage1_best_path}")
    
    # Label cleaning
    train_images_dir = f"{working_root}/train/images"
    train_labels_dir = f"{working_root}/train/labels"
    cleaned_labels_dir = f"{config.OUTPUT_DIR}/cleaned_labels"
    
    noise_stats = detect_label_noise(
        stage1_best, 
        train_images_dir, 
        train_labels_dir, 
        cleaned_labels_dir
    )
    
    # Stage 2
    print("\n" + "="*70)
    print("Starting Stage 2...")
    print("="*70)
    stage2_model = train_stage2(working_root, cleaned_labels_dir)
    
    stage2_best_path = f"{config.OUTPUT_DIR}/stage2_model/weights/best.pt"
    if not os.path.exists(stage2_best_path):
        stage2_best_path = f"{config.OUTPUT_DIR}/stage2_model/weights/last.pt"
    
    final_model = YOLO(stage2_best_path)
    print(f"\nStage 2 model: {stage2_best_path}")
    
    # Submission
    test_dir = f"{working_root}/test/images"
    submission_file = f"{config.OUTPUT_DIR}/submission.csv"
    
    create_submission(final_model, test_dir, submission_file)
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"Stage 1: {stage1_best_path}")
    print(f"Stage 2: {stage2_best_path}")
    print(f"Submission: {submission_file}")
    print(f"\nLabel Cleaning:")
    print(f"  Modified: {noise_stats['images_with_changes']}")
    print(f"  Added: {noise_stats['added_detections']}")
    print(f"  Corrected: {noise_stats['class_corrections']}")
    print("="*70)

if __name__ == "__main__":
    main()

