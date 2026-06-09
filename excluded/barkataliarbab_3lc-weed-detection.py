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


# Cell 1: Install packages
!pip install ultralytics opencv-python-headless

# Cell 2: Run your script
import os
import shutil
import cv2
from ultralytics import YOLO
# ... rest of your code



import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob


# ============================================================================
# COTTON WEED DETECTION - COMPLETE NOTEBOOK WITH PATH FIXES
# ============================================================================

# 1. IMPORTS
import os
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import yaml
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Ultralytics
from ultralytics import YOLO
import torch

# Set up environment
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ============================================================================
# 2. FIX DATASET YAML FILE
# ============================================================================

print("\n" + "="*50)
print("FIXING DATASET PATHS")
print("="*50)

# Original dataset YAML path
original_yaml_path = "/kaggle/input/the-3lc-cotton-weed-detection-challenge/cotton_weed_competition_dataset/dataset.yaml"

# Read the original YAML
with open(original_yaml_path, 'r') as f:
    original_yaml_content = yaml.safe_load(f)

print("Original YAML content:")
print(f"Path: {original_yaml_content.get('path', 'Not found')}")
print(f"Train: {original_yaml_content.get('train', 'Not found')}")
print(f"Val: {original_yaml_content.get('val', 'Not found')}")
print(f"Test: {original_yaml_content.get('test', 'Not found')}")
print(f"Classes: {original_yaml_content.get('names', 'Not found')}")

# Create corrected YAML with absolute paths
corrected_yaml = f"""
# Corrected dataset paths for Kaggle
path: /kaggle/input/the-3lc-cotton-weed-detection-challenge/cotton_weed_competition_dataset
train: train/images
val: val/images
test: test/images

# Class names
names: {original_yaml_content.get('names', {})}
nc: {len(original_yaml_content.get('names', {}))}
"""

# Save corrected YAML
corrected_yaml_path = "/kaggle/working/corrected_dataset.yaml"
with open(corrected_yaml_path, 'w') as f:
    f.write(corrected_yaml)

print(f"\n✓ Corrected YAML saved to: {corrected_yaml_path}")

# ============================================================================
# 3. CHECK DATASET STRUCTURE
# ============================================================================

def check_dataset_structure():
    """Verify dataset structure is correct"""
    print("\n" + "="*50)
    print("CHECKING DATASET STRUCTURE")
    print("="*50)
    
    base_path = "/kaggle/input/the-3lc-cotton-weed-detection-challenge/cotton_weed_competition_dataset"
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        print(f"\n{split.upper()}:")
        print("-" * 20)
        
        # Check images
        images_path = os.path.join(base_path, split, "images")
        if os.path.exists(images_path):
            image_files = list(Path(images_path).glob("*.jpg"))
            print(f"  Images: {len(image_files)} files")
            if image_files:
                # Show first few image sizes
                for img_path in image_files[:2]:
                    try:
                        with Image.open(img_path) as img:
                            print(f"    Sample: {img_path.name} - Size: {img.size}")
                    except:
                        print(f"    Sample: {img_path.name} - Cannot read")
        else:
            print(f"  Images: Directory not found")
        
        # Check labels
        labels_path = os.path.join(base_path, split, "labels")
        if os.path.exists(labels_path):
            label_files = list(Path(labels_path).glob("*.txt"))
            print(f"  Labels: {len(label_files)} files")
        else:
            print(f"  Labels: Directory not found")

# Run check
check_dataset_structure()

# ============================================================================
# 4. LOAD OR TRAIN MODEL (WITH FIXED PATHS)
# ============================================================================

print("\n" + "="*50)
print("MODEL SETUP")
print("="*50)

# Try to find existing model first
existing_models = []
for model_file in ['best.pt', 'last.pt']:
    model_path = f"/kaggle/working/{model_file}"
    if os.path.exists(model_path):
        existing_models.append(model_path)

if existing_models:
    print(f"Found existing models: {existing_models}")
    model = YOLO(existing_models[0])
    print(f"✓ Loaded existing model: {existing_models[0]}")
    print(f"  Model classes: {model.names}")
else:
    print("No existing model found. Training new model...")
    
    # Load a small model for quick training
    model = YOLO('yolov8n.pt')
    
    print(f"\nStarting training with corrected dataset YAML...")
    print(f"Dataset YAML: {corrected_yaml_path}")
    
    # Train the model with corrected paths
    try:
        results = model.train(
            data=corrected_yaml_path,
            epochs=30,
            imgsz=640,
            batch=16,
            patience=10,
            device=0 if torch.cuda.is_available() else 'cpu',
            workers=4,
            lr0=0.01,
            optimizer='AdamW',
            cos_lr=True,
            save=True,
            save_period=5,
            project='/kaggle/working',
            name='cotton_weed_model',
            exist_ok=True,
            verbose=True
        )
        
        print("✓ Model training completed!")
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        print("\nTrying alternative approach with direct paths...")
        
        # Alternative: Create YAML with full paths
        alternative_yaml = f"""
path: /kaggle/input/the-3lc-cotton-weed-detection-challenge/cotton_weed_competition_dataset
train: /kaggle/input/the-3lc-cotton-weed-detection-challenge/cotton_weed_competition_dataset/train/images
val: /kaggle/input/the-3lc-cotton-weed-detection-challenge/cotton_weed_competition_dataset/val/images
test: /kaggle/input/the-3lc-cotton-weed-detection-challenge/cotton_weed_competition_dataset/test/images

names: {original_yaml_content.get('names', {})}
nc: {len(original_yaml_content.get('names', {}))}
"""
        
        alt_yaml_path = "/kaggle/working/alternative_dataset.yaml"
        with open(alt_yaml_path, 'w') as f:
            f.write(alternative_yaml)
        
        print(f"Created alternative YAML: {alt_yaml_path}")
        
        # Try training again
        results = model.train(
            data=alt_yaml_path,
            epochs=20,
            imgsz=640,
            batch=8,
            device=0 if torch.cuda.is_available() else 'cpu',
            workers=2,
            verbose=True,
            project='/kaggle/working',
            name='quick_train'
        )

# ============================================================================
# 5. VERIFY MODEL
# ============================================================================

print("\n" + "="*50)
print("MODEL VERIFICATION")
print("="*50)

if hasattr(model, 'names'):
    print(f"Model classes: {model.names}")
    print(f"Number of classes: {len(model.names)}")
    
    # Expected classes
    expected_classes = original_yaml_content.get('names', {0: 'carpetweed', 1: 'morningglory', 2: 'palmer_amaranth'})
    
    print("\nExpected vs Actual classes:")
    print("-" * 40)
    for cls_id, expected_name in expected_classes.items():
        if cls_id in model.names:
            actual_name = model.names[cls_id]
            if actual_name.lower() == expected_name.lower():
                print(f"✓ Class {cls_id}: {actual_name}")
            else:
                print(f"⚠ Class {cls_id}: {actual_name} (expected {expected_name})")
        else:
            print(f"✗ Class {cls_id}: NOT FOUND (expected {expected_name})")
else:
    print("✗ Model doesn't have class names attribute")

# ============================================================================
# 6. PREDICTION FUNCTION
# ============================================================================

def predict_image(model, img_path, conf_thresh=0.25):
    """Predict on a single image"""
    try:
        results = model.predict(
            source=str(img_path),
            conf=conf_thresh,
            iou=0.5,
            imgsz=640,
            max_det=300,
            verbose=False,
            device=0 if torch.cuda.is_available() else 'cpu'
        )
        
        if results and len(results) > 0:
            return results[0]
        return None
    except Exception as e:
        print(f"Error predicting {img_path}: {e}")
        return None

# ============================================================================
# 7. GENERATE SUBMISSION
# ============================================================================

def generate_submission(model):
    """Generate submission.csv file"""
    print("\n" + "="*50)
    print("GENERATING SUBMISSION")
    print("="*50)
    
    # Test images path
    test_path = "/kaggle/input/the-3lc-cotton-weed-detection-challenge/cotton_weed_competition_dataset/test/images"
    test_images = list(Path(test_path).glob("*.jpg"))
    
    if not test_images:
        print("✗ No test images found!")
        return pd.DataFrame()
    
    print(f"Found {len(test_images)} test images")
    
    # Process images
    submission_data = []
    
    for img_path in tqdm(test_images, desc="Processing"):
        result = predict_image(model, img_path, conf_thresh=0.01)  # Low threshold to catch more predictions
        
        prediction_string = ""
        
        if result and hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            img_height, img_width = result.orig_shape
            
            box_lines = []
            for box in boxes:
                try:
                    # Get box data
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf)
                    cls_id = int(box.cls)
                    
                    # Convert to YOLO format
                    x_center = ((x1 + x2) / 2) / img_width
                    y_center = ((y1 + y2) / 2) / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    # Ensure valid values
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    width = max(0.0, min(1.0, width))
                    height = max(0.0, min(1.0, height))
                    
                    # Add to prediction string
                    box_line = f"{cls_id} {confidence:.6f} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    box_lines.append(box_line)
                    
                except:
                    continue
            
            # Sort by confidence
            box_lines.sort(key=lambda x: float(x.split()[1]), reverse=True)
            prediction_string = " ".join(box_lines)
        
        if not prediction_string:
            prediction_string = "no boxes"
        
        submission_data.append({
            "image_id": img_path.stem,
            "prediction_string": prediction_string
        })
    
    # Create DataFrame
    submission_df = pd.DataFrame(submission_data)
    
    # Save to CSV
    submission_csv = "/kaggle/working/submission.csv"
    submission_df.to_csv(submission_csv, index=False)
    
    print(f"\n✓ Submission saved to: {submission_csv}")
    print(f"  Total images: {len(submission_df)}")
    
    # Count predictions
    images_with_preds = (submission_df['prediction_string'] != "no boxes").sum()
    print(f"  Images with predictions: {images_with_preds}")
    
    return submission_df

# Generate submission
submission_df = generate_submission(model)

# ============================================================================
# 8. VALIDATE SUBMISSION
# ============================================================================

def validate_submission(submission_df):
    """Validate submission format and content"""
    print("\n" + "="*50)
    print("VALIDATING SUBMISSION")
    print("="*50)
    
    if submission_df.empty:
        print("✗ Empty submission")
        return False
    
    # Check required columns
    required_cols = ['image_id', 'prediction_string']
    if not all(col in submission_df.columns for col in required_cols):
        print(f"✗ Missing required columns. Found: {list(submission_df.columns)}")
        return False
    
    print("✓ Has required columns")
    
    # Check for invalid class IDs
    invalid_class_ids = []
    valid_class_ids = set([0, 1, 2])  # Should be 0, 1, 2 for 3-class problem
    
    for idx, row in submission_df.iterrows():
        if row['prediction_string'] != "no boxes":
            parts = row['prediction_string'].split()
            for i in range(0, len(parts), 6):
                try:
                    cls_id = int(parts[i])
                    if cls_id not in valid_class_ids:
                        invalid_class_ids.append((row['image_id'], cls_id))
                except:
                    continue
    
    if invalid_class_ids:
        print(f"✗ Found {len(invalid_class_ids)} invalid class IDs")
        print(f"  First few: {invalid_class_ids[:5]}")
        print("  Should only contain 0, 1, or 2")
        return False
    else:
        print("✓ All class IDs are valid (0, 1, 2)")
    
    # Check prediction format
    sample_predictions = []
    for idx, row in submission_df.head(3).iterrows():
        if row['prediction_string'] != "no boxes":
            parts = row['prediction_string'].split()
            if len(parts) % 6 != 0:
                print(f"✗ Invalid format for {row['image_id']}: {len(parts)} parts")
                return False
            sample_predictions.append((row['image_id'], len(parts)//6))
    
    print("✓ Prediction format is correct")
    
    # Show sample predictions
    print("\nSample predictions:")
    print("-" * 40)
    for img_id, num_preds in sample_predictions[:3]:
        print(f"  {img_id}: {num_preds} boxes")
    
    return True

# Validate
is_valid = validate_submission(submission_df)

# ============================================================================
# 9. QUICK VISUAL CHECK
# ============================================================================

def visualize_sample(model, num_samples=2):
    """Visualize a few predictions"""
    print("\n" + "="*50)
    print("VISUAL SAMPLE CHECK")
    print("="*50)
    
    test_path = "/kaggle/input/the-3lc-cotton-weed-detection-challenge/cotton_weed_competition_dataset/test/images"
    test_images = list(Path(test_path).glob("*.jpg"))[:num_samples]
    
    if not test_images:
        print("No test images for visualization")
        return
    
    fig, axes = plt.subplots(1, num_samples, figsize=(5*num_samples, 5))
    if num_samples == 1:
        axes = [axes]
    
    for idx, (ax, img_path) in enumerate(zip(axes, test_images)):
        try:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                ax.text(0.5, 0.5, "Failed to load", ha='center', va='center')
                ax.set_title(f"{img_path.stem}\nLoad failed")
                ax.axis('off')
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get prediction from submission
            img_pred = submission_df[submission_df['image_id'] == img_path.stem]
            
            if not img_pred.empty:
                pred_str = img_pred.iloc[0]['prediction_string']
                
                if pred_str != "no boxes":
                    parts = pred_str.split()
                    img_h, img_w = img.shape[:2]
                    
                    for i in range(0, min(10, len(parts)), 6):  # Show first 10 max
                        try:
                            cls_id = int(parts[i])
                            conf = float(parts[i+1])
                            x_center = float(parts[i+2])
                            y_center = float(parts[i+3])
                            width = float(parts[i+4])
                            height = float(parts[i+5])
                            
                            # Convert to pixel coordinates
                            x1 = int((x_center - width/2) * img_w)
                            y1 = int((y_center - height/2) * img_h)
                            x2 = int((x_center + width/2) * img_w)
                            y2 = int((y_center + height/2) * img_h)
                            
                            # Clamp
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(img_w, x2), min(img_h, y2)
                            
                            # Draw if confidence > 0.1
                            if conf > 0.1:
                                # Choose color based on class
                                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
                                color = colors[cls_id % len(colors)]
                                
                                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                                
                                # Label
                                cls_name = model.names.get(cls_id, f"C{cls_id}")
                                label = f"{cls_name}:{conf:.2f}"
                                cv2.putText(img, label, (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        except:
                            continue
                
                num_preds = 0 if pred_str == "no boxes" else len(parts)//6
                ax.set_title(f"{img_path.stem}\n{num_preds} preds")
            else:
                ax.set_title(f"{img_path.stem}\nNo preds found")
            
            ax.imshow(img)
            ax.axis('off')
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error", ha='center', va='center')
            ax.set_title(f"{img_path.stem}\nError")
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Visualize if validation passed
if is_valid and not submission_df.empty:
    visualize_sample(model, num_samples=2)

# ============================================================================
# 10. FINAL OUTPUT
# ============================================================================

print("\n" + "="*50)
print("FINAL OUTPUT SUMMARY")
print("="*50)

if not submission_df.empty:
    # Basic stats
    total_images = len(submission_df)
    images_with_preds = (submission_df['prediction_string'] != "no boxes").sum()
    
    print(f"\nSubmission Statistics:")
    print("-" * 40)
    print(f"Total images: {total_images}")
    print(f"Images with predictions: {images_with_preds}")
    print(f"Images without predictions: {total_images - images_with_preds}")
    
    # Count boxes
    total_boxes = 0
    for pred_str in submission_df['prediction_string']:
        if pred_str != "no boxes":
            total_boxes += len(pred_str.split()) // 6
    
    print(f"Total boxes predicted: {total_boxes}")
    
    if images_with_preds > 0:
        print(f"Average boxes per image: {total_boxes/images_with_preds:.2f}")
    
    # Show format
    print(f"\nSubmission format (first 3 rows):")
    print("-" * 40)
    print(submission_df.head(3).to_string())
    
    # Save summary
    with open("/kaggle/working/submission_summary.txt", "w") as f:
        f.write("COTTON WEED DETECTION SUBMISSION SUMMARY\n")
        f.write("="*50 + "\n")
        f.write(f"Total images: {total_images}\n")
        f.write(f"Images with predictions: {images_with_preds}\n")
        f.write(f"Total boxes: {total_boxes}\n")
        f.write(f"Validation passed: {is_valid}\n")
    
    print(f"\n✓ Summary saved to: /kaggle/working/submission_summary.txt")
    
    if is_valid:
        print("\n" + "="*50)
        print("✅ SUBMISSION READY FOR KAGGLE!")
        print("="*50)
        print("\nFiles generated:")
        print("1. submission.csv - Main submission file")
        print("2. submission_summary.txt - Summary statistics")
        print("\nSubmit 'submission.csv' to the competition!")
    else:
        print("\n" + "="*50)
        print("❌ SUBMISSION HAS ISSUES")
        print("="*50)
        print("\nFix the validation errors before submitting!")
else:
    print("✗ No submission generated!")

# ============================================================================
# 11. CLEANUP
# ============================================================================

print("\n" + "="*50)
print("CLEANUP")
print("="*50)

# Free memory
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")

print("\n" + "="*60)
print("NOTEBOOK COMPLETED")
print("="*60)

