!pip install ultralytics "numpy<2.0"


import pandas as pd
import os
import shutil
import cv2
import torch
from sklearn.model_selection import KFold
from ultralytics import YOLO

# --- CONFIG ---
DATASET_DIR = '/kaggle/working/yolo_dataset_5fold' 
BBOX_CSV = '/kaggle/input/solidworks-ai-hackathon/train_bboxes.csv'
TRAIN_IMG_DIR = '/kaggle/input/solidworks-ai-hackathon/train/train'

# Class mapping
class_map = {'bolt': 0, 'locatingpin': 1, 'nut': 2, 'washer': 3}

# --- SETUP ---
# Clean up previous runs to avoid mixing data
if os.path.exists(DATASET_DIR):
    shutil.rmtree(DATASET_DIR)
os.makedirs(DATASET_DIR, exist_ok=True)

# Load Data
df = pd.read_csv(BBOX_CSV)
print("Columns found:", df.columns.tolist())
unique_images = df['image_name'].unique()

# K-Fold Definition
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Helper: Normalize BBox for YOLO
def normalize_bbox(row, img_width, img_height):
    dw = 1. / img_width
    dh = 1. / img_height
    x_center = (row['x_min'] + row['x_max']) / 2.0
    y_center = (row['y_min'] + row['y_max']) / 2.0
    w = row['x_max'] - row['x_min']
    h = row['y_max'] - row['y_min']
    return x_center * dw, y_center * dh, w * dw, h * dh

#--GPU CHECK--
devices = 'cpu'
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f"GPU Count: {gpu_count}")
    #use all available GPUs
    devices = list(range(gpu_count)) 
else:
    print("No GPU found. Training will be slow.")

# --- MAIN LOOP ---
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(unique_images)):
    print(f"\n--- Starting Fold {fold_idx+1}/5 ---")
    
    # 1. Setup Directories for this Fold
    fold_dir = os.path.join(DATASET_DIR, f'fold_{fold_idx}')
    for split in ['train', 'val']:
        os.makedirs(os.path.join(fold_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, 'labels', split), exist_ok=True)
    
    # 2. Distribute Images & Create Labels
    train_imgs = unique_images[train_idx]
    val_imgs = unique_images[val_idx]
    
    for split, img_list in [('train', train_imgs), ('val', val_imgs)]:
        for img_name in img_list:
            src_img_path = os.path.join(TRAIN_IMG_DIR, img_name)
            dst_img_path = os.path.join(fold_dir, 'images', split, img_name)
            dst_label_path = os.path.join(fold_dir, 'labels', split, img_name.replace('.png', '.txt').replace('.jpg', '.txt'))
            
            # Copy Image
            if os.path.exists(src_img_path):
                shutil.copy(src_img_path, dst_img_path)
                
                # Read Image Dims
                img = cv2.imread(src_img_path)
                if img is None: continue
                h_img, w_img, _ = img.shape
                
                # Create Label File
                img_df = df[df['image_name'] == img_name]
                with open(dst_label_path, 'w') as f:
                    for _, row in img_df.iterrows():
                        cls_id = class_map[row['class']] 
                        xc, yc, w, h = normalize_bbox(row, w_img, h_img)
                        f.write(f"{cls_id} {xc} {yc} {w} {h}\n")
    
    # 3. Create YAML for this fold
    yaml_path = f'{fold_dir}/data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(f"""
path: {os.path.abspath(fold_dir)}
train: images/train
val: images/val
nc: 4
names: ['bolt', 'locatingpin', 'nut', 'washer']
""")
    
    # 4. Train Model (GPU Enabled)
    model = YOLO('yolov8n.pt') 
    model.train(
        data=yaml_path, 
        epochs=15, 
        imgsz=640, 
        batch=32,       # Higher batch size for GPU
        project='solidworks_hackathon',
        name=f'fold_{fold_idx}',
        device=devices, # Uses [0, 1] if available
        verbose=False
    )


import pandas as pd
import os
import glob
from ultralytics import YOLO
import torch

# --- CONFIG ---
# This matches the 'project' and 'name' from your training script
MODELS_DIR = 'solidworks_hackathon' 
# Adjust this path if your test images are elsewhere
TEST_IMG_DIR = '/kaggle/input/solidworks-ai-hackathon/test/test' 

# The order MUST match your class_map {'bolt': 0, 'locatingpin': 1, ...}
CLASSES = ['bolt', 'locatingpin', 'nut', 'washer']

def main():
    # 1. Find the 5 trained models
    model_paths = []
    for fold in range(5):
        # Your code saves to: solidworks_hackathon/fold_{fold_idx}/weights/best.pt
        path = f"{MODELS_DIR}/fold_{fold}/weights/best.pt"
        if os.path.exists(path):
            model_paths.append(path)
            print(f"âœ… Found model: {path}")
        else:
            print(f"âš ï¸� Missing model for fold {fold} (Training might have failed or is running)")

    if not model_paths:
        print("â�Œ No models found! Wait for training to finish.")
        return

    # 2. Load Models
    print(f"Loading {len(model_paths)} models for Ensemble...")
    models = [YOLO(p) for p in model_paths]

    # 3. Find Test Images
    # Using glob to find .png or .jpg inside the test folder
    test_images = glob.glob(os.path.join(TEST_IMG_DIR, '*.png')) + \
                  glob.glob(os.path.join(TEST_IMG_DIR, '*.jpg'))
    
    if len(test_images) == 0:
        print(f"â�Œ No images found in {TEST_IMG_DIR}. Check the path!")
        return
        
    print(f"Predicting on {len(test_images)} images...")

    results_list = []

    # 4. Inference Loop
    for img_path in test_images:
        filename = os.path.basename(img_path)
        
        all_counts = []

        # Run each model on the image
        for model in models:
            # augment=True enables TTA (Test Time Augmentation - flips/scales image for better accuracy)
            # conf=0.20 is a safe threshold. If you miss parts, lower it to 0.15.
            results = model.predict(img_path, augment=True, conf=0.20, verbose=False)
            
            # Count parts for this model
            counts = {cls: 0 for cls in CLASSES}
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    counts[CLASSES[cls_id]] += 1
            all_counts.append(counts)
        
        # 5. Ensemble Strategy: Majority Vote / Average
        # For object counting, taking the AVERAGE count across models is usually safest
        # followed by rounding to the nearest integer.
        
        final_counts = {}
        for cls in CLASSES:
            # Get count for this part from all 5 models
            cls_vals = [c[cls] for c in all_counts] 
            
            # Technique: Average and Round
            avg_val = sum(cls_vals) / len(cls_vals)
            final_counts[cls] = int(round(avg_val))
            
            # Alternative: Max Voting (Use this if you think models miss parts often)
            # final_counts[cls] = max(set(cls_vals), key=cls_vals.count)

        # 6. Add to list
        row = {'image_name': filename}
        row.update(final_counts)
        results_list.append(row)

    # 5. Save Submission
    df = pd.DataFrame(results_list)
    # Reorder columns strictly
    df = df[['image_name', 'bolt', 'locatingpin', 'nut', 'washer']]
    df.to_csv('submission.csv', index=False)
    print("\nğŸ�‰ DONE! Saved 'submission.csv'. Download it and submit!")

if __name__ == "__main__":
    main()










