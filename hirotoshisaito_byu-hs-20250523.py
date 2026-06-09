## YOROãƒ‡ãƒ¼ã‚¿å�‘ã�‘å‰�å‡¦ç�†

import os
import numpy as np
import pandas as pd
from PIL import Image
import shutil
import time
import yaml
from pathlib import Path
from tqdm.notebook import tqdm  # Jupyter/Kaggleç’°å¢ƒç”¨ã�®é€²æ�—ãƒ�ãƒ¼

# å†�ç�¾æ€§ã�®ã�Ÿã‚�ã�«ä¹±æ•°ã‚·ãƒ¼ãƒ‰ã‚’å›ºå®š
np.random.seed(42)

# Kaggleãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆã�®ãƒ‘ã‚¹ã‚’å®šç¾©
data_path = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/"
train_dir = os.path.join(data_path, "train")

# YOLOå½¢å¼�ã�®ãƒ‡ãƒ¼ã‚¿æ§‹é€ ã‚’å®šç¾©
yolo_dataset_dir = "/kaggle/working/yolo_dataset"
yolo_images_train = os.path.join(yolo_dataset_dir, "images", "train")
yolo_images_val = os.path.join(yolo_dataset_dir, "images", "val")
yolo_labels_train = os.path.join(yolo_dataset_dir, "labels", "train")
yolo_labels_val = os.path.join(yolo_dataset_dir, "labels", "val")

# å¿…è¦�ã�ªãƒ‡ã‚£ãƒ¬ã‚¯ãƒˆãƒªã‚’ä½œæˆ�
for dir_path in [yolo_images_train, yolo_images_val, yolo_labels_train, yolo_labels_val]:
    os.makedirs(dir_path, exist_ok=True)

# å®šæ•°ã‚’å®šç¾©
TRUST = 4  # ä¸­å¿ƒã‚¹ãƒ©ã‚¤ã‚¹ã�®å‰�å¾Œã�«å�«ã‚�ã‚‹ã‚¹ãƒ©ã‚¤ã‚¹æ•°ï¼ˆå�ˆè¨ˆ2*TRUST+1æ�šï¼‰
BOX_SIZE = 24  # ã‚¢ãƒ�ãƒ†ãƒ¼ã‚·ãƒ§ãƒ³ç”¨ãƒ�ã‚¦ãƒ³ãƒ‡ã‚£ãƒ³ã‚°ãƒœãƒƒã‚¯ã‚¹ã�®ã‚µã‚¤ã‚ºï¼ˆãƒ”ã‚¯ã‚»ãƒ«å�˜ä½�ï¼‰
TRAIN_SPLIT = 0.8  # å­¦ç¿’ãƒ‡ãƒ¼ã‚¿ã�¨æ¤œè¨¼ãƒ‡ãƒ¼ã‚¿ã�®åˆ†å‰²æ¯”ï¼ˆ80%:20%ï¼‰

# ã‚¹ãƒ©ã‚¤ã‚¹ç”»åƒ�ã�®æ­£è¦�åŒ–é–¢æ•°
def normalize_slice(slice_data):
    """
    ã‚¹ãƒ©ã‚¤ã‚¹ç”»åƒ�ã‚’2ãƒ‘ãƒ¼ã‚»ãƒ³ã‚¿ã‚¤ãƒ«ã�¨98ãƒ‘ãƒ¼ã‚»ãƒ³ã‚¿ã‚¤ãƒ«ã�§ã‚¯ãƒªãƒƒãƒ—ã�—ã�¦æ­£è¦�åŒ–
    """
    p2 = np.percentile(slice_data, 2)
    p98 = np.percentile(slice_data, 98)
    clipped_data = np.clip(slice_data, p2, p98)
    normalized = 255 * (clipped_data - p2) / (p98 - p2)
    return np.uint8(normalized)

# YOLOç”¨ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆã‚’æº–å‚™ã�™ã‚‹é–¢æ•°
def prepare_yolo_dataset(trust=TRUST, train_split=TRAIN_SPLIT):
    """
    é�­æ¯›ãƒ¢ãƒ¼ã‚¿ãƒ¼ã‚’å�«ã‚€ã‚¹ãƒ©ã‚¤ã‚¹ã‚’æŠ½å‡ºã�—ã�¦ã€�YOLOç”¨ç”»åƒ�ã�¨ãƒ©ãƒ™ãƒ«ã‚’ä¿�å­˜
    """
    # ã‚¢ãƒ�ãƒ†ãƒ¼ã‚·ãƒ§ãƒ³CSVã‚’èª­ã�¿è¾¼ã�¿
    labels_df = pd.read_csv(os.path.join(data_path, "train_labels.csv"))
    
    # ãƒ¢ãƒ¼ã‚¿ãƒ¼ã�®ç·�æ•°ã‚’è¡¨ç¤º
    total_motors = labels_df['Number of motors'].sum()
    print(f"ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆå†…ã�®ãƒ¢ãƒ¼ã‚¿ãƒ¼ç·�æ•°: {total_motors}")
    
    # ãƒ¢ãƒ¼ã‚¿ãƒ¼ã‚’å�«ã‚€ãƒ¦ãƒ‹ãƒ¼ã‚¯ã�ªãƒˆãƒ¢ã‚°ãƒ©ãƒ ã‚’å�–å¾—
    tomo_df = labels_df[labels_df['Number of motors'] > 0].copy()
    unique_tomos = tomo_df['tomo_id'].unique()
    print(f"ãƒ¢ãƒ¼ã‚¿ãƒ¼ã‚’å�«ã‚€ãƒ¦ãƒ‹ãƒ¼ã‚¯ã�ªãƒˆãƒ¢ã‚°ãƒ©ãƒ æ•°: {len(unique_tomos)}")
    
    # ãƒˆãƒ¢ã‚°ãƒ©ãƒ å�˜ä½�ã�§å­¦ç¿’/æ¤œè¨¼ç”¨ã�«åˆ†å‰²ï¼ˆå�Œã�˜ãƒˆãƒ¢ã‚°ãƒ©ãƒ ã�¯ç‰‡æ–¹ã�«ã�—ã�‹å±�ã�•ã�ªã�„ï¼‰
    np.random.shuffle(unique_tomos)
    split_idx = int(len(unique_tomos) * train_split)
    train_tomos = unique_tomos[:split_idx]
    val_tomos = unique_tomos[split_idx:]
    print(f"åˆ†å‰²çµ�æ�œ: å­¦ç¿’ç”¨ {len(train_tomos)}å€‹, æ¤œè¨¼ç”¨ {len(val_tomos)}å€‹")

    # ãƒˆãƒ¢ã‚°ãƒ©ãƒ ã‚’å‡¦ç�†ã�™ã‚‹é–¢æ•°
    def process_tomogram_set(tomogram_ids, images_dir, labels_dir, set_name):
        motor_counts = []
        for tomo_id in tomogram_ids:
            tomo_motors = labels_df[labels_df['tomo_id'] == tomo_id]
            for _, motor in tomo_motors.iterrows():
                if pd.isna(motor['Motor axis 0']):
                    continue
                motor_counts.append(
                    (tomo_id, 
                     int(motor['Motor axis 0']), 
                     int(motor['Motor axis 1']), 
                     int(motor['Motor axis 2']),
                     int(motor['Array shape (axis 0)']))
                )
        
        print(f"{set_name}ã‚»ãƒƒãƒˆã�§å‡¦ç�†ã�™ã‚‹ã‚¹ãƒ©ã‚¤ã‚¹ã�®æ¦‚ç®—æ•°: {len(motor_counts) * (2 * trust + 1)}")
        
        processed_slices = 0
        
        # ãƒ¢ãƒ¼ã‚¿ãƒ¼ã�”ã�¨ã�«å‡¦ç�†
        for tomo_id, z_center, y_center, x_center, z_max in tqdm(motor_counts, desc=f"{set_name}ãƒ¢ãƒ¼ã‚¿ãƒ¼å‡¦ç�†ä¸­"):
            z_min = max(0, z_center - trust)
            z_max = min(z_max - 1, z_center + trust)
            
            for z in range(z_min, z_max + 1):
                slice_filename = f"slice_{z:04d}.jpg"
                src_path = os.path.join(train_dir, tomo_id, slice_filename)
                
                if not os.path.exists(src_path):
                    print(f"è­¦å‘Š: {src_path} ã�Œå­˜åœ¨ã�—ã�¾ã�›ã‚“ã€‚ã‚¹ã‚­ãƒƒãƒ—ã�—ã�¾ã�™ã€‚")
                    continue
                
                img = Image.open(src_path)
                img_array = np.array(img)
                normalized_img = normalize_slice(img_array)
                
                # å‡ºåŠ›ç”¨ãƒ•ã‚¡ã‚¤ãƒ«å��ã‚’ãƒ¦ãƒ‹ãƒ¼ã‚¯ã�«ä½œæˆ�
                dest_filename = f"{tomo_id}_z{z:04d}_y{y_center:04d}_x{x_center:04d}.jpg"
                dest_path = os.path.join(images_dir, dest_filename)
                
                # ç”»åƒ�ã‚’ä¿�å­˜
                Image.fromarray(normalized_img).save(dest_path)
                
                # ç”»åƒ�ã‚µã‚¤ã‚ºã‚’å�–å¾—
                img_width, img_height = img.size
                
                # YOLOå½¢å¼�ã�®ãƒ©ãƒ™ãƒ«ã‚’ä½œæˆ�ï¼ˆæ­£è¦�åŒ–ã�•ã‚Œã�Ÿä¸­å¿ƒåº§æ¨™ã�¨ã‚µã‚¤ã‚ºï¼‰
                x_center_norm = x_center / img_width
                y_center_norm = y_center / img_height
                box_width_norm = BOX_SIZE / img_width
                box_height_norm = BOX_SIZE / img_height
                
                label_path = os.path.join(labels_dir, dest_filename.replace('.jpg', '.txt'))
                with open(label_path, 'w') as f:
                    f.write(f"0 {x_center_norm} {y_center_norm} {box_width_norm} {box_height_norm}\n")
                
                processed_slices += 1
        
        return processed_slices, len(motor_counts)
    
    # å­¦ç¿’ãƒ‡ãƒ¼ã‚¿å‡¦ç�†
    train_slices, train_motors = process_tomogram_set(train_tomos, yolo_images_train, yolo_labels_train, "å­¦ç¿’")
    
    # æ¤œè¨¼ãƒ‡ãƒ¼ã‚¿å‡¦ç�†
    val_slices, val_motors = process_tomogram_set(val_tomos, yolo_images_val, yolo_labels_val, "æ¤œè¨¼")
    
    # YOLOã�®å­¦ç¿’è¨­å®šç”¨ã�®YAMLãƒ•ã‚¡ã‚¤ãƒ«ã‚’ä½œæˆ�
    yaml_content = {
        'path': yolo_dataset_dir,
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'motor'}
    }
    
    with open(os.path.join(yolo_dataset_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    # å‡¦ç�†ã�®ã‚µãƒ�ãƒªã‚’è¡¨ç¤º
    print(f"\nå‡¦ç�†çµ�æ�œ:")
    print(f"- å­¦ç¿’ãƒ‡ãƒ¼ã‚¿: {len(train_tomos)} ãƒˆãƒ¢ã‚°ãƒ©ãƒ , {train_motors} ãƒ¢ãƒ¼ã‚¿ãƒ¼, {train_slices} ã‚¹ãƒ©ã‚¤ã‚¹")
    print(f"- æ¤œè¨¼ãƒ‡ãƒ¼ã‚¿: {len(val_tomos)} ãƒˆãƒ¢ã‚°ãƒ©ãƒ , {val_motors} ãƒ¢ãƒ¼ã‚¿ãƒ¼, {val_slices} ã‚¹ãƒ©ã‚¤ã‚¹")
    print(f"- å�ˆè¨ˆ: {len(train_tomos) + len(val_tomos)} ãƒˆãƒ¢ã‚°ãƒ©ãƒ , {train_motors + val_motors} ãƒ¢ãƒ¼ã‚¿ãƒ¼, {train_slices + val_slices} ã‚¹ãƒ©ã‚¤ã‚¹")
    
    return {
        "dataset_dir": yolo_dataset_dir,
        "yaml_path": os.path.join(yolo_dataset_dir, 'dataset.yaml'),
        "train_tomograms": len(train_tomos),
        "val_tomograms": len(val_tomos),
        "train_motors": train_motors,
        "val_motors": val_motors,
        "train_slices": train_slices,
        "val_slices": val_slices
    }

# å‰�å‡¦ç�†ã‚’å®Ÿè¡Œ
summary = prepare_yolo_dataset(TRUST)
print(f"\nå‰�å‡¦ç�†å®Œäº†:")
print(f"- å­¦ç¿’ãƒ‡ãƒ¼ã‚¿: {summary['train_tomograms']} ãƒˆãƒ¢ã‚°ãƒ©ãƒ , {summary['train_motors']} ãƒ¢ãƒ¼ã‚¿ãƒ¼, {summary['train_slices']} ã‚¹ãƒ©ã‚¤ã‚¹")
print(f"- æ¤œè¨¼ãƒ‡ãƒ¼ã‚¿: {summary['val_tomograms']} ãƒˆãƒ¢ã‚°ãƒ©ãƒ , {summary['val_motors']} ãƒ¢ãƒ¼ã‚¿ãƒ¼, {summary['val_slices']} ã‚¹ãƒ©ã‚¤ã‚¹")
print(f"- ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆä¿�å­˜å…ˆ: {summary['dataset_dir']}")
print(f"- YOLOè¨­å®šãƒ•ã‚¡ã‚¤ãƒ«: {summary['yaml_path']}")
print("\nYOLOå­¦ç¿’ã�®æº–å‚™ã�Œæ•´ã�„ã�¾ã�—ã�Ÿï¼�")



# YOLOv8ã�®ã‚¤ãƒ³ã‚¹ãƒˆãƒ¼ãƒ«
!pip install -q ultralytics

# ãƒ©ã‚¤ãƒ–ãƒ©ãƒªã�®ã‚¤ãƒ³ãƒ�ãƒ¼ãƒˆ
import os
import torch
import numpy as np
import random
from ultralytics import YOLO

# å†�ç�¾æ€§ã�®ã�Ÿã‚�ã�®ã‚·ãƒ¼ãƒ‰è¨­å®š
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# ãƒ‘ã‚¹ã�®å®šç¾©
dataset_path = "/kaggle/working/yolo_dataset"
yaml_path = os.path.join(dataset_path, "dataset.yaml")
output_dir = "/kaggle/working/motor_yolo_train"

# ãƒ¢ãƒ‡ãƒ«ã�®å®šç¾©
model = YOLO("yolov8n.pt")

# å­¦ç¿’ã�®å®Ÿè¡Œï¼ˆæ˜�ç¤ºçš„ã�«ãƒ‡ã‚£ãƒ¬ã‚¯ãƒˆãƒªã‚’æŒ‡å®šï¼‰
model.train(
    data=yaml_path,
    epochs=5,
    imgsz=256,
    batch=8,
    project="/kaggle/working",  # ä¸Šä½�ãƒ‡ã‚£ãƒ¬ã‚¯ãƒˆãƒª
    name="motor_yolo_train",    # ã‚µãƒ–ãƒ‡ã‚£ãƒ¬ã‚¯ãƒˆãƒª
    exist_ok=True               # æ—¢å­˜ãƒ•ã‚©ãƒ«ãƒ€ã�Œã�‚ã�£ã�¦ã‚‚OK
)

# ãƒ¢ãƒ‡ãƒ«ä¿�å­˜ãƒ‘ã‚¹ã�®ç¢ºèª�
trained_model_path = os.path.join(output_dir, "weights", "best.pt")

if os.path.exists(trained_model_path):
    print(f"âœ… ãƒ¢ãƒ‡ãƒ«ä¿�å­˜æˆ�åŠŸ: {trained_model_path}")
else:
    # å®Ÿéš›ã�«å­˜åœ¨ã�™ã‚‹ãƒ‘ã‚¹ã‚’åˆ—æŒ™ã�—ã�¦ãƒ’ãƒ³ãƒˆã‚’å‡ºã�™
    print("â�Œ ãƒ¢ãƒ‡ãƒ«ã�Œä¿�å­˜ã�•ã‚Œã�¦ã�„ã�¾ã�›ã‚“ã€‚ä¿�å­˜å…ˆã‚’æ�¢ç´¢ã�—ã�¾ã�™...")
    for root, dirs, files in os.walk("/kaggle/working"):
        for file in files:
            if file == "best.pt":
                print(f"ğŸ”� ç™ºè¦‹: {os.path.join(root, file)}")



import os

weights_dir = "/kaggle/working/motor_yolo_train/weights"
if os.path.exists(weights_dir):
    print("ğŸ“‚ weightsãƒ•ã‚©ãƒ«ãƒ€ã�®å†…å®¹ï¼š", os.listdir(weights_dir))
else:
    print("â�Œ weightsãƒ•ã‚©ãƒ«ãƒ€ã�Œå­˜åœ¨ã�—ã�¾ã�›ã‚“ã€‚")


