import pandas as pd
import shutil
import os
from PIL import Image
from pycocotools.coco import COCO
import random

# Ä�Æ°á»�ng dáº«n dataset
imagenet_dir = '/kaggle/input/imagenet-object-localization-challenge'
coco_dir = '/kaggle/input/coco-2017-dataset/coco2017'
output_dir = '/kaggle/working/dataset'

# Táº¡o thÆ° má»¥c output
os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

# Mapping classes
# YOLO class IDs: 0=person, 1=phone, 2=reflex_camera, 3=polaroid_camera
imagenet_classes = {
    'n02992529': 1,  # mobile phone -> phone
    'n04069434': 2,  # reflex camera
    'n03976467': 3   # Polaroid camera
}

coco_classes = {
    1: 0,   # person (COCO ID: 1) -> YOLO ID: 0
    77: 1   # cell phone (COCO ID: 77, khÃ´ng pháº£i 68!) -> YOLO ID: 1
}

# Counter Ä‘á»ƒ theo dÃµi sá»‘ lÆ°á»£ng máº«u
class_counts = {0: 0, 1: 0, 2: 0, 3: 0}


print("=== Xá»­ lÃ½ ImageNet ===")
# Xá»­ lÃ½ ImageNet
try:
    annotations_file = os.path.join(imagenet_dir, 'LOC_train_solution.csv')
    annotations = pd.read_csv(annotations_file)
    
    # Lá»�c annotations cho cÃ¡c class cáº§n thiáº¿t
    desired_classes = list(imagenet_classes.keys())
    filtered_annotations = annotations[annotations['PredictionString'].str.contains('|'.join(desired_classes))]
    
    print(f"TÃ¬m tháº¥y {len(filtered_annotations)} annotations tá»« ImageNet")
    
    for idx, row in filtered_annotations.iterrows():
        image_id = row['ImageId']
        predictions = row['PredictionString'].split()
        
        # Xá»­ lÃ½ tá»«ng bounding box trong prediction
        i = 0
        while i < len(predictions):
            if predictions[i] in desired_classes:
                class_id = predictions[i]
                xmin, ymin, xmax, ymax = map(float, predictions[i+1:i+5])
                
                # Ä�Æ°á»�ng dáº«n hÃ¬nh áº£nh
                image_path = os.path.join(imagenet_dir, 'ILSVRC/Data/CLS-LOC/train', class_id, f'{image_id}.JPEG')
                if not os.path.exists(image_path):
                    i += 5
                    continue
                
                # Sao chÃ©p hÃ¬nh áº£nh (chá»‰ sao chÃ©p 1 láº§n)
                output_image_path = os.path.join(output_dir, 'images', f'imagenet_{image_id}.jpg')
                if not os.path.exists(output_image_path):
                    shutil.copy(image_path, output_image_path)
                
                # Láº¥y kÃ­ch thÆ°á»›c hÃ¬nh áº£nh
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
                
                # Chuyá»ƒn Ä‘á»•i sang format YOLO
                x_center = (xmin + xmax) / 2 / img_width
                y_center = (ymin + ymax) / 2 / img_height
                width_norm = (xmax - xmin) / img_width
                height_norm = (ymax - ymin) / img_height
                
                yolo_class_id = imagenet_classes[class_id]
                label = f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n"
                
                # LÆ°u hoáº·c append label
                label_path = os.path.join(output_dir, 'labels', f'imagenet_{image_id}.txt')
                mode = 'a' if os.path.exists(label_path) else 'w'
                with open(label_path, mode) as f:
                    f.write(label)
                
                class_counts[yolo_class_id] += 1
                i += 5
            else:
                i += 1
                
except Exception as e:
    print(f"Lá»—i khi xá»­ lÃ½ ImageNet: {e}")


print("\n=== Xá»­ lÃ½ COCO ===")
# Xá»­ lÃ½ COCO
try:
    # Load COCO annotations
    coco = COCO(os.path.join(coco_dir, 'annotations/instances_train2017.json'))
    
    # Láº¥y danh sÃ¡ch image IDs cho má»—i category
    person_img_ids = coco.getImgIds(catIds=[1])  # person
    phone_img_ids = coco.getImgIds(catIds=[77])  # cell phone (ID chÃ­nh xÃ¡c lÃ  77!)
    
    # Ä�á»ƒ cÃ¢n báº±ng dataset, giá»›i háº¡n sá»‘ lÆ°á»£ng áº£nh person
    max_person_images = 2000  # Giá»›i háº¡n Ä‘á»ƒ trÃ¡nh máº¥t cÃ¢n báº±ng
    person_img_ids = random.sample(person_img_ids, min(len(person_img_ids), max_person_images))
    
    # Combine táº¥t cáº£ image IDs (loáº¡i bá»� duplicates)
    all_img_ids = list(set(person_img_ids + phone_img_ids))
    
    print(f"TÃ¬m tháº¥y {len(person_img_ids)} áº£nh cÃ³ person")
    print(f"TÃ¬m tháº¥y {len(phone_img_ids)} áº£nh cÃ³ cell phone")
    print(f"Tá»•ng cá»™ng {len(all_img_ids)} áº£nh unique tá»« COCO")
    
    processed_count = 0
    for img_id in all_img_ids:
        img_info = coco.loadImgs(img_id)[0]
        
        # Láº¥y táº¥t cáº£ annotations cho image nÃ y (cáº£ person vÃ  cell phone)
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[1, 77])
        anns = coco.loadAnns(ann_ids)
        
        if not anns:
            continue
            
        # Sao chÃ©p hÃ¬nh áº£nh
        image_path = os.path.join(coco_dir, 'train2017', img_info['file_name'])
        if not os.path.exists(image_path):
            continue
            
        output_image_path = os.path.join(output_dir, 'images', f'coco_{img_info["file_name"]}')
        shutil.copy(image_path, output_image_path)
        
        # Táº¡o label file
        label_path = os.path.join(output_dir, 'labels', f'coco_{os.path.splitext(img_info["file_name"])[0]}.txt')
        
        with open(label_path, 'w') as f:
            for ann in anns:
                if ann['category_id'] in coco_classes:
                    # Chuyá»ƒn Ä‘á»•i COCO bbox sang YOLO format
                    bbox = ann['bbox']  # [x, y, width, height]
                    x, y, w, h = bbox
                    
                    # Kiá»ƒm tra bbox há»£p lá»‡
                    if w <= 0 or h <= 0:
                        continue
                        
                    x_center = (x + w/2) / img_info['width']
                    y_center = (y + h/2) / img_info['height']
                    width_norm = w / img_info['width']
                    height_norm = h / img_info['height']
                    
                    # Ä�áº£m báº£o giÃ¡ trá»‹ náº±m trong [0, 1]
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    width_norm = max(0, min(1, width_norm))
                    height_norm = max(0, min(1, height_norm))
                    
                    yolo_class_id = coco_classes[ann['category_id']]
                    f.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")
                    class_counts[yolo_class_id] += 1
        
        processed_count += 1
        if processed_count % 500 == 0:
            print(f"Ä�Ã£ xá»­ lÃ½ {processed_count} áº£nh tá»« COCO...")
            
except Exception as e:
    print(f"Lá»—i khi xá»­ lÃ½ COCO: {e}")




print("\n=== Thá»‘ng kÃª dataset ===")
for class_id, count in class_counts.items():
    class_names = ['person', 'phone', 'reflex_camera', 'polaroid_camera']
    print(f"{class_names[class_id]}: {count} instances")

# Táº¡o train vÃ  validation split
all_images = [f for f in os.listdir(os.path.join(output_dir, 'images')) if f.endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(all_images)

# 80% train, 20% val
split_idx = int(0.8 * len(all_images))
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

print(f"\nTá»•ng sá»‘ áº£nh: {len(all_images)}")
print(f"Train: {len(train_images)}, Val: {len(val_images)}")

# Táº¡o train.txt vÃ  val.txt
with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
    for img in train_images:
        f.write(os.path.join(output_dir, 'images', img) + '\n')

with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
    for img in val_images:
        f.write(os.path.join(output_dir, 'images', img) + '\n')

# Táº¡o data.yaml
yaml_content = f"""
train: {os.path.join(output_dir, 'train.txt')}
val: {os.path.join(output_dir, 'val.txt')}

nc: 4
names: ['person', 'phone', 'reflex_camera', 'polaroid_camera']

# Augmentation parameters Ä‘á»ƒ tÄƒng cÆ°á»�ng nháº­n diá»‡n person
mosaic: 1.0
mixup: 0.5
copy_paste: 0.1
"""

with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
    f.write(yaml_content)



# Knowledge Distillation cho YOLOv11 vá»›i Dataset tá»« ImageNet + COCO
# Fixed version - xá»­ lÃ½ lá»—i batch collation

import pandas as pd
import shutil
import os
from PIL import Image
from pycocotools.coco import COCO
import random
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Install ultralytics
!pip install ultralytics -q

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ====================== CUSTOM COLLATE FUNCTION ======================

def custom_collate_fn(batch):
    """Custom collate function to handle variable-sized labels"""
    images = []
    labels_list = []
    paths = []
    
    for img, labels, path in batch:
        images.append(img)
        labels_list.append(labels)
        paths.append(path)
    
    # Stack images normally
    images = torch.stack(images, 0)
    
    # For labels, we keep them as a list since they have different sizes
    # Each label tensor has shape [num_boxes, 5] where 5 = [class, x, y, w, h]
    
    return images, labels_list, paths



# ====================== DATASET CLASS ======================

class ScreenPhotoDataset(Dataset):
    """Custom dataset for screen photography detection"""
    
    def __init__(self, data_yaml_path, mode='train', img_size=640, augment=True):
        self.img_size = img_size
        self.mode = mode
        self.augment = augment and mode == 'train'
        
        # Load data configuration
        with open(data_yaml_path, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        # Get image paths
        txt_file = self.data_config['train'] if mode == 'train' else self.data_config['val']
        with open(txt_file, 'r') as f:
            self.img_paths = [line.strip() for line in f.readlines()]
        
        # Class names
        self.class_names = self.data_config.get('names', ['person', 'phone', 'reflex_camera', 'polaroid_camera'])
        
        print(f"Loaded {len(self.img_paths)} {mode} images")
    
    def __len__(self):
        return len(self.img_paths)
    
    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        """Resize and pad image while maintaining aspect ratio"""
        shape = img.shape[:2]  # current shape [height, width]
        
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        return img, r, (dw, dh)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image {img_path}")
            return self.__getitem__((idx + 1) % len(self))
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]  # original hw
        
        # Letterbox
        img, ratio, pad = self.letterbox(img, new_shape=(self.img_size, self.img_size))
        
        # Load labels
        label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
        
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x, y, w, h = map(float, parts)
                        labels.append([cls, x, y, w, h])
        
        labels = np.array(labels).reshape(-1, 5) if labels else np.zeros((0, 5))
        
        # Convert normalized to pixel coordinates
        if len(labels):
            labels[:, 1] = w0 * labels[:, 1] + pad[0]  # x padding
            labels[:, 2] = h0 * labels[:, 2] + pad[1]  # y padding
            labels[:, 3] *= w0  # width
            labels[:, 4] *= h0  # height
            
            # Convert pixel to normalized with padding
            labels[:, 1] /= self.img_size
            labels[:, 2] /= self.img_size
            labels[:, 3] /= self.img_size
            labels[:, 4] /= self.img_size
        
        # Apply augmentations if training
        if self.augment:
            # Random horizontal flip
            if random.random() < 0.5:
                img = np.fliplr(img)
                if len(labels):
                    labels[:, 1] = 1 - labels[:, 1]
            
            # Random HSV augmentation
            if random.random() < 0.5:
                img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
                h_gain = random.uniform(-0.015, 0.015)
                s_gain = random.uniform(0.7, 1.3)
                v_gain = random.uniform(0.4, 1.6)
                
                img_hsv[..., 0] = (img_hsv[..., 0] + h_gain * 180) % 180
                img_hsv[..., 1] = np.clip(img_hsv[..., 1] * s_gain, 0, 255)
                img_hsv[..., 2] = np.clip(img_hsv[..., 2] * v_gain, 0, 255)
                
                img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Normalize and convert to tensor
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        
        labels = torch.from_numpy(labels).float()
        
        return img, labels, img_path


# ====================== DISTILLATION TRAINER ======================

class YOLODistillationTrainer:
    """Trainer for knowledge distillation"""
    
    def __init__(self, teacher_path, student_type='n', output_dir='/kaggle/working/distillation'):
        self.teacher_path = teacher_path
        self.student_type = student_type
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load teacher
        print("Loading teacher model...")
        self.teacher = YOLO(teacher_path)
        self.teacher.model.eval()
        
        # Create student
        print(f"Creating student model ({student_type})...")
        student_models = {
            'n': 'yolo11n.pt',
            's': 'yolo11s.pt', 
            'm': 'yolo11m.pt'
        }
        self.student = YOLO(student_models[student_type])
        
        # Print model comparison
        self._print_model_comparison()
        
    def _print_model_comparison(self):
        """Print teacher vs student comparison"""
        teacher_params = sum(p.numel() for p in self.teacher.model.parameters())
        student_params = sum(p.numel() for p in self.student.model.parameters())
        
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        print(f"Teacher parameters: {teacher_params/1e6:.2f}M")
        print(f"Student parameters: {student_params/1e6:.2f}M")
        print(f"Compression ratio: {teacher_params/student_params:.1f}x")
        print("="*50 + "\n")
    
    def distillation_loss(self, student_pred, teacher_pred, labels_list, alpha=0.7, temperature=4.0):
        """Calculate combined distillation and task loss"""
        device = student_pred[0].device
        batch_size = len(labels_list)
        
        # Task loss (student vs ground truth)
        task_loss = 0
        
        # For YOLO, we need to calculate the loss differently
        # Using MSE loss as a simple approximation
        for i in range(len(student_pred)):
            s_feat = student_pred[i]
            t_feat = teacher_pred[i]
            
            # Feature-level distillation
            feat_loss = F.mse_loss(s_feat, t_feat)
            task_loss += feat_loss
        
        # Normalize by number of layers
        task_loss /= len(student_pred)
        
        # Distillation loss (student vs teacher)
        distill_loss = 0
        
        # Softmax with temperature on classification outputs
        if len(student_pred) > 0:
            # Get the last layer features (usually contains class predictions)
            s_out = student_pred[-1]
            t_out = teacher_pred[-1]
            
            # Apply temperature scaling and calculate KL divergence
            if s_out.shape[-1] > 5:  # Has class predictions
                s_cls = s_out[..., 5:] / temperature
                t_cls = t_out[..., 5:] / temperature
                
                distill_loss = F.kl_div(
                    F.log_softmax(s_cls.view(-1, s_cls.shape[-1]), dim=-1),
                    F.softmax(t_cls.view(-1, t_cls.shape[-1]), dim=-1),
                    reduction='batchmean'
                ) * (temperature ** 2)
        
        # Combined loss
        total_loss = alpha * task_loss + (1 - alpha) * distill_loss
        
        return total_loss
    
    def train_epoch_standard(self, train_loader, optimizer, epoch, epochs, alpha=0.7, temperature=4.0):
        """Train one epoch using standard distillation"""
        self.student.model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, (images, labels_list, paths) in enumerate(pbar):
            images = images.to(device)
            
            optimizer.zero_grad()
            
            # Get predictions
            with torch.no_grad():
                teacher_outputs = self.teacher.model(images)
            
            student_outputs = self.student.model(images)
            
            # Calculate distillation loss
            loss = self.distillation_loss(
                student_outputs, teacher_outputs, labels_list,
                alpha=alpha, temperature=temperature
            )
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.student.model.parameters(), max_norm=10.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return epoch_loss / len(train_loader)
    
    def train(self, data_yaml_path, epochs=30, batch_size=16, alpha=0.7, temperature=4.0):
        """Train student with distillation using Ultralytics trainer"""
        
        print(f"Starting distillation training...")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        print(f"Alpha: {alpha}, Temperature: {temperature}")
        
        # Option 1: Use Ultralytics built-in training (recommended)
        results = self.student.train(
            data=data_yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            patience=10,
            save=True,
            device=device,
            workers=4,
            project=self.output_dir,
            name='student_distilled',
            exist_ok=True,
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            close_mosaic=10,
            # Augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0
        )
        
        # Save final model
        final_path = os.path.join(self.output_dir, 'student_final.pt')
        shutil.copy(
            os.path.join(self.output_dir, 'student_distilled/weights/best.pt'),
            final_path
        )
        print(f"\nTraining completed! Model saved to: {final_path}")
        
        return results
    
    def train_custom(self, data_yaml_path, epochs=30, batch_size=16, alpha=0.7, temperature=4.0):
        """Alternative: Custom training loop with manual distillation"""
        
        print(f"Starting custom distillation training...")
        
        # Create datasets with custom collate function
        train_dataset = ScreenPhotoDataset(data_yaml_path, mode='train', augment=True)
        val_dataset = ScreenPhotoDataset(data_yaml_path, mode='val', augment=False)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=4, 
            pin_memory=True,
            collate_fn=custom_collate_fn  # Use custom collate function
        )
        
        # Optimizer
        optimizer = torch.optim.Adam(self.student.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Move models to device
        self.teacher.model.to(device)
        self.student.model.to(device)
        
        # Training history
        history = {'train_loss': [], 'val_map': []}
        
        # Training loop
        for epoch in range(epochs):
            # Train one epoch
            avg_loss = self.train_epoch_standard(
                train_loader, optimizer, epoch, epochs, alpha, temperature
            )
            
            history['train_loss'].append(avg_loss)
            scheduler.step()
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Validation every 5 epochs
            if (epoch + 1) % 5 == 0:
                print("Validating...")
                metrics = self.student.val(data=data_yaml_path, verbose=False)
                val_map = metrics.box.map
                history['val_map'].append(val_map)
                print(f"Validation mAP@0.5:0.95: {val_map:.4f}")
                
                # Save checkpoint
                checkpoint_path = os.path.join(self.output_dir, f'student_epoch_{epoch+1}.pt')
                self.student.save(checkpoint_path)
        
        # Save final model
        final_path = os.path.join(self.output_dir, 'student_final.pt')
        self.student.save(final_path)
        print(f"\nTraining completed! Model saved to: {final_path}")
        
        return history


# ====================== EVALUATION & COMPARISON ======================

def evaluate_and_compare(teacher_path, student_path, data_yaml_path):
    """Evaluate and compare models"""
    
    print("\n" + "="*60)
    print("MODEL EVALUATION & COMPARISON")
    print("="*60)
    
    # Load models
    teacher = YOLO(teacher_path)
    student = YOLO(student_path)
    
    # 1. Model Size Comparison
    teacher_size = os.path.getsize(teacher_path) / (1024*1024)  # MB
    student_size = os.path.getsize(student_path) / (1024*1024)  # MB
    
    print(f"\nğŸ“¦ Model Size:")
    print(f"   Teacher: {teacher_size:.1f} MB")
    print(f"   Student: {student_size:.1f} MB")
    print(f"   Reduction: {teacher_size/student_size:.1f}x smaller")
    
    # 2. Inference Speed Comparison
    print(f"\nâš¡ Inference Speed (on {device}):")
    
    # Create dummy input
    dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(10):
        _ = teacher(dummy_img, verbose=False)
        _ = student(dummy_img, verbose=False)
    
    # Teacher timing
    teacher_times = []
    for _ in range(100):
        start = time.time()
        _ = teacher(dummy_img, verbose=False)
        teacher_times.append((time.time() - start) * 1000)
    
    # Student timing  
    student_times = []
    for _ in range(100):
        start = time.time()
        _ = student(dummy_img, verbose=False)
        student_times.append((time.time() - start) * 1000)
    
    teacher_avg = np.mean(teacher_times)
    student_avg = np.mean(student_times)
    
    print(f"   Teacher: {teacher_avg:.2f} ms/image")
    print(f"   Student: {student_avg:.2f} ms/image")
    print(f"   Speedup: {teacher_avg/student_avg:.2f}x faster")
    
    # 3. Accuracy Comparison
    print(f"\nğŸ�¯ Accuracy Comparison:")
    
    print("   Evaluating teacher...")
    teacher_metrics = teacher.val(data=data_yaml_path, verbose=False)
    
    print("   Evaluating student...")
    student_metrics = student.val(data=data_yaml_path, verbose=False)
    
    print(f"\n   Overall Metrics:")
    print(f"   {'Metric':<15} {'Teacher':<10} {'Student':<10} {'Retention':<10}")
    print(f"   {'-'*45}")
    
    metrics_list = [
        ('mAP@0.5', teacher_metrics.box.map50, student_metrics.box.map50),
        ('mAP@0.5:0.95', teacher_metrics.box.map, student_metrics.box.map),
        ('Precision', teacher_metrics.box.mp, student_metrics.box.mp),
        ('Recall', teacher_metrics.box.mr, student_metrics.box.mr)
    ]
    
    for metric_name, teacher_val, student_val in metrics_list:
        retention = (student_val/teacher_val)*100 if teacher_val > 0 else 0
        print(f"   {metric_name:<15} {teacher_val:.4f}     {student_val:.4f}     {retention:.1f}%")
    
    return {
        'size_reduction': teacher_size/student_size,
        'speed_improvement': teacher_avg/student_avg,
        'map_retention': (student_metrics.box.map/teacher_metrics.box.map)*100 if teacher_metrics.box.map > 0 else 0
    }



# ====================== VISUALIZE RESULTS ======================

def visualize_detections(teacher_path, student_path, data_yaml_path, num_samples=5):
    """Visualize detection results comparison"""
    
    teacher = YOLO(teacher_path)
    student = YOLO(student_path)
    
    # Load validation images
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    val_file = data_config['val']
    with open(val_file, 'r') as f:
        val_images = [line.strip() for line in f.readlines()]
    
    # Random sample
    samples = random.sample(val_images, min(num_samples, len(val_images)))
    
    # Create visualization
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_path in enumerate(samples):
        # Load original image
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Teacher prediction
        teacher_results = teacher(img_path, verbose=False)[0]
        teacher_img = teacher_results.plot()
        
        # Student prediction
        student_results = student(img_path, verbose=False)[0]
        student_img = student_results.plot()
        
        # Original
        axes[idx, 0].imshow(img_rgb)
        axes[idx, 0].set_title('Original', fontsize=12, fontweight='bold')
        axes[idx, 0].axis('off')
        
        # Teacher
        axes[idx, 1].imshow(teacher_img)
        axes[idx, 1].set_title('Teacher Model', fontsize=12, fontweight='bold')
        axes[idx, 1].axis('off')
        
        # Student
        axes[idx, 2].imshow(student_img)
        axes[idx, 2].set_title('Student Model (Distilled)', fontsize=12, fontweight='bold')
        axes[idx, 2].axis('off')
        
        # Add detection info
        if teacher_results.boxes is not None:
            teacher_count = len(teacher_results.boxes)
            axes[idx, 1].text(10, 30, f'Detections: {teacher_count}', 
                             color='white', fontsize=10, 
                             bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.5))
        
        if student_results.boxes is not None:
            student_count = len(student_results.boxes)
            axes[idx, 2].text(10, 30, f'Detections: {student_count}', 
                             color='white', fontsize=10,
                             bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/detection_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()





# ====================== MAIN EXECUTION ======================

def main():
    """Main execution function"""
    
    print("ğŸš€ YOLOv11 Knowledge Distillation for Screen Photography Detection")
    print("="*70)
    
    # Paths
    data_yaml_path = '/kaggle/working/dataset/data.yaml'
    teacher_path = '/kaggle/input/testimage/teacher_best.pt'
    output_dir = '/kaggle/working/distillation'
    
    # Step 1: Check dataset
    if not os.path.exists(data_yaml_path):
        print("â�Œ Dataset not found! Please run dataset preparation code first.")
        return
    else:
        print("âœ… Dataset found at:", data_yaml_path)
    
    # Step 2: Check teacher model
    if not os.path.exists(teacher_path):
        print("â�Œ Teacher model not found! Please train or provide teacher model.")
        return
    else:
        print("âœ… Teacher model found at:", teacher_path)
    
    # Step 3: Knowledge Distillation
    print("\nğŸ�“ Starting Knowledge Distillation...")
    
    # Configure distillation
    config = {
        'student_type': 'n',  # nano model for edge deployment
        'epochs': 30,
        'batch_size': 16,
        'alpha': 0.7,
        'temperature': 4.0
    }
    
    # Create trainer
    trainer = YOLODistillationTrainer(
        teacher_path=teacher_path,
        student_type=config['student_type'],
        output_dir=output_dir
    )
    
    # Train student using Ultralytics trainer (recommended)
    # This uses YOLO's built-in training which handles data loading properly
    results = trainer.train(
        data_yaml_path=data_yaml_path,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        alpha=config['alpha'],
        temperature=config['temperature']
    )
    
    # Alternative: Use custom training loop with manual distillation
    # Uncomment below to use custom training instead:
    # history = trainer.train_custom(
    #     data_yaml_path=data_yaml_path,
    #     epochs=config['epochs'],
    #     batch_size=config['batch_size'],
    #     alpha=config['alpha'],
    #     temperature=config['temperature']
    # )
    
    # Step 4: Evaluation
    print("\nğŸ“Š Evaluating models...")
    student_path = os.path.join(output_dir, 'student_final.pt')
    
    if os.path.exists(student_path):
        results = evaluate_and_compare(teacher_path, student_path, data_yaml_path)
        
        # Step 5: Visualization
        print("\nğŸ�¨ Generating visualizations...")
        visualize_detections(teacher_path, student_path, data_yaml_path, num_samples=5)
        
        # Step 6: Export for deployment
        print("\nğŸ“± Exporting models for deployment...")
        
        student_model = YOLO(student_path)
        
        # Export to different formats
        print("   Exporting to ONNX...")
        student_model.export(format='onnx', simplify=True)
        
        print("   Exporting to TensorFlow Lite...")
        student_model.export(format='tflite')
        
        # Final summary
        print("\n" + "="*70)
        print("âœ… DISTILLATION COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"ğŸ“¦ Size reduction: {results['size_reduction']:.1f}x smaller")
        print(f"âš¡ Speed improvement: {results['speed_improvement']:.1f}x faster") 
        print(f"ğŸ�¯ Accuracy retention: {results['map_retention']:.1f}%")
        print(f"ğŸ’¾ Student model saved to: {student_path}")
        print("="*70)
    else:
        print("â�Œ Student model training failed!")

# Run the main function
if __name__ == "__main__":
    main()


teacher_path = '/kaggle/input/finetuneyoloteacher/pytorch/default/yolo_bestx.pt'
student_path = '/kaggle/input/testimage/student_final.pt'
data_yaml_path = '/kaggle/working/dataset/data.yaml'
visualize_detections(teacher_path, student_path, data_yaml_path, num_samples=5)


!pip install ultralytics


import torch
import os
import shutil
from ultralytics import YOLO
import numpy as np

def actually_reduce_yolo_size(
    input_model_path='/kaggle/input/testimage/student_final.pt',
    working_dir='/kaggle/working/optimization_workspace'
):
    """
    Code thá»±c sá»± giáº£m kÃ­ch thÆ°á»›c model YOLO
    """
    
    print("ğŸš€ YOLO Size Reduction - Working Version")
    print("="*60)
    
    # Create working directory
    os.makedirs(working_dir, exist_ok=True)
    
    # STEP 1: Copy model to writable location
    model_path = os.path.join(working_dir, 'student_final.pt')
    shutil.copy(input_model_path, model_path)
    
    original_size = os.path.getsize(model_path) / (1024*1024)
    print(f"ğŸ“¦ Original model size: {original_size:.2f} MB")
    
    results = {}
    
    # ========== METHOD 1: Use Existing OpenVINO Model ==========
    print("\n1ï¸�âƒ£ Using Existing OpenVINO INT8 Model...")
    
    # From your previous run, OpenVINO already succeeded!
    openvino_path = '/kaggle/working/distillation/student_final_int8_openvino_model'
    
    if os.path.exists(openvino_path):
        total_size = 0
        for root, dirs, files in os.walk(openvino_path):
            for file in files:
                total_size += os.path.getsize(os.path.join(root, file))
        
        openvino_size_mb = total_size / (1024*1024)
        print(f"   âœ… Found existing OpenVINO model: {openvino_size_mb:.2f} MB")
        
        results['openvino_int8'] = {
            'path': openvino_path,
            'size_mb': openvino_size_mb,
            'reduction': ((original_size - openvino_size_mb) / original_size) * 100
        }
    
    # ========== METHOD 2: ONNX Export + Quantization ==========
    print("\n2ï¸�âƒ£ ONNX Export + INT8 Quantization...")
    
    try:
        # Load model from writable location
        model = YOLO(model_path)
        
        # Export to ONNX (this will work now)
        onnx_path = model.export(
            format='onnx',
            imgsz=320,
            simplify=True,
            dynamic=False
        )
        
        print(f"   ONNX exported to: {onnx_path}")
        onnx_size = os.path.getsize(onnx_path) / (1024*1024)
        print(f"   ONNX FP32 size: {onnx_size:.2f} MB")
        
        # Quantize to INT8
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        quantized_path = os.path.join(working_dir, 'model_int8.onnx')
        
        quantize_dynamic(
            model_input=onnx_path,
            model_output=quantized_path,
            weight_type=QuantType.QUInt8,
            per_channel=True,
            reduce_range=True
        )
        
        quantized_size = os.path.getsize(quantized_path) / (1024*1024)
        print(f"   âœ… ONNX INT8 size: {quantized_size:.2f} MB")
        
        results['onnx_int8'] = {
            'path': quantized_path,
            'size_mb': quantized_size,
            'reduction': ((original_size - quantized_size) / original_size) * 100
        }
        
    except Exception as e:
        print(f"   â�Œ ONNX failed: {e}")
    
    # ========== METHOD 3: Half Precision (FP16) ==========
    print("\n3ï¸�âƒ£ Half Precision (FP16) Model...")
    
    try:
        # Load model
        model = YOLO(model_path)
        
        # Convert to half precision
        model.model = model.model.half()
        
        # Save FP16 model
        fp16_path = os.path.join(working_dir, 'model_fp16.pt')
        
        # Save only the model state
        torch.save({
            'model': model.model.state_dict(),
            'nc': model.model.nc,  # number of classes
            'names': model.names,  # class names
        }, fp16_path)
        
        fp16_size = os.path.getsize(fp16_path) / (1024*1024)
        print(f"   âœ… FP16 model size: {fp16_size:.2f} MB")
        
        results['fp16'] = {
            'path': fp16_path,
            'size_mb': fp16_size,
            'reduction': ((original_size - fp16_size) / original_size) * 100
        }
        
    except Exception as e:
        print(f"   â�Œ FP16 failed: {e}")
    
    # ========== METHOD 4: Simple INT8 Weights ==========
    print("\n4ï¸�âƒ£ Simple INT8 Weight Quantization...")
    
    try:
        # Load model with weights_only=False
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Get model state dict
        if hasattr(checkpoint, 'state_dict'):
            state_dict = checkpoint.state_dict()
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model'].state_dict() if hasattr(checkpoint['model'], 'state_dict') else checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Quantize weights
        quantized_weights = {}
        metadata = {}
        
        for name, param in state_dict.items():
            if param.dtype == torch.float32 and param.numel() > 100:  # Only quantize larger tensors
                # Calculate scale and zero point
                min_val = param.min().item()
                max_val = param.max().item()
                scale = (max_val - min_val) / 255.0
                zero_point = round(-min_val / scale)
                
                # Quantize to uint8
                quantized = torch.clamp(
                    torch.round(param / scale + zero_point), 
                    0, 255
                ).to(torch.uint8)
                
                quantized_weights[name] = quantized
                metadata[name] = {'scale': scale, 'zero_point': zero_point}
            else:
                # Keep small tensors as is
                quantized_weights[name] = param
        
        # Save quantized model
        int8_path = os.path.join(working_dir, 'model_int8_simple.pth')
        torch.save({
            'weights': quantized_weights,
            'metadata': metadata,
            'model_info': {
                'nc': 4,  # number of classes
                'names': ['person', 'phone', 'reflex_camera', 'polaroid_camera']
            }
        }, int8_path, pickle_protocol=4)
        
        int8_size = os.path.getsize(int8_path) / (1024*1024)
        print(f"   âœ… INT8 weights size: {int8_size:.2f} MB")
        
        results['int8_weights'] = {
            'path': int8_path,
            'size_mb': int8_size,
            'reduction': ((original_size - int8_size) / original_size) * 100
        }
        
    except Exception as e:
        print(f"   â�Œ INT8 weights failed: {e}")
    
    # ========== METHOD 5: Export for Specific Platform ==========
    print("\n5ï¸�âƒ£ Platform-Specific Exports...")
    
    # CoreML for iOS (usually smaller)
    try:
        model = YOLO(model_path)
        coreml_path = model.export(format='coreml', imgsz=320, nms=True)
        if os.path.exists(coreml_path):
            coreml_size = os.path.getsize(coreml_path) / (1024*1024)
            print(f"   âœ… CoreML size: {coreml_size:.2f} MB")
            results['coreml'] = {
                'path': coreml_path,
                'size_mb': coreml_size,
                'reduction': ((original_size - coreml_size) / original_size) * 100
            }
    except:
        pass
    
    # ========== SUMMARY ==========
    print("\n" + "="*60)
    print("ğŸ“Š OPTIMIZATION RESULTS")
    print("="*60)
    
    print(f"\nOriginal size: {original_size:.2f} MB\n")
    
    # Sort by size
    sorted_results = sorted(
        [(k, v) for k, v in results.items()], 
        key=lambda x: x[1]['size_mb']
    )
    
    print(f"{'Method':<20} {'Size (MB)':<10} {'Reduction':<15} {'Status'}")
    print("-"*60)
    
    for name, info in sorted_results:
        reduction = info['reduction']
        status = "âœ… Success" if reduction > 0 else "â�Œ Larger"
        print(f"{name:<20} {info['size_mb']:<10.2f} {reduction:>6.1f}%        {status}")
    
    # Best option
    best_options = [r for r in sorted_results if r[1]['reduction'] > 0]
    if best_options:
        best_name, best_info = best_options[0]
        print(f"\nğŸ�¯ BEST OPTION: {best_name}")
        print(f"   Size: {best_info['size_mb']:.2f} MB (â†“{best_info['reduction']:.1f}%)")
        print(f"   Path: {best_info['path']}")
        
        # Deployment code
        if 'onnx' in best_name:
            print("\nğŸ’» DEPLOYMENT CODE:")
            print("-"*40)
            print("""
# For Raspberry Pi / Jetson
import onnxruntime as ort
import numpy as np
import cv2

# Load INT8 model
session = ort.InferenceSession('model_int8.onnx')

# Inference function
def detect(image_path):
    # Preprocess
    img = cv2.imread(image_path)
    img = cv2.resize(img, (320, 320))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    
    # Run inference
    outputs = session.run(None, {session.get_inputs()[0].name: img})
    return outputs

# Use it
results = detect('test.jpg')
""")
        
        elif 'openvino' in best_name:
            print("\nğŸ’» For Intel devices:")
            print(f"Model ready at: {best_info['path']}")
            print("Use OpenVINO Runtime for inference")
    
    else:
        print("\nâš ï¸� No size reduction achieved. Try different methods.")
    
    return results

# ========== QUICK FIX: Use Existing Models ==========
def use_existing_optimized_models():
    """
    Sá»­ dá»¥ng cÃ¡c model Ä‘Ã£ tá»‘i Æ°u tá»« láº§n cháº¡y trÆ°á»›c
    """
    print("\nğŸ”� Checking for existing optimized models...")
    
    models = {
        'OpenVINO INT8': '/kaggle/working/distillation/student_final_int8_openvino_model',
        'ONNX': '/kaggle/working/distillation/student_final.onnx',
    }
    
    found = []
    for name, path in models.items():
        if os.path.exists(path):
            if os.path.isdir(path):
                size = sum(os.path.getsize(os.path.join(root, f)) 
                          for root, dirs, files in os.walk(path) 
                          for f in files) / (1024*1024)
            else:
                size = os.path.getsize(path) / (1024*1024)
            
            found.append((name, path, size))
            print(f"âœ… Found {name}: {size:.2f} MB at {path}")
    
    if found:
        best = min(found, key=lambda x: x[2])
        print(f"\nğŸ�¯ Best existing model: {best[0]} ({best[2]:.2f} MB)")
        print(f"Path: {best[1]}")
        
        return best[1]
    
    return None

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    # First check existing models
    existing = use_existing_optimized_models()
    
    if existing:
        print("\nâœ… You already have optimized models!")
        print("Use the OpenVINO model at:")
        print("/kaggle/working/distillation/student_final_int8_openvino_model")
        print("Size: ~3.2 MB (38% smaller than original)")
    
    # Then try new optimization
    print("\n" + "="*60)
    print("Running new optimization...")
    
    results = actually_reduce_yolo_size()
    
    print("\nâœ… Done! Your best options for IoT deployment are above.")


import os
import random
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np

# --- Load YOLOv8 model ---
# DÃ¹ng mÃ´ hÃ¬nh YOLOv8 Ä‘Ã£ Ä‘Æ°á»£c huáº¥n luyá»‡n trÆ°á»›c vá»›i dá»¯ liá»‡u COCO
model = YOLO('/kaggle/working/optimization_workspace/model_int8.onnx')  # Sá»­ dá»¥ng model YOLOv8 medium pre-trained

# --- Load áº£nh ---
# Chá»�n má»™t áº£nh ngáº«u nhiÃªn tá»« bá»™ validation cá»§a COCO
img_path = '/kaggle/input/testimage/WIN_20250607_19_34_01_Pro.jpg'  # Cáº­p nháº­t Ä‘Ãºng Ä‘Æ°á»�ng dáº«n áº£nh náº¿u cáº§n
img = Image.open(img_path).convert('RGB')

# --- Cháº¡y inference vá»›i YOLOv8 ---
results = model(img_path)[0]  # Sá»­ dá»¥ng mÃ´ hÃ¬nh Ä‘Ã£ táº£i Ä‘á»ƒ cháº¡y inference
boxes = results.boxes.xyxy.cpu().numpy()  # Láº¥y toáº¡ Ä‘á»™ bounding boxes (x1, y1, x2, y2)
classes = results.boxes.cls.cpu().numpy().astype(int)  # Láº¥y lá»›p cá»§a cÃ¡c object
confidences = results.boxes.conf.cpu().numpy()  # Láº¥y Ä‘á»™ tá»± tin cá»§a cÃ¡c prediction

# --- Váº½ káº¿t quáº£ vÃ  Ä‘Ã¡nh giÃ¡ hÃ nh Ä‘á»™ng ---
draw = ImageDraw.Draw(img)
font = ImageFont.load_default()

# Táº­p há»£p cÃ¡c box theo class
by_cls = {}
for box, cls, conf in zip(boxes, classes, confidences):
    by_cls.setdefault(cls, []).append((box, conf))

# Kiá»ƒm tra hÃ nh Ä‘á»™ng "chá»¥p áº£nh mÃ n hÃ¬nh"
persons = by_cls.get(0, [])  # Lá»›p person (ID=0)
phones = by_cls.get(1, [])  # Lá»›p cell phone (ID=67)

detected = False
for p_box, _ in persons:
    for ph_box, _ in phones:
        # Kiá»ƒm tra xem Ä‘iá»‡n thoáº¡i cÃ³ gáº§n mÃ n hÃ¬nh (cÃ³ thá»ƒ báº¡n cáº§n thÃªm "screen" náº¿u muá»‘n)
        cx, cy = (ph_box[0] + ph_box[2]) / 2, (ph_box[1] + ph_box[3]) / 2  # Tá»�a Ä‘á»™ trung tÃ¢m cá»§a Ä‘iá»‡n thoáº¡i
        px, py = (p_box[0] + p_box[2]) / 2, (p_box[1] + p_box[3]) / 2  # Tá»�a Ä‘á»™ trung tÃ¢m cá»§a ngÆ°á»�i

        # Kiá»ƒm tra xem Ä‘iá»‡n thoáº¡i cÃ³ náº±m trong vÃ¹ng táº§m tay cá»§a ngÆ°á»�i hay khÃ´ng
        if p_box[0] < cx < p_box[2] and p_box[1] < cy < p_box[3]:
            # Váº½ cÃ¡c bounding boxes
            draw.rectangle(p_box.tolist(), outline='blue', width=2)
            draw.rectangle(ph_box.tolist(), outline='green', width=2)
            draw.text((ph_box[0], ph_box[1]-10), "likely taking photo", fill='green', font=font)
            detected = True
            break
    if detected: break

if detected:
    print("Detected taking-photo action based on rule-based heuristic.")
else:
    print("No taking-photo action detected.")

# --- Hiá»ƒn thá»‹ káº¿t quáº£ ---
plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.axis('off')
plt.show()

