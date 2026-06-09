%%capture
# Example: Download YOLOv8 weights (adjust for YOLOv11 if available)
!wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt -O /kaggle/working/yolo8x.pt
!wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov11x.pt -O /kaggle/working/yolo11x.pt
# Install necessary packages
!pip install ultralytics
!pip install ensemble_boxes
!pip install -U ultralytics
!pip install optuna
# Cache clearing
!pip install --no-cache-dir torch torchvision
!pip install --no-cache-dir ultralytics
!pip install --no-cache-dir albumentations
!pip install --no-cache-dir ensemble-boxes
!pip install --no-cache-dir pycocotools


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
import csv
import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import csv
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from tqdm import tqdm
import warnings
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou, nms
from tqdm import tqdm
import torchvision.transforms.v2 as T
import pandas as pd
import csv
from ensemble_boxes import weighted_boxes_fusion
from torchvision.ops import box_iou, nms
import optuna

warnings.filterwarnings('ignore')

# Verify installations
try:
    from ultralytics import YOLO
    print("[notice] All dependencies installed successfully")
    print(f"[notice] PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"[error] Failed to import dependencies: {e}")
    print("[error] Please restart the kernel and re-run the script.")
    raise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Dataset configuration
data_yaml = """
path: /kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2
train: train/images
val: val/images
test: testImages/images
nc: 1
names: ['object']
"""

# Hyperparameters Configuration
TRAIN_EPOCHS = 100
IMG_SIZE = 640
PATIENCE = 10
BATCH_SIZE = 8
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_path = "/kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2"


# Save data.yaml
os.makedirs('/kaggle/working', exist_ok=True)
with open('/kaggle/working/data.yaml', 'w') as file:
    file.write(data_yaml)


# Data transforms
train_transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ColorJitter(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

val_transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


# --- Custom Dataset ---
class SoupCanDataset(Dataset):
    def __init__(self, image_dir, label_dir, transforms=None):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transforms = transforms
        self.images = [p for p in self.image_dir.glob("*") if p.suffix.lower() in ['.png', '.jpg', '.jpeg']]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.label_dir / f"{img_path.stem}.txt"

        img = Image.open(img_path).convert("RGB")
        img_width, img_height = img.size

        boxes = []
        labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    try:
                        cls_id, x_center, y_center, width, height = map(float, line.strip().split())
                        x1 = (x_center - width / 2) * img_width
                        y1 = (y_center - height / 2) * img_height
                        x2 = (x_center + width / 2) * img_width
                        y2 = (y_center + height / 2) * img_height
                        boxes.append([x1, y1, x2, y2])
                        labels.append(int(cls_id))
                    except ValueError:
                        print(f"[warning] Invalid label in {label_path}: {line.strip()}")

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.empty((0,)),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
        }

        if self.transforms:
            img = self.transforms(img)
            if len(boxes) > 0:
                scale = IMG_SIZE / max(img_width, img_height)
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
                target["boxes"] = boxes.clamp(min=0, max=IMG_SIZE-1)

        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

# Create datasets and dataloaders
train_dataset = SoupCanDataset(f"{base_path}/train/images", f"{base_path}/train/labels", transforms=train_transforms)
val_dataset = SoupCanDataset(f"{base_path}/val/images", f"{base_path}/val/labels", transforms=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn)


print("[notice] Training YOLO8n...")
try:
    yolo8n_model = YOLO("yolov8n.pt")
    yolo8n_results = yolo8n_model.train(
        data="/kaggle/working/data.yaml",
        epochs=TRAIN_EPOCHS,
        imgsz=IMG_SIZE,
        patience=PATIENCE,
        cos_lr=True,
        dropout=0.2,
        mosaic=0.5,
        lr0=0.001,
        optimizer="Adam",
        momentum=0.9,
        weight_decay=0.0005,
        single_cls=True,
        plots=True,
        cache=True,
        flipud=0.5,
        scale=0.8,
        name="yolo8n_trained",
        verbose=True
    )
except Exception as e:
    print(f"[error] YOLO8n training failed: {e}")
    raise


print("[notice] Training YOLO8x...")
try:
    yolo8x_model = YOLO("yolov8x.pt")
    yolo8x_results = yolo8x_model.train(
        data="/kaggle/working/data.yaml",
        epochs=TRAIN_EPOCHS,
        imgsz=IMG_SIZE,
        patience=PATIENCE,
        cos_lr=True,
        dropout=0.4,
        mosaic=0.2,
        lr0=0.0001,
        optimizer="SGD",
        momentum=0.975,
        weight_decay=0.0001,
        single_cls=True,
        plots=True,
        cache=True,
        flipud=0.25,
        scale=1.0,
        name="yolo8x_trained",
        verbose=True
    )
except Exception as e:
    print(f"[error] YOLO8x training failed: {e}")
    raise


# Data Transformations
train_transforms = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.RandomHorizontalFlip(p=0.5),
    T.Resize(size=(IMG_SIZE, IMG_SIZE)),
])

val_transforms = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Resize(size=(IMG_SIZE, IMG_SIZE)),
])

# Custom Dataset
class SoupCanDataset(Dataset):
    def __init__(self, image_dir, label_dir, transforms=None):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transforms = transforms
        self.images = [p for p in self.image_dir.glob("*") if p.suffix.lower() in ['.png', '.jpg', '.jpeg']]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.label_dir / f"{img_path.stem}.txt"

        img = Image.open(img_path).convert("RGB")
        img_width, img_height = img.size

        boxes = []
        labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    try:
                        cls_id, x_center, y_center, width, height = map(float, line.strip().split())
                        x1 = (x_center - width / 2) * img_width
                        y1 = (y_center - height / 2) * img_height
                        x2 = (x_center + width / 2) * img_width
                        y2 = (y_center + height / 2) * img_height
                        boxes.append([x1, y1, x2, y2])
                        labels.append(int(cls_id))
                    except ValueError:
                        print(f"[warning] Invalid label in {label_path}: {line.strip()}")

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.empty((0,)),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
        }

        if self.transforms:
            img = self.transforms(img)
            if len(boxes) > 0:
                scale = IMG_SIZE / max(img_width, img_height)
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
                target["boxes"] = boxes.clamp(min=0, max=IMG_SIZE-1)

        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

# Create datasets and dataloaders
train_dataset = SoupCanDataset(f"{base_path}/train/images", f"{base_path}/train/labels", transforms=train_transforms)
val_dataset = SoupCanDataset(f"{base_path}/val/images", f"{base_path}/val/labels", transforms=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn)


# Train Faster R-CNN with ResNet50-FPN
print("[notice] Training Faster R-CNN with ResNet50-FPN...")
def get_faster_rcnn_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)  # +1 for background
    return model

faster_rcnn_model = get_faster_rcnn_model(num_classes=1)  # 1 class (object) + background
faster_rcnn_model.to(device)

optimizer = torch.optim.SGD(faster_rcnn_model.parameters(), lr=0.0001, momentum=0.975, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_EPOCHS)

def train_faster_rcnn(model, train_loader, val_loader, epochs, patience):
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            train_loss += losses.item()
        
        train_loss /= len(train_loader)
        
        # Validation with loss computation in training mode
        model.train()  # Temporarily set to train mode for loss
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
        
        val_loss /= len(val_loader)
        print(f"[notice] Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "/kaggle/working/faster_rcnn_best.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("[notice] Early stopping triggered")
                break
        
        scheduler.step()

try:
    train_faster_rcnn(faster_rcnn_model, train_loader, val_loader, TRAIN_EPOCHS, PATIENCE)
except Exception as e:
    print(f"[error] Faster R-CNN training failed: {e}")
    raise


# Load and Prepare Models for Ensemble
print("[notice] Loading trained models for ensemble...")
yolo8x_model = YOLO("/kaggle/working/runs/detect/yolo8n_trained/weights/best.pt")  
yolo8n_model = YOLO("/kaggle/working/runs/detect/yolo8x_trained/weights/best.pt")
faster_rcnn_model.load_state_dict(torch.load("/kaggle/working/faster_rcnn_best.pt"))
faster_rcnn_model.to(device)
faster_rcnn_model.eval()


# Ensemble Validation
@torch.no_grad()
def run_ensemble_inference(image_tensor, conf_thres=0.25, iou_thres=0.5):
    """
    Runs inference with YOLO and Faster R-CNN models and combines their results.
    Accepts a PyTorch tensor with a batch dimension.
    """
    # YOLOv8x inference
    yolo8x_results = yolo8x_model.predict(image_tensor, conf=conf_thres, iou=iou_thres, verbose=False)
    if len(yolo8x_results[0].boxes.data) == 0:
        yolo8x_preds = torch.empty((0, 6), device=device)
    else:
        yolo8x_preds = yolo8x_results[0].boxes.data.to(device)

    # YOLOv8n inference
    yolo8n_results = yolo8n_model.predict(image_tensor, conf=conf_thres, iou=iou_thres, verbose=False)
    if len(yolo8n_results[0].boxes.data) == 0:
        yolo8n_preds = torch.empty((0, 6), device=device)
    else:
        yolo8n_preds = yolo8n_results[0].boxes.data.to(device)

    # Faster R-CNN inference
    faster_rcnn_image = image_tensor.to(device)  # Move to GPU directly
    faster_rcnn_results = faster_rcnn_model([faster_rcnn_image[0]])  # Pass as list, single image per call
    if len(faster_rcnn_results[0]['boxes']) == 0:
        faster_rcnn_preds = torch.empty((0, 6), device=device)
    else:
        boxes = faster_rcnn_results[0]['boxes'].to(device)
        scores = faster_rcnn_results[0]['scores'].to(device)
        labels = faster_rcnn_results[0]['labels'].to(device)
        mask = (labels == 1)  # Filter for class 1 (object)
        faster_rcnn_preds = torch.cat((boxes[mask], scores[mask].unsqueeze(1), torch.zeros_like(scores[mask].unsqueeze(1))), dim=1)

    # Combine detections
    combined_detections = torch.cat((yolo8x_preds, yolo8n_preds, faster_rcnn_preds), dim=0)

    if combined_detections.shape[0] == 0:
        return torch.empty((0, 4), device=device), torch.empty((0,), device=device), torch.empty((0,), device=device)

    # Apply NMS
    combined_boxes = combined_detections[:, :4]
    combined_scores = combined_detections[:, 4]
    combined_classes = combined_detections[:, 5]

    keep_indices = nms(combined_boxes, combined_scores, iou_thres)
    
    final_boxes = combined_boxes[keep_indices]
    final_scores = combined_scores[keep_indices]
    final_classes = combined_classes[keep_indices]

    # Scale boxes to match IMG_SIZE
    orig_shape = yolo8x_results[0].orig_shape
    scale_x = IMG_SIZE / orig_shape[1]
    scale_y = IMG_SIZE / orig_shape[0]
    final_boxes[:, [0, 2]] *= scale_x
    final_boxes[:, [1, 3]] *= scale_y
    final_boxes = final_boxes.clamp(min=0, max=IMG_SIZE-1)

    return final_boxes, final_scores, final_classes

@torch.no_grad()
def validate_ensemble(val_loader):
    """
    Validates the ensemble model on the validation dataset.
    """
    print("[notice] Validating ensemble model...")
    all_metrics = []

    for images, targets in tqdm(val_loader, desc="Validating Ensemble"):
        images = torch.stack(images).to(device)
        batch_size = len(images)

        final_boxes_batch = []
        final_scores_batch = []
        final_classes_batch = []
        for i in range(batch_size):
            final_boxes, final_scores, final_classes = run_ensemble_inference(images[i].unsqueeze(0))
            final_boxes_batch.append(final_boxes)
            final_scores_batch.append(final_scores)
            final_classes_batch.append(final_classes)

        for i in range(batch_size):
            gt_boxes = targets[i]['boxes'].to(device)
            if len(gt_boxes) == 0 or len(final_boxes_batch[i]) == 0:
                all_metrics.append({'precision': 0, 'recall': 0})
                continue

            iou_matrix = box_iou(gt_boxes, final_boxes_batch[i])
            detected_count = 0

            for gt_idx in range(len(gt_boxes)):
                if torch.max(iou_matrix[gt_idx]) >= 0.5:
                    detected_count += 1

            true_positives = detected_count
            predicted_positives = len(final_boxes_batch[i])
            actual_positives = len(gt_boxes)

            precision = true_positives / predicted_positives if predicted_positives > 0 else 0
            recall = true_positives / actual_positives if actual_positives > 0 else 0
            all_metrics.append({'precision': precision, 'recall': recall})

    if all_metrics:
        avg_precision = np.mean([m['precision'] for m in all_metrics])
        avg_recall = np.mean([m['recall'] for m in all_metrics])
        print(f"Ensemble Validation Results:")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
    else:
        print("No detections found. Validation metrics not calculated.")

# Run the validation on the ensembled models
validate_ensemble(val_loader)


# Ensemble Validation with Hyperparameter Tuning
@torch.no_grad()
def run_ensemble_inference(image_tensor, conf_thres, iou_thres):
    """
    Runs inference with YOLO and Faster R-CNN models and combines their results.
    Accepts a PyTorch tensor with a batch dimension.
    """
    # YOLOv8x inference
    yolo8x_results = yolo8x_model.predict(image_tensor, conf=conf_thres, iou=iou_thres, verbose=False)
    if len(yolo8x_results[0].boxes.data) == 0:
        yolo8x_preds = torch.empty((0, 6), device=device)
    else:
        yolo8x_preds = yolo8x_results[0].boxes.data.to(device)

    # YOLOv8n inference
    yolo8n_results = yolo8n_model.predict(image_tensor, conf=conf_thres, iou=iou_thres, verbose=False)
    if len(yolo8n_results[0].boxes.data) == 0:
        yolo8n_preds = torch.empty((0, 6), device=device)
    else:
        yolo8n_preds = yolo8n_results[0].boxes.data.to(device)

    # Faster R-CNN inference
    faster_rcnn_image = image_tensor.to(device)
    faster_rcnn_results = faster_rcnn_model([faster_rcnn_image[0]])
    if len(faster_rcnn_results[0]['boxes']) == 0:
        faster_rcnn_preds = torch.empty((0, 6), device=device)
    else:
        boxes = faster_rcnn_results[0]['boxes'].to(device)
        scores = faster_rcnn_results[0]['scores'].to(device)
        labels = faster_rcnn_results[0]['labels'].to(device)
        mask = (labels == 1)
        faster_rcnn_preds = torch.cat((boxes[mask], scores[mask].unsqueeze(1), torch.zeros_like(scores[mask].unsqueeze(1))), dim=1)

    # Combine detections
    combined_detections = torch.cat((yolo8x_preds, yolo8n_preds, faster_rcnn_preds), dim=0)

    if combined_detections.shape[0] == 0:
        return torch.empty((0, 4), device=device), torch.empty((0,), device=device), torch.empty((0,), device=device)

    # Apply NMS
    combined_boxes = combined_detections[:, :4]
    combined_scores = combined_detections[:, 4]
    combined_classes = combined_detections[:, 5]

    keep_indices = nms(combined_boxes, combined_scores, iou_thres)
    
    final_boxes = combined_boxes[keep_indices]
    final_scores = combined_scores[keep_indices]
    final_classes = combined_classes[keep_indices]

    # Scale boxes to match IMG_SIZE
    orig_shape = yolo8x_results[0].orig_shape
    scale_x = IMG_SIZE / orig_shape[1]
    scale_y = IMG_SIZE / orig_shape[0]
    final_boxes[:, [0, 2]] *= scale_x
    final_boxes[:, [1, 3]] *= scale_y
    final_boxes = final_boxes.clamp(min=0, max=IMG_SIZE-1)

    return final_boxes, final_scores, final_classes

@torch.no_grad()
def validate_ensemble(val_loader, conf_thres, iou_thres):
    """
    Validates the ensemble model on the validation dataset.
    """
    all_metrics = []

    for images, targets in tqdm(val_loader, desc="Validating Ensemble"):
        images = torch.stack(images).to(device)
        batch_size = len(images)

        final_boxes_batch = []
        final_scores_batch = []
        final_classes_batch = []
        for i in range(batch_size):
            final_boxes, final_scores, final_classes = run_ensemble_inference(images[i].unsqueeze(0), conf_thres, iou_thres)
            final_boxes_batch.append(final_boxes)
            final_scores_batch.append(final_scores)
            final_classes_batch.append(final_classes)

        for i in range(batch_size):
            gt_boxes = targets[i]['boxes'].to(device)
            if len(gt_boxes) == 0 or len(final_boxes_batch[i]) == 0:
                all_metrics.append({'precision': 0, 'recall': 0})
                continue

            iou_matrix = box_iou(gt_boxes, final_boxes_batch[i])
            detected_count = 0

            for gt_idx in range(len(gt_boxes)):
                if torch.max(iou_matrix[gt_idx]) >= 0.5:
                    detected_count += 1

            true_positives = detected_count
            predicted_positives = len(final_boxes_batch[i])
            actual_positives = len(gt_boxes)

            precision = true_positives / predicted_positives if predicted_positives > 0 else 0
            recall = true_positives / actual_positives if actual_positives > 0 else 0
            all_metrics.append({'precision': precision, 'recall': recall})

    if all_metrics:
        avg_precision = np.mean([m['precision'] for m in all_metrics])
        avg_recall = np.mean([m['recall'] for m in all_metrics])
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        print(f"Validation Results - conf_thres={conf_thres:.2f}, iou_thres={iou_thres:.2f}:")
        print(f"Average Precision: {avg_precision:.4f}, Average Recall: {avg_recall:.4f}, F1 Score: {f1_score:.4f}")
    else:
        f1_score = 0
        print(f"Validation Results - conf_thres={conf_thres:.2f}, iou_thres={iou_thres:.2f}: No detections found, F1 Score: {f1_score:.4f}")

    return f1_score

def objective(trial):
    """
    Optuna objective function to tune hyperparameters.
    """
    conf_thres = trial.suggest_float("conf_thres", 0.1, 0.9)
    iou_thres = trial.suggest_float("iou_thres", 0.3, 0.7)
    
    f1_score = validate_ensemble(val_loader, conf_thres, iou_thres)
    return f1_score

# Create and run Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Print best parameters and F1 score
best_params = study.best_params
best_f1 = study.best_value
print(f"Best hyperparameters: {best_params}")
print(f"Best F1 Score: {best_f1:.4f}")


# Load and Prepare Models for Ensemble
print("[notice] Loading trained models for ensemble...")
yolo8x_model = YOLO("/kaggle/working/runs/detect/yolo8n_trained/weights/best.pt")
yolo8n_model = YOLO("/kaggle/working/runs/detect/yolo8x_trained/weights/best.pt")
faster_rcnn_model.load_state_dict(torch.load("/kaggle/working/faster_rcnn_best.pt"))
faster_rcnn_model.to(device)
faster_rcnn_model.eval()


CONF_THRESHOLD = 0.2793820431159878
IOU_THRESHOLD = 0.5125192058617429

# Ensemble Inference
test_images_path = f"{base_path}/testImages/images"
output_dir = "/kaggle/working/predictions/labels"
os.makedirs(output_dir, exist_ok=True)

test_transforms = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Resize(size=(IMG_SIZE, IMG_SIZE)),
])

for img_path in tqdm(list(Path(test_images_path).glob("*")), desc="Predicting"):
    if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
        continue
    
    img_name = img_path.stem
    img = Image.open(img_path).convert("RGB")
    img_width, img_height = img.size
    
    # Preprocess image
    img_tensor = test_transforms(img).unsqueeze(0).to(device)  # Add batch dimension
    
    # YOLOv8x predictions
    yolo8x_boxes = []
    yolo8x_scores = []
    yolo8x_labels = []
    try:
        yolo8x_results = yolo8x_model.predict(img_path, conf=CONF_THRESHOLD, verbose=False)
        for result in yolo8x_results:
            boxes = result.boxes.data
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2, conf, cls_id = box.tolist()
                    if conf >= CONF_THRESHOLD:
                        # Normalize to [0, 1] relative to original image size
                        yolo8x_boxes.append([x1/img_width, y1/img_height, x2/img_width, y2/img_height])
                        yolo8x_scores.append(conf)
                        yolo8x_labels.append(int(cls_id))
    except Exception as e:
        print(f"[warning] YOLOv8x prediction failed for {img_name}: {e}")
    
    # YOLOv8n predictions
    yolo8n_boxes = []
    yolo8n_scores = []
    yolo8n_labels = []
    try:
        yolo8n_results = yolo8n_model.predict(img_path, conf=CONF_THRESHOLD, verbose=False)
        for result in yolo8n_results:
            boxes = result.boxes.data
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2, conf, cls_id = box.tolist()
                    if conf >= CONF_THRESHOLD:
                        yolo8n_boxes.append([x1/img_width, y1/img_height, x2/img_width, y2/img_height])
                        yolo8n_scores.append(conf)
                        yolo8n_labels.append(int(cls_id))
    except Exception as e:
        print(f"[warning] YOLOv8n prediction failed for {img_name}: {e}")
    
    # Faster R-CNN predictions
    frcnn_boxes = []
    frcnn_scores = []
    frcnn_labels = []
    try:
        with torch.no_grad():
            predictions = faster_rcnn_model([img_tensor[0]])[0]
            for box, score, label in zip(predictions['boxes'], predictions['scores'], predictions['labels']):
                if score >= CONF_THRESHOLD and label == 1:  # Class 1 is object
                    x1, y1, x2, y2 = box.tolist()
                    # Scale boxes to original image size and normalize
                    scale_x = img_width / IMG_SIZE
                    scale_y = img_height / IMG_SIZE
                    x1, x2 = x1 * scale_x / img_width, x2 * scale_x / img_width
                    y1, y2 = y1 * scale_y / img_height, y2 * scale_y / img_height
                    frcnn_boxes.append([x1, y1, x2, y2])
                    frcnn_scores.append(score.item())
                    frcnn_labels.append(0)  # Map to class 0 for consistency
    except Exception as e:
        print(f"[warning] Faster R-CNN prediction failed for {img_name}: {e}")
    
    # Ensemble using Weighted Boxes Fusion
    boxes_list = [yolo8x_boxes, yolo8n_boxes, frcnn_boxes]
    scores_list = [yolo8x_scores, yolo8n_scores, frcnn_scores]
    labels_list = [yolo8x_labels, yolo8n_labels, frcnn_labels]
    weights = [1.0, 0.9, 0.8]  # YOLOv8x > YOLOv8n > Faster R-CNN
    
    try:
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=IOU_THRESHOLD,
            skip_box_thr=CONF_THRESHOLD
        )
    except Exception as e:
        print(f"[warning] WBF failed for {img_name}: {e}")
        fused_boxes, fused_scores, fused_labels = [], [], []
    
    # Save ensemble predictions in YOLO format
    output_txt = Path(output_dir) / f"{img_name}.txt"
    with open(output_txt, "w") as f:
        for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
            x1, y1, x2, y2 = box
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            f.write(f"{int(label)} {score:.6f} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print(f"[notice] All ensemble detections saved to: {output_dir}")


# Convert Predictions to Submission CSV
def predictions_to_csv(
    preds_folder: str = "/kaggle/working/predictions/labels",
    output_csv: str = "/kaggle/working/submission.csv",
    test_images_folder: str = "/kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2/testImages/images",
    allowed_extensions: tuple = (".jpg", ".png", ".jpeg")
):
    preds_path = Path(preds_folder)
    test_images_path = Path(test_images_folder)
    
    test_images = {p.stem for p in test_images_path.glob("*") if p.suffix.lower() in allowed_extensions}
    predictions = []
    predicted_images = set()
    
    for txt_file in preds_path.glob("*.txt"):
        image_id = txt_file.stem
        predicted_images.add(image_id)
        with open(txt_file, "r") as f:
            valid_lines = [line.strip() for line in f if len(line.strip().split()) == 6]
        pred_str = " ".join(valid_lines) if valid_lines else "no boxes"
        predictions.append({"image_id": image_id, "prediction_string": pred_str})
    
    missing_images = test_images - predicted_images
    for image_id in missing_images:
        predictions.append({"image_id": image_id, "prediction_string": "no boxes"})
    
    submission_df = pd.DataFrame(predictions)
    submission_df.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"[notice] Submission saved to {output_csv}")

predictions_to_csv()

