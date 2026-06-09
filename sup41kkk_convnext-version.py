import os
import random
import numpy as np
import pandas as pd
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import BackboneWithFPN
import timm


DATA_DIR = Path('/kaggle/input/where-are-the-seagulls/data')
TRAIN_IMG_DIR = DATA_DIR / 'train/images'
TRAIN_LBL_DIR = DATA_DIR / 'train/labels'
TEST_IMG_DIR = DATA_DIR / 'test/images'


BATCH_SIZE = 8
NUM_WORKERS = os.cpu_count() // 2
NUM_CLASSES = 2 
IMG_SIZE = 512
SEED = 42


pl.seed_everything(SEED, workers=True)


if torch.cuda.is_available():
    gpu_info = !nvidia-smi --query-gpu=gpu_name --format=csv,noheader
    print(f"âœ… GPU Ğ´Ğ¾Ñ�Ñ‚ÑƒĞ¿ĞµĞ½: {gpu_info[0]}")
    DEVICE = "cuda"
else:
    print("âš ï¸� GPU Ğ½Ğµ Ğ½Ğ°Ğ¹Ğ´ĞµĞ½, Ğ¸Ñ�Ğ¿Ğ¾Ğ»ÑŒĞ·ÑƒĞµÑ‚Ñ�Ñ� CPU.")
    DEVICE = "cpu"



print("ğŸš€ Ğ�Ğ°Ñ‡Ğ°Ğ»Ğ¾ Ğ°Ğ½Ğ°Ğ»Ğ¸Ğ·Ğ° Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…...")

# ĞŸĞ¾Ğ»ÑƒÑ‡Ğ°ĞµĞ¼ Ñ�Ğ¿Ğ¸Ñ�ĞºĞ¸ Ñ„Ğ°Ğ¹Ğ»Ğ¾Ğ²
train_image_files = sorted([p for p in TRAIN_IMG_DIR.glob('*.jpg')])
train_label_files = sorted([p for p in TRAIN_LBL_DIR.glob('*.txt')])

print(f"Ğ’Ñ�ĞµĞ³Ğ¾ Ğ¸Ğ·Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ğ¹ Ğ´Ğ»Ñ� Ñ‚Ñ€ĞµĞ½Ğ¸Ñ€Ğ¾Ğ²ĞºĞ¸: {len(train_image_files)}")
print(f"Ğ’Ñ�ĞµĞ³Ğ¾ Ñ„Ğ°Ğ¹Ğ»Ğ¾Ğ² Ñ� Ğ°Ğ½Ğ½Ğ¾Ñ‚Ğ°Ñ†Ğ¸Ñ�Ğ¼Ğ¸: {len(train_label_files)}")

# ĞŸÑ€Ğ¾Ğ²ĞµÑ€ĞºĞ° Ğ½Ğ° Ğ´ÑƒĞ±Ğ»Ğ¸ĞºĞ°Ñ‚Ñ‹ Ğ¸ Ğ¿Ğ¾Ğ´Ñ�Ñ‡ĞµÑ‚ Ñ�Ñ‚Ğ°Ñ‚Ğ¸Ñ�Ñ‚Ğ¸ĞºĞ¸
total_boxes = 0
empty_images_count = 0
objects_per_image = []
box_areas = []

for label_file in tqdm(train_label_files, desc="Ğ�Ğ½Ğ°Ğ»Ğ¸Ğ· Ğ°Ğ½Ğ½Ğ¾Ñ‚Ğ°Ñ†Ğ¸Ğ¹"):
    with open(label_file, 'r') as f:
        lines = f.readlines()
        
    # ĞŸÑ€Ğ¾Ğ²ĞµÑ€ĞºĞ° Ğ½Ğ° Ğ´ÑƒĞ±Ğ»Ğ¸ĞºĞ°Ñ‚Ñ‹
    unique_lines = set(lines)
    if len(unique_lines) < len(lines):
        print(f"âš ï¸� Ğ�Ğ°Ğ¹Ğ´ĞµĞ½Ñ‹ Ğ´ÑƒĞ±Ğ»Ğ¸ĞºĞ°Ñ‚Ñ‹ Ğ² Ñ„Ğ°Ğ¹Ğ»Ğµ: {label_file.name}")
    
    if not unique_lines:
        empty_images_count += 1
        objects_per_image.append(0)
    else:
        num_objects = len(unique_lines)
        total_boxes += num_objects
        objects_per_image.append(num_objects)
        for line in unique_lines:
            _, _, _, w, h = map(float, line.strip().split())
            box_areas.append(w * h)

print(f"\nĞ�Ğ½Ğ°Ğ»Ğ¸Ğ· Ğ·Ğ°Ğ²ĞµÑ€ÑˆĞµĞ½.")
print(f"Ğ�Ğ±Ñ‰ĞµĞµ ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ bounding box'Ğ¾Ğ²: {total_boxes}")
print(f"Ğ”Ğ¾Ğ»Ñ� Ğ¿ÑƒÑ�Ñ‚Ñ‹Ñ… ĞºĞ°Ğ´Ñ€Ğ¾Ğ² (Ğ±ĞµĞ· Ñ‡Ğ°ĞµĞº): {empty_images_count / len(train_label_files):.2%}")


fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.histplot(objects_per_image, discrete=True, ax=axes[0])
axes[0].set_title('Ğ Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ğµ Ñ‡Ğ¸Ñ�Ğ»Ğ° Ñ‡Ğ°ĞµĞº Ğ½Ğ° Ğ¸Ğ·Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ğ¸')
axes[0].set_xlabel('ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ğ¾Ğ±ÑŠĞµĞºÑ‚Ğ¾Ğ²')
axes[0].set_ylabel('Ğ§Ğ°Ñ�Ñ‚Ğ¾Ñ‚Ğ°')

sns.histplot(box_areas, bins=50, ax=axes[1])
axes[1].set_title('Ğ Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ğµ Ğ¾Ñ‚Ğ½Ğ¾Ñ�Ğ¸Ñ‚ĞµĞ»ÑŒĞ½Ğ¾Ğ¹ Ğ¿Ğ»Ğ¾Ñ‰Ğ°Ğ´Ğ¸ Ğ±Ğ¾ĞºÑ�Ğ¾Ğ²')
axes[1].set_xlabel('ĞŸĞ»Ğ¾Ñ‰Ğ°Ğ´ÑŒ (w * h, Ğ½Ğ¾Ñ€Ğ¼Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ½Ğ°Ñ�)')
axes[1].set_ylabel('Ğ§Ğ°Ñ�Ñ‚Ğ¾Ñ‚Ğ°')
plt.show()

def display_image_with_boxes(img_path, lbl_path):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    
    if lbl_path.exists():
        with open(lbl_path, 'r') as f:
            for line in f.readlines():
                _, cx, cy, bw, bh = map(float, line.strip().split())
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(f"ĞŸÑ€Ğ¸Ğ¼ĞµÑ€: {img_path.name}")
    plt.axis('off')
    plt.show()

print("\nğŸ–¼ï¸� ĞŸÑ€Ğ¸Ğ¼ĞµÑ€ Ğ¸Ğ·Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ñ� Ñ� Ğ°Ğ½Ğ½Ğ¾Ñ‚Ğ°Ñ†Ğ¸Ñ�Ğ¼Ğ¸:")
first_non_empty_idx = next(i for i, count in enumerate(objects_per_image) if count > 0)
display_image_with_boxes(train_image_files[first_non_empty_idx], train_label_files[first_non_empty_idx])



def yolo_to_pascal(boxes, img_height, img_width):
    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)

    boxes[:, 0] *= img_width
    boxes[:, 1] *= img_height
    boxes[:, 2] *= img_width
    boxes[:, 3] *= img_height

    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    return np.stack([x1, y1, x2, y2], axis=1)

class SeagullDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        lbl_path = self.label_paths[idx]

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]

        boxes = []
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                for line in f.readlines():
                    boxes.append(list(map(float, line.strip().split()))[1:])

        target = {}
        if boxes:
            pascal_boxes = yolo_to_pascal(np.array(boxes), img_h, img_w)
            labels = [1] * len(pascal_boxes)
        else:
            pascal_boxes = np.empty((0, 4), dtype=np.float32)
            labels = []

        if self.transform:
            transformed = self.transform(image=image, bboxes=pascal_boxes, labels=labels)
            image = transformed['image'] 
            
            target_boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            if len(transformed['bboxes']) == 0:
                 target_boxes = torch.empty((0, 4), dtype=torch.float32)
            
            target['boxes'] = target_boxes
            target['labels'] = torch.tensor(transformed['labels'], dtype=torch.int64)
        else:
            target['boxes'] = torch.tensor(pascal_boxes, dtype=torch.float32)
            target['labels'] = torch.tensor(labels, dtype=torch.int64)


        return image, target

def get_transforms(train=True):
    if train:
        return A.Compose([
            A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.5, 1.0), p=0.9),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=10, min_visibility=0.2))
    else:
        return A.Compose([
            A.Resize(height=IMG_SIZE, width=IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

from sklearn.model_selection import train_test_split
if 'train_image_files' in globals() and train_image_files:
    train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
        train_image_files, train_label_files, test_size=0.2, random_state=SEED, shuffle=True
    )

    train_dataset = SeagullDataset(train_imgs, train_lbls, transform=get_transforms(train=True))
    val_dataset = SeagullDataset(val_imgs, val_lbls, transform=get_transforms(train=False))

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True
    )

    print(f"âœ… Ğ¡Ğ¾Ğ·Ğ´Ğ°Ğ½Ñ‹ DataLoader'Ñ‹: {len(train_loader)} train-Ğ±Ğ°Ñ‚Ñ‡ĞµĞ¹, {len(val_loader)} val-Ğ±Ğ°Ñ‚Ñ‡ĞµĞ¹.")
else:
    print("âš ï¸� ĞŸÑ€Ğ¾Ğ¿ÑƒÑ�ĞºĞ°ĞµĞ¼ Ñ�Ğ¾Ğ·Ğ´Ğ°Ğ½Ğ¸Ğµ DataLoader'Ğ¾Ğ², Ñ‚Ğ°Ğº ĞºĞ°Ğº Ñ‚Ñ€ĞµĞ½Ğ¸Ñ€Ğ¾Ğ²Ğ¾Ñ‡Ğ½Ñ‹Ğµ Ñ„Ğ°Ğ¹Ğ»Ñ‹ Ğ½Ğµ Ğ±Ñ‹Ğ»Ğ¸ Ğ½Ğ°Ğ¹Ğ´ĞµĞ½Ñ‹.")



from torch import nn
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import pytorch_lightning as pl
import timm

class TimmFPNBackbone(nn.Module):
    def __init__(self, backbone_name: str, out_indices=(1, 2, 3), fpn_out_channels=256):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            features_only=True,
            pretrained=True,
            out_indices=out_indices
        )
        in_channels_list = self.backbone.feature_info.channels()
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=fpn_out_channels,
        )
        self.out_channels = fpn_out_channels

    def forward(self, x):
        features = self.backbone(x)
        features_dict = {str(i): f for i, f in enumerate(features)}
        return self.fpn(features_dict)

class SeagullDetector(pl.LightningModule):
    def __init__(self, backbone_name='convnext_base', lr=1e-4, weight_decay=1e-4, total_steps=0):
        super().__init__()
        self.save_hyperparameters()

        backbone_with_fpn = TimmFPNBackbone(self.hparams.backbone_name, out_indices=(1, 2, 3))

        anchor_sizes = ((64,), (128,), (256,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

        rpn_anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios
        )

        self.model = FasterRCNN(
            backbone_with_fpn,
            num_classes=NUM_CLASSES,
            rpn_anchor_generator=rpn_anchor_generator, # <--- ĞŸĞ•Ğ Ğ•Ğ”Ğ�Ğ•Ğœ Ğ•Ğ“Ğ� Ğ¡Ğ®Ğ”Ğ�
            box_roi_pool=None
        )

        self.val_map = MeanAveragePrecision(box_format='xyxy', class_metrics=False)

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.forward(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        for k, v in loss_dict.items():
            self.log(f'train_{k}', v, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.forward(images)
        self.val_map.update(preds, targets)

    def on_validation_epoch_end(self):
        map_metrics = self.val_map.compute()
        self.log_dict(map_metrics, prog_bar=True, logger=True)
        self.val_map.reset()

    def configure_optimizers(self):
        return get_optimizer_and_scheduler(self)

try:
    temp_model = SeagullDetector()
    total_params = sum(p.numel() for p in temp_model.parameters())
    trainable_params = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)

    print(f"âœ… ĞœĞ¾Ğ´ĞµĞ»ÑŒ ÑƒÑ�Ğ¿ĞµÑˆĞ½Ğ¾ Ñ�Ğ¾Ğ·Ğ´Ğ°Ğ½Ğ° Ğ½Ğ° Ğ¾Ñ�Ğ½Ğ¾Ğ²Ğµ {temp_model.hparams.backbone_name}")
    print(f"Ğ�Ğ±Ñ‰ĞµĞµ Ñ‡Ğ¸Ñ�Ğ»Ğ¾ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ğ¾Ğ²: {total_params / 1e6:.2f}M")
    print(f"Ğ§Ğ¸Ñ�Ğ»Ğ¾ Ğ¾Ğ±ÑƒÑ‡Ğ°ĞµĞ¼Ñ‹Ñ… Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ğ¾Ğ²: {trainable_params / 1e6:.2f}M")
    del temp_model
except Exception as e:
    print(f"â�Œ Ğ�Ğµ ÑƒĞ´Ğ°Ğ»Ğ¾Ñ�ÑŒ Ñ�Ğ¾Ğ·Ğ´Ğ°Ñ‚ÑŒ Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ Ğ´Ğ»Ñ� Ğ¿Ğ¾Ğ´Ñ�Ñ‡ĞµÑ‚Ğ° Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ğ¾Ğ². Ğ�ÑˆĞ¸Ğ±ĞºĞ°: {e}")



from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

def get_optimizer_and_scheduler(pl_module):
    # Ğ�Ğ¿Ñ‚Ğ¸Ğ¼Ğ¸Ğ·Ğ°Ñ‚Ğ¾Ñ€ AdamW
    optimizer = AdamW(
        pl_module.parameters(), 
        lr=pl_module.hparams.lr, 
        weight_decay=pl_module.hparams.weight_decay
    )
    
    # Scheduler OneCycleLR Ğ´Ğ»Ñ� Ğ±Ñ‹Ñ�Ñ‚Ñ€Ğ¾Ğ³Ğ¾ Ñ�Ñ…Ğ¾Ğ¶Ğ´ĞµĞ½Ğ¸Ñ�
    scheduler = OneCycleLR(
        optimizer,
        max_lr=pl_module.hparams.lr,
        total_steps=pl_module.hparams.total_steps,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0,
    )
    
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step", 
            "frequency": 1
        },
    }

print("âœ… Ğ¤ÑƒĞ½ĞºÑ†Ğ¸Ñ� Ğ´Ğ»Ñ� Ñ�Ğ¾Ğ·Ğ´Ğ°Ğ½Ğ¸Ñ� Ğ¾Ğ¿Ñ‚Ğ¸Ğ¼Ğ¸Ğ·Ğ°Ñ‚Ğ¾Ñ€Ğ° Ğ¸ scheduler'Ğ° Ğ¾Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ°.")



MAX_EPOCHS = 25


total_steps = len(train_loader) * MAX_EPOCHS
model = SeagullDetector(
    lr=1e-4, 
    weight_decay=1e-3, 
    total_steps=total_steps
)


checkpoint_callback = ModelCheckpoint(
    monitor='map',
    dirpath='checkpoints/',
    filename='best-seagull-detector-{epoch:02d}-{map:.4f}',
    save_top_k=1,
    mode='max'
)

early_stopping_callback = EarlyStopping(
    monitor='map',
    patience=5,
    mode='max'
)

lr_monitor = LearningRateMonitor(logging_interval='step')

trainer = pl.Trainer(
    accelerator='gpu' if DEVICE == 'cuda' else 'cpu',
    devices=1,
    max_epochs=MAX_EPOCHS,
    precision="16-mixed", 
    logger=logger,
    callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
    deterministic=False, 
    log_every_n_steps=10
)

print("ğŸš€ Ğ—Ğ°Ğ¿ÑƒÑ�ĞºĞ°ĞµĞ¼ Ñ‚Ñ€ĞµĞ½Ğ¸Ñ€Ğ¾Ğ²ĞºÑƒ Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸...")
# Ğ—Ğ°Ğ¿ÑƒÑ�ĞºĞ°ĞµĞ¼ Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ğµ
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
print("âœ… Ğ¢Ñ€ĞµĞ½Ğ¸Ñ€Ğ¾Ğ²ĞºĞ° Ğ·Ğ°Ğ²ĞµÑ€ÑˆĞµĞ½Ğ°!")

# ĞŸÑƒÑ‚ÑŒ Ğº Ğ»ÑƒÑ‡ÑˆĞµĞ¹ Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸
best_model_path = checkpoint_callback.best_model_path
print(f"Ğ›ÑƒÑ‡ÑˆĞ°Ñ� Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½ĞµĞ½Ğ° Ğ²: {best_model_path}")


print("ğŸ“ˆ ĞŸÑ€Ğ¾Ğ²Ğ¾Ğ´Ğ¸Ğ¼ Ñ„Ğ¸Ğ½Ğ°Ğ»ÑŒĞ½ÑƒÑ� Ğ¾Ñ†ĞµĞ½ĞºÑƒ Ğ½Ğ° Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¾Ğ½Ğ½Ğ¾Ğ¼ Ñ�ĞµÑ‚Ğµ...")

if not best_model_path:
    print("Ğ�Ğµ Ğ½Ğ°Ğ¹Ğ´ĞµĞ½ Ğ¿ÑƒÑ‚ÑŒ Ğº Ğ»ÑƒÑ‡ÑˆĞµĞ¹ Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸.")
else:
    model = SeagullDetector.load_from_checkpoint(best_model_path)
    
    val_results = trainer.validate(model, dataloaders=val_loader)


def pascal_to_yolo(boxes, img_height, img_width):
    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    dw = 1.0 / img_width
    dh = 1.0 / img_height
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    cx *= dw
    w *= dw
    cy *= dh
    h *= dh
    return np.stack([cx, cy, w, h], axis=1)

class SeagullTestDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image)
        return transformed['image'], str(img_path)

if TEST_IMG_DIR.exists():
    test_image_files = sorted([p for p in TEST_IMG_DIR.glob('*.jpg')])
    print(f"Ğ�Ğ°Ğ¹Ğ´ĞµĞ½Ğ¾ {len(test_image_files)} Ñ‚ĞµÑ�Ñ‚Ğ¾Ğ²Ñ‹Ñ… Ğ¸Ğ·Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ğ¹.")

    test_transform = A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    test_dataset = SeagullTestDataset(test_image_files, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    if 'model' not in globals() and 'best_model_path' in globals() and best_model_path:
        print(f"Ğ—Ğ°Ğ³Ñ€ÑƒĞ·ĞºĞ° Ğ»ÑƒÑ‡ÑˆĞµĞ¹ Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸ Ğ¸Ğ· {best_model_path}")
        model = SeagullDetector.load_from_checkpoint(best_model_path)

    model.to(DEVICE)
    model.eval()

    results = []
    CONF_THRESHOLD = 0.5

    for images, img_paths in tqdm(test_loader, desc="Predicting"):
        images = images.to(DEVICE)
        with torch.no_grad():
            preds = model(images)

        for i, pred in enumerate(preds):
            img_path_str = img_paths[i]
            
            orig_img = cv2.imread(img_path_str)
            orig_h, orig_w = orig_img.shape[:2]

            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            valid_boxes = boxes[scores >= CONF_THRESHOLD]
            
            if len(valid_boxes) == 0:
                final_bbox_str = '-1'
            else:
                valid_boxes[:, [0, 2]] *= orig_w / IMG_SIZE
                valid_boxes[:, [1, 3]] *= orig_h / IMG_SIZE
                yolo_boxes = pascal_to_yolo(valid_boxes, orig_h, orig_w)
                box_strings = [f"0 {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}" for yolo_box in yolo_boxes]
                final_bbox_str = " ".join(box_strings)
                
            results.append({
                'index': len(results), 
                'filename': os.path.basename(img_path_str), 
                'bbox': final_bbox_str
            })

    submission_df = pd.DataFrame(results)

    print(f"\nâœ… Ğ¤Ğ°Ğ¹Ğ» submission.csv ÑƒÑ�Ğ¿ĞµÑˆĞ½Ğ¾ Ñ�Ğ¾Ğ·Ğ´Ğ°Ğ½ Ğ¸ Ñ�Ğ¾Ğ´ĞµÑ€Ğ¶Ğ¸Ñ‚ {len(submission_df)} Ñ�Ñ‚Ñ€Ğ¾Ğº.")
    submission_df.to_csv('submission.csv', index=False)
    
    print("\nĞŸÑ€Ğ¸Ğ¼ĞµÑ€ Ñ�Ğ³ĞµĞ½ĞµÑ€Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ½Ğ¾Ğ³Ğ¾ Ñ„Ğ°Ğ¹Ğ»Ğ°:")
    print(submission_df.head())
else:
    print(f"âš ï¸� Ğ¢ĞµÑ�Ñ‚Ğ¾Ğ²Ğ°Ñ� Ğ´Ğ¸Ñ€ĞµĞºÑ‚Ğ¾Ñ€Ğ¸Ñ� Ğ½Ğµ Ğ½Ğ°Ğ¹Ğ´ĞµĞ½Ğ°: {TEST_IMG_DIR}")




