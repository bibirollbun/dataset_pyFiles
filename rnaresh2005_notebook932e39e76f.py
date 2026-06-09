






pip install -U ultralytics









# Python cell
from pathlib import Path
import os, random, time, json, math
import numpy as np, pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision.transforms.functional as TF

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# ultralytics
from ultralytics import YOLO

# global paths (update if needed)
WORK_DIR = Path("/kaggle/working/vindr")
IMG_DIR = WORK_DIR / "images"
LAB_DIR = WORK_DIR / "labels"
MODEL_DIR = Path("/kaggle/working/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = Path("/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train.csv")
SAMPLE_SUB = Path("/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/sample_submission.csv")
TEST_DICOM_DIR = Path("/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/test")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)



pip install -U ultralytics


# Python cell
from pathlib import Path
import os, random, time, json, math
import numpy as np, pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision.transforms.functional as TF

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# ultralytics
from ultralytics import YOLO

# global paths (update if needed)
WORK_DIR = Path("/kaggle/working/vindr")
IMG_DIR = WORK_DIR / "images"
LAB_DIR = WORK_DIR / "labels"
MODEL_DIR = Path("/kaggle/working/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = Path("/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train.csv")
SAMPLE_SUB = Path("/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/sample_submission.csv")
TEST_DICOM_DIR = Path("/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/test")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)



# import shutil
# import os

# # Source (input) and destination (working/output)
# src = "/kaggle/input/chest-xray/kaggle/working/vindr"       # change this to your dataset
# dst = "/kaggle/working/vindr"     # destination folder

# # If destination exists, remove it first (optional)
# if os.path.exists(dst):
#     shutil.rmtree(dst)

# # Copy entire folder
# shutil.copytree(src, dst)

# print(f"✅ Copied {src} → {dst}")



# Build class list (match earlier)
class_names = [
    "Aortic_enlargement","Atelectasis","Calcification","Cardiomegaly",
    "Consolidation","ILD","Infiltration","Lung_Opacity","Nodule_Mass",
    "Other_lesion","Pleural_effusion","Pleural_thickening",
    "Pneumothorax","Pulmonary_fibrosis","No_finding"
]
NUM_CLASSES = len(class_names)

# Read train.csv and build multi-hot label dict
df = pd.read_csv(TRAIN_CSV)
# group by image_id
targets = {}
for img_id, g in df.groupby("image_id"):
    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    for cid in g['class_id'].values:
        vec[int(cid)] = 1.0
    targets[img_id] = vec

# Some images may be missing in df (no annotation) -> treat as No_finding
# Ensure that for converted images, we have label vectors
train_img_dir = IMG_DIR / "train"
val_img_dir   = IMG_DIR / "val"

train_ids = [p.stem for p in train_img_dir.glob("*.jpg")]
val_ids   = [p.stem for p in val_img_dir.glob("*.jpg")]

# if an image not in targets -> treat as no finding (class 14 = No_finding)
for img in train_ids + val_ids:
    if img not in targets:
        vec = np.zeros(NUM_CLASSES, dtype=np.float32)
        vec[14] = 1.0
        targets[img] = vec

# Dataset class
class MultiLabelCXRDataset(Dataset):
    def __init__(self, image_dir, img_ids, targets_dict, transform=None):
        self.image_dir = Path(image_dir)
        self.img_ids = img_ids
        self.targets = targets_dict
        self.transform = transform

    def __len__(self): return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = self.image_dir / f"{img_id}.jpg"
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.targets[img_id], dtype=torch.float32)
        return img, label, img_id



BATCH = 32

train_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_ds = MultiLabelCXRDataset(train_img_dir, train_ids, targets, transform=train_tfms)
val_ds   = MultiLabelCXRDataset(val_img_dir, val_ids, targets, transform=val_tfms)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
print("Train / Val sizes:", len(train_ds), len(val_ds))



import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from torch.amp import autocast, GradScaler

# Model
NUM_CLASSES = 15  # change as needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_cls = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model_cls.fc = nn.Linear(model_cls.fc.in_features, NUM_CLASSES)
model_cls = model_cls.to(device)

# Loss, optimizer, scheduler
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model_cls.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# AMP
scaler = GradScaler("cuda")



import os

labels_dir = "/kaggle/working/vindr/labels"

for split in ["train", "val", "test"]:
    folder = os.path.join(labels_dir, split)
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            path = os.path.join(folder, file)
            with open(path, "r") as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # force all labels to class 0
                    parts[0] = "0"
                    new_lines.append(" ".join(parts) + "\n")
            
            # overwrite
            with open(path, "w") as f:
                f.writelines(new_lines)

print("✅ All labels remapped to single class 0")



import yaml
yolo_yaml = {
    "path": str(WORK_DIR),   # base path
    "train": "images/train",
    "val":   "images/val",
    "test":  "images/test",
    "nc": 1,
    "names": ["abnormal"]
}
YAML_PATH = WORK_DIR / "yolov8_abnormal.yaml"
with open(YAML_PATH, 'w') as f:
    yaml.dump(yolo_yaml, f)
print("Saved YAML:", YAML_PATH)



pip install -U ultralytics


from ultralytics import YOLO

yolo_model = YOLO("yolov8n.pt")   # lightweight test model
results = yolo_model.train(
    data=str(YAML_PATH),
    epochs=20,
    imgsz=640,
    batch=8,
    project=str(WORK_DIR/"yolov8_runs"),
    name="abnormal_singleclass"
)





