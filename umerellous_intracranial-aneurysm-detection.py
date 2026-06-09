import os
import random
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import GradScaler, autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
import ast
import shutil
from sklearn.model_selection import train_test_split
from collections import defaultdict
import kaggle_evaluation.rsna_inference_server
from sklearn.metrics import classification_report, f1_score, roc_auc_score
import polars as pl
import kagglehub
import json
import gc


import logging
logging.getLogger('pydicom').setLevel(logging.ERROR)


import warnings
warnings.filterwarnings("ignore") 


# Load detection labels
df_labels = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv")


df_labels.head()


class CFG:
    seed = 42
    img_size = 224
    model_name = 'tf_efficientnetv2_s'
    lr = 1e-4
    epochs = 10
    batch_size = 16
    num_workers = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_csv_path = '/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv'
    train_localizers_csv_path = '/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv'
    series_data_path = '/kaggle/input/rsna-intracranial-aneurysm-detection/series'
    output_dir = '/kaggle/working/'
    val_split = 0.2

os.makedirs(CFG.output_dir, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CFG.seed)


def load_dicom_as_array(dcm_path, window=(0, 100), output_size=CFG.img_size):
    dcm = pydicom.dcmread(dcm_path)
    arr = dcm.pixel_array.astype(np.float32)

    # Apply rescale slope/intercept if present
    slope = getattr(dcm, "RescaleSlope", 1.0)
    intercept = getattr(dcm, "RescaleIntercept", 0.0)
    arr = arr * slope + intercept

    # Handle dimensionality
    if arr.ndim == 3:
        arr = arr[arr.shape[0] // 2]
    elif arr.ndim == 4:
        arr = arr[arr.shape[0] // 2]
        if arr.ndim == 3 and arr.shape[-1] != 3:
            arr = arr[..., 0]

    # Clip to window and normalize to [0, 1]
    w_min, w_max = window
    arr = np.clip(arr, w_min, w_max)
    arr = (arr - w_min) / (w_max - w_min + 1e-5)

    # Convert to 3-channel RGB and resize
    arr = np.stack([arr] * 3, axis=-1)
    arr = Image.fromarray((arr * 255).astype(np.uint8)).resize((output_size, output_size))
    return np.array(arr)


class RSNADataset(Dataset):
    def __init__(self, df, series_path, transform=None):
        self.df = df.reset_index(drop=True)
        self.series_path = series_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        series_id = row['series_id']
        series_folder = os.path.join(self.series_path, str(series_id))
        dcm_files = sorted(glob(os.path.join(series_folder, '*.dcm')))
        
        # Fallback for corrupted/missing DICOMs
        if not dcm_files:
            dummy_image = torch.zeros(3, CFG.img_size, CFG.img_size) 
            dummy_label = torch.tensor(0.0, dtype=torch.float32) 
            dummy_loc = torch.full((13,), -1.0)  # 13 locations
            return dummy_image, dummy_label, dummy_loc
    
        mid_idx = (len(dcm_files) - 1) // 2
        try:
            image = load_dicom_as_array(dcm_files[mid_idx])
        except Exception as e:
            dummy_image = torch.zeros(3, CFG.img_size, CFG.img_size)
            dummy_label = torch.tensor(0.0, dtype=torch.float32)
            dummy_loc = torch.full((13,), -1.0)
            return dummy_image, dummy_label, dummy_loc
    
        # Transform or basic normalization
        if self.transform:
            image = self.transform(image=image)['image']
        else:
            image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
    
        # Binary class label
        label = torch.tensor(row['present'], dtype=torch.float32)
    
        # Per-location labels
        loc_cols = [
            "Left Infraclinoid Internal Carotid Artery",
            "Right Infraclinoid Internal Carotid Artery",
            "Left Supraclinoid Internal Carotid Artery",
            "Right Supraclinoid Internal Carotid Artery",
            "Left Middle Cerebral Artery",
            "Right Middle Cerebral Artery",
            "Anterior Communicating Artery",
            "Left Anterior Cerebral Artery",
            "Right Anterior Cerebral Artery",
            "Left Posterior Communicating Artery",
            "Right Posterior Communicating Artery",
            "Basilar Tip",
            "Other Posterior Circulation",
        ]
        loc_values = row[loc_cols].values.astype(np.float32)
        loc_values = np.nan_to_num(loc_values, nan=-1.0)
        loc = torch.tensor(loc_values)
    
        return image, label, loc


train_transform = A.Compose([
    A.Resize(CFG.img_size, CFG.img_size),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(CFG.img_size, CFG.img_size),
    A.Normalize(),
    ToTensorV2()
])


class DualHeadModel(nn.Module):
    def __init__(self, backbone_name=CFG.model_name):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        in_features = self.backbone.num_features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
        
        self.localization_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 13)
        )

    def forward(self, x):
        features = self.backbone.forward_features(x)
        pooled = self.pool(features).view(features.size(0), -1)
        class_out = self.classifier_head(pooled)
        loc_out = self.localization_head(pooled)
        return class_out[:, 0], loc_out


class DualLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_no_reduce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, class_pred, class_true, loc_pred, loc_true):
        class_loss = self.bce(class_pred, class_true)
        mask = (loc_true != -1).float()
        loc_loss_raw = self.bce_no_reduce(loc_pred, loc_true)
        masked_loc_loss = loc_loss_raw * mask
        loc_loss = masked_loc_loss.sum() / (mask.sum() + 1e-6)
        total_loss = class_loss + self.alpha * loc_loss
        return total_loss


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=CFG.epochs):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, class_targets, loc_targets in train_loader:
            inputs = inputs.to(device)
            class_targets = class_targets.to(device)
            loc_targets = loc_targets.to(device)

            optimizer.zero_grad()
            class_outputs, loc_outputs = model(inputs)

            loss = criterion(class_outputs, class_targets, loc_outputs, loc_targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, class_targets, loc_targets in val_loader:
                inputs = inputs.to(device)
                class_targets = class_targets.to(device)
                loc_targets = loc_targets.to(device)

                class_outputs, loc_outputs = model(inputs)
                loss = criterion(class_outputs, class_targets, loc_outputs, loc_targets)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("Saved best model!")
            torch.save(model.state_dict(), 'best_model.pth')


train_df = pd.read_csv(CFG.train_csv_path)
train_df.rename(columns={'Aneurysm Present': 'present'}, inplace=True)
train_df.rename(columns={'SeriesInstanceUID': 'series_id'}, inplace=True)


df_train, df_val = train_test_split(
    train_df,
    test_size=0.2,
    stratify=train_df['present'],
    random_state=42
)

df_val, df_test = train_test_split(
    df_val,
    test_size=0.5,
    stratify=df_val['present'],
    random_state=42
)


train_dataset = RSNADataset(df_train, series_path=CFG.series_data_path, transform=train_transform)
val_dataset   = RSNADataset(df_val,   series_path=CFG.series_data_path, transform=val_transform)
test_dataset  = RSNADataset(df_test,  series_path=CFG.series_data_path, transform=val_transform)


train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers)
val_loader   = DataLoader(val_dataset,   batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)
test_loader  = DataLoader(test_dataset,  batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DualHeadModel().to(device)


criterion = DualLoss(alpha=1.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


# train_model(
#     model=model,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     criterion=criterion,
#     optimizer=optimizer,
#     scheduler=scheduler,
#     device=device
# )


model = DualHeadModel()
model.load_state_dict(torch.load('/kaggle/input/m/umerellous/rsna-intracranial-aneurysm-detection/keras/default/1/best_model.pth',map_location=device))
model.to(CFG.device) 
model.eval()
print("Model loaded successfully!")


def evaluate_model(model, test_loader, device):
    model.eval()
    all_present_preds = []
    all_present_targets = []
    all_loc_preds = []
    all_loc_targets = []

    with torch.no_grad():
        for images, present_targets, loc_targets in tqdm(test_loader):
            images = images.to(device)
            present_targets = present_targets.to(device)
            loc_targets = loc_targets.to(device)

            present_preds, loc_preds = model(images)

            # Apply sigmoid to get probabilities
            present_preds = torch.sigmoid(present_preds)
            loc_preds = torch.sigmoid(loc_preds)

            # Store results
            all_present_preds.append(present_preds.cpu())
            all_present_targets.append(present_targets.cpu())
            all_loc_preds.append(loc_preds.cpu())
            all_loc_targets.append(loc_targets.cpu())

    # Stack results
    all_present_preds = torch.cat(all_present_preds).numpy()
    all_present_targets = torch.cat(all_present_targets).numpy()
    all_loc_preds = torch.cat(all_loc_preds).numpy()
    all_loc_targets = torch.cat(all_loc_targets).numpy()

    print("Classification Report:")
    print(classification_report(all_present_targets, (all_present_preds > 0.5).astype(int)))


evaluate_model(model, test_loader, device)


transform = A.Compose([
    A.Resize(CFG.img_size, CFG.img_size),
    A.Normalize(),
    ToTensorV2()
])


LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
]


def predict(series_path: str) -> pd.DataFrame:
    try:
        series_id = os.path.basename(series_path)
        dicom_files = sorted([
            os.path.join(dp, f)
            for dp, _, filenames in os.walk(series_path)
            for f in filenames if f.endswith(".dcm")
        ], key=lambda x: int(pydicom.dcmread(x, stop_before_pixels=True).InstanceNumber))
    
        images = []
        for file in dicom_files:
            dcm = pydicom.dcmread(file)
            img = dcm.pixel_array.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-5)
            if len(img.shape) == 2:
                img = np.stack([img]*3, axis=-1)
            img = transform(image=img)["image"]
            images.append(img)
    
        images = torch.stack(images).to(CFG.device)
        model.to(CFG.device)
        model.eval()
    
        all_present_preds = []
        all_loc_preds = []
    
        with torch.no_grad():
            for img in images:
                img = img.unsqueeze(0)
                present, loc = model(img)
                present = torch.sigmoid(present).cpu().item()
                loc = torch.sigmoid(loc).cpu().numpy().flatten()
                all_present_preds.append(present)
                all_loc_preds.append(loc)
    
        final_present = float(np.max(all_present_preds))
        final_loc = np.max(all_loc_preds, axis=0).tolist()
    
        df = pd.DataFrame([{
            "SeriesInstanceUID": series_id,
            **{label: float(v) for label, v in zip(LABEL_COLS, final_loc)},
            "Aneurysm Present": final_present
        }])
    
        if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
            df.to_parquet("/kaggle/working/submission.parquet", index=False)
            print("Saved submission.parquet")
            
        return df

    except Exception as e:
        print(f"Prediction failed for {series_path}: {e}")
        fallback = pd.DataFrame([{
            **{label: 0.1 for label in LABEL_COLS},
            "Aneurysm Present": 0.1
        }])
        
        return fallback

    finally:
        shared_dir = "/kaggle/shared"
        shutil.rmtree(shared_dir, ignore_errors=True)
        os.makedirs(shared_dir, exist_ok=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


shared_dir = '/kaggle/shared'
if os.path.exists(shared_dir):
    for f in os.listdir(shared_dir):
        file_path = os.path.join(shared_dir, f)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}')


inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway()
    display(pl.read_parquet('/kaggle/working/submission.parquet'))




