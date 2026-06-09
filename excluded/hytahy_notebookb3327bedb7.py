# =================================================================================
# KAGGLE NOTEBOOK: GRAND X-RAY SLAM - CHIẾN LƯỢC "TITAN REGULATOR" v2.1
# Tác giả: Kaggle Grandmaster (Persona)
# Phiên bản: 2.1 - Sửa lỗi TypeError, Albumentations Warnings, và GradScaler Deprecation
# Mục tiêu: Đạt hiệu suất SOTA (>0.96) với siêu kiến trúc EVA-02
# =================================================================================

# --- 1. IMPORT CÁC THƯ VIỆN CẦN THIẾT ---
import os
import gc
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import timm
from tqdm.notebook import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

print("Tất cả thư viện đã được import thành công.")

# --- 2. CẤU HÌNH TOÀN CỤC (CONFIGURATION) ---

class Config:
    DEBUG = False
    MODEL_NAME = "eva02_large_patch14_448.mim_in22k_ft_in1k"
    IMAGE_SIZE = 448
    EPOCHS_FROZEN = 2
    LR_FROZEN = 1e-3
    EPOCHS_FINETUNE = 4
    LR_FINETUNE = 1e-5
    WEIGHT_DECAY_FROZEN = 1e-6
    WEIGHT_DECAY_FINETUNE = 1e-4
    LABEL_SMOOTHING = 0.05
    EARLY_STOPPING_PATIENCE = 2
    DATA_ROOT = "/kaggle/input/grand-xray-slam-division-a"
    TRAIN_CSV = os.path.join(DATA_ROOT, "train1.csv")
    TEST_CSV = os.path.join(DATA_ROOT, "sample_submission_1.csv")
    TRAIN_IMAGE_DIR = os.path.join(DATA_ROOT, "train1")
    TEST_IMAGE_DIR = os.path.join(DATA_ROOT, "test1")
    TARGET_COLS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
                   'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 
                   'Lung Opacity', 'No Finding', 'Pleural Effusion', 
                   'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
    NUM_CLASSES = len(TARGET_COLS)
    BATCH_SIZE = 2
    N_SPLITS = 5
    VALIDATION_FOLD = 0
    USE_AMP = True
    NUM_WORKERS = 2
    SEED = 42
    ACCUMULATION_STEPS = 8 # Mô phỏng batch size hiệu quả là 4 * 4 = 16
    USE_CHECKPOINTING = True

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Sử dụng thiết bị: {Config.DEVICE}")
print(f"Sử dụng mô hình: {Config.MODEL_NAME} với kích thước ảnh {Config.IMAGE_SIZE}")
print(f"Chế độ DEBUG: {'BẬT' if Config.DEBUG else 'TẮT'}")

# --- 3. HÀM TIỆN ÍCH & PIPELINE DỮ LIỆU ---

def set_seed(seed=Config.SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed()

def get_transforms(size, is_train=True):
    """
    Pipeline tăng cường dữ liệu đã được cập nhật để sửa lỗi warnings.
    """
    if is_train:
        return A.Compose([
            A.Resize(size, size, interpolation=cv2.INTER_AREA),
            A.HorizontalFlip(p=0.5),
            # Sửa lỗi warning: Sử dụng các tham số đúng cho Affine
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), rotate=(-20, 20), p=0.75, 
                     cval=0, interpolation=cv2.INTER_AREA),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.FancyPCA(),
                A.HueSaturationValue(),
            ], p=0.7),
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),
            ], p=0.5),
            A.GaussNoise(p=0.5),
            # # Sửa lỗi warning: Sử dụng Cutout, là tên mới của CoarseDropout
            # A.Cutout(num_holes=12, max_h_size=int(size*0.12), max_w_size=int(size*0.12), 
            #          fill_value=0, p=0.75),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(size, size, interpolation=cv2.INTER_AREA),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

class ChestXRayDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, is_test=False):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test
        # Lấy trước nhãn để tối ưu hóa
        if not is_test:
            self.labels = self.df[Config.TARGET_COLS].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['Image_name'])
        try:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Lỗi khi đọc ảnh: {img_path}. Lỗi: {e}")
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        if self.is_test:
            return image
        else:
            # SỬA LỖI TYPEERROR: Lấy trực tiếp từ self.labels đã được ép kiểu
            labels = self.labels[idx]
            return image, torch.from_numpy(labels)

# --- 4. MÔ HÌNH & CÁC LỚP PHÒNG THỦ ---

class TitanModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True, checkpointing=False):
        super().__init__()
        # Tạo model mà không ép tham số checkpointing
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=num_classes
        )

        # Nếu model có hỗ trợ checkpointing thì bật qua method
        if checkpointing and hasattr(self.model, "set_grad_checkpointing"):
            self.model.set_grad_checkpointing(enable=True)
    def forward(self, x):
        return self.model(x)
        
    def freeze_backbone(self):
        print("Đóng băng các lớp backbone...")
        for name, param in self.model.named_parameters():
            if not name.startswith('head.'):
                param.requires_grad = False
                
    def unfreeze_backbone(self):
        print("Mở băng toàn bộ mô hình để fine-tune...")
        for param in self.model.parameters():
            param.requires_grad = True

class SmoothedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None, smoothing=0.1):
        super(SmoothedBCEWithLogitsLoss, self).__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    def forward(self, outputs, labels):
        labels_smoothed = torch.where(labels > 0.5, 1.0 - self.smoothing, self.smoothing)
        loss = self.bce(outputs, labels_smoothed)
        return loss.mean()

class EarlyStopping:
    def __init__(self, patience=2, delta=0.0001):
        self.patience = patience; self.counter = 0; self.best_score = None
        self.early_stop = False; self.delta = delta
    def __call__(self, val_score, model, path):
        if self.best_score is None:
            self.best_score = val_score; self.save_checkpoint(model, path)
        elif val_score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = val_score; self.save_checkpoint(model, path); self.counter = 0
    def save_checkpoint(self, model, path):
        print(f'Validation score improved ({self.best_score:.4f}). Saving model to {path} ...')
        torch.save(model.state_dict(), path)

# --- 5. HÀM HUẤN LUYỆN & ĐÁNH GIÁ ---

def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device, scaler):
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training", leave=False)
    
    # <<< THAY ĐỔI: Logic tích lũy gradient
    for step, (images, labels) in pbar:
        images, labels = images.to(device), labels.to(device)
        
        with torch.amp.autocast(device_type=device.type, enabled=Config.USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Scale loss theo số bước tích lũy
            loss = loss / Config.ACCUMULATION_STEPS
            
        scaler.scale(loss).backward()
        
        # Chỉ cập nhật trọng số sau ACCUMULATION_STEPS
        if (step + 1) % Config.ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()
        
        running_loss += loss.item() * Config.ACCUMULATION_STEPS # Nhân lại để log loss đúng
        pbar.set_postfix(loss=f'{running_loss / (step + 1):.4f}')
        
    return running_loss / len(dataloader)

def validate_one_epoch(model, dataloader, device):
    model.eval(); all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=Config.USE_AMP):
                outputs = model(images)
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds); all_labels = np.concatenate(all_labels)
    return roc_auc_score(all_labels, all_preds, average='macro')

# --- 6. PIPELINE HUẤN LUYỆN CHÍNH ---

print("===== BẮT ĐẦU CHUẨN BỊ DỮ LIỆU =====")
train_df = pd.read_csv(Config.TRAIN_CSV)
# if Config.DEBUG:
print("!!! CHẾ ĐỘ DEBUG ĐANG BẬT, SỬ DỤNG 1000 MẪU !!!")
# train_df = train_df.sample(n=10000, random_state=Config.SEED).reset_index(drop=True)

train_df['Patient_ID'] = train_df['Image_name'].apply(lambda x: x.split('_')[0])

sgkf = StratifiedGroupKFold(n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.SEED)
y_stratify = train_df[Config.TARGET_COLS].sum(axis=1)
train_df['fold'] = -1
for fold, (train_idx, val_idx) in enumerate(sgkf.split(train_df, y_stratify, groups=train_df['Patient_ID'])):
    train_df.loc[val_idx, 'fold'] = fold

train_fold_df = train_df[train_df['fold'] != Config.VALIDATION_FOLD].reset_index(drop=True)
val_fold_df = train_df[train_df['fold'] == Config.VALIDATION_FOLD].reset_index(drop=True)
print(f"Huấn luyện trên {len(train_fold_df)} mẫu, kiểm định trên {len(val_fold_df)} mẫu.")

train_dataset = ChestXRayDataset(train_fold_df, Config.TRAIN_IMAGE_DIR, get_transforms(Config.IMAGE_SIZE, is_train=True))
val_dataset = ChestXRayDataset(val_fold_df, Config.TRAIN_IMAGE_DIR, get_transforms(Config.IMAGE_SIZE, is_train=False))
train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE*2, shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)

model = TitanModel(
    Config.MODEL_NAME, 
    Config.NUM_CLASSES,
    checkpointing=Config.USE_CHECKPOINTING
).to(Config.DEVICE)

pos_counts = train_fold_df[Config.TARGET_COLS].sum()
neg_counts = len(train_fold_df) - pos_counts
pos_weight = (neg_counts / pos_counts).values
criterion = SmoothedBCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32).to(Config.DEVICE), 
                                     smoothing=Config.LABEL_SMOOTHING)
# SỬA LỖI GRADSCALER
scaler = torch.amp.GradScaler(device=Config.DEVICE.type, enabled=Config.USE_AMP)

# === GIAI ĐOẠN 1: HUẤN LUYỆN CLASSIFIER (FROZEN) ===
print("\n===== BẮT ĐẦU GIAI ĐOẠN 1: FROZEN BACKBONE =====")
model.freeze_backbone()
# Quan trọng: đảm bảo optimizer chỉ nhận các tham số có thể huấn luyện
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=Config.LR_FROZEN, 
    weight_decay=Config.WEIGHT_DECAY_FROZEN
)
early_stopper = EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE)

for epoch in range(Config.EPOCHS_FROZEN):
    print(f"\n--- Epoch Frozen {epoch + 1}/{Config.EPOCHS_FROZEN} ---")
    train_loss = train_one_epoch(model, train_loader, optimizer, None, criterion, Config.DEVICE, scaler)
    val_auc = validate_one_epoch(model, val_loader, Config.DEVICE)
    print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}")
    early_stopper(val_auc, model, "best_model_frozen.pth")
    if early_stopper.early_stop:
        print("Dừng sớm trong giai đoạn frozen.")
        break

print("\n===== BẮT ĐẦU GIAI ĐOẠN 2: FINE-TUNING =====")
model.load_state_dict(torch.load("best_model_frozen.pth"))
model.unfreeze_backbone()
optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR_FINETUNE, weight_decay=Config.WEIGHT_DECAY_FINETUNE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * Config.EPOCHS_FINETUNE)
early_stopper = EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE, delta=0.0001)

for epoch in range(Config.EPOCHS_FINETUNE):
    print(f"\n--- Epoch Finetune {epoch + 1}/{Config.EPOCHS_FINETUNE} ---")
    train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, Config.DEVICE, scaler)
    val_auc = validate_one_epoch(model, val_loader, Config.DEVICE)
    print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}")
    early_stopper(val_auc, model, "best_model_finetuned.pth")
    if early_stopper.early_stop:
        print("Dừng sớm trong giai đoạn fine-tuning.")
        break
        
gc.collect()
torch.cuda.empty_cache()

print("\n===== BẮT ĐẦU SUY LUẬN TRÊN TẬP TEST =====")
test_df = pd.read_csv(Config.TEST_CSV)
test_dataset = ChestXRayDataset(test_df, Config.TEST_IMAGE_DIR, get_transforms(Config.IMAGE_SIZE, is_train=False), is_test=True)
test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE*2, shuffle=False, num_workers=Config.NUM_WORKERS)

model.load_state_dict(torch.load("best_model_finetuned.pth"))
model.eval()
all_preds = []

with torch.no_grad():
    for images in tqdm(test_loader, desc="Inference"):
        images = images.to(Config.DEVICE)
        with torch.amp.autocast(device_type=Config.DEVICE.type, enabled=Config.USE_AMP):
            preds_orig = torch.sigmoid(model(images))
            images_flipped = torch.flip(images, dims=[3])
            preds_flipped = torch.sigmoid(model(images_flipped))
            avg_preds = (preds_orig + preds_flipped) / 2.0
            all_preds.append(avg_preds.cpu().numpy())

predictions = np.concatenate(all_preds)

print("\n===== TẠO FILE SUBMISSION =====")
submission_df = pd.DataFrame(predictions, columns=Config.TARGET_COLS)
submission_df.insert(0, 'Image_name', test_df['Image_name'])
submission_df.to_csv("submission.csv", index=False)

print("\nĐã tạo file submission.csv thành công!")
print("Một vài hàng đầu của submission.csv:")
print(submission_df.head())

