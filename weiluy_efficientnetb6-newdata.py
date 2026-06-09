import cv2
import numpy as np
import os
import pandas as pd
import random
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations import Compose, Normalize, HorizontalFlip, Rotate, ColorJitter, Resize
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from scipy.optimize import minimize

# ==============================================================================
# 0. 全局设置
# ==============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# ==============================================================================
# 1. 预处理函数
# ==============================================================================
def circle_crop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return image
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return image[y:y + h, x:x + w]

def apply_ben_graham_preprocessing(image, sigmaX=30):
    blurred_image = cv2.GaussianBlur(image, (0, 0), sigmaX)
    return cv2.addWeighted(image, 4, blurred_image, -4, 128)

def preprocess_directory(input_dir, output_dir, image_size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir); print(f"创建目录: {output_dir}")
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for filename in tqdm(image_files, desc=f"处理中 {os.path.basename(input_dir)}"):
        input_path, output_path = os.path.join(input_dir, filename), os.path.join(output_dir, filename)
        image = cv2.imread(input_path)
        if image is None: continue
        processed_image = circle_crop(image)
        processed_image = cv2.resize(processed_image, (image_size, image_size))
        processed_image = apply_ben_graham_preprocessing(processed_image)
        cv2.imwrite(output_path, processed_image)
    print(f"处理完成！处理后的图片已保存至: {output_dir}")

# ==============================================================================
# 2. PyTorch 模型与数据类
# ==============================================================================
def replace_batchnorm_with_groupnorm(module, num_groups=32):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            if num_channels % num_groups == 0:
                setattr(module, name, nn.GroupNorm(num_groups=num_groups, num_channels=num_channels))
        else:
            replace_batchnorm_with_groupnorm(child, num_groups)

class EfficientNetModel(nn.Module):
    def __init__(self, model_name='efficientnet_b6', pretrained_model_path=None):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False)
        replace_batchnorm_with_groupnorm(self.model)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(in_features, 512), nn.Mish(),
            nn.Dropout(p=0.5), nn.Linear(512, 1)
        )
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            print(f"加载自定义预训练权重: {pretrained_model_path}")
            self.model.load_state_dict(torch.load(pretrained_model_path), strict=False)
        elif pretrained_model_path:
             print(f"警告: 预训练模型路径未找到: {pretrained_model_path}")
    def forward(self, x): return self.model(x)

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, labels_df, transform=None, is_test=False):
        self.img_dir, self.labels_df, self.transform, self.is_test = img_dir, labels_df, transform, is_test
    def __len__(self): return len(self.labels_df)
    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_id = row['id_code']
        img_path = os.path.join(self.img_dir, img_id + '.png')
        try:
            img = cv2.imread(img_path)
            if img is None: return self.__getitem__((idx + 1) % len(self))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform: img = self.transform(image=img)["image"]
            if self.is_test: return img
            else:
                label = float(row['diagnosis'])
                return img, label
        except Exception: return self.__getitem__((idx + 1) % len(self))

def get_transforms(data_type='train', image_size=512):
    if data_type == 'train':
        return Compose([Resize(image_size, image_size), Rotate(limit=40, p=0.5), ColorJitter(p=0.3), HorizontalFlip(p=0.5), Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])
    else:
        return Compose([Resize(image_size, image_size), Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])

# ==============================================================================
# 3. 训练与预测函数
# ==============================================================================
def train_fold(model, train_loader, valid_loader, device, num_epochs, lr, fold):
    criterion, optimizer = nn.MSELoss(), optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    best_valid_kappa, patience, epochs_no_improve = 0.0, 5, 0
    save_path = os.path.join(os.getcwd(), f'best_model_fold_{fold}.pth')
    for epoch in range(num_epochs):
        model.train(); running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Fold {fold} - Epoch {epoch+1} 训练"):
            images, labels = images.to(device), labels.to(device).float().view(-1, 1)
            optimizer.zero_grad(); outputs = model(images); loss = criterion(outputs, labels)
            loss.backward(); optimizer.step(); running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        
        model.eval(); all_valid_labels, all_valid_preds = [], []
        with torch.no_grad():
            for images, labels in tqdm(valid_loader, desc=f"Fold {fold} - Epoch {epoch+1} 验证"):
                
                # <--- 关键修正 ---
                # 必须先将数据移动到device，再送入模型
                images = images.to(device)
                outputs = model(images).squeeze()
                # --- 修正结束 ---

                all_valid_labels.extend(labels.cpu().numpy()); all_valid_preds.extend(np.atleast_1d(outputs.cpu().numpy()))
        
        valid_kappa = cohen_kappa_score(all_valid_labels, np.round(all_valid_preds).astype(int), weights='quadratic')
        print(f"\nFold {fold} - Epoch {epoch+1}\n  训练损失: {train_loss:.4f}\n  验证 Kappa: {valid_kappa:.4f}")
        scheduler.step()
        if valid_kappa > best_valid_kappa:
            best_valid_kappa, epochs_no_improve = valid_kappa, 0
            torch.save(model.state_dict(), save_path); print(f"  -> 新最佳模型已保存! Kappa: {best_valid_kappa:.4f}")
        else: epochs_no_improve += 1
        if epochs_no_improve >= patience: print(f"触发早停。"); break
    
    print(f"\nFold {fold} 训练结束。最佳模型保存在: {save_path}")
    return save_path, best_valid_kappa

class OptimizedRounder:
    def __init__(self): self.coef_ = 0
    def _kappa_loss(self, coef, X, y): return -cohen_kappa_score(y, self.predict(X, coef), weights='quadratic')
    def fit(self, X, y):
        loss_partial = lambda coef: self._kappa_loss(coef, X, y)
        self.coef_ = minimize(loss_partial, [0.5, 1.5, 2.5, 3.5], method='nelder-mead')['x']
        print(f"优化后的阈值: {self.coef_}")
    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]: X_p[i] = 0
            elif pred < coef[1]: X_p[i] = 1
            elif pred < coef[2]: X_p[i] = 2
            elif pred < coef[3]: X_p[i] = 3
            else: X_p[i] = 4
        return X_p.astype(int)

def get_predictions(loader, model, device):
    model.eval(); all_preds = []
    with torch.no_grad():
        for images in tqdm(loader, desc="获取预测结果"):
            images = images.to(device); outputs = model(images).squeeze()
            all_preds.extend(np.atleast_1d(outputs.cpu().numpy()))
    return np.array(all_preds)

# ==============================================================================
# 4. 主执行逻辑
# ==============================================================================
if __name__ == "__main__":
    # --- 步骤 1: 定义路径和参数 ---
    IMAGE_SIZE = 512; BATCH_SIZE = 4; NUM_WORKERS = 2
    FINETUNE_EPOCHS = 25; FINETUNE_LR = 3e-4; N_SPLITS = 5
    PRETRAINED_MODEL_PATH = '/kaggle/input/pretrained_2015_efficientnet_b6_256/pytorch/default/1/pretrained_2015_efficientnet_b6_256.pth'
    
    RAW_DATA_DIR = '/kaggle/input/aptos2019-blindness-detection'
    source_train_dir = os.path.join(RAW_DATA_DIR, 'train_images')
    source_test_dir = os.path.join(RAW_DATA_DIR, 'test_images')
    train_label_file = os.path.join(RAW_DATA_DIR, 'train.csv')
    test_label_file = os.path.join(RAW_DATA_DIR, 'test.csv')

    preprocessed_input_dir = '/kaggle/input/images-preprocessed-512'
    working_dir = '/kaggle/working'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的设备: {device}")

    # --- 步骤 2: 智能检查并确定数据路径 ---
    final_train_dir = os.path.join(preprocessed_input_dir, 'train_images_preprocessed_512')
    final_test_dir = os.path.join(preprocessed_input_dir, 'test_images_preprocessed_512')
    if not (os.path.exists(final_train_dir) and os.path.exists(final_test_dir)):
        print(f"未在 {preprocessed_input_dir} 找到预处理数据，将执行实时预处理。")
        final_train_dir, final_test_dir = os.path.join(working_dir, f'train_images_preprocessed_{IMAGE_SIZE}'), os.path.join(working_dir, f'test_images_preprocessed_{IMAGE_SIZE}')
        preprocess_directory(source_train_dir, final_train_dir, IMAGE_SIZE)
        preprocess_directory(source_test_dir, final_test_dir, IMAGE_SIZE)
    else: print(f"成功找到已有的预处理数据: {preprocessed_input_dir}")

    # --- 步骤 3: 5折交叉验证训练 ---
    df = pd.read_csv(train_label_file)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    all_folds_exist = all([os.path.exists(f'best_model_fold_{f}.pth') for f in range(N_SPLITS)])
    if not all_folds_exist:
        for fold, (train_idx, val_idx) in enumerate(skf.split(df['id_code'], df['diagnosis'])):
            if os.path.exists(f'best_model_fold_{fold}.pth'): print(f"Fold {fold} 模型已存在，跳过。"); continue
            print(f"\n--- 开始训练 Fold {fold} ---")
            train_df, valid_df = df.iloc[train_idx], df.iloc[val_idx]
            train_dataset = CustomImageDataset(final_train_dir, train_df, transform=get_transforms('train', IMAGE_SIZE))
            valid_dataset = CustomImageDataset(final_train_dir, valid_df, transform=get_transforms('valid', IMAGE_SIZE))
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
            valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
            model = EfficientNetModel(pretrained_model_path=PRETRAINED_MODEL_PATH).to(device)
            train_fold(model, train_loader, valid_loader, device, FINETUNE_EPOCHS, FINETUNE_LR, fold)
    else: print("\n--- 所有5折模型文件均已存在，跳过训练阶段 ---")

    # --- 步骤 4: OOF预测与阈值优化 ---
    print("\n--- 步骤 4: 生成OOF预测并优化阈值 ---")
    df = pd.read_csv(train_label_file)
    oof_preds = np.zeros((len(df), 1))
    for fold, (train_idx, val_idx) in enumerate(skf.split(df['id_code'], df['diagnosis'])):
        valid_df = df.iloc[val_idx]
        valid_dataset = CustomImageDataset(final_train_dir, valid_df, transform=get_transforms('valid', IMAGE_SIZE))
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        model = EfficientNetModel(pretrained_model_path=None).to(device)
        model.load_state_dict(torch.load(f'best_model_fold_{fold}.pth'))
        oof_preds[val_idx] = get_predictions(valid_loader, model, device).reshape(-1, 1)

    true_labels = df['diagnosis'].values
    rounder = OptimizedRounder(); rounder.fit(oof_preds.flatten(), true_labels)
    optimized_coefficients = rounder.coef_
    final_oof_preds = rounder.predict(oof_preds.flatten(), optimized_coefficients)
    oof_kappa = cohen_kappa_score(true_labels, final_oof_preds, weights='quadratic')
    print(f"\n优化后的 OOF Kappa 分数: {oof_kappa:.4f}")

    # --- 步骤 5: 集成预测测试集并生成提交文件 ---
    print("\n--- 步骤 5: 集成预测测试集并提交 ---")
    test_df = pd.read_csv(test_label_file)
    test_dataset = CustomImageDataset(final_test_dir, test_df, transform=get_transforms('test', IMAGE_SIZE), is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    all_test_preds = []
    for fold in range(N_SPLITS):
        print(f"--- 使用 Fold {fold} 模型进行测试集预测 ---")
        model = EfficientNetModel(pretrained_model_path=None).to(device)
        model.load_state_dict(torch.load(f'best_model_fold_{fold}.pth'))
        all_test_preds.append(get_predictions(test_loader, model, device))
    avg_test_preds = np.mean(all_test_preds, axis=0)
    final_test_labels = rounder.predict(avg_test_preds, optimized_coefficients)
    submission_df = pd.read_csv(test_label_file)
    submission_df['diagnosis'] = final_test_labels
    submission_df.to_csv('submission.csv', index=False)
    print("\n--- 提交文件 'submission.csv' 已生成 ---")
    print("--- 脚本执行结束 ---")

