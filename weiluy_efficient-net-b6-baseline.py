import cv2
import numpy as np
import os
import pandas as pd
import random
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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
# 1. PyTorch 模型与数据类 (回归版本)
# ==============================================================================
class EfficientNetModelRegression(nn.Module):
    def __init__(self, model_name, pretrained_model_path=None):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False)
        
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 1)
        )

        if pretrained_model_path and os.path.exists(pretrained_model_path):
            print(f"正在从本地路径加载预训练权重: {pretrained_model_path}")
            self.model.load_state_dict(torch.load(pretrained_model_path), strict=False)
        elif pretrained_model_path:
            print(f"警告: 预训练模型路径未找到: {pretrained_model_path}")

    def forward(self, x):
        return self.model(x)

class CustomImageDatasetRegression(Dataset):
    def __init__(self, img_dir, labels_df, transform=None):
        self.img_dir = img_dir
        self.labels_df = labels_df
        self.transform = transform
        
    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_id = self.labels_df.iloc[idx, 0]
        label = float(self.labels_df.iloc[idx, 1]) 
        img_path = os.path.join(self.img_dir, img_id + '.png')
        
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(image=img)["image"]
            return img, label
        except Exception as e:
            print(f"读取图片失败: {img_path}, error: {e}")
            return self.__getitem__((idx + 1) % len(self))

def get_transforms_exp3(data_type='train'):
    return Compose([
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# ==============================================================================
# 2. 训练与预测函数 (回归版本)
# ==============================================================================
def train_regression(model, train_loader, valid_loader, device, num_epochs, lr, image_size):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    best_valid_kappa = 0.0
    epochs_no_improve = 0
    patience = 5
    save_path = os.path.join(os.getcwd(), f'exp3_model_kappa_{image_size}.pth')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} 训练中"):
            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        
        model.eval()
        all_valid_labels = []
        all_valid_preds = []
        with torch.no_grad():
            for images, labels in tqdm(valid_loader, desc="验证中"):
                images = images.to(device)
                outputs = model(images)
                
                all_valid_labels.extend(labels.cpu().numpy())
                
                # --- BUG修复: 修复在最后一个batch大小为1时产生的0维数组错误 ---
                # 将squeeze后的结果转为numpy数组
                preds_np = outputs.squeeze().cpu().numpy()
                # 使用np.atleast_1d确保结果至少是一维的，这样extend才能正常工作
                all_valid_preds.extend(np.atleast_1d(preds_np))

        rounded_preds = np.round(all_valid_preds)
        clipped_preds = np.clip(rounded_preds, 0, 4).astype(int)
        
        valid_kappa = cohen_kappa_score(all_valid_labels, clipped_preds, weights='quadratic')
        
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"  训练损失 (MSE): {train_loss:.4f}")
        print(f"  验证集 Kappa (四舍五入后): {valid_kappa:.4f}")
        print(f"  当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step()
        
        if valid_kappa > best_valid_kappa:
            best_valid_kappa = valid_kappa
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> 已保存新最佳模型! 验证集 Kappa: {best_valid_kappa:.4f}")
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"\n在 {patience} 个 epoch 没有提升后触发早停。")
            break

def predict_test_data_exp3(test_csv_path, img_dir, model, device):
    test_df = pd.read_csv(test_csv_path)
    predictions = []
    test_transform = get_transforms_exp3('test')
    model.eval()
    for img_id in tqdm(test_df['id_code'], desc="在测试集上预测"):
        img_path = os.path.join(img_dir, img_id + '.png')
        if not os.path.exists(img_path):
             img_path = os.path.join(img_dir, img_id + '.jpeg')
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_tensor = test_transform(image=img)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            pred_continuous = model(image_tensor).squeeze().item()
            pred_class = int(np.clip(np.round(pred_continuous), 0, 4))
            predictions.append(pred_class)
            
    test_df['diagnosis'] = predictions
    submission_file = os.path.join(os.getcwd(), 'submission.csv')
    test_df.to_csv(submission_file, index=False)
    print(f"\n实验3的提交文件已保存至: {submission_file}")

# ==============================================================================
# 3. 主执行逻辑 (实验3版本)
# ==============================================================================
if __name__ == "__main__":
    print("--- 正在执行实验 3: 实验 2 + 回归建模 ---")
    
    MODEL_NAME = 'efficientnet_b6'
    IMAGE_SIZE = 512 
    BATCH_SIZE = 4
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 2
    
    COMPETITION_DATA_DIR = '/kaggle/input/aptos2019-blindness-detection'
    train_label_file = os.path.join(COMPETITION_DATA_DIR, 'train.csv')
    test_label_file = os.path.join(COMPETITION_DATA_DIR, 'test.csv')

    PREPROCESSED_DATA_DIR = '/kaggle/input/images-preprocessed-512'
    preprocessed_train_dir = os.path.join(PREPROCESSED_DATA_DIR, 'train_images_preprocessed_512')
    preprocessed_test_dir = os.path.join(PREPROCESSED_DATA_DIR, 'test_images_preprocessed_512')

    PRETRAINED_MODEL_PATH = '/kaggle/input/efficientnet-b6/pytorch_model_effi_b6.bin'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的设备: {device}")
    print(f"加载训练图片路径: {preprocessed_train_dir}")
    print(f"加载测试图片路径: {preprocessed_test_dir}")

    all_labels_df = pd.read_csv(train_label_file)
    train_df, valid_df = train_test_split(
        all_labels_df, test_size=0.2, random_state=42, stratify=all_labels_df['diagnosis']
    )
    
    train_dataset = CustomImageDatasetRegression(preprocessed_train_dir, train_df, transform=get_transforms_exp3('train'))
    valid_dataset = CustomImageDatasetRegression(preprocessed_train_dir, valid_df, transform=get_transforms_exp3('valid'))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    print("\n开始训练回归模型...")
    model = EfficientNetModelRegression(MODEL_NAME, pretrained_model_path=PRETRAINED_MODEL_PATH).to(device)
    train_regression(model, train_loader, valid_loader, device, NUM_EPOCHS, LEARNING_RATE, IMAGE_SIZE)
    
    print("\n--- 加载最佳回归模型并在测试集上预测 ---")
    model_path = os.path.join(os.getcwd(), f'exp3_model_kappa_{IMAGE_SIZE}.pth')
    
    if os.path.exists(model_path):
        model_for_pred = EfficientNetModelRegression(MODEL_NAME).to(device)
        model_for_pred.load_state_dict(torch.load(model_path, map_location=device))
        
        predict_test_data_exp3(test_label_file, preprocessed_test_dir, model_for_pred, device)
    else:
        print(f"错误: 未找到已训练的模型文件: {model_path}")

    print("\n--- 实验 3 脚本执行结束 ---")

