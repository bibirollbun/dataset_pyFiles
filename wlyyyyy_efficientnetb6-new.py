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
    def __init__(self, model_name, pretrained_model_path=None):
        super().__init__()
        # 如果没有提供本地路径，则使用timm的默认ImageNet预训练权重
        use_timm_pretrained = pretrained_model_path is None
        self.model = timm.create_model(model_name, pretrained=use_timm_pretrained)
        
        replace_batchnorm_with_groupnorm(self.model)
        
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.Mish(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 1)
        )
        
        # 如果提供了本地路径，则加载它
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            print(f"加载自定义本地预训练权重: {pretrained_model_path}")
            # 使用 strict=False 增加加载的灵活性
            self.model.load_state_dict(torch.load(pretrained_model_path), strict=False)
        elif pretrained_model_path:
            print(f"警告: 预训练模型路径未找到: {pretrained_model_path}")

    def forward(self, x):
        return self.model(x)

# 这个Dataset类与您的图片目录结构完全匹配
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, labels_df, transform=None):
        self.img_dir = img_dir
        self.labels_df = labels_df.copy()
        self.transform = transform
        self.label_to_folder = {
            0: 'No_DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferate_DR'
        }
        self.labels_df['level'] = self.labels_df['level'].astype(int)
        self.labels_df['folder_name'] = self.labels_df['level'].map(self.label_to_folder)
        
    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_id = row['image']
        label = float(row['level'])
        folder_name = row['folder_name']

        img_path_jpeg = os.path.join(self.img_dir, folder_name, str(img_id) + '.jpeg')
        img_path_png = os.path.join(self.img_dir, folder_name, str(img_id) + '.png')

        if os.path.exists(img_path_jpeg):
            img_path = img_path_jpeg
        elif os.path.exists(img_path_png):
            img_path = img_path_png
        else:
            # 如果找不到文件，则递归地加载下一个样本
            return self.__getitem__((idx + 1) % len(self))

        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(image=img)["image"]
            return img, label
        except Exception:
            # 如果读取文件出错，同样递归加载下一个
            return self.__getitem__((idx + 1) % len(self))

def get_transforms(data_type='train', image_size=512):
    if data_type == 'train':
        return Compose([
            Resize(image_size, image_size),
            Rotate(limit=40, p=0.5),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.3),
            HorizontalFlip(p=0.5),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return Compose([
            Resize(image_size, image_size),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

# ==============================================================================
# 3. 训练函数
# ==============================================================================
def pretrain_on_2015(model, train_loader, device, num_epochs, lr, save_path):
    print("\n--- 开始在2015数据集上进行预训练 ---")
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"预训练 Epoch {epoch + 1}/{num_epochs}"):
            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        scheduler.step()
        
        print(f"Epoch {epoch + 1}/{num_epochs} | 训练损失: {train_loss:.4f} | 学习率: {optimizer.param_groups[0]['lr']:.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"--- 预训练完成，模型已保存至: {save_path} ---")

# ==============================================================================
# 4. 主执行逻辑
# ==============================================================================
if __name__ == "__main__":
    # --- 步骤 1: 定义路径和参数 ---
    print("--- 步骤 1: 初始化路径和参数 ---")
    MODEL_NAME = 'efficientnet_b6'
    BATCH_SIZE = 4
    NUM_WORKERS = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的设备: {device}")

    # <--- 修改: 定义您下载的b6模型权重路径
    BASE_MODEL_PATH = '/kaggle/input/efficientnet-b6/pytorch_model_effi_b6.bin'

    # --- 2015 预训练阶段参数 ---
    PRETRAIN_IMAGE_SIZE = 256
    PRETRAIN_EPOCHS = 20
    PRETRAIN_LR = 3e-4
    # 确保这个路径是包含 No_DR, Mild 等子文件夹的父目录
    PRETRAIN_DATA_DIR = '/kaggle/input/new-data-preprocessed-images-256/new_data_preprocessed_images_256' 
    PRETRAIN_LABEL_FILE = '/kaggle/input/new-data-labels/trainLabels.csv'
    PRETRAINED_2015_MODEL_PATH = os.path.join(os.getcwd(), f'pretrained_2015_{MODEL_NAME}_{PRETRAIN_IMAGE_SIZE}.pth')
    
    # --- 阶段一: 在2015数据集上进行预训练 ---
    if not os.path.exists(PRETRAINED_2015_MODEL_PATH):
        # 1. 加载2015标签文件
        # <--- 修正: 直接使用 'image' 和 'level' 列，不再重命名
        pretrain_df = pd.read_csv(PRETRAIN_LABEL_FILE)
        
        # 2. 准备数据集和加载器
        # 这个Dataset会自动处理子文件夹和文件格式
        pretrain_dataset = CustomImageDataset(
            PRETRAIN_DATA_DIR, 
            pretrain_df, 
            transform=get_transforms('train', image_size=PRETRAIN_IMAGE_SIZE)
        )
        pretrain_loader = DataLoader(pretrain_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        
        # 3. 初始化模型 (加载您指定的本地.bin文件)
        pretrain_model = EfficientNetModel(MODEL_NAME, pretrained_model_path=BASE_MODEL_PATH).to(device)
        
        # 4. 执行预训练
        pretrain_on_2015(pretrain_model, pretrain_loader, device, PRETRAIN_EPOCHS, PRETRAIN_LR, PRETRAINED_2015_MODEL_PATH)
    else:
        print(f"--- 找到已存在的2015预训练模型，跳过预训练: {PRETRAINED_2015_MODEL_PATH} ---")

    print("\n--- 脚本执行结束 ---")
    print("已成功执行预训练流程。如果您想继续进行微调，请取消后续代码的注释。")

