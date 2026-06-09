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
# 1. 预处理函数 (用于实时处理原始图片)
# ==============================================================================
def circle_crop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    cropped_image = image[y:y + h, x:x + w]
    return cropped_image

def apply_ben_graham_preprocessing(image, sigmaX=30):
    blurred_image = cv2.GaussianBlur(image, (0, 0), sigmaX)
    processed_image = cv2.addWeighted(image, 4, blurred_image, -4, 128)
    return processed_image

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
    def __init__(self, model_name, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        replace_batchnorm_with_groupnorm(self.model)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 1)
        )
    def forward(self, x):
        return self.model(x)

class FineTuneDataset(Dataset):
    def __init__(self, labels_df, image_size, transform=None):
        self.labels_df = labels_df.copy()
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_path = row['full_path']
        label = float(row['diagnosis'])
        
        try:
            # 实时进行预处理
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = circle_crop(img)
            img = apply_ben_graham_preprocessing(img)
            img = cv2.resize(img, (self.image_size, self.image_size))
            
            if self.transform:
                img = self.transform(image=img)["image"]
            return img, label
        except Exception as e:
            print(f"警告: 处理图片 {img_path} 时出错: {e}, 将加载下一张")
            return self.__getitem__((idx + 1) % len(self))

def get_transforms(data_type='train'):
    if data_type == 'train':
        return Compose([
            Rotate(limit=40, p=0.5),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.3),
            HorizontalFlip(p=0.5),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else: # 'valid'
        return Compose([
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

# ==============================================================================
# 3. 训练函数
# ==============================================================================
def train(model, train_loader, valid_loader, device, num_epochs, lr, image_size):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7) # 使用更小的最小学习率

    best_valid_kappa = 0.0
    epochs_no_improve = 0
    patience = 3 # 微调时可以使用更小的耐心值
    save_path = os.path.join(os.getcwd(), f'final_finetuned_model_kappa_{image_size}.pth')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - 微调中"):
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
            for images, labels in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} - 验证中"):
                images = images.to(device)
                outputs = model(images).squeeze()
                all_valid_labels.extend(labels.cpu().numpy())
                preds_np = outputs.cpu().numpy()
                all_valid_preds.extend(np.atleast_1d(preds_np))
        
        rounded_preds = np.clip(np.round(all_valid_preds).astype(int), 0, 4)
        valid_kappa = cohen_kappa_score(all_valid_labels, rounded_preds, weights='quadratic')

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"  训练损失: {train_loss:.4f}")
        print(f"  验证 Kappa (四舍五入): {valid_kappa:.4f}")
        print(f"  当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step()

        if valid_kappa > best_valid_kappa:
            best_valid_kappa = valid_kappa
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> 已保存新最佳模型! 验证 Kappa: {best_valid_kappa:.4f}")
            print(f"   最终模型权重已保存至: {save_path}")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"\n在 {patience} 个 epoch 没有提升后触发早停。")
            break
    
    print(f"\n微调结束。最佳模型权重保存在: {save_path}")

# ==============================================================================
# 4. 主执行逻辑
# ==============================================================================
if __name__ == "__main__":
    # --- 步骤 1: 定义路径和参数 ---
    print("--- 步骤 1: 初始化路径和参数 ---")
    
    MODEL_NAME = 'efficientnet_b4'
    IMAGE_SIZE = 256
    BATCH_SIZE = 32
    NUM_EPOCHS = 10 # 微调时使用较少的epochs
    LEARNING_RATE = 5e-5 # 微调时使用较小的学习率
    
    # --- 路径定义 ---
    # 第一阶段训练好的权重
    STAGE1_WEIGHTS_PATH = '/kaggle/input/best-model-kappa-256/best_model_kappa_256.pth'
    
    # APTOS 2019 竞赛的原始数据路径
    APTOS_DATA_DIR = '/kaggle/input/aptos2019-blindness-detection'
    ORIGINAL_TRAIN_CSV = os.path.join(APTOS_DATA_DIR, 'train.csv')
    ORIGINAL_TRAIN_IMAGES_DIR = os.path.join(APTOS_DATA_DIR, 'train_images')
    ORIGINAL_TEST_IMAGES_DIR = os.path.join(APTOS_DATA_DIR, 'test_images')
    
    # 第二阶段生成的伪标签文件
    PSEUDO_LABELS_CSV = '/kaggle/input/efficientnetb4-newdataset/pseudo_labels.csv'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的设备: {device}")
    
    # --- 步骤 2: 加载并合并数据 ---
    print("\n--- 步骤 2: 加载并合并真实标签和伪标签数据 ---")
    
    # 加载真实训练数据
    real_labels_df = pd.read_csv(ORIGINAL_TRAIN_CSV)
    real_labels_df['full_path'] = real_labels_df['id_code'].apply(lambda x: os.path.join(ORIGINAL_TRAIN_IMAGES_DIR, f"{x}.png"))
    
    # 加载伪标签数据
    pseudo_labels_df = pd.read_csv(PSEUDO_LABELS_CSV)
    pseudo_labels_df['full_path'] = pseudo_labels_df['id_code'].apply(lambda x: os.path.join(ORIGINAL_TEST_IMAGES_DIR, f"{x}.png"))

    # 合并成一个大的训练集
    combined_df = pd.concat([real_labels_df, pseudo_labels_df], ignore_index=True)
    print(f"真实标签: {len(real_labels_df)}, 伪标签: {len(pseudo_labels_df)}")
    print(f"合并后的总训练数据量: {len(combined_df)}")

    # 从合并后的数据集中划分出训练集和验证集
    # 注意：验证集仍然只使用带真实标签的数据，这样评估才最准确
    train_df, _ = train_test_split(combined_df, test_size=0.01, random_state=42) # 这里只是为了打乱顺序
    valid_df = real_labels_df.sample(frac=0.2, random_state=42) # 从真实标签中抽20%作为验证集
    
    train_dataset = FineTuneDataset(combined_df, IMAGE_SIZE, transform=get_transforms('train'))
    valid_dataset = FineTuneDataset(valid_df, IMAGE_SIZE, transform=get_transforms('valid'))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    print("混合数据加载器创建完毕。")
    
    # --- 步骤 3: 执行模型微调 ---
    print("\n--- 步骤 3: 开始最终模型微调 ---")
    
    model = EfficientNetModel(MODEL_NAME, pretrained=False).to(device)
    
    if os.path.exists(STAGE1_WEIGHTS_PATH):
        print(f"正在加载第一阶段的预训练权重: {STAGE1_WEIGHTS_PATH}")
        model.load_state_dict(torch.load(STAGE1_WEIGHTS_PATH))
    else:
        print(f"错误: 找不到第一阶段的权重文件: {STAGE1_WEIGHTS_PATH}")
        exit()

    train(model, train_loader, valid_loader, device, NUM_EPOCHS, LEARNING_RATE, IMAGE_SIZE)

    print("\n--- 脚本执行结束 ---")

