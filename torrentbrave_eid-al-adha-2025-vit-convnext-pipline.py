# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path
# from PIL import Image
# from tqdm.notebook import tqdm

# # Set style for better visualizations
# plt.style.use('fivethirtyeight')
# sns.set_style('whitegrid')

# # Ignore warnings
# import warnings
# warnings.filterwarnings('ignore')

# # Set Kaggle paths
# HOME = "/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images"
# # Define paths for train and test data
# train_dir = Path(f'{HOME}/train')
# test_dir = Path(f'{HOME}/test')
# labels_file = Path(f'{HOME}/train_labels.csv')

# print("📁 Setup Complete!")


# # Load training labels
# train_labels = pd.read_csv(labels_file)

# # Display basic information
# print("📝 Training Labels Info:")
# print(f"Total number of labeled images: {len(train_labels)}")
# print(f"\nFirst few entries:")
# display(train_labels.head())

# # Check for missing values
# print("\n🔍 Missing Values Check:")
# display(train_labels.isnull().sum())


# # Calculate class distribution
# class_dist = train_labels['label'].value_counts()

# # Create a bar plot using matplotlib
# plt.figure(figsize=(12, 6))
# bars = plt.bar(class_dist.index, class_dist.values)
# plt.title('Distribution of Sheep Breeds', pad=20)
# plt.xlabel('Breed')
# plt.ylabel('Number of Images')

# # Color the bars using a color map
# colors = plt.cm.viridis(np.linspace(0, 1, len(class_dist)))
# for bar, color in zip(bars, colors):
#     bar.set_color(color)

# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # Print class distribution statistics
# print("\n📊 Class Distribution Statistics:")
# for breed, count in class_dist.items():
#     print(f"{breed}: {count} images ({count/len(train_labels)*100:.2f}%)")


# def load_and_analyze_image(img_path):
#     """Load and return image dimensions and basic stats"""
#     img = Image.open(img_path)
#     return img.size, np.array(img).mean()

# # Analyze a sample of images
# sample_size = min(100, len(train_labels))
# sample_images = train_labels['filename'].sample(sample_size)

# # Collect image statistics
# image_sizes = []
# image_means = []

# print("🔍 Analyzing sample images...")
# for img_file in tqdm(sample_images):
#     img_path = train_dir / img_file
#     if img_path.exists():
#         size, mean_val = load_and_analyze_image(img_path)
#         image_sizes.append(size)
#         image_means.append(mean_val)

# # Convert to DataFrame for analysis
# img_stats = pd.DataFrame({
#     'width': [s[0] for s in image_sizes],
#     'height': [s[1] for s in image_sizes],
#     'mean_pixel_value': image_means
# })

# # Display statistics
# print("\n📊 Image Statistics:")
# display(img_stats.describe())

# # Plot image dimensions distribution using matplotlib
# plt.figure(figsize=(10, 8))
# scatter = plt.scatter(img_stats['width'], img_stats['height'], 
#                      c=img_stats['mean_pixel_value'], 
#                      cmap='viridis')
# plt.colorbar(scatter, label='Mean Pixel Value')
# plt.title('Image Dimensions Distribution')
# plt.xlabel('Width (pixels)')
# plt.ylabel('Height (pixels)')
# plt.tight_layout()
# plt.show()


# # Function to display sample images
# def display_sample_images(df, samples_per_breed=5):
#     # Get unique breeds
#     breeds = df['label'].unique()
#     n_breeds = len(breeds)
    
#     # Create a figure with subplots for each breed
#     plt.figure(figsize=(20, 4*n_breeds))
    
#     # For each breed
#     for breed_idx, breed in enumerate(breeds):
#         # Get 5 random samples for this breed
#         breed_samples = df[df['label'] == breed].sample(min(samples_per_breed, len(df[df['label'] == breed])))
        
#         # Display each sample
#         for sample_idx, (_, row) in enumerate(breed_samples.iterrows(), 1):
#             plt.subplot(n_breeds, samples_per_breed, breed_idx * samples_per_breed + sample_idx)
#             img = Image.open(train_dir / row['filename'])
#             plt.imshow(img)
#             plt.title(f"{breed}")
#             plt.axis('off')
    
#     plt.tight_layout()
#     plt.show()

# print("🖼️ Sample Images from Each Breed (5 samples per breed):")
# display_sample_images(train_labels, samples_per_breed=5)


# import os
# import gc
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torch.cuda.amp import GradScaler, autocast
# from sklearn.model_selection import train_test_split
# from PIL import Image
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import timm
# from tqdm.auto import tqdm
# import torch.nn.functional as F

# class Config:
#     DATA_PATH = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/'
#     IMG_SIZE = 416
#     BATCH_SIZE = 2
#     GRAD_ACCUMULATION_STEPS = 8
#     EPOCHS = 20
#     MODEL_NAME = 'convnext_large.fb_in22k_ft_in1k_384'
#     LEARNING_RATE = 5e-5
#     LR_MIN = 1e-6
#     WARMUP_EPOCHS = 1
#     WEIGHT_DECAY = 1e-6
#     DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#     NUM_WORKERS = 0
#     MIXUP_CUTMIX_ALPHA = 0.5

# def get_transforms(img_size):
#     train_transforms = A.Compose([
#         A.Resize(img_size, img_size),
#         A.HorizontalFlip(p=0.5),
#         A.ShiftScaleRotate(p=0.3, scale_limit=0.2, rotate_limit=30, border_mode=0),
#         A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.7),
#         A.CoarseDropout(max_holes=8, max_height=int(img_size*0.1), max_width=int(img_size*0.1), p=0.5),
#         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ToTensorV2(),
#     ])
#     valid_transforms = A.Compose([
#         A.Resize(img_size, img_size),
#         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ToTensorV2(),
#     ])
#     return train_transforms, valid_transforms

# class SheepDataset(Dataset):
#     def __init__(self, df, transforms=None, is_test=False):
#         self.df = df
#         self.transforms = transforms
#         self.is_test = is_test
#         self.image_dir = os.path.join(Config.DATA_PATH, 'test' if is_test else 'train')

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         img_path = os.path.join(self.image_dir, row['filename'])
#         image = Image.open(img_path).convert('RGB')
#         image = np.array(image)
#         if self.transforms:
#             image = self.transforms(image=image)['image']
#         if self.is_test:
#             return image, row['filename']
#         else:
#             return image, torch.tensor(row['label_int'], dtype=torch.long)

# def mixup_data(x, y, alpha=1.0):
#     if alpha > 0: lam = np.random.beta(alpha, alpha)
#     else: lam = 1
#     batch_size = x.size()[0]
#     index = torch.randperm(batch_size).to(Config.DEVICE)
#     mixed_x = lam * x + (1 - lam) * x[index, :]
#     y_a, y_b = y, y[index]
#     return mixed_x, y_a, y_b, lam

# def cutmix_data(x, y, alpha=1.0):
#     lam = np.random.beta(alpha, alpha)
#     rand_index = torch.randperm(x.size()[0]).to(Config.DEVICE)
#     y_a, y_b = y, y[rand_index]
#     bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
#     x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
#     lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
#     return x, y_a, y_b, lam

# def rand_bbox(size, lam):
#     W, H = size[2], size[3]
#     cut_rat = np.sqrt(1. - lam)
#     cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
#     cx, cy = np.random.randint(W), np.random.randint(H)
#     bbx1 = np.clip(cx - cut_w // 2, 0, W)
#     bby1 = np.clip(cy - cut_h // 2, 0, H)
#     bbx2 = np.clip(cx + cut_w // 2, 0, W)
#     bby2 = np.clip(cy + cut_h // 2, 0, H)
#     return bbx1, bby1, bbx2, bby2

# def mixup_criterion(criterion, pred, y_a, y_b, lam):
#     return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, scaler):
#     model.train()
#     total_loss, correct_predictions, total_samples = 0, 0, 0
#     device_type = Config.DEVICE.split(':')[0]
#     progress = tqdm(dataloader, desc='Training', leave=False)
#     optimizer.zero_grad()
#     for i, (images, labels) in enumerate(progress):
#         images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
#         total_samples += labels.size(0)
        
#         r = np.random.rand()
#         if Config.MIXUP_CUTMIX_ALPHA > 0 and r < 0.5:
#             images, targets_a, targets_b, lam = mixup_data(images, labels, Config.MIXUP_CUTMIX_ALPHA)
#         elif Config.MIXUP_CUTMIX_ALPHA > 0 and r < 0.9:
#             images, targets_a, targets_b, lam = cutmix_data(images, labels, Config.MIXUP_CUTMIX_ALPHA)
#         else:
#             targets_a, targets_b, lam = labels, labels, 1.0

#         with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
#             outputs = model(images)
#             loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        
#         loss = loss / Config.GRAD_ACCUMULATION_STEPS
#         scaler.scale(loss).backward()
        
#         if (i + 1) % Config.GRAD_ACCUMULATION_STEPS == 0:
#             scaler.step(optimizer)
#             scaler.update()
#             optimizer.zero_grad()
        
#         scheduler.step()
#         total_loss += loss.item() * Config.GRAD_ACCUMULATION_STEPS
        
#         preds = outputs.argmax(dim=1)
#         correct_predictions += (lam * (preds == targets_a).sum().item() + (1 - lam) * (preds == targets_b).sum().item())
#         progress.set_postfix(loss=loss.item() * Config.GRAD_ACCUMULATION_STEPS, lr=scheduler.get_last_lr()[0])

#     return total_loss / len(dataloader), correct_predictions / total_samples

# @torch.no_grad()
# def validate(model, dataloader, criterion):
#     model.eval()
#     total_loss, correct_predictions, total_samples = 0, 0, 0
#     device_type = Config.DEVICE.split(':')[0]
#     for images, labels in dataloader:
#         images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
#         total_samples += len(labels)
#         with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#         total_loss += loss.item()
#         correct_predictions += (outputs.argmax(dim=1) == labels).sum().item()
#     return total_loss / len(dataloader), correct_predictions / total_samples

# def main():
#     print(f"Using device: {Config.DEVICE}")
#     df = pd.read_csv(os.path.join(Config.DATA_PATH, 'train_labels.csv'))

#     class_names = sorted(df['label'].unique())
#     class_to_int = {name: i for i, name in enumerate(class_names)}
#     int_to_class = {i: name for i, name in enumerate(class_names)}
#     df['label_int'] = df['label'].map(class_to_int)
    
#     train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label'])

#     train_transforms, valid_transforms = get_transforms(Config.IMG_SIZE)
#     train_dataset = SheepDataset(train_df, transforms=train_transforms)
#     val_dataset = SheepDataset(val_df, transforms=valid_transforms)
    
#     train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
#     val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE * 2, shuffle=False, num_workers=Config.NUM_WORKERS)

#     model = timm.create_model(Config.MODEL_NAME, pretrained=True, num_classes=len(class_names), drop_path_rate=0.25)
#     model.to(Config.DEVICE)

#     criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
#     scaler = torch.amp.GradScaler(enabled=(Config.DEVICE == 'cuda'))
    
#     num_train_steps = len(train_loader) * Config.EPOCHS
#     scheduler = torch.optim.lr_scheduler.OneCycleLR(
#         optimizer, max_lr=Config.LEARNING_RATE, total_steps=num_train_steps,
#         pct_start=float(Config.WARMUP_EPOCHS) / Config.EPOCHS
#     )

#     best_accuracy = 0
#     best_model_path = 'best_model.pth'

#     for epoch in range(Config.EPOCHS):
#         print(f"\n--- Epoch {epoch+1}/{Config.EPOCHS} ---")
#         train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, scaler)
#         val_loss, val_accuracy = validate(model, val_loader, criterion)
#         print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_accuracy:.4f} | Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.4f}")

#         if val_accuracy > best_accuracy:
#             print(f"🚀 Accuracy improved! Saving model to {best_model_path}")
#             best_accuracy = val_accuracy
#             torch.save(model.state_dict(), best_model_path)

#     print("\nTraining finished. Loading best model for inference with TTA.")
#     model.load_state_dict(torch.load(best_model_path))

#     test_image_dir = os.path.join(Config.DATA_PATH, 'test')
#     test_filenames = [f for f in os.listdir(test_image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
#     test_df = pd.DataFrame({'filename': test_filenames})
    
#     test_dataset = SheepDataset(test_df, transforms=valid_transforms, is_test=True)
#     test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE * 2, shuffle=False, num_workers=Config.NUM_WORKERS)

#     model.eval()
#     predictions, filenames = [], []
#     device_type = Config.DEVICE.split(':')[0]
#     with torch.no_grad():
#         for images, fns in tqdm(test_loader, desc='Predicting with TTA'):
#             images = images.to(Config.DEVICE)
#             with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
#                 outputs_original = model(images)
#                 outputs_flipped = model(torch.flip(images, dims=[3]))
#             avg_probs = (F.softmax(outputs_original, dim=1) + F.softmax(outputs_flipped, dim=1)) / 2
#             preds = avg_probs.argmax(dim=1).cpu().numpy()
#             predictions.extend(preds)
#             filenames.extend(fns)

#     submission_df = pd.DataFrame({'filename': filenames, 'label_int': predictions})
#     submission_df['label'] = submission_df['label_int'].map(int_to_class)
#     final_submission = submission_df[['filename', 'label']]
#     final_submission.to_csv('submission.csv', index=False)

#     print("\n Submission file created successfully!")

# if __name__ == '__main__':
#     main()


# # 添加资源监控
# import psutil

# def log_resources():
#     # 显存监控
#     if torch.cuda.is_available():
#         alloc = torch.cuda.memory_allocated() / 1024**3
#         reserv = torch.cuda.memory_reserved() / 1024**3
#         print(f"GPU显存: 已用 {alloc:.2f}GB / 保留 {reserv:.2f}GB")
    
#     # 内存监控
#     mem = psutil.virtual_memory()
#     print(f"内存: 已用 {mem.used/1024**3:.2f}GB / 总计 {mem.total/1024**3:.2f}GB")
    
#     # 磁盘监控
#     disk = psutil.disk_usage('/kaggle')
#     print(f"磁盘: 已用 {disk.used/1024**3:.2f}GB / 剩余 {disk.free/1024**3:.2f}GB")


import warnings
warnings.filterwarnings("ignore")


import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
from transformers import ViTFeatureExtractor, ViTForImageClassification
from transformers import TrainingArguments, Trainer
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
from torch.optim import AdamW
from sklearn.utils.class_weight import compute_class_weight

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"可用 GPU 数量: {torch.cuda.device_count()}")

# 数据路径
TRAIN_DIR = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train'
TEST_DIR = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/test'
TRAIN_LABELS_PATH = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv'
DUMMY_SUB_PATH = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/dummy_sub.csv'

# 加载训练标签
train_labels = pd.read_csv(TRAIN_LABELS_PATH)

# ====================================
# 1. 增强型标签到索引映射（处理类别不平衡）
# ====================================
# 分析类别分布
class_counts = train_labels['label'].value_counts()
print("\n类别分布统计:")
print(class_counts)

# 创建标签到索引的映射
label_to_idx = {label: idx for idx, label in enumerate(sorted(train_labels['label'].unique()))}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
NUM_CLASSES = len(label_to_idx)

# ====================================
# 2. 自适应数据增强策略
# ====================================
def get_train_augmentations():
    return A.Compose([
        A.Resize(height=256, width=256),  # 明确指定height和width
        A.RandomResizedCrop(
            size=(224, 224),  # 使用size参数替代height和width
            scale=(0.8, 1.0), 
            ratio=(0.75, 1.33),  # 添加默认宽高比范围
            interpolation=cv2.INTER_LINEAR,  # 明确插值方法
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.Affine(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.7),  # 使用Affine替代ShiftScaleRotate
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.7),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.MotionBlur(blur_limit=7, p=0.2),  # 移除了GlassBlur以简化
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=0, p=0.5),  # 使用CoarseDropout替代Cutout
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_augmentations():
    return A.Compose([
        A.Resize(height=256, width=256),
        A.CenterCrop(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# ====================================
# 3. 改进的数据集类（添加数据增强）
# ====================================
class EnhancedSheepDataset(Dataset):
    def __init__(self, image_dir, labels_df=None, is_train=True):
        self.image_dir = image_dir
        self.is_train = is_train
        self.augment = get_train_augmentations() if is_train else get_val_augmentations()
        
        if is_train:
            self.labels_df = labels_df
            self.image_names = labels_df['filename'].tolist()
        else:
            self.image_names = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # 使用OpenCV读取图像（支持Albumentations）
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 应用数据增强
        augmented = self.augment(image=image)
        pixel_values = augmented['image']
        
        if self.is_train:
            label = self.labels_df.loc[self.labels_df['filename'] == img_name, 'label'].values[0]
            label_idx = label_to_idx[label]
            return {'pixel_values': pixel_values, 'labels': torch.tensor(label_idx, dtype=torch.long)}
        else:
            return {'pixel_values': pixel_values, 'filename': img_name}

# ====================================
# 4. 交叉验证支持（5折交叉验证）
# ====================================
def get_kfold_datasets(k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_labels['filename'], train_labels['label'])):
        train_df = train_labels.iloc[train_idx]
        val_df = train_labels.iloc[val_idx]
        
        train_dataset = EnhancedSheepDataset(TRAIN_DIR, train_df)
        val_dataset = EnhancedSheepDataset(TRAIN_DIR, val_df)
        
        folds.append({
            'fold': fold_idx,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset
        })
    
    return folds

# ====================================
# 5. 焦点损失函数（改善类别不平衡）
# ====================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ====================================
# 6. 自定义训练器（集成焦点损失）
# ====================================
class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(device) if class_weights is not None else None
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # 使用焦点损失函数
        loss_fct = FocalLoss(
            alpha=self.class_weights, 
            gamma=self.gamma
        )
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# ====================================
# 7. 增强型评估函数（计算F1分数）
# ====================================
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    
    # 计算多个评估指标
    accuracy = (predictions == labels).mean()
    macro_f1 = f1_score(labels, predictions, average='macro')
    weighted_f1 = f1_score(labels, predictions, average='weighted')
    
    # 输出混淆矩阵（诊断错误模式）
    cm = confusion_matrix(labels, predictions)
    print("\n混淆矩阵:")
    print(cm)
    
    # 计算每类F1分数
    per_class_f1 = f1_score(labels, predictions, average=None)
    print("\n每类F1分数:")
    for idx, f1 in enumerate(per_class_f1):
        print(f"{idx_to_label[idx]}: {f1:.4f}")
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "avg_f1": (macro_f1 + weighted_f1) / 2
    }

# ====================================
# 8. 模型训练主函数
# ====================================
def train_and_evaluate():
    # 创建5折交叉验证数据集
    folds = get_kfold_datasets(k=5)

    # 模型集成预测列表
    test_predictions = []

    # 统计每折验证集的结果指标
    metrics_per_fold = []
    
    for fold_data in folds:
        fold_idx = fold_data['fold']
        print(f"\n================== 训练折数 {fold_idx+1}/5 ==================")
        
        # 动态计算类别权重（基于当前fold）
        train_labels_fold = fold_data['train_dataset'].labels_df['label'].map(label_to_idx)
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(train_labels_fold), 
            y=train_labels_fold
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        
        # 重新初始化模型
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=NUM_CLASSES,
            id2label=idx_to_label,
            label2id=label_to_idx,
            ignore_mismatched_sizes=True,
        )#.to(device) 让 Trainer自动分配多 GPU 除非再用单卡或者CPU上处理再加上
        # model.gradient_checkpointing_enable() 显存优化，时间换空间
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=f'/kaggle/tmp/results_fold_{fold_idx}',
            num_train_epochs=30,  # 增加训练轮次
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=300,
            weight_decay=0.01,
            logging_dir=f'/kaggle/tmp/logs_fold_{fold_idx}',
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            # 关键修改：禁用所有模型保存
            save_strategy="no",  # 完全禁用保存
            save_steps=None,     # 不需要保存步数
            save_total_limit=0,  # 不保留任何检查点
            # save_strategy="steps",
            # save_steps=100,
            # save_total_limit=1,
            learning_rate=5e-5,  # 调整学习率
            metric_for_best_model='avg_f1',  # 使用平均F1作为主要评估指标
            greater_is_better=True,
            bf16=torch.cuda.is_available(),
            disable_tqdm=False,
            report_to=["tensorboard"],
            remove_unused_columns=False,
            label_names=["labels"],
        )
        
        # 创建自定义Trainer（包含焦点损失）
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=fold_data['train_dataset'],
            eval_dataset=fold_data['val_dataset'],
            compute_metrics=compute_metrics,
            class_weights=class_weights,
            gamma=1.5  # 焦点损失参数
        )
        
        # 训练模型
        trainer.train()
        
        # 在验证集上评估
        eval_results = trainer.evaluate()
        print(f"\nFold {fold_idx} 验证集性能:")
        print(eval_results)

        # 每次 fold 后添加
        metrics_per_fold.append(eval_results)
        
        # 保存模型
        # trainer.save_model(f"best_vit_model_fold_{fold_idx}")
        
        # 在测试集上进行预测（用于后续集成）
        test_set = EnhancedSheepDataset(TEST_DIR, is_train=False)
        test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
        
        fold_test_predictions = []
        filenames = []
        
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"预测折数 {fold_idx}"):
                inputs = batch['pixel_values'].to(device)
                output = model(pixel_values=inputs)
                logits = output.logits
                probs = torch.softmax(logits, dim=-1)
                
                fold_test_predictions.append(probs.detach().cpu().numpy())
                filenames.extend(batch['filename'])
        
        test_predictions.append(np.concatenate(fold_test_predictions))

        # 在每个fold结束后
        import gc; gc.collect()
        torch.cuda.empty_cache()
        # 显式删除不再需要的对象
        del trainer

    print("\n========== K 折交叉验证结果汇总 ==========")
    for metric_name in metrics_per_fold[0].keys():
        values = [fold_result[metric_name] for fold_result in metrics_per_fold]
        mean_val = np.mean(values)
        range_val = np.max(values) - np.min(values)
        print(f"{metric_name}: 均值 = {mean_val:.4f}, 波动 = ±{range_val/2:.4f}")
    
    return np.array(test_predictions), filenames

# ====================================
# 9. 模型集成与后处理
# ====================================
def ensemble_predictions(predictions, filenames):
    # 2. 概率阈值处理（可选）
    # 可以提高低置信度预测的可靠性
    
    # 1. 加权集成（按验证集性能加权）
    fold_weights = [m['eval_avg_f1'] for m in metrics_per_fold]
    weighted_avg = np.average(predictions, axis=0, weights=fold_weights)
    final_predictions = np.argmax(weighted_avg, axis=1)
    
    # 2. 创建提交文件
    predicted_labels = [idx_to_label[idx] for idx in final_predictions]
    
    submission = pd.DataFrame({
        'filename': filenames,
        'label': predicted_labels
    })
    
    # 确保顺序与示例提交文件一致
    dummy_sub = pd.read_csv(DUMMY_SUB_PATH)
    submission = submission.sort_values('filename').reset_index(drop=True)
    
    return submission

# ====================================
# 主执行流程
# ====================================
if __name__ == "__main__":
    os.makedirs("/kaggle/tmp/logs", exist_ok=True)
    os.makedirs("/kaggle/tmp/results", exist_ok=True)
    
    # 训练模型并获取测试集预测
    test_predictions, test_filenames = train_and_evaluate()
    
    # 集成模型预测
    submission = ensemble_predictions(test_predictions, test_filenames)
    
    # 保存预测结果
    submission.to_csv('submission.csv', index=False)
    print("\n预测结果已保存至 submission.csv")
    print(f"测试集预测样本数: {len(submission)}") 


# !pip install optuna-integration[pytorch_lightning]


# import optuna
# from optuna.integration import PyTorchLightningPruningCallback
# import os
# import pandas as pd
# import torch
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# from sklearn.model_selection import train_test_split, StratifiedKFold
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
# from tqdm import tqdm
# from transformers import ViTFeatureExtractor, ViTForImageClassification
# from transformers import TrainingArguments, Trainer
# import cv2
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import torch.nn as nn
# from torch.optim import AdamW
# from sklearn.utils.class_weight import compute_class_weight

# # 设置随机种子保证可重复性
# torch.manual_seed(42)
# np.random.seed(42)

# # 检查是否有可用的GPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")
# print(f"可用 GPU 数量: {torch.cuda.device_count()}")

# # 数据路径
# TRAIN_DIR = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train'
# TEST_DIR = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/test'
# TRAIN_LABELS_PATH = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv'
# DUMMY_SUB_PATH = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/dummy_sub.csv'

# # 加载训练标签
# train_labels = pd.read_csv(TRAIN_LABELS_PATH)

# # ====================================
# # 1. 增强型标签到索引映射（处理类别不平衡）
# # ====================================
# # 分析类别分布
# class_counts = train_labels['label'].value_counts()
# print("\n类别分布统计:")
# print(class_counts)

# # 创建标签到索引的映射
# label_to_idx = {label: idx for idx, label in enumerate(sorted(train_labels['label'].unique()))}
# idx_to_label = {idx: label for label, idx in label_to_idx.items()}
# NUM_CLASSES = len(label_to_idx)

# # ====================================
# # 2. 增强型数据增强策略（针对小数据集优化）
# # ====================================
# def get_train_augmentations():
#     return A.Compose([
#         # 核心增强（100%应用）
#         A.Resize(height=256, weight=256),
#         A.RandomResizedCrop(height=224, weight=224, scale=(0.6, 1.0), ratio=(0.6, 1.4)),
        
#         # 几何变换组（90%概率应用其中1-2种）
#         A.OneOf([
#             A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=30, p=0.7),
#             A.ElasticTransform(alpha=120, sigma=60, p=0.3),
#         ], p=0.9),
        
#         # 颜色变换组（80%概率）
#         A.OneOf([
#             A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1, p=0.5),
#             A.FancyPCA(alpha=0.3, p=0.3),
#             A.ToGray(p=0.2),
#         ], p=0.8),
        
#         # 遮挡与噪声组（70%概率）
#         A.OneOf([
#             A.CoarseDropout(max_holes=16, max_height=32, max_width=32, p=0.5),
#             A.GridDropout(ratio=0.25, p=0.3),
#             A.GaussianNoise(var_limit=(30, 70), p=0.2),
#         ], p=0.7),
        
#         # 环境模拟组（50%概率）
#         A.OneOf([
#             A.RandomShadow(p=0.4),
#             A.RandomFog(fog_coef_upper=0.4, p=0.3),
#             A.RandomSunFlare(p=0.3),
#         ], p=0.5),
        
#         # 标准化
#         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ToTensorV2(),
#     ])

# def get_val_augmentations():
#     return A.Compose([
#         A.Resize(height=256, width=256),
#         A.CenterCrop(height=224, width=224),
#         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ToTensorV2(),
#     ])

# # ====================================
# # 3. 改进的数据集类（添加数据增强）
# # ====================================
# class EnhancedSheepDataset(Dataset):
#     def __init__(self, image_dir, labels_df=None, is_train=True):
#         self.image_dir = image_dir
#         self.is_train = is_train
#         self.augment = get_train_augmentations() if is_train else get_val_augmentations()
        
#         if is_train:
#             self.labels_df = labels_df
#             self.image_names = labels_df['filename'].tolist()
#         else:
#             self.image_names = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    
#     def __len__(self):
#         return len(self.image_names)
    
#     def __getitem__(self, idx):
#         img_name = self.image_names[idx]
#         img_path = os.path.join(self.image_dir, img_name)
        
#         # 使用OpenCV读取图像（支持Albumentations）
#         image = cv2.imread(img_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#         # 应用数据增强
#         augmented = self.augment(image=image)
#         pixel_values = augmented['image']
        
#         if self.is_train:
#             label = self.labels_df.loc[self.labels_df['filename'] == img_name, 'label'].values[0]
#             label_idx = label_to_idx[label]
#             return {'pixel_values': pixel_values, 'labels': torch.tensor(label_idx, dtype=torch.long)}
#         else:
#             return {'pixel_values': pixel_values, 'filename': img_name}

# # ====================================
# # 4. 交叉验证支持（5折交叉验证）
# # ====================================
# def get_kfold_datasets(k=5):
#     skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
#     folds = []
#     for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_labels['filename'], train_labels['label'])):
#         train_df = train_labels.iloc[train_idx]
#         val_df = train_labels.iloc[val_idx]
        
#         train_dataset = EnhancedSheepDataset(TRAIN_DIR, train_df)
#         val_dataset = EnhancedSheepDataset(TRAIN_DIR, val_df)
        
#         folds.append({
#             'fold': fold_idx,
#             'train_dataset': train_dataset,
#             'val_dataset': val_dataset
#         })
    
#     return folds

# # ====================================
# # 5. 焦点损失函数（改善类别不平衡）
# # ====================================
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = reduction

#     def forward(self, inputs, targets):
#         ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
#         pt = torch.exp(-ce_loss)
#         focal_loss = (1 - pt) ** self.gamma * ce_loss
        
#         if self.alpha is not None:
#             focal_loss = self.alpha[targets] * focal_loss
        
#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         else:
#             return focal_loss

# # ====================================
# # 6. 自定义训练器（集成焦点损失）
# # ====================================
# class CustomTrainer(Trainer):
#     def __init__(self, class_weights=None, gamma=2.0, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.class_weights = class_weights.to(device) if class_weights is not None else None
#         self.gamma = gamma

#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)
#         logits = outputs.logits
        
#         # 使用焦点损失函数
#         loss_fct = FocalLoss(
#             alpha=self.class_weights, 
#             gamma=self.gamma
#         )
#         loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
#         return (loss, outputs) if return_outputs else loss

# # ====================================
# # 7. 增强型评估函数（计算F1分数）
# # ====================================
# def compute_metrics(p):
#     predictions, labels = p
#     predictions = np.argmax(predictions, axis=1)
    
#     # 计算多个评估指标
#     accuracy = (predictions == labels).mean()
#     macro_f1 = f1_score(labels, predictions, average='macro')
#     weighted_f1 = f1_score(labels, predictions, average='weighted')
    
#     # 输出混淆矩阵（诊断错误模式）
#     cm = confusion_matrix(labels, predictions)
#     print("\n混淆矩阵:")
#     print(cm)
    
#     # 计算每类F1分数
#     per_class_f1 = f1_score(labels, predictions, average=None)
#     print("\n每类F1分数:")
#     for idx, f1 in enumerate(per_class_f1):
#         print(f"{idx_to_label[idx]}: {f1:.4f}")
    
#     return {
#         "accuracy": accuracy,
#         "macro_f1": macro_f1,
#         "weighted_f1": weighted_f1,
#         "avg_f1": (macro_f1 + weighted_f1) / 2
#     }

# # ====================================
# # 8. Optuna目标函数
# # ====================================
# def objective(trial):
#     # 定义要优化的超参数范围
#     params = {
#         'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True),
#         'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
#         'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
#         'hidden_dropout_prob': trial.suggest_float('hidden_dropout_prob', 0.05, 0.3),
#         'attention_probs_dropout_prob': trial.suggest_float('attention_probs_dropout_prob', 0.05, 0.3),
#         'gamma': trial.suggest_float('gamma', 1.0, 3.0),
#         'warmup_ratio': trial.suggest_float('warmup_ratio', 0.01, 0.2),
#         'num_epochs': trial.suggest_int('num_epochs', 15, 30),
#         'max_grad_norm': trial.suggest_float('max_grad_norm', 0.1, 5.0),
#         'label_smoothing': trial.suggest_float('label_smoothing', 0.05, 0.3),  # 新增标签平滑
#     }
    
#     print(f"\n=== 开始Optuna试验 #{trial.number} ===")
#     print(f"超参数: {params}")
    
#     # 创建5折交叉验证数据集
#     folds = get_kfold_datasets(k=5)
    
#     # 存储每折的验证集性能
#     fold_metrics = []
    
#     for fold_data in folds:
#         fold_idx = fold_data['fold']
#         print(f"\n== 训练折数 {fold_idx+1}/5 ==")

#         # 动态计算类别权重（基于当前fold）
#         train_labels_fold = fold_data['train_dataset'].labels_df['label'].map(label_to_idx)
#         class_weights = compute_class_weight(
#             'balanced', 
#             classes=np.unique(train_labels_fold), 
#             y=train_labels_fold
#         )
#         class_weights = torch.tensor(class_weights, dtype=torch.float)
        
#         # 重新初始化模型
#         model = ViTForImageClassification.from_pretrained(
#             "google/vit-huge-patch14-224-in21k",
#             num_labels=NUM_CLASSES,
#             id2label=idx_to_label,
#             label2id=label_to_idx,
#             ignore_mismatched_sizes=True,
#             hidden_dropout_prob=params['hidden_dropout_prob'],
#             attention_probs_dropout_prob=params['attention_probs_dropout_prob']
#         ) 
#         # model.gradient_checkpointing_enable() 显存优化，时间换空间
#         # 冻结模型前80%的层（迁移学习优化）
#         total_layers = len(model.vit.encoder.layer)
#         freeze_until = int(total_layers * 0.8)
        
#         for i, layer in enumerate(model.vit.encoder.layer):
#             if i < freeze_until:
#                 for param in layer.parameters():
#                     param.requires_grad = False

#         # 训练参数
#         training_args = TrainingArguments(
#             output_dir=f'./optuna_results/trial_{trial.number}_fold_{fold_idx}',
#             num_train_epochs=params['num_epochs'],
#             per_device_train_batch_size=params['batch_size'],
#             per_device_eval_batch_size=64,
#             warmup_ratio=params['warmup_ratio'],
#             weight_decay=params['weight_decay'],
#             logging_dir=f'./optuna_logs/trial_{trial.number}_fold_{fold_idx}',
#             logging_strategy="epoch",
#             eval_strategy="epoch",
#             save_strategy="no",
#             learning_rate=params['learning_rate'],
#             lr_scheduler_type="cosine",
#             gradient_accumulation_steps=1,
#             metric_for_best_model='avg_f1',
#             greater_is_better=True,
#             bf16=torch.cuda.is_available(),
#             tf32=True,
#             optim="adamw_torch_fused",
#             max_grad_norm=params['max_grad_norm'],
#             report_to=[],
#             remove_unused_columns=True,
#             label_names=["labels"],
#             dataloader_num_workers=2,
#             dataloader_pin_memory=True,
#             label_smoothing_factor=params['label_smoothing'],  # 标签平滑
#         )
        
#         # 创建自定义Trainer（包含焦点损失）
#         trainer = CustomTrainer(
#             model=model,
#             args=training_args,
#             train_dataset=fold_data['train_dataset'],
#             eval_dataset=fold_data['val_dataset'],
#             compute_metrics=compute_metrics,
#             class_weights=class_weights,
#             gamma=params['gamma']
#         )
        
#         # 训练模型
#         trainer.train()
        
#         # 在验证集上评估
#         eval_results = trainer.evaluate()
#         print(f"Fold {fold_idx} 验证集性能: {eval_results}")
        
#         # 存储当前fold的验证集avg_f1
#         fold_metrics.append(eval_results['eval_avg_f1'])
        
#         # 清理内存
#         del model, trainer
#         torch.cuda.empty_cache()
    
#     # 计算5折的平均avg_f1
#     avg_f1 = np.mean(fold_metrics)
#     print(f"\n试验 #{trial.number} 完成 | 平均验证集F1: {avg_f1:.4f}")
    
#     return avg_f1

# # ====================================
# # 9. 使用最佳超参数训练最终模型
# # ====================================
# def train_with_best_params(best_params):
#     print("\n=== 使用最佳超参数训练最终模型 ===")
#     print(f"最佳超参数: {best_params}")
    
#     # 创建5折交叉验证数据集
#     folds = get_kfold_datasets(k=5)
    
#     # 模型集成预测列表
#     test_predictions = []
#     metrics_per_fold = []
    
#     for fold_data in folds:
#         fold_idx = fold_data['fold']
#         print(f"\n== 训练折数 {fold_idx+1}/5 ==")

#         # 动态计算类别权重（基于当前fold）
#         train_labels_fold = fold_data['train_dataset'].labels_df['label'].map(label_to_idx)
#         class_weights = compute_class_weight(
#             'balanced', 
#             classes=np.unique(train_labels_fold), 
#             y=train_labels_fold
#         )
#         class_weights = torch.tensor(class_weights, dtype=torch.float)
        
#         # 重新初始化模型
#         model = ViTForImageClassification.from_pretrained(
#             "google/vit-huge-patch14-224-in21k",
#             num_labels=NUM_CLASSES,
#             id2label=idx_to_label,
#             label2id=label_to_idx,
#             ignore_mismatched_sizes=True,
#             hidden_dropout_prob=best_params['hidden_dropout_prob'],
#             attention_probs_dropout_prob=best_params['attention_probs_dropout_prob']
#         )
#         # model.gradient_checkpointing_enable() 显存优化，时间换空间
#         # 冻结模型前80%的层
#         total_layers = len(model.vit.encoder.layer)
#         freeze_until = int(total_layers * 0.8)
        
#         for i, layer in enumerate(model.vit.encoder.layer):
#             if i < freeze_until:
#                 for param in layer.parameters():
#                     param.requires_grad = False
        
#         # 训练参数
#         training_args = TrainingArguments(
#             output_dir=f'./best_results/fold_{fold_idx}',
#             num_train_epochs=best_params['num_epochs'],
#             per_device_train_batch_size=best_params['batch_size'],
#             per_device_eval_batch_size=64,
#             warmup_ratio=best_params['warmup_ratio'],
#             weight_decay=best_params['weight_decay'],
#             logging_dir=f'./best_logs/fold_{fold_idx}',
#             logging_strategy="epoch",
#             eval_strategy="epoch",
#             save_strategy="no",
#             learning_rate=best_params['learning_rate'],
#             lr_scheduler_type="cosine",
#             gradient_accumulation_steps=1,
#             metric_for_best_model='avg_f1',
#             greater_is_better=True,
#             bf16=torch.cuda.is_available(),
#             tf32=True,
#             optim="adamw_torch_fused",
#             max_grad_norm=best_params['max_grad_norm'],
#             report_to=[],
#             remove_unused_columns=True,
#             label_names=["labels"],
#             dataloader_num_workers=2,
#             dataloader_pin_memory=True,
#             label_smoothing_factor=best_params.get('label_smoothing', 0.1),
#         )
        
#         # 创建自定义Trainer
#         trainer = CustomTrainer(
#             model=model,
#             args=training_args,
#             train_dataset=fold_data['train_dataset'],
#             eval_dataset=fold_data['val_dataset'],
#             compute_metrics=compute_metrics,
#             class_weights=class_weights,
#             gamma=best_params['gamma']
#         )
        
#         # 训练模型
#         trainer.train()
        
#         # 在验证集上评估
#         eval_results = trainer.evaluate()
#         print(f"Fold {fold_idx} 验证集性能: {eval_results}")
#         metrics_per_fold.append(eval_results)
        
#         # 在测试集上进行预测
#         test_set = EnhancedSheepDataset(TEST_DIR, is_train=False)
#         test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
        
#         fold_test_predictions = []
#         filenames = []
        
#         model.eval()
#         with torch.no_grad():
#             for batch in tqdm(test_loader, desc=f"预测折数 {fold_idx}"):
#                 inputs = batch['pixel_values'].to(device)
#                 output = model(pixel_values=inputs)
#                 logits = output.logits
#                 probs = torch.softmax(logits, dim=-1)
                
#                 fold_test_predictions.append(probs.detach().cpu().numpy())
#                 filenames.extend(batch['filename'])
        
#         test_predictions.append(np.concatenate(fold_test_predictions))

#         # 清理内存
#         del model, trainer
#         torch.cuda.empty_cache()
    
#     return np.array(test_predictions), filenames, metrics_per_fold

# # ====================================
# # 10. 模型集成与后处理
# # ====================================
# def ensemble_predictions(predictions, filenames, metrics_per_fold):
#     # 1. 加权集成（按验证集性能加权）
#     fold_weights = [m['eval_avg_f1'] for m in metrics_per_fold]
#     weighted_avg = np.average(predictions, axis=0, weights=fold_weights)
#     final_predictions = np.argmax(weighted_avg, axis=1)
    
#     # 2. 创建提交文件
#     predicted_labels = [idx_to_label[idx] for idx in final_predictions]
    
#     submission = pd.DataFrame({
#         'filename': filenames,
#         'label': predicted_labels
#     })
    
#     # 3. 确保顺序与示例提交文件一致
#     dummy_sub = pd.read_csv(DUMMY_SUB_PATH)
#     submission = submission.sort_values('filename').reset_index(drop=True)
    
#     return submission

# # ====================================
# # 主执行流程
# # ====================================
# if __name__ == "__main__":
#     # 创建Optuna研究
#     study = optuna.create_study(
#         direction='maximize',
#         sampler=optuna.samplers.TPESampler(seed=42),
#         pruner=optuna.pruners.MedianPruner(n_warmup_steps=3)
#     )
    
#     # 运行Optuna优化
#     study.optimize(objective, n_trials=10, timeout=24*3600)
    
#     # 输出最佳试验结果
#     print("\n=== Optuna优化完成 ===")
#     print(f"最佳试验ID: {study.best_trial.number}")
#     print(f"最佳验证集F1: {study.best_value:.4f}")
#     print(f"最佳超参数: {study.best_params}")
    
#     # 使用最佳超参数训练最终模型
#     test_predictions, test_filenames, metrics_per_fold = train_with_best_params(study.best_params)
    
#     # 集成模型预测
#     submission = ensemble_predictions(test_predictions, test_filenames, metrics_per_fold)
    
#     # 保存预测结果
#     submission.to_csv('submission.csv', index=False)
#     print("\n预测结果已保存至 submission.csv")
#     print(f"测试集预测样本数: {len(submission)}")
    
#     # 保存Optuna研究结果
#     optuna.visualization.plot_optimization_history(study).write_image("optuna_history.png")
#     optuna.visualization.plot_param_importances(study).write_image("param_importances.png")
    
#     # 保存所有试验结果到CSV
#     trials_df = study.trials_dataframe()
#     trials_df.to_csv("optuna_trials.csv", index=False)


# import optuna
# from optuna.integration import PyTorchLightningPruningCallback

# # 在代码开头添加
# import os
# import pandas as pd
# import torch
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# from sklearn.model_selection import train_test_split, StratifiedKFold
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
# from tqdm import tqdm
# from transformers import ViTFeatureExtractor, ViTForImageClassification
# from transformers import TrainingArguments, Trainer
# import cv2
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import torch.nn as nn
# from torch.optim import AdamW
# from sklearn.utils.class_weight import compute_class_weight

# # 设置随机种子保证可重复性
# torch.manual_seed(42)
# np.random.seed(42)

# # 检查是否有可用的GPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")
# print(f"可用 GPU 数量: {torch.cuda.device_count()}")

# # 数据路径
# TRAIN_DIR = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train'
# TEST_DIR = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/test'
# TRAIN_LABELS_PATH = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv'
# DUMMY_SUB_PATH = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/dummy_sub.csv'

# # 加载训练标签
# train_labels = pd.read_csv(TRAIN_LABELS_PATH)

# # ====================================
# # 1. 增强型标签到索引映射（处理类别不平衡）
# # ====================================
# # 分析类别分布
# class_counts = train_labels['label'].value_counts()
# print("\n类别分布统计:")
# print(class_counts)

# # 创建标签到索引的映射
# label_to_idx = {label: idx for idx, label in enumerate(sorted(train_labels['label'].unique()))}
# idx_to_label = {idx: label for label, idx in label_to_idx.items()}
# NUM_CLASSES = len(label_to_idx)

# # ====================================
# # 2. 自适应数据增强策略
# # ====================================
# def get_train_augmentations():
#     return A.Compose([
#         A.Resize(height=256, width=256),  # 明确指定height和width
#         A.RandomResizedCrop(
#             size=(224, 224),  # 使用size参数替代height和width
#             scale=(0.8, 1.0), 
#             ratio=(0.75, 1.33),  # 添加默认宽高比范围
#             interpolation=cv2.INTER_LINEAR,  # 明确插值方法
#             p=1.0
#         ),
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.3),
#         A.RandomRotate90(p=0.3),
#         A.Affine(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.7),  # 使用Affine替代ShiftScaleRotate
#         A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.7),
#         A.OneOf([
#             A.GaussianBlur(blur_limit=(3, 7), p=0.5),
#             A.MotionBlur(blur_limit=7, p=0.2),  # 移除了GlassBlur以简化
#         ], p=0.3),
#         A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
#         A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
#         A.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=0, p=0.5),  # 使用CoarseDropout替代Cutout
#         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ToTensorV2(),
#     ])

# def get_val_augmentations():
#     return A.Compose([
#         A.Resize(height=256, width=256),
#         A.CenterCrop(height=224, width=224),
#         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ToTensorV2(),
#     ])

# # ====================================
# # 3. 改进的数据集类（添加数据增强）
# # ====================================
# class EnhancedSheepDataset(Dataset):
#     def __init__(self, image_dir, labels_df=None, is_train=True):
#         self.image_dir = image_dir
#         self.is_train = is_train
#         self.augment = get_train_augmentations() if is_train else get_val_augmentations()
        
#         if is_train:
#             self.labels_df = labels_df
#             self.image_names = labels_df['filename'].tolist()
#         else:
#             self.image_names = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    
#     def __len__(self):
#         return len(self.image_names)
    
#     def __getitem__(self, idx):
#         img_name = self.image_names[idx]
#         img_path = os.path.join(self.image_dir, img_name)
        
#         # 使用OpenCV读取图像（支持Albumentations）
#         image = cv2.imread(img_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#         # 应用数据增强
#         augmented = self.augment(image=image)
#         pixel_values = augmented['image']
        
#         if self.is_train:
#             label = self.labels_df.loc[self.labels_df['filename'] == img_name, 'label'].values[0]
#             label_idx = label_to_idx[label]
#             return {'pixel_values': pixel_values, 'labels': torch.tensor(label_idx, dtype=torch.long)}
#         else:
#             return {'pixel_values': pixel_values, 'filename': img_name}

# # ====================================
# # 4. 交叉验证支持（5折交叉验证）
# # ====================================
# def get_kfold_datasets(k=5):
#     skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
#     folds = []
#     for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_labels['filename'], train_labels['label'])):
#         train_df = train_labels.iloc[train_idx]
#         val_df = train_labels.iloc[val_idx]
        
#         train_dataset = EnhancedSheepDataset(TRAIN_DIR, train_df)
#         val_dataset = EnhancedSheepDataset(TRAIN_DIR, val_df)
        
#         folds.append({
#             'fold': fold_idx,
#             'train_dataset': train_dataset,
#             'val_dataset': val_dataset
#         })
    
#     return folds

# # ====================================
# # 5. 焦点损失函数（改善类别不平衡）
# # ====================================
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = reduction

#     def forward(self, inputs, targets):
#         ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
#         pt = torch.exp(-ce_loss)
#         focal_loss = (1 - pt) ** self.gamma * ce_loss
        
#         if self.alpha is not None:
#             focal_loss = self.alpha[targets] * focal_loss
        
#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         else:
#             return focal_loss

# # ====================================
# # 6. 自定义训练器（集成焦点损失）
# # ====================================
# class CustomTrainer(Trainer):
#     def __init__(self, class_weights=None, gamma=2.0, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.class_weights = class_weights.to(device) if class_weights is not None else None
#         self.gamma = gamma

#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)
#         logits = outputs.logits
        
#         # 使用焦点损失函数
#         loss_fct = FocalLoss(
#             alpha=self.class_weights, 
#             gamma=self.gamma
#         )
#         loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
#         return (loss, outputs) if return_outputs else loss

# # ====================================
# # 7. 增强型评估函数（计算F1分数）
# # ====================================
# def compute_metrics(p):
#     predictions, labels = p
#     predictions = np.argmax(predictions, axis=1)
    
#     # 计算多个评估指标
#     accuracy = (predictions == labels).mean()
#     macro_f1 = f1_score(labels, predictions, average='macro')
#     weighted_f1 = f1_score(labels, predictions, average='weighted')
    
#     # 输出混淆矩阵（诊断错误模式）
#     cm = confusion_matrix(labels, predictions)
#     print("\n混淆矩阵:")
#     print(cm)
    
#     # 计算每类F1分数
#     per_class_f1 = f1_score(labels, predictions, average=None)
#     print("\n每类F1分数:")
#     for idx, f1 in enumerate(per_class_f1):
#         print(f"{idx_to_label[idx]}: {f1:.4f}")
    
#     return {
#         "accuracy": accuracy,
#         "macro_f1": macro_f1,
#         "weighted_f1": weighted_f1,
#         "avg_f1": (macro_f1 + weighted_f1) / 2
#     }

# # ====================================
# # 8. Optuna目标函数
# # ====================================
# def objective(trial):
#     # 定义要优化的超参数范围
#     params = {
#         'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True),
#         'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
#         'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
#         'hidden_dropout_prob': trial.suggest_float('hidden_dropout_prob', 0.05, 0.3),
#         'attention_probs_dropout_prob': trial.suggest_float('attention_probs_dropout_prob', 0.05, 0.3),
#         'gamma': trial.suggest_float('gamma', 1.0, 3.0),
#         'warmup_ratio': trial.suggest_float('warmup_ratio', 0.01, 0.2),
#         'num_epochs': trial.suggest_int('num_epochs', 15, 30),
#         'max_grad_norm': trial.suggest_float('max_grad_norm', 0.1, 5.0),  # 新增梯度裁剪阈值优化
#     }
    
#     print(f"\n=== 开始Optuna试验 #{trial.number} ===")
#     print(f"超参数: {params}")
    
#     # 创建5折交叉验证数据集
#     folds = get_kfold_datasets(k=5)
    
#     # 存储每折的验证集性能
#     fold_metrics = []
    
#     for fold_data in folds:
#         fold_idx = fold_data['fold']
#         print(f"\n== 训练折数 {fold_idx+1}/5 ==")

#         # 动态计算类别权重（基于当前fold）
#         train_labels_fold = fold_data['train_dataset'].labels_df['label'].map(label_to_idx)
#         class_weights = compute_class_weight(
#             'balanced', 
#             classes=np.unique(train_labels_fold), 
#             y=train_labels_fold
#         )
#         class_weights = torch.tensor(class_weights, dtype=torch.float)
        
#         # 重新初始化模型 - 使用正确的参数名
#         model = ViTForImageClassification.from_pretrained(
#             "google/vit-huge-patch14-224-in21k",
#             num_labels=NUM_CLASSES,
#             id2label=idx_to_label,
#             label2id=label_to_idx,
#             ignore_mismatched_sizes=True,
#             hidden_dropout_prob=params['hidden_dropout_prob'],
#             attention_probs_dropout_prob=params['attention_probs_dropout_prob']
#         ) 

#         # model.gradient_checkpointing_enable() 显存优化，时间换空间
#         # 冻结模型前80%的层（迁移学习优化）
#         # total_layers = len(model.vit.encoder.layer)
#         # freeze_until = int(total_layers * 0.8)
        
#         # for i, layer in enumerate(model.vit.encoder.layer):
#         #     if i < freeze_until:
#         #         for param in layer.parameters():
#         #             param.requires_grad = False

#         # 冻结所有 encoder 层，只训练分类头（适用于小数据）
#         for param in model.vit.parameters():
#             param.requires_grad = False
#         for param in model.classifier.parameters():
#             param.requires_grad = True
                    
#         # 训练参数
#         training_args = TrainingArguments(
#             output_dir=f'./optuna_results/trial_{trial.number}_fold_{fold_idx}',
#             num_train_epochs=params['num_epochs'],
#             per_device_train_batch_size=params['batch_size'],
#             per_device_eval_batch_size=64,
#             warmup_ratio=params['warmup_ratio'],
#             weight_decay=params['weight_decay'],
#             logging_dir=f'./optuna_logs/trial_{trial.number}_fold_{fold_idx}',
#             logging_strategy="epoch",
#             eval_strategy="epoch",
#             save_strategy="no",  # 禁用保存以节省空间
#             learning_rate=params['learning_rate'],
#             lr_scheduler_type="cosine",
#             gradient_accumulation_steps=1,
#             metric_for_best_model='avg_f1',
#             greater_is_better=True,
#             bf16=torch.cuda.is_available(),
#             tf32=True,
#             optim="adamw_torch_fused",
#             max_grad_norm=params['max_grad_norm'],
#             report_to=[],
#             remove_unused_columns=True,
#             label_names=["labels"],
#             dataloader_num_workers=2,
#             dataloader_pin_memory=True,
#         )
        
#         # 创建自定义Trainer（包含焦点损失）
#         trainer = CustomTrainer(
#             model=model,
#             args=training_args,
#             train_dataset=fold_data['train_dataset'],
#             eval_dataset=fold_data['val_dataset'],
#             compute_metrics=compute_metrics,
#             class_weights=class_weights,
#             gamma=params['gamma']  # 焦点损失参数
#         )

#         # 第一个阶段：只训练 classifier
#         for param in model.vit.parameters():
#             param.requires_grad = False
#         trainer.train()
        
#         # 第二阶段：解冻后几层 + classifier
#         for i in range(total_layers - 4, total_layers):
#             for param in model.vit.encoder.layer[i].parameters():
#                 param.requires_grad = True
#         trainer.train()

#         # # 训练模型
#         # trainer.train()
        
#         # 在验证集上评估
#         eval_results = trainer.evaluate()
#         print(f"Fold {fold_idx} 验证集性能: {eval_results}")
        
#         # 存储当前fold的验证集avg_f1
#         fold_metrics.append(eval_results['eval_avg_f1'])
        
#         # 清理内存
#         del model, trainer
#         torch.cuda.empty_cache()
    
#     # 计算5折的平均avg_f1
#     avg_f1 = np.mean(fold_metrics)
#     print(f"\n试验 #{trial.number} 完成 | 平均验证集F1: {avg_f1:.4f}")
    
#     return avg_f1

# # ====================================
# # 9. 使用最佳超参数训练最终模型
# # ====================================
# def train_with_best_params(best_params):
#     print("\n=== 使用最佳超参数训练最终模型 ===")
#     print(f"最佳超参数: {best_params}")
    
#     # 创建5折交叉验证数据集
#     folds = get_kfold_datasets(k=5)
    
#     # 模型集成预测列表
#     test_predictions = []
#     metrics_per_fold = []
    
#     for fold_data in folds:
#         fold_idx = fold_data['fold']
#         print(f"\n== 训练折数 {fold_idx+1}/5 ==")

#         # 动态计算类别权重（基于当前fold）
#         train_labels_fold = fold_data['train_dataset'].labels_df['label'].map(label_to_idx)
#         class_weights = compute_class_weight(
#             'balanced', 
#             classes=np.unique(train_labels_fold), 
#             y=train_labels_fold
#         )
#         class_weights = torch.tensor(class_weights, dtype=torch.float)
        
#         # 重新初始化模型 - 使用正确的参数名
#         model = ViTForImageClassification.from_pretrained(
#             "google/vit-huge-patch14-224-in21k",
#             num_labels=NUM_CLASSES,
#             id2label=idx_to_label,
#             label2id=label_to_idx,
#             ignore_mismatched_sizes=True,
#             hidden_dropout_prob=best_params['hidden_dropout_prob'],
#             attention_probs_dropout_prob=best_params['attention_probs_dropout_prob']
#         )
        
#         # model.gradient_checkpointing_enable() 显存优化，时间换空间
#         # 冻结模型前80%的层（迁移学习优化）
#         # total_layers = len(model.vit.encoder.layer)
#         # freeze_until = int(total_layers * 0.8)
        
#         # for i, layer in enumerate(model.vit.encoder.layer):
#         #     if i < freeze_until:
#         #         for param in layer.parameters():
#         #             param.requires_grad = False

#         # 冻结所有 encoder 层，只训练分类头（适用于小数据）
#         for param in model.vit.parameters():
#             param.requires_grad = False
#         for param in model.classifier.parameters():
#             param.requires_grad = True
        
#         # 训练参数
#         training_args = TrainingArguments(
#             output_dir=f'./best_results/fold_{fold_idx}',
#             num_train_epochs=best_params['num_epochs'],
#             per_device_train_batch_size=best_params['batch_size'],
#             per_device_eval_batch_size=64,
#             warmup_ratio=best_params['warmup_ratio'],
#             weight_decay=best_params['weight_decay'],
#             logging_dir=f'./best_logs/fold_{fold_idx}',
#             logging_strategy="epoch",
#             eval_strategy="epoch",
#             save_strategy="no",
#             learning_rate=best_params['learning_rate'],
#             lr_scheduler_type="cosine",
#             gradient_accumulation_steps=1,
#             metric_for_best_model='avg_f1',
#             greater_is_better=True,
#             bf16=torch.cuda.is_available(),
#             tf32=True,
#             optim="adamw_torch_fused",
#             max_grad_norm=best_params['max_grad_norm'],
#             report_to=[],
#             remove_unused_columns=True,
#             label_names=["labels"],
#             dataloader_num_workers=2,
#             dataloader_pin_memory=True,
#         )
        
#         # 创建自定义Trainer
#         trainer = CustomTrainer(
#             model=model,
#             args=training_args,
#             train_dataset=fold_data['train_dataset'],
#             eval_dataset=fold_data['val_dataset'],
#             compute_metrics=compute_metrics,
#             class_weights=class_weights,
#             gamma=best_params['gamma']
#         )

#         # 训练模型
#         trainer.train()
        
#         # 在验证集上评估
#         eval_results = trainer.evaluate()
#         print(f"Fold {fold_idx} 验证集性能: {eval_results}")
#         metrics_per_fold.append(eval_results)
        
#         # 在测试集上进行预测
#         test_set = EnhancedSheepDataset(TEST_DIR, is_train=False)
#         test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
        
#         fold_test_predictions = []
#         filenames = []
        
#         model.eval()
#         with torch.no_grad():
#             for batch in tqdm(test_loader, desc=f"预测折数 {fold_idx}"):
#                 inputs = batch['pixel_values'].to(device)
#                 output = model(pixel_values=inputs)
#                 logits = output.logits
#                 probs = torch.softmax(logits, dim=-1)
                
#                 fold_test_predictions.append(probs.detach().cpu().numpy())
#                 filenames.extend(batch['filename'])
        
#         test_predictions.append(np.concatenate(fold_test_predictions))

#         # 清理内存
#         del model, trainer
#         torch.cuda.empty_cache()
    
#     return np.array(test_predictions), filenames, metrics_per_fold

# # ====================================
# # 10. 模型集成与后处理
# # ====================================
# def ensemble_predictions(predictions, filenames, metrics_per_fold):
#     # 1. 加权集成（按验证集性能加权）
#     fold_weights = [m['eval_avg_f1'] for m in metrics_per_fold]
#     weighted_avg = np.average(predictions, axis=0, weights=fold_weights)
#     final_predictions = np.argmax(weighted_avg, axis=1)
    
#     # 2. 创建提交文件
#     predicted_labels = [idx_to_label[idx] for idx in final_predictions]
    
#     submission = pd.DataFrame({
#         'filename': filenames,
#         'label': predicted_labels
#     })
    
#     # 3. 确保顺序与示例提交文件一致
#     dummy_sub = pd.read_csv(DUMMY_SUB_PATH)
#     submission = submission.sort_values('filename').reset_index(drop=True)
    
#     return submission

# # ====================================
# # 主执行流程
# # ====================================
# if __name__ == "__main__":
#     # 创建Optuna研究
#     study = optuna.create_study(
#         direction='maximize',  # 最大化验证集F1分数
#         sampler=optuna.samplers.TPESampler(seed=42),
#         pruner=optuna.pruners.MedianPruner(n_warmup_steps=3)
#     )
    
#     # 运行Optuna优化
#     study.optimize(objective, n_trials=10, timeout=24*3600)  # 24小时超时
    
#     # 输出最佳试验结果
#     print("\n=== Optuna优化完成 ===")
#     print(f"最佳试验ID: {study.best_trial.number}")
#     print(f"最佳验证集F1: {study.best_value:.4f}")
#     print(f"最佳超参数: {study.best_params}")
    
#     # 使用最佳超参数训练最终模型
#     test_predictions, test_filenames, metrics_per_fold = train_with_best_params(study.best_params)
    
#     # 集成模型预测
#     submission = ensemble_predictions(test_predictions, test_filenames, metrics_per_fold)
    
#     # 保存预测结果
#     submission.to_csv('submission.csv', index=False)
#     print("\n预测结果已保存至 submission.csv")
#     print(f"测试集预测样本数: {len(submission)}")
    
#     # 保存Optuna研究结果
#     optuna.visualization.plot_optimization_history(study).write_image("optuna_history.png")
#     optuna.visualization.plot_param_importances(study).write_image("param_importances.png")
    
#     # 保存所有试验结果到CSV
#     trials_df = study.trials_dataframe()
#     trials_df.to_csv("optuna_trials.csv", index=False)




# import os
# import pandas as pd
# import torch
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# from sklearn.model_selection import train_test_split, StratifiedKFold
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
# from tqdm import tqdm
# from transformers import ViTFeatureExtractor, ViTForImageClassification
# from transformers import TrainingArguments, Trainer
# import cv2
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import torch.nn as nn
# from torch.optim import AdamW
# from sklearn.utils.class_weight import compute_class_weight

# # 设置随机种子保证可重复性
# torch.manual_seed(42)
# np.random.seed(42)

# # 检查是否有可用的GPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")
# print(f"可用 GPU 数量: {torch.cuda.device_count()}")

# # 数据路径
# TRAIN_DIR = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train'
# TEST_DIR = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/test'
# TRAIN_LABELS_PATH = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv'
# DUMMY_SUB_PATH = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/dummy_sub.csv'

# # 加载训练标签
# train_labels = pd.read_csv(TRAIN_LABELS_PATH)

# # ====================================
# # 1. 增强型标签到索引映射（处理类别不平衡）
# # ====================================
# # 分析类别分布
# class_counts = train_labels['label'].value_counts()
# print("\n类别分布统计:")
# print(class_counts)

# # 创建标签到索引的映射
# label_to_idx = {label: idx for idx, label in enumerate(sorted(train_labels['label'].unique()))}
# idx_to_label = {idx: label for label, idx in label_to_idx.items()}
# NUM_CLASSES = len(label_to_idx)

# # ====================================
# # 2. 自适应数据增强策略
# # ====================================
# def get_train_augmentations():
#     return A.Compose([
#         A.Resize(height=256, width=256),  # 明确指定height和width
#         A.RandomResizedCrop(
#             size=(224, 224),  # 使用size参数替代height和width
#             scale=(0.8, 1.0), 
#             ratio=(0.75, 1.33),  # 添加默认宽高比范围
#             interpolation=cv2.INTER_LINEAR,  # 明确插值方法
#             p=1.0
#         ),
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.3),
#         A.RandomRotate90(p=0.3),
#         A.Affine(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.7),  # 使用Affine替代ShiftScaleRotate
#         A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.7),
#         A.OneOf([
#             A.GaussianBlur(blur_limit=(3, 7), p=0.5),
#             A.MotionBlur(blur_limit=7, p=0.2),  # 移除了GlassBlur以简化
#         ], p=0.3),
#         A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
#         A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
#         A.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=0, p=0.5),  # 使用CoarseDropout替代Cutout
#         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ToTensorV2(),
#     ])

# def get_val_augmentations():
#     return A.Compose([
#         A.Resize(height=256, width=256),
#         A.CenterCrop(height=224, width=224),
#         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ToTensorV2(),
#     ])

# # ====================================
# # 3. 改进的数据集类（添加数据增强）
# # ====================================
# class EnhancedSheepDataset(Dataset):
#     def __init__(self, image_dir, labels_df=None, is_train=True):
#         self.image_dir = image_dir
#         self.is_train = is_train
#         self.augment = get_train_augmentations() if is_train else get_val_augmentations()
        
#         if is_train:
#             self.labels_df = labels_df
#             self.image_names = labels_df['filename'].tolist()
#         else:
#             self.image_names = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    
#     def __len__(self):
#         return len(self.image_names)
    
#     def __getitem__(self, idx):
#         img_name = self.image_names[idx]
#         img_path = os.path.join(self.image_dir, img_name)
        
#         # 使用OpenCV读取图像（支持Albumentations）
#         image = cv2.imread(img_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#         # 应用数据增强
#         augmented = self.augment(image=image)
#         pixel_values = augmented['image']
        
#         if self.is_train:
#             label = self.labels_df.loc[self.labels_df['filename'] == img_name, 'label'].values[0]
#             label_idx = label_to_idx[label]
#             return {'pixel_values': pixel_values, 'labels': torch.tensor(label_idx, dtype=torch.long)}
#         else:
#             return {'pixel_values': pixel_values, 'filename': img_name}

# # ====================================
# # 4. 交叉验证支持（5折交叉验证）
# # ====================================
# def get_kfold_datasets(k=5):
#     skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
#     folds = []
#     for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_labels['filename'], train_labels['label'])):
#         train_df = train_labels.iloc[train_idx]
#         val_df = train_labels.iloc[val_idx]
        
#         train_dataset = EnhancedSheepDataset(TRAIN_DIR, train_df)
#         val_dataset = EnhancedSheepDataset(TRAIN_DIR, val_df)
        
#         folds.append({
#             'fold': fold_idx,
#             'train_dataset': train_dataset,
#             'val_dataset': val_dataset
#         })
    
#     return folds

# # ====================================
# # 5. 焦点损失函数（改善类别不平衡）
# # ====================================
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = reduction

#     def forward(self, inputs, targets):
#         ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
#         pt = torch.exp(-ce_loss)
#         focal_loss = (1 - pt) ** self.gamma * ce_loss
        
#         if self.alpha is not None:
#             focal_loss = self.alpha[targets] * focal_loss
        
#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         else:
#             return focal_loss

# # ====================================
# # 6. 自定义训练器（集成焦点损失）
# # ====================================
# class CustomTrainer(Trainer):
#     def __init__(self, class_weights=None, gamma=2.0, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.class_weights = class_weights.to(device) if class_weights is not None else None
#         self.gamma = gamma

#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)
#         logits = outputs.logits
        
#         # 使用焦点损失函数
#         loss_fct = FocalLoss(
#             alpha=self.class_weights, 
#             gamma=self.gamma
#         )
#         loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
#         return (loss, outputs) if return_outputs else loss

# # ====================================
# # 7. 增强型评估函数（计算F1分数）
# # ====================================
# def compute_metrics(p):
#     predictions, labels = p
#     predictions = np.argmax(predictions, axis=1)
    
#     # 计算多个评估指标
#     accuracy = (predictions == labels).mean()
#     macro_f1 = f1_score(labels, predictions, average='macro')
#     weighted_f1 = f1_score(labels, predictions, average='weighted')
    
#     # 输出混淆矩阵（诊断错误模式）
#     cm = confusion_matrix(labels, predictions)
#     print("\n混淆矩阵:")
#     print(cm)
    
#     # 计算每类F1分数
#     per_class_f1 = f1_score(labels, predictions, average=None)
#     print("\n每类F1分数:")
#     for idx, f1 in enumerate(per_class_f1):
#         print(f"{idx_to_label[idx]}: {f1:.4f}")
    
#     return {
#         "accuracy": accuracy,
#         "macro_f1": macro_f1,
#         "weighted_f1": weighted_f1,
#         "avg_f1": (macro_f1 + weighted_f1) / 2
#     }

# # ====================================
# # 8. 模型训练主函数
# # ====================================
# def train_and_evaluate():
#     # 创建5折交叉验证数据集
#     folds = get_kfold_datasets(k=5)
    
#     # 模型集成预测列表
#     test_predictions = []

#     # 统计每折验证集的结果指标
#     metrics_per_fold = []
    
#     for fold_data in folds:
#         fold_idx = fold_data['fold']
#         print(f"\n================== 训练折数 {fold_idx+1}/5 ==================")
        
#         # 动态计算类别权重（基于当前fold）
#         train_labels_fold = fold_data['train_dataset'].labels_df['label'].map(label_to_idx)
#         class_weights = compute_class_weight(
#             'balanced', 
#             classes=np.unique(train_labels_fold), 
#             y=train_labels_fold
#         )
#         class_weights = torch.tensor(class_weights, dtype=torch.float)
#         print("\n类别权重:")
#             for idx, weight in enumerate(class_weights):
#         print(f"{idx_to_label[idx]}: {weight:.4f}")
        
#         # 加载预训练ViT模型
#         model = ViTForImageClassification.from_pretrained(
#             "google/vit-base-patch16-224-in21k",
#             num_labels=NUM_CLASSES,
#             id2label=idx_to_label,
#             label2id=label_to_idx,
#             ignore_mismatched_sizes=True
#         )#.to(device) 让 Trainer自动分配多 GPU 除非再用单卡或者CPU上处理再加上
        
#         # 训练参数
#         training_args = TrainingArguments(
#             output_dir=f'/kaggle/tmp/results_fold_{fold_idx}',
#             num_train_epochs=30,  # 增加训练轮次
#             per_device_train_batch_size=16,
#             per_device_eval_batch_size=16,
#             warmup_steps=300,
#             weight_decay=0.01,
#             logging_dir=f'/kaggle/tmp/logs_fold_{fold_idx}',
#             logging_steps=10,
#             eval_strategy="steps",
#             eval_steps=100,
#             # 关键修改：禁用所有模型保存
#             save_strategy="no",  # 完全禁用保存
#             save_steps=None,     # 不需要保存步数
#             save_total_limit=0,  # 不保留任何检查点
#             # save_strategy="steps",
#             # save_steps=100,
#             # save_total_limit=1,
#             learning_rate=5e-5,  # 调整学习率
#             metric_for_best_model='avg_f1',  # 使用平均F1作为主要评估指标
#             greater_is_better=True,
#             bf16=torch.cuda.is_available(),
#             disable_tqdm=False,
#             report_to=["tensorboard"],
#             remove_unused_columns=False,
#             label_names=["labels"],
#         )
        
#         # 创建自定义Trainer（包含焦点损失）
#         trainer = CustomTrainer(
#             model=model,
#             args=training_args,
#             train_dataset=fold_data['train_dataset'],
#             eval_dataset=fold_data['val_dataset'],
#             compute_metrics=compute_metrics,
#             class_weights=class_weights,
#             gamma=1.5  # 焦点损失参数
#         )
        
#         # 训练模型
#         trainer.train()
        
#         # 在验证集上评估
#         eval_results = trainer.evaluate()
#         print(f"\nFold {fold_idx} 验证集性能:")
#         print(eval_results)

#         # 每次 fold 后添加
#         metrics_per_fold.append(eval_results)
        
#         # 保存模型
#         # trainer.save_model(f"best_vit_model_fold_{fold_idx}")
        
#         # 在测试集上进行预测（用于后续集成）
#         test_set = EnhancedSheepDataset(TEST_DIR, is_train=False)
#         test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
        
#         fold_test_predictions = []
#         filenames = []
        
#         model.eval()
#         with torch.no_grad():
#             for batch in tqdm(test_loader, desc=f"预测折数 {fold_idx}"):
#                 inputs = batch['pixel_values'].to(device)
#                 output = model(pixel_values=inputs)
#                 logits = output.logits
#                 probs = torch.softmax(logits, dim=-1)
                
#                 fold_test_predictions.append(probs.detach().cpu().numpy())
#                 filenames.extend(batch['filename'])
        
#         test_predictions.append(np.concatenate(fold_test_predictions))

#         # 在每个fold结束后
#         import gc; gc.collect()
#         torch.cuda.empty_cache()
#         # 显式删除不再需要的对象
#         del trainer, model

#     print("\n========== K 折交叉验证结果汇总 ==========")
#     for metric_name in metrics_per_fold[0].keys():
#         values = [fold_result[metric_name] for fold_result in metrics_per_fold]
#         mean_val = np.mean(values)
#         range_val = np.max(values) - np.min(values)
#         print(f"{metric_name}: 均值 = {mean_val:.4f}, 波动 = ±{range_val/2:.4f}")
    
#     return np.array(test_predictions), filenames

# # ====================================
# # 9. 模型集成与后处理
# # ====================================
# def ensemble_predictions(predictions, filenames):
    
#     # 1. 加权集成（按验证集性能加权）
#     fold_weights = [m['eval_avg_f1'] for m in metrics_per_fold]
#     weighted_avg = np.average(predictions, axis=0, weights=fold_weights)
#     final_predictions = np.argmax(weighted_avg, axis=1)
    
#     # 2. 创建提交文件
#     predicted_labels = [idx_to_label[idx] for idx in final_predictions]
    
#     submission = pd.DataFrame({
#         'filename': filenames,
#         'label': predicted_labels
#     })
    
#     # 确保顺序与示例提交文件一致
#     dummy_sub = pd.read_csv(DUMMY_SUB_PATH)
#     submission = submission.sort_values('filename').reset_index(drop=True)
    
#     return submission

# # ====================================
# # 主执行流程
# # ====================================
# if __name__ == "__main__":
#     os.makedirs("/kaggle/tmp/logs", exist_ok=True)
#     os.makedirs("/kaggle/tmp/results", exist_ok=True)
    
#     # 训练模型并获取测试集预测
#     test_predictions, test_filenames = train_and_evaluate()
    
#     # 集成模型预测
#     submission = ensemble_predictions(test_predictions, test_filenames)
    
#     # 保存预测结果
#     submission.to_csv('submission.csv', index=False)
#     print("\n预测结果已保存至 submission.csv")
#     print(f"测试集预测样本数: {len(submission)}") 



