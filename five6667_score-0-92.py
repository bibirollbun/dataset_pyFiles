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
    # 计算类别权重（处理类别不平衡）
    labels = train_labels['label'].map(label_to_idx).values
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(labels), 
        y=labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print("\n类别权重:")
    for idx, weight in enumerate(class_weights):
        print(f"{idx_to_label[idx]}: {weight:.4f}")
    
    # 加载预训练ViT模型
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=NUM_CLASSES,
        id2label=idx_to_label,
        label2id=label_to_idx,
        ignore_mismatched_sizes=True
    ).to(device)
    
    # 创建5折交叉验证数据集
    folds = get_kfold_datasets(k=5)
    
    # 模型集成预测列表
    test_predictions = []
    
    for fold_data in folds:
        fold_idx = fold_data['fold']
        print(f"\n================== 训练折数 {fold_idx+1}/5 ==================")
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=f'./results_fold_{fold_idx}',
            num_train_epochs=100,  # 增加训练轮次
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=300,
            weight_decay=0.01,
            logging_dir=f'./logs_fold_{fold_idx}',
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=1,
            learning_rate=5e-5,  # 调整学习率
            metric_for_best_model='avg_f1',  # 使用平均F1作为主要评估指标
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
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
        
        # 保存模型
        trainer.save_model(f"best_vit_model_fold_{fold_idx}")
        
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
    
    return np.array(test_predictions), filenames

# ====================================
# 9. 模型集成与后处理
# ====================================
def ensemble_predictions(predictions, filenames):
    # 1. 对多个模型预测进行平均
    avg_predictions = np.mean(predictions, axis=0)
    
    # 2. 概率阈值处理（可选）
    # 可以提高低置信度预测的可靠性
    
    # 3. 创建提交文件
    final_predictions = np.argmax(avg_predictions, axis=1)
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
    # 训练模型并获取测试集预测
    test_predictions, test_filenames = train_and_evaluate()
    
    # 集成模型预测
    submission = ensemble_predictions(test_predictions, test_filenames)
    
    # 保存预测结果
    submission.to_csv('submission.csv', index=False)
    print("\n预测结果已保存至 submission.csv")
    print(f"测试集预测样本数: {len(submission)}") 

