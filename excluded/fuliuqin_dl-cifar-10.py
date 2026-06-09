pip install py7zr


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # 添加这行
from pathlib import Path
import py7zr
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split, Dataset
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from PIL import Image


torch.manual_seed(42)
np.random.seed(42)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


INPUT_DIR = Path("/kaggle/input/cifar-10")
OUTPUT_DIR = Path("/kaggle/working")


def extract_cifar10():
    if (OUTPUT_DIR/"train").exists() and (OUTPUT_DIR/"test").exists():
        print("数据已解压，跳过解压步骤")
        return
    (OUTPUT_DIR/"train").mkdir(exist_ok=True, parents=True)
    (OUTPUT_DIR/"test").mkdir(exist_ok=True, parents=True)
    with py7zr.SevenZipFile(INPUT_DIR/"train.7z", mode='r') as archive:
        archive.extractall(path=OUTPUT_DIR/"train")
    with py7zr.SevenZipFile(INPUT_DIR/"test.7z", mode='r') as archive:
        archive.extractall(path=OUTPUT_DIR/"test")
    labels = pd.read_csv(INPUT_DIR/"trainLabels.csv")
    labels.to_csv(OUTPUT_DIR/"trainLabels.csv", index=False)
    print(f"解压完成！文件保存至: {OUTPUT_DIR}")

if not (OUTPUT_DIR/"train").exists():
    extract_cifar10()


# 加载标签
labels_df = pd.read_csv(OUTPUT_DIR/"trainLabels.csv")
label_map = dict(zip(labels_df["id"], labels_df["label"]))
class_names = sorted(labels_df['label'].unique().tolist())
num_classes = len(class_names)
print(f"类别数量: {num_classes}, 类别名称: {class_names}")


class CIFAR10Dataset(Dataset):
    def __init__(self, root_dir, label_map=None, transform=None, is_test=False):
        self.root = Path(root_dir)
        self.transform = transform
        self.is_test = is_test
        self.images = sorted([f for f in self.root.glob("*.png")])
        
        if not is_test:
            self.labels = []
            self.class_to_idx = {c: i for i, c in enumerate(class_names)}
            for img_path in self.images:
                img_id = int(img_path.stem)
                self.labels.append(self.class_to_idx[label_map[img_id]])
        else:
            self.filenames = [f.name for f in self.images]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.is_test:
            return img, self.filenames[idx]
        else:
            return img, self.labels[idx]


import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
train_transform = transforms.Compose([
    transforms.RandomResizedCrop( 32, 
        scale=(0.9, 1.0),  
        ratio=(0.9, 1.1)   
    ),
    transforms.RandomHorizontalFlip(p=0.3), 
    transforms.RandomAffine(
        degrees=8,         
        translate=(0.05, 0.05),  
        scale=None,    
        shear=0        
    ),
transforms.ColorJitter(
    brightness=0.1,  
    contrast=0.1,     
    saturation=0.1,  
    hue=0.05      
),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])
train_dataset = CIFAR10Dataset(
    root_dir=OUTPUT_DIR/"train/train",
    label_map=label_map,
    transform=train_transform
)

test_dataset = CIFAR10Dataset(
    root_dir=OUTPUT_DIR/"test/test",
    transform=test_transform,
    is_test=True
)


# 划分训练集和验证集
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

val_dataset = CIFAR10Dataset(
    root_dir=OUTPUT_DIR/"train/train",
    label_map=label_map,
    transform=test_transform,
    is_test=False
)
val_dataset.images = [train_dataset.images[i] for i in val_subset.indices]
val_dataset.labels = [train_dataset.labels[i] for i in val_subset.indices]


# 数据加载器
batch_size = 64
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
train_labels = []
for inputs, labels in train_loader:
    train_labels.extend(labels.cpu().numpy())  # Convert labels to numpy array and flatten
label_counts = pd.Series(train_labels).value_counts()
plt.figure(figsize=(12, 6))
sns.barplot(x=label_counts.index, y=label_counts.values, palette="viridis")  # Use seaborn for styling
plt.title("Training Set Label Frequency Distribution", fontsize=14)
plt.xlabel("Label", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.xticks(rotation=45, ha="right")  
plt.grid(axis="y", alpha=0.3)  
for x, y in zip(label_counts.index, label_counts.values):
    plt.text(x, y, str(y), ha="center", va="bottom")
plt.tight_layout()  
plt.show()


def visualize_augmentations(dataset, num_augmentations=5):
    if not isinstance(dataset, CIFAR10Dataset):
        raise TypeError("dataset必须是CIFAR10Dataset实例")
    
    # 读取原始PIL图像（避免数据集transform的影响）
    img_path = dataset.images[0]
    orig_img = Image.open(img_path).convert("RGB")
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, num_augmentations + 1, 1)
    plt.imshow(orig_img)
    plt.title("Original")
    plt.axis('off')
    
    # 构建与train_transform对齐的可视化增强流程（不含ToTensor/Normalize）
    vis_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            32, 
            scale=(0.9, 1.0), 
            ratio=(0.9, 1.1)
        ),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomAffine(
            degrees=8, 
            translate=(0.05, 0.05), 
            scale=None, 
            shear=0
        ),
transforms.ColorJitter(
    brightness=0.1,   # 亮度 ±10% 变化（人眼可察觉，但不极端）
    contrast=0.1,     # 对比度 ±10% 变化
    saturation=0.1,   # 饱和度 ±10% 变化
    hue=0.05          # 色调微小偏移（避免色偏严重）
)
    ])
    
    for i in range(num_augmentations):
        # 应用增强（保持PIL格式）
        aug_img = vis_transform(orig_img)
        
        # 手动模拟训练流程的ToTensor和Normalize
        aug_tensor = transforms.ToTensor()(aug_img)
        aug_tensor = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )(aug_tensor)
        
        # 转换为PIL图像显示（自动反归一化）
        aug_img_pil = transforms.ToPILImage()(aug_tensor)
        
        plt.subplot(1, num_augmentations + 1, i + 2)
        plt.imshow(aug_img_pil)
        plt.title(f"Aug {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("data_augmentations.png")
    plt.show()


visualize_augmentations(val_dataset, num_augmentations=3)


# 定义模型（以ResNet34为例，修复原始代码中的层配置）
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = nn.ReLU()(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def create_resnet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes).to(device)


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, num_epochs=20, model_name="model"):
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        train_progress = tqdm(train_loader, desc='Train', unit='batch')
        for inputs, labels in train_progress:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)
            
            train_progress.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{(preds == labels).float().mean():.4f}'
            })
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.float() / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        print(f'Train: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}')
        
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        
        val_progress = tqdm(val_loader, desc='Val', unit='batch')
        with torch.no_grad():
            for inputs, labels in val_progress:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels)
                
                val_progress.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{(preds == labels).float().mean():.4f}'
                })
        
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.float() / len(val_loader.dataset)
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc.item())
        print(f'Val: Loss={val_epoch_loss:.4f}, Acc={val_epoch_acc:.4f}')
        
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, f"{OUTPUT_DIR}/{model_name}_best.pth")
            print("★ 保存最佳模型")
        
        if scheduler:
            scheduler.step()
        print()
    
    model.load_state_dict(best_model_wts)
    return model, history


# 创建并训练模型
model = create_resnet34(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4) #1e-4
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1) #7 #0.1

model, history = train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler,
    num_epochs=20, model_name="resnet34"
)


# 训练历史可视化
def plot_history(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig(f"{OUTPUT_DIR}/training_history.png")
    plt.show()

plot_history(history)



from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
import numpy as np

def evaluate_model(model, loader, class_names, model_name):
    """Evaluate the model and generate various metrics including per-class accuracy"""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 计算每个类别单独的准确率（新增）
    per_class_acc = {}
    for cls in range(len(class_names)):
        cls_labels = np.array(all_labels) == cls
        cls_preds = np.array(all_preds) == cls
        per_class_acc[class_names[cls]] = np.sum(cls_labels & cls_preds) / np.sum(cls_labels) if np.sum(cls_labels) != 0 else 0.0
    
    # 打印每个类别准确率（新增）
    print(f"\n{model_name} Per-Class Accuracy:")
    for cls, acc in per_class_acc.items():
        print(f"  {cls}: {acc:.4f}")
    
    # 生成包含准确率的分类报告（保留原有）
    print(f"\n{model_name} Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # 绘制混淆矩阵（保留原有）
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(OUTPUT_DIR/f'{model_name}_confusion_matrix.png')
    plt.show()
    
    # 计算AUC和绘制ROC曲线（保留原有）
    y_test_bin = label_binarize(all_labels, classes=range(len(class_names)))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], np.array(all_probs)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig(OUTPUT_DIR/f'{model_name}_roc_curve.png')
    plt.show()
    
    return all_preds, all_probs

# 评估自定义ResNet模型（调用方式不变）
print("\nEvaluating Custom ResNet Model...")
resnet_val_preds, resnet_val_probs = evaluate_model(
    model, val_loader, class_names, 'Custom ResNet'
)


def predict_test_set(model, test_loader, class_names):
    model.eval()
    predictions = []
    file_ids = []
    
    # 添加softmax计算置信度
    confidences = []
    
    with torch.no_grad():
        for inputs, filenames in tqdm(test_loader, desc="预测进度"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # 获取预测结果和置信度
            probs = torch.nn.functional.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, 1)
            
            predictions.extend(preds.cpu().numpy())
            confidences.extend(max_probs.cpu().numpy())
            
            for f in filenames:
                try:
                    file_id = int(f.split('.')[0])
                    file_ids.append(file_id)
                except ValueError:
                    # 记录错误文件名但不中断流程
                    print(f"警告: 无效文件名格式 - {f}")

    # 分析置信度分布
    print("\n预测置信度分析:")
    print(f"平均置信度: {np.mean(confidences):.4f}")
    print(f"最低置信度: {np.min(confidences):.4f}")
    print(f"最高置信度: {np.max(confidences):.4f}")
    
    plt.hist(confidences, bins=50)
    plt.title("Distribution of Prediction Confidences")
    plt.xlabel("Confidence Score")
    plt.ylabel("Number of Samples")
    plt.show()
    # 创建提交文件
    idx_to_class = {i: cls for i, cls in enumerate(class_names)}
    predicted_classes = [idx_to_class[p] for p in predictions]
    
    submission = pd.DataFrame({
        'id': file_ids,
        'label': predicted_classes
    }).sort_values('id')
    
    # 检查ID范围
    print(f"\nID范围: {submission['id'].min()} - {submission['id'].max()}")
    print(f"总样本数: {len(submission)}")
    
    return submission


# 使用表现最好的模型进行预测
print("\n预测测试集...")#resnet_model，pretrained_model
submission_df = predict_test_set(model,test_loader, class_names)

# 保存提交文件
submission_path = OUTPUT_DIR / 'submission.csv'
submission_df.to_csv(submission_path, index=False)
print(f"\n提交文件已保存至: {submission_path}")

# 验证提交文件格式
print("\n提交文件格式验证:")
print(f"行数: {len(submission_df)}")
print(f"列名: {submission_df.columns.tolist()}")
print(f"ID范围: {submission_df['id'].min()} 到 {submission_df['id'].max()}")
print(f"类别分布:\n{submission_df['label'].value_counts()}")

# 检查ID是否连续
expected_ids = set(range(submission_df['id'].min(), submission_df['id'].max() + 1))
actual_ids = set(submission_df['id'])
missing_ids = expected_ids - actual_ids
if missing_ids:
    print(f"警告: 缺少 {len(missing_ids)} 个ID! 首尾缺少的ID: {sorted(missing_ids)[:5]}...{sorted(missing_ids)[-5:]}")
else:
    print("所有ID连续无缺失")


submission_df


import seaborn as sns
import matplotlib.pyplot as plt

# 创建 countplot
plt.figure(figsize=(10, 6))
ax = sns.countplot(
    data=submission_df, 
    x='label',  # 指定分类列
    order=submission_df['label'].value_counts().index,  # 按频率排序
    palette="viridis"  # 设置颜色为 viridis
)

# 添加数值标签
for p in ax.patches:
    ax.annotate(
        f'{p.get_height()}',
        (p.get_x() + p.get_width() / 2, p.get_height()),
        ha='center', va='baseline', fontsize=10, xytext=(0, 5),
        textcoords='offset points'
    )

# 美化图表（改为英文）
plt.title('Label Frequency Distribution', fontsize=14)
plt.xlabel('Class Label', fontsize=12)
plt.ylabel('Occurrence Frequency', fontsize=12)
plt.xticks(rotation=15)  # 旋转标签防止重叠
plt.tight_layout()
plt.show()




