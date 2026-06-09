import torch
from torch import optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F#added import
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from copy import deepcopy
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
#此处引入头文件，需要可随时添加
# 忽略警告
import warnings
warnings.filterwarnings('ignore')

# 设备设置（GPU/CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据路径与常量
DATA_DIR = '/kaggle/input/dog-breed-identification'  # 数据集根目录
IMAGE_SIZE = (128, 128)  # 图像统一尺寸
BATCH_SIZE = 32          # 批次大小
NUM_CLASSES = 120        # 狗品种总数

#读取数据的标签
labels = pd.read_csv('/kaggle/input/dog-breed-identification/labels.csv')
labels.head()

#id是train数据集样本图片的名称 ，breed是标签，运行后正常显示则读取成功。（用于测试数据集是否正常导入note）



breed_count = labels['breed'].value_counts()
breed_count.head(10)#查看前十个类别
#breed_count.shape 输入此命令可查看数据集涉及所有分类，共120个。（用于测试数据集是否有损坏）



# 自定义数据集类
class DogBreedDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths  # 图像路径列表
        self.labels = labels        # 标签（数值编码）
        self.transform = transform  # 图像变换函数

    def __len__(self):
        return len(self.filepaths)  # 数据集大小

    def __getitem__(self, idx):
        # 读取图像并转换为RGB格式
        img_path = self.filepaths.iloc[idx]
        image = Image.open(img_path).convert('RGB')

        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 返回（图像，标签）
        label = self.labels.iloc[idx]
        return image, label


# 图像变换定义（区分训练/验证/测试集）
# 训练集：含数据增强
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),  # Moderate flip
    transforms.RandomRotation(degrees=10),   # Small rotation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 验证集：无增强，基础处理
val_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 测试集：与验证集处理一致（无增强，确保评估公平）
test_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 数据加载与三部分划分（8:1:1）

# 1. 读取原始标签文件
labels = pd.read_csv(os.path.join(DATA_DIR, 'labels.csv'))
print(f"原始数据集总样本数: {labels.shape[0]}")  # 输出 10222

# 2. 生成图像路径
labels['filepath'] = os.path.join(DATA_DIR, 'train/') + labels['id'] + '.jpg'

# 3. 编码品种标签（文本→数值）
le = LabelEncoder()
labels['breed_encoded'] = le.fit_transform(labels['breed'])
print(f"品种数: {labels['breed_encoded'].nunique()}")  # 输出 120

# 4. 第一次划分：先从原始数据中拆分出测试集（10%）
# 剩余90%用于后续拆分为训练集和验证集
X_temp, X_test, y_temp, y_test = train_test_split(
    labels['filepath'],
    labels['breed_encoded'],
    test_size=0.1,  # 测试集占10%
    random_state=10,
    stratify=labels['breed_encoded']  # 分层抽样，保持类别比例
)

# 5. 第二次划分：从剩余90%中拆分出验证集（10% of 总数据）
# 最终训练集占80%，验证集10%，测试集10%
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=1/9, 
    random_state=10,
    stratify=y_temp  # 再次分层，确保验证集类别比例正确
)


# ----------------------------
# 创建数据集实例
# ----------------------------
train_dataset = DogBreedDataset(
    filepaths=X_train,
    labels=y_train,
    transform=train_transform
)

val_dataset = DogBreedDataset(
    filepaths=X_val,
    labels=y_val,
    transform=val_transform
)

test_dataset = DogBreedDataset(
    filepaths=X_test,
    labels=y_test,
    transform=test_transform
)


# ----------------------------
# 创建DataLoader
# ----------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


# ----------------------------
# 验证结果
# ----------------------------
total = len(train_dataset) + len(val_dataset) + len(test_dataset)
print(f"总样本数: {total}")
print(f"训练集样本数: {len(train_dataset)} ({len(train_dataset)/total:.0%})")  # 约80%
print(f"验证集样本数: {len(val_dataset)} ({len(val_dataset)/total:.0%})")    # 约10%
print(f"测试集样本数: {len(test_dataset)} ({len(test_dataset)/total:.0%})")  # 约10%


#YOUR CODE 包括训练函数定义，GPU等部分此处我直接引用了personal assignment部分的代码，因为似乎可以通用。
# Define batch size and create data loaders.
batch_size = 64

# Data loader (this provides queues and threads in a very simple way).
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

# Define the device.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def train_and_val_with_lr_scheduler(model, train_loader, val_loader, loss_fn, optimizer, lr_scheduler, num_epochs, device, print_step=20):
    # record the loss and accuracy
    train_losses = []
    val_accuracies = []

    best_val_acc = 0.0
    best_model_state = None
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        model.train()
        size = len(train_loader.dataset)
        epoch_loss = 0  # 这里初始化累积 loss

        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            epoch_loss += loss.item() 
            loss.backward()
            optimizer.step()
            if batch % print_step == 0:
                loss_val, current = loss.item(), (batch+1)*len(X)
                print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

        # Trigger the scheduler step and print info if learning rate is reduced
        old_lr = optimizer.param_groups[0]['lr']
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f"\nLearning rate reduced from {old_lr} to {new_lr}.\n")

         # 记录训练 loss
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # 验证准确率
        val_acc = test(model, val_loader, device)
        val_accuracies.append(val_acc)
        lr_scheduler.step(val_acc)  # 传入验证准确率，自动调整学习率

    return train_losses, val_accuracies

def test(model, test_loader, device):
    # Test the neural network
    correct = 0
    total = 0

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculation
    with torch.inference_mode():
        for inputs, labels in test_loader:

            # Move the inputs and labels to the GPU if available
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Get the predicted class
            _, predicted = torch.max(outputs, dim=1)  # output size is (batch_size, num_classes)

            # Update the total number of samples and correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate the accuracy
    accuracy = 100 * correct / total
    print(f"val_Accuracy: {accuracy:.2f}%")
    return accuracy

# Define the cross-entropy loss function for classification
loss_fn = nn.CrossEntropyLoss()


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import efficientnet_b0

# Define a model using pretrained ResNet18
class EfficientNetTransfer(nn.Module):
    def __init__(self, num_classes=120):
        super(EfficientNetTransfer, self).__init__()
        self.model = efficientnet_b0(pretrained=True)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# Instantiate the model and move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNetTransfer(num_classes=120).to(device)
print(model)






# Set all layers trainable
for param in model.parameters():
    param.requires_grad = True
# Define optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.0008, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3, verbose=True, min_lr=1e-6
)


# Start training
num_epochs = 30  # Optional to reduce for faster testing
train_losses, val_accuracies = train_and_val_with_lr_scheduler(
    model, train_loader, val_loader,
    loss_fn, optimizer, lr_scheduler,
    num_epochs=num_epochs,
    device=device
)
print('\nTest results for the baseline network:')
test(model, test_loader, device)


%matplotlib inline

import matplotlib.pyplot as plt

def plot_train_val(train_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color=color)
    ax1.plot(epochs, train_losses, color=color, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # 创建第二个y轴
    color = 'tab:red'
    ax2.set_ylabel('Validation Accuracy (%)', color=color)
    ax2.plot(epochs, val_accuracies, color=color, label='Val Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Training Loss and Validation Accuracy')
    plt.show()
plot_train_val(train_losses, val_accuracies)


