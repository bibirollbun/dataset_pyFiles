import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

torch.backends.cudnn.benchmark = True


!sudo apt-get install p7zip-full
!7z x "/kaggle/input/cifar-10/train.7z"


IMG_DIR = '/kaggle/working/train'
LABELS_FILE = '/kaggle/input/cifar-10/trainLabels.csv'


class CustomCIFAR10Dataset(Dataset):
    def __init__(self, img_dir, labels_file, transform=None):
        # 初始化数据集，加载图像目录和标签文件
        self.img_dir = img_dir
        self.transform = transform
        self.labels = pd.read_csv(labels_file)  # 读取标签文件
        self.labels.set_index('id', inplace=True)  # 将标签文件的索引设置为 'id'

    def __len__(self):
        # 返回数据集的大小
        return len(self.labels)

    def __getitem__(self, idx):
        # 根据索引获取图像和标签
        img_name = f"{self.labels.index[idx]}.png"  # 获取图像文件名
        img_path = os.path.join(self.img_dir, img_name)  # 获取图像路径
        image = Image.open(img_path).convert('RGB')  # 打开图像并转换为 RGB 模式
        if self.transform:
            image = self.transform(image)  # 如果有变换操作，则应用变换
        label = self.labels.loc[self.labels.index[idx], 'label']  # 获取标签
        label = self.label_to_index(label)  # 将标签转换为索引
        return image, label  # 返回图像和标签

    def label_to_index(self, label):
        # 将标签转换为索引
        label_map = {
            'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
            'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
        }
        return label_map[label]


# 数据预处理
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

# 构造数据集
dataset = CustomCIFAR10Dataset(img_dir=IMG_DIR, labels_file=LABELS_FILE, transform=train_transforms)

# 划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 使用不同的变换
train_dataset.dataset.transform = train_transforms
val_dataset.dataset.transform = test_transforms

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, pin_memory=True)


# 检查是否有可用的 GPU，如果有则使用 GPU，否则使用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载预训练的 ResNet-18 模型
model = torchvision.models.resnet18(pretrained=True)

# 修改最后一层全连接层，使其输出 10 个类别（对应 CIFAR-10 数据集的 10 个类别）
model.fc = nn.Linear(512, 10)

# 将模型移动到指定设备（GPU 或 CPU）
model = model.to(device)

# 定义损失函数为交叉熵损失
criterion = nn.CrossEntropyLoss()

# 定义优化器为随机梯度下降（SGD），学习率为 0.001，动量为 0.9
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 定义梯度缩放器，用于混合精度训练
scaler = GradScaler()


num_epochs = 4
train_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        progress_bar.set_postfix(loss=running_loss/(total/labels.size(0)), accuracy=100.*correct/total)
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast():
                outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    val_acc = 100. * correct / total
    val_accuracies.append(val_acc)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.show()


# 保存模型
torch.save(model.state_dict(), 'resnet18_cifar10.pth')


import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

torch.backends.cudnn.benchmark = True


# !sudo apt-get install p7zip-full
!7z x "/kaggle/input/cifar-10/test.7z"


IMG_DIR = '/kaggle/workingtest'
SUBMISSION_FILE = './submission.csv'


class CustomCIFAR10TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')  # 使用 PIL 图像
        if self.transform:
            image = self.transform(image)
        return image, img_name

# 数据预处理
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 构造测试数据集
test_dataset = CustomCIFAR10TestDataset(img_dir=IMG_DIR, transform=test_transforms)

# 数据加载器
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, pin_memory=True)

# 定义标签映射
index_to_label = {
    0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
    5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
}

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.resnet18(weights=None)
model.fc = nn.Linear(512, 10)
model.load_state_dict(torch.load('resnet18_cifar10.pth'))
model = model.to(device)

# 确保模型处于评估模式
model.eval()

# 存储预测结果
predictions = []

with torch.no_grad():
    for images, img_names in tqdm(test_loader, desc="Inference"):
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        predicted = predicted.cpu().numpy()
        for img_name, pred in zip(img_names, predicted):
            label = index_to_label[pred]
            predictions.append((img_name, label))

# 保存预测结果到 submission.csv
submission_df = pd.DataFrame(predictions, columns=['id', 'label'])
submission_df['id'] = submission_df['id'].str.replace('.png', '')
# submission_df.to_csv('submission.csv', index=False)


# submission id转为int 按照id排序 保存到'../data/cifar-10/submission.csv'
submission_df['id'] = submission_df['id'].astype(int)
submission_df = submission_df.sort_values('id')
submission_df.to_csv(SUBMISSION_FILE, index=False)


submission_df

