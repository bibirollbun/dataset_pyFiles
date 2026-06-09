pip install d2l


import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 超参数设置
batch_size = 128
num_epochs = 20
learning_rate = 0.001
num_classes = 10
data_dir = "./data/cifar10"  # 数据集存储路径


# 定义训练集数据增强
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 定义测试集数据增强（仅标准化）
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 加载CIFAR-10数据集
train_dataset = datasets.CIFAR10(
    root=data_dir, train=True, download=True, transform=train_transform)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

test_dataset = datasets.CIFAR10(
    root=data_dir, train=False, download=True, transform=test_transform)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 类别名称
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')


# 定义ResNet-18模型
def get_resnet_model(num_classes=10):
    # 加载预训练的ResNet-18
    model = models.resnet18(pretrained=True)
    # 修改最后一层全连接层以适应CIFAR-10的10个类别
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

# 也可以使用CNN模型作为基线
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 初始化模型
# model = SimpleCNN(num_classes).to(device)  # 简单CNN模型
model = get_resnet_model(num_classes).to(device)  # ResNet-18模型


# 定义ResNet-18模型
def get_resnet_model(num_classes=10):
    # 加载预训练的ResNet-18
    model = models.resnet18(pretrained=True)
    # 修改最后一层全连接层以适应CIFAR-10的10个类别
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

# 也可以使用CNN模型作为基线
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 初始化模型
# model = SimpleCNN(num_classes).to(device)  # 简单CNN模型
model = get_resnet_model(num_classes).to(device)  # ResNet-18模型


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# 训练函数
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    with tqdm(total=len(train_loader)) as pbar:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            pbar.set_description(f'Epoch: {epoch} Loss: {train_loss/(batch_idx+1):.3f} Acc: {100.*correct/total:.3f}%')
            pbar.update(1)
    
    return train_loss / len(train_loader), 100. * correct / total

# 评估函数
def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as pbar:
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                test_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                pbar.set_description(f'Test Loss: {test_loss/(len(test_loader)):.3f} Acc: {100.*correct/total:.3f}%')
                pbar.update(1)
    
    return test_loss / len(test_loader), 100. * correct / total

# 训练过程
train_losses, train_accs = [], []
test_losses, test_accs = [], []

best_acc = 0.0
best_model_path = "cifar10_best_model.pth"

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch)
    test_loss, test_acc = test(model, test_loader, criterion)
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    # 保存最佳模型
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved at epoch {epoch}, accuracy: {best_acc:.2f}%")
    
    scheduler.step()

# 绘制训练过程
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(test_accs, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()


# 加载最佳模型
model.load_state_dict(torch.load(best_model_path))
model.eval()

# 对测试集进行预测
all_predictions = []
all_ids = list(range(len(test_dataset)))

with torch.no_grad():
    for data, _ in tqdm(test_loader):
        data = data.to(device)
        output = model(data)
        _, pred = torch.max(output, 1)
        all_predictions.extend(pred.cpu().numpy())

# 准备提交文件
import pandas as pd

# 创建提交数据框
submission = pd.DataFrame({
    'id': all_ids,
    'label': all_predictions
})

# 保存为CSV文件
submission.to_csv('cifar10_submission.csv', index=False)
print("Submission file created: cifar10_submission.csv")

# 可视化一些预测结果
def visualize_predictions(model, test_loader, classes, num_samples=9):
    model.eval()
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs, 1)
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()
    
    for i in range(num_samples):
        img = images[i].cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        # 反标准化
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f"True: {classes[labels[i]]}\nPred: {classes[predicted[i].cpu()]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# 可视化预测结果
visualize_predictions(model, test_loader, classes)

