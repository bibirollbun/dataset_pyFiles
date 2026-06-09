import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


# 设置随机种子
seed=42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 检查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 数据预处理和增强
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# 加载CIFAR-10数据集
print("加载数据集...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)



train_size = len(trainset)
test_size = len(testset)
print(f"训练集样本数量：{train_size}")
print(f"测试集样本数量：{test_size}")


class_names = trainset.classes
print(f"类别名称：{class_names}")


# 定义类别
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# 定义一个改进的ResNet模型
class ModifiedResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedResNet18, self).__init__()
        # 使用预训练的ResNet18，但修改第一个卷积层以适应32x32的图像
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # 修改第一层卷积以适应CIFAR-10的32x32图像
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 修改最后的全连接层以适应10个类别
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.model.fc.in_features, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)



# 初始化模型
print("初始化模型...")
model = ModifiedResNet18(num_classes=10)
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)



# 训练模型
def train_model(model, trainloader, criterion, optimizer, scheduler, num_epochs=100):
    model.train()
    train_losses = []
    train_accs = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(trainloader)
        for i, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar.set_description(f'Epoch: {epoch+1}/{num_epochs} | Loss: {running_loss/(i+1):.3f} | Acc: {100.*correct/total:.3f}%')
        
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100. * correct / total
        
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        scheduler.step()
        
        # 每10个epoch保存一次模型
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f'cifar10_model_epoch_{epoch+1}.pth')
    
    return train_losses, train_accs



# 评估模型
def evaluate_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    accuracy = 100. * correct / total
    print(f'测试准确率: {accuracy:.2f}%')
    
    return accuracy, all_preds, all_targets



# 训练模型
print("开始训练模型...")
train_losses, train_accs = train_model(model, 
                                       trainloader, 
                                       criterion, 
                                       optimizer, 
                                       scheduler, 
                                       num_epochs=20)

# 评估模型
print("评估模型...")
accuracy, predictions, targets = evaluate_model(model, testloader)


# 绘制训练过程
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Epoch-Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(train_accs)
plt.title('Epoch-Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.savefig('training_history.png')
plt.show()

# 保存最终模型
torch.save(model.state_dict(), 'cifar10_final_model.pth')


# 创建提交文件
def create_submission(model, testloader, filename='submission.csv'):
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for inputs, _ in tqdm(testloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
    
    # 创建提交文件
    submission = pd.DataFrame({
        'id': range(len(all_preds)),
        'label': all_preds
    })
    submission.to_csv(filename, index=False)
    print(f"提交文件已保存为 {filename}")

# 创建提交文件
create_submission(model, testloader)

print("完成!")


