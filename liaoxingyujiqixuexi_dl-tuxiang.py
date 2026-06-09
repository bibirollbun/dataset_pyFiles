pip install d2l


pip install py7zr


pip install efficientnet_pytorch


import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet18
import numpy as np
import pandas as pd

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)


# 1. 数据预处理
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop(224, scale=(0.64, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载CIFAR10数据集
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

# 划分训练集和验证集
n = len(train_data)
indices = torch.randperm(n)
split = int(n * 0.9)
train_subset = torch.utils.data.Subset(train_data, indices[:split])
valid_subset = torch.utils.data.Subset(train_data, indices[split:])

batch_size = 64
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

class_names = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Boat', 'Truck']


print("\n查看数据集信息...")
print(f"训练集样本数: {len(train_subset)}")
print(f"验证集样本数: {len(valid_subset)}")
print(f"测试集样本数: {len(test_data)}")
print("\n类别分布:")

# 统计训练集类别分布
train_labels = [train_data.targets[i] for i in train_subset.indices]
train_class_counts = {class_name: train_labels.count(i) for i, class_name in enumerate(class_names)}
print("训练集:", train_class_counts)

# 统计验证集类别分布
valid_labels = [train_data.targets[i] for i in valid_subset.indices]
valid_class_counts = {class_name: valid_labels.count(i) for i, class_name in enumerate(class_names)}
print("验证集:", valid_class_counts)

# 统计测试集类别分布
test_labels = test_data.targets
test_class_counts = {class_name: test_labels.count(i) for i, class_name in enumerate(class_names)}
print("测试集:", test_class_counts)


# 添加查看处理前后图像的代码
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """反归一化图像以便显示"""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

print("\n显示原始图像和处理后图像对比...")
# 获取一个批次的原始图像和处理后图像
original_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

original_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=original_transform)
sample_indices = torch.randperm(len(original_data))[:8]  # 随机选择8个样本

plt.figure(figsize=(12, 6))
for i, idx in enumerate(sample_indices):
    # 原始图像
    original_img, label = original_data[idx]
    plt.subplot(2, 8, i+1)
    plt.imshow(original_img.permute(1, 2, 0))
    plt.title(f"Original\n{class_names[label]}")
    plt.axis('off')
    
    # 处理后的图像
    processed_img, _ = train_data[idx]
    processed_img = denormalize(processed_img)  # 反归一化
    plt.subplot(2, 8, i+9)
    plt.imshow(processed_img.permute(1, 2, 0))
    plt.title(f"Processed\n{class_names[label]}")
    plt.axis('off')

plt.tight_layout()
plt.show()


# 2. 模型准备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 准备EfficientNet模型
print("\n准备EfficientNet模型...")
effnet = EfficientNet.from_pretrained('efficientnet-b0')
# 冻结所有层
for param in effnet.parameters():
    param.requires_grad = False
# 解冻最后几层
for param in effnet._conv_head.parameters():
    param.requires_grad = True
for param in effnet._bn1.parameters():
    param.requires_grad = True
for param in effnet._fc.parameters():
    param.requires_grad = True
# 修改最后的全连接层
effnet._fc = nn.Linear(effnet._fc.in_features, 10)
effnet = effnet.to(device)



# 准备ResNet模型
print("\n准备ResNet模型...")
resnet = resnet18(pretrained=True)
# 冻结所有层
for param in resnet.parameters():
    param.requires_grad = False
# 解冻最后几层
for param in resnet.layer4.parameters():
    param.requires_grad = True
for param in resnet.fc.parameters():
    param.requires_grad = True
# 修改最后的全连接层
resnet.fc = nn.Linear(resnet.fc.in_features, 10)
resnet = resnet.to(device)


# 3. 训练配置
criterion = nn.CrossEntropyLoss()
epochs = 10
effnet_train_losses = []
effnet_train_accs = []
effnet_val_accs = []
resnet_train_losses = []
resnet_train_accs = []
resnet_val_accs = []
final_train_losses = []
final_train_accs = []


# 4. 训练EfficientNet
print("\n开始训练EfficientNet...")
effnet_optimizer = optim.Adam(filter(lambda p: p.requires_grad, effnet.parameters()), lr=1e-3, weight_decay=5e-4)
effnet_scheduler = optim.lr_scheduler.StepLR(effnet_optimizer, step_size=2, gamma=0.9)
best_effnet_acc = 0.0

for epoch in range(epochs):
    effnet.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader, desc=f"EfficientNet 第 {epoch+1} 轮"):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        effnet_optimizer.zero_grad(set_to_none=True)
        outputs = effnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        effnet_optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    effnet_scheduler.step()
    train_loss = total_loss / len(train_loader)
    train_acc = correct / total
    
    # 记录训练损失和准确率
    effnet_train_losses.append(train_loss)
    effnet_train_accs.append(train_acc)
    
    # 验证
    effnet.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = effnet(inputs)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total += labels.size(0)
    
    val_acc = val_correct / val_total
    effnet_val_accs.append(val_acc)
    
    if val_acc > best_effnet_acc:
        best_effnet_acc = val_acc
        torch.save(effnet.state_dict(), 'best_effnet.pth')
    
    print(f"第 {epoch+1} 轮: 损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}, 验证准确率: {val_acc:.4f}")


# 5. 训练ResNet
print("\n开始训练ResNet...")
resnet_optimizer = optim.Adam(filter(lambda p: p.requires_grad, resnet.parameters()), lr=1e-3, weight_decay=5e-4)
resnet_scheduler = optim.lr_scheduler.StepLR(resnet_optimizer, step_size=2, gamma=0.9)
best_resnet_acc = 0.0

for epoch in range(epochs):
    resnet.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader, desc=f"ResNet 第 {epoch+1} 轮"):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        resnet_optimizer.zero_grad(set_to_none=True)
        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        resnet_optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    resnet_scheduler.step()
    train_loss = total_loss / len(train_loader)
    train_acc = correct / total
    
    # 记录训练损失和准确率
    resnet_train_losses.append(train_loss)
    resnet_train_accs.append(train_acc)
    
    # 验证
    resnet.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = resnet(inputs)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total += labels.size(0)
    
    val_acc = val_correct / val_total
    resnet_val_accs.append(val_acc)
    
    if val_acc > best_resnet_acc:
        best_resnet_acc = val_acc
        torch.save(resnet.state_dict(), 'best_resnet.pth')
    
    print(f"第 {epoch+1} 轮: 损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}, 验证准确率: {val_acc:.4f}")


# 绘制EfficientNet和ResNet的训练曲线
plt.figure(figsize=(12, 5))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(effnet_train_losses, label='EfficientNet Training loss')
plt.plot(resnet_train_losses, label='ResNet Training loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training loss curve')
plt.legend()

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(effnet_train_accs, label='EfficientNet Training accuracy')
plt.plot(effnet_val_accs, label='EfficientNet Verify accuracy', linestyle='--')
plt.plot(resnet_train_accs, label='ResNet Training accuracy')
plt.plot(resnet_val_accs, label='ResNet Verify accuracy', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Curves')
plt.legend()

plt.tight_layout()
plt.show()


# 6. 选择最优模型
if best_effnet_acc > best_resnet_acc:
    best_model = effnet
    best_model.load_state_dict(torch.load('best_effnet.pth'))
    print("\n选择EfficientNet作为最优模型")
else:
    best_model = resnet
    best_model.load_state_dict(torch.load('best_resnet.pth'))
    print("\n选择ResNet作为最优模型")


# 7. 在完整训练集上训练最优模型
print("\n在完整训练集上训练最优模型...")
full_train_loader = DataLoader(
    torch.utils.data.ConcatDataset([train_loader.dataset, valid_loader.dataset]),
    batch_size=64, shuffle=True, num_workers=4, pin_memory=True
)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, best_model.parameters()), lr=1e-3, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

for epoch in range(10):
    best_model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(full_train_loader, desc=f"完整训练 第 {epoch+1} 轮"):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        outputs = best_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    scheduler.step()
    train_loss = total_loss / len(full_train_loader)
    train_acc = correct / total
    
    # 记录最终模型的训练损失和准确率
    final_train_losses.append(train_loss)
    final_train_accs.append(train_acc)
    
    print(f"第 {epoch+1} 轮: 损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")


# 绘制最终模型的训练曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(final_train_losses, label='Training loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('The final model training loss curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(final_train_accs, label='Training accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('The final model training accuracy curve')
plt.legend()

plt.tight_layout()
plt.show()

# 保存最终模型
torch.save(best_model.state_dict(), 'final_best_model.pth')
print("已保存最终模型: final_best_model.pth")


# 8. 在测试集上评估
print("\n在测试集上评估最优模型...")
best_model.eval()
test_correct = 0
test_total = 0
all_preds = []
all_labels = []
all_probs = []
all_images = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = best_model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        test_correct += predicted.eq(labels).sum().item()
        test_total += labels.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_images.extend(inputs.cpu().numpy())

test_acc = test_correct / test_total
print(f"测试准确率: {test_acc:.4f}")


# 10. 输出分类报告和混淆矩阵
print("\n分类报告:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# 显示混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(all_labels, all_preds), 
            annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.show()


# 9. 保存预测结果
print("\n保存预测结果...")
results = []
for i in range(len(all_preds)):
    results.append({
        '真实标签': class_names[all_labels[i]],
        '预测标签': class_names[all_preds[i]],
        '预测概率': max(all_probs[i]),
        '是否正确': all_labels[i] == all_preds[i]
    })

results_df = pd.DataFrame(results)
results_df.to_csv('predictions_results.csv', index=False, encoding='utf-8-sig')
print("预测结果已保存到 predictions_results.csv")


import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet18
import numpy as np
import pandas as pd

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)

# 1. 数据准备
print("准备数据...")
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop(224, scale=(0.64, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载CIFAR10数据集
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

# 划分训练集和验证集
n = len(train_data)
indices = torch.randperm(n)
split = int(n * 0.9)
train_subset = torch.utils.data.Subset(train_data, indices[:split])
valid_subset = torch.utils.data.Subset(train_data, indices[split:])

batch_size = 64
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

class_names = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Boat', 'Truck']

# 添加数据查看代码
print("\n查看数据集信息...")
print(f"训练集样本数: {len(train_subset)}")
print(f"验证集样本数: {len(valid_subset)}")
print(f"测试集样本数: {len(test_data)}")
print("\n类别分布:")

# 统计训练集类别分布
train_labels = [train_data.targets[i] for i in train_subset.indices]
train_class_counts = {class_name: train_labels.count(i) for i, class_name in enumerate(class_names)}
print("训练集:", train_class_counts)

# 统计验证集类别分布
valid_labels = [train_data.targets[i] for i in valid_subset.indices]
valid_class_counts = {class_name: valid_labels.count(i) for i, class_name in enumerate(class_names)}
print("验证集:", valid_class_counts)

# 统计测试集类别分布
test_labels = test_data.targets
test_class_counts = {class_name: test_labels.count(i) for i, class_name in enumerate(class_names)}
print("测试集:", test_class_counts)

# 添加查看处理前后图像的代码
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """反归一化图像以便显示"""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

print("\n显示原始图像和处理后图像对比...")
# 获取一个批次的原始图像和处理后图像
original_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

original_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=original_transform)
sample_indices = torch.randperm(len(original_data))[:8]  # 随机选择8个样本

plt.figure(figsize=(12, 6))
for i, idx in enumerate(sample_indices):
    # 原始图像
    original_img, label = original_data[idx]
    plt.subplot(2, 8, i+1)
    plt.imshow(original_img.permute(1, 2, 0))
    plt.title(f"Original\n{class_names[label]}")
    plt.axis('off')
    
    # 处理后的图像
    processed_img, _ = train_data[idx]
    processed_img = denormalize(processed_img)  # 反归一化
    plt.subplot(2, 8, i+9)
    plt.imshow(processed_img.permute(1, 2, 0))
    plt.title(f"Processed\n{class_names[label]}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# 2. 模型准备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 准备EfficientNet模型
print("\n准备EfficientNet模型...")
effnet = EfficientNet.from_pretrained('efficientnet-b0')
# 冻结所有层
for param in effnet.parameters():
    param.requires_grad = False
# 解冻最后几层
for param in effnet._conv_head.parameters():
    param.requires_grad = True
for param in effnet._bn1.parameters():
    param.requires_grad = True
for param in effnet._fc.parameters():
    param.requires_grad = True
# 修改最后的全连接层
effnet._fc = nn.Linear(effnet._fc.in_features, 10)
effnet = effnet.to(device)

# 准备ResNet模型
print("\n准备ResNet模型...")
resnet = resnet18(pretrained=True)
# 冻结所有层
for param in resnet.parameters():
    param.requires_grad = False
# 解冻最后几层
for param in resnet.layer4.parameters():
    param.requires_grad = True
for param in resnet.fc.parameters():
    param.requires_grad = True
# 修改最后的全连接层
resnet.fc = nn.Linear(resnet.fc.in_features, 10)
resnet = resnet.to(device)

# 3. 训练配置
criterion = nn.CrossEntropyLoss()
epochs = 5

# 初始化用于存储损失和准确率的列表
effnet_train_losses = []
effnet_train_accs = []
effnet_val_accs = []

resnet_train_losses = []
resnet_train_accs = []
resnet_val_accs = []

final_train_losses = []
final_train_accs = []

# 4. 训练EfficientNet
print("\n开始训练EfficientNet...")
effnet_optimizer = optim.Adam(filter(lambda p: p.requires_grad, effnet.parameters()), lr=1e-3, weight_decay=5e-4)
effnet_scheduler = optim.lr_scheduler.StepLR(effnet_optimizer, step_size=2, gamma=0.9)
best_effnet_acc = 0.0

for epoch in range(epochs):
    effnet.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader, desc=f"EfficientNet 第 {epoch+1} 轮"):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        effnet_optimizer.zero_grad(set_to_none=True)
        outputs = effnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        effnet_optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    effnet_scheduler.step()
    train_loss = total_loss / len(train_loader)
    train_acc = correct / total
    
    # 记录训练损失和准确率
    effnet_train_losses.append(train_loss)
    effnet_train_accs.append(train_acc)
    
    # 验证
    effnet.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = effnet(inputs)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total += labels.size(0)
    
    val_acc = val_correct / val_total
    effnet_val_accs.append(val_acc)
    
    if val_acc > best_effnet_acc:
        best_effnet_acc = val_acc
        torch.save(effnet.state_dict(), 'best_effnet.pth')
    
    print(f"第 {epoch+1} 轮: 损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}, 验证准确率: {val_acc:.4f}")

# 5. 训练ResNet
print("\n开始训练ResNet...")
resnet_optimizer = optim.Adam(filter(lambda p: p.requires_grad, resnet.parameters()), lr=1e-3, weight_decay=5e-4)
resnet_scheduler = optim.lr_scheduler.StepLR(resnet_optimizer, step_size=2, gamma=0.9)
best_resnet_acc = 0.0

for epoch in range(epochs):
    resnet.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader, desc=f"ResNet 第 {epoch+1} 轮"):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        resnet_optimizer.zero_grad(set_to_none=True)
        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        resnet_optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    resnet_scheduler.step()
    train_loss = total_loss / len(train_loader)
    train_acc = correct / total
    
    # 记录训练损失和准确率
    resnet_train_losses.append(train_loss)
    resnet_train_accs.append(train_acc)
    
    # 验证
    resnet.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = resnet(inputs)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total += labels.size(0)
    
    val_acc = val_correct / val_total
    resnet_val_accs.append(val_acc)
    
    if val_acc > best_resnet_acc:
        best_resnet_acc = val_acc
        torch.save(resnet.state_dict(), 'best_resnet.pth')
    
    print(f"第 {epoch+1} 轮: 损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}, 验证准确率: {val_acc:.4f}")

# 绘制EfficientNet和ResNet的训练曲线
plt.figure(figsize=(12, 5))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(effnet_train_losses, label='EfficientNet Training loss')
plt.plot(resnet_train_losses, label='ResNet Training loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training loss curve')
plt.legend()

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(effnet_train_accs, label='EfficientNet Training accuracy')
plt.plot(effnet_val_accs, label='EfficientNet Verify accuracy', linestyle='--')
plt.plot(resnet_train_accs, label='ResNet Training accuracy')
plt.plot(resnet_val_accs, label='ResNet Verify accuracy', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Curves')
plt.legend()

plt.tight_layout()
plt.show()

# 6. 选择最优模型
if best_effnet_acc > best_resnet_acc:
    best_model = effnet
    best_model.load_state_dict(torch.load('best_effnet.pth'))
    print("\n选择EfficientNet作为最优模型")
else:
    best_model = resnet
    best_model.load_state_dict(torch.load('best_resnet.pth'))
    print("\n选择ResNet作为最优模型")

# 7. 在完整训练集上训练最优模型
print("\n在完整训练集上训练最优模型...")
full_train_loader = DataLoader(
    torch.utils.data.ConcatDataset([train_loader.dataset, valid_loader.dataset]),
    batch_size=64, shuffle=True, num_workers=4, pin_memory=True
)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, best_model.parameters()), lr=1e-3, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

for epoch in range(10):
    best_model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(full_train_loader, desc=f"完整训练 第 {epoch+1} 轮"):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        outputs = best_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    scheduler.step()
    train_loss = total_loss / len(full_train_loader)
    train_acc = correct / total
    
    # 记录最终模型的训练损失和准确率
    final_train_losses.append(train_loss)
    final_train_accs.append(train_acc)
    
    print(f"第 {epoch+1} 轮: 损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")

# 绘制最终模型的训练曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(final_train_losses, label='Training loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('The final model training loss curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(final_train_accs, label='Training accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('The final model training accuracy curve')
plt.legend()

plt.tight_layout()
plt.show()

# 保存最终模型
torch.save(best_model.state_dict(), 'final_best_model.pth')
print("已保存最终模型: final_best_model.pth")

# 8. 在测试集上评估
print("\n在测试集上评估最优模型...")
best_model.eval()
test_correct = 0
test_total = 0
all_preds = []
all_labels = []
all_probs = []
all_images = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = best_model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        test_correct += predicted.eq(labels).sum().item()
        test_total += labels.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_images.extend(inputs.cpu().numpy())

test_acc = test_correct / test_total
print(f"测试准确率: {test_acc:.4f}")

# 9. 保存预测结果
print("\n保存预测结果...")
results = []
for i in range(len(all_preds)):
    results.append({
        '真实标签': class_names[all_labels[i]],
        '预测标签': class_names[all_preds[i]],
        '预测概率': max(all_probs[i]),
        '是否正确': all_labels[i] == all_preds[i]
    })

results_df = pd.DataFrame(results)
results_df.to_csv('predictions_results.csv', index=False, encoding='utf-8-sig')
print("预测结果已保存到 predictions_results.csv")

# 10. 输出分类报告和混淆矩阵
print("\n分类报告:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# 显示混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(all_labels, all_preds), 
            annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.show()

