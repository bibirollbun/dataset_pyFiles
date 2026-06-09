import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# начнём с того, что в данной задаче очень интересный формат представления изображений
# научимся переводить его в более привычный для нас

train_data = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
test_data = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

train_data


test_data


train_data.iloc[0, 1:].values.shape


# проверим, удалось ли получить ожидаемый результат такими преобразованиями
plt.imshow((train_data.iloc[0, 1:].values).reshape(28, 28), cmap='gray');


val_data = train_data.iloc[48000 : ]
train_data = train_data.iloc[ : 48000]


# применим такую трансформацию ко всем изображениям

train_data_y = train_data['label']
train_data_y = train_data_y.reset_index(drop=True)
train_data = train_data.drop(columns=['label'])
train_data = train_data.iloc[:, :].values
shp = len(train_data)
train_data = train_data.reshape(shp, 28, 28)

val_data_y = val_data['label']
val_data_y = val_data_y.reset_index(drop=True)
val_data = val_data.drop(columns=['label'])
val_data = val_data.iloc[:, :].values
shp = len(val_data)
val_data = val_data.reshape(shp, 28, 28)

test_data_ID = test_data['id'].reset_index(drop=True)
test_data = test_data.drop(columns=['id'])
test_data = test_data.iloc[:, :].values
shp = len(test_data)
test_data = test_data.reshape(shp, 28, 28)


from torch.utils.data import Dataset, DataLoader

# стандартный класс
class Image(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = image.astype(np.uint8)

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            return image


train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(20),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(0.5),
    transforms.Normalize((0.5,), (0.5,))
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


train_dataset = Image(train_data, train_data_y, transform=train_transforms)
val_dataset = Image(val_data, val_data_y, transform=test_transforms)
test_dataset = Image(test_data, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)


# убедимся, что загрузка данных отработала корректно
iter_ = iter(train_loader)
images, labels = next(iter_)

fig, axes = plt.subplots(1, 10, figsize=(12, 4))
for i in range(10):
    img = images[i].numpy().squeeze()
    axes[i].imshow(img, cmap="gray")
    axes[i].set_title(f"Label: {labels[i].item()}")
    axes[i].axis("off")
plt.show()


# переходим к архитектуре; как уже говорилось ранее, мудрить слишком сильно не будем

n_classes = 10
class BasicBlockNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.shortcut = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, padding=0, stride=1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.relu_common = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(in_features=32 * 4 * 4, out_features=n_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu_common(out)
        out = self.avgpool(out)
        out = out.view(x.shape[0], -1)
        out = self.fc(out)
        return out

net = BasicBlockNet()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = net.to(device)


criterion = nn.CrossEntropyLoss()

def train_epoch(model, optimizer, train_loader):
    loss_log = []
    acc_log = []
    model.train()
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        
        loss_log.append(loss.item())
        
        pred = logits.argmax(dim=1)
        acc = (pred == target).sum() / target.shape[0]
        acc_log.append(acc.item()) 
        
        acc_log.append(acc.item()) 

    return loss_log, acc_log

def train(model, optimizer, n_epochs, train_loader, scheduler=None):
    train_loss_log, train_acc_log = [], []

    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, optimizer, train_loader)
        
        train_loss_log.extend(train_loss)
        train_acc_log.extend(train_acc)

        print(f"Epoch {epoch}")
        print(f" train loss: {np.mean(train_loss)}, train acc: {np.mean(train_acc)}")
        
        if scheduler is not None:
            scheduler.step()

    return train_loss_log, train_acc_log


optimizer = optim.Adam(net.parameters(), lr=0.005)  
tr_loss_log, tr_acc_log = train(net, optimizer, 10, train_loader)


# проверим качество на валидационной выборке

net.eval()
val_accuracy = []

with torch.no_grad():
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        logits = net(data)
        pred = logits.argmax(dim=1)
        accuracy = (pred == target).sum() / target.shape[0]
        val_accuracy.append(accuracy.item()) 

print(np.mean(val_accuracy))


# это было написано в надежде, что можно будет просто заслать submission.csv
net.eval()
predictions = []

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        logits = net(data)
        pred = logits.argmax(dim=1).cpu().numpy()
        predictions.extend(pred)

result = pd.DataFrame({"id": test_data_ID, "label": predictions})
result.to_csv('submission.csv', index=False)

