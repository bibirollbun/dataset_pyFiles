ls /kaggle/input/cifar-10


# !7z x /kaggle/input/cifar-10/train.7z -o/kaggle/working/train


import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import random

from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch.optim as optim
from tqdm import tqdm
import torchvision.datasets as dsets
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

print(torch.__version__)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device


CLASS_TO_IDX = {
    'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
    'dog': 5,      'frog': 6,       'horse': 7, 'ship': 8, 'truck': 9
}

IDX_TO_CLASS = {v: k for k,v in CLASS_TO_IDX.items()}

composed_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465] , std=[0.247, 0.243, 0.261]), 
    transforms.RandomErasing(),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465] , std=[0.247, 0.243, 0.261]), 
])


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[idx, 0]) + '.png')
        image = Image.open(img_path).convert("RGB")
        label = self.img_labels.iloc[idx, 1]
        label = CLASS_TO_IDX[label]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# cifar = CustomImageDataset(
#     annotations_file='/kaggle/input/cifar10-7z-extraction/trainLabels.csv', 
#     img_dir='/kaggle/input/cifar10-7z-extraction/train', 
#     transform=composed_transform
# )

# train_dataset =  dsets.CIFAR10(root='./data', train=True, download=True, transform = composed_transform)
# validation_dataset = dsets.CIFAR10(root='./data', train=False, download=True, transform = val_transform)

# plt.imshow(train_dataset[0][0].permute(1,2,0))
# print(IDX_TO_CLASS[train_dataset[0][1]])

# train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
# val_loader = DataLoader(dataset=validation_dataset, batch_size=256, shuffle=False)


cifar_size = len(os.listdir('/kaggle/input/cifar10-7z-extraction/train'))
all_indices = np.arange(cifar_size)
train_idx, val_idx = train_test_split(all_indices, train_size=0.8, random_state=42, shuffle=True)

train_subset = Subset(
    CustomImageDataset(
        annotations_file='/kaggle/input/cifar10-7z-extraction/trainLabels.csv', 
        img_dir='/kaggle/input/cifar10-7z-extraction/train', 
        transform=composed_transform
    ),
    train_idx
)
val_subset = Subset(
    CustomImageDataset(
        annotations_file='/kaggle/input/cifar10-7z-extraction/trainLabels.csv', 
        img_dir='/kaggle/input/cifar10-7z-extraction/train', 
        transform=val_transform
    ),
    val_idx
)

train_loader = DataLoader(train_subset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=256, shuffle=False)


plt.imshow(train_subset[0][0].permute(1,2,0))
print(IDX_TO_CLASS[train_subset[0][1]])


class ResNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, stride=1):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bnorm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bnorm2 = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        residual = x
        x = self.relu(self.bnorm1(self.conv1(x)))
        x = self.bnorm2(self.conv2(x))
        # if self.conv1x1:
        #     residual = self.conv1x1(residual)
        return self.relu(x + self.skip(residual))
    

class Model(nn.Module):
    def __init__(self, num_classes=10):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnrom = nn.BatchNorm2d(64)
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res_block1a = ResNet(64, 64)
        self.res_block1b = ResNet(64, 64)

        self.res_block2a = ResNet(64, 128, stride=1)
        self.res_block2b = ResNet(128, 128)

        self.res_block3a = ResNet(128, 256, stride=1)
        self.res_block3b = ResNet(256, 256)

        self.res_block4a = ResNet(256, 512, stride=1)
        self.res_block4b = ResNet(512, 512)

        self.global_AvgPool = nn.AdaptiveAvgPool2d((1,1))
        # self.fc1 = nn.Linear(512, 1024)
        # self.fc2 = nn.Linear(1024, 1024)
        # self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, num_classes) # Most modern ResNet‑style models do this: GlobalAvgPool → Linear(512→10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Suitable initialization for ReLU
                nn.init.normal_(m.weight, mean=0.0, std=0.05)
                nn.init.constant_(m.bias, 0.0)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.bnrom(self.conv1(x)))
        x = self.res_block1b(self.res_block1a(x))
        x = self.res_block2b(self.maxPool(self.res_block2a(x)))
        x = self.res_block3b(self.maxPool(self.res_block3a(x)))
        x = self.res_block4b(self.maxPool(self.res_block4a(x)))

        # use adaptive_avg_pool2d 1.1k to achieve global average pooling, 
        # just set the output size to (1, 1)
        x = self.global_AvgPool(x)
        x = torch.flatten(x, 1)
        # x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        x = self.fc4(self.dropout(x))
        return x


lr = 3e-4

model = Model().to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)


epochs = 100
train_loss_lst = []
train_acc_lst = []
val_loss_lst = []
val_acc_lst = []

def freeze_bn(m):
    if isinstance(m, nn.BatchNorm2d):
        m.eval()
        for p in m.parameters():
            p.requires_grad = False


for epoch in tqdm(range(epochs)):
    train_loss = 0.0
    train_acc = 0.0
    count = 0

    model.train()
    if epoch == 5:
        model.apply(freeze_bn)
        
    for X_train, y_train in train_loader:
        X_train, y_train = X_train.float().to(device), y_train.to(device)
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += (torch.argmax(outputs, 1) == y_train).sum().item()
        count += len(y_train)
    scheduler.step()

    train_loss /= len(train_loader)
    train_loss_lst.append(train_loss)
    train_acc /= count
    train_acc_lst.append(train_acc)

    val_loss = 0.0
    val_acc = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.float().to(device), y_val.to(device)
            outputs = model(X_val)
            loss = criterion(outputs, y_val)
            val_loss += loss.item()
            val_acc += (torch.argmax(outputs, 1) == y_val).sum().item()
            count += len(y_val)

    val_loss /= len(val_loader)
    val_loss_lst.append(val_loss)
    val_acc /= count
    val_acc_lst.append(val_acc)

    print(f"EPOCH {epoch+1}/{epochs}, Train_Loss: {train_loss:.4f}, Train_Acc: {train_acc:.4f}, Validation Loss: {val_loss:.4f}, Val_Acc: {val_acc:.4f}")


fig, ax = plt.subplots(2, 2, figsize=(12, 10))
ax[0, 0].plot(train_loss_lst, color='green')
ax[0, 0].set(xlabel='Epoch', ylabel='Loss')
ax[0, 0].set_title('Training Loss')

ax[0, 1].plot(val_loss_lst, color='orange')
ax[0, 1].set(xlabel='Epoch', ylabel='Loss')
ax[0, 1].set_title('Validation Loss')

ax[1, 0].plot(train_acc_lst, color='green')
ax[1, 0].set(xlabel='Epoch', ylabel='Accuracy')
ax[1, 0].set_title('Training Accuracy')

ax[1, 1].plot(val_acc_lst, color='orange')
ax[1, 1].set(xlabel='Epoch', ylabel='Accuracy')
ax[1, 1].set_title('Validation Accuracy')

plt.show()


class TestDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = annotations_file
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(self.img_labels[idx]))
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0


sample = pd.read_csv('/kaggle/input/cifar10-7z-extraction/sampleSubmission.csv')
test_files = sample["id"].astype(str) + ".png"
cifar_test = TestDataset(
    annotations_file=test_files.tolist(), 
    img_dir='/kaggle/input/cifar10-7z-extraction/test', 
    transform=val_transform
)
test_loader = DataLoader(cifar_test, batch_size=256, shuffle=False)


preds = []

model.eval()
with torch.no_grad():
    for X_test, _ in test_loader:
        X_test = X_test.to(device)
        labels = model(X_test).argmax(dim=1).type(torch.int32).cpu().tolist()
        preds.extend(labels)


sample["label"] = [IDX_TO_CLASS[prediction] for prediction in preds]
sample.to_csv("submission.csv", index=False)

