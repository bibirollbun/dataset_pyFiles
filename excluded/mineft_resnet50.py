import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
import torchvision
from torch.utils.data import random_split
import torch.optim as optim





data = pd.read_csv('/kaggle/input/classify-leaves/train.csv')


label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])


transforms_train = torchvision.transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.5, 1),ratio=(3/4, 4/3)),
    transforms.ToTensor()
])
transforms_test = torchvision.transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        image = Image.open(os.path.join("/kaggle/input/classify-leaves", image_path))
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# 创建自定义数据集
dataset = CustomDataset(data, transform=None)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(38))

#  对训练集应用数据增强的transform
train_dataset.dataset.transform = transforms_train

#  对测试集应用测试的transform
test_dataset.dataset.transform = transforms_test

train_iter = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_iter = DataLoader(test_dataset, batch_size=64, shuffle=False)
for X,y in train_iter:
    print(X.shape, y.shape)
    print(X.dtype, y.dtype)
    print(y)
    break


def train(net,train_loader,valid_loader,num_epochs,lr,device):
    epoch = num_epochs
    losses = []
    optimizer = optim.SGD(net.parameters(), lr, weight_decay=1e-3)
    loss = nn.CrossEntropyLoss(reduction='mean')
    for i in range(epoch):
        acc = 0
        loss_sum = 0
        for x, y in train_loader:
            net = net.to(device)
            x = torch.as_tensor(x, dtype=torch.float)
            x = x.to(device)
            y = y.to(device)
            y_hat = net(x)
            loss_temp = loss(y_hat, y)
            loss_sum += loss_temp
            optimizer.zero_grad()
            loss_temp.backward()
            optimizer.step()
            acc += torch.sum(y_hat.argmax(dim=1).type(y.dtype) == y)
        losses.append(loss_sum.cpu().detach().numpy() / len(train_loader))
        print( "epoch: ", i, "loss=", loss_sum.item(), "训练集准确度=",(acc/(len(train_loader)*train_loader.batch_size)).item(),end="")
        test_acc = 0
        for x, y in valid_loader:
            x = x.to(device)
            x = torch.as_tensor(x, dtype=torch.float)
            y = y.to(device)
            y_hat = net(x)
            test_acc += torch.sum(y_hat.argmax(dim=1).type(y.dtype) == y)
        print("验证集准确度", (test_acc / (len(valid_loader)*valid_loader.batch_size)).item())
       


lr, num_epochs = 0.01, 20
net = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
in_features = net.fc.in_features
net.fc = nn.Linear(in_features, 176)
train(net, train_iter, test_iter, num_epochs, lr, device = torch.device("cuda:0"))


test = pd.read_csv('/kaggle/input/classify-leaves/test.csv')
class testDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 0]
        image = Image.open(os.path.join("/kaggle/input/classify-leaves", image_path))
        if self.transform:
            image = self.transform(image)
        return image

# 创建自定义数据集
test_dataset = testDataset(test, transform=transforms_test)
test_iter = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda:0")
predict = torch.tensor([]).to(device)

with torch.no_grad():
    for x in test_iter:
        net = net.to(device)
        x = x.to(device)
        x = torch.as_tensor(x, dtype=torch.float)
        y_hat = net(x)
        predict = torch.cat((y_hat, predict), dim=0)
    predict = torch.argmax(predict, dim=1).reshape(-1)
predict_cpu = predict.cpu().numpy()
result = pd.read_csv("/kaggle/input/classify-leaves/test.csv")
result["label"] = pd.Series(predict_cpu)
result["label"] = label_encoder.inverse_transform(result["label"])
result.to_csv("result.csv", index=False)


print(result)

