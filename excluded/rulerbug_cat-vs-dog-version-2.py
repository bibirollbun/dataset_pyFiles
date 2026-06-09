import os
import torch
import numpy as np


class NasConfig():
    def __init__(self, dir):
        self.root_dir = dir
        self.train_dir = os.path.join(self.root_dir, "train")
        self.test_dir = os.path.join(self.root_dir, "samples")
        self.number_cls = 2
        self.cls_map = {"dog": 0, "cat": 1}
        self.index_map = {0: "dog", 1: "cat"}
        self.read_train()
        self.read_test()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def read_train(self):
        # 读取训练数据
        self.train_datas = []
        for file in os.listdir(self.train_dir):
            if file.endswith(".jpg"):
                img = os.path.join(self.train_dir, file)
                label = self.cls_map[file.split(".")[0]]
                self.train_datas.append([img, label])

    def read_test(self):
        # 读取训练数据
        self.test_datas = []
        for file in os.listdir(self.test_dir):
            if file.endswith(".jpg"):
                img = os.path.join(self.test_dir, file)
                idStr = file.split(".")[0]
                self.test_datas.append([img,int(idStr)])

     


ROOT_PATH = "/kaggle/input/vc-master-24-2-dogs-vs-cats"


config = NasConfig(ROOT_PATH)


from torch.utils.data import Dataset, DataLoader
from PIL import Image

class AnimoDataset(Dataset):
    def __init__(self,datas,transform=None):
        self.datas = datas
        self.transform = transform
    def __len__(self):
        return len(self.datas)
    def __getitem__(self,index):
        img_path,label = self.datas[index]
        image = Image.open(img_path).convert('RGB')

        # 应用变换
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label)
        return image, label


from torchvision import transforms

from torch.utils.data import random_split
def loader_create( batch_size=512,resize=(64, 64),split_ratio=0.80):
    transform_val = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_train = transforms.Compose([
        transforms.Resize(resize),
       transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 调整颜色
        transforms.RandomRotation(15),  # 随机旋转
        transforms.ToTensor(),  # 转为张量 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_size = int(len(config.train_datas) * split_ratio)
    test_size = len(config.train_datas) - train_size
    train_dataset, vaild_dataset = random_split(config.train_datas, [train_size, test_size])
    train_dataset = AnimoDataset(train_dataset,transform_train)
    test_dataset = AnimoDataset(config.test_datas,transform_val)   
    vaild_dataset = AnimoDataset(vaild_dataset,transform_val)   
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=1)
    vaild_dataset = torch.utils.data.DataLoader(vaild_dataset, batch_size=batch_size, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size*2, num_workers=1)
    return train_loader,vaild_dataset,test_loader


import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DiyModel(nn.Module):
    def __init__(self, number_class=10):
        super(DiyModel, self).__init__()
        self.number_class = number_class
        self.conv1 = nn.Sequential(
            ConvBlock(3, 64, kernel_size=3, stride=1, padding=0),
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=0),
            ConvBlock(128, 256, kernel_size=3, stride=2, padding=0),
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=0),
        )
        self.conv2 = nn.Sequential(

            ConvBlock(512, 256, kernel_size=3, stride=1, padding=0),
            ConvBlock(256, 128, kernel_size=3, stride=2, padding=0),
            ConvBlock(128, 64, kernel_size=3, stride=1, padding=0),
        )
        self.conv3 = nn.Sequential(
            ConvBlock(64 + 512, 256, kernel_size=3, stride=1, padding=0),
            ConvBlock(256, 128, kernel_size=3, stride=1, padding=0),
            ConvBlock(128, 64, kernel_size=3, stride=1, padding=0),
        )
        self.dropout = nn.Dropout(0.6)
        self.fc0 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, number_class)

    def forward(self, x):
        top = self.conv1(x)
        bottom = self.conv2(top)
        bottom = F.interpolate(bottom, top.size()[2:])
        out = torch.cat([top, bottom], 1)
        out = self.conv3(out)
        out = self.dropout(out)
        out = self.fc0(out)
        # print(out.size())
        out = out.view(out.size(0), -1)
        x = self.fc(out)
        return x



from tqdm import tqdm

def fit(model, train_l, test_l, epochs, lr, device, warmup_epochs=5):
    import torch
    from torch.optim.lr_scheduler import LambdaLR

    # 将模型移到指定设备
    model = model.to(device)
    
    # 定义损失函数和优化器
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=1e-4)

    # 定义学习率热身和后续衰减调度器
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # 热身阶段，线性增加学习率
        else:
            return 0.1 ** ((epoch - warmup_epochs) // 10)  # 后续衰减
    
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    for epoch in range(epochs):
        model.train()
        datas = tqdm(train_l)
        
        for imgs, labels in datas:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            l = loss(outputs, labels)
            l.backward()
            optimizer.step()
            datas.set_description(f"Epoch: {epoch + 1}/{epochs} Loss: {l.item():.4f}")
        
        print(f"Epoch: {epoch + 1} Loss: {l.item():.4f}")

        # 更新学习率
        scheduler.step()

        # 验证阶段
        with torch.no_grad():
            correct = 0
            count = 0
            model.eval()
            test_ls = tqdm(test_l)
            for imgs, labels in test_ls:
                imgs = imgs.to(device)
                labels = labels.to(device)
                count += len(labels)
                outputs = model(imgs)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                test_ls.set_description(f"Accuracy: {100 * correct / count:.2f}%")
            print(f"Epoch: {epoch + 1} Accuracy: {100 * correct / count:.2f}%")

    print("Training complete.")


train_loader,vaild_dataset,test_loader= loader_create(batch_size=64)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = DiyModel(config.number_cls)

model = model.to(device)


fit(model, train_loader,vaild_dataset,25,1e-3,device)


predictions = []
ids = []
model.eval()
with torch.no_grad():
    for t_x,index in test_loader:
        prediction = model(t_x .cuda())
        predictions.extend(prediction.cpu().detach().numpy())
        ids.extend(index)
predictions = np.array(predictions)
ids = np.array(ids)


predicted_labels = np.argmax(predictions, axis=1)


import pandas as pd
result =  pd.DataFrame()
result


predicted_str_labels = []
predicted_ids = ids
for l in predicted_labels:
    predicted_str_labels.append(config.index_map[l])


result["id"] = predicted_ids
result['label'] = predicted_str_labels



result


result.to_csv( '/kaggle/working/submission.csv', index = False )

