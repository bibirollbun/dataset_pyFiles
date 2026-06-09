# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 可视化

import seaborn as sns
import matplotlib.pyplot as plt

# PyTorch

import torch as T
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision import transforms


device = T.device('cuda' if T.cuda.is_available() else 'cpu')


# 读取文件
train_data = pd.read_csv("/kaggle/input/dog-breed-identification/labels.csv")
# Train data shape
print(f"Train dataset shape: {train_data.shape}")
# Sample of the train_data DataFrame
train_data.head()


breed_classes = train_data.breed.value_counts().reset_index()
plt.figure(figsize=(20,8))
sns.barplot(breed_classes, x='breed', y='count', palette="flare")
plt.xticks(rotation=90)
plt.title("Distribution of the breed classes")
plt.show()


breed_classes['count'].describe()


breed_classes['breed'].nunique()


le = LabelEncoder()
train_data.loc[:,'breed'] = le.fit_transform(train_data.loc[:,'breed']) 


train_data.head()


class Dog_Breed_Dataset(Dataset):
    
    def __init__(self, df: pd.DataFrame, img_base_path: str, split: str, transforms = None):        
        self.df = df
        self.img_base_path = img_base_path
        self.split = split
        self.transforms = transforms
        
    def __getitem__(self, index):
        # 图片路径
        img_path = os.path.join(self.img_base_path + self.df.loc[index,'id'] + '.jpg')
        # 读取图片
        img = Image.open(img_path)        
      
        if self.transforms:
            img = self.transforms(img)
        
        if self.split != 'test':
            y = self.df.loc[index, 'breed']                     
            return img, y
        else:
            img_id = self.df.loc[index, 'id']
            return img_id, img
    
    def __len__(self):
        return len(self.df)    


train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transforms = transforms.Compose([
    transforms.Resize((224,224)),    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



train, val = train_test_split(train_data, test_size=0.2, random_state=42, stratify=train_data['breed'])

train = train.reset_index(drop=True)
val = val.reset_index(drop=True)


train_dataset = Dog_Breed_Dataset(
    df=train,
    img_base_path='/kaggle/input/dog-breed-identification/train/',
    split='train',
    transforms=train_transforms
)
validation_dataset = Dog_Breed_Dataset(
    df=val,
    img_base_path='/kaggle/input/dog-breed-identification/train/',
    split='val',
    transforms=test_transforms
)

train_dl = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
validation_dl = DataLoader(validation_dataset, batch_size=64, shuffle=False, num_workers=4)


print(f"Train data length: {len(train_dl.dataset)}, Validation data length: {len(validation_dl.dataset)}")


import torch as T


def train_model(train_dl, val_dl, model, epochs=30):
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    
    val_loss_history = []
    train_recall_history = []
    val_recall_history = []
   
    best_val_loss = 1_000_000.0

    device = next(model.parameters()).device

    def evaluate(model, data_loader):
        correct_pred = 0
        total_loss = 0
        true_positives = 0
        actual_positives = 0
        model.eval()
        with T.no_grad():
            for x, y in data_loader:
                # Convert data to Tensor
                x = x.clone().detach().to(device)
                y = y.clone().detach().long().to(device)
                # 预测
                preds = model(x)
                # 计算损失
                loss = model.criterion(preds, y)
                total_loss += loss.item()
                # 计算正确的预测
                preds = T.argmax(preds, dim=1)
                correct_pred += (preds.long().unsqueeze(1) == y.unsqueeze(1)).sum().item()
                # Calculate true positives and actual positives for recall
                true_positives += ((preds == 1) & (y == 1)).sum().item()
                actual_positives += (y == 1).sum().item()
        accuracy = correct_pred / len(data_loader.dataset)
        average_loss = total_loss / len(data_loader)
        recall = true_positives / actual_positives if actual_positives > 0 else 0
        return accuracy, average_loss, recall

    for epoch in range(epochs):
        print("=" * 20, "Epoch: ", str(epoch), "=" * 20)

        train_correct_pred = 0
        train_loss = 0
        train_true_positives = 0
        train_actual_positives = 0

        # Set to training mode
        model.train()
        for x, y in train_dl:
            # Convert data to Tensor
            x = x.clone().detach().to(device).requires_grad_(True)
            y = y.clone().detach().long().to(device)
            # Reset gradients
            model.optim.zero_grad()
            # Predict
            preds = model(x)
            # Compute the loss
            loss = model.criterion(preds, y)
            # Compute the gradients
            loss.backward()
            # Update weights
            model.optim.step()
            # Count the correct predictions
            preds = T.argmax(preds, dim=1)
            train_correct_pred += (preds.long().unsqueeze(1) == y.unsqueeze(1)).sum().item()
            train_loss += loss.item()
            # Calculate true positives and actual positives for recall
            train_true_positives += ((preds == 1) & (y == 1)).sum().item()
            train_actual_positives += (y == 1).sum().item()

        train_acc = train_correct_pred / len(train_dl.dataset)
        train_loss /= len(train_dl)
        train_recall = train_true_positives / train_actual_positives if train_actual_positives > 0 else 0

        train_acc_history.append(train_acc)
        train_loss_history.append(train_loss)
        train_recall_history.append(train_recall)

        val_acc, val_loss, val_recall = evaluate(model, val_dl)
        model.scheduler.step()

        val_acc_history.append(val_acc)
        val_loss_history.append(val_loss)
        val_recall_history.append(val_recall)

        print(f"Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, Train Recall: {train_recall:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}, Val Recall: {val_recall:.4f}")

    return train_acc_history, val_acc_history, train_loss_history, val_loss_history, train_recall_history, val_recall_history, model


import torch
import torch.nn as nn
import torchvision.models as models

inception = models.inception_v3(weights=None)  


model_path = '/kaggle/input/inception/pytorch/default/1/inception_v3_google-0cc3c7bd.pth'  
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
inception.load_state_dict(torch.load(model_path))

inception_model = nn.Sequential(
    inception.Conv2d_1a_3x3,
    inception.Conv2d_2a_3x3,
    inception.Conv2d_2b_3x3,
    inception.maxpool1,
    inception.Conv2d_3b_1x1,
    inception.Conv2d_4a_3x3,
    inception.maxpool2,
    inception.Mixed_5b,
    inception.Mixed_5c,
    inception.Mixed_5d,
    inception.Mixed_6a,
    inception.Mixed_6b,
    inception.Mixed_6c,
    inception.Mixed_6d,
    inception.Mixed_6e,
    inception.Mixed_7a,
    inception.Mixed_7b,
    inception.Mixed_7c,
    inception.avgpool
)


model_path = '/kaggle/input/resnet/pytorch/default/1/resnet50-0676ba61.pth'

# 验证文件存在性
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# 加载本地模型
resnet50 = models.resnet50(weights=None)
resnet50.load_state_dict(torch.load(model_path))
resnet50_model = nn.Sequential(
    resnet50.conv1,
    resnet50.bn1,
    resnet50.relu,
    resnet50.maxpool,
    resnet50.layer1,
    resnet50.layer2,
    resnet50.layer3,
    resnet50.layer4,
    resnet50.avgpool
)


# Freeze parameters of pretrained models
for param in resnet50_model.parameters():    
    param.requires_grad = False
    
for param in inception_model.parameters():    
    param.requires_grad = False


class Model(nn.Module):
    def __init__(self, inception_model, resnet50_model):
        super(Model, self).__init__()

        self.inception_model = inception_model
        self.resnet50_model = resnet50_model

        self.output = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 120)
        )

        self.to('cuda' if T.cuda.is_available() else 'cpu')
        # Optimizer
        self.optim = T.optim.Adam(self.output.parameters(), lr=0.001)
        # Loss
        self.criterion = T.nn.CrossEntropyLoss()
        # Scheduler
        self.scheduler = T.optim.lr_scheduler.StepLR(self.optim, step_size=7, gamma=0.1)

    def forward(self, x):
        X1 = self.inception_model(x)
        X2 = self.resnet50_model(x)
        X1 = X1.view(X1.size(0), -1)
        X2 = X2.view(X2.size(0), -1)

        X = T.cat([X1, X2], dim=1)

        P = self.output(X)

        return P

    def get_weights(self):
        return self.output.state_dict()

    def load_weights(self, weights):
        self.output.load_state_dict(weights)


model = Model(inception_model, resnet50_model)


train_acc_history, val_acc_history, train_loss_history, val_loss_history, train_recall_history, val_recall_history, model = train_model(train_dl, validation_dl, model)


# 绘制准确率曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(range(len(train_acc_history)), train_acc_history, label="Training accuracy")
plt.plot(range(len(val_acc_history)), val_acc_history, label="Validation accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.grid(True)

# 绘制损失曲线
plt.subplot(1, 3, 2)
plt.plot(range(len(train_loss_history)), train_loss_history, label="Training Loss")
plt.plot(range(len(val_loss_history)), val_loss_history, label="Validation Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(range(len(train_recall_history)), train_recall_history, label="Training Recall")
plt.plot(range(len(val_recall_history)), val_recall_history, label="Validation Recall")
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Recall Over Epochs')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


#test_data = pd.DataFrame([])
# 获取测试目录下的所有文件名
test_dir = '/kaggle/input/dog-breed-identification/test/'
filenames = os.listdir(test_dir)

# 创建包含测试数据的 DataFrame
test_data = pd.DataFrame({'id': filenames})

# 去除文件名中的 .jpg 后缀
test_data['id'] = test_data['id'].str.replace('.jpg', '', regex=False)


# Dataset shape
print(f"Test dataset shape: {test_data.shape}")
# Sample of the train_data DataFrame
test_data.head()


test_dataset = Dog_Breed_Dataset(
    df=test_data,
    img_base_path='/kaggle/input/dog-breed-identification/test/',
    split='test',
    transforms=test_transforms
)

test_dl = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)


def test_model(test_dl, model):
    all_ids = []
    all_preds = []
    device = next(model.parameters()).device
    for ids, images in test_dl:
        all_ids.extend(ids)
        images = images.to(device)
        preds = model(images)
        prob_preds = T.nn.functional.softmax(preds, dim=1)
        prob_preds = prob_preds.detach().cpu().numpy()
        all_preds.extend(prob_preds)

    result_df = pd.DataFrame(all_preds)
    result_df.insert(0, 'id', all_ids)
    return result_df


test_preds = test_model(test_dl, model)
test_preds.shape


test_preds.head()


# Set columns to breed names
num_classes = []
for num_class in test_preds.columns:
    if num_class != 'id':  # 跳过id列
        num_classes.append(num_class)

num_classes = np.array(num_classes)
num_classes = le.inverse_transform(num_classes)
test_preds.columns = ['id'] + list(num_classes)  # 将id列放在最前面


test_preds.head()


test_preds.to_csv('submission.csv', index=None)







