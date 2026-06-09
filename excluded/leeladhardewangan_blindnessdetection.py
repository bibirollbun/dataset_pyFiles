# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/train.csv")
test = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/test.csv")
sample = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/sample_submission.csv")


train.head()



train.loc[train['id_code']=='ef476be214d4']


train.shape


train.diagnosis.value_counts()


train.diagnosis.value_counts().plot(kind='pie', autopct="%1.1f%%")


sample.head()


sample.shape


test.shape


test


train['image_path'] =  "/kaggle/input/aptos2019-blindness-detection/train_images/" + train['id_code'] +'.png'


train


test['file_path'] = "/kaggle/input/aptos2019-blindness-detection/test_images/"+test['id_code'] +'.png'
test


train = train.rename(columns={'image_path':'file_path', 'diagnosis':'label'})
train


import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import resnet18


class EyeDataset(Dataset):
    def __init__(self, df, transform):
        self.df  = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx,'file_path']
        label = self.df.loc[idx, 'label']

        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std =[0.229,0.224,0.225])
])

dataset = EyeDataset(train, transform=transform)
train_laoder = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print(device)


model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, train['label'].nunique())
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)


epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct =  0
    total = 0

    for images, labels in train_laoder:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds==labels).sum().item()
        total += labels.size(0)

    acc = correct/total
    print(f"Epoch {epoch+1}/{epochs}, loss : {total_loss: .4f}, Acc: {acc:.4f}")

torch.save(model.state_dict(), "resnet_eye_model.pth")


val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class TestDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.loc[idx, 'file_path']
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image


test_ds = TestDataset(test, transform=val_tfms)
test_loader = DataLoader(test_ds, batch_size= 32, shuffle=False)

model.eval()
all_preds = []
with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds)

sub = pd.DataFrame({
    'id_code':test['file_path'],
    'prediction': all_preds
})
sub.head()



sub['prediction'].value_counts().plot(kind='bar')




