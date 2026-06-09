!pip install py7zr


import py7zr

# Путь к архиву
archive_path = '/kaggle/input/cifar-10/train.7z'

# Папка для распаковки
extract_to = '/kaggle/working/'

# Распаковка архива
with py7zr.SevenZipFile(archive_path, mode='r') as z:
    z.extractall(path=extract_to)

print(f"Архив распакован в {extract_to}")



import py7zr

# Путь к архиву
archive_path = '/kaggle/input/cifar-10/test.7z'

# Папка для распаковки
extract_to = '/kaggle/working/'

# Распаковка архива
with py7zr.SevenZipFile(archive_path, mode='r') as z:
    z.extractall(path=extract_to)

print(f"Архив распакован в {extract_to}")


import pandas as pd
import numpy as np
labels = pd.read_csv('/kaggle/input/cifar-10/trainLabels.csv')
labels


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(labels['label'])

labels['label'] = le.transform(labels['label'])


import os
from torch.utils.data import Dataset
from PIL import Image


d={}

for i in range(len(labels)):
    d[str(labels.loc[i,'id'])] = labels.loc[i,'label']


class CustomImageDataset(Dataset):
    def __init__(self,image_dir,labels,transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir,fname) for fname in os.listdir(image_dir) if fname.endswith('.png')]
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self,idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        filename = os.path.basename(image_path)
        label = d[filename[0:len(filename)-4]]

        if self.transform:
            image = self.transform(image)

        return image,label



from torch.utils.data import DataLoader
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(((224,224))),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = CustomImageDataset(image_dir='/kaggle/working/train',labels=labels,transform = transform)
dataloader = DataLoader(dataset,batch_size=8,shuffle=True)




import torch.nn as nn
import torch.optim as optim


data_for_train = dataloader


import torch.nn.functional as F
import torch


from torchvision import models, transforms


import torch.nn as nn
import torch.nn.functional as F


# Модель
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.base = models.resnet18(pretrained=True)

        # Разморозим только последние слои
        for name, param in self.base.named_parameters():
            if 'layer4' in name or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)

    def forward(self, x):
        return self.base(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNetClassifier(num_classes=10).to(device)

# Loss + Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# Обучение
epochs = 2
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in data_for_train:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data_for_train):.4f}, Accuracy: {correct/total:.4f}")



sample = pd.read_csv('/kaggle/input/cifar-10/sampleSubmission.csv')


class ComfortableData(Dataset):
    def __init__(self,image_dir,transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.labels = labels
        self.image_paths = [os.path.join(image_dir,fname) for fname in os.listdir(image_dir) if fname.endswith('.png')]
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self,idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        filename = os.path.basename(image_path)
        
        if self.transform:
            image = self.transform(image)

        return image, os.path.basename(image_path)[:-4]



comfort_data = ComfortableData(image_dir='/kaggle/working/test',transform = transform)
data_for_test = DataLoader(comfort_data, batch_size = 8, shuffle=False)
    
model.eval()  # Важный шаг
preds = [0]*(len(sample)+1)

with torch.no_grad():
    for image, ID in data_for_test:
        image = image.to(device)
        predictions = model(image)  # (batch_size, 10)
        predicted_classes = torch.argmax(predictions, dim=1)  # (batch_size,)
        
        # Сохраняем предсказания по ID
        for i in range(len(ID)):
            preds[int(ID[i])] = le.inverse_transform([predicted_classes[i].item()])[0]
            
    


import pandas as pd

# Преобразуем в DataFrame
df = pd.DataFrame({
    'id': sample['id'],
    'label': preds[1:len(preds)]
})
df.to_csv("submission.csv", index=False)



for i in os.listdir('/kaggle/working/test'):
    print(i)
    break


