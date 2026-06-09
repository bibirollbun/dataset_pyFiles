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


import zipfile

#extracting the ZIP file
zip_path = '/kaggle/input/dogs-vs-cats/train.zip'
extract_dir = '/kaggle/working/'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
    
#extracting the ZIP file
zip_path = '/kaggle/input/dogs-vs-cats/test1.zip'
extract_dir = '/kaggle/working/'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import cv2
from PIL import Image
import torch.nn.functional as F


IMG_SIZE = 128

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class DogCatData(Dataset):
    def __init__(self, img_dir, IMG_SIZE = 32, transform = None):
        self.img_dir = img_dir
        self.file_names = sorted(os.listdir(img_dir))
        self.transform = transform
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        img = cv2.imread(os.path.join(self.img_dir, file_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 1 = dog, 0 = cat
        label = 0 if file_name.startswith("cat") else 1
        
        img = Image.fromarray(img)  # ðŸ”¥ THIS LINE FIXES THE ERROR

        transformed_img = self.transform(img)



        return transformed_img, label
        
    def __len__(self):
        return len(self.file_names)
        
        


working_dir = "/kaggle/working/train/"

train_data = DogCatData(working_dir, IMG_SIZE, transform)
train_loader = DataLoader(train_data, shuffle = True, batch_size = 4)


class DogCatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = (3,3), padding = 1) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size = (3,3), padding = 1) 
        self.fc1 = nn.Linear(128 * 128 * 64, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DogCatModel().to(device)


epoch = 5
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)



for i in range(epoch):
    model.train()
    total_loss = 0.0
    j = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        y_pred = model(images)
        
        loss = loss_fn(y_pred, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss
        if j%1000 == 0:
            print(j, "loss:", loss)
        j+=1
    print(f"Epoch: {i} Total loss: {total_loss}")
        



torch.save(model.state_dict(), "model.pth")


def sort_int(n):
    n = n.split(".")[0]
    return int(n)


class DogCatDataTest(Dataset):
    def __init__(self, img_dir, IMG_SIZE = 32, transform = None):
        self.img_dir = img_dir
        self.file_names = sorted(os.listdir(img_dir), key=sort_int)
        self.transform = transform
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        img = cv2.imread(os.path.join(self.img_dir, file_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)  # ðŸ”¥ THIS LINE FIXES THE ERROR
        transformed_img = self.transform(img)
        return file_name, transformed_img
        
    def __len__(self):
        return len(self.file_names)


working_dir = "/kaggle/working/test1/"

test_data = DogCatDataTest(working_dir, IMG_SIZE, transform)
test_loader = DataLoader(test_data, batch_size = 4)


name = []
value = []
l = 0
for file_name, images in test_loader:
    l+=1
    images = images.to(device)
    
    y_pred = model(images)

    predicted = torch.argmax(y_pred, dim=1)

    predicted_np = predicted.cpu().numpy()

    for temp in range(len(predicted_np)):
        name.append(int(file_name[temp].split(".")[0]))
        value.append(predicted_np[temp])
    
    
    if l%500 == 0:
        print(l)
    
print(len(li))


from matplotlib import pyplot as plt
loc = "/kaggle/working/test1/"
file_name = "1000.jpg"
img = cv2.imread(os.path.join(loc, file_name))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)


len(test_loader)








name_arr = np.array(name)
value_arr = np.array(value)

df = pd.DataFrame({"id" : name_arr, "label" : value_arr})
df.to_csv("foo.csv", index=False)


name[:10]




