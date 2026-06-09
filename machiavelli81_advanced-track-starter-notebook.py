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
        break

import tqdm

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class SmallNet(nn.Module):
    def __init__(self, num_classes=86):
        super(SmallNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),  # 32
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
        )
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=1, stride=2),
            nn.BatchNorm2d(24),
        )
        
        self.layer2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=3, stride=2, padding=1),  # 16
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.Conv2d(36, 36, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(36),
        )
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(24, 36, kernel_size=1, stride=2),
            nn.BatchNorm2d(36),
        )
        
        self.layer3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=3, stride=2, padding=1),  # 8
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
        )
        self.shortcut3 = nn.Sequential(
            nn.Conv2d(36, 48, kernel_size=1, stride=2),
            nn.BatchNorm2d(48),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(48, num_classes)
    
    def forward(self, x):
        # print(x.shape)
        # return self.fc1(x.view(x.size(0), -1))
        out = self.relu(self.bn1(self.conv1(x)))
        identity = out
        
        out = self.layer1(out)
        out += self.shortcut1(identity)
        out = self.relu(out)
        identity = out
        
        out = self.layer2(out)
        out += self.shortcut2(identity)
        out = self.relu(out)
        identity = out
        
        out = self.layer3(out)
        out += self.shortcut3(identity)
        out = self.relu(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out


model = SmallNet(num_classes = 10)
model.load_state_dict(torch.load("/kaggle/input/student-model/pytorch/default/1/student_model.pth",
                                 map_location="cpu"))


# ============================================================
# DO NOT REMOVE - COUNT MODEL PARAMETERS
try:
    total_params = sum(p.numel() for p in model.parameters())
except NameError:
    total_params = -1
    print("Model not defined yet; total_params set to -1 as placeholder.")
# ============================================================


# ============================================================
# DO NOT REMOVE - WRITE meta.txt
with open("meta.txt", "w") as f:
    f.write(f"Total parameters: {total_params}\n")
    f.write("Input image size: 256x256\n")
# ============================================================


import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.transforms import ToTensor

class ImageData(Dataset):
    def __init__(self, csv_file, augment=False):
        self.data = pd.read_csv(csv_file)
        self.augment = augment
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        image = Image.open(f"/kaggle/input/acm-ai-mini-ml-challenge-advanced-track/test/{img_path}").convert('RGB')
        
        if self.augment:
            image = self.augmentation(image)
        
        image = ToTensor()(image)
        
        
        return image



dataset = ImageData("/kaggle/input/acm-ai-mini-ml-challenge-advanced-track/test.csv")


labels = []
for i in tqdm.tqdm(range(len(dataset))):
    img = dataset[i].unsqueeze(0)
    y = model(img).squeeze(0)
    labels.append(torch.argmax(y).item())


import numpy as np
np.unique(labels, return_counts = True)


label_map = {'AnnualCrop': 0, 'Forest': 1, 'HerbaceousVegetation': 2, 'Highway': 3, 'Industrial': 4, 'Pasture': 5, 'PermanentCrop': 6, 'Residential': 7, 'River': 8, 'SeaLake': 9}


inv_label = {v:k for k,v in label_map.items()}


labels_r = [inv_label[l] for l in labels]


labels_r


df = pd.read_csv("/kaggle/input/acm-ai-mini-ml-challenge-advanced-track/test.csv")


df["target"] = labels_r
del df["path"]


df.to_csv("/kaggle/working/submission.csv")


df




