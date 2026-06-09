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


%pip install py7zr


import py7zr
import os

def unzip_file(zip_filepath, extract_to='.'):
    if not os.path.isfile(zip_filepath):
        print(f"Error: {zip_filepath} does not exist.")
        return
    
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    try:
        with py7zr.SevenZipFile(zip_filepath, 'r') as zip_ref:
            # extract .7z files to extract_to dir
            zip_ref.extractall(extract_to)
            print(f"All files have been extracted to {extract_to}.")
    except Exception as e:
        print(f"An error occurred while extracting the file {zip_filepath}: {e}")


extract_to = '/kaggle/working'
unzip_file('/kaggle/input/cifar-10/test.7z', extract_to)
unzip_file('/kaggle/input/cifar-10/train.7z', extract_to)


import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


"""
2. Data warapping, loading.
"""


LABEL_CLASS_LIST = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, label_file=None, transform=None, target_transform=None, train=True):
        """
        img_dir (string): Directory with all the images.  
        label_file (string): Path to the csv file with labels.      
        transform (callable, optional): Optional transform to be applied on a sample.  
        target_transform (callable, optional): Optional transform to be applied on a target.
        train (bool): default train dataset, if False, test dataset.
        """
        super().__init__()
        self.img_dir = img_dir
        self.label_file = label_file
        self.train = train
        if self.train:
            self.img_labels = pd.read_csv(self.label_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels) if self.train else len(os.listdir(self.img_dir))

    def __getitem__(self, index):
        target = LABEL_CLASS_LIST.index(self.img_labels.iloc[index, 1]) if self.train else None
        img_path = os.path.join(self.img_dir, str(index + 1) + '.png')
        img = Image.open(img_path)   # ImageFile.ImageFile object (PIL Image)
        if self.transform:
            img = self.transform(img)
        if target is not None and self.target_transform:
            target = self.target_transform(target)
        return (img, target) if self.train else img
    

train_img_dir = "/kaggle/working/train"
test_img_dir = "/kaggle/working/test"
label_file = "/kaggle/input/cifar-10/trainLabels.csv"


train_transform = transforms.Compose([
    transforms.Resize(40),
    transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])    # learn from trained ImageNet Dataset
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


train_dataset = CustomImageDataset(train_img_dir, label_file=label_file, transform=train_transform)
test_dataset = CustomImageDataset(test_img_dir, transform=test_transform, train=False)

train_batch_size = 1024
test_batch_size = 10240
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size)


len(train_dataset), len(test_dataset)


img, target = train_dataset[0]


img, img.shape, target


target = test_dataset[0]


target, target.shape


len(train_dataloader), len(test_dataloader)


import torch
torch.cuda.is_available()


"""
model building
"""
import torch
from torch import nn


class MyModel(nn.Module):
    def __init__(self, in_channels, out_features):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),  # (N, 16 * 4 * 4)
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, out_features)    # Last layer not need to activate, instead keep original logits beacuse loss function use CrossEntropyLoss.
        )

    def forward(self, x):
        return self.layer(x)


# Check if CUDA is available and set the device accordingly.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


in_channels = train_dataset[0][0].shape[0]    
out_features = len(LABEL_CLASS_LIST)
# Move model to GPU
model = MyModel(in_channels, out_features).to(device)

lr = 0.002
weight_decay = 0.001
# Ensure that loss function and optimizer are defined after moving the model to the device
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()


"""
4. training
"""


def train(epochs):
    batch_num = len(train_dataloader)   # (len(train_dataset) + train_batch_size - 1) // train_batch_size
    model.train()
    for epoch in range(epochs):
        total_mean_loss = 0
        total_correct_class_count = 0
        for input_datas, labels in train_dataloader:
            # Move inputs and labels to the device
            input_datas, labels = input_datas.to(device), labels.to(device)
            optimizer.zero_grad()
            output_datas = model(input_datas)            
            loss = criterion(output_datas, labels)
            loss.backward()
            optimizer.step()
            total_mean_loss += loss.item()
            total_correct_class_count += torch.sum(torch.argmax(output_datas, dim=1) == labels).item()
        accurancy = round((total_correct_class_count / len(train_dataset)) * 100, 2)
        print(f"epoch={epoch}, accurancy is: {accurancy} %")
        if epoch % 2 == 0:            
            print(f"epoch={epoch}, mean loss is: {total_mean_loss / batch_num}")

epochs = 33
train(epochs)


sample_data = pd.read_csv("/kaggle/input/cifar-10/sampleSubmission.csv")


sample_data.head()


from tqdm import tqdm


pre_labels = []
with torch.no_grad():
    model.eval()
    for input_datas in tqdm(test_dataloader):
        input_datas = input_datas.to(device)
        output_datas = model(input_datas)
        pre_labels.extend(torch.argmax(output_datas, dim=1).cpu().numpy())


len(pre_labels)


pre_labels[:10]


pre_labels = [LABEL_CLASS_LIST[label] for label in pre_labels]


pre_labels[:10]


sample_data['label'] = np.array(pre_labels)


sample_data.head()


sample_data.to_csv('/kaggle/working/submission.csv', index=False)
print("Submission file created successfully!")


import shutil

shutil.rmtree('/kaggle/working/train')
shutil.rmtree('/kaggle/working/test')

