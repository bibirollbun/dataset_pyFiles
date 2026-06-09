# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import multiprocessing

multiprocessing.set_start_method("spawn", force=True)  # Fixes CUDA issue

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for elems in os.listdir('/kaggle/input'):
    print(elems)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


DATA_DIR = "/kaggle/input/ai-vs-human-generated-dataset/"


import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
# Load train.csv
#df = pd.read_csv('train.csv')

DATA_DIR = "/kaggle/input/ai-vs-human-generated-dataset/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define image dataset class
class ImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.data = df  # Load CSV file
        self.img_dir = img_dir  # Image folder path
        self.transform = transform  # Image transformations
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        #print(f"Getting item {idx}")
        img_path = os.path.join(self.img_dir, self.data.iloc[idx]['file_name'])
        #print(f"Image path \"{img_path}\"")
        image = Image.open(img_path).convert("RGB")  # Load image on demand
        #print("Loaded image")
        label = torch.tensor(self.data.iloc[idx]['label']).to(device).float()  # Assuming label column exists
        #print("Loaded label")
        #label = label.float().to(device)
        
        if self.transform:
            image = self.transform(image).to(device).float()
            #print(f"Type of image: {image.dtype}")
            #print(f"Type of labels: {label.dtype}")
            #print("Transformed")
        else:
            #print("No Transform!!!")
            pass
        return image, label




import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 37 * 37, 128)  # Adjust the input size based on the output of the conv layers
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = x.view(-1, 128 * 37 * 37)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
# if __name__ == "__main__":
#     multiprocessing.freeze_support()
#     multiprocessing.set_start_method("spawn", force=True)
# Load train.csv
#df = pd.read_csv('train.csv')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DATA_DIR = "./data"

EPOCHS = 2
BATCH_SIZE = 64
LEARNING_RATE = 0.001

transform = transforms.Compose([
    #transforms.Resize((300, 300), interpolation=transforms.InterpolationMode.BICUBIC),  # Resize to fit model input
    #transforms.RandomHorizontalFlip(),  # 
    transforms.CenterCrop(300),
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet's mean and standard deviation
                         std=[0.229, 0.224, 0.225])  # Normalize
])

csv_file = os.path.join(DATA_DIR, "train.csv")
df = pd.read_csv(csv_file)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
# Create dataset and DataLoader
train_dataset = ImageDataset(train_df, img_dir=DATA_DIR, transform=transform)
dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

model = ConvNet()
model = model.to(device)

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.8, 0.9))
#scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.5)

loss_fn = torch.nn.BCEWithLogitsLoss()

# Example: Iterate through the dataset

import matplotlib.pyplot as plt

def plot_activations(x):
    x = x.detach().cpu().numpy().flatten()
    plt.hist(x, bins=50)
    plt.show()


for epoch in range(EPOCHS):
    model.train()
    for i, (images, labels) in enumerate(dataloader):
        optim.zero_grad()
        output = model(images)
        loss = loss_fn(output.flatten(), labels)
        loss.backward()
        optim.step()
        #print(images.shape, labels)
        inputs = images
        inputs = inputs.to(device)
        #activations = model.conv1(inputs)  # Check first convolution layer
        #plot_activations(activations)
        #print(f"Mean: {activations.mean().item():.4f}, Std: {activations.std().item():.4f}")

        sys.stdout.write(f"\rLoss: {loss.item()} Batch: {i+1}/{len(dataloader)} Epoch: {epoch+1}/{EPOCHS}")
    
    # Validation phase
    model.eval()
    val_dataset = ImageDataset(val_df, img_dir=DATA_DIR, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    correct = 0
    total = 0

    with torch.no_grad():
        all_predictions = []
        all_labels = []
        for i, (images, labels) in enumerate(val_dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted.flatten() == labels).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            sys.stdout.write(f"\rBatch: {i+1}/{len(val_dataloader)}")

            
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)

        print(f'Validation Precision: {precision:.2f}')
        print(f'Validation Recall: {recall:.2f}')
                
        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy:.2f}%')



import os
test_dir = os.path.join(DATA_DIR, "test_data_v2")
test_images = [os.path.join(test_dir, img) for img in os.listdir(test_dir) if img.endswith('.jpg')]

test_transform = transforms.Compose([
    transforms.Resize((300, 300), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class TestDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return img_path, image

test_dataset = TestDataset(test_images, transform=test_transform)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model.eval()
submission = []

with torch.no_grad():
    for img_paths, images in test_dataloader:
        images = images.to(device)
        outputs = model(images)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        for img_path, pred in zip(img_paths, predicted):
            submission.append({"id": img_path, "label": int(pred.item())})
            sys.stdout.write(f"\rProcessed: {len(submission)}/{len(test_dataset)}")

submission_df = pd.DataFrame(submission)
submission_df.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")

