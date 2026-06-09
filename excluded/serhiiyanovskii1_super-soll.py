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


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torchvision.transforms as transforms
import random
import timm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- Model Definition ---
class DenseNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.timm_model = timm.create_model('densenet201', pretrained=True, in_chans=1)
        self.fc = nn.Linear(1000, 30)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.timm_model(x)
        x = self.relu(x)
        x = self.fc(x)
        return x

# --- Dataset Class ---
class MyDataset(data.Dataset):
    def __init__(self, mode, images, targets=None):
        self.mode = mode
        self.images = images
        self.targets = targets

    def transform(self, image, target=None):
        image = transforms.ToTensor()(image)
        if self.mode == 'train':
            if random.random() > 0.5:
                image = transforms.GaussianBlur(5)(image)
            if random.random() > 0.5:
                angle = random.randint(-20, 20)
                image = transforms.functional.rotate(image, angle)
        return (image, target) if self.mode != 'test' else image

    def __getitem__(self, index):
        if self.mode != 'test':
            return self.transform(self.images[index], self.targets[index])
        return self.transform(self.images[index])

    def __len__(self):
        return self.images.shape[0]

# --- Helper Functions ---
def reshape_images(images):
    images_reshaped = np.zeros((images.shape[0], 96, 96, 1))
    for i, img in enumerate(images):
        img = np.array([int(num) for num in img.split(' ')])
        images_reshaped[i] = img.reshape((96, 96, 1))
    return images_reshaped

# --- Data Loading ---
train_data = pd.read_csv('/kaggle/input/facial-keypoints-detection/training.zip', compression='zip')
train_data = train_data.sample(frac=1).reset_index(drop=True)
features_name = list(train_data.columns)
features_name.remove('Image')

# Reshape images and split data
images = train_data['Image']
targets = train_data.drop(columns=['Image']).fillna(-1).to_numpy()
images = reshape_images(images)
X_train, X_val, y_train, y_val = train_test_split(images, targets, test_size=0.1, random_state=42)

# Create datasets and data loaders
train_dataset = MyDataset('train', X_train, y_train)
val_dataset = MyDataset('val', X_val, y_val)
train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
val_loader = data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

# --- Model, Loss, Optimizer, and Scheduler ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DenseNetModel().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.25)
loss_func = nn.MSELoss()
num_epochs = 210
# --- Training Loop ---
train_losses, val_losses = [], []
for epoch in range(1, num_epochs + 1):
    print(f'Epoch {epoch}/{num_epochs}')
    
    # Training
    model.train()
    train_loss = 0
    for samples, targets in train_loader:
        samples, targets = samples.to(device).float(), targets.to(device).float()
        preds = model(samples)
        preds[targets == -1] = -1  # Ignore missing keypoints
        loss = loss_func(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * samples.size(0)

    train_loss = np.sqrt(train_loss / len(train_loader.dataset))
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for samples, targets in val_loader:
            samples, targets = samples.to(device).float(), targets.to(device).float()
            preds = model(samples)
            preds[targets == -1] = -1
            loss = loss_func(preds, targets)
            val_loss += loss.item() * samples.size(0)

    val_loss = np.sqrt(val_loss / len(val_loader.dataset))
    val_losses.append(val_loss)
    print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# Plot training and validation losses
plt.plot(range(num_epochs), train_losses, label='Train Loss')
plt.plot(range(num_epochs), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# --- Test Data Prediction ---
test_data = pd.read_csv('/kaggle/input/facial-keypoints-detection/test.zip', compression='zip')
test_images = reshape_images(test_data['Image'])
test_dataset = MyDataset('test', test_images)
test_loader = data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

model.eval()
predictions = []
with torch.no_grad():
    for samples in test_loader:
        samples = samples.to(device).float()
        preds = model(samples)
        predictions.extend(preds.cpu().numpy().flatten().tolist())

# --- Submission ---
df_preds = pd.DataFrame({'ImageId': np.repeat(np.arange(1, len(test_images) + 1), 30),
                         'FeatureName': features_name * len(test_images),
                         'Location': predictions})
lookup = pd.read_csv('/kaggle/input/facial-keypoints-detection/IdLookupTable.csv')
lookup = lookup.drop('Location',axis = 1)
print(lookup)
print(df_preds)
submission = df_preds.merge(lookup, on=['ImageId', 'FeatureName'], how='inner')[['RowId', 'Location']]
submission.to_csv('submission.csv', index=False)
print('Submission saved.')


