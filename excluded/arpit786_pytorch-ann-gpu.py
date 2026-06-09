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


import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Set random seeds for reproducibility
torch.manual_seed(6379)


df = pd.read_csv('/kaggle/input/image-classifications/train.csv')
df.head()


df.shape


# Create a 4x4 grid of images
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
fig.suptitle("First 16 Images", fontsize=16)

# Plot the first 16 images from the dataset
for i, ax in enumerate(axes.flat):
    img = df.iloc[i, 1:65537].values.reshape(256, 256)  # Reshape to 256x256
    ax.imshow(img)  # Display in grayscale
    ax.axis('off')  # Remove axis for a cleaner look
    ax.set_title(f"Label: {df.iloc[i, 0]}")  # Show the label

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title
plt.show()


# train test split
# Separate columns
ids = df.iloc[:, 0]              # Sample ID
X = df.iloc[:, 1:65537]          # Features
y = df.iloc[:, 65537]            # Label


# Split INCLUDING ids
X_train, X_test, y_train, y_test, id_train, id_val = train_test_split(
    X, y, ids,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# scaling the feautures
X_train = X_train/255.0
X_test = X_test/255.0


X_train.shape


X_test


X_test.shape


# create CustomDataset Class
class CustomDataset(Dataset):

  def __init__(self, features, labels):

    self.features = torch.tensor(features, dtype=torch.float32)
    self.labels = torch.tensor(labels, dtype=torch.long)

  def __len__(self):

    return len(self.features)

  def __getitem__(self, index):

    return self.features[index], self.labels[index]


# create train_dataset object
train_dataset = CustomDataset(X_train.values, y_train.values)


train_dataset[0]


# create test_dataset object
test_dataset = CustomDataset(X_test.values, y_test.values)


test_dataset


print(type(X_train), X_train.shape)
print(type(y_train), y_train.shape)


# create train and test loader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


train_loader


# Define NN Class
class MyNN(nn.Module):
    def __init__(self,num_features):
        super().__init__()
        self.model= nn.Sequential(
            nn.Linear(num_features,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,2)
        
    )

    def forward(self,x):
        return self.model(x)



# setting learning rate and epochs
epochs=70
learning_rate=0.1


# Instantiate the model
model=MyNN(X_train.shape[1])
model = model.to(device)
# Loss Function
criterion= nn.CrossEntropyLoss()

# Optimizer
optimizer= optim.SGD(model.parameters(),lr=learning_rate)


len(train_loader)


# Training the loop
for epoch in range(epochs):

    total_epoch_loss=0
    for batch_features,batch_labels in train_loader:
        # Move data to GPU
        batch_features,batch_labels= batch_features.to(device),batch_labels.to(device)
        # Forward Pass
        outputs= model(batch_features)

        # Calculate_Loss
        loss= criterion(outputs,batch_labels)

        # Back pass
        optimizer.zero_grad()
        loss.backward()

        # Update_grads
        optimizer.step()

        total_epoch_loss= total_epoch_loss+ loss.item()

    avg_loss= total_epoch_loss/len(train_loader)
    print(f'epoch:{epoch+1},Loss:{avg_loss}')
    


model.eval()


# Evaluation code
total=0
correct=0

with torch.no_grad():
    for batch_features,batch_labels in train_loader:
        # Move data to GPU
         # move data to gpu
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
    
        outputs = model(batch_features)
    
        _, predicted = torch.max(outputs, 1)
    
        total = total + batch_labels.shape[0]
    
        correct = correct + (predicted == batch_labels).sum().item()
    
    print(correct/total)


# Evaluation Code
total=0
correct=0

with torch.no_grad():
    for batch_features,batch_labels in test_loader:
        # Move data to GPU
        batch_features,batch_labels= batch_features.to(device),batch_labels.to(device)
        outputs=model(batch_features)

        _,predicted= torch.max(outputs,1)

        total= total + batch_labels.shape[0]
        correct= correct+ (predicted==batch_labels).sum().item()

print(correct/total)


predicted.shape


test_df = pd.read_csv("/kaggle/input/image-classifications/test.csv")

test_ids = test_df.iloc[:, 0]
X_test = test_df.iloc[:, 1:65537]


class TestDataset(Dataset):
    def __init__(self, features, ids):
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.ids = ids.values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.ids[index]



test_dataset = TestDataset(X_test, test_ids)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


all_preds = []
all_ids = []

with torch.no_grad():
    for batch_features, batch_ids in test_loader:
        batch_features = batch_features.to(device)

        outputs = model(batch_features)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_ids.extend(batch_ids)


submission = pd.DataFrame({
    "ID": all_ids,
    "Class": all_preds
})

submission.to_csv("submission.csv", index=False)


