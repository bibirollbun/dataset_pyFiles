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


# Advanced Modeling
# Line 1
class MyNN(nn.Module):

    # Line 3
    def __init__(self, input_dim, output_dim, num_hidden_layers,
                 neurons_per_layer, dropout_rate):

        # Line 6
        super().__init__()

        # Line 8
        layers = []

        # Line 10
        current_input_dim = input_dim

        # Line 12
        for i in range(num_hidden_layers):

            # Line 15
            if i == num_hidden_layers - 1:
                hidden_neurons = max(output_dim*2, neurons_per_layer)
            else:
                hidden_neurons = max(
                    output_dim,
                    neurons_per_layer // (i + 1)
                )

            # Line 22
            layers.append(
                nn.Linear(current_input_dim, hidden_neurons)
            )

            # Line 25
            layers.append(
                nn.BatchNorm1d(hidden_neurons)
            )

            # Line 28
            layers.append(nn.ReLU())

            # Line 30
            layers.append(
                nn.Dropout(dropout_rate)
            )

            # Line 33
            current_input_dim = hidden_neurons

        # Line 36
        layers.append(
            nn.Linear(current_input_dim, output_dim)
        )

        # Line 40
        self.model = nn.Sequential(*layers)

    # Line 43
    def forward(self, x):

        # Line 46
        return self.model(x)


# objective function
def objective(trial):

    # next hyperparameter value from the search space
    num_hidden_layers= trial.suggest_int("num_hidden_layers",1,8)
    neurons_per_layers= trial.suggest_int("neurons_per_layer",8,520,step=32)
    epochs= trial.suggest_int("epoch",10,70,step=10)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    dropout_rate= trial.suggest_float("dropout_rate",0.1,0.5,step=0.1)
    Batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    optimizer_name = trial.suggest_categorical("optimizer", ['Adam', 'SGD', 'RMSprop'])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)


    # create train and test loader
    train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True,pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Batch_size, shuffle=False,pin_memory=True)

    # Model Initialisation
    input_dim= 65536
    output_dim=2
    # Creating Model
    model = MyNN(input_dim, output_dim, num_hidden_layers, neurons_per_layers, dropout_rate)
    model.to(device)
    # Optimizer Selection
    criterion= nn.CrossEntropyLoss()
    
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

      
    # Training loop
    # training loop
    
    for epoch in range(epochs):
        
        for batch_features, batch_labels in train_loader:
        
          # move data to gpu
          batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        
          # forward pass
          outputs = model(batch_features)
        
          # calculate loss
          loss = criterion(outputs, batch_labels)
        
          # back pass
          optimizer.zero_grad()
          loss.backward()
        
          # update grads
          optimizer.step()
    
    
    # evaluation
    model.eval()
    # evaluation on test data
    total = 0
    correct = 0
    
    with torch.no_grad():
        
        for batch_features, batch_labels in test_loader:
        
          # move data to gpu
          batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        
          outputs = model(batch_features)
        
          _, predicted = torch.max(outputs, 1)
        
          total = total + batch_labels.shape[0]
        
          correct = correct + (predicted == batch_labels).sum().item()
        
        accuracy = correct/total
    return accuracy


!pip install optuna


import optuna

study = optuna.create_study(direction='maximize')


study.optimize(objective, n_trials=20)


study.best_value


study.best_params


best = study.best_params


model = MyNN(
    input_dim=65536,
    output_dim=2,
    num_hidden_layers=best["num_hidden_layers"],
    neurons_per_layer=best["neurons_per_layer"],
    dropout_rate=best["dropout_rate"]
).to(device)



criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=best["learning_rate"],
    weight_decay=best["weight_decay"]
)


full_train_dataset = CustomDataset(X.values, y.values)


train_loader = DataLoader(
    full_train_dataset,   # ← train + val combined
    batch_size=best["batch_size"],
    shuffle=True,
    pin_memory=True
)

model.train()


for epoch in range(best["epoch"]):
    for batch_features, batch_labels in train_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

        optimizer.zero_grad()
        loss = criterion(model(batch_features), batch_labels)
        loss.backward()
        optimizer.step()


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
test_loader = DataLoader(test_dataset, batch_size=best["batch_size"], shuffle=False,pin_memory=True)


model.eval()

all_ids = []
all_preds = []

with torch.no_grad():
    for batch_features, batch_ids in test_loader:
        batch_features = batch_features.to(device)

        outputs = model(batch_features)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_ids.extend(batch_ids)



submission = pd.DataFrame({
    "ID": all_ids,
    "Class": all_preds
})

submission.to_csv("submission.csv", index=False)


