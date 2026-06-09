import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader


train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

train_df.head()


train_df.describe()


plt.figure(figsize=(10, 8)) 
sns.heatmap(train_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.9)
plt.title("Correlation between Features")


plt.figure(figsize=(10, 8))
sns.histplot(train_df["rainfall"], bins=2)
plt.title("Rainfall Distribution", fontsize=14, fontweight="bold")


for col in train_df.columns:
    plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=train_df, x="rainfall", y=col)
    plt.subplot(1, 2, 2)
    sns.histplot(data=train_df, x=col, hue="rainfall", kde=True)


class PredictModel(nn.Module): 
    def __init__(self):
        super(PredictModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(12, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class RainfallDataset:
    def __init__(self, df, split="train"):
        if split == "train":
            self.features = df.iloc[:1900, :-1].values
            self.labels = df.iloc[:1900, -1].values
        elif split == "val":
            self.features = df.iloc[1900:, :-1].values
            self.labels = df.iloc[1900:, -1].values
        else:
            self.features = df.iloc[:, :-1].values
            self.labels = df.iloc[:, -1].values
    
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        data = torch.tensor(self.features[idx], dtype=torch.float32)  
        label = torch.tensor(self.labels[idx], dtype=torch.float32) 
        return data, label


train_loader = DataLoader(RainfallDataset(train_df, split="train"), batch_size=32, shuffle=True)
val_loader = DataLoader(RainfallDataset(train_df, split="val"), batch_size=32, shuffle=False)

criterion = nn.BCELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PredictModel()
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)



num_epochs = 500
best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    for data, labels in train_loader:
        data = data.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * data.size(0)

    epoch_train_loss = running_train_loss / len(train_loader.dataset)

    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for data, labels in val_loader:
            data = data.to(device)
            labels = labels.to(device).unsqueeze(1)

            output = model(data)
            loss = criterion(output, labels)

            running_val_loss += loss.item() * data.size(0)

    epoch_val_loss = running_val_loss / len(val_loader.dataset)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), "best_model.pth")
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break



import torch
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = PredictModel()
model.load_state_dict(torch.load('best_model.pth'))
model.to(device)
model.eval()

predictions = []

for _, row in test_df.iterrows():
    data = torch.tensor(row.values, dtype=torch.float32).unsqueeze(0)
    data = data.to(device)
    
    with torch.no_grad():
        output = model(data)
        
        pred = output.squeeze().cpu().numpy()
        
        predictions.append(pred)

submission_df = pd.DataFrame({'id': test_df['id'], 'prediction': predictions})
print(predictions[0])

submission_df.to_csv("submission.csv", index = False)
submission_df.head()



import os

file_path = "/kaggle/working/submission.csv"
if os.path.exists(file_path):
    os.remove(file_path)
    print(f"{file_path} has been deleted")
else:
    print(f"The file {file_path} does not exist")


