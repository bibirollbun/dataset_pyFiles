# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import polars as pl

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

import kaggle_evaluation.cmi_inference_server


full_train_data = pl.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')#[:30057]


full_train_data


tof_columns = []
for column in full_train_data.columns:
    if 'tof' in column:
        tof_columns.append(column)
        
tof_columns[0:10]


last_sequence_id = full_train_data['sequence_id'][0]
count = 0
X = []
y = [full_train_data['gesture'][0]]
one_X = []
for i in range(full_train_data.shape[0]):
    sequence_id = full_train_data['sequence_id'][i]
    if sequence_id == last_sequence_id:
        count += 1
        if count > 29:
            continue
    else:
        last_sequence_id = sequence_id
        X.append(one_X)
        y.append(full_train_data['gesture'][i])
        one_X = []
        count = 1

    one_X.append(np.array(full_train_data[tof_columns][1]).reshape(5, 8, 8))
X.append(one_X)


X = np.array(X).transpose(0,2,1,3,4).astype(np.float32)
y = np.array(y)


# X = full_train_data.drop(['row_id', 'sequence_type', 'sequence_id', 'sequence_counter',
#                          'subject','orientation', 'behavior', 'phase', 'gesture',
#                          'acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z', 'thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5'])
# y = full_train_data['gesture']


train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2)


label_encoder = LabelEncoder()
train_y = label_encoder.fit_transform(train_y)
val_y = label_encoder.transform(val_y)


import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

train_set = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
val_set = TensorDataset(torch.from_numpy(val_X), torch.from_numpy(val_y))

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)


import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3DClassifier(nn.Module):
    def __init__(self, num_classes=18):
        super(Conv3DClassifier, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels=5, out_channels=16, kernel_size=3, padding=1),  # (B, 16, 30, 8, 8)
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),  # (B, 16, 15, 4, 4)

            nn.Conv3d(16, 32, kernel_size=3, padding=1),  # (B, 32, 15, 4, 4)
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3, 2, 2)),  # (B, 32, 5, 2, 2)

            nn.Conv3d(32, 64, kernel_size=3, padding=1),  # (B, 64, 5, 2, 2)
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))  # (B, 64, 1, 1, 1)
        )

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)  # Flatten to (B, 64)
        x = self.classifier(x)
        return x


import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Conv3DClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)  # shape: (B, 5, 30, 8, 8)
        labels = labels.to(device)  # shape: (B,)

        optimizer.zero_grad()
        outputs = model(inputs)  # shape: (B, 18)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / len(dataloader.dataset)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / len(dataloader.dataset)
    return avg_loss, accuracy



num_epochs = 100

for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
    print(f"  Val   Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")


torch.save(model.state_dict(), "conv3d_classifier.pth")


full_test_data = pl.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')


count = 0
X = []
one_X = []
for i in range(full_test_data.shape[0]):
    count += 1
    if count > 30:
        break
    one_X.append(np.array(full_test_data[tof_columns][1]).reshape(5, 8, 8))
X.append(one_X)


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    count = 0
    test_X = []
    one_X = []
    for i in range(sequence.shape[0]):
        count += 1
        if count > 30:
            break
        one_X.append(np.array(sequence[tof_columns][1]).reshape(5, 8, 8))
    test_X.append(one_X)
    test_X = np.array(test_X).transpose(0,2,1,3,4).astype(np.float32)
    
    re = label_encoder.inverse_transform((model(torch.from_numpy(test_X)).detach().numpy().argmax(),))[0]
    return str(re)


inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )




