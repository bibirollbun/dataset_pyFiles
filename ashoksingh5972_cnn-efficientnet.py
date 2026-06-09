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


import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import time
import cv2
warnings.filterwarnings('ignore')


start_time = time.time()
BASE_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/"


brain_activities = ['Seizure', 'GPD', 'LRDA', 'Other', 'GRDA', 'LPD']
activity_mapping = {activity: idx for idx, activity in enumerate(brain_activities)}


df = pd.read_csv(f"{BASE_DIR}train.csv")
# Split 80% Train, 20% Temp (Validation + Test)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

# Split 10% Validation, 10% Test from Temp
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save to CSV
train_df.to_csv("train.csv", index=False)
val_df.to_csv("validation.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("Splitting done! Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))


class ChunkedBrainActivityDataset(Dataset):
    def __init__(self, csv_file, base_dir, activity_mapping):
        self.df = csv_file
        self.base_dir = base_dir
        self.activity_mapping = activity_mapping
        self.resize_transform = transforms.Resize((224, 224))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        spect_id, label, offset = self.df.iloc[idx][["spectrogram_id", "expert_consensus", "spectrogram_label_offset_seconds"]]

        temp_df = pd.read_parquet(f'{self.base_dir}/train_spectrograms/{spect_id}.parquet')
        temp_df.drop(['time'], axis=1, inplace=True)

        start = int(offset) // 2
        temp_df = temp_df[start:start+300]
        temp_df = np.log1p(temp_df)
        temp_df /= temp_df.max()
        temp_arr = np.nan_to_num(temp_df.to_numpy(), nan=1e-4)

        # Use OpenCV to apply a colormap and convert to RGB
        temp_arr_uint8 = np.uint8(255 * temp_arr)
        rgb_image = cv2.applyColorMap(temp_arr_uint8, cv2.COLORMAP_JET)

        # Normalize to [0, 1] and convert to tensor
        rgb_image = rgb_image.astype(np.float32) / 255.0
        rgb_image_tensor = torch.tensor(rgb_image).permute(2, 0, 1)  # (C, H, W)
        rgb_image_tensor = self.resize_transform(rgb_image_tensor)

        y = self.activity_mapping[label]
        y_tensor = torch.nn.functional.one_hot(torch.tensor(y, dtype=torch.long), num_classes=6).float()

        return rgb_image_tensor, y_tensor


# Now create DataLoader with the chunked dataset
# chunk_size = 1000  # Adjust chunk size according to memory constraints

train_dataset = ChunkedBrainActivityDataset(csv_file=train_df, base_dir=BASE_DIR, activity_mapping=activity_mapping)
val_dataset = ChunkedBrainActivityDataset(csv_file=val_df, base_dir=BASE_DIR, activity_mapping=activity_mapping)
test_dataset = ChunkedBrainActivityDataset(csv_file=test_df, base_dir=BASE_DIR, activity_mapping=activity_mapping)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=12, pin_memory=True, prefetch_factor=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=12, pin_memory=True, prefetch_factor=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=12, pin_memory=True, prefetch_factor=2)


model = models.efficientnet_v2_s(pretrained=True)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 6)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)


num_epochs = 101
patience = 15
best_val_loss = float('inf')
patience_counter = 0


train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    # tr_cntr=0
    for data in tqdm(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        _, labels_ = torch.max(labels.data, 1)
        correct += (predicted == labels_).sum().item()
                    
        # batch_accuracy = 100 * correct / total
        # print(f"Train Batch: {tr_cntr}/585, Loss: {loss.item():.4f}, Accuracy: {batch_accuracy:.2f}%")
        # tr_cntr+=1
        
        torch.cuda.empty_cache()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    # val_cntr=0
    with torch.no_grad():
        for data in tqdm(val_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # print("Data for some epoch loaded!")

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            _, labels_ = torch.max(labels.data, 1)
            correct += (predicted == labels_).sum().item()
            # batch_accuracy = 100 * correct / total
            # print(f"Val Batch: {val_cntr}/126, Loss: {loss.item():.4f}, Accuracy: {batch_accuracy:.2f}%")
            # val_cntr+=1
            torch.cuda.empty_cache()

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    elapsed_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, , Time: {elapsed_time}")
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'HMS_model_v1_efficientnet_v2_s.pth')
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered")
        break

    torch.cuda.empty_cache()

# Testing
if(os.path.isfile('HMS_model_v1_efficientnet_v2_s.pth')):
    model.load_state_dict(torch.load('HMS_model_v1_efficientnet_v2_s.pth'))
model.eval()

test_loss = 0.0
correct = 0
total = 0
# test_cntr=0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        _, labels_ = torch.max(labels.data, 1)
        correct += (predicted == labels_).sum().item()
        # batch_accuracy = 100 * correct / total
        # print(f"Test Batch: {test_cntr}/126, Loss: {loss.item():.4f}, Accuracy: {batch_accuracy:.2f}%")
        # test_cntr+=1
        
        torch.cuda.empty_cache()

test_loss /= len(test_loader)
test_accuracy = 100 * correct / total

print(f"Test Accuracy: {test_accuracy:.2f}%")


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(train_accuracies)
plt.plot(val_accuracies)
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(train_losses)
plt.plot(val_losses)
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()


!ls




