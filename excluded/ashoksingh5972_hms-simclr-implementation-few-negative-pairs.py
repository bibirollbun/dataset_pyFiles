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


!pip install efficientnet_pytorch


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

df_toy = df.sample(frac=0.2, random_state=42)
# Split 80% Train, 20% Temp (Validation + Test)
train_df, temp_df = train_test_split(df_toy, test_size=0.4, random_state=42)

# Split 10% Validation, 10% Test from Temp
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save to CSV
train_df.to_csv("train.csv", index=False)
val_df.to_csv("validation.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("Splitting done! Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))


class ChunkedBrainActivityDataset(Dataset):
    def __init__(self, csv_file, base_dir, activity_mapping,md):
        self.df = csv_file
        self.base_dir = base_dir
        self.activity_mapping = activity_mapping
        self.resize_transform = transforms.Resize((224, 224))
        self.md = md

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

train_dataset = ChunkedBrainActivityDataset(csv_file=train_df, base_dir=BASE_DIR, activity_mapping=activity_mapping,md = "lr")
val_dataset = ChunkedBrainActivityDataset(csv_file=val_df, base_dir=BASE_DIR, activity_mapping=activity_mapping,md = "lr")
test_dataset = ChunkedBrainActivityDataset(csv_file=test_df, base_dir=BASE_DIR, activity_mapping=activity_mapping,md = "lr")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers= 2, pin_memory=True, prefetch_factor=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers= 2, pin_memory=True, prefetch_factor=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers= 2, pin_memory=True, prefetch_factor=2)


import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

class SimCLR(nn.Module):
    def __init__(self, base_model, projection_dim=128):
        super(SimCLR, self).__init__()
        self.encoder = base_model
        self.encoder.fc = nn.Identity()  # Remove the final fully connected layer
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, projection_dim)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        return h, z

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1, neg_pair_fraction=0.2):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.neg_pair_fraction = neg_pair_fraction
    
    def forward(self, out1, out2):
        # Normalize the outputs
        out1 = torch.nn.functional.normalize(out1, dim=1)
        out2 = torch.nn.functional.normalize(out2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.exp(torch.mm(out1, out2.T) / self.temperature)
        
        # Positive pairs are on the diagonal
        pos_pairs = torch.diag(sim_matrix)
        
        # Select only a fraction of negative pairs
        num_neg = int(self.neg_pair_fraction * (sim_matrix.size(1) - 1))
        neg_pairs = torch.topk(sim_matrix, num_neg, dim=1, largest=False)[0].sum(dim=1)
        
        # Compute loss
        loss = -torch.log(pos_pairs / neg_pairs).mean()
        
        return loss

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model and move it to the appropriate device
base_model = models.resnet50(pretrained=False)
model = SimCLR(base_model).to(device)

# Define the loss function and optimizer
criterion = NTXentLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


class RandomAugmentation(nn.Module):
    def __init__(self):
        super(RandomAugmentation, self).__init__()
        self.augmentations = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, x):
        if len(x.shape) == 4:
            return torch.stack([self.augmentations(to_pil_image(img)) for img in x])
        elif len(x.shape) == 3:
            return self.augmentations(to_pil_image(x))
        else:
            raise ValueError(f"Invalid input shape {x.shape}")


num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, _ in train_loader:
        images = images.cpu()
        aug1, aug2 = RandomAugmentation()(images), RandomAugmentation()(images)
        aug1, aug2 = aug1.to(device), aug2.to(device)
        _, z1 = model(aug1)
        _, z2 = model(aug2)
        loss = criterion(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader.dataset):.4f}")


# Fine-tuning with logistic regression
class FineTuneModel(nn.Module):
    def __init__(self, encoder, num_classes=6):
        super(FineTuneModel, self).__init__()
        self.encoder = encoder
        self.logistic_regression = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        h = self.encoder(x)
        logits = self.logistic_regression(h)
        return logits

fine_tune_model = FineTuneModel(model.encoder).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(fine_tune_model.logistic_regression.parameters(), lr=0.001)

num_epochs = 20
for epoch in range(num_epochs):
    fine_tune_model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)
        labels = torch.argmax(targets, dim=1)
        
        optimizer.zero_grad()
        logits = fine_tune_model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(logits, dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")


fine_tune_model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for images, targets in test_loader:
        images = images.to(device)
        targets = targets.to(device)
        labels = torch.argmax(targets, dim=1)
        logits = fine_tune_model(images)
        _, preds = torch.max(logits, dim=1)
        test_total += labels.size(0)
        test_correct += (preds == labels).sum().item()

test_accuracy = 100 * test_correct / test_total
print(f"Test Accuracy: {test_accuracy:.2f}%")

