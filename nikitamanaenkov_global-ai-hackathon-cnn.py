import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


PATCH_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 200
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PatchDataset(Dataset):
    def __init__(self, image_patches, labels=None, transform=None):
        self.image_patches = image_patches
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_patches)

    def __getitem__(self, idx):
        img = self.image_patches[idx]
        if self.transform:
            img = self.transform(img)
        if self.labels is not None:
            return img, self.labels[idx]
        return img


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
])

class CellTypeDataset(Dataset):
    def __init__(self, coordinates, labels, patch_size=64, transform=None):
        self.coordinates = coordinates
        self.labels = labels
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        x, y = self.coordinates[idx]
        patch = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
        patch[x % self.patch_size, y % self.patch_size] = 1.0

        patch = (patch - patch.mean()) / patch.std()
        patch = torch.tensor(patch, dtype=torch.float32).unsqueeze(0) 

        if self.transform:
            patch = self.transform(patch)

        label = torch.tensor(np.argmax(self.labels[idx]), dtype=torch.long)
        return patch, label



class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4)
        )
        
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 35)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = torch.relu(self.fc2(x))
        return self.fc3(x)



with h5py.File("/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5", "r") as f:
    train_spots = f["spots/Train"]
    train_spot_tables = {slide: pd.DataFrame(np.array(train_spots[slide])) for slide in train_spots.keys()}

train_df = pd.concat(train_spot_tables.values(), ignore_index=True)

X = train_df[['x', 'y']].values
y = train_df.iloc[:, 2:].values

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)

train_dataset = CellTypeDataset(X_train, y_train, patch_size=PATCH_SIZE, transform=transform)
valid_dataset = CellTypeDataset(X_valid, y_valid, patch_size=PATCH_SIZE) 

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

model = CNNModel().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

best_val_loss = float('inf')
patience = 5
wait = 0


for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            valid_loss += loss.item()

    train_loss /= len(train_loader)
    valid_loss /= len(valid_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

    if valid_loss < best_val_loss:
        best_val_loss = valid_loss
        wait = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered.")
            break



model.load_state_dict(torch.load("best_model.pt"))
model.eval()

with h5py.File("/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5", "r") as f:
    test_spots = f["spots/Test"]
    test_spot_table = pd.DataFrame(np.array(test_spots['S_7']))

X_test = test_spot_table[['x', 'y']].values
X_test_resized = np.zeros((X_test.shape[0], 1, PATCH_SIZE, PATCH_SIZE))

for i, (x, y) in enumerate(X_test):
    patch = np.zeros((PATCH_SIZE, PATCH_SIZE))
    patch[x % PATCH_SIZE, y % PATCH_SIZE] = 1
    patch = (patch - 0.5) / 0.5  # Normalize
    X_test_resized[i] = patch.reshape(1, PATCH_SIZE, PATCH_SIZE)

X_test_tensor = torch.tensor(X_test_resized, dtype=torch.float32).to(DEVICE)

with torch.no_grad():
    test_preds = model(X_test_tensor)
    test_preds_softmax = torch.softmax(test_preds, dim=1).cpu().numpy()


submission_df = pd.DataFrame(test_preds, columns=[f"C{i+1}" for i in range(35)])
submission_df.insert(0, 'ID', test_spot_table.index)
submission_df.to_csv("/kaggle/working/submission.csv", index=False)

print("Submission file 'submission.csv' created!")
submission_df.head()

