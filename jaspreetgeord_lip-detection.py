import torch
import torch.nn as nn
import torchvision
import cv2
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


with zipfile.ZipFile("/kaggle/input/facial-keypoints-detection/training.zip", "r") as zip_ref:
    zip_ref.extractall()
with zipfile.ZipFile("/kaggle/input/facial-keypoints-detection/test.zip", "r") as zip_ref:
    zip_ref.extractall()


train_data_df = pd.read_csv('training.csv')
test_data_df = pd.read_csv('test.csv')
train_data_df = train_data_df[[col for col in train_data_df.columns if col.startswith("mouth") or col.startswith("Image")]]
test_data_df = test_data_df[[col for col in test_data_df.columns if col.startswith("mouth") or col.startswith("Image")]]

train_data_df.head(1)


from sklearn.impute import KNNImputer

print(train_data_df.isna().sum())  

imputer = KNNImputer(n_neighbors=50)  
train_data_df.iloc[:, :-1] = imputer.fit_transform(train_data_df.iloc[:, :-1])

print(train_data_df.isna().sum())  


def convert_images(image_data):
    image = np.array(image_data, dtype=np.uint8)
    image = image.reshape(96, 96) 
    return image

def process_images(data, labels_df, start_idx, end_idx):
    converted_images = []
    labels = []
    for i in range(start_idx, end_idx):
        image_data = data[i].split(" ")
        converted_images.append(convert_images(image_data))
        labels.append(labels_df.iloc[i, :8]) 
    return converted_images, labels

train_images_data = train_data_df.iloc[:6500, -1]
val_images_data = train_data_df.iloc[6500:, -1]

train_converted_images, train_label = process_images(train_images_data, train_data_df, 0, len(train_images_data))
val_converted_images, val_label = process_images(val_images_data, train_data_df, 6500, len(train_data_df))


x_coords = train_label[0][::2]  
y_coords = train_label[0][1::2] 

plt.imshow(train_converted_images[0], cmap="gray")

plt.scatter(x_coords, y_coords, c="red", marker="o")

plt.title("Image with Facial Keypoints")


class LipDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 96x96 -> 48x48
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 48x48 -> 24x24
       
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 24x24 -> 12x12
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6))
        )

        self.fc = nn.Sequential(
            nn.Linear(512*6*6, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 8)
        )

      
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) 
        
        x = self.fc(x)
        return x


class FacialKeypointsDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.images, self.labels = self._preprocess_data()
    
    def _preprocess_data(self):
        images, labels = [], []
        for _, row in self.df.iterrows():
            img = np.array(row['Image'].split(), dtype=np.float32).reshape(96, 96) / 255.0
            keypoints = row.values[:8].astype(np.float32) / 96.0
            images.append(img)
            labels.append(keypoints)
        return images, labels

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image = self.images[idx].copy()
        label = self.labels[idx].copy()
        
        image = image.reshape(1, 96, 96)
        image = torch.from_numpy(image).float()
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.from_numpy(label).float()


dataset = FacialKeypointsDataset(train_data_df)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LipDetectionModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)


num_epochs = 100
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device) 
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item() * images.size(0)

    epoch_train_loss = running_train_loss / len(train_loader.dataset)

    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item() * images.size(0)

    epoch_val_loss = running_val_loss / len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f} - Val Loss: {epoch_val_loss:.4f}")

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), "lip_detection_model.pth")
        print("Best model saved!")





import os
os.listdir("/kaggle/input/lip-detection-model/pytorch/default/1/")


import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("/kaggle/input/lip-detection-model/pytorch/default/1/final_best_model.pth", map_location=device)
model = LipDetectionModel().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

test_data_df = pd.read_csv("/kaggle/working/test.csv")
sample = test_data_df.sample(n=1).iloc[0]
img_data = np.array(sample["Image"].split(), dtype=np.float32).reshape(96, 96) / 255.0
img_tensor = torch.tensor(img_data).unsqueeze(0).unsqueeze(0).float().to(device)

with torch.no_grad():
    output = model(img_tensor)
    coords = output.detach().cpu().numpy().flatten()

pred_x_coords_denorm = np.clip(coords[0::2] * 96.0, 0, 96)
pred_y_coords_denorm = np.clip(coords[1::2] * 96.0, 0, 96)
x_start = max(min(pred_x_coords_denorm) - 15, 0)
y_start = max(min(pred_y_coords_denorm) - 15, 0)
length = min(abs(max(pred_x_coords_denorm) - min(pred_x_coords_denorm)) + 30, 96 - x_start)
width = min(abs(max(pred_y_coords_denorm) - min(pred_y_coords_denorm)) + 30, 96 - y_start)

print(f"Vorhergesagte Koordinaten: {list(zip(pred_x_coords_denorm, pred_y_coords_denorm))}")
print(f"Rechteck - x_start: {x_start}, y_start: {y_start}, Länge: {length}, Breite: {width}")

fig, ax = plt.subplots()
ax.imshow(img_data, cmap="gray")
ax.scatter(pred_x_coords_denorm, pred_y_coords_denorm, c="red", marker="x")
rect = patches.Rectangle((x_start, y_start), length, width, linewidth=2, edgecolor="blue", facecolor="none")
ax.add_patch(rect)
plt.axis("off")
plt.show()


