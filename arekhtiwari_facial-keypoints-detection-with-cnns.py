# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import zipfile
import torch.nn as nn
import torch.nn.functional as F

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Unzip training data
with zipfile.ZipFile('/kaggle/input/facial-keypoints-detection/training.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working')

# Unzip test data
with zipfile.ZipFile('/kaggle/input/facial-keypoints-detection/test.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working')


# Load training and test CSVs
train_df = pd.read_csv('/kaggle/working/training.csv')
test_df = pd.read_csv('/kaggle/working/test.csv')

# Also load the lookup table and submission sample
lookup_df = pd.read_csv('/kaggle/input/facial-keypoints-detection/IdLookupTable.csv')
submission_df = pd.read_csv('/kaggle/input/facial-keypoints-detection/SampleSubmission.csv')


# Parse image data
train_df['Image'] = train_df['Image'].apply(lambda x: np.fromstring(x, sep=' ') if isinstance(x, str) else np.nan)
train_df = train_df.dropna(subset=['Image'])  # Drop bad rows

X_train = np.vstack(train_df['Image'].values).astype(np.float32) / 255.0
X_train = X_train.reshape(-1, 96, 96, 1)

y_train = train_df.drop('Image', axis=1).values  # Labels, shape: (num_samples, 30)


# Pick an index to visualize
index = 200

# Get image (reshape to 96x96)
image = X_train[index].reshape(96, 96)

# Get keypoints (x, y pairs)
keypoints = y_train[index]

# Plot
plt.imshow(image, cmap='gray')
plt.scatter(keypoints[0::2], keypoints[1::2], c='r', marker='x')  # every other is x, then y
plt.title("Training Image with Keypoints")
plt.axis('off')
plt.show()


# Load training data
df = pd.read_csv('/kaggle/working/training.csv')
df = df.dropna()  # remove rows with missing keypoints

# Convert image string to numpy array
df['Image'] = df['Image'].apply(lambda x: np.fromstring(x, sep=' ').reshape(96, 96).astype(np.uint8))

# Split features and targets
X = np.stack(df['Image'].values)
y = df.drop('Image', axis=1).values.astype(np.float32)

# Normalize images and keypoints
X = X / 255.0
y = y / 96.0  # normalize coordinates to [0, 1]


class FaceKeypointsDataset(Dataset):
    def __init__(self, images, keypoints, transform=None):
        self.images = images.astype(np.float32)  # ensures images are float32
        self.keypoints = keypoints.astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        keypoint = self.keypoints[idx]

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # add channel dim

        return image, torch.tensor(keypoint, dtype=torch.float32)

# Transform: convert to tensor and add channel dim
transform = transforms.Compose([
    transforms.ToTensor(),  # shape (1, 96, 96)
])

dataset = FaceKeypointsDataset(X, y, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


class KeypointModel(nn.Module):
    def __init__(self):
        super(KeypointModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 22 * 22, 128)
        self.fc2 = nn.Linear(128, 30)  # 15 keypoints × 2

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> [batch, 32, 47, 47]
        x = self.pool(F.relu(self.conv2(x)))  # -> [batch, 64, 22, 22]
        x = x.view(-1, 64 * 22 * 22)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = KeypointModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 30
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, keypoints in dataloader:
        # print(images.dtype)  # should be torch.float32
        images = images.to(device)
        keypoints = keypoints.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, keypoints)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader):.4f}")


# Load test.csv
test_df = pd.read_csv('/kaggle/working/test.csv')
test_df['Image'] = test_df['Image'].apply(lambda x: np.fromstring(x, sep=' ').reshape(96, 96).astype(np.float32))
X_test = np.stack(test_df['Image'].values) / 255.0  # normalize

print(X_test.shape)  # Should be (num_test_images, 96, 96)


class TestDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # [1, 96, 96]
        return image

test_dataset = TestDataset(X_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


model.eval()
predictions = []

with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = model(images)
        outputs = outputs.cpu().numpy()
        predictions.append(outputs)

predictions = np.vstack(predictions)  # Shape: [1783, 30]
predictions = predictions * 96  # de-normalize back to image coordinates


id_lookup = pd.read_csv('/kaggle/input/facial-keypoints-detection/IdLookupTable.csv')

# Get feature names in correct order
feature_names = pd.read_csv('/kaggle/working/training.csv').columns[:-1]

# Prepare final submission
submission = []

for idx, row in id_lookup.iterrows():
    image_id = row['ImageId'] - 1  # 0-indexed
    feature_name = row['FeatureName']
    
    feature_index = list(feature_names).index(feature_name)
    predicted_value = predictions[image_id][feature_index]
    
    submission.append(predicted_value)

submission_df = pd.DataFrame({
    'RowId': id_lookup['RowId'],
    'Location': submission
})

submission_df.to_csv('submission.csv', index=False)

