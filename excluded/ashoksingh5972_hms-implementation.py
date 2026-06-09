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


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader

# # Assuming your ChunkedBrainActivityDataset class and the DataLoader code
# # for train_loader, val_loader, and test_loader are already defined.

# # Define a logistic regression model for multi-class classification.
# class LogisticRegressionModel(nn.Module):
#     def __init__(self, input_dim, num_classes):
#         super(LogisticRegressionModel, self).__init__()
#         self.linear = nn.Linear(input_dim, num_classes)
        
#     def forward(self, x):
#         # Flatten the input tensor: (batch, 3, 224, 224) => (batch, 3*224*224)
#         x = x.view(x.size(0), -1)
#         # Return the raw logits (CrossEntropyLoss applies softmax internally)
#         return self.linear(x)

# # Set device to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Calculate the size of the flattened image.
# # Your images are of shape: (3, 224, 224)
# input_dim = 3 * 224 * 224
# num_classes = 6

# # Instantiate the model, loss function, and optimizer.
# model = LogisticRegressionModel(input_dim, num_classes).to(device)

# # CrossEntropyLoss expects integer labels, not one-hot vectors.
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)

# # Number of training epochs
# num_epochs = 10

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#     for images, targets in train_loader:
#         # Move the batch to the device
#         images = images.to(device)
#         targets = targets.to(device)
#         # Convert one-hot targets to integer labels.
#         labels = torch.argmax(targets, dim=1)

#         optimizer.zero_grad()
#         logits = model(images)
#         loss = criterion(logits, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item() * images.size(0)
#         # Calculate accuracy for the batch
#         _, preds = torch.max(logits, 1)
#         total += labels.size(0)
#         correct += (preds == labels).sum().item()
        
#     epoch_loss = running_loss / total
#     epoch_acc = 100 * correct / total
#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")


# model.eval() # Set the model to evaluation mode
# correct_test = 0
# total_test = 0

# with torch.no_grad(): # Disables gradient calculation
#     for images, targets in test_loader:
#         images = images.to(device)
#         targets = targets.to(device)
#         # Convert one-hot target vectors to scalar class labels
#         labels = torch.argmax(targets, dim=1)
#         # Forward pass to get predictions
#         logits = model(images)
#         _, preds = torch.max(logits, dim=1)
#         total_test += labels.size(0)
#         correct_test += (preds == labels).sum().item()
#     test_accuracy = 100 * correct_test / total_test
#     print(f"Test Accuracy: {test_accuracy:.2f}%")


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models  # ✅ Correct import

# Define the model that uses EfficientNet as an encoder with logistic regression as the classifier
class EfficientNetV2EncoderLogisticRegression(nn.Module):
    def __init__(self, num_classes=6):
        super(EfficientNetV2EncoderLogisticRegression, self).__init__()
        # ✅ Load pretrained EfficientNetV2-S correctly
        self.encoder = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        
        # Get the feature size from the last layer
        n_features = self.encoder.classifier[1].in_features
        
        # Remove the classifier head
        self.encoder.classifier = nn.Identity()
        
        # Add a logistic regression layer for classification
        self.logistic_regression = nn.Linear(n_features, num_classes)

    def forward(self, x):
        features = self.encoder(x)  # Extract features
        logits = self.logistic_regression(features)  # Apply classifier
        return logits

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model and move it to the appropriate device
num_classes = 6
model = EfficientNetV2EncoderLogisticRegression(num_classes=num_classes).to(device)

# Load pretrained model weights
pretrained_path = "/kaggle/input/cnn_efficientnet_v1/pytorch/default/1/HMS_model_v1_efficientnet_v2_s.pth"
pretrained_dict = torch.load(pretrained_path, map_location=device)

# Get model's current state dict
model_dict = model.state_dict()

# ✅ Load only matching layers (ignore classifier mismatch)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

# ✅ Do NOT redefine classifier again (already done inside the model class)
# Move model to device
model.to(device)

print("Model loaded and classifier updated successfully!")

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)
        labels = torch.argmax(targets, dim=1)  # Convert one-hot encoded to class index
        
        optimizer.zero_grad()
        logits = model(images)
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

# Evaluate on the test dataset
model.eval()
correct_test = 0
total_test = 0
with torch.no_grad():
    for images, targets in test_loader:
        images = images.to(device)
        targets = targets.to(device)
        labels = torch.argmax(targets, dim=1)
        logits = model(images)
        _, preds = torch.max(logits, dim=1)
        total_test += labels.size(0)
        correct_test += (preds == labels).sum().item()

test_accuracy = 100 * correct_test / total_test
print(f"Test Accuracy: {test_accuracy:.2f}%")



import torch
import torch.nn.functional as F

def compute_rsm(model, data_loader, device):
    """
    Compute Representational Similarity Matrix (RSM) using cosine similarity.
    
    Args:
        model: The trained model.
        data_loader: DataLoader for the dataset.
        device: Torch device (CPU/GPU).
    
    Returns:
        rsm: Tensor representing the RSM (N x N similarity matrix).
    """
    model.eval()
    features_list = []

    # Extract features
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            features = model.encoder(images)  # Extract features from encoder
            features_list.append(features.cpu())  # Move to CPU to save memory

    # Stack features into a single tensor
    features = torch.cat(features_list, dim=0)  # Shape: (N, feature_dim)

    # Compute cosine similarity
    rsm = F.cosine_similarity(
        features.unsqueeze(1),  # Shape: (N, 1, feature_dim)
        features.unsqueeze(0),  # Shape: (1, N, feature_dim)
        dim=2
    )

    return rsm


# Compute RSM for the test dataset
rsm = compute_rsm(model, test_loader, device)

# Convert to NumPy and visualize
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 8))
plt.imshow(rsm.cpu().numpy(), cmap="coolwarm", interpolation="nearest")
plt.colorbar(label="Cosine Similarity")
plt.title("Representational Similarity Matrix (RSM)")
plt.xlabel("Samples")
plt.ylabel("Samples")
plt.show()





