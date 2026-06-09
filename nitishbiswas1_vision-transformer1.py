# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session





import json
import os
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# Function to compute bounding box from polygon points
def getc_bbox(points, image_width, image_height):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x = max(0, min(xs))
    max_x = min(image_width, max(xs))
    min_y = max(0, min(ys))
    max_y = min(image_height, max(ys))
    if min_x >= max_x or min_y >= max_y:
        return None  # Invalid ROI
    return [int(min_x), int(min_y), int(max_x), int(max_y)]
# Paths (update these)
train_dir = "/kaggle/input/glioma-mcd-2025/Data_122824/Glioma_MDC_2025_training/"

# Load JSON files
json_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.json')]

# 1. Class Distribution
labels = []
for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)
    for shape in data['shapes']:
        label = 1 if shape['label'] == 'Mitosis' else 0
        labels.append(label)
total_rois = len(labels)
mitosis_count = sum(labels)
non_mitosis_count = total_rois - mitosis_count
print(f"Total ROIs: {total_rois}")
print(f"Mitosis: {mitosis_count} ({mitosis_count/total_rois*100:.2f}%)")
print(f"Non-mitosis: {non_mitosis_count} ({non_mitosis_count/total_rois*100:.2f}%)")

# 2. ROI Size Distribution
areas = []
for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)
    image_width, image_height = Image.open(os.path.join(train_dir, json_file.replace('.json', '.jpg'))).size
    
    for shape in data['shapes']:
        bbox = getc_bbox(shape['points'],image_width, image_height)
        if not bbox:
            continue
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        areas.append(area)
areas = np.array(areas)
print(f"ROI Area Stats - Mean: {areas.mean():.2f}, Median: {np.median(areas):.2f}, Min: {areas.min():.2f}, Max: {areas.max():.2f}, Std: {areas.std():.2f}")

# 3. ROIs per Image
rois_per_image = [len(json.load(open(jf))['shapes']) for jf in json_files]
rois_per_image = np.array(rois_per_image)
print(f"ROIs per Image - Mean: {rois_per_image.mean():.2f}, Median: {np.median(rois_per_image):.2f}, Min: {rois_per_image.min()}, Max: {rois_per_image.max()}, Std: {rois_per_image.std():.2f}")

# 4. Spatial Distribution
centroids = []
for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)
    for shape in data['shapes']:
        points = np.array(shape['points'])
        centroid = points.mean(axis=0)  # [x, y]
        centroids.append(centroid)
centroids = np.array(centroids)
grid, _, _ = np.histogram2d(centroids[:, 0], centroids[:, 1], bins=8, range=[[0, 512], [0, 512]])
print("Spatial Distribution (8x8 grid counts):")
print(grid.astype(int))

# 5. Image Intensity Stats (sample 50 images)
image_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.jpg')][:50]
means, stds = [], []
for img_file in image_files:
    img = np.array(Image.open(img_file).convert('RGB')) / 255.0
    means.append(img.mean(axis=(0, 1)))
    stds.append(img.std(axis=(0, 1)))
means = np.mean(means, axis=0)
stds = np.mean(stds, axis=0)
print(f"Image Intensity - R: mean={means[0]:.3f}, std={stds[0]:.3f}; G: mean={means[1]:.3f}, std={stds[1]:.3f}; B: mean={means[2]:.3f}, std={stds[2]:.3f}")

# 6. Mitosis vs. Non-mitosis Feature Comparison
mitosis_intensities, non_mitosis_intensities = [], []
for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)
    image_width, image_height = Image.open(os.path.join(train_dir, json_file.replace('.json', '.jpg'))).size
    img = np.array(Image.open(os.path.join(train_dir, json_file.replace('.json', '.jpg'))).convert('RGB')) / 255.0
    for shape in data['shapes']:
        bbox = getc_bbox(shape['points'],image_width, image_height)
        if not bbox:
            continue
        roi = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        intensity = roi.mean()
        if shape['label'] == 'Mitosis':
            mitosis_intensities.append(intensity)
        else:
            non_mitosis_intensities.append(intensity)
print(f"Mitosis ROI Intensity - Mean: {np.mean(mitosis_intensities):.3f}, Std: {np.std(mitosis_intensities):.3f}")
print(f"Non-mitosis ROI Intensity - Mean: {np.mean(non_mitosis_intensities):.3f}, Std: {np.std(non_mitosis_intensities):.3f}")


import json
import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_dir = "/kaggle/input/glioma-mcd-2025/Data_122824/Glioma_MDC_2025_training/"



def get_bbox(points, img_width, img_height):
    """Compute bounding box from polygon points, ensuring it stays within image bounds."""
    xs = [p[0] for p in points]  # Extract x-coordinates
    ys = [p[1] for p in points]  # Extract y-coordinates
    min_x = max(0, int(math.floor(min(xs))))  # Clamp to 0 as minimum
    max_x = min(img_width, int(math.ceil(max(xs))))  # Clamp to image width as maximum
    min_y = max(0, int(math.floor(min(ys))))  # Clamp to 0 as minimum
    max_y = min(img_height, int(math.ceil(max(ys))))  # Clamp to image height as maximum
    
    # Check if the bounding box is valid
    if min_x >= max_x or min_y >= max_y:
        return None  # Return None for invalid bounding boxes
    return [min_x, min_y, max_x, max_y]


def parse_training_json(json_path, train_dir):
    """Extract training samples from JSON annotations, handling varying image sizes."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    image_path = os.path.join(train_dir, json_file.replace('.json', '.jpg'))
    img = Image.open(image_path)  # Open image to get its size
    img_width, img_height = img.size  # Get actual dimensions
    samples = []
    for shape in data['shapes']:
        label = 1 if shape['label'] == 'Mitosis' else 0  # Binary label
        bbox = get_bbox(shape['points'], img_width, img_height)
        if bbox is not None:  # Only include valid bounding boxes
            samples.append((image_path, bbox, label))
    return samples


# Parse test JSON files with dynamic image sizes
def parse_test_json(json_path, test_dir):
    """Extract test samples from JSON annotations, handling varying image sizes."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    image_path = os.path.join(train_dir, json_file.replace('.json', '.jpg'))
    img = Image.open(image_path)  # Open image to get its size
    img_width, img_height = img.size  # Get actual dimensions
    samples = []
    for shape in data['shapes']:
        label_id = shape['label']  # e.g., "Blank1"
        bbox = get_bbox(shape['points'], img_width, img_height)
        if bbox is not None:  # Only include valid bounding boxes
            samples.append((image_path, bbox, label_id))
    return samples



# Custom Dataset for training/validation with empty ROI handling
class MitosisDataset(Dataset):
    """Dataset class for training and validation data."""
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        self.image_cache = {}  # Cache images to avoid reloading

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, bbox, label = self.samples[idx]
        # Load or retrieve image from cache
        if image_path not in self.image_cache:
            image = Image.open(image_path).convert('RGB')
            self.image_cache[image_path] = image
        else:
            image = self.image_cache[image_path]
        
        # Crop and handle empty ROIs
        cropped = image.crop(bbox)
        if cropped.size == (0, 0):  # Skip empty crops
            return None
        resized = cropped.resize((224, 224))  # Resize for model input
        if self.transform:
            resized = self.transform(resized)
        return resized, label


# Custom Dataset for test data with empty ROI handling
class TestDataset(Dataset):
    """Dataset class for test data."""
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        self.image_cache = {}  # Cache images to avoid reloading

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, bbox, label_id = self.samples[idx]
        # Load or retrieve image from cache
        if image_path not in self.image_cache:
            image = Image.open(image_path).convert('RGB')
            self.image_cache[image_path] = image
        else:
            image = self.image_cache[image_path]
        
        # Crop and handle empty ROIs
        cropped = image.crop(bbox)
        if cropped.size == (0, 0):  # Skip empty crops
            return None
        resized = cropped.resize((224, 224))  # Resize for model input
        if self.transform:
            resized = self.transform(resized)
        image_id = os.path.basename(image_path).split('.')[0]
        return resized, image_id, label_id




# Define transformations for training and validation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.620, 0.405, 0.633], std=[0.162, 0.150, 0.123]),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.620, 0.405, 0.633], std=[0.162, 0.150, 0.123]),
])


train_dir = "/kaggle/input/glioma-mcd-2025/Data_122824/Glioma_MDC_2025_training/"
test_dir = "/kaggle/input/glioma-mcd-2025/Data_122824/Glioma_MDC_2025_test"





training_json_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.json')]
all_training_samples = []
for json_file in training_json_files:
    all_training_samples.extend(parse_training_json(json_file, train_dir))


test_json_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.json')]
all_test_samples = []
for json_file in test_json_files:
    all_test_samples.extend(parse_test_json(json_file, test_dir))


train_samples, val_samples = train_test_split(all_training_samples, test_size=0.2, random_state=42)


train_dataset = MitosisDataset(train_samples, transform=train_transform)
val_dataset = MitosisDataset(val_samples, transform=val_transform)
test_dataset = TestDataset(all_test_samples, transform=val_transform)


# Custom collate function to filter out None values (empty ROIs)
def collate_fn(batch):
    batch = [item for item in batch if item is not None]  # Remove None items
    if len(batch) == 0:
        return None  # Return None if batch is empty
    return torch.utils.data.dataloader.default_collate(batch)


# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)


# Load Vision Transformer model and modify the head
model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
in_features = model.heads.head.in_features
model.heads.head = nn.Linear(in_features, 1)  # Binary classification
model.to(device)


# Define loss, optimizer, and scheduler
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)


# Training function with empty batch handling
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    best_f1 = 0
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_train = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", leave=False)
        for batch in train_bar:
            if batch is None:  # Skip empty batches
                continue
            images, labels = batch
            images, labels = images.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Update running loss and accuracy
            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            running_corrects += torch.sum(preds == labels).item()
            total_train += labels.size(0)
            running_loss += loss.item()

        # Update live training metrics
            current_loss = running_loss / total_train
            current_acc = running_corrects / total_train
            train_bar.set_postfix({'Loss': f'{current_loss:.4f}', 'Acc': f'{current_acc:.4f}'})
        train_loss = running_loss / total_train
        train_acc = running_corrects / total_train
        # Validation phase
        model.eval()
        val_loss = 0.0
        running_val_loss = 0.0
        running_val_corrects = 0
        total_val = 0
        preds_list, true_labels = [], []
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]", leave=False)
        with torch.no_grad():
            for batch in val_bar:
                if batch is None:  # Skip empty batches
                    continue
                images, labels = batch
                images, labels = images.to(device), labels.float().to(device)
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                running_val_corrects += torch.sum(preds == labels).item()
                total_val += labels.size(0)
                
                preds_list.extend(preds.cpu().numpy().tolist())
                true_labels.extend(labels.cpu().numpy().tolist())
                
                current_val_loss = running_val_loss / total_val
                current_val_acc = running_val_corrects / total_val
                val_bar.set_postfix({'Loss': f'{current_val_loss:.4f}', 'Acc': f'{current_val_acc:.4f}'})
        
        val_loss = running_val_loss / total_val
        val_acc = running_val_corrects / total_val
        f1 = f1_score(true_labels, preds_list)
        
        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, F1: {f1:.4f}')
        
        # Adjust learning rate based on F1 score and save best model
        scheduler.step(f1)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), '/kaggle/working/best_model.pth')



# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50)


model.load_state_dict(torch.load('/kaggle/working/best_model.pth'))
model.eval()

# Predict on test set with empty ROI handling
predictions_dict = {}
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting", leave=False):
        if batch is None:  # Skip empty batches
            continue
        images, image_ids, label_ids = batch
        images = images.to(device)
        outputs = model(images).squeeze()
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        for i in range(len(image_ids)):
            predictions_dict[(image_ids[i], label_ids[i])] = preds[i]

# Prepare submission file
print(predictions_dict)
submission = pd.read_csv('/kaggle/input/glioma-mcd-2025/archive/Submission_template.csv')
submission['Prediction'] = submission.apply(
    lambda row: predictions_dict.get((row['Image ID'], row['Label ID']), 0), axis=1
)
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Submission file 'submission.csv' created successfully!")




