import os
from PIL import Image
import pydicom

# import other libraries
# import system libs
import os
import time
import shutil
import pathlib
import itertools
from pathlib import Path
import multiprocessing as mp
from tqdm.notebook import tqdm
from joblib import Parallel, delayed

# import data handling tools
import cv2
import pydicom
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt

# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import categorical_crossentropy

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

print ('modules loaded')


import os

# Ø§Ù„Ù…Ø³Ø§Ø± Ù„Ù…Ø¬Ù„Ø¯ Ø§Ù„ØµÙˆØ±
image_folder = '/kaggle/input/rsna-bcd-1024x512-preprocessed/'

# Ø§Ø¹Ø±Ø¶ Ø£ÙˆÙ„ 5 Ù�Ø§ÙŠÙ„Ø§Øª Ø¹Ø´Ø§Ù† Ù†ØªØ£ÙƒØ¯
print(os.listdir(image_folder)[:5])


# Ø§Ø³ØªÙŠØ±Ø§Ø¯ Ø§Ù„Ù…ÙƒØªØ¨Ø§Øª Ø§Ù„Ù…Ø·Ù„ÙˆØ¨Ø©
import pandas as pd
import os

# ØªØ­Ø¯ÙŠØ¯ Ø§Ù„Ù…Ø³Ø§Ø±Ø§Øª
dataset_path = '/kaggle/input/rsna-bcd-1024x512-preprocessed'  # Ù‡Ù†Ø§ ØªØ­Ø·ÙŠ Ø§Ù„Ù…Ø³Ø§Ø± Ø§Ù„Ø£Ø³Ø§Ø³ÙŠ Ù„Ù„Ø¯Ø§ØªØ§ Ø¹Ù†Ø¯Ùƒ
csv_path = os.path.join(dataset_path, '/kaggle/input/rsna-breast-cancer-detection/train.csv')
images_path = os.path.join(dataset_path, '/kaggle/input/rsna-bcd-1024x512-preprocessed/train_images')

# Ù‚Ø±Ø§Ø¡Ø© Ù…Ù„Ù� train.csv
df = pd.read_csv(csv_path)

# Ø¥Ù†Ø´Ø§Ø¡ Ø¹Ù…ÙˆØ¯ Ø¬Ø¯ÙŠØ¯ Ù„Ù…Ø³Ø§Ø± ÙƒÙ„ ØµÙˆØ±Ø©
df['image_path'] = df['image_id'].apply(lambda x: os.path.join(images_path, f"{x}.png"))

# Ø¹Ø±Ø¶ Ø£ÙˆÙ„ 5 ØµÙ�ÙˆÙ� Ù„Ù„ØªØ£ÙƒØ¯ Ø£Ù† ÙƒÙ„ Ø­Ø§Ø¬Ø© ØµØ­
print(df.head())

# ÙƒÙ…Ø§Ù† Ù†Ø·Ø¨Ø¹ Ø¹Ø¯Ø¯ Ø§Ù„ØµÙˆØ± ÙˆØ§Ù„Ù„Ø§Ø¨ÙŠÙ„Ø² Ù„Ùˆ ØªØ­Ø¨ÙŠ
print(f"Number of images: {len(df)}")
print(f"Labels distribution:\n{df['cancer'].value_counts()}")


# Ù…Ø³Ø§Ø± Ø§Ù„Ø¯Ø§ØªØ§ Ø§Ù„Ø¬Ø¯ÙŠØ¯Ø© Ø§Ù„Ù„ÙŠ Ù�ÙŠÙ‡Ø§ Ø§Ù„ØµÙˆØ± Ø§Ù„Ù…Ø¹Ø§Ù„Ø¬Ø©
data_path = '/kaggle/input/rsna-bcd-1024x512-preprocessed'
csv_path = os.path.join(data_path, '/kaggle/input/rsna-breast-cancer-detection/train.csv')
images_folder = os.path.join(data_path, '/kaggle/input/rsna-bcd-1024x512-preprocessed/train_images')

# Ù‚Ø±Ø§Ø¡Ø© Ù…Ù„Ù� Ø§Ù„Ù„Ø§Ø¨Ù„Ø²
df = pd.read_csv(csv_path)

# Ø¥Ù†Ø´Ø§Ø¡ Ø¹Ù…ÙˆØ¯ Ù…Ø³Ø§Ø±Ø§Øª Ø§Ù„ØµÙˆØ±
df['image_path'] = df['patient_id'].astype(str) + '/' + df['image_id'].astype(str) + '.png'  # Ø§Ù…ØªØ¯Ø§Ø¯ Ø§Ù„ØµÙˆØ± png Ù…Ø´ jpg
df['image_path'] = df['image_path'].apply(lambda x: os.path.join(images_folder, x))

# Ù†Ø¸Ø±Ø© Ø³Ø±ÙŠØ¹Ø©
print(df[['patient_id', 'image_id', 'image_path']].head())


# Ø§Ø³ØªÙŠØ±Ø§Ø¯ Ø§Ù„Ù…ÙƒØªØ¨Ø§Øª Ø§Ù„Ù…Ø·Ù„ÙˆØ¨Ø©
import pandas as pd
import os

# ØªØ­Ø¯ÙŠØ¯ Ø§Ù„Ù…Ø³Ø§Ø±Ø§Øª
import os
import pandas as pd

# Ø§Ù„Ù…Ø³Ø§Ø± Ø§Ù„Ø£Ø³Ø§Ø³ÙŠ Ù„Ù„Ø¯Ø§ØªØ§
dataset_path = '/kaggle/input/rsna-bcd-1024x512-preprocessed'
csv_path = os.path.join(dataset_path, '/kaggle/input/rsna-breast-cancer-detection/train.csv')  # ØªÙ… ØªØ¹Ø¯ÙŠÙ„ Ø§Ù„Ù…Ø³Ø§Ø± Ù‡Ù†Ø§
images_path = os.path.join(dataset_path, '/kaggle/input/rsna-bcd-1024x512-preprocessed/train_images')  # ØªÙ… ØªØ¹Ø¯ÙŠÙ„ Ø§Ù„Ù…Ø³Ø§Ø± Ù‡Ù†Ø§

# Ù‚Ø±Ø§Ø¡Ø© Ù…Ù„Ù� train.csv
df = pd.read_csv(csv_path)

# Ø¥Ù†Ø´Ø§Ø¡ Ø¹Ù…ÙˆØ¯ Ø¬Ø¯ÙŠØ¯ Ù„Ù…Ø³Ø§Ø± ÙƒÙ„ ØµÙˆØ±Ø© Ø¨Ø§Ø³ØªØ®Ø¯Ø§Ù… "image_id" Ùˆ "patient_id"
df['image_path'] = df['patient_id'].astype(str) + '/' + df['image_id'].astype(str) + '.png'
df['image_path'] = df['image_path'].apply(lambda x: os.path.join(images_path, x))
# Ø¹Ø±Ø¶ Ø£ÙˆÙ„ 5 ØµÙ�ÙˆÙ� Ù„Ù„ØªØ£ÙƒØ¯ Ù…Ù† Ø§Ù„Ù…Ø³Ø§Ø±Ø§Øª
print(df.head())





from sklearn.model_selection import train_test_split

# Ø£ÙˆÙ„ Ø­Ø§Ø¬Ø©: Ù†Ù‚Ø³Ù… Ø¥Ù„Ù‰ train (80%) Ùˆ (validation + test) (20%)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['cancer'])

# Ø«Ø§Ù†ÙŠ Ø­Ø§Ø¬Ø©: Ù†Ù‚Ø³Ù… Ø§Ù„Ù€ temp_df Ø¨Ø§Ù„ØªØ³Ø§ÙˆÙŠ Ø¥Ù„Ù‰ validation Ùˆ test (ÙƒÙ„ ÙˆØ§Ø­Ø¯ 10%)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['cancer'])

# Ø·Ø¨Ø§Ø¹Ø© Ø£Ø­Ø¬Ø§Ù… ÙƒÙ„ Ù…Ø¬Ù…ÙˆØ¹Ø©
print(f"Train size: {len(train_df)}")
print(f"Validation size: {len(valid_df)}")
print(f"Test size: {len(test_df)}")


# Ø¥Ø¹Ø¯Ø§Ø¯ Ø§Ù„ØªØ­ÙˆÙŠÙ„Ø§Øª transformations
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
img_size = (256, 256)

train_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

valid_test_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
])

# ÙƒÙ„Ø§Ø³ Ø§Ù„Ø¯Ø§ØªØ§ Ø§Ù„Ø®Ø§Øµ Ø¨ÙŠÙ†Ø§
class BreastCancerDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'image_path']
        label = self.df.loc[idx, 'cancer']
        
        # Ù�ØªØ­ Ø§Ù„ØµÙˆØ±Ø©
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)

# Ø¥Ù†Ø´Ø§Ø¡ Ø§Ù„Ø¯Ø§ØªØ§Ø³ØªØ³
train_dataset = BreastCancerDataset(train_df, transform=train_transform)
valid_dataset = BreastCancerDataset(valid_df, transform=valid_test_transform)
test_dataset = BreastCancerDataset(test_df, transform=valid_test_transform)

# Ø¥Ù†Ø´Ø§Ø¡ Ø§Ù„Ø¯Ø§ØªØ§Ù„ÙˆØ¯Ø±Ø²
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

# Ø§Ø®ØªØ¨Ø§Ø± Ø§Ù„Ø´ÙƒÙ„
for images, labels in train_loader:
    print(f"Images batch shape: {images.shape}")
    print(f"Labels batch shape: {labels.shape}")
    break


import matplotlib.pyplot as plt
import numpy as np

# Function to show a batch of images
def show_sample(data_loader):
    images, labels = next(iter(data_loader))  # Get one batch
    images = images.numpy()

    plt.figure(figsize=(12, 6))
    for i in range(8):  # Ø¹Ø±Ø¶ Ø£ÙˆÙ„ 8 ØµÙˆØ±
        plt.subplot(2, 4, i+1)
        img = images[i].transpose((1, 2, 0))  # Ø±Ø¬Ø¹Ù†Ø§ Ø§Ù„Ø´ÙƒÙ„ (HWC)
        plt.imshow(img)
        plt.title(f"Label: {labels[i].item()}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Example: Show sample from training data
show_sample(train_loader)


import torch.nn as nn
import torchvision.models as models

# Load a pre-trained EfficientNetB5
model = models.efficientnet_b5(weights='IMAGENET1K_V1')

#Modify the classifier to match the number of classes (2 classes: 0 and 1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define Loss function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ØªØ­Ø¯ÙŠØ¯ Ø§Ù„Ø¬Ù‡Ø§Ø² (GPU Ø£Ùˆ CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ØªØ¹Ø±ÙŠÙ� Ø§Ù„Ù†Ù…ÙˆØ°Ø¬ (Ù‡Ù†Ø§ ÙŠØªÙ… Ø§Ø³ØªØ®Ø¯Ø§Ù… EfficientNet ÙƒÙ…Ø«Ø§Ù„)
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)  # ØªØµÙ†ÙŠÙ� Ø«Ù†Ø§Ø¦ÙŠ (Ù…Ø±ÙŠØ¶ / ØºÙŠØ± Ù…Ø±ÙŠØ¶)
model = model.to(device)

# ØªØ¹Ø±ÙŠÙ� Ø§Ù„Ù€ Loss Function
criterion = nn.BCEWithLogitsLoss()

# ØªØ¹Ø±ÙŠÙ� Ø§Ù„Ù€ Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Ø¹Ø¯Ø¯ Ø§Ù„Ø¥ÙŠØ¨ÙˆÙƒØ³ (epochs)
epochs = 5

# ØªØ¯Ø±ÙŠØ¨ Ø§Ù„Ù†Ù…ÙˆØ°Ø¬
for epoch in range(epochs):
    model.train()  # ÙˆØ¶Ø¹ Ø§Ù„Ù†Ù…ÙˆØ°Ø¬ Ù�ÙŠ ÙˆØ¶Ø¹ Ø§Ù„ØªØ¯Ø±ÙŠØ¨
    running_loss = 0.0
    
    # Ø§Ø³ØªØ®Ø¯Ø§Ù… tqdm Ù„Ø¹Ø±Ø¶ Ø´Ø±ÙŠØ· Ø§Ù„ØªÙ‚Ø¯Ù… Ø£Ø«Ù†Ø§Ø¡ Ø§Ù„ØªØ¯Ø±ÙŠØ¨
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    
    for images, labels in pbar:
        # Ù†Ù‚Ù„ Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª Ø¥Ù„Ù‰ Ø§Ù„Ø¬Ù‡Ø§Ø² (GPU Ø£Ùˆ CPU)
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)  # ØªØ£ÙƒØ¯ Ø£Ù† labels Ù�ÙŠ Ø´ÙƒÙ„ [batch_size, 1]
        
        optimizer.zero_grad()  # Ø¥Ø¹Ø§Ø¯Ø© ØªØ¹ÙŠÙŠÙ† Ø§Ù„ØªØ¯Ø±Ø¬Ø§Øª
        outputs = model(images)  # ØªÙ…Ø±ÙŠØ± Ø§Ù„ØµÙˆØ± Ø¹Ø¨Ø± Ø§Ù„Ù†Ù…ÙˆØ°Ø¬
        
        loss = criterion(outputs, labels)  # Ø­Ø³Ø§Ø¨ Ø§Ù„Ø®Ø³Ø§Ø±Ø©
        loss.backward()  # Ø­Ø³Ø§Ø¨ Ø§Ù„ØªØ¯Ø±Ø¬Ø§Øª
        optimizer.step()  # ØªØ­Ø¯ÙŠØ« Ø§Ù„Ø£ÙˆØ²Ø§Ù†
        
        running_loss += loss.item()  # Ø¬Ù…Ø¹ Ø§Ù„Ø®Ø³Ø§Ø±Ø©
        
        # ØªØ­Ø¯ÙŠØ« Ø´Ø±ÙŠØ· Ø§Ù„ØªÙ‚Ø¯Ù… Ø¨Ø¹Ø±Ø¶ Ø§Ù„Ø®Ø³Ø§Ø±Ø©
        pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
    
    # Ø·Ø¨Ø§Ø¹Ø© Ø§Ù„Ø®Ø³Ø§Ø±Ø© Ù�ÙŠ Ù†Ù‡Ø§ÙŠØ© ÙƒÙ„ epoch
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f}")

    # ØªÙ‚ÙŠÙŠÙ… Ø§Ù„Ø£Ø¯Ø§Ø¡ Ø¹Ù„Ù‰ Ø¨ÙŠØ§Ù†Ø§Øª Ø§Ù„ØªØ­Ù‚Ù‚ Ø¨Ø¹Ø¯ ÙƒÙ„ epoch
    model.eval()  # ÙˆØ¶Ø¹ Ø§Ù„Ù†Ù…ÙˆØ°Ø¬ Ù�ÙŠ ÙˆØ¶Ø¹ Ø§Ù„ØªØ­Ù‚Ù‚
    correct = 0
    total = 0
    with torch.no_grad():  # Ù„Ø§ Ø­Ø§Ø¬Ø© Ù„Ø­Ø³Ø§Ø¨ Ø§Ù„ØªØ¯Ø±Ø¬Ø§Øª Ø£Ø«Ù†Ø§Ø¡ Ø§Ù„ØªØ­Ù‚Ù‚
        for images, labels in valid_loader:
            # Ù†Ù‚Ù„ Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª Ø¥Ù„Ù‰ Ø§Ù„Ø¬Ù‡Ø§Ø²
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            
            # ØªØ·Ø¨ÙŠÙ‚ Ø¯Ø§Ù„Ø© sigmoide ÙˆØªØ­ÙˆÙŠÙ„ Ø§Ù„Ù…Ø®Ø±Ø¬Ø§Øª Ø¥Ù„Ù‰ ØªÙ†Ø¨Ø¤Ø§Øª Ø«Ù†Ø§Ø¦ÙŠØ©
            preds = torch.sigmoid(outputs) > 0.5
            
            correct += (preds == labels).sum().item()  # Ø­Ø³Ø§Ø¨ Ø¹Ø¯Ø¯ Ø§Ù„ØªÙ†Ø¨Ø¤Ø§Øª Ø§Ù„ØµØ­ÙŠØ­Ø©
            total += labels.size(0)  # Ø¥Ø¬Ù…Ø§Ù„ÙŠ Ø¹Ø¯Ø¯ Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª
            
    # Ø­Ø³Ø§Ø¨ Ø§Ù„Ø¯Ù‚Ø©
    acc = correct / total
    print(f"Validation Accuracy after Epoch {epoch+1}: {acc:.4f}")


# ÙˆØ¶Ø¹ Ø§Ù„Ù†Ù…ÙˆØ°Ø¬ Ù�ÙŠ ÙˆØ¶Ø¹ Ø§Ù„ØªÙ‚ÙŠÙŠÙ… (eval)
model.eval()  
correct = 0
total = 0

# Ø¹Ø¯Ù… Ø§Ù„Ø­Ø§Ø¬Ø© Ù„Ø­Ø³Ø§Ø¨ Ø§Ù„ØªØ¯Ø±Ø¬Ø§Øª Ø£Ø«Ù†Ø§Ø¡ Ø§Ù„ØªØ­Ù‚Ù‚
with torch.no_grad():
    for images, labels in test_loader:
        # Ù†Ù‚Ù„ Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª Ø¥Ù„Ù‰ Ø§Ù„Ø¬Ù‡Ø§Ø² (GPU Ø£Ùˆ CPU)
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        
        # ØªÙ…Ø±ÙŠØ± Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª Ø¹Ø¨Ø± Ø§Ù„Ù†Ù…ÙˆØ°Ø¬
        outputs = model(images)
        
        # ØªØ·Ø¨ÙŠÙ‚ Ø¯Ø§Ù„Ø© sigmoide Ù„ØªØ­ÙˆÙŠÙ„ Ø§Ù„Ù…Ø®Ø±Ø¬Ø§Øª Ø¥Ù„Ù‰ ØªÙ†Ø¨Ø¤Ø§Øª
        preds = torch.sigmoid(outputs) > 0.5  # Ø§Ù„ØªÙ†Ø¨Ø¤ Ø¥Ø°Ø§ ÙƒØ§Ù†Øª Ø§Ù„Ù†ØªÙŠØ¬Ø© 0 Ø£Ùˆ 1
        
        # Ø­Ø³Ø§Ø¨ Ø¹Ø¯Ø¯ Ø§Ù„ØªÙ†Ø¨Ø¤Ø§Øª Ø§Ù„ØµØ­ÙŠØ­Ø©
        correct += (preds == labels).sum().item()
        total += labels.size(0)  # Ø¥Ø¬Ù…Ø§Ù„ÙŠ Ø¹Ø¯Ø¯ Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª

# Ø­Ø³Ø§Ø¨ Ø¯Ù‚Ø© Ø§Ù„Ù†Ù…ÙˆØ°Ø¬ Ø¹Ù„Ù‰ Ø¨ÙŠØ§Ù†Ø§Øª Ø§Ù„Ø§Ø®ØªØ¨Ø§Ø±
acc = correct / total
print(f"Test Accuracy: {acc:.4f}")







