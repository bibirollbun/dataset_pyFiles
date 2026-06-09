# ğŸ“Œ Step 1: Import needed libraries
import os
import pandas as pd
import numpy as np
import pydicom
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import timm  # For pre-trained models like EfficientNet







labels_df = pd.read_csv('/kaggle/input/rsna-breast-cancer-detection/train.csv')



base_image_path = '/kaggle/input/rsna-bcd-1024x512-preprocessed/train_images'

labels_df['image_path'] = base_image_path + '/' + labels_df['patient_id'].astype(str) + '/' + labels_df['image_id'].astype(str) + '.png'




labels_df['exists'] = labels_df['image_path'].apply(lambda x: os.path.exists(x))
print(labels_df['exists'].value_counts())




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




# Ø£ÙˆÙ„Ø§Ù‹: ØªÙ‚Ø³ÙŠÙ… 80% ØªØ¯Ø±ÙŠØ¨ Ùˆ20% (Ù‡Ù‚Ø³Ù…Ù‡Ù… Ù„Ø§Ø­Ù‚Ù‹Ø§ Ù„Ù€ val/test)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['cancer'])

# Ø«Ø§Ù†ÙŠØ§Ù‹: ØªÙ‚Ø³ÙŠÙ… 20% Ø§Ù„Ø¨Ø§Ù‚ÙŠÙŠÙ† Ø¥Ù„Ù‰ 10% Ù�Ø§Ù„ÙŠØ¯ÙŠØ´Ù† Ùˆ10% ØªÙŠØ³Øª
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['cancer'])

# Ø·Ø¨Ø§Ø¹Ø© Ø­Ø¬Ù… ÙƒÙ„ Ù…Ø¬Ù…ÙˆØ¹Ø© Ù„Ù„ØªØ£ÙƒØ¯
print(f"Train size: {len(train_df)}")
print(f"Validation size: {len(valid_df)}")
print(f"Test size: {len(test_df)}")




# Ø¥Ø¹Ø¯Ø§Ø¯ Ø§Ù„ØªØ­ÙˆÙŠÙ„Ø§Øª transformations
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




# Ø¯Ø§Ù„Ø© Ù„Ø¹Ø±Ø¶ Ù…Ø¬Ù…ÙˆØ¹Ø© ØµÙˆØ± Ù…Ø¹ Ø§Ù„Ù„Ø§Ø¨Ù„Ø²
def show_batch(loader):
    images, labels = next(iter(loader))  # Ù†Ø¬ÙŠØ¨ Ø£ÙˆÙ„ batch
    images = images[:8]  # Ù†Ø¹Ø±Ø¶ Ø£ÙˆÙ„ 8 ØµÙˆØ± Ù…Ø«Ù„Ø§Ù‹
    labels = labels[:8]
    
    plt.figure(figsize=(16, 8))
    for i in range(len(images)):
        img = images[i].permute(1, 2, 0).numpy()  # ØªØ­ÙˆÙŠÙ„ Ù…Ù† (C, H, W) Ø¥Ù„Ù‰ (H, W, C)
        plt.subplot(2, 4, i + 1)
        plt.imshow(img)
        plt.title(f'Label: {int(labels[i].item())}')
        plt.axis('off')
    plt.show()

# Ø§Ø³ØªØ¯Ø¹Ø§Ø¡ Ø§Ù„Ø¯Ø§Ù„Ø©
show_batch(train_loader)




# Ø§Ù„ØªØ£ÙƒØ¯ Ù…Ù† ÙˆØ¬ÙˆØ¯ GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# ØªØ­Ù…ÙŠÙ„ Ù…ÙˆØ¯ÙŠÙ„ EfficientNetB5
from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights

# Ù†Ø³ØªØ®Ø¯Ù… Ø§Ù„Ù€ Weights Ø§Ù„Ù…Ø³Ø¨Ù‚Ø© Ù„Ùˆ Ø¹Ø§ÙŠØ²ÙŠÙ† (Transfer Learning)
weights = EfficientNet_B5_Weights.IMAGENET1K_V1
model = efficientnet_b5(weights=weights)

# ØªØ¹Ø¯ÙŠÙ„ Ø¢Ø®Ø± Ø·Ø¨Ù‚Ø© Ù„ÙŠÙ†Ø§Ø³Ø¨ Ø¹Ø¯Ø¯ Ø§Ù„ÙƒÙ„Ø§Ø³Ø§Øª (Ù‡Ù†Ø§ 2: Cancer / No Cancer)
model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=1)

# Ù†Ø±Ø³Ù„ Ø§Ù„Ù…ÙˆØ¯ÙŠÙ„ Ù„Ù„Ù€ device (GPU Ø£Ùˆ CPU)
model = model.to(device)

# Ø·Ø¨Ø§Ø¹Ø© Ù…Ù„Ø®Øµ Ù„Ù„Ù…ÙˆØ¯ÙŠÙ„
print(model)



# ØªØ¹Ø¨ÙŠØ± Ø¹Ù† ØªÙ…Ø±ÙŠØ± ØµÙˆØ±Ø© Ø®Ù„Ø§Ù„ Ø§Ù„Ù…ÙˆØ¯ÙŠÙ„ ÙˆØ§Ø³ØªØ®Ø±Ø§Ø¬ Ø§Ù„Ù€ features

sample_batch = next(iter(train_loader))
images, labels = sample_batch
images = images.to(device)

# Ù…Ø±Ø± ØµÙˆØ±Ø© Ø®Ù„Ø§Ù„ Ø§Ù„Ù…ÙˆØ¯ÙŠÙ„ Ø¨Ø¯ÙˆÙ† ØªØ­Ø¯ÙŠØ« Ø§Ù„Ù…Ø¹Ø§Ù…Ù„Ø§Øª
with torch.no_grad():
    features = model.features(images)  # Ø§Ø³ØªØ®Ø±Ø§Ø¬ Ø§Ù„Ù€ features Ù‚Ø¨Ù„ Ø·Ø¨Ù‚Ø© Ø§Ù„ØªØµÙ†ÙŠÙ� Ø§Ù„Ù†Ù‡Ø§Ø¦ÙŠØ©

print(f'Extracted feature map shape: {features.shape}')




# ØªØ­Ø¯ÙŠØ¯ Ø§Ù„Ø¬Ù‡Ø§Ø²
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ØªØ¹Ø±ÙŠÙ� Ø§Ù„Ù�Ù‚Ø¯
criterion = nn.BCEWithLogitsLoss()

# ØªØ¹Ø±ÙŠÙ� Ø§Ù„Ø£ÙˆØ¨ØªÙ…ÙŠÙ…Ø§ÙŠØ²Ø±
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Ø¹Ø¯Ø¯ Ø§Ù„Ø¥ÙŠØ¨ÙˆÙƒØ³
epochs = 5

# Ø§Ù„ØªØ¯Ø±ÙŠØ¨
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
    
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f}")

    # ØªÙ‚ÙŠÙŠÙ… Ø¨Ø¹Ø¯ ÙƒÙ„ Ø§ÙŠØ¨ÙˆÙƒ
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"Validation Accuracy after Epoch {epoch+1}: {acc:.4f}")



correct = 0
total = 0

with torch.no_grad():  # We don't need gradients for testing
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)  # Forward pass
        _, predicted = torch.max(outputs, 1)  # Get predicted class
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")


