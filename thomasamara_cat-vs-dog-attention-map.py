import numpy as np
import torch
from torchvision import models
import zipfile


train = "/kaggle/input/dogs-vs-cats/train.zip"
test  = "/kaggle/input/dogs-vs-cats/test1.zip"


with zipfile.ZipFile(train, 'r') as zip_ref:
    zip_ref.extractall("/kaggle/working/train")


with zipfile.ZipFile(test, 'r') as zip_ref:
    zip_ref.extractall("/kaggle/working/test")


train_data = "/kaggle/working/train/train" #it contain images with name like cat.1.jpg, cat.2.jpg, dog.1.jpg, etc
test_data = "/kaggle/working/test/test1" #the images naming is only 1.jpg, 2.jpg, etc


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Toy feature map: 3 pixels, 2 channels
X_pixels = torch.tensor([[1., 0.],   # Pixel A
                         [0., 1.],   # Pixel B
                         [1., 1.]])  # Pixel C

# Weight matrices (kept the same for comparison)
W_Q = torch.tensor([[1., 0.],
                    [0., 1.]])
W_K = torch.tensor([[1., 1.],
                    [0., 1.]])
W_V = torch.tensor([[1., 0.],
                    [0., 1.]])

# --- 1. Add the [CLS] Token ---
# In a real model, this would be a learnable nn.Parameter.
# Here, we'll just define it as a tensor for demonstration.
cls_token = torch.tensor([[-0.5, 0.5]]) # A distinct starting feature

# Prepend the CLS token to the sequence of pixels
X_with_cls = torch.cat([cls_token, X_pixels], dim=0)


print("--- Input with [CLS] Token ---")
print(X_with_cls)


# --- 2. Run the Attention Mechanism ---
# The process is the same, but now on a 4x2 input tensor
Q = X_with_cls @ W_Q
K = X_with_cls @ W_K
V = X_with_cls @ W_V

scores = Q @ K.T
attn = F.softmax(scores, dim=-1)

out = attn @ V

print("\n--- New 4x4 Attention Weights ---")
print(attn)


# --- 3. Visualize the New Attention Map ---
attn_np = attn.detach().numpy()

plt.figure(figsize=(7, 6))
sns.heatmap(attn_np, annot=True, fmt=".4f", cmap="viridis",
            xticklabels=['[CLS]', 'Pixel A', 'Pixel B', 'Pixel C'],
            yticklabels=['[CLS]', 'Pixel A', 'Pixel B', 'Pixel C'])

plt.title("[CLS] Token Self-Attention Map")
plt.xlabel("Tokens Being Looked At (Keys)")
plt.ylabel("Tokens Doing the Looking (Queries)")
plt.show()



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns

import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
import math


# ======================================================================================
# 1. Configuration
# ======================================================================================
TRAIN_DIR = "/kaggle/working/train/train"
TEST_DIR = "/kaggle/working/test/test1"
SUBMISSION_FILE = "submission.csv"

IMG_SIZE = 224
PATCH_SIZE = 16
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 10 
EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ======================================================================================
# 2. Custom Dataset for Cats vs. Dogs (Unchanged)
# ======================================================================================
class CatsDogsDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_test=False):
        self.data_dir = data_dir
        self.transform = transform
        self.is_test = is_test
        self.image_files = sorted(os.listdir(self.data_dir), key=lambda x: int(x.split('.')[0]) if self.is_test else (x.split('.')[0], int(x.split('.')[1])))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if not self.is_test:
            label = 0 if 'cat' in img_name else 1
            return image, torch.tensor(label, dtype=torch.long)
        else:
            image_id = int(img_name.split('.')[0])
            return image, image_id


# ======================================================================================
# 3. Image Transformations (Unchanged)
# ======================================================================================
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}


# ======================================================================================
# 4. Create DataLoaders (Unchanged)
# ======================================================================================
full_train_dataset = CatsDogsDataset(TRAIN_DIR, transform=data_transforms['train'])
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
val_dataset.dataset.transform = data_transforms['val'] 
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")


# ======================================================================================
# 5. Vision Transformer from Scratch (Returns Attention)
# ======================================================================================
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x): return self.proj(x).flatten(2).transpose(1, 2)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.qkv, self.out_proj = nn.Linear(embed_dim, embed_dim * 3), nn.Linear(embed_dim, embed_dim)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn_scores.softmax(dim=-1)
        context = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(context), attn

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Linear(embed_dim * 4, embed_dim))
    def forward(self, x):
        attn_output, attn_weights = self.attn(self.norm1(x))
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, num_heads, num_layers):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.encoder_layers = nn.ModuleList([TransformerEncoderBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        for layer in self.encoder_layers: x, attn_weights = layer(x)
        cls_output = self.norm(x[:, 0])
        return self.head(cls_output), attn_weights


# ======================================================================================
# 6. Instantiate the Model
# ======================================================================================
model = VisionTransformer(
    img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_channels=3, num_classes=2,
    embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS
).to(DEVICE)


# ======================================================================================
# 7. NEW: Epoch-by-Epoch Visualization Function
# ======================================================================================
def visualize_attention_by_epoch(model, dataloader, device, epoch):
    print(f"\n--- Visualizing Attention for Epoch {epoch} ---")
    model.eval()
    
    correct_samples, incorrect_samples = [], []
    
    # Find 5 correct and 5 incorrect samples
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits, attn_weights = model(images)
            preds = torch.argmax(logits, dim=1)
            
            for i in range(images.size(0)):
                is_correct = (preds[i] == labels[i]).item()
                sample = {
                    'image': images[i], 'label': labels[i], 
                    'pred': preds[i], 'attn': attn_weights[i]
                }
                if is_correct and len(correct_samples) < 5:
                    correct_samples.append(sample)
                elif not is_correct and len(incorrect_samples) < 5:
                    incorrect_samples.append(sample)
            
            if len(correct_samples) >= 5 and len(incorrect_samples) >= 5:
                break

    # If not enough incorrect samples, fill with correct ones
    samples_to_show = incorrect_samples + correct_samples
    samples_to_show = samples_to_show[:10]
    
    if not samples_to_show:
        print("Not enough samples to visualize.")
        return

    inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
    
    fig, axes = plt.subplots(2, len(samples_to_show), figsize=(20, 5))
    
    for i, sample in enumerate(samples_to_show):
        img = inv_normalize(sample['image']).cpu().permute(1, 2, 0).numpy()
        
        # Process attention map
        cls_attn = sample['attn'][:, 0, 1:].mean(dim=0) # Avg heads, CLS token -> patches
        num_patches_side = int(math.sqrt(cls_attn.shape[0]))
        attn_map = cls_attn.cpu().reshape(num_patches_side, num_patches_side).numpy()
        
        # Plot original image
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        true_label = "Dog" if sample['label'].item() == 1 else "Cat"
        pred_label = "Dog" if sample['pred'].item() == 1 else "Cat"
        color = "green" if true_label == pred_label else "red"
        axes[0, i].set_title(f"True: {true_label}\nPred: {pred_label}", color=color)
        
        # Plot attention map
        axes[1, i].imshow(attn_map, cmap='viridis')
        axes[1, i].axis('off')

    plt.suptitle(f"Attention Maps - Epoch {epoch}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ======================================================================================
# 8. Training Loop (MODIFIED to call visualization)
# ======================================================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
    model.train()
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
    
    # --- Validation and Visualization after each epoch ---
    model.eval()
    val_corrects = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            logits, _ = model(inputs)
            preds = torch.argmax(logits, 1)
            val_corrects += torch.sum(preds == labels.data)
    
    val_acc = val_corrects.double() / len(val_dataset)
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    # Call the new visualization function
    visualize_attention_by_epoch(model, val_loader, DEVICE, epoch + 1)

print("\n--- Training Finished ---")



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
import timm # PyTorch Image Models library

import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
import math


# ======================================================================================
# 1. Configuration
# ======================================================================================
TRAIN_DIR = "/kaggle/working/train/train"
TEST_DIR = "/kaggle/working/test/test1"
SUBMISSION_FILE = "submission.csv"

# Model and training parameters
IMG_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 5 # Pre-trained models need fewer epochs for fine-tuning
MODEL_NAME = 'vit_base_patch16_224' # Pre-trained Vision Transformer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ======================================================================================
# 2. Custom Dataset for Cats vs. Dogs (Unchanged)
# ======================================================================================
class CatsDogsDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_test=False):
        self.data_dir = data_dir
        self.transform = transform
        self.is_test = is_test
        self.image_files = sorted(os.listdir(self.data_dir), key=lambda x: int(x.split('.')[0]) if self.is_test else (x.split('.')[0], int(x.split('.')[1])))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if not self.is_test:
            label = 0 if 'cat' in img_name else 1
            return image, torch.tensor(label, dtype=torch.long)
        else:
            image_id = int(img_name.split('.')[0])
            return image, image_id


# ======================================================================================
# 3. Image Transformations (Unchanged)
# ======================================================================================
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}


# ======================================================================================
# 4. Create DataLoaders (Unchanged)
# ======================================================================================
full_train_dataset = CatsDogsDataset(TRAIN_DIR, transform=data_transforms['train'])
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
val_dataset.dataset.transform = data_transforms['val'] 
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")


# ======================================================================================
# 5. Instantiate the Pre-trained Model
# ======================================================================================
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=2).to(DEVICE)
model


print(f"Model '{MODEL_NAME}' loaded with {sum(p.numel() for p in model.parameters()):,} parameters.")


# ======================================================================================
# 6. NEW: Monkey-Patching the Attention Block to Capture Attention
# ======================================================================================

def new_forward(self, x):
    # This is a recreation of the original timm forward pass
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    
    # *** KEY CHANGE: Save the attention map to an attribute ***
    self.attn_map = attn
    
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

# We use __get__ to bind the new method to the instance of the attention block
model.blocks[-1].attn.forward = new_forward.__get__(model.blocks[-1].attn)


# ======================================================================================
# 7. Epoch-by-Epoch Visualization Function (MODIFIED to use new attribute)
# ======================================================================================
def visualize_attention_by_epoch(model, dataloader, device, epoch):
    print(f"\n--- Visualizing Attention for Epoch {epoch} ---")
    model.eval()
    
    correct_samples, incorrect_samples = [], []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            
            # *** KEY CHANGE: Get attention map from our new attribute ***
            attn_weights = model.blocks[-1].attn.attn_map
            
            for i in range(images.size(0)):
                is_correct = (preds[i] == labels[i]).item()
                sample = {'image': images[i], 'label': labels[i], 'pred': preds[i], 'attn': attn_weights[i]}
                if is_correct and len(correct_samples) < 5: correct_samples.append(sample)
                elif not is_correct and len(incorrect_samples) < 5: incorrect_samples.append(sample)
            
            if len(correct_samples) >= 5 and len(incorrect_samples) >= 5: break

    samples_to_show = incorrect_samples + correct_samples
    samples_to_show = samples_to_show[:10]
    
    if not samples_to_show:
        print("Not enough samples to visualize.")
        return

    inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
    fig, axes = plt.subplots(2, len(samples_to_show), figsize=(20, 5))
    
    for i, sample in enumerate(samples_to_show):
        img = inv_normalize(sample['image']).cpu().permute(1, 2, 0).numpy()
        cls_attn = sample['attn'][:, 0, 1:].mean(dim=0)
        num_patches_side = int(math.sqrt(cls_attn.shape[0]))
        attn_map = cls_attn.cpu().reshape(num_patches_side, num_patches_side).numpy()
        
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        true_label = "Dog" if sample['label'].item() == 1 else "Cat"
        pred_label = "Dog" if sample['pred'].item() == 1 else "Cat"
        color = "green" if true_label == pred_label else "red"
        axes[0, i].set_title(f"True: {true_label}\nPred: {pred_label}", color=color)
        
        axes[1, i].imshow(attn_map, cmap='viridis')
        axes[1, i].axis('off')

    plt.suptitle(f"Attention Maps - Epoch {epoch}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ======================================================================================
# 8. Training Loop (Unchanged)
# ======================================================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
    model.train()
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
    
    model.eval()
    val_corrects = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            logits = model(inputs)
            preds = torch.argmax(logits, 1)
            val_corrects += torch.sum(preds == labels.data)
    
    val_acc = val_corrects.double() / len(val_dataset)
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    visualize_attention_by_epoch(model, val_loader, DEVICE, epoch + 1)

print("\n--- Training Finished ---")

