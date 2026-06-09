# # IMPORTANT: SOME KAGGLE DATA SOURCES ARE PRIVATE
# # RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES.
# import kagglehub
# kagglehub.login()


# # IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# # THEN FEEL FREE TO DELETE THIS CELL.
# # NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# # ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# # NOTEBOOK.

# srifoton_25_machine_learning_competition_path = kagglehub.competition_download('srifoton-25-machine-learning-competition')

# print('Data source import complete.')



pip install tensorflow


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models


# srifoton_25_machine_learning_competition_path
pip install albumentations


pip install albumentations


img_size = (224, 224)   # ukuran standar untuk CNN
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    "/kaggle/input/srifoton-25-machine-learning-competition/train/train",
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "/kaggle/input/srifoton-25-machine-learning-competition/val/val",
    image_size=img_size,
    batch_size=batch_size
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "/kaggle/input/srifoton-25-machine-learning-competition/test",
    image_size=img_size,
    batch_size=batch_size
)


# resnet50.a1_in1k
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
import timm
from safetensors.torch import save_file, load_file
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold, train_test_split
import os
from PIL import Image
import pandas as pd
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- Seed & Device ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# --- Dataset setup ---
data_dir = "/kaggle/input/srifoton-25-machine-learning-competition"
img_size, batch_size = 224, 8

# Custom Dataset Class untuk Multiple Augmentation
class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, num_augments=3):
        self.original_dataset = original_dataset
        self.num_augments = num_augments
        
        # Transform untuk augmentasi
        self.augment_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-30, 30)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
            ], p=0.3),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.05, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
        ])
        
        # Transform untuk original (tanpa augmentasi random)
        self.original_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.original_dataset) * (self.num_augments + 1)  # +1 untuk original
    
    def __getitem__(self, idx):
        # Hitung index asli dan augmentation index
        original_idx = idx // (self.num_augments + 1)
        aug_idx = idx % (self.num_augments + 1)
        
        # Ambil original image dan label
        img_path = self.original_dataset.samples[original_idx][0]
        label = self.original_dataset.samples[original_idx][1]
        img = Image.open(img_path).convert("RGB")
        
        # Apply transform sesuai index
        if aug_idx == 0:
            # Original image
            img = self.original_transform(img)
        else:
            # Augmented versions
            img = self.augment_transform(img)
            
        return img, label

# Validation/test transform tanpa augmentasi
val_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# Load original datasets
original_train = datasets.ImageFolder(root=f"{data_dir}/train/train")
original_val = datasets.ImageFolder(root=f"{data_dir}/val/val")

# Buat augmented datasets (1x augmented + 1 original = 2x total)
augmented_train = AugmentedDataset(original_train, num_augments=1)
augmented_val = AugmentedDataset(original_val, num_augments=1)

# Gabung train + val eksternal
full_dataset = ConcatDataset([augmented_train, augmented_val])

# Ambil target labels (duplikasi sesuai augmentasi)
train_targets = []
for target in original_train.targets:
    train_targets.extend([target] * 2)  # 2x karena 1 augment + 1 original

val_targets = []
for target in original_val.targets:
    val_targets.extend([target] * 2)

targets = np.array(train_targets + val_targets)

num_classes = len(original_train.classes)
print("Jumlah Class:", num_classes)
print("Classes:", original_train.classes)
print("Original samples (train):", len(original_train))
print("Original samples (val):", len(original_val))
print("Augmented samples (train):", len(augmented_train))
print("Augmented samples (val):", len(augmented_val))
print("Total samples (train+val) after augmentation:", len(full_dataset))

# --- Split Dataset for Training and Validation ---
# Use train_test_split to create a validation set (e.g., 80% train, 20% val)
train_idx, val_idx = train_test_split(
    np.arange(len(full_dataset)),
    test_size=0.2,
    stratify=targets,
    random_state=seed
)

train_subset = Subset(full_dataset, train_idx)
val_subset = Subset(full_dataset, val_idx)

# Create DataLoaders
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)

print(f"Training data: {len(train_subset)} samples")
print(f"Validation data: {len(val_subset)} samples")

# --- Training Final Menggunakan Seluruh Data dengan Early Stopping ---
# Hyperparameters
epochs, lr = 15, 5e-5
patience = 3  # Number of epochs to wait before stopping if no improvement
best_val_loss = float('inf')
patience_counter = 0
best_model_path = "best_model.safetensors"

# Define Model, Optimizer, and Scheduler
final_model = timm.create_model(
    "convnext_base_in22k",
    pretrained=True,
    num_classes=num_classes
).to(device)

optimizer = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=1e-2)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
criterion = nn.CrossEntropyLoss()

print("\nMemulai training final dengan early stopping...")

final_model.train()
for epoch in range(epochs):
    # Training loop
    epoch_loss = 0.0
    final_model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = final_model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_train_loss = epoch_loss / len(train_loader)
    
    # Validation loop
    final_model.eval()
    val_loss = 0.0
    val_preds, val_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = final_model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(labels.cpu().numpy())
    
    avg_val_loss = val_loss / len(val_loader)
    val_f1 = f1_score(val_labels, val_preds, average='macro')
    
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Early stopping logic
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        # Save the best model
        save_file(final_model.state_dict(), best_model_path)
        print(f"Validation loss improved. Saving best model to {best_model_path}")
    else:
        patience_counter += 1
        print(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}")
    
    if patience_counter >= patience:
        print(f"\nEarly stopping triggered after {epoch+1} epochs.")
        break
    
    # Step the scheduler
    scheduler.step()

print("\nTraining final selesai.")

# Load the best model for inference
final_model.load_state_dict(load_file(best_model_path))
print(f"Loaded best model from {best_model_path}")

# --- Prediksi pada Data Test dengan Test-Time Augmentation (TTA) ---
print("\nMemulai prediksi pada data test dengan TTA...")
test_dir = f"{data_dir}/test/test"
test_files = sorted(os.listdir(test_dir))

final_model.eval()
all_test_preds = []
with torch.no_grad():
    for fname in test_files:
        img_path = os.path.join(test_dir, fname)
        img = Image.open(img_path).convert("RGB")
        
        # TTA: Prediksi pada gambar asli dan versi flip
        # 1. Gambar asli
        img_original = val_transform(img).unsqueeze(0).to(device)
        probs_original = torch.softmax(final_model(img_original), dim=1)
        
        # 2. Gambar yang di-flip horizontal
        img_flipped = transforms.functional.hflip(img)
        img_flipped_tensor = val_transform(img_flipped).unsqueeze(0).to(device)
        probs_flipped = torch.softmax(final_model(img_flipped_tensor), dim=1)
        
        # 3. Rata-ratakan probabilitas dari kedua versi
        avg_probs = (probs_original + probs_flipped) / 2
        pred = avg_probs.argmax(1).cpu().item()
        all_test_preds.append(pred)

print("Prediksi dengan TTA selesai.")

# --- Buat dan Simpan File Submisi ---
submission_df = pd.DataFrame({
    'Id': test_files,
    'Predicted': all_test_preds
})
submission_df.to_csv("submission.csv", index=False)

print("\nFile submisi 'submission.csv' berhasil dibuat!")
print(submission_df.head())


# resnet50.a1_in1k
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
import timm
from safetensors.torch import save_file, load_file
import numpy as np
import random
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import pandas as pd
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- Seed & Device ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# --- Dataset setup ---
data_dir = "/kaggle/input/srifoton-25-machine-learning-competition"
img_size, batch_size = 224, 8

# Custom Dataset Class untuk Multiple Augmentation
class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, num_augments=1):
        self.original_dataset = original_dataset
        self.num_augments = num_augments
        
        # Transform untuk augmentasi (lebih ringan untuk X-ray)
        self.augment_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(degrees=(-30, 30)),  # kecil saja
            transforms.RandomAffine(
                degrees=0, 
                translate=(0.05, 0.05),
                scale=(0.95, 1.05)
            ),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
            ], p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        
        # Transform untuk original (konsisten dengan val/test)
        self.original_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        
    def __len__(self):
        return len(self.original_dataset) * (self.num_augments + 1)
    
    def __getitem__(self, idx):
        original_idx = idx // (self.num_augments + 1)
        aug_idx = idx % (self.num_augments + 1)
        
        img_path, label = self.original_dataset.samples[original_idx]
        img = Image.open(img_path).convert("RGB")
        
        if aug_idx == 0:
            img = self.original_transform(img)
        else:
            img = self.augment_transform(img)
            
        return img, label

# Validation/test transform
val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load datasets
original_train = datasets.ImageFolder(root=f"{data_dir}/train/train")
original_val = datasets.ImageFolder(root=f"{data_dir}/val/val")

# Buat augmented datasets (2 augment + 1 original = 3x total)
augmented_train = AugmentedDataset(original_train, num_augments=1)
augmented_val = AugmentedDataset(original_val, num_augments=1)

# Gabung train + val eksternal
full_dataset = ConcatDataset([augmented_train, augmented_val])

# Ambil target labels (duplikasi sesuai augmentasi)
train_targets = []
for target in original_train.targets:
    train_targets.extend([target] * 2)  # 2 augment + 1 original

val_targets = []
for target in original_val.targets:
    val_targets.extend([target] * 2)

targets = np.array(train_targets + val_targets)

num_classes = len(original_train.classes)
print("Jumlah Class:", num_classes)
print("Classes:", original_train.classes)
print("Original samples (train):", len(original_train))
print("Original samples (val):", len(original_val))
print("Augmented samples (train):", len(augmented_train))
print("Augmented samples (val):", len(augmented_val))
print("Total samples (train+val) after augmentation:", len(full_dataset))

# --- Split Dataset for Training and Validation ---
train_idx, val_idx = train_test_split(
    np.arange(len(full_dataset)),
    test_size=0.1,
    stratify=targets,
    random_state=seed
)

train_subset = Subset(full_dataset, train_idx)
val_subset = Subset(full_dataset, val_idx)

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)

print(f"Training data: {len(train_subset)} samples")
print(f"Validation data: {len(val_subset)} samples")

# --- Training Final dengan Early Stopping ---
epochs, lr = 10, 5e-5
patience = 3
best_val_loss = float('inf')
patience_counter = 0
best_model_path = "best_model.safetensors"

final_model = timm.create_model(
    "tf_efficientnetv2_m_in21ft1k",
    pretrained=True,
    num_classes=num_classes
).to(device)

optimizer = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=1e-2)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-7)
criterion = nn.CrossEntropyLoss()

print("\nMemulai training final dengan early stopping...")

for epoch in range(epochs):
    # Training loop
    final_model.train()
    epoch_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = final_model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_train_loss = epoch_loss / len(train_loader)
    
    # Validation loop
    final_model.eval()
    val_loss, val_preds, val_labels = 0.0, [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = final_model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(labels.cpu().numpy())
    
    avg_val_loss = val_loss / len(val_loader)
    val_f1 = f1_score(val_labels, val_preds, average='macro')
    
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f} | "
          f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Early stopping logic
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        save_file(final_model.state_dict(), best_model_path)
        print(f"Validation loss improved. Saving best model to {best_model_path}")
    else:
        patience_counter += 1
        print(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}")
    
    if patience_counter >= patience:
        print(f"\nEarly stopping triggered after {epoch+1} epochs.")
        break
    
    scheduler.step()

print("\nTraining final selesai.")

# Load best model
final_model.load_state_dict(load_file(best_model_path))
print(f"Loaded best model from {best_model_path}")

# --- Prediksi Test dengan TTA ---
print("\nMemulai prediksi pada data test dengan TTA...")
test_dir = f"{data_dir}/test/test"
test_files = sorted(os.listdir(test_dir))

final_model.eval()
all_test_preds = []
with torch.no_grad():
    for fname in test_files:
        img_path = os.path.join(test_dir, fname)
        img = Image.open(img_path).convert("RGB")
        
        # 1. Original
        img_original = val_transform(img).unsqueeze(0).to(device)
        probs_original = torch.softmax(final_model(img_original), dim=1)
        
        # 2. Flipped
        img_flipped = transforms.functional.hflip(img)
        img_flipped_tensor = val_transform(img_flipped).unsqueeze(0).to(device)
        probs_flipped = torch.softmax(final_model(img_flipped_tensor), dim=1)
        
        # Average
        avg_probs = (probs_original + probs_flipped) / 2
        pred = avg_probs.argmax(1).cpu().item()
        all_test_preds.append(pred)

print("Prediksi dengan TTA selesai.")

# --- Simpan Submission ---
submission_df = pd.DataFrame({
    'Id': test_files,
    'Predicted': all_test_preds
})
submission_df.to_csv("submission.csv", index=False)

print("\nFile submisi 'submission.csv' berhasil dibuat!")
print(submission_df.head())


# resnet50.a1_in1k
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
import timm
from safetensors.torch import save_file, load_file
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold, train_test_split
import os
from PIL import Image
import pandas as pd
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- Seed & Device ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# --- Dataset setup ---
data_dir = "/kaggle/input/srifoton-25-machine-learning-competition"
img_size, batch_size = 224, 8

# Custom Dataset Class untuk Multiple Augmentation
class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, num_augments=3):
        self.original_dataset = original_dataset
        self.num_augments = num_augments
        
        # Transform untuk augmentasi
        self.augment_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-30, 30)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
            ], p=0.3),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.05, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
        ])
        
        # Transform untuk original (tanpa augmentasi random)
        self.original_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.original_dataset) * (self.num_augments + 1)  # +1 untuk original
    
    def __getitem__(self, idx):
        # Hitung index asli dan augmentation index
        original_idx = idx // (self.num_augments + 1)
        aug_idx = idx % (self.num_augments + 1)
        
        # Ambil original image dan label
        img_path = self.original_dataset.samples[original_idx][0]
        label = self.original_dataset.samples[original_idx][1]
        img = Image.open(img_path).convert("RGB")
        
        # Apply transform sesuai index
        if aug_idx == 0:
            # Original image
            img = self.original_transform(img)
        else:
            # Augmented versions
            img = self.augment_transform(img)
            
        return img, label

# Validation/test transform tanpa augmentasi
val_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# Load original datasets
original_train = datasets.ImageFolder(root=f"{data_dir}/train/train")
original_val = datasets.ImageFolder(root=f"{data_dir}/val/val")

# Buat augmented datasets (1x augmented + 1 original = 2x total)
augmented_train = AugmentedDataset(original_train, num_augments=1)
augmented_val = AugmentedDataset(original_val, num_augments=1)

# Gabung train + val eksternal
full_dataset = ConcatDataset([augmented_train, augmented_val])

# Ambil target labels (duplikasi sesuai augmentasi)
train_targets = []
for target in original_train.targets:
    train_targets.extend([target] * 2)  # 2x karena 1 augment + 1 original

val_targets = []
for target in original_val.targets:
    val_targets.extend([target] * 2)

targets = np.array(train_targets + val_targets)

num_classes = len(original_train.classes)
print("Jumlah Class:", num_classes)
print("Classes:", original_train.classes)
print("Original samples (train):", len(original_train))
print("Original samples (val):", len(original_val))
print("Augmented samples (train):", len(augmented_train))
print("Augmented samples (val):", len(augmented_val))
print("Total samples (train+val) after augmentation:", len(full_dataset))

# --- Split Dataset for Training and Validation ---
# Use train_test_split to create a validation set (e.g., 80% train, 20% val)
train_idx, val_idx = train_test_split(
    np.arange(len(full_dataset)),
    test_size=0.2,
    stratify=targets,
    random_state=seed
)

train_subset = Subset(full_dataset, train_idx)
val_subset = Subset(full_dataset, val_idx)

# Create DataLoaders
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)

print(f"Training data: {len(train_subset)} samples")
print(f"Validation data: {len(val_subset)} samples")

# --- Training Final Menggunakan Seluruh Data dengan Early Stopping ---
# Hyperparameters
epochs, lr = 2, 5e-5
patience = 3  # Number of epochs to wait before stopping if no improvement
best_val_loss = float('inf')
patience_counter = 0
best_model_path = "best_model.safetensors"

# Define Model, Optimizer, and Scheduler
final_model = timm.create_model(
    "tf_efficientnetv2_s_in21ft1k",
    pretrained=True,
    num_classes=num_classes
).to(device)

optimizer = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=1e-2)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
criterion = nn.CrossEntropyLoss()

print("\nMemulai training final dengan early stopping...")

final_model.train()
for epoch in range(epochs):
    # Training loop
    epoch_loss = 0.0
    final_model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = final_model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_train_loss = epoch_loss / len(train_loader)
    
    # Validation loop
    final_model.eval()
    val_loss = 0.0
    val_preds, val_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = final_model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(labels.cpu().numpy())
    
    avg_val_loss = val_loss / len(val_loader)
    val_f1 = f1_score(val_labels, val_preds, average='macro')
    
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Early stopping logic
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        # Save the best model
        save_file(final_model.state_dict(), best_model_path)
        print(f"Validation loss improved. Saving best model to {best_model_path}")
    else:
        patience_counter += 1
        print(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}")
    
    if patience_counter >= patience:
        print(f"\nEarly stopping triggered after {epoch+1} epochs.")
        break
    
    # Step the scheduler
    scheduler.step()

print("\nTraining final selesai.")

# Load the best model for inference
final_model.load_state_dict(load_file(best_model_path))
print(f"Loaded best model from {best_model_path}")

# --- Prediksi pada Data Test dengan Test-Time Augmentation (TTA) ---
print("\nMemulai prediksi pada data test dengan TTA...")
test_dir = f"{data_dir}/test/test"
test_files = sorted(os.listdir(test_dir))

final_model.eval()
all_test_preds = []
with torch.no_grad():
    for fname in test_files:
        img_path = os.path.join(test_dir, fname)
        img = Image.open(img_path).convert("RGB")
        
        # TTA: Prediksi pada gambar asli dan versi flip
        # 1. Gambar asli
        img_original = val_transform(img).unsqueeze(0).to(device)
        probs_original = torch.softmax(final_model(img_original), dim=1)
        
        # 2. Gambar yang di-flip horizontal
        img_flipped = transforms.functional.hflip(img)
        img_flipped_tensor = val_transform(img_flipped).unsqueeze(0).to(device)
        probs_flipped = torch.softmax(final_model(img_flipped_tensor), dim=1)
        
        # 3. Rata-ratakan probabilitas dari kedua versi
        avg_probs = (probs_original + probs_flipped) / 2
        pred = avg_probs.argmax(1).cpu().item()
        all_test_preds.append(pred)

print("Prediksi dengan TTA selesai.")

# --- Buat dan Simpan File Submisi ---
submission_df = pd.DataFrame({
    'Id': test_files,
    'Predicted': all_test_preds
})
submission_df.to_csv("submission.csv", index=False)

print("\nFile submisi 'submission.csv' berhasil dibuat!")
print(submission_df.head())


# resnet50.a1_in1k
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
import timm
from safetensors.torch import save_file, load_file
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
import os
from PIL import Image
import pandas as pd
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR
import albumentations as A
import albumentations.pytorch as A_pytorch
import cv2

# --- Seed & Device ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# --- Dataset setup ---
data_dir = "/kaggle/input/srifoton-25-machine-learning-competition"
img_size, batch_size = 384, 8

# Custom function for salt-and-pepper noise using Albumentations
def add_salt_pepper_noise(image):
    """
    Add salt-and-pepper noise using Albumentations.
    """
    return A.Compose([
        A.RandomSaltAndPepper(p=0.01, salt=1.0, pepper=1.0)
    ])(image=image)['image']

# Custom Dataset Class dengan Albumentations
class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, num_augments=1):
        self.original_dataset = original_dataset
        self.num_augments = num_augments
        
        # Albumentations pipeline untuk augmentasi (menggantikan TrivialAugment)
        self.augment_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=(-45, 45), p=0.5),
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.1, p=0.5),
                A.GridDistortion(p=0.5),
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
            ], p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1, p=0.5),
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.3),
            ], p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.pytorch.ToTensorV2(),
        ])
        
        # Transform untuk original (tanpa augmentasi random, hanya resize dan normalize)
        self.original_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.pytorch.ToTensorV2(),
        ])
        
    def __len__(self):
        return len(self.original_dataset) * (self.num_augments + 1)  # +1 untuk original
    
    def __getitem__(self, idx):
        # Hitung index asli dan augmentation index
        original_idx = idx // (self.num_augments + 1)
        aug_idx = idx % (self.num_augments + 1)
        
        # Ambil original image dan label
        img_path = self.original_dataset.samples[original_idx][0]
        label = self.original_dataset.samples[original_idx][1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transform sesuai index
        if aug_idx == 0:
            # Original image
            transformed = self.original_transform(image=image)
            img = transformed['image']
        else:
            # Augmented versions
            transformed = self.augment_transform(image=image)
            img = transformed['image']
            
        return img, torch.tensor(label)

# Validation/test transform dengan Albumentations (tanpa augmentasi)
val_transform_alb = A.Compose([
    A.Resize(img_size, img_size),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.pytorch.ToTensorV2(),
])

# Load original datasets
original_train = datasets.ImageFolder(root=f"{data_dir}/train/train")
original_val = datasets.ImageFolder(root=f"{data_dir}/val/val")

# Buat augmented datasets (1x augmented + 1 original = 2x total)
augmented_train = AugmentedDataset(original_train, num_augments=1)
augmented_val = AugmentedDataset(original_val, num_augments=1)

# gabung train + val eksternal
full_dataset = ConcatDataset([augmented_train, augmented_val])

# ambil target labels (duplikasi sesuai augmentasi)
train_targets = []
for target in original_train.targets:
    train_targets.extend([target] * 2)  # 2x karena 1 augment + 1 original

val_targets = []
for target in original_val.targets:
    val_targets.extend([target] * 2)

targets = np.array(train_targets + val_targets)

num_classes = len(original_train.classes)
print("Jumlah Class:", num_classes)
print("Classes:", original_train.classes)
print("Original samples (train):", len(original_train))
print("Original samples (val):", len(original_val))
print("Augmented samples (train):", len(augmented_train))
print("Augmented samples (val):", len(augmented_val))
print("Total samples (train+val) after augmentation:", len(full_dataset))

# --- Training Final Menggunakan Seluruh Data (Tanpa K-Fold) ---
full_train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
print(f"Total data untuk training final: {len(full_dataset)}")

# --- Hyperparameters ---
epochs, lr = 10, 5e-5

# --- Model, Optimizer, and Scheduler ---
final_model = timm.create_model(
    "tf_efficientnetv2_m_in21ft1k",
    pretrained=True,
    num_classes=num_classes
).to(device)

optimizer = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=1e-2)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
criterion = nn.CrossEntropyLoss()

print("\nMemulai training final dengan model dan optimizer baru...")
final_model.train()

for epoch in range(epochs):
    epoch_loss = 0.0
    for imgs, labels in full_train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = final_model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    scheduler.step()
    avg_loss = epoch_loss / len(full_train_loader)
    print(f"Epoch {epoch+1}/{epochs} selesai | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

print("\nTraining final selesai.")

# --- Prediksi pada Data Test dengan Test-Time Augmentation (TTA) ---
print("\nMemulai prediksi pada data test dengan TTA...")
test_dir = f"{data_dir}/test/test"
test_files = sorted(os.listdir(test_dir))

final_model.eval()
all_test_preds = []
with torch.no_grad():
    for fname in test_files:
        img_path = os.path.join(test_dir, fname)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # --- TTA: Prediksi pada gambar asli ---
        transformed_original = val_transform_alb(image=image)
        img_original = transformed_original['image'].unsqueeze(0).to(device)
        probs_original = torch.softmax(final_model(img_original), dim=1)
        
        # --- TTA: Gambar yang di-flip horizontal ---
        flip_transform = A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.pytorch.ToTensorV2(),
        ])
        transformed_flipped = flip_transform(image=image)
        img_flipped = transformed_flipped['image'].unsqueeze(0).to(device)
        probs_flipped = torch.softmax(final_model(img_flipped), dim=1)
        
        # Rata-ratakan probabilitas
        avg_probs = (probs_original + probs_flipped) / 2
        pred = avg_probs.argmax(1).cpu().item()
        all_test_preds.append(pred)

print("Prediksi dengan TTA selesai.")

# --- Buat dan Simpan File Submisi ---
submission_df = pd.DataFrame({
    'Id': test_files,
    'Predicted': all_test_preds
})
submission_df.to_csv("submission.csv", index=False)

print("\nFile submisi 'submission.csv' berhasil dibuat!")
print(submission_df.head())


# resnet50.a1_in1k
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
import timm
from safetensors.torch import save_file, load_file
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold, train_test_split
import os
from PIL import Image
import pandas as pd
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- Seed & Device ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# --- Dataset setup ---
data_dir = "/kaggle/input/srifoton-25-machine-learning-competition"
img_size, batch_size = 300, 8

from PIL import ImageFilter
import torchvision.transforms.functional as F

class SharpenAndGritty:
    def __call__(self, img):
        # Sharpen (setara 200%)
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=3))
        # Gritty = tambahin noise sederhana
        np_img = np.array(img)
        noise = np.random.normal(0, 25, np_img.shape).astype(np.int16)  # 100% noise
        np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(np_img)


# Custom Dataset Class untuk Multiple Augmentation
class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, num_augments=3):
        self.original_dataset = original_dataset
        self.num_augments = num_augments
        
        # Transform untuk augmentasi
        self.augment_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-35, 35)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
            ], p=0.3),
            SharpenAndGritty(),  
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.05, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
        ])
        
        # Transform untuk original (tanpa augmentasi random)
        self.original_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.original_dataset) * (self.num_augments + 1)  # +1 untuk original
    
    def __getitem__(self, idx):
        # Hitung index asli dan augmentation index
        original_idx = idx // (self.num_augments + 1)
        aug_idx = idx % (self.num_augments + 1)
        
        # Ambil original image dan label
        img_path = self.original_dataset.samples[original_idx][0]
        label = self.original_dataset.samples[original_idx][1]
        img = Image.open(img_path).convert("RGB")
        
        # Apply transform sesuai index
        if aug_idx == 0:
            # Original image
            img = self.original_transform(img)
        else:
            # Augmented versions
            img = self.augment_transform(img)
            
        return img, label

# Validation/test transform tanpa augmentasi
val_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# Load original datasets
original_train = datasets.ImageFolder(root=f"{data_dir}/train/train")
original_val = datasets.ImageFolder(root=f"{data_dir}/val/val")

# Buat augmented datasets (1x augmented + 1 original = 2x total)
augmented_train = AugmentedDataset(original_train, num_augments=1)
augmented_val = AugmentedDataset(original_val, num_augments=1)

# Gabung train + val eksternal
full_dataset = ConcatDataset([augmented_train, augmented_val])

# Ambil target labels (duplikasi sesuai augmentasi)
train_targets = []
for target in original_train.targets:
    train_targets.extend([target] * 2)  # 2x karena 1 augment + 1 original

val_targets = []
for target in original_val.targets:
    val_targets.extend([target] * 2)

targets = np.array(train_targets + val_targets)

num_classes = len(original_train.classes)
print("Jumlah Class:", num_classes)
print("Classes:", original_train.classes)
print("Original samples (train):", len(original_train))
print("Original samples (val):", len(original_val))
print("Augmented samples (train):", len(augmented_train))
print("Augmented samples (val):", len(augmented_val))
print("Total samples (train+val) after augmentation:", len(full_dataset))

# --- Split Dataset for Training and Validation ---
# Use train_test_split to create a validation set (e.g., 80% train, 20% val)
train_idx, val_idx = train_test_split(
    np.arange(len(full_dataset)),
    test_size=0.2,
    stratify=targets,
    random_state=seed
)

train_subset = Subset(full_dataset, train_idx)
val_subset = Subset(full_dataset, val_idx)

# Create DataLoaders
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)

print(f"Training data: {len(train_subset)} samples")
print(f"Validation data: {len(val_subset)} samples")

# --- Training Final Menggunakan Seluruh Data dengan Early Stopping ---
# Hyperparameters
epochs, lr = 15, 5e-5
patience = 3  # Number of epochs to wait before stopping if no improvement
best_val_loss = float('inf')
patience_counter = 0
best_model_path = "best_model.safetensors"

# xception41.tf_in1k epoch 1 : 0.5363

# Define Model, Optimizer, and Scheduler
final_model = timm.create_model(
    "xception41.tf_in1k",
    pretrained=True,
    num_classes=num_classes
).to(device)

optimizer = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=1e-2)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
criterion = nn.CrossEntropyLoss()

print("\nMemulai training final dengan early stopping...")

final_model.train()
for epoch in range(epochs):
    # Training loop
    epoch_loss = 0.0
    final_model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = final_model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_train_loss = epoch_loss / len(train_loader)
    
    # Validation loop
    final_model.eval()
    val_loss = 0.0
    val_preds, val_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = final_model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(labels.cpu().numpy())
    
    avg_val_loss = val_loss / len(val_loader)
    val_f1 = f1_score(val_labels, val_preds, average='macro')
    
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Early stopping logic
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        # Save the best model
        save_file(final_model.state_dict(), best_model_path)
        print(f"Validation loss improved. Saving best model to {best_model_path}")
    else:
        patience_counter += 1
        print(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}")
    
    if patience_counter >= patience:
        print(f"\nEarly stopping triggered after {epoch+1} epochs.")
        break
    
    # Step the scheduler
    scheduler.step()

print("\nTraining final selesai.")

# Load the best model for inference
final_model.load_state_dict(load_file(best_model_path))
print(f"Loaded best model from {best_model_path}")

# --- Prediksi pada Data Test dengan Test-Time Augmentation (TTA) ---
print("\nMemulai prediksi pada data test dengan TTA...")
test_dir = f"{data_dir}/test/test"
test_files = sorted(os.listdir(test_dir))

final_model.eval()
all_test_preds = []
with torch.no_grad():
    for fname in test_files:
        img_path = os.path.join(test_dir, fname)
        img = Image.open(img_path).convert("RGB")
        
        # TTA: Prediksi pada gambar asli dan versi flip
        # 1. Gambar asli
        img_original = val_transform(img).unsqueeze(0).to(device)
        probs_original = torch.softmax(final_model(img_original), dim=1)
        
        # 2. Gambar yang di-flip horizontal
        img_flipped = transforms.functional.hflip(img)
        img_flipped_tensor = val_transform(img_flipped).unsqueeze(0).to(device)
        probs_flipped = torch.softmax(final_model(img_flipped_tensor), dim=1)
        
        # 3. Rata-ratakan probabilitas dari kedua versi
        avg_probs = (probs_original + probs_flipped) / 2
        pred = avg_probs.argmax(1).cpu().item()
        all_test_preds.append(pred)

print("Prediksi dengan TTA selesai.")

# --- Buat dan Simpan File Submisi ---
submission_df = pd.DataFrame({
    'Id': test_files,
    'Predicted': all_test_preds
})
submission_df.to_csv("submission.csv", index=False)

print("\nFile submisi 'submission.csv' berhasil dibuat!")
print(submission_df.head())


# resnet50.a1_in1kx
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
import timm
from safetensors.torch import save_file, load_file
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
import os
from PIL import Image
import pandas as pd
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR
import albumentations as A
import albumentations.pytorch as A_pytorch
import cv2

# --- Seed & Device ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# --- Dataset setup ---
data_dir = "/kaggle/input/srifoton-25-machine-learning-competition"
img_size, batch_size = 384, 8

# Custom function for salt-and-pepper noise using Albumentations
def add_salt_pepper_noise(image):
    """
    Add salt-and-pepper noise using Albumentations.
    """
    return A.Compose([
        A.RandomSaltAndPepper(p=0.01, salt=1.0, pepper=1.0)
    ])(image=image)['image']

# Custom Dataset Class dengan Albumentations
class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, num_augments=1):
        self.original_dataset = original_dataset
        self.num_augments = num_augments
        
        # Albumentations pipeline untuk augmentasi (menggantikan TrivialAugment)
        self.augment_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=(-45, 45), p=0.5),
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.1, p=0.5),
                A.GridDistortion(p=0.5),
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
            ], p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1, p=0.5),
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.3),
            ], p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.pytorch.ToTensorV2(),
        ])
        
        # Transform untuk original (tanpa augmentasi random, hanya resize dan normalize)
        self.original_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.pytorch.ToTensorV2(),
        ])
        
    def __len__(self):
        return len(self.original_dataset) * (self.num_augments + 1)  # +1 untuk original
    
    def __getitem__(self, idx):
        # Hitung index asli dan augmentation index
        original_idx = idx // (self.num_augments + 1)
        aug_idx = idx % (self.num_augments + 1)
        
        # Ambil original image dan label
        img_path = self.original_dataset.samples[original_idx][0]
        label = self.original_dataset.samples[original_idx][1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transform sesuai index
        if aug_idx == 0:
            # Original image
            transformed = self.original_transform(image=image)
            img = transformed['image']
        else:
            # Augmented versions
            transformed = self.augment_transform(image=image)
            img = transformed['image']
            
        return img, torch.tensor(label)

# Validation/test transform dengan Albumentations (tanpa augmentasi)
val_transform_alb = A.Compose([
    A.Resize(img_size, img_size),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.pytorch.ToTensorV2(),
])

# Load original datasets
original_train = datasets.ImageFolder(root=f"{data_dir}/train/train")
original_val = datasets.ImageFolder(root=f"{data_dir}/val/val")

# Buat augmented datasets (1x augmented + 1 original = 2x total)
augmented_train = AugmentedDataset(original_train, num_augments=1)
augmented_val = AugmentedDataset(original_val, num_augments=1)

# gabung train + val eksternal
full_dataset = ConcatDataset([augmented_train, augmented_val])

# ambil target labels (duplikasi sesuai augmentasi)
train_targets = []
for target in original_train.targets:
    train_targets.extend([target] * 2)  # 2x karena 1 augment + 1 original

val_targets = []
for target in original_val.targets:
    val_targets.extend([target] * 2)

targets = np.array(train_targets + val_targets)

num_classes = len(original_train.classes)
print("Jumlah Class:", num_classes)
print("Classes:", original_train.classes)
print("Original samples (train):", len(original_train))
print("Original samples (val):", len(original_val))
print("Augmented samples (train):", len(augmented_train))
print("Augmented samples (val):", len(augmented_val))
print("Total samples (train+val) after augmentation:", len(full_dataset))

# --- Training Final Menggunakan Seluruh Data (Tanpa K-Fold) ---
full_train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
print(f"Total data untuk training final: {len(full_dataset)}")

# --- Hyperparameters ---
epochs, lr = 10, 5e-5

# --- Model, Optimizer, and Scheduler ---
final_model = timm.create_model(
    "tf_efficientnetv2_m_in21ft1k",
    pretrained=True,
    num_classes=num_classes
).to(device)

optimizer = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=1e-2)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
criterion = nn.CrossEntropyLoss()

print("\nMemulai training final dengan model dan optimizer baru...")
final_model.train()

for epoch in range(epochs):
    epoch_loss = 0.0
    for imgs, labels in full_train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = final_model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    scheduler.step()
    avg_loss = epoch_loss / len(full_train_loader)
    print(f"Epoch {epoch+1}/{epochs} selesai | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

print("\nTraining final selesai.")

# --- Prediksi pada Data Test dengan Test-Time Augmentation (TTA) ---
print("\nMemulai prediksi pada data test dengan TTA...")
test_dir = f"{data_dir}/test/test"
test_files = sorted(os.listdir(test_dir))

final_model.eval()
all_test_preds = []
with torch.no_grad():
    for fname in test_files:
        img_path = os.path.join(test_dir, fname)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # --- TTA: Prediksi pada gambar asli ---
        transformed_original = val_transform_alb(image=image)
        img_original = transformed_original['image'].unsqueeze(0).to(device)
        probs_original = torch.softmax(final_model(img_original), dim=1)
        
        # --- TTA: Gambar yang di-flip horizontal ---
        flip_transform = A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.pytorch.ToTensorV2(),
        ])
        transformed_flipped = flip_transform(image=image)
        img_flipped = transformed_flipped['image'].unsqueeze(0).to(device)
        probs_flipped = torch.softmax(final_model(img_flipped), dim=1)
        
        # Rata-ratakan probabilitas
        avg_probs = (probs_original + probs_flipped) / 2
        pred = avg_probs.argmax(1).cpu().item()
        all_test_preds.append(pred)

print("Prediksi dengan TTA selesai.")

# --- Buat dan Simpan File Submisi ---
submission_df = pd.DataFrame({
    'Id': test_files,
    'Predicted': all_test_preds
})
submission_df.to_csv("submission.csv", index=False)

print("\nFile submisi 'submission.csv' berhasil dibuat!")
print(submission_df.head())


# resnet50.a1_in1k
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
import timm
from safetensors.torch import save_file, load_file
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
import os
from PIL import Image
import pandas as pd
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- Seed & Device ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# --- Dataset setup ---
data_dir = "/kaggle/input/srifoton-25-machine-learning-competition"
img_size, batch_size = 384, 8

# Custom Dataset Class untuk Multiple Augmentation
class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, num_augments=3):
        self.original_dataset = original_dataset
        self.num_augments = num_augments
        
        # Transform untuk augmentasi
        self.augment_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-30, 30)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
            ], p=0.3),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        ])
        
        # Transform untuk original (tanpa augmentasi random)
        self.original_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.original_dataset) * (self.num_augments + 1)  # +1 untuk original
    
    def __getitem__(self, idx):
        # Hitung index asli dan augmentation index
        original_idx = idx // (self.num_augments + 1)
        aug_idx = idx % (self.num_augments + 1)
        
        # Ambil original image dan label
        img_path = self.original_dataset.samples[original_idx][0]
        label = self.original_dataset.samples[original_idx][1]
        img = Image.open(img_path).convert("RGB")
        
        # Apply transform sesuai index
        if aug_idx == 0:
            # Original image
            img = self.original_transform(img)
        else:
            # Augmented versions
            img = self.augment_transform(img)
            
        return img, label

# Validation/test transform tanpa augmentasi
val_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# Load original datasets
original_train = datasets.ImageFolder(root=f"{data_dir}/train/train")
original_val = datasets.ImageFolder(root=f"{data_dir}/val/val")

# Buat augmented datasets (1x augmented + 1 original = 2x total)
augmented_train = AugmentedDataset(original_train, num_augments=1)
augmented_val = AugmentedDataset(original_val, num_augments=1)

# gabung train + val eksternal
full_dataset = ConcatDataset([augmented_train, augmented_val])

# ambil target labels (duplikasi sesuai augmentasi)
train_targets = []
for target in original_train.targets:
    train_targets.extend([target] * 2)  # 2x karena 1 augment + 1 original

val_targets = []
for target in original_val.targets:
    val_targets.extend([target] * 2)

targets = np.array(train_targets + val_targets)

num_classes = len(original_train.classes)
print("Jumlah Class:", num_classes)
print("Classes:", original_train.classes)
print("Original samples (train):", len(original_train))
print("Original samples (val):", len(original_val))
print("Augmented samples (train):", len(augmented_train))
print("Augmented samples (val):", len(augmented_val))
print("Total samples (train+val) after augmentation:", len(full_dataset))

# --- Training Final Menggunakan Seluruh Data (Tanpa K-Fold) ---


# 1. Buat DataLoader untuk seluruh dataset Anda (tidak ada perubahan)
full_train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
print(f"Total data untuk training final: {len(full_dataset)}")

# 2. Atur Hyperparameter (tidak ada perubahan)
epochs, lr = 5, 5e-5

# 3. Definisikan Model, Optimizer, dan Scheduler (BAGIAN YANG DIUBAH)
# Ganti nama model menjadi ini
final_model = timm.create_model(
    "tf_efficientnetv2_m_in21ft1k", # <-- Coba gunakan nama ini
    pretrained=True,
    num_classes=num_classes
).to(device)

# --- PERUBAHAN 2: Ganti Optimizer dan Tambah Scheduler ---
optimizer = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=1e-2) # <-- Menggunakan AdamW
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6) # <-- Menambahkan Scheduler

criterion = nn.CrossEntropyLoss()

print("\nMemulai training final dengan model dan optimizer baru...")
final_model.train()

for epoch in range(epochs):
    epoch_loss = 0.0
    # Loop training
    for imgs, labels in full_train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = final_model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    # --- PERUBAHAN 3: Panggil scheduler di akhir epoch ---
    scheduler.step()
    
    avg_loss = epoch_loss / len(full_train_loader)
    print(f"Epoch {epoch+1}/{epochs} selesai | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

print("\nTraining final selesai.")

# --- 4. Prediksi pada Data Test dengan Test-Time Augmentation (TTA) ---
print("\nMemulai prediksi pada data test dengan TTA...")
test_dir = f"{data_dir}/test/test"
test_files = sorted(os.listdir(test_dir))

final_model.eval()
all_test_preds = []
with torch.no_grad():
    for fname in test_files:
        img_path = os.path.join(test_dir, fname)
        img = Image.open(img_path).convert("RGB")
        
        # --- TTA: Prediksi pada gambar asli dan versi flip ---
        
        # 1. Gambar asli
        img_original = val_transform(img).unsqueeze(0).to(device)
        probs_original = torch.softmax(final_model(img_original), dim=1)
        
        # 2. Gambar yang di-flip horizontal
        img_flipped = transforms.functional.hflip(img)
        img_flipped_tensor = val_transform(img_flipped).unsqueeze(0).to(device)
        probs_flipped = torch.softmax(final_model(img_flipped_tensor), dim=1)
        
        # 3. Rata-ratakan probabilitas dari kedua versi
        avg_probs = (probs_original + probs_flipped) / 2
        pred = avg_probs.argmax(1).cpu().item()
        all_test_preds.append(pred)

print("Prediksi dengan TTA selesai.")

# --- 5. Buat dan Simpan File Submisi ---
submission_df = pd.DataFrame({
    'Id': test_files,
    'Predicted': all_test_preds
})
submission_df.to_csv("submission.csv", index=False)

print("\nFile submisi 'submission.csv' berhasil dibuat!")
print(submission_df.head())


# resnet50.a1_in1k - OPTIMIZED VERSION
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
import timm
from safetensors.torch import save_file, load_file
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
import os
from PIL import Image
import pandas as pd
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import torch.nn.functional as F

# --- Seed & Device ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# --- Dataset setup ---
data_dir = "/kaggle/input/srifoton-25-machine-learning-competition"
img_size, batch_size = 320, 12  # Optimized size & batch untuk balance speed/accuracy

# Custom Dataset Class untuk Multiple Augmentation - OPTIMIZED
class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, num_augments=2, is_train=True):
        self.original_dataset = original_dataset
        self.num_augments = num_augments
        self.is_train = is_train
        
        # Transform untuk training - LESS AGGRESSIVE untuk pneumonia classification
        self.train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-15, 15)),  # Reduced from 30
            transforms.RandomAffine(
                degrees=0, 
                translate=(0.05, 0.05),  # Reduced from 0.1
                scale=(0.95, 1.05)       # Reduced from 0.9-1.1
            ),
            transforms.ColorJitter(
                brightness=0.1,    # Reduced from 0.2
                contrast=0.15,     # Reduced from 0.2
                saturation=0.05    # Reduced from 0.1
            ),
            # Remove GaussianBlur - dapat menghilangkan detail penting untuk pneumonia
            transforms.ToTensor(),
            # Normalization untuk medical images
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.05, scale=(0.01, 0.05), ratio=(0.5, 2.0))  # More conservative
        ])
        
        # Transform untuk validation/test
        self.val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        if self.is_train:
            return len(self.original_dataset) * (self.num_augments + 1)
        else:
            return len(self.original_dataset)
    
    def __getitem__(self, idx):
        if self.is_train:
            # Training mode dengan augmentasi
            original_idx = idx // (self.num_augments + 1)
            aug_idx = idx % (self.num_augments + 1)
            
            img_path = self.original_dataset.samples[original_idx][0]
            label = self.original_dataset.samples[original_idx][1]
            img = Image.open(img_path).convert("RGB")
            
            if aug_idx == 0:
                img = self.val_transform(img)  # Original tanpa random aug
            else:
                img = self.train_transform(img)
        else:
            # Validation mode tanpa augmentasi
            img_path = self.original_dataset.samples[idx][0]
            label = self.original_dataset.samples[idx][1]
            img = Image.open(img_path).convert("RGB")
            img = self.val_transform(img)
            
        return img, label

# Load datasets
original_train = datasets.ImageFolder(root=f"{data_dir}/train/train")
original_val = datasets.ImageFolder(root=f"{data_dir}/val/val")

# Buat augmented datasets
augmented_train = AugmentedDataset(original_train, num_augments=2, is_train=True)
val_dataset = AugmentedDataset(original_val, num_augments=0, is_train=False)

# Gabung train + val untuk training
full_train_dataset = ConcatDataset([augmented_train, val_dataset])

num_classes = len(original_train.classes)
print("Jumlah Class:", num_classes)
print("Classes:", original_train.classes)
print("Total training samples:", len(full_train_dataset))

# --- Model Definition dengan Ensemble Internal ---
class OptimizedModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Gunakan model yang lebih ringan tapi tetap powerful
        self.backbone = timm.create_model(
            'efficientnet_b3a',  # Lebih ringan dari efficientnetv2_m
            pretrained=True,
            num_classes=0,  # Remove classifier
        )
        
        # Custom classifier dengan dropout
        feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# --- Training Setup ---
full_train_loader = DataLoader(
    full_train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=4,
    pin_memory=True,
    drop_last=True
)

epochs, base_lr = 8, 3e-4  # Increased epochs, adjusted LR

# Model setup
model = OptimizedModel(num_classes).to(device)

# Optimizer dengan weight decay yang tepat
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=base_lr, 
    weight_decay=0.01,
    betas=(0.9, 0.999)
)

# OneCycleLR scheduler - lebih baik untuk medical images
scheduler = OneCycleLR(
    optimizer,
    max_lr=base_lr * 3,
    total_steps=epochs * len(full_train_loader),
    pct_start=0.1,
    anneal_strategy='cos'
)

# Loss function dengan label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

print(f"\nMemulai training dengan {epochs} epochs...")
model.train()

best_loss = float('inf')
for epoch in range(epochs):
    epoch_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (imgs, labels) in enumerate(full_train_loader):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping untuk stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        epoch_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if batch_idx % 50 == 0:
            print(f'Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(full_train_loader)} | '
                  f'Loss: {loss.item():.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    avg_loss = epoch_loss / len(full_train_loader)
    accuracy = 100. * correct / total
    
    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}% | "
          f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': avg_loss
        }, 'best_model.pth')

print("\nTraining selesai!")

# --- Enhanced Test-Time Augmentation (TTA) ---
print("\nMemulai prediksi dengan Enhanced TTA...")
test_dir = f"{data_dir}/test/test"
test_files = sorted(os.listdir(test_dir))

# Load best model
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# TTA transforms
tta_transforms = [
    transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.functional.hflip,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((int(img_size * 1.1), int(img_size * 1.1))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
]

all_test_preds = []
with torch.no_grad():
    for fname in test_files:
        img_path = os.path.join(test_dir, fname)
        img = Image.open(img_path).convert("RGB")
        
        # Collect predictions from multiple TTA
        tta_probs = []
        for transform in tta_transforms:
            if callable(transform.transforms[1]):  # For hflip
                img_transformed = transform.transforms[1](img)
                img_tensor = transforms.Compose(transform.transforms[2:])(img_transformed)
            else:
                img_tensor = transform(img)
            
            img_tensor = img_tensor.unsqueeze(0).to(device)
            probs = F.softmax(model(img_tensor), dim=1)
            tta_probs.append(probs)
        
        # Average all TTA predictions
        avg_probs = torch.stack(tta_probs).mean(0)
        pred = avg_probs.argmax(1).cpu().item()
        all_test_preds.append(pred)

# Create submission
submission_df = pd.DataFrame({
    'Id': test_files,
    'Predicted': all_test_preds
})
submission_df.to_csv("submission_optimized.csv", index=False)

print("\nFile submisi 'submission_optimized.csv' berhasil dibuat!")
print(f"Total prediksi: {len(all_test_preds)}")
print("Sample prediksi:")
print(submission_df.head())


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
import timm
from safetensors.torch import save_file, load_file
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
import os
from PIL import Image
import pandas as pd
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- Seed & Device ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# --- Dataset setup ---
data_dir = "/kaggle/input/srifoton-25-machine-learning-competition"
img_size, batch_size = 224, 8

# Custom Dataset Class untuk Multiple Augmentation
class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, num_augments=3):
        self.original_dataset = original_dataset
        self.num_augments = num_augments
        
        # Transform untuk augmentasi
        self.augment_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-15, 15)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
            ], p=0.3),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        ])
        
        # Transform untuk original (tanpa augmentasi random)
        self.original_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.original_dataset) * (self.num_augments + 1)  # +1 untuk original
    
    def __getitem__(self, idx):
        # Hitung index asli dan augmentation index
        original_idx = idx // (self.num_augments + 1)
        aug_idx = idx % (self.num_augments + 1)
        
        # Ambil original image dan label
        img_path = self.original_dataset.samples[original_idx][0]
        label = self.original_dataset.samples[original_idx][1]
        img = Image.open(img_path).convert("RGB")
        
        # Apply transform sesuai index
        if aug_idx == 0:
            # Original image
            img = self.original_transform(img)
        else:
            # Augmented versions
            img = self.augment_transform(img)
            
        return img, label

# Validation/test transform tanpa augmentasi
val_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# Load original datasets
original_train = datasets.ImageFolder(root=f"{data_dir}/train/train")
original_val = datasets.ImageFolder(root=f"{data_dir}/val/val")

# Buat augmented datasets (1x augmented + 1 original = 2x total)
augmented_train = AugmentedDataset(original_train, num_augments=1)
augmented_val = AugmentedDataset(original_val, num_augments=1)

# gabung train + val eksternal
full_dataset = ConcatDataset([augmented_train, augmented_val])

# ambil target labels (duplikasi sesuai augmentasi)
train_targets = []
for target in original_train.targets:
    train_targets.extend([target] * 2)  # 2x karena 1 augment + 1 original

val_targets = []
for target in original_val.targets:
    val_targets.extend([target] * 2)

targets = np.array(train_targets + val_targets)

num_classes = len(original_train.classes)
print("Jumlah Class:", num_classes)
print("Classes:", original_train.classes)
print("Original samples (train):", len(original_train))
print("Original samples (val):", len(original_val))
print("Augmented samples (train):", len(augmented_train))
print("Augmented samples (val):", len(augmented_val))
print("Total samples (train+val) after augmentation:", len(full_dataset))

# --- Training Final Menggunakan Seluruh Data (Tanpa K-Fold) ---


# 1. Buat DataLoader untuk seluruh dataset Anda (tidak ada perubahan)
full_train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
print(f"Total data untuk training final: {len(full_dataset)}")

# 2. Atur Hyperparameter (tidak ada perubahan)
epochs, lr = 5, 5e-5

# 3. Definisikan Model, Optimizer, dan Scheduler (BAGIAN YANG DIUBAH)
# Ganti nama model menjadi ini
final_model = timm.create_model(
    "tf_efficientnetv2_s", # <-- Coba gunakan nama ini
    pretrained=True,
    num_classes=num_classes
).to(device)

# --- PERUBAHAN 2: Ganti Optimizer dan Tambah Scheduler ---
optimizer = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=1e-2) # <-- Menggunakan AdamW
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6) # <-- Menambahkan Scheduler

criterion = nn.CrossEntropyLoss()

print("\nMemulai training final dengan model dan optimizer baru...")
final_model.train()

for epoch in range(epochs):
    epoch_loss = 0.0
    # Loop training
    for imgs, labels in full_train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = final_model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    # --- PERUBAHAN 3: Panggil scheduler di akhir epoch ---
    scheduler.step()
    
    avg_loss = epoch_loss / len(full_train_loader)
    print(f"Epoch {epoch+1}/{epochs} selesai | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

print("\nTraining final selesai.")

# --- 4. Prediksi pada Data Test dengan Test-Time Augmentation (TTA) ---
print("\nMemulai prediksi pada data test dengan TTA...")
test_dir = f"{data_dir}/test/test"
test_files = sorted(os.listdir(test_dir))

final_model.eval()
all_test_preds = []
with torch.no_grad():
    for fname in test_files:
        img_path = os.path.join(test_dir, fname)
        img = Image.open(img_path).convert("RGB")
        
        # --- TTA: Prediksi pada gambar asli dan versi flip ---
        
        # 1. Gambar asli
        img_original = val_transform(img).unsqueeze(0).to(device)
        probs_original = torch.softmax(final_model(img_original), dim=1)
        
        # 2. Gambar yang di-flip horizontal
        img_flipped = transforms.functional.hflip(img)
        img_flipped_tensor = val_transform(img_flipped).unsqueeze(0).to(device)
        probs_flipped = torch.softmax(final_model(img_flipped_tensor), dim=1)
        
        # 3. Rata-ratakan probabilitas dari kedua versi
        avg_probs = (probs_original + probs_flipped) / 2
        pred = avg_probs.argmax(1).cpu().item()
        all_test_preds.append(pred)

print("Prediksi dengan TTA selesai.")

# --- 5. Buat dan Simpan File Submisi ---
submission_df = pd.DataFrame({
    'Id': test_files,
    'Predicted': all_test_preds
})
submission_df.to_csv("submission.csv", index=False)

print("\nFile submisi 'submission.csv' berhasil dibuat!")
print(submission_df.head())


import tensorflow as tf
from tensorflow.keras.applications import VGG16, DenseNet121, InceptionV3, ResNet50
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.ensemble import VotingClassifier  # For soft voting ensemble
import lime
import lime.lime_image
from tensorflow.keras import backend as K

# Dataset paths
train_dir = "/kaggle/input/srifoton-25-machine-learning-competition/train/train"
val_dir = "/kaggle/input/srifoton-25-machine-learning-competition/val/val"
test_dir = "/kaggle/input/srifoton-25-machine-learning-competition/test"
img_size = (224, 224)
batch_size = 32
num_classes = 5
class_names = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']

# ========================
# 1. ENHANCED PREPROCESSING
# ========================

class EnhancedPreprocessing:
    @staticmethod
    def enhance_image(image):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        image = cv2.convertScaleAbs(image, beta=10)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            enhanced = clahe.apply(image)
            if len(enhanced.shape) == 2:
                enhanced = np.expand_dims(enhanced, axis=-1)
                enhanced = np.repeat(enhanced, 3, axis=-1)
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), sigmaX=0)
        enhanced = enhanced.astype(np.float32) / 255.0
        return enhanced

def create_enhanced_data_generators():
    train_datagen = ImageDataGenerator(
        preprocessing_function=EnhancedPreprocessing.enhance_image,
        rotation_range=45,
        width_shift_range=0.15,
        height_shift_range=0.15,
        brightness_range=[0.7, 1.3],
        zoom_range=0.2,
        shear_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(
        preprocessing_function=EnhancedPreprocessing.enhance_image
    )
    return train_datagen, test_datagen

# ========================
# 2. MODEL BUILDING (STEP 2.6: DL Methods)
# ========================

class ModelBuilder:
    def __init__(self, num_classes=5):
        self.num_classes = num_classes
        self.models = {}

    def build_vgg16(self, dropout_rate=0.5, learning_rate=1e-4):
        backbone = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        backbone.trainable = False
        model = models.Sequential([
            backbone,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu', kernel_regularizer=l2(0.02)),
            layers.Dropout(dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.models['vgg16'] = model
        return model

    def build_densenet121(self, dropout_rate=0.5, learning_rate=1e-4):
        backbone = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        backbone.trainable = False
        model = models.Sequential([
            backbone,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu', kernel_regularizer=l2(0.02)),
            layers.Dropout(dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.models['densenet121'] = model
        return model

    def build_inceptionv3(self, dropout_rate=0.5, learning_rate=1e-4):
        backbone = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        backbone.trainable = False
        model = models.Sequential([
            backbone,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu', kernel_regularizer=l2(0.02)),
            layers.Dropout(dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.models['inceptionv3'] = model
        return model

    def build_resnet50(self, dropout_rate=0.5, learning_rate=1e-4):
        backbone = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        backbone.trainable = False
        model = models.Sequential([
            backbone,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu', kernel_regularizer=l2(0.02)),
            layers.Dropout(dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.models['resnet50'] = model
        return model

    def build_ensemble(self, models_dict):
        # Simple soft voting ensemble using prediction probabilities
        def ensemble_predict(X):
            predictions = np.zeros((len(X), self.num_classes))
            for name, model in models_dict.items():
                preds = model.predict(X)
                predictions += preds
            return np.argmax(predictions, axis=1)
        
        def ensemble_predict_proba(X):
            predictions = np.zeros((len(X), self.num_classes))
            for name, model in models_dict.items():
                preds = model.predict(X)
                predictions += preds
            predictions /= len(models_dict)
            return predictions
        
        return ensemble_predict, ensemble_predict_proba

# ========================
# 3. TRAINING FUNCTION
# ========================

def train_single_model(model, train_ds, val_ds, epochs=10, callbacks=None):
    if callbacks is None:
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint(f'{model.name}_best.h5', monitor='val_loss', save_best_only=True)
        ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    return history

# ========================
# 4. EVALUATION (STEP 4.8)
# ========================

def evaluate_model(model, val_ds):
    y_true = []
    y_pred = []
    for batch in val_ds:
        images, labels = batch
        preds = model.predict(images)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

# ========================
# 5. EXPLAINABILITY (XAI - LIME & Grad-CAM)
# ========================

def lime_explanation(model, image, num_samples=1000):
    # LIME for image
    predictor = lambda x: model.predict(x / 255.0)  # Assuming image is 0-255
    explainer = lime.lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image, predictor, num_samples=num_samples)
    return explanation

def grad_cam(model, image, layer_name='block5_conv3'):  # Default for VGG16
    # Grad-CAM implementation
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, tf.argmax(predictions[0])]
    
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    
    gate_f = tf.cast(output > 0, 'float32')
    gate_g = tf.cast(grads > 0, 'float32')
    guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads
    
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.nn.relu(output) * weights[:, tf.newaxis, tf.newaxis, tf.newaxis], axis=-1)
    
    # Resize to input shape
    cam = cv2.resize(cam.numpy()[0], (224, 224))
    cam = cam / np.max(cam)
    return cam

# ========================
# 6. MAIN PIPELINE (ALUR SESUAI DIAGRAM)
# ========================

def main():
    print("ğŸš€ Pneumonia Classification Pipeline (STEP 1.5 - 4.8)")
    print("=" * 60)
    print("Classes: Bacterial Pneumonia, COVID, Normal, Tuberculosis, Viral Pneumonia")
    print("=" * 60)

    # STEP 1.5: Load Input Datasets
    print("\nğŸ“‚ STEP 1.5: Loading Input Datasets...")
    train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, image_size=img_size, batch_size=batch_size, shuffle=True)
    val_ds = tf.keras.utils.image_dataset_from_directory(val_dir, image_size=img_size, batch_size=batch_size, shuffle=False)
    test_ds = tf.keras.utils.image_dataset_from_directory(test_dir, image_size=img_size, batch_size=batch_size, shuffle=False)
    print("âœ… Datasets loaded successfully")

    # Apply preprocessing
    train_datagen, test_datagen = create_enhanced_data_generators()

    def apply_datagen(dataset, datagen, is_train=True):
        def gen():
            for images, labels in dataset:
                images = images.numpy()
                labels = labels.numpy().astype(np.int32)
                for i in range(images.shape[0]):
                    if is_train:
                        img = datagen.random_transform(images[i])
                    else:
                        img = datagen.standardize(images[i])
                    yield img, labels[i]
        return tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = apply_datagen(train_ds, train_datagen, is_train=True)
    val_ds = apply_datagen(val_ds, test_datagen, is_train=False)
    test_ds = apply_datagen(test_ds, test_datagen, is_train=False)

    # STEP 2.6: Build DL Methods
    print("\nğŸ”§ STEP 2.6: Building DL Methods...")
    builder = ModelBuilder(num_classes)
    builder.build_vgg16()
    builder.build_densenet121()
    builder.build_inceptionv3()
    builder.build_resnet50()

    # Train individual models
    histories = {}
    for name, model in builder.models.items():
        print(f"\nTraining {name}...")
        model.name = name
        histories[name] = train_single_model(model, train_ds, val_ds, epochs=10)

    # STEP 3.7: Classification & Ensemble
    print("\nğŸ�—ï¸� STEP 3.7: Ensemble Classification...")
    ensemble_predict, ensemble_predict_proba = builder.build_ensemble(builder.models)

    # Evaluate individual and ensemble
    print("\nğŸ“Š STEP 4.8: Evaluation...")
    results = {}
    for name, model in builder.models.items():
        print(f"\nEvaluating {name}:")
        results[name] = evaluate_model(model, val_ds)

    # Ensemble evaluation
    y_true = []
    y_pred_ensemble = []
    for batch in val_ds:
        images, labels = batch
        preds = ensemble_predict_proba(images.numpy())
        y_pred_ensemble.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())
    
    y_true = np.array(y_true)
    y_pred_ensemble = np.array(y_pred_ensemble)
    acc_ens = accuracy_score(y_true, y_pred_ensemble)
    prec_ens, rec_ens, f1_ens, _ = precision_recall_fscore_support(y_true, y_pred_ensemble, average='weighted')
    results['ensemble'] = {'accuracy': acc_ens, 'precision': prec_ens, 'recall': rec_ens, 'f1': f1_ens}
    print(f"\nEnsemble Evaluation:")
    print(f"Accuracy: {acc_ens:.4f}")
    print(f"Precision: {prec_ens:.4f}")
    print(f"Recall: {rec_ens:.4f}")
    print(f"F1-Score: {f1_ens:.4f}")

    # XAI: LIME & Grad-CAM (Example on one sample)
    print("\nğŸ§  XAI Explanations (Sample Image)...")
    sample_batch = next(iter(val_ds))
    sample_image, sample_label = sample_batch[0][0], sample_batch[1][0]
    sample_image = (sample_image.numpy() * 255).astype(np.uint8)  # For LIME

    # LIME on VGG16
    lime_exp = lime_explanation(builder.models['vgg16'], sample_image[0])
    plt.figure()
    plt.imshow(sample_image[0])
    plt.title(f"LIME Explanation for {class_names[sample_label.numpy()]}")
    plt.show()

    # Grad-CAM on VGG16
    cam = grad_cam(builder.models['vgg16'], tf.expand_dims(sample_image[0]/255.0, 0))
    plt.figure()
    plt.imshow(sample_image[0])
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title(f"Grad-CAM for {class_names[np.argmax(builder.models['vgg16'].predict(sample_image[0:1]/255.0))]}")
    plt.show()

    # STEP 4.8: Test Predictions
    print("\nğŸ”® Final Test Predictions (Ensemble)...")
    test_predictions = []
    test_filenames = sorted(os.listdir(test_dir))
    for batch in test_ds:
        images = batch[0]
        preds = ensemble_predict_proba(images.numpy())
        classes = np.argmax(preds, axis=1)
        test_predictions.extend(classes)
    
    pred_df = pd.DataFrame({
        'Id': test_filenames,
        'Predicted': test_predictions
    })
    pred_df.to_csv('predictions.csv', index=False)
    print("âœ… Predictions saved to 'predictions.csv'")

    print("\nğŸ�‰ Pipeline Completed!")

if __name__ == "__main__":
    main()


import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import cv2

# ========================
# 1. ENHANCED PREPROCESSING
# ========================

class EnhancedPreprocessing:
    """
    Optimalisasi preprocessing khusus untuk pneumonia classification
    Focus: pencahayaan, ketajaman, dan kontras untuk membedakan pola bakterial vs viral
    """

    @staticmethod
    def enhance_image(image):
        """
        Enhanced preprocessing untuk pneumonia X-ray
        - CLAHE untuk kontras adaptif
        - Gaussian blur removal untuk ketajaman
        - Histogram equalization untuk pencahayaan optimal
        """
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        if len(image.shape) == 3:
            # RGB image
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale
            enhanced = clahe.apply(image)
            if len(enhanced.shape) == 2:
                enhanced = np.expand_dims(enhanced, axis=-1)
                enhanced = np.repeat(enhanced, 3, axis=-1)

        # Sharpening kernel untuk mempertegas batas antara area normal dan infected
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)

        # Normalize back to [0,1]
        enhanced = enhanced.astype(np.float32) / 255.0

        return enhanced

def create_enhanced_data_generators():
    """
    Data augmentation yang dioptimalkan untuk pneumonia
    Focus: variasi pencahayaan dan rotasi kecil (sesuai kondisi X-ray)
    """
    train_datagen = ImageDataGenerator(
        preprocessing_function=EnhancedPreprocessing.enhance_image,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2],
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest',
        validation_split=0.2
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=EnhancedPreprocessing.enhance_image,
        validation_split=0.2
    )

    return train_datagen, val_datagen

# ========================
# 2. MODEL ARCHITECTURE
# ========================

class PneumoniaClassificationSystem:
    def __init__(self, num_classes=5):
        self.num_classes = num_classes
        self.class_names = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']
        self.model = None

    def build_model(self, dropout_rate=0.3, optimizer='adam', learning_rate=1e-4):
        """
        Model dengan transfer learning menggunakan VGG16
        Classes: [0]Bacterial, [1]COVID, [2]Normal, [3]TB, [4]Viral
        """
        # Load VGG16 dengan weights pre-trained dari ImageNet, tanpa top layers
        backbone = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Freeze backbone initially
        backbone.trainable = False

        # Build model
        self.model = models.Sequential([
            backbone,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        # Configure optimizer
        if optimizer.lower() == 'adam':
            opt = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'rmsprop':
            opt = optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            opt = optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(
            optimizer=opt,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return self.model

def train_model(train_ds, val_ds, dropout_rate=0.3, optimizer='adam', learning_rate=1e-4):
    """
    Training pipeline untuk model VGG16
    """
    print("ğŸ”¥ Building Pneumonia Classification System with VGG16...")
    print(f"Classes: {['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']}")

    # Initialize system
    system = PneumoniaClassificationSystem(num_classes=5)

    # Build model
    model = system.build_model(dropout_rate=dropout_rate, optimizer=optimizer, learning_rate=learning_rate)
    print(f"Model built with {model.count_params():,} parameters")

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1),
        ModelCheckpoint('vgg16_pneumonia_best.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, verbose=1, min_lr=1e-7)
    ]

    # Phase 1: Training with frozen backbone
    print("\nğŸ”’ Phase 1: Training with frozen backbone...")
    history_1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=callbacks,
        verbose=1
    )

    # Phase 2: Fine-tuning
    print("\nğŸ”“ Phase 2: Fine-tuning model...")
    model.layers[0].trainable = True

    # Unfreeze last 4 blocks of VGG16 (careful fine-tuning)
    for layer in model.layers[0].layers[:-5]:
        layer.trainable = False

    trainable_count = sum([1 for layer in model.layers[0].layers if layer.trainable])
    print(f"Unfrozen layers: {trainable_count}/total layers")

    # Recompile with lower learning rate
    if optimizer.lower() == 'adam':
        opt = optimizers.Adam(learning_rate=learning_rate/10)
    elif optimizer.lower() == 'rmsprop':
        opt = optimizers.RMSprop(learning_rate=learning_rate/10)
    elif optimizer.lower() == 'sgd':
        opt = optimizers.SGD(learning_rate=learning_rate/10, momentum=0.9)
    else:
        opt = optimizers.Adam(learning_rate=learning_rate/10)

    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history_2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=callbacks,
        verbose=1
    )

    return system, model, (history_1, history_2)

# ========================
# 3. PREDICTION
# ========================

def predict_test_dataset(model, test_ds, confidence_threshold=0.7):
    """
    Predict pada test dataset dengan comprehensive output
    """
    class_names = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']
    
    print("ğŸ”® PREDICTING TEST DATASET...")
    print(f"Classes: {class_names}")
    print(f"Confidence threshold: {confidence_threshold}")
    
    # Collect all predictions
    all_predictions = []
    all_confidences = []
    all_images = []
    batch_count = 0
    
    print("\nğŸ“Š Processing test batches...")
    
    for batch in test_ds:
        images = batch[0]  # Robust handling: assume first element is images
        batch_count += 1
        print(f"Processing batch {batch_count}...", end=' ')
        
        # Model predictions
        preds = model.predict(images, verbose=0)
        classes = np.argmax(preds, axis=1)
        confidences = np.max(preds, axis=1)
        
        batch_results = []
        
        for i in range(len(images)):
            img = images[i].numpy()
            pred = classes[i]
            conf = confidences[i]
            probs = preds[i]
            
            result = {
                'image_index': len(all_predictions) + i,
                'prediction': pred,
                'class_name': class_names[pred],
                'confidence': conf,
                'probabilities': probs,
                'prediction_method': 'VGG16'
            }
            
            batch_results.append(result)
        
        all_predictions.extend(batch_results)
        all_images.extend(images.numpy())
        print(f"âœ… {len(batch_results)} predictions")
    
    print(f"\nğŸ�¯ PREDICTION SUMMARY:")
    print(f"Total images processed: {len(all_predictions)}")
    
    # Count predictions by class
    class_counts = {}
    confidence_stats = {'high': 0, 'medium': 0, 'low': 0}
    
    for pred in all_predictions:
        # Count by final class
        class_name = pred['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Count by confidence level
        conf = pred['confidence']
        if conf >= 0.8:
            confidence_stats['high'] += 1
        elif conf >= 0.6:
            confidence_stats['medium'] += 1
        else:
            confidence_stats['low'] += 1
    
    print(f"\nğŸ�·ï¸� PREDICTED CLASS DISTRIBUTION:")
    for class_name, count in class_counts.items():
        print(f"   {class_name}: {count} ({count/len(all_predictions)*100:.1f}%)")
    
    print(f"\nğŸ�¯ CONFIDENCE DISTRIBUTION:")
    print(f"   High (â‰¥0.8): {confidence_stats['high']} ({confidence_stats['high']/len(all_predictions)*100:.1f}%)")
    print(f"   Medium (0.6-0.8): {confidence_stats['medium']} ({confidence_stats['medium']/len(all_predictions)*100:.1f}%)")
    print(f"   Low (<0.6): {confidence_stats['low']} ({confidence_stats['low']/len(all_predictions)*100:.1f}%)")
    
    # Calculate average confidences
    all_confidences = [pred['confidence'] for pred in all_predictions]
    print(f"\nğŸ“Š CONFIDENCE STATISTICS:")
    print(f"   Mean: {np.mean(all_confidences):.3f}")
    print(f"   Std: {np.std(all_confidences):.3f}")
    print(f"   Min: {np.min(all_confidences):.3f}")
    print(f"   Max: {np.max(all_confidences):.3f}")
    
    return all_predictions, np.array(all_images)

# ========================
# 4. MAIN EXECUTION
# ========================

def main():
    """
    Main execution pipeline untuk pneumonia classification dengan VGG16
    """
    print("ğŸš€ Pneumonia Classification System with VGG16")
    print("=" * 60)
    print("Classes Mapping:")
    print("  [0] Bacterial Pneumonia")
    print("  [1] Corona Virus Disease (COVID-19)")
    print("  [2] Normal")
    print("  [3] Tuberculosis")
    print("  [4] Viral Pneumonia")
    print("=" * 60)

    # Setup GPU dan Mixed Precision
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("âœ… GPU detected:", gpus)
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("âœ… GPU memory growth enabled")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("âš ï¸� No GPU detected. Training akan menggunakan CPU (slower)")

    # Enable mixed precision untuk GPU optimization
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("âœ… Mixed precision enabled for faster training")

    print("\nğŸ”§ Key System Enhancements:")
    print("1. ğŸ“¸ CLAHE + Sharpening preprocessing untuk X-ray contrast optimization")
    print("2. âš¡ Transfer Learning dengan VGG16")
    print("3. ğŸ�—ï¸� Single-stage architecture dengan fine-tuning")

    print("\n" + "="*60)
    print("ğŸ“‹ USAGE INSTRUCTIONS:")
    print("="*60)
    print("1. Load your dataset dengan format yang sesuai:")
    print("   train_ds = tf.keras.preprocessing.image_dataset_from_directory(...)")
    print("   val_ds = tf.keras.preprocessing.image_dataset_from_directory(...)")
    print("   test_ds = tf.keras.preprocessing.image_dataset_from_directory(...)")
    print("")
    print("2. Training dengan custom parameters:")
    print("   system, model, histories = train_model(train_ds, val_ds, dropout_rate=0.3, optimizer='adam', learning_rate=1e-4)")
    print("")
    print("3. Predict on test dataset:")
    print("   predictions, test_images = predict_test_dataset(model, test_ds, confidence_threshold=0.7)")

    print("\nâœ… System ready untuk deployment!")

    # Apply enhanced preprocessing
    train_datagen, val_datagen = create_enhanced_data_generators()

    # Train the system with custom configuration
    system, model, histories = train_model(train_ds, val_ds, dropout_rate=0.3, optimizer='adam', learning_rate=1e-4)

    # Predict on test dataset
    predictions, test_images = predict_test_dataset(model, test_ds, confidence_threshold=0.7)

    print("ğŸ�‰ Prediction selesai!")

if __name__ == "__main__":
    main()


import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import pandas as pd  # NEW: Added for CSV output

# ========================
# 1. ENHANCED PREPROCESSING
# ========================

class EnhancedPreprocessing:
    """
    Optimalisasi preprocessing khusus untuk pneumonia classification
    Focus: pencahayaan, ketajaman, dan kontras untuk membedakan pola bakterial vs viral
    """
    @staticmethod
    def enhance_image(image):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            enhanced = clahe.apply(image)
            if len(enhanced.shape) == 2:
                enhanced = np.expand_dims(enhanced, axis=-1)
                enhanced = np.repeat(enhanced, 3, axis=-1)

        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        enhanced = enhanced.astype(np.float32) / 255.0
        return enhanced

def create_enhanced_data_generators():
    """
    Data augmentation yang dioptimalkan untuk pneumonia
    """
    train_datagen = ImageDataGenerator(
        preprocessing_function=EnhancedPreprocessing.enhance_image,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        brightness_range=[0.7, 1.3],
        zoom_range=0.2,
        shear_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=EnhancedPreprocessing.enhance_image,
        validation_split=0.2
    )

    return train_datagen, val_datagen

# ========================
# 2. MODEL ARCHITECTURE
# ========================

class PneumoniaClassificationSystem:
    def __init__(self, num_classes=5):
        self.num_classes = num_classes
        self.class_names = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']
        self.model = None

    def build_model(self, dropout_rate=0.4, optimizer='adam', learning_rate=1e-4):
        """
        Model dengan transfer learning menggunakan VGG16
        """
        backbone = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        backbone.trainable = False

        self.model = models.Sequential([
            backbone,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
            layers.Dropout(dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        if optimizer.lower() == 'adam':
            opt = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'rmsprop':
            opt = optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            opt = optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(
            optimizer=opt,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return self.model

def train_model(train_ds, val_ds, dropout_rate=0.4, optimizer='adam', learning_rate=1e-4):
    """
    Training pipeline untuk model VGG16
    """
    print("ğŸ”¥ Building Pneumonia Classification System with VGG16...")
    print(f"Classes: {['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']}")

    system = PneumoniaClassificationSystem(num_classes=5)
    model = system.build_model(dropout_rate=dropout_rate, optimizer=optimizer, learning_rate=learning_rate)
    print(f"Model built with {model.count_params():,} parameters")

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1),
        ModelCheckpoint('vgg16_pneumonia_best.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, verbose=1, min_lr=1e-7)
    ]

    print("\nğŸ”’ Phase 1: Training with frozen backbone...")
    history_1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=callbacks,
        verbose=1
    )

    def plot_history(history, phase):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'b', label='Training accuracy')
        plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
        plt.title(f'{phase} - Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title(f'{phase} - Loss')
        plt.legend()
        plt.show()

    plot_history(history_1, "Phase 1")

    print("\nğŸ”“ Phase 2: Fine-tuning model...")
    model.layers[0].trainable = True
    for layer in model.layers[0].layers[:-5]:
        layer.trainable = False

    trainable_count = sum([1 for layer in model.layers[0].layers if layer.trainable])
    print(f"Unfrozen layers: {trainable_count}/total layers")

    if optimizer.lower() == 'adam':
        opt = optimizers.Adam(learning_rate=learning_rate/10)
    elif optimizer.lower() == 'rmsprop':
        opt = optimizers.RMSprop(learning_rate=learning_rate/10)
    elif optimizer.lower() == 'sgd':
        opt = optimizers.SGD(learning_rate=learning_rate/10, momentum=0.9)
    else:
        opt = optimizers.Adam(learning_rate=learning_rate/10)

    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history_2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=callbacks,
        verbose=1
    )

    plot_history(history_2, "Phase 2")

    return system, model, (history_1, history_2)

# ========================
# 3. PREDICTION
# ========================

def predict_test_dataset(model, test_ds, confidence_threshold=0.7):
    """
    Predict pada test dataset dan simpan ke CSV dengan kolom Id dan Predicted
    MODIFIED: Added CSV output with image names and predicted class indices
    """
    class_names = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']
    
    print("ğŸ”® PREDICTING TEST DATASET...")
    print(f"Classes: {class_names}")
    print(f"Confidence threshold: {confidence_threshold}")
    
    all_predictions = []
    all_confidences = []
    all_images = []
    all_filenames = []
    batch_count = 0
    
    print("\nğŸ“Š Processing test batches...")
    
    # Assume test_ds has file_paths attribute (from image_dataset_from_directory)
    try:
        file_paths = test_ds.file_paths
    except AttributeError:
        print("âš ï¸� Warning: test_ds does not have file_paths. Using index-based IDs.")
        file_paths = [f"image_{i}" for i in range(1000)]  # Fallback for unknown dataset size
    
    file_index = 0
    
    for batch in test_ds:
        images = batch[0]
        batch_count += 1
        print(f"Processing batch {batch_count}...", end=' ')
        
        preds = model.predict(images, verbose=0)
        classes = np.argmax(preds, axis=1)
        confidences = np.max(preds, axis=1)
        
        batch_results = []
        batch_filenames = []
        
        for i in range(len(images)):
            img = images[i].numpy()
            pred = classes[i]
            conf = confidences[i]
            probs = preds[i]
            
            # Get filename or fallback to index-based ID
            if file_index < len(file_paths):
                filename = file_paths[file_index]
                filename = filename.split('/')[-1]  # Extract just the filename
            else:
                filename = f"image_{file_index}"
            
            result = {
                'image_index': len(all_predictions) + i,
                'prediction': pred,
                'class_name': class_names[pred],
                'confidence': conf,
                'probabilities': probs,
                'prediction_method': 'VGG16',
                'filename': filename
            }
            
            batch_results.append(result)
            batch_filenames.append(filename)
            file_index += 1
        
        all_predictions.extend(batch_results)
        all_images.extend(images.numpy())
        all_filenames.extend(batch_filenames)
        print(f"âœ… {len(batch_results)} predictions")
    
    print(f"\nğŸ�¯ PREDICTION SUMMARY:")
    print(f"Total images processed: {len(all_predictions)}")
    
    class_counts = {}
    confidence_stats = {'high': 0, 'medium': 0, 'low': 0}
    
    for pred in all_predictions:
        class_name = pred['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        conf = pred['confidence']
        if conf >= 0.8:
            confidence_stats['high'] += 1
        elif conf >= 0.6:
            confidence_stats['medium'] += 1
        else:
            confidence_stats['low'] += 1
    
    print(f"\nğŸ�·ï¸� PREDICTED CLASS DISTRIBUTION:")
    for class_name, count in class_counts.items():
        print(f"   {class_name}: {count} ({count/len(all_predictions)*100:.1f}%)")
    
    print(f"\nğŸ�¯ CONFIDENCE DISTRIBUTION:")
    print(f"   High (â‰¥0.8): {confidence_stats['high']} ({confidence_stats['high']/len(all_predictions)*100:.1f}%)")
    print(f"   Medium (0.6-0.8): {confidence_stats['medium']} ({confidence_stats['medium']/len(all_predictions)*100:.1f}%)")
    print(f"   Low (<0.6): {confidence_stats['low']} ({confidence_stats['low']/len(all_predictions)*100:.1f}%)")
    
    all_confidences = [pred['confidence'] for pred in all_predictions]
    print(f"\nğŸ“Š CONFIDENCE STATISTICS:")
    print(f"   Mean: {np.mean(all_confidences):.3f}")
    print(f"   Std: {np.std(all_confidences):.3f}")
    print(f"   Min: {np.min(all_confidences):.3f}")
    print(f"   Max: {np.max(all_confidences):.3f}")
    
    # NEW: Save predictions to CSV
    print("\nğŸ’¾ Saving predictions to 'predictions.csv'...")
    pred_df = pd.DataFrame({
        'Id': [pred['filename'] for pred in all_predictions],
        'Predicted': [pred['prediction'] for pred in all_predictions]
    })
    pred_df.to_csv('predictions.csv', index=False)
    print("âœ… Predictions saved to 'predictions.csv'")
    
    return all_predictions, np.array(all_images)

# ========================
# 4. MAIN EXECUTION
# ========================

def main():
    """
    Main execution pipeline untuk pneumonia classification dengan VGG16
    """
    print("ğŸš€ Pneumonia Classification System with VGG16")
    print("=" * 60)
    print("Classes Mapping:")
    print("  [0] Bacterial Pneumonia")
    print("  [1] Corona Virus Disease (COVID-19)")
    print("  [2] Normal")
    print("  [3] Tuberculosis")
    print("  [4] Viral Pneumonia")
    print("=" * 60)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("âœ… GPU detected:", gpus)
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("âœ… GPU memory growth enabled")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("âš ï¸� No GPU detected. Training akan menggunakan CPU (slower)")

    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("âœ… Mixed precision enabled for faster training")

    print("\nğŸ”§ Key System Enhancements:")
    print("1. ğŸ“¸ CLAHE + Sharpening preprocessing untuk X-ray contrast optimization")
    print("2. âš¡ Transfer Learning dengan VGG16")
    print("3. ğŸ�—ï¸� Single-stage architecture dengan fine-tuning")
    print("4. ğŸ›¡ï¸� Anti-overfitting: Enhanced augmentation, L2 regularization, increased dropout")

    print("\n" + "="*60)
    print("ğŸ“‹ USAGE INSTRUCTIONS:")
    print("1. Load your dataset dengan format yang sesuai:")
    print("   train_ds = tf.keras.preprocessing.image_dataset_from_directory(...)")
    print("   val_ds = tf.keras.preprocessing.image_dataset_from_directory(...)")
    print("   test_ds = tf.keras.preprocessing.image_dataset_from_directory(...)")
    print("")
    print("2. Training dengan custom parameters:")
    print("   system, model, histories = train_model(train_ds, val_ds, dropout_rate=0.4, optimizer='adam', learning_rate=1e-4)")
    print("")
    print("3. Predict on test dataset:")
    print("   predictions, test_images = predict_test_dataset(model, test_ds, confidence_threshold=0.7)")

    print("\nâœ… System ready untuk deployment!")

    train_datagen, val_datagen = create_enhanced_data_generators()
    system, model, histories = train_model(train_ds, val_ds, dropout_rate=0.8, optimizer='adamW', learning_rate=1e-4)
    predictions, test_images = predict_test_dataset(model, test_ds, confidence_threshold=0.7)

    print("ğŸ�‰ Prediction selesai!")

if __name__ == "__main__":
    main()


import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import pandas as pd

# ========================
# 1. ENHANCED PREPROCESSING
# ========================

class EnhancedPreprocessing:
    """
    Optimalisasi preprocessing khusus untuk pneumonia classification
    Focus: pencahayaan, ketajaman, dan kontras untuk membedakan pola bakterial vs viral
    """
    @staticmethod
    def enhance_image(image):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            enhanced = clahe.apply(image)
            if len(enhanced.shape) == 2:
                enhanced = np.expand_dims(enhanced, axis=-1)
                enhanced = np.repeat(enhanced, 3, axis=-1)

        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        enhanced = enhanced.astype(np.float32) / 255.0
        return enhanced

def create_enhanced_data_generators():
    """
    Data augmentation yang dioptimalkan untuk pneumonia
    """
    train_datagen = ImageDataGenerator(
        preprocessing_function=EnhancedPreprocessing.enhance_image,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        brightness_range=[0.7, 1.3],
        zoom_range=0.2,
        shear_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=EnhancedPreprocessing.enhance_image,
        validation_split=0.2
    )

    return train_datagen, val_datagen

# ========================
# 2. MODEL ARCHITECTURE
# ========================

class PneumoniaClassificationSystem:
    def __init__(self, num_classes=5):
        self.num_classes = num_classes
        self.class_names = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']
        self.model = None

    def build_model(self, dropout_rate=0.5, optimizer='adam', learning_rate=1e-4):
        """
        Model dengan transfer learning menggunakan DenseNet121
        MODIFIED: Removed AdamW, using Adam optimizer
        """
        backbone = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        backbone.trainable = False

        self.model = models.Sequential([
            backbone,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
            layers.Dropout(dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        if optimizer.lower() == 'adam':
            opt = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'rmsprop':
            opt = optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            opt = optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(
            optimizer=opt,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return self.model

def train_model(train_ds, val_ds, dropout_rate=0.5, optimizer='adam', learning_rate=1e-4):
    """
    Training pipeline untuk model DenseNet121
    MODIFIED: Removed AdamW, using Adam optimizer
    """
    print("ğŸ”¥ Building Pneumonia Classification System with DenseNet121...")
    print(f"Classes: {['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']}")

    system = PneumoniaClassificationSystem(num_classes=5)
    model = system.build_model(dropout_rate=dropout_rate, optimizer=optimizer, learning_rate=learning_rate)
    print(f"Model built with {model.count_params():,} parameters")

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1),
        ModelCheckpoint('densenet_pneumonia_best.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, verbose=1, min_lr=1e-7)
    ]

    print("\nğŸ”’ Phase 1: Training with frozen backbone...")
    history_1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=callbacks,
        verbose=1
    )

    def plot_history(history, phase):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'b', label='Training accuracy')
        plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
        plt.title(f'{phase} - Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title(f'{phase} - Loss')
        plt.legend()
        plt.show()

    plot_history(history_1, "Phase 1")

    print("\nğŸ”“ Phase 2: Fine-tuning model...")
    model.layers[0].trainable = True
    for layer in model.layers[0].layers[:-20]:
        layer.trainable = False

    trainable_count = sum([1 for layer in model.layers[0].layers if layer.trainable])
    print(f"Unfrozen layers: {trainable_count}/total layers")

    if optimizer.lower() == 'adam':
        opt = optimizers.Adam(learning_rate=learning_rate/10)
    elif optimizer.lower() == 'rmsprop':
        opt = optimizers.RMSprop(learning_rate=learning_rate/10)
    elif optimizer.lower() == 'sgd':
        opt = optimizers.SGD(learning_rate=learning_rate/10, momentum=0.9)
    else:
        opt = optimizers.Adam(learning_rate=learning_rate/10)

    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history_2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=callbacks,
        verbose=1
    )

    plot_history(history_2, "Phase 2")

    return system, model, (history_1, history_2)

# ========================
# 3. PREDICTION
# ========================

def predict_test_dataset(model, test_ds, confidence_threshold=0.7):
    """
    Predict pada test dataset dan simpan ke CSV dengan kolom Id dan Predicted
    """
    class_names = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']
    
    print("ğŸ”® PREDICTING TEST DATASET...")
    print(f"Classes: {class_names}")
    print(f"Confidence threshold: {confidence_threshold}")
    
    all_predictions = []
    all_confidences = []
    all_images = []
    all_filenames = []
    batch_count = 0
    
    print("\nğŸ“Š Processing test batches...")
    
    try:
        file_paths = test_ds.file_paths
    except AttributeError:
        print("âš ï¸� Warning: test_ds does not have file_paths. Using index-based IDs.")
        file_paths = [f"image_{i}" for i in range(1000)]
    
    file_index = 0
    
    for batch in test_ds:
        images = batch[0]
        batch_count += 1
        print(f"Processing batch {batch_count}...", end=' ')
        
        preds = model.predict(images, verbose=0)
        classes = np.argmax(preds, axis=1)
        confidences = np.max(preds, axis=1)
        
        batch_results = []
        batch_filenames = []
        
        for i in range(len(images)):
            img = images[i].numpy()
            pred = classes[i]
            conf = confidences[i]
            probs = preds[i]
            
            if file_index < len(file_paths):
                filename = file_paths[file_index]
                filename = filename.split('/')[-1]
            else:
                filename = f"image_{file_index}"
            
            result = {
                'image_index': len(all_predictions) + i,
                'prediction': pred,
                'class_name': class_names[pred],
                'confidence': conf,
                'probabilities': probs,
                'prediction_method': 'DenseNet121',
                'filename': filename
            }
            
            batch_results.append(result)
            batch_filenames.append(filename)
            file_index += 1
        
        all_predictions.extend(batch_results)
        all_images.extend(images.numpy())
        all_filenames.extend(batch_filenames)
        print(f"âœ… {len(batch_results)} predictions")
    
    print(f"\nğŸ�¯ PREDICTION SUMMARY:")
    print(f"Total images processed: {len(all_predictions)}")
    
    class_counts = {}
    confidence_stats = {'high': 0, 'medium': 0, 'low': 0}
    
    for pred in all_predictions:
        class_name = pred['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        conf = pred['confidence']
        if conf >= 0.8:
            confidence_stats['high'] += 1
        elif conf >= 0.6:
            confidence_stats['medium'] += 1
        else:
            confidence_stats['low'] += 1
    
    print(f"\nğŸ�·ï¸� PREDICTED CLASS DISTRIBUTION:")
    for class_name, count in class_counts.items():
        print(f"   {class_name}: {count} ({count/len(all_predictions)*100:.1f}%)")
    
    print(f"\nğŸ�¯ CONFIDENCE DISTRIBUTION:")
    print(f"   High (â‰¥0.8): {confidence_stats['high']} ({confidence_stats['high']/len(all_predictions)*100:.1f}%)")
    print(f"   Medium (0.6-0.8): {confidence_stats['medium']} ({confidence_stats['medium']/len(all_predictions)*100:.1f}%)")
    print(f"   Low (<0.6): {confidence_stats['low']} ({confidence_stats['low']/len(all_predictions)*100:.1f}%)")
    
    all_confidences = [pred['confidence'] for pred in all_predictions]
    print(f"\nğŸ“Š CONFIDENCE STATISTICS:")
    print(f"   Mean: {np.mean(all_confidences):.3f}")
    print(f"   Std: {np.std(all_confidences):.3f}")
    print(f"   Min: {np.min(all_confidences):.3f}")
    print(f"   Max: {np.max(all_confidences):.3f}")
    
    print("\nğŸ’¾ Saving predictions to 'predictions.csv'...")
    pred_df = pd.DataFrame({
        'Id': [pred['filename'] for pred in all_predictions],
        'Predicted': [pred['prediction'] for pred in all_predictions]
    })
    pred_df.to_csv('predictions.csv', index=False)
    print("âœ… Predictions saved to 'predictions.csv'")
    
    return all_predictions, np.array(all_images)

# ========================
# 4. MAIN EXECUTION
# ========================

def main():
    """
    Main execution pipeline untuk pneumonia classification dengan DenseNet121
    MODIFIED: Changed optimizer to 'adam' due to tensorflow_addons incompatibility
    """
    print("ğŸš€ Pneumonia Classification System with DenseNet121")
    print("=" * 60)
    print("Classes Mapping:")
    print("  [0] Bacterial Pneumonia")
    print("  [1] Corona Virus Disease (COVID-19)")
    print("  [2] Normal")
    print("  [3] Tuberculosis")
    print("  [4] Viral Pneumonia")
    print("=" * 60)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("âœ… GPU detected:", gpus)
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("âœ… GPU memory growth enabled")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("âš ï¸� No GPU detected. Training akan menggunakan CPU (slower)")

    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("âœ… Mixed precision enabled for faster training")

    print("\nğŸ”§ Key System Enhancements:")
    print("1. ğŸ“¸ CLAHE + Sharpening preprocessing untuk X-ray contrast optimization")
    print("2. âš¡ Transfer Learning dengan DenseNet121")
    print("3. ğŸ�—ï¸� Single-stage architecture dengan fine-tuning")
    print("4. ğŸ›¡ï¸� Anti-overfitting: Enhanced augmentation, L2 regularization, increased dropout")

    print("\n" + "="*60)
    print("ğŸ“‹ USAGE INSTRUCTIONS:")
    print("1. Load your dataset dengan format yang sesuai:")
    print("   train_ds = tf.keras.preprocessing.image_dataset_from_directory(...)")
    print("   val_ds = tf.keras.preprocessing.image_dataset_from_directory(...)")
    print("   test_ds = tf.keras.preprocessing.image_dataset_from_directory(...)")
    print("")
    print("2. Training dengan custom parameters:")
    print("   system, model, histories = train_model(train_ds, val_ds, dropout_rate=0.5, optimizer='adam', learning_rate=1e-4)")
    print("")
    print("3. Predict on test dataset:")
    print("   predictions, test_images = predict_test_dataset(model, test_ds, confidence_threshold=0.7)")

    print("\nâœ… System ready untuk deployment!")

    train_datagen, val_datagen = create_enhanced_data_generators()
    system, model, histories = train_model(train_ds, val_ds, dropout_rate=0.5, optimizer='adam', learning_rate=1e-4)
    predictions, test_images = predict_test_dataset(model, test_ds, confidence_threshold=0.7)

    print("ğŸ�‰ Prediction selesai!")

if __name__ == "__main__":
    main()


import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os

# ========================
# 1. ENHANCED PREPROCESSING
# ========================

class EnhancedPreprocessing:
    """
    Optimalisasi preprocessing khusus untuk pneumonia classification
    Focus: kontras ringan menggunakan CLAHE, brightness adjustment, dan Gaussian blur
    """
    @staticmethod
    def enhance_image(image):
        """
        Preprocessing untuk pneumonia X-ray
        - Brightness adjustment untuk pencahayaan
        - CLAHE ringan untuk kontras adaptif
        - Gaussian blur untuk mengurangi noise
        """
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Brightness adjustment
        image = cv2.convertScaleAbs(image, beta=10)

        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            enhanced = clahe.apply(image)
            if len(enhanced.shape) == 2:
                enhanced = np.expand_dims(enhanced, axis=-1)
                enhanced = np.repeat(enhanced, 3, axis=-1)

        # Gaussian blur
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), sigmaX=0)

        # Normalize back to [0,1]
        enhanced = enhanced.astype(np.float32) / 255.0
        return enhanced

def create_enhanced_data_generators():
    """
    Data augmentation yang dioptimalkan untuk pneumonia
    """
    train_datagen = ImageDataGenerator(
        preprocessing_function=EnhancedPreprocessing.enhance_image,
        rotation_range=45,
        width_shift_range=0.15,
        height_shift_range=0.15,
        brightness_range=[0.7, 1.3],
        zoom_range=0.2,
        shear_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=EnhancedPreprocessing.enhance_image
    )

    return train_datagen, test_datagen

# ========================
# 2. MODEL ARCHITECTURE
# ========================

class PneumoniaClassificationSystem:
    def __init__(self, num_classes=5):
        self.num_classes = num_classes
        self.class_names = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']
        self.model = None

    def build_model(self, dropout_rate=0.5, optimizer='adam', learning_rate=1e-4):
        """
        Model dengan transfer learning menggunakan DenseNet121
        """
        backbone = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        backbone.trainable = False

        self.model = models.Sequential([
            backbone,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
            layers.Dropout(dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        if optimizer.lower() == 'adam':
            opt = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'rmsprop':
            opt = optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            opt = optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(
            optimizer=opt,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return self.model

def clean_low_confidence_samples(model, dataset, confidence_threshold=0.6):
    """
    Filter out training samples with low confidence predictions
    """
    print("\nğŸ§¹ Cleaning low-confidence samples...")
    images, labels, indices = [], [], []
    for batch_idx, batch in enumerate(dataset.unbatch().batch(32)):
        batch_images, batch_labels = batch
        images.append(batch_images.numpy())
        labels.append(batch_labels.numpy())
        indices.extend([batch_idx * 32 + i for i in range(batch_images.shape[0])])

    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    indices = np.array(indices)

    # Predict probabilities
    probs = model.predict(images, verbose=1)
    confidences = np.max(probs, axis=1)

    # Keep samples with confidence >= threshold
    keep_mask = confidences >= confidence_threshold
    filtered_images = images[keep_mask]
    filtered_labels = labels[keep_mask]

    print(f"ğŸ“Š CleanLab Summary:")
    print(f"   Total samples: {len(images)}")
    print(f"   Kept samples: {len(filtered_images)} ({len(filtered_images)/len(images)*100:.1f}%)")
    print(f"   Dropped samples: {len(images) - len(filtered_images)}")

    # Create new dataset
    filtered_dataset = tf.data.Dataset.from_tensor_slices((filtered_images, filtered_labels))
    filtered_dataset = filtered_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    return filtered_dataset

def train_model(combined_ds, dropout_rate=0.5, optimizer='adam', learning_rate=1e-4):
    """
    Training pipeline untuk model DenseNet121
    """
    print("ğŸ”¥ Building Pneumonia Classification System with DenseNet121...")
    print(f"Classes: {['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']}")

    system = PneumoniaClassificationSystem(num_classes=5)
    model = system.build_model(dropout_rate=dropout_rate, optimizer=optimizer, learning_rate=learning_rate)
    print(f"Model built with {model.count_params():,} parameters")

    callbacks = [
        EarlyStopping(monitor='loss', patience=7, restore_best_weights=True, verbose=1),
        ModelCheckpoint('densenet_pneumonia_best.h5', monitor='loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='loss', factor=0.2, patience=4, verbose=1, min_lr=1e-7)
    ]

    print("\nğŸ”’ Phase 1: Initial training with frozen backbone...")
    history_1 = model.fit(
        combined_ds,
        epochs=15,
        callbacks=callbacks,
        verbose=1
    )

    def plot_history(history, phase):
        acc = history.history['accuracy']
        loss = history.history['loss']
        epochs = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'b', label='Training accuracy')
        plt.title(f'{phase} - Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.title(f'{phase} - Loss')
        plt.legend()
        plt.show()

    plot_history(history_1, "Phase 1")

    print("\nğŸ§¹ Applying CleanLab to filter low-confidence samples...")
    cleaned_train_ds = clean_low_confidence_samples(model, combined_ds)

    print("\nğŸ”’ Phase 2: Training with cleaned dataset...")
    history_2 = model.fit(
        cleaned_train_ds,
        epochs=15,
        callbacks=callbacks,
        verbose=1
    )

    plot_history(history_2, "Phase 2")

    print("\nğŸ”“ Phase 3: Fine-tuning model...")
    model.layers[0].trainable = True
    for layer in model.layers[0].layers[:-20]:
        layer.trainable = False

    trainable_count = sum([1 for layer in model.layers[0].layers if layer.trainable])
    print(f"Unfrozen layers: {trainable_count}/total layers")

    opt = optimizers.Adam(learning_rate=learning_rate/10)
    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history_3 = model.fit(
        cleaned_train_ds,
        epochs=15,
        callbacks=callbacks,
        verbose=1
    )

    plot_history(history_3, "Phase 3")

    return system, model, (history_1, history_2, history_3)

# ========================
# 3. PREDICTION
# ========================

def predict_test_dataset(model, test_ds, confidence_threshold=0.7):
    """
    Predict pada test dataset dan simpan ke CSV dengan kolom Id dan Predicted
    """
    class_names = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']
    
    print("ğŸ”® PREDICTING TEST DATASET...")
    print(f"Classes: {class_names}")
    print(f"Confidence threshold: {confidence_threshold}")
    
    all_predictions = []
    all_confidences = []
    all_images = []
    all_filenames = []
    batch_count = 0
    
    print("\nğŸ“Š Processing test batches...")
    
    try:
        file_paths = test_ds.file_paths
    except AttributeError:
        print("âš ï¸� Warning: test_ds does not have file_paths. Using index-based IDs.")
        file_paths = [f"image_{i}.jpg" for i in range(2025)]
    
    file_index = 0
    
    for batch in test_ds:
        images = batch[0] if isinstance(batch, tuple) else batch
        batch_count += 1
        print(f"Processing batch {batch_count}...", end=' ')
        
        preds = model.predict(images, verbose=0)
        classes = np.argmax(preds, axis=1)
        confidences = np.max(preds, axis=1)
        
        batch_results = []
        batch_filenames = []
        
        for i in range(len(images)):
            img = images[i].numpy()
            pred = classes[i]
            conf = confidences[i]
            probs = preds[i]
            
            if file_index < len(file_paths):
                filename = file_paths[file_index]
                filename = filename.split('/')[-1]
            else:
                filename = f"image_{file_index}.jpg"
            
            result = {
                'image_index': len(all_predictions) + i,
                'prediction': pred,
                'class_name': class_names[pred],
                'confidence': conf,
                'probabilities': probs,
                'prediction_method': 'DenseNet121',
                'filename': filename
            }
            
            batch_results.append(result)
            batch_filenames.append(filename)
            file_index += 1
        
        all_predictions.extend(batch_results)
        all_images.extend(images.numpy())
        all_filenames.extend(batch_filenames)
        print(f"âœ… {len(batch_results)} predictions")
    
    print(f"\nğŸ�¯ PREDICTION SUMMARY:")
    print(f"Total images processed: {len(all_predictions)}")
    
    class_counts = {}
    confidence_stats = {'high': 0, 'medium': 0, 'low': 0}
    
    for pred in all_predictions:
        class_name = pred['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        conf = pred['confidence']
        if conf >= 0.8:
            confidence_stats['high'] += 1
        elif conf >= 0.6:
            confidence_stats['medium'] += 1
        else:
            confidence_stats['low'] += 1
    
    print(f"\nğŸ�·ï¸� PREDICTED CLASS DISTRIBUTION:")
    for class_name, count in class_counts.items():
        print(f"   {class_name}: {count} ({count/len(all_predictions)*100:.1f}%)")
    
    print(f"\nğŸ�¯ CONFIDENCE DISTRIBUTION:")
    print(f"   High (â‰¥0.8): {confidence_stats['high']} ({confidence_stats['high']/len(all_predictions)*100:.1f}%)")
    print(f"   Medium (0.6-0.8): {confidence_stats['medium']} ({confidence_stats['medium']/len(all_predictions)*100:.1f}%)")
    print(f"   Low (<0.6): {confidence_stats['low']} ({confidence_stats['low']/len(all_predictions)*100:.1f}%)")
    
    all_confidences = [pred['confidence'] for pred in all_predictions]
    print(f"\nğŸ“Š CONFIDENCE STATISTICS:")
    print(f"   Mean: {np.mean(all_confidences):.3f}")
    print(f"   Std: {np.std(all_confidences):.3f}")
    print(f"   Min: {np.min(all_confidences):.3f}")
    print(f"   Max: {np.max(all_confidences):.3f}")
    
    print("\nğŸ’¾ Saving predictions to 'predictions.csv'...")
    pred_df = pd.DataFrame({
        'Id': [pred['filename'] for pred in all_predictions],
        'Predicted': [pred['prediction'] for pred in all_predictions]
    })
    pred_df.to_csv('predictions.csv', index=False)
    print("âœ… Predictions saved to 'predictions.csv'")
    
    return all_predictions, np.array(all_images)

# ========================
# 4. MAIN EXECUTION
# ========================

def main():
    """
    Main execution pipeline untuk pneumonia classification dengan DenseNet121
    """
    print("ğŸš€ Pneumonia Classification System with DenseNet121")
    print("=" * 60)
    print("Classes Mapping:")
    print("  [0] Bacterial Pneumonia")
    print("  [1] Corona Virus Disease (COVID-19)")
    print("  [2] Normal")
    print("  [3] Tuberculosis")
    print("  [4] Viral Pneumonia")
    print("=" * 60)

    # Dataset paths
    train_dir = "/kaggle/input/srifoton-25-machine-learning-competition/train/train"
    val_dir = "/kaggle/input/srifoton-25-machine-learning-competition/val/val"
    test_dir = "/kaggle/input/srifoton-25-machine-learning-competition/test"
    img_size = (224, 224)
    batch_size = 32

    # Load datasets
    try:
        print("\nğŸ“‚ Loading datasets...")
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=img_size,
            batch_size=batch_size,
            shuffle=True
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            image_size=img_size,
            batch_size=batch_size,
            shuffle=True
        )
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=img_size,
            batch_size=batch_size,
            shuffle=False
        )
        print("âœ… Datasets loaded successfully")
    except Exception as e:
        print(f"â�Œ Error loading datasets: {e}")
        return

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("âœ… GPU detected:", gpus)
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("âœ… GPU memory growth enabled")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("âš ï¸� No GPU detected. Training akan menggunakan CPU (slower)")

    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("âœ… Mixed precision enabled for faster training")

    print("\nğŸ”§ Key System Enhancements:")
    print("1. ğŸ“¸ Mild CLAHE, brightness adjustment, Gaussian blur untuk X-ray preprocessing")
    print("2. âš¡ Transfer Learning dengan DenseNet121")
    print("3. ğŸ�—ï¸� Single-stage architecture dengan fine-tuning")
    print("4. ğŸ›¡ï¸� Anti-overfitting: Enhanced augmentation (0-45 deg rotation), L2 regularization, CleanLab filtering")

    print("\n" + "="*60)
    print("ğŸ“‹ USAGE INSTRUCTIONS:")
    print("1. Load your dataset dengan format yang sesuai:")
    print("   train_ds = tf.keras.preprocessing.image_dataset_from_directory(...)")
    print("   test_ds = tf.keras.preprocessing.image_dataset_from_directory(...)")
    print("")
    print("2. Combine train and validation datasets:")
    print("   combined_ds = train_ds.concatenate(val_ds)")
    print("")
    print("3. Training dengan custom parameters:")
    print("   system, model, histories = train_model(combined_ds, dropout_rate=0.5, optimizer='adam', learning_rate=1e-4)")
    print("")
    print("4. Predict on test dataset:")
    print("   predictions, test_images = predict_test_dataset(model, test_ds, confidence_threshold=0.7)")

    print("\nâœ… System ready untuk deployment!")

    # Combine train and validation datasets
    print("\nğŸ“¥ Combining train and validation datasets...")
    combined_ds = train_ds.concatenate(val_ds)
    combined_ds = combined_ds.unbatch().batch(32).prefetch(tf.data.AUTOTUNE)

    # Apply preprocessing to datasets
    train_datagen, test_datagen = create_enhanced_data_generators()

    def apply_datagen(dataset, datagen):
        def gen():
            for images, labels in dataset:
                images = images.numpy()
                for i in range(images.shape[0]):
                    img = datagen.random_transform(images[i])
                    yield img, labels[i]
        return tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        ).batch(32).prefetch(tf.data.AUTOTUNE)

    combined_ds = apply_datagen(combined_ds, train_datagen)
    test_ds = apply_datagen(test_ds, test_datagen)

    system, model, histories = train_model(combined_ds, dropout_rate=0.5, optimizer='adam', learning_rate=1e-4)
    predictions, test_images = predict_test_dataset(model, test_ds, confidence_threshold=0.7)

    print("ğŸ�‰ Prediction selesai!")

if __name__ == "__main__":
    main()


import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os

# Dataset paths
train_dir = "/kaggle/input/srifoton-25-machine-learning-competition/train/train"
val_dir = "/kaggle/input/srifoton-25-machine-learning-competition/val/val"
test_dir = "/kaggle/input/srifoton-25-machine-learning-competition/test"
img_size = (224, 224)
batch_size = 32

# ========================
# 1. ENHANCED PREPROCESSING
# ========================

class EnhancedPreprocessing:
    """
    Optimized preprocessing for pneumonia classification
    Focus: light contrast enhancement using CLAHE, brightness adjustment, and Gaussian blur
    """
    @staticmethod
    def enhance_image(image):
        """
        Preprocessing for pneumonia X-ray
        - Brightness adjustment for lighting
        - Light CLAHE for adaptive contrast
        - Gaussian blur to reduce noise
        """
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Brightness adjustment
        image = cv2.convertScaleAbs(image, beta=10)

        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            enhanced = clahe.apply(image)
            if len(enhanced.shape) == 2:
                enhanced = np.expand_dims(enhanced, axis=-1)
                enhanced = np.repeat(enhanced, 3, axis=-1)

        # Gaussian blur
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), sigmaX=0)

        # Normalize back to [0,1]
        enhanced = enhanced.astype(np.float32) / 255.0
        return enhanced

def create_enhanced_data_generators():
    """
    Data augmentation optimized for pneumonia
    """
    train_datagen = ImageDataGenerator(
        preprocessing_function=EnhancedPreprocessing.enhance_image,
        rotation_range=45,
        width_shift_range=0.15,
        height_shift_range=0.15,
        brightness_range=[0.7, 1.3],
        zoom_range=0.2,
        shear_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=EnhancedPreprocessing.enhance_image
    )

    return train_datagen, test_datagen

# ========================
# 2. MODEL ARCHITECTURE
# ========================

class PneumoniaClassificationSystem:
    def __init__(self, num_classes=5):
        self.num_classes = num_classes
        self.class_names = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']
        self.model = None

    def build_model(self, dropout_rate=0.5, optimizer='adam', learning_rate=1e-4):
        """
        Model with transfer learning using VGG16
        """
        try:
            backbone = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            backbone.trainable = False

            self.model = models.Sequential([
                backbone,
                layers.GlobalAveragePooling2D(),
                layers.Dense(256, activation='relu', kernel_regularizer=l2(0.02)),
                layers.Dropout(dropout_rate),
                layers.Dense(self.num_classes, activation='softmax')
            ])

            if optimizer.lower() == 'adam':
                opt = optimizers.Adam(learning_rate=learning_rate)
            elif optimizer.lower() == 'rmsprop':
                opt = optimizers.RMSprop(learning_rate=learning_rate)
            elif optimizer.lower() == 'sgd':
                opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
            else:
                opt = optimizers.Adam(learning_rate=learning_rate)

            self.model.compile(
                optimizer=opt,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            return self.model
        except Exception as e:
            print(f"â�Œ Error building model: {e}")
            raise

def clean_low_confidence_samples(model, dataset, confidence_threshold=0.6):
    """
    Filter out training samples with low confidence predictions
    """
    print("\nğŸ§¹ Cleaning low-confidence samples...")
    images, labels = [], []
    for batch in dataset.unbatch().batch(32):
        batch_images, batch_labels = batch
        images.append(batch_images.numpy())
        labels.append(batch_labels.numpy())

    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Predict probabilities
    probs = model.predict(images, verbose=1)
    confidences = np.max(probs, axis=1)

    # Keep samples with confidence >= threshold
    keep_mask = confidences >= confidence_threshold
    filtered_images = images[keep_mask]
    filtered_labels = labels[keep_mask]

    print(f"ğŸ“Š CleanLab Summary:")
    print(f"   Total samples: {len(images)}")
    print(f"   Kept samples: {len(filtered_images)} ({len(filtered_images)/len(images)*100:.1f}%)")
    print(f"   Dropped samples: {len(images) - len(filtered_images)}")

    # Create new dataset
    filtered_dataset = tf.data.Dataset.from_tensor_slices((filtered_images, filtered_labels))
    filtered_dataset = filtered_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    return filtered_dataset, len(filtered_images)

def generate_pseudolabels(model, test_ds, confidence_threshold=0.95):
    """
    Generate pseudolabels for test dataset with high-confidence predictions
    """
    print("\nğŸ”� Generating pseudolabels for test dataset...")
    images, pseudolabels = [], []
    for batch in test_ds:
        batch_images = batch[0] if isinstance(batch, tuple) else batch
        preds = model.predict(batch_images, verbose=0)
        classes = np.argmax(preds, axis=1).astype(np.int32)
        confidences = np.max(preds, axis=1)

        # Select high-confidence predictions
        high_conf_mask = confidences >= confidence_threshold
        images.append(batch_images[high_conf_mask].numpy())
        pseudolabels.append(classes[high_conf_mask])

    images = np.concatenate(images, axis=0) if images else np.array([])
    pseudolabels = np.concatenate(pseudolabels, axis=0) if pseudolabels else np.array([])

    print(f"ğŸ“Š Pseudolabeling Summary:")
    total_test = sum([len(b[0]) for b in test_ds])
    print(f"   Total test images: {total_test}")
    print(f"   Pseudolabeled images: {len(images)} ({len(images)/total_test*100:.1f}%)")
    print(f"   Dropped images: {total_test - len(images)}")

    if len(images) > 0:
        pseudo_dataset = tf.data.Dataset.from_tensor_slices((images, pseudolabels))
        pseudo_dataset = pseudo_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
        return pseudo_dataset, len(images)
    else:
        print("âš ï¸� No high-confidence pseudolabels generated.")
        return None, 0

def train_model(train_ds, val_ds, dropout_rate=0.5, optimizer='adam', learning_rate=1e-4):
    """
    Training pipeline for VGG16 model
    """
    print("ğŸ”¥ Building Pneumonia Classification System with VGG16...")
    print(f"Classes: {['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']}")

    system = PneumoniaClassificationSystem(num_classes=5)
    model = system.build_model(dropout_rate=dropout_rate, optimizer=optimizer, learning_rate=learning_rate)
    print(f"Model built with {model.count_params():,} parameters")

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
        ModelCheckpoint('vgg16_pneumonia_best.h5', monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, verbose=1, min_lr=1e-7)
    ]

    print("\nğŸ”’ Phase 1: Initial training with frozen backbone...")
    history_1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        callbacks=callbacks,
        verbose=1
    )

    print("\nğŸ§¹ Applying CleanLab to filter low-confidence samples...")
    cleaned_train_ds, num_cleaned = clean_low_confidence_samples(model, train_ds)

    print("\nğŸ”� Generating pseudolabels for test dataset...")
    pseudo_ds, num_pseudo = generate_pseudolabels(model, test_ds)

    if pseudo_ds is not None:
        print("\nğŸ“¥ Combining cleaned training dataset with pseudolabeled data...")
        train_with_pseudo_ds = cleaned_train_ds.concatenate(pseudo_ds)
    else:
        print("\nâš ï¸� No pseudolabels available. Using cleaned training dataset only.")
        train_with_pseudo_ds = cleaned_train_ds

    train_with_pseudo_ds = train_with_pseudo_ds.shuffle(buffer_size=10000).batch(32).prefetch(tf.data.AUTOTUNE)

    print("\nğŸ”’ Phase 2: Training with cleaned dataset and pseudolabels...")
    history_2 = model.fit(
        train_with_pseudo_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=callbacks,
        verbose=1
    )

    print("\nğŸ”“ Phase 3: Fine-tuning model...")
    model.layers[0].trainable = True
    for layer in model.layers[0].layers[:-5]:
        layer.trainable = False

    trainable_count = sum([layer.trainable for layer in model.layers[0].layers])
    print(f"Unfrozen layers: {trainable_count}/total layers")

    opt = optimizers.Adam(learning_rate=learning_rate/10)
    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history_3 = model.fit(
        train_with_pseudo_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=callbacks,
        verbose=1
    )

    return system, model, (history_1, history_2, history_3)

# ========================
# 3. PREDICTION
# ========================

def predict_test_dataset(model, test_ds, confidence_threshold=0.7):
    """
    Predict on test dataset and save to CSV with Id and Predicted columns
    """
    class_names = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']
    
    print("ğŸ”® PREDICTING TEST DATASET...")
    print(f"Classes: {class_names}")
    print(f"Confidence threshold: {confidence_threshold}")
    
    all_predictions = []
    all_images = []
    batch_count = 0
    
    print("\nğŸ“Š Processing test batches...")
    
    # Get file paths for correct IDs
    try:
        file_paths = test_ds.file_paths
    except AttributeError:
        print("âš ï¸� Warning: test_ds does not have file_paths. Attempting to list from directory.")
        file_paths = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    file_index = 0
    
    for batch in test_ds:
        images = batch[0] if isinstance(batch, tuple) else batch
        batch_count += 1
        print(f"Processing batch {batch_count}...", end=' ')
        
        preds = model.predict(images, verbose=0)
        classes = np.argmax(preds, axis=1)
        confidences = np.max(preds, axis=1)
        
        batch_results = []
        
        for i in range(len(images)):
            pred = classes[i]
            conf = confidences[i]
            probs = preds[i]
            
            if file_index < len(file_paths):
                filename = os.path.basename(file_paths[file_index])
            else:
                filename = f"image_{file_index}.jpg"
            
            result = {
                'prediction': pred,
                'class_name': class_names[pred],
                'confidence': conf,
                'filename': filename
            }
            
            batch_results.append(result)
            file_index += 1
        
        all_predictions.extend(batch_results)
        all_images.extend(images.numpy())
        print(f"âœ… {len(batch_results)} predictions")
    
    print(f"\nğŸ�¯ PREDICTION SUMMARY:")
    print(f"Total images processed: {len(all_predictions)}")
    
    class_counts = {}
    confidence_stats = {'high': 0, 'medium': 0, 'low': 0}
    
    for pred in all_predictions:
        class_name = pred['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        conf = pred['confidence']
        if conf >= 0.8:
            confidence_stats['high'] += 1
        elif conf >= 0.6:
            confidence_stats['medium'] += 1
        else:
            confidence_stats['low'] += 1
    
    print(f"\nğŸ�·ï¸� PREDICTED CLASS DISTRIBUTION:")
    for class_name, count in class_counts.items():
        print(f"   {class_name}: {count} ({count/len(all_predictions)*100:.1f}%)")
    
    print(f"\nğŸ�¯ CONFIDENCE DISTRIBUTION:")
    print(f"   High (â‰¥0.8): {confidence_stats['high']} ({confidence_stats['high']/len(all_predictions)*100:.1f}%)")
    print(f"   Medium (0.6-0.8): {confidence_stats['medium']} ({confidence_stats['medium']/len(all_predictions)*100:.1f}%)")
    print(f"   Low (<0.6): {confidence_stats['low']} ({confidence_stats['low']/len(all_predictions)*100:.1f}%)")
    
    all_confidences = [pred['confidence'] for pred in all_predictions]
    print(f"\nğŸ“Š CONFIDENCE STATISTICS:")
    print(f"   Mean: {np.mean(all_confidences):.3f}")
    print(f"   Std: {np.std(all_confidences):.3f}")
    print(f"   Min: {np.min(all_confidences):.3f}")
    print(f"   Max: {np.max(all_confidences):.3f}")
    
    print("\nğŸ’¾ Saving predictions to 'predictions.csv'...")
    pred_df = pd.DataFrame({
        'Id': [pred['filename'] for pred in all_predictions],
        'Predicted': [pred['prediction'] for pred in all_predictions]
    })
    pred_df.to_csv('predictions.csv', index=False)
    print("âœ… Predictions saved to 'predictions.csv'")
    
    return all_predictions, np.array(all_images)

# ========================
# 4. MAIN EXECUTION
# ========================

def main():
    """
    Main execution pipeline for pneumonia classification with VGG16
    """
    print("ğŸš€ Pneumonia Classification System with VGG16")
    print("=" * 60)
    print("Classes Mapping:")
    print("  [0] Bacterial Pneumonia")
    print("  [1] Corona Virus Disease (COVID-19)")
    print("  [2] Normal")
    print("  [3] Tuberculosis")
    print("  [4] Viral Pneumonia")
    print("=" * 60)

    # Load datasets
    try:
        print("\nğŸ“‚ Loading datasets...")
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=img_size,
            batch_size=batch_size,
            shuffle=True
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            image_size=img_size,
            batch_size=batch_size,
            shuffle=True
        )
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=img_size,
            batch_size=batch_size,
            shuffle=False
        )
        print("âœ… Datasets loaded successfully")
    except Exception as e:
        print(f"â�Œ Error loading datasets: {e}")
        return

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("âœ… GPU detected:", gpus)
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("âœ… GPU memory growth enabled")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("âš ï¸� No GPU detected. Training will use CPU (slower)")

    # Note: Removed mixed precision to avoid shape-related errors

    print("\nğŸ”§ Key System Enhancements:")
    print("1. ğŸ“¸ Mild CLAHE, brightness adjustment, Gaussian blur for X-ray preprocessing")
    print("2. âš¡ Transfer Learning with VGG16")
    print("3. ğŸ�—ï¸� Single-stage architecture with fine-tuning")
    print("4. ğŸ›¡ï¸� Anti-overfitting: Enhanced augmentation (0-45 deg rotation), L2 regularization, CleanLab filtering")
    print("5. ğŸ“ˆ Pseudolabeling for dataset augmentation with 95% confidence threshold")

    print("\n" + "="*60)
    print("ğŸ“‹ USAGE INSTRUCTIONS:")
    print("1. Load your dataset with the appropriate format:")
    print("   train_ds = tf.keras.preprocessing.image_dataset_from_directory(...)")
    print("   val_ds = tf.keras.preprocessing.image_dataset_from_directory(...)")
    print("   test_ds = tf.keras.preprocessing.image_dataset_from_directory(...)")
    print("")
    print("2. Training with custom parameters:")
    print("   system, model, histories = train_model(train_ds, val_ds, dropout_rate=0.5, optimizer='adam', learning_rate=1e-4)")
    print("")
    print("3. Predict on test dataset:")
    print("   predictions, test_images = predict_test_dataset(model, test_ds, confidence_threshold=0.7)")

    print("\nâœ… System ready for deployment!")

    # Apply preprocessing to datasets
    train_datagen, test_datagen = create_enhanced_data_generators()

    def apply_datagen(dataset, datagen):
        def gen():
            for images, labels in dataset:
                images = images.numpy()
                labels = labels.numpy().astype(np.int32)  # Ensure labels are int32
                for i in range(images.shape[0]):
                    img = datagen.random_transform(images[i])
                    yield img, labels[i]
        return tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        ).batch(32).prefetch(tf.data.AUTOTUNE)

    train_ds = apply_datagen(train_ds, train_datagen)
    val_ds = apply_datagen(val_ds, test_datagen)
    test_ds = apply_datagen(test_ds, test_datagen)

    system, model, histories = train_model(train_ds, val_ds, dropout_rate=0.5, optimizer='adam', learning_rate=1e-4)
    predictions, test_images = predict_test_dataset(model, test_ds, confidence_threshold=0.7)

    print("ğŸ�‰ Prediction completed!")

if __name__ == "__main__":
    main(
        
    )


pip install tensorflow_addons


import matplotlib.pyplot as plt

class_names = train_ds.class_names
plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):  # Take one batch to display examples
    for i in range(len(class_names)):
        # Find indices of images belonging to the current class
        class_indices = tf.where(labels == i).numpy().flatten()
        # Take up to 3 indices from the current class
        display_indices = class_indices[:3]

        for j, idx in enumerate(display_indices):
            ax = plt.subplot(len(class_names), 3, i * 3 + j + 1)
            plt.imshow(images[idx].numpy().astype("uint8"))
            plt.title(class_names[labels[idx]])
            plt.axis("off")

plt.tight_layout()
plt.show()


import collections

# Function to count class distribution
def count_class_distribution(dataset):
    class_counts = collections.defaultdict(int)
    for _, labels in dataset:
        for label in labels.numpy():
            class_counts[label] += 1
    return dict(class_counts)

train_class_counts = count_class_distribution(train_ds)
val_class_counts = count_class_distribution(val_ds)

print("Training set class distribution:")
for class_index, count in train_class_counts.items():
    print(f"Class {class_names[class_index]}: {count}")

print("\nValidation set class distribution:")
for class_index, count in val_class_counts.items():
    print(f"Class {class_names[class_index]}: {count}")

# Optional: Visualize the distribution
plt.figure(figsize=(10, 5))
plt.bar(class_names, train_class_counts.values())
plt.title("Training Set Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(class_names, val_class_counts.values(), color='orange')
plt.title("Validation Set Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.show()


import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# STEP 1: ROI EXTRACTION (Region of Interest - Segmentasi Paru)
# ============================================================================

def create_lung_mask_opencv(image):
    """Helper untuk OpenCV: ekstrak mask paru dari grayscale."""
    image_np = image.numpy()  # Convert TF tensor ke NumPy
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image_np.astype(np.uint8)

    # Convert ke grayscale jika RGB atau single-channel
    if len(image_np.shape) == 3 and image_np.shape[-1] == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np if len(image_np.shape) == 2 else image_np[:, :, 0]

    # Otsu thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    # Normalize dan tambah channel dim
    mask = cleaned.astype(np.float32) / 255.0
    mask = np.expand_dims(mask, axis=-1)  # Shape: (224, 224, 1)
    return mask

def create_lung_mask(image):
    """Wrapper TF untuk mask."""
    if len(image.shape) == 2:
        image = tf.expand_dims(image, axis=-1)  # (224, 224) -> (224, 224, 1)
    elif len(image.shape) != 3:
        raise ValueError(f"Input ke create_lung_mask harus 2D atau 3D, diterima shape: {image.shape}")

    mask = tf.py_function(
        func=create_lung_mask_opencv,
        inp=[image],
        Tout=tf.float32
    )
    mask.set_shape([224, 224, 1])
    return mask

def extract_lung_roi(image):
    """Apply mask ke image (broadcasting untuk channel 3)."""
    if len(image.shape) == 2:
        image = tf.expand_dims(image, axis=-1)  # (224, 224) -> (224, 224, 1)
        image = tf.repeat(image, 3, axis=-1)  # Buat RGB: (224, 224, 3)
    elif len(image.shape) == 3:
        if image.shape[-1] == 1:
            image = tf.repeat(image, 3, axis=-1)  # Convert single-channel ke RGB
    else:
        raise ValueError(f"Input ke extract_lung_roi harus 2D atau 3D, diterima shape: {image.shape}")

    lung_mask = create_lung_mask(image)
    roi_image = image * lung_mask  # Output: (224, 224, 3) untuk RGB
    return roi_image

# ============================================================================
# STEP 2: CONTRAST ENHANCEMENT dengan CLAHE
# ============================================================================

def apply_clahe_opencv(image):
    """Helper CLAHE: proses RGB atau grayscale."""
    image_np = image.numpy()
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image_np.astype(np.uint8)

    if len(image_np.shape) == 3 and image_np.shape[-1] == 3:
        # RGB: CLAHE pada L channel di LAB
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        # Grayscale: apply CLAHE, lalu repeat ke 3 channel
        gray = image_np if len(image_np.shape) == 2 else image_np[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        enhanced = np.repeat(np.expand_dims(enhanced_gray, axis=-1), 3, axis=-1)  # Buat RGB

    return enhanced.astype(np.float32) / 255.0

def apply_clahe(image):
    """Wrapper TF untuk CLAHE."""
    enhanced = tf.py_function(
        func=apply_clahe_opencv,
        inp=[image],
        Tout=tf.float32
    )
    enhanced.set_shape([224, 224, 3])  # Output selalu RGB
    return enhanced

# ============================================================================
# STEP 3: PIXEL NORMALIZATION & CLIPPING
# ============================================================================

def normalize_and_clip(image):
    """Normalize & clip ke [0,1]."""
    image = tf.cast(image, tf.float32)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

# ============================================================================
# STEP 4: DATA AUGMENTATION (Hanya untuk Training)
# ============================================================================

def augment_image(image, label):
    """Augmentasi: pastikan 3D sebelum tf.image ops."""
    image = tf.ensure_shape(image, [224, 224, 3])  # Pastikan shape
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    image = tf.image.rot90(image, k=k)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

# ============================================================================
# STEP 5: PIPELINE PREPROCESSING LENGKAP
# ============================================================================

def preprocess_for_training(image, label):
    """Pipeline preprocessing untuk data TRAINING."""
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1]
    image = extract_lung_roi(image)
    image = apply_clahe(image)
    image = normalize_and_clip(image)
    image, label = augment_image(image, label)
    return image, label

def preprocess_for_validation(image, label):
    """Pipeline preprocessing untuk data VALIDATION/TEST."""
    image = tf.cast(image, tf.float32) / 255.0
    image = extract_lung_roi(image)
    image = apply_clahe(image)
    image = normalize_and_clip(image)
    if label is None:
        return image  # Untuk test
    return image, label

# ============================================================================
# APPLY TO DATASET
# ============================================================================

# Load dataset
img_size = (224, 224)
batch_size = 32

try:
    train_ds = train_ds
    val_ds = val_ds
    test_ds = test_ds
except Exception as e:
    print("â�Œ Error loading dataset:", str(e))
    raise

# Unbatch datasets to process single images
train_ds_unbatched = train_ds.unbatch()
val_ds_unbatched = val_ds.unbatch()
test_ds_unbatched = test_ds.unbatch()

# Apply preprocessing
train_ds_processed = (train_ds_unbatched
                     .map(preprocess_for_training, num_parallel_calls=tf.data.AUTOTUNE)
                     .batch(batch_size)  # Re-batch after preprocessing
                     .prefetch(tf.data.AUTOTUNE))

val_ds_processed = (val_ds_unbatched
                   .map(preprocess_for_validation, num_parallel_calls=tf.data.AUTOTUNE)
                   .batch(batch_size)
                   .prefetch(tf.data.AUTOTUNE))

test_ds_processed = (test_ds_unbatched
                    .map(lambda img: preprocess_for_validation(img, None), num_parallel_calls=tf.data.AUTOTUNE)
                    .batch(batch_size)
                    .prefetch(tf.data.AUTOTUNE))


for images, labels in train_ds_unbatched.take(1):
    image = tf.cast(images, tf.float32) / 255.0
    roi = extract_lung_roi(image)
    clahe = apply_clahe(roi)
    norm = normalize_and_clip(clahe)
    aug, _ = augment_image(norm, labels)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(image, cmap='gray' if image.shape[-1] == 1 else None)
    plt.title("Original")
    plt.axis('off')
    plt.subplot(1, 4, 2)
    plt.imshow(roi, cmap='gray' if roi.shape[-1] == 1 else None)
    plt.title("ROI")
    plt.axis('off')
    plt.subplot(1, 4, 3)
    plt.imshow(clahe, cmap='gray' if clahe.shape[-1] == 1 else None)
    plt.title("CLAHE")
    plt.axis('off')
    plt.subplot(1, 4, 4)
    plt.imshow(aug, cmap='gray' if aug.shape[-1] == 1 else None)
    plt.title("Augmented")
    plt.axis('off')
    plt.show()
    break


# Define class_names if it's not already defined
if 'class_names' not in locals() and 'class_names' not in globals():
    try:
        class_names = train_ds.class_names
    except NameError:
        print("Error: 'train_ds' is not defined. Please run the cell that loads the dataset first.")
        class_names = [] # Set to empty list to avoid further errors

if class_names:
    print("Class Index Mapping:")
    for i, class_name in enumerate(class_names):
        print(f"Index {i}: {class_name}")
else:
    print("Class names could not be retrieved.")


test_ds


import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, DenseNet121
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import cv2

# ========================
# 1. ENHANCED PREPROCESSING
# ========================

class EnhancedPreprocessing:
    """
    Optimalisasi preprocessing khusus untuk pneumonia classification
    Focus: pencahayaan, ketajaman, dan kontras untuk membedakan pola bakterial vs viral
    """

    @staticmethod
    def enhance_image(image):
        """
        Enhanced preprocessing untuk pneumonia X-ray
        - CLAHE untuk kontras adaptif
        - Gaussian blur removal untuk ketajaman
        - Histogram equalization untuk pencahayaan optimal
        """
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Bagus untuk enhance perbedaan antara area padat dan menyebar
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        if len(image.shape) == 3:
            # RGB image
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale
            enhanced = clahe.apply(image)
            if len(enhanced.shape) == 2:
                enhanced = np.expand_dims(enhanced, axis=-1)
                enhanced = np.repeat(enhanced, 3, axis=-1)

        # Sharpening kernel untuk mempertegas batas antara area normal dan infected
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)

        # Normalize back to [0,1]
        enhanced = enhanced.astype(np.float32) / 255.0

        return enhanced

def create_enhanced_data_generators():
    """
    Data augmentation yang dioptimalkan untuk pneumonia
    Focus: variasi pencahayaan dan rotasi kecil (sesuai kondisi X-ray)
    """
    train_datagen = ImageDataGenerator(
        preprocessing_function=EnhancedPreprocessing.enhance_image,
        rotation_range=10,  # Rotasi kecil saja (X-ray biasanya standar orientasi)
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2],  # Variasi pencahayaan
        zoom_range=0.1,
        horizontal_flip=False,  # X-ray tidak boleh flip horizontal
        fill_mode='nearest',
        validation_split=0.2
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=EnhancedPreprocessing.enhance_image,
        validation_split=0.2
    )

    return train_datagen, val_datagen

# ========================
# 2. DUAL POOLING STRATEGY
# ========================

class DualPoolingLayer(layers.Layer):
    """
    Kombinasi Max Pooling + Global Average Pooling
    - Max Pooling: menangkap pola padat/intens (bacterial pneumonia)
    - Global Average Pooling: menangkap pola menyebar (viral pneumonia)
    Weight: 0.5 untuk setiap pooling strategy
    """

    def __init__(self, **kwargs):
        super(DualPoolingLayer, self).__init__(**kwargs)
        self.max_pool = layers.GlobalMaxPooling2D()
        self.avg_pool = layers.GlobalAveragePooling2D()

    def call(self, inputs):
        max_features = self.max_pool(inputs)
        avg_features = self.avg_pool(inputs)

        # Combine dengan weight 0.5 masing-masing
        combined = 0.5 * max_features + 0.5 * avg_features
        return combined

    def get_config(self):
        return super(DualPoolingLayer, self).get_config()

# ========================
# 3. TWO-STAGE MODEL ARCHITECTURE
# ========================

class PneumoniaClassificationSystem:
    def __init__(self, num_classes=5):  # 5 classes: BACTERIAL, COVID, NORMAL, TB, VIRAL
        self.num_classes = num_classes
        self.class_names = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']
        self.general_model = None
        self.specialist_model = None

    def build_general_model(self, backbone_name='resnet50'):
        """
        Model 1: General 5-class classifier
        Classes: [0]Bacterial, [1]COVID, [2]Normal, [3]TB, [4]Viral
        Focus: Enhanced differentiation untuk Bacterial (0) vs Viral (4) Pneumonia
        Backbone options: resnet50, resnet101, densenet121, mobilenetv2
        """
        if backbone_name.lower() == 'resnet50':
            backbone = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        elif backbone_name.lower() == 'resnet101':
            backbone = tf.keras.applications.ResNet101(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        elif backbone_name.lower() == 'densenet121':
            backbone = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        elif backbone_name.lower() == 'mobilenetv2':
            backbone = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        else:
            backbone = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Freeze backbone initially
        backbone.trainable = False

        # Build general model
        self.general_model = models.Sequential([
            backbone,
            DualPoolingLayer(),  # Dual pooling strategy
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        return self.general_model

    def build_specialist_model(self):
        """
        Model 2: Specialist untuk BACTERIAL (0) vs VIRAL (4) Pneumonia ONLY
        Menggunakan InceptionV3 (berdasarkan research paper)
        Enhanced architecture khusus untuk membedakan pola bakterial vs viral
        """
        backbone = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        backbone.trainable = False

        # Enhanced architecture khusus untuk Bacterial vs Viral differentiation
        self.specialist_model = models.Sequential([
            backbone,
            DualPoolingLayer(),  # CRITICAL: Max pooling untuk bacterial (padat) + Avg pooling untuk viral (menyebar)
            layers.Dense(1024, activation='relu', name='specialist_dense_1'),
            layers.Dropout(0.6),
            layers.Dense(512, activation='relu', name='specialist_dense_2'),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu', name='specialist_dense_3'),
            layers.Dropout(0.3),
            # Output layer untuk binary classification: Bacterial vs Viral
            layers.Dense(2, activation='softmax', name='specialist_output')  # [Bacterial, Viral] only
        ])

        return self.specialist_model

def create_bacterial_viral_dataset(train_ds, val_ds):
    """
    Filter dataset untuk specialist model (Bacterial vs Viral Pneumonia only)
    Index 0: Bacterial Pneumonia -> Label 0
    Index 4: Viral Pneumonia -> Label 1
    """
    def filter_and_relabel(dataset):
        filtered_images = []
        filtered_labels = []

        for images, labels in dataset:
            # Convert to numpy for easier processing
            images_np = images.numpy()
            labels_np = labels.numpy()

            # Filter untuk bacterial (0) dan viral (4) pneumonia
            mask = (labels_np == 0) | (labels_np == 4)

            if np.any(mask):
                selected_images = images_np[mask]
                selected_labels = labels_np[mask]

                # Relabel: Bacterial (0) -> 0, Viral (4) -> 1
                selected_labels[selected_labels == 4] = 1

                filtered_images.extend(selected_images)
                filtered_labels.extend(selected_labels)

        # Convert back to tensorflow dataset
        if filtered_images:
            filtered_images = np.array(filtered_images)
            filtered_labels = np.array(filtered_labels)

            filtered_ds = tf.data.Dataset.from_tensor_slices((filtered_images, filtered_labels))
            filtered_ds = filtered_ds.batch(32).prefetch(tf.data.AUTOTUNE)
            return filtered_ds
        else:
            return None

    train_specialist = filter_and_relabel(train_ds)
    val_specialist = filter_and_relabel(val_ds)

    print(f"âœ… Specialist dataset created for Bacterial vs Viral classification")
    return train_specialist, val_specialist

def train_two_stage_system(train_ds, val_ds):
    """
    Two-stage training pipeline for 5-class pneumonia classification
    Stage 1: General 5-class model dengan focus pada Bacterial vs Viral
    Stage 2: Specialist Binary model (Bacterial vs Viral only)
    """
    class_names = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']
    print("ğŸ”¥ Building Enhanced 5-Class Pneumonia Classification System...")
    print(f"Classes: {class_names}")

    # Initialize system
    system = PneumoniaClassificationSystem(num_classes=5)

    # ===== STAGE 1: General 5-Class Model =====
    print("\nğŸ“� Stage 1: Training General 5-Class Model")
    print("Focus: Enhanced Bacterial vs Viral differentiation with dual pooling")

    general_model = system.build_general_model('resnet50')
    general_model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"Model built with {general_model.count_params():,} parameters")

    # Callbacks for stage 1
    callbacks_stage1 = [
        EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1),
        ModelCheckpoint('general_model_5class_best.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, verbose=1, min_lr=1e-7)
    ]

    # Phase 1.1: Frozen backbone training
    print("\nğŸ”’ Phase 1.1: Training with frozen backbone...")
    history_general_1 = general_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,  # More epochs for 5-class
        callbacks=callbacks_stage1,
        verbose=1
    )

    # Phase 1.2: Fine-tuning
    print("\nğŸ”“ Phase 1.2: Fine-tuning general model...")
    # Unfreeze top layers gradually
    general_model.layers[0].trainable = True

    # Only unfreeze last 30 layers for careful fine-tuning
    for layer in general_model.layers[0].layers[:-30]:
        layer.trainable = False

    trainable_count = sum([1 for layer in general_model.layers[0].layers if layer.trainable])
    print(f"Unfrozen layers: {trainable_count}/total layers")

    general_model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history_general_2 = general_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=callbacks_stage1,
        verbose=1
    )

    # ===== STAGE 2: Specialist Binary Model (Bacterial vs Viral) =====
    print("\nğŸ“� Stage 2: Training Specialist Model (Bacterial vs Viral ONLY)")
    print("Using InceptionV3 with dual pooling for optimal bacterial vs viral differentiation")

    # Create filtered dataset for bacterial vs viral only
    train_specialist, val_specialist = create_bacterial_viral_dataset(train_ds, val_ds)

    if train_specialist is not None and val_specialist is not None:
        specialist_model = system.build_specialist_model()
        specialist_model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"Specialist model built with {specialist_model.count_params():,} parameters")

        callbacks_stage2 = [
            EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
            ModelCheckpoint('specialist_bacterial_viral_best.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, verbose=1, min_lr=1e-7)
        ]

        # Phase 2.1: Frozen training
        print("\nğŸ¦  Phase 2.1: Training specialist with frozen InceptionV3...")
        history_specialist_1 = specialist_model.fit(
            train_specialist,
            validation_data=val_specialist,
            epochs=15,
            callbacks=callbacks_stage2,
            verbose=1
        )

        # Phase 2.2: Fine-tuning specialist
        print("\nâš¡ Phase 2.2: Fine-tuning specialist model...")
        specialist_model.layers[0].trainable = True

        # Unfreeze carefully - InceptionV3 is sensitive
        for layer in specialist_model.layers[0].layers[:-50]:
            layer.trainable = False

        specialist_trainable = sum([1 for layer in specialist_model.layers[0].layers if layer.trainable])
        print(f"Specialist unfrozen layers: {specialist_trainable}")

        specialist_model.compile(
            optimizer=optimizers.Adam(learning_rate=5e-6),  # Very low LR for specialist
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        history_specialist_2 = specialist_model.fit(
            train_specialist,
            validation_data=val_specialist,
            epochs=12,
            callbacks=callbacks_stage2,
            verbose=1
        )

        return system, general_model, specialist_model, (history_general_1, history_general_2, history_specialist_1, history_specialist_2)
    else:
        print("â�Œ No bacterial/viral samples found for specialist training")
        return system, general_model, None, (history_general_1, history_general_2, None, None)

# ========================
# 5. EVALUATION & INFERENCE
# ========================

def evaluate_enhanced_system(model, test_ds, specialist_model=None):
    """
    Comprehensive evaluation untuk 5-class system dengan focus khusus pada Bacterial vs Viral
    """
    class_names = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']

    print("ğŸ”� Evaluating Enhanced 5-Class Pneumonia Classification System...")
    print(f"Classes: {class_names}")

    # ===== GENERAL MODEL EVALUATION =====
    print("\nğŸ“Š GENERAL MODEL (5-Class) EVALUATION:")

    # Predictions
    y_true = []
    y_pred = []
    y_pred_proba = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_proba.extend(preds)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    y_pred_proba = np.array(y_pred_proba)

    # Overall Classification Report
    print("\nğŸ“‹ Overall Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('5-Class Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, [name.replace(' ', '\n') for name in class_names], rotation=0, fontsize=8)
    plt.yticks(tick_marks, [name.replace(' ', '\n') for name in class_names], fontsize=8)

    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10)

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # ===== BACTERIAL vs VIRAL ANALYSIS (KEY FOCUS) =====
    print("\nğŸ¦  BACTERIAL vs VIRAL PNEUMONIA DETAILED ANALYSIS:")

    bacterial_idx = 0  # Bacterial Pneumonia
    viral_idx = 4      # Viral Pneumonia

    # Extract bacterial vs viral cases
    bacterial_mask = np.array(y_true) == bacterial_idx
    viral_mask = np.array(y_true) == viral_idx

    bacterial_cases = np.sum(bacterial_mask)
    viral_cases = np.sum(viral_mask)

    print(f"Total Bacterial Pneumonia cases: {bacterial_cases}")
    print(f"Total Viral Pneumonia cases: {viral_cases}")

    if bacterial_cases > 0 and viral_cases > 0:
        # Bacterial performance
        bacterial_predicted_as_bacterial = cm[bacterial_idx, bacterial_idx]
        bacterial_predicted_as_viral = cm[bacterial_idx, viral_idx]

        # Viral performance
        viral_predicted_as_viral = cm[viral_idx, viral_idx]
        viral_predicted_as_bacterial = cm[viral_idx, bacterial_idx]

        # Precision and Recall
        bacterial_precision = cm[bacterial_idx, bacterial_idx] / (cm[:, bacterial_idx].sum() + 1e-10)
        viral_precision = cm[viral_idx, viral_idx] / (cm[:, viral_idx].sum() + 1e-10)

        bacterial_recall = cm[bacterial_idx, bacterial_idx] / (cm[bacterial_idx, :].sum() + 1e-10)
        viral_recall = cm[viral_idx, viral_idx] / (cm[viral_idx, :].sum() + 1e-10)

        bacterial_f1 = 2 * (bacterial_precision * bacterial_recall) / (bacterial_precision + bacterial_recall + 1e-10)
        viral_f1 = 2 * (viral_precision * viral_recall) / (viral_precision + viral_recall + 1e-10)

        print(f"\nğŸ“ˆ Bacterial Pneumonia Performance:")
        print(f"   Precision: {bacterial_precision:.3f}")
        print(f"   Recall: {bacterial_recall:.3f}")
        print(f"   F1-Score: {bacterial_f1:.3f}")
        print(f"   Correctly classified: {bacterial_predicted_as_bacterial}/{bacterial_cases}")
        print(f"   Confused with Viral: {bacterial_predicted_as_viral}")

        print(f"\nğŸ“ˆ Viral Pneumonia Performance:")
        print(f"   Precision: {viral_precision:.3f}")
        print(f"   Recall: {viral_recall:.3f}")
        print(f"   F1-Score: {viral_f1:.3f}")
        print(f"   Correctly classified: {viral_predicted_as_viral}/{viral_cases}")
        print(f"   Confused with Bacterial: {viral_predicted_as_bacterial}")

        # Confidence analysis
        bacterial_confidences = y_pred_proba[bacterial_mask][:, bacterial_idx]
        viral_confidences = y_pred_proba[viral_mask][:, viral_idx]

        print(f"\nğŸ�¯ Confidence Analysis:")
        print(f"   Bacterial Pneumonia - Mean Confidence: {np.mean(bacterial_confidences):.3f} (Â±{np.std(bacterial_confidences):.3f})")
        print(f"   Viral Pneumonia - Mean Confidence: {np.mean(viral_confidences):.3f} (Â±{np.std(viral_confidences):.3f})")

        # Bacterial vs Viral confusion subplot
        plt.subplot(2, 2, 2)
        bv_cm = np.array([[bacterial_predicted_as_bacterial, bacterial_predicted_as_viral],
                         [viral_predicted_as_bacterial, viral_predicted_as_viral]])
        plt.imshow(bv_cm, interpolation='nearest', cmap=plt.cm.Reds)
        plt.title('Bacterial vs Viral\nConfusion Matrix')
        plt.colorbar()
        plt.xticks([0, 1], ['Bacterial\nPredicted', 'Viral\nPredicted'])
        plt.yticks([0, 1], ['Bacterial\nTrue', 'Viral\nTrue'])

        for i, j in np.ndindex(bv_cm.shape):
            plt.text(j, i, format(bv_cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if bv_cm[i, j] > bv_cm.max()/2 else "black",
                    fontsize=12)

        # Confidence distribution plot
        plt.subplot(2, 2, 3)
        plt.hist(bacterial_confidences, bins=20, alpha=0.7, label=f'Bacterial (n={len(bacterial_confidences)})', color='red')
        plt.hist(viral_confidences, bins=20, alpha=0.7, label=f'Viral (n={len(viral_confidences)})', color='blue')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution\nBacterial vs Viral')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # ===== SPECIALIST MODEL EVALUATION (jika ada) =====
    if specialist_model is not None:
        print("\nğŸ�¯ SPECIALIST MODEL (Bacterial vs Viral Binary) EVALUATION:")

        # Create specialist test dataset
        test_specialist, _ = create_bacterial_viral_dataset(test_ds, test_ds)

        if test_specialist is not None:
            specialist_y_true = []
            specialist_y_pred = []
            specialist_y_pred_proba = []

            for images, labels in test_specialist:
                preds = specialist_model.predict(images, verbose=0)
                specialist_y_pred_proba.extend(preds)
                specialist_y_pred.extend(np.argmax(preds, axis=1))
                specialist_y_true.extend(labels.numpy())

            specialist_y_pred_proba = np.array(specialist_y_pred_proba)

            # Binary classification report
            binary_class_names = ['Bacterial Pneumonia', 'Viral Pneumonia']
            print("\nğŸ“‹ Specialist Binary Classification Report:")
            print(classification_report(specialist_y_true, specialist_y_pred,
                                      target_names=binary_class_names, digits=3))

            # Binary confusion matrix
            binary_cm = confusion_matrix(specialist_y_true, specialist_y_pred)
            plt.subplot(2, 2, 4)
            plt.imshow(binary_cm, interpolation='nearest', cmap=plt.cm.Greens)
            plt.title('Specialist Model\nBinary Classification')
            plt.colorbar()
            plt.xticks([0, 1], ['Bacterial', 'Viral'])
            plt.yticks([0, 1], ['Bacterial', 'Viral'])

            for i, j in np.ndindex(binary_cm.shape):
                plt.text(j, i, format(binary_cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if binary_cm[i, j] > binary_cm.max()/2 else "black",
                        fontsize=12)

            # Specialist confidence analysis
            specialist_bacterial_conf = specialist_y_pred_proba[np.array(specialist_y_true) == 0][:, 0]
            specialist_viral_conf = specialist_y_pred_proba[np.array(specialist_y_true) == 1][:, 1]

            print(f"\nğŸ�¯ Specialist Model Confidence:")
            print(f"   Bacterial: {np.mean(specialist_bacterial_conf):.3f} (Â±{np.std(specialist_bacterial_conf):.3f})")
            print(f"   Viral: {np.mean(specialist_viral_conf):.3f} (Â±{np.std(specialist_viral_conf):.3f})")

    plt.tight_layout()
    plt.show()

    # Summary
    print("\n" + "="*60)
    print("ğŸ�† SUMMARY - Enhanced Pneumonia Classification Performance")
    print("="*60)
    print(f"Overall 5-Class Accuracy: {np.mean(np.array(y_true) == np.array(y_pred)):.3f}")
    if bacterial_cases > 0 and viral_cases > 0:
        print(f"Bacterial Pneumonia F1-Score: {bacterial_f1:.3f}")
        print(f"Viral Pneumonia F1-Score: {viral_f1:.3f}")
        print(f"Bacterial vs Viral Differentiation: {((bacterial_predicted_as_bacterial + viral_predicted_as_viral) / (bacterial_cases + viral_cases)):.3f}")
        print("="*60)
        print(f"Precision: {viral_precision:.3f}, Recall: {viral_recall:.3f}")


        # Confidence analysis
        bacterial_confidences = y_pred_proba[np.array(y_true) == bacterial_idx][:, bacterial_idx]
        viral_confidences = y_pred_proba[np.array(y_true) == viral_idx][:, viral_idx]

        print(f"\nConfidence Analysis:")
        print(f"Bacterial Pneumonia - Mean Confidence: {np.mean(bacterial_confidences):.3f}")
        print(f"Viral Pneumonia - Mean Confidence: {np.mean(viral_confidences):.3f}")

# ========================
# 6. MAIN EXECUTION
# ========================

def plot_training_history(histories):
    """
    Plot training histories dari kedua stage
    """
    history_general_1, history_general_2, history_specialist_1, history_specialist_2 = histories

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # General Model Phase 1
    if history_general_1:
        axes[0,0].plot(history_general_1.history['accuracy'], label='Train Acc', color='blue')
        axes[0,0].plot(history_general_1.history['val_accuracy'], label='Val Acc', color='red')
        axes[0,0].set_title('General Model Phase 1\n(Frozen Backbone)')
        axes[0,0].legend()
        axes[0,0].grid(True)

        axes[1,0].plot(history_general_1.history['loss'], label='Train Loss', color='blue')
        axes[1,0].plot(history_general_1.history['val_loss'], label='Val Loss', color='red')
        axes[1,0].set_title('General Model Phase 1 Loss')
        axes[1,0].legend()
        axes[1,0].grid(True)

    # General Model Phase 2
    if history_general_2:
        axes[0,1].plot(history_general_2.history['accuracy'], label='Train Acc', color='blue')
        axes[0,1].plot(history_general_2.history['val_accuracy'], label='Val Acc', color='red')
        axes[0,1].set_title('General Model Phase 2\n(Fine-tuning)')
        axes[0,1].legend()
        axes[0,1].grid(True)

        axes[1,1].plot(history_general_2.history['loss'], label='Train Loss', color='blue')
        axes[1,1].plot(history_general_2.history['val_loss'], label='Val Loss', color='red')
        axes[1,1].set_title('General Model Phase 2 Loss')
        axes[1,1].legend()
        axes[1,1].grid(True)

    # Specialist Model (jika ada)
    if history_specialist_1 and history_specialist_2:
        # Combined specialist history
        combined_acc = history_specialist_1.history['accuracy'] + history_specialist_2.history['accuracy']
        combined_val_acc = history_specialist_1.history['val_accuracy'] + history_specialist_2.history['val_accuracy']
        combined_loss = history_specialist_1.history['loss'] + history_specialist_2.history['loss']
        combined_val_loss = history_specialist_1.history['val_loss'] + history_specialist_2.history['val_loss']

        axes[0,2].plot(combined_acc, label='Train Acc', color='green')
        axes[0,2].plot(combined_val_acc, label='Val Acc', color='orange')
        axes[0,2].axvline(x=len(history_specialist_1.history['accuracy']), color='black', linestyle='--', alpha=0.7, label='Fine-tuning Start')
        axes[0,2].set_title('Specialist Model\n(Bacterial vs Viral)')
        axes[0,2].legend()
        axes[0,2].grid(True)

        axes[1,2].plot(combined_loss, label='Train Loss', color='green')
        axes[1,2].plot(combined_val_loss, label='Val Loss', color='orange')
        axes[1,2].axvline(x=len(history_specialist_1.history['loss']), color='black', linestyle='--', alpha=0.7, label='Fine-tuning Start')
        axes[1,2].set_title('Specialist Model Loss')
        axes[1,2].legend()
        axes[1,2].grid(True)
    else:
        axes[0,2].text(0.5, 0.5, 'Specialist Model\nNot Trained', ha='center', va='center', transform=axes[0,2].transAxes)
        axes[1,2].text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=axes[1,2].transAxes)

    plt.tight_layout()
    plt.show()

def main():
    """
    Main execution pipeline untuk 5-class pneumonia classification
    """
    print("ğŸš€ Enhanced 5-Class Pneumonia Classification System")
    print("=" * 60)
    print("Classes Mapping:")
    print("  [0] Bacterial Pneumonia")
    print("  [1] Corona Virus Disease (COVID-19)")
    print("  [2] Normal")
    print("  [3] Tuberculosis")
    print("  [4] Viral Pneumonia")
    print("=" * 60)

    # Setup GPU dan Mixed Precision
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("âœ… GPU detected:", gpus)
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("âœ… GPU memory growth enabled")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("âš ï¸� No GPU detected. Training akan menggunakan CPU (slower)")

    # Enable mixed precision untuk GPU optimization
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("âœ… Mixed precision enabled for faster training")

    print("\nğŸ”§ Key System Enhancements:")
    print("1. ğŸ“¸ CLAHE + Sharpening preprocessing untuk X-ray contrast optimization")
    print("2. âš¡ Dual Pooling Strategy:")
    print("   - Max Pooling: Deteksi pola padat (Bacterial Pneumonia)")
    print("   - Average Pooling: Deteksi pola menyebar (Viral Pneumonia)")
    print("   - Weight: 0.5 + 0.5 combination")
    print("3. ğŸ�—ï¸� Two-Stage Architecture:")
    print("   - General Model: ResNet50 untuk 5-class classification")
    print("   - Specialist Model: InceptionV3 untuk Bacterial vs Viral binary")
    print("4. ğŸ“Š Comprehensive evaluation dengan Bacterial vs Viral focus analysis")
    print("5. ğŸ�¯ Enhanced training pipeline dengan careful fine-tuning")

    print("\n" + "="*60)
    print("ğŸ“‹ USAGE INSTRUCTIONS:")
    print("="*60)
    print("1. Load your dataset dengan format yang sesuai:")
    print("   train_ds = tf.keras.preprocessing.image_dataset_from_directory(...)")
    print("   val_ds = tf.keras.preprocessing.image_dataset_from_directory(...)")
    print("")
    print("2. Uncomment dan jalankan training pipeline:")
    print("   system, general_model, specialist_model, histories = train_two_stage_system(train_ds, val_ds)")
    print("")
    print("3. Evaluate models:")
    print("   evaluate_enhanced_system(general_model, val_ds, specialist_model)")
    print("   plot_training_history(histories)")
    print("")
    print("4. Save models:")
    print("   general_model.save('enhanced_5class_pneumonia_model.h5')")
    print("   if specialist_model:")
    print("       specialist_model.save('bacterial_viral_specialist_model.h5')")

    print("\nğŸ’¡ EXPECTED IMPROVEMENTS:")
    print("- Enhanced Bacterial vs Viral differentiation accuracy")
    print("- Better confidence scores untuk pneumonia predictions")
    print("- Reduced confusion antara Bacterial dan Viral pneumonia")
    print("- Overall improved 5-class classification performance")
    print("- Robust performance across different X-ray image qualities")

    print("\nâœ… System ready untuk deployment!")

    # Example training pipeline (uncomment to use)

    # Apply enhanced preprocessing
    train_datagen, val_datagen = create_enhanced_data_generators()

    # Train the system
    system, general_model, specialist_model, histories = train_two_stage_system(train_ds, val_ds)

    # Evaluate
    evaluate_enhanced_system(general_model, val_ds, specialist_model)
    plot_training_history(histories)

    # Save models
    general_model.save('enhanced_5class_pneumonia_model.h5')
    if specialist_model:
        specialist_model.save('bacterial_viral_specialist_model.h5')

    print("ğŸ�‰ Training dan evaluation selesai!")
if __name__ == "__main__":
    main()


import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, DenseNet121
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import cv2

# ========================
# 1. ENHANCED PREPROCESSING
# ========================

class EnhancedPreprocessing:
    """
    Optimalisasi preprocessing khusus untuk pneumonia classification
    Focus: pencahayaan, ketajaman, dan kontras untuk membedakan pola bakterial vs viral
    """

    @staticmethod
    def enhance_image(image):
        """
        Enhanced preprocessing untuk pneumonia X-ray
        - CLAHE untuk kontras adaptif
        - Gaussian blur removal untuk ketajaman
        - Histogram equalization untuk pencahayaan optimal
        """
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Bagus untuk enhance perbedaan antara area padat dan menyebar
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        if len(image.shape) == 3:
            # RGB image
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale
            enhanced = clahe.apply(image)
            if len(enhanced.shape) == 2:
                enhanced = np.expand_dims(enhanced, axis=-1)
                enhanced = np.repeat(enhanced, 3, axis=-1)

        # Sharpening kernel untuk mempertegas batas antara area normal dan infected
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)

        # Normalize back to [0,1]
        enhanced = enhanced.astype(np.float32) / 255.0

        return enhanced

def create_enhanced_data_generators():
    """
    Data augmentation yang dioptimalkan untuk pneumonia
    Focus: variasi pencahayaan dan rotasi kecil (sesuai kondisi X-ray)
    """
    train_datagen = ImageDataGenerator(
        preprocessing_function=EnhancedPreprocessing.enhance_image,
        rotation_range=10,  # Rotasi kecil saja (X-ray biasanya standar orientasi)
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2],  # Variasi pencahayaan
        zoom_range=0.1,
        horizontal_flip=False,  # X-ray tidak boleh flip horizontal
        fill_mode='nearest',
        validation_split=0.2
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=EnhancedPreprocessing.enhance_image,
        validation_split=0.2
    )

    return train_datagen, val_datagen

# ========================
# 2. DUAL POOLING STRATEGY
# ========================

class DualPoolingLayer(layers.Layer):
    """
    Kombinasi Max Pooling + Global Average Pooling
    - Max Pooling: menangkap pola padat/intens (bacterial pneumonia)
    - Global Average Pooling: menangkap pola menyebar (viral pneumonia)
    Weight: 0.5 untuk setiap pooling strategy
    """

    def __init__(self, **kwargs):
        super(DualPoolingLayer, self).__init__(**kwargs)
        self.max_pool = layers.GlobalMaxPooling2D()
        self.avg_pool = layers.GlobalAveragePooling2D()

    def call(self, inputs):
        max_features = self.max_pool(inputs)
        avg_features = self.avg_pool(inputs)

        # Combine dengan weight 0.5 masing-masing
        combined = 0.5 * max_features + 0.5 * avg_features
        return combined

    def get_config(self):
        return super(DualPoolingLayer, self).get_config()

# ========================
# 3. VISION TRANSFORMER IMPLEMENTATION
# ========================

class VisionTransformer(layers.Layer):
    """
    Vision Transformer implementation untuk specialist model
    Menggantikan InceptionV3 dengan ViT architecture
    """
    def __init__(self, image_size=224, patch_size=16, num_layers=12, d_model=768,
                 num_heads=12, mlp_dim=3072, dropout_rate=0.1, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate

        # Patch embedding
        self.patch_embedding = layers.Conv2D(
            filters=d_model,
            kernel_size=patch_size,
            strides=patch_size,
            padding='valid',
            name='patch_embedding'
        )

        # Class token
        self.class_token = self.add_weight(
            shape=(1, 1, d_model),
            initializer='random_normal',
            trainable=True,
            name='class_token'
        )

        # Position embeddings
        self.position_embedding = self.add_weight(
            shape=(1, self.num_patches + 1, d_model),
            initializer='random_normal',
            trainable=True,
            name='position_embedding'
        )

        # Transformer blocks
        self.transformer_blocks = []
        for i in range(num_layers):
            self.transformer_blocks.append([
                layers.LayerNormalization(epsilon=1e-6, name=f'ln1_{i}'),
                layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads,
                                        dropout=dropout_rate, name=f'mha_{i}'),
                layers.Dropout(dropout_rate, name=f'dropout1_{i}'),
                layers.LayerNormalization(epsilon=1e-6, name=f'ln2_{i}'),
                layers.Dense(mlp_dim, activation='gelu', name=f'mlp1_{i}'),
                layers.Dropout(dropout_rate, name=f'dropout2_{i}'),
                layers.Dense(d_model, name=f'mlp2_{i}'),
                layers.Dropout(dropout_rate, name=f'dropout3_{i}')
            ])

        # Final layer norm
        self.final_ln = layers.LayerNormalization(epsilon=1e-6, name='final_ln')

    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]

        # Patch embedding
        patches = self.patch_embedding(inputs)  # (batch, h//p, w//p, d_model)
        patches = tf.reshape(patches, [batch_size, self.num_patches, self.d_model])

        # Add class token
        class_tokens = tf.broadcast_to(self.class_token, [batch_size, 1, self.d_model])
        patches = tf.concat([class_tokens, patches], axis=1)

        # Add position embedding
        patches = patches + self.position_embedding

        # Apply transformer blocks
        for block in self.transformer_blocks:
            ln1, mha, dropout1, ln2, mlp1, dropout2, mlp2, dropout3 = block

            # Multi-head attention
            attn_input = ln1(patches)
            attn_output = mha(attn_input, attn_input, training=training)
            attn_output = dropout1(attn_output, training=training)
            patches = patches + attn_output

            # MLP
            mlp_input = ln2(patches)
            mlp_output = mlp1(mlp_input)
            mlp_output = dropout2(mlp_output, training=training)
            mlp_output = mlp2(mlp_output)
            mlp_output = dropout3(mlp_output, training=training)
            patches = patches + mlp_output

        # Final layer norm and extract class token
        patches = self.final_ln(patches)
        class_token_output = patches[:, 0]  # Extract class token

        return class_token_output

    def get_config(self):
        config = super().get_config()
        config.update({
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'd_model': self.d_model,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'mlp_dim': self.mlp_dim,
            'dropout_rate': self.dropout_rate
        })
        return config

# ========================
# 3. TWO-STAGE MODEL ARCHITECTURE
# ========================

class PneumoniaClassificationSystem:
    def __init__(self, num_classes=5):  # 5 classes: BACTERIAL, COVID, NORMAL, TB, VIRAL
        self.num_classes = num_classes
        self.class_names = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']
        self.general_model = None
        self.specialist_model = None

    def build_general_model(self, backbone_name='densenet121'):
        """
        Model 1: General 5-class classifier dengan additional dropout
        Classes: [0]Bacterial, [1]COVID, [2]Normal, [3]TB, [4]Viral
        Focus: Enhanced differentiation untuk Bacterial (0) vs Viral (4) Pneumonia
        Backbone options: resnet50, resnet101, densenet121, mobilenetv2
        """
        if backbone_name.lower() == 'resnet50':
            backbone = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        elif backbone_name.lower() == 'resnet101':
            backbone = tf.keras.applications.ResNet101(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        elif backbone_name.lower() == 'densenet121':
            backbone = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        elif backbone_name.lower() == 'mobilenetv2':
            backbone = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        else:
            backbone = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Freeze backbone initially
        backbone.trainable = False

        # Build general model dengan additional dropout
        self.general_model = models.Sequential([
            backbone,
            DualPoolingLayer(),  # Dual pooling strategy
            layers.Dropout(0.4),  # Additional dropout after pooling
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.4),  # Additional dropout
            layers.Dense(128, activation='relu'),  # Additional dense layer
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        return self.general_model

    def build_specialist_model(self):
        """
        Model 2: Specialist untuk BACTERIAL (0) vs VIRAL (4) Pneumonia ONLY
        Menggunakan Vision Transformer (ViT) menggantikan InceptionV3
        Enhanced architecture khusus untuk membedakan pola bakterial vs viral
        """
        # Build ViT-based specialist model
        inputs = layers.Input(shape=(224, 224, 3))

        # Vision Transformer with increased dropout
        vit_features = VisionTransformer(
            image_size=224,
            patch_size=8,
            num_layers=8,  # Smaller ViT for medical images
            d_model=512,
            num_heads=8,
            mlp_dim=2048,
            dropout_rate=0.2  # Higher dropout in ViT
        )(inputs)

        # Additional dropout layers
        x = layers.Dropout(0.6)(vit_features)  # Higher dropout
        x = layers.Dense(1024, activation='relu', name='specialist_dense_1')(x)
        x = layers.Dropout(0.6)(x)
        x = layers.Dense(512, activation='relu', name='specialist_dense_2')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu', name='specialist_dense_3')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation='relu', name='specialist_dense_4')(x)  # Additional layer
        x = layers.Dropout(0.3)(x)
        # Output layer untuk binary classification: Bacterial vs Viral
        outputs = layers.Dense(2, activation='softmax', name='specialist_output')(x)  # [Bacterial, Viral] only

        self.specialist_model = models.Model(inputs=inputs, outputs=outputs)

        return self.specialist_model

def create_bacterial_viral_dataset(train_ds, val_ds):
    """
    Filter dataset untuk specialist model (Bacterial vs Viral Pneumonia only)
    Index 0: Bacterial Pneumonia -> Label 0
    Index 4: Viral Pneumonia -> Label 1
    """
    def filter_and_relabel(dataset):
        filtered_images = []
        filtered_labels = []

        for images, labels in dataset:
            # Convert to numpy for easier processing
            images_np = images.numpy()
            labels_np = labels.numpy()

            # Filter untuk bacterial (0) dan viral (4) pneumonia
            mask = (labels_np == 0) | (labels_np == 4)

            if np.any(mask):
                selected_images = images_np[mask]
                selected_labels = labels_np[mask]

                # Relabel: Bacterial (0) -> 0, Viral (4) -> 1
                selected_labels[selected_labels == 4] = 1

                filtered_images.extend(selected_images)
                filtered_labels.extend(selected_labels)

        # Convert back to tensorflow dataset
        if filtered_images:
            filtered_images = np.array(filtered_images)
            filtered_labels = np.array(filtered_labels)

            filtered_ds = tf.data.Dataset.from_tensor_slices((filtered_images, filtered_labels))
            filtered_ds = filtered_ds.batch(32).prefetch(tf.data.AUTOTUNE)
            return filtered_ds
        else:
            return None

    train_specialist = filter_and_relabel(train_ds)
    val_specialist = filter_and_relabel(val_ds)

    print(f"âœ… Specialist dataset created for Bacterial vs Viral classification")
    return train_specialist, val_specialist

def train_two_stage_system(train_ds, val_ds):
    """
    Two-stage training pipeline for 5-class pneumonia classification
    Stage 1: General 5-class model dengan focus pada Bacterial vs Viral
    Stage 2: Specialist ViT model (Bacterial vs Viral only)
    """
    class_names = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']
    print("ğŸ”¥ Building Enhanced 5-Class Pneumonia Classification System...")
    print(f"Classes: {class_names}")

    # Initialize system
    system = PneumoniaClassificationSystem(num_classes=5)

    # ===== STAGE 1: General 5-Class Model =====
    print("\nğŸ“� Stage 1: Training General 5-Class Model with Additional Dropout")
    print("Focus: Enhanced Bacterial vs Viral differentiation with dual pooling")

    general_model = system.build_general_model('resnet50')
    general_model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"Model built with {general_model.count_params():,} parameters")

    # Callbacks for stage 1
    callbacks_stage1 = [
        EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1),
        ModelCheckpoint('general_model_5class_best.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, verbose=1, min_lr=1e-7)
    ]

    # Phase 1.1: Frozen backbone training
    print("\nğŸ”’ Phase 1.1: Training with frozen backbone...")
    history_general_1 = general_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,  # More epochs for 5-class
        callbacks=callbacks_stage1,
        verbose=1
    )

    # Phase 1.2: Fine-tuning
    print("\nğŸ”“ Phase 1.2: Fine-tuning general model...")
    # Unfreeze top layers gradually
    general_model.layers[0].trainable = True

    # Only unfreeze last 30 layers for careful fine-tuning
    for layer in general_model.layers[0].layers[:-30]:
        layer.trainable = False

    trainable_count = sum([1 for layer in general_model.layers[0].layers if layer.trainable])
    print(f"Unfrozen layers: {trainable_count}/total layers")

    general_model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history_general_2 = general_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=callbacks_stage1,
        verbose=1
    )

    # ===== STAGE 2: Specialist ViT Model (Bacterial vs Viral) =====
    print("\nğŸ“� Stage 2: Training Specialist ViT Model (Bacterial vs Viral ONLY)")
    print("Using Vision Transformer with enhanced dropout for optimal bacterial vs viral differentiation")

    # Create filtered dataset for bacterial vs viral only
    train_specialist, val_specialist = create_bacterial_viral_dataset(train_ds, val_ds)

    if train_specialist is not None and val_specialist is not None:
        specialist_model = system.build_specialist_model()
        specialist_model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"Specialist ViT model built with {specialist_model.count_params():,} parameters")

        callbacks_stage2 = [
            EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
            ModelCheckpoint('specialist_bacterial_viral_vit_best.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, verbose=1, min_lr=1e-7)
        ]

        # Phase 2.1: Initial training
        print("\nğŸ¦  Phase 2.1: Training specialist ViT model...")
        history_specialist_1 = specialist_model.fit(
            train_specialist,
            validation_data=val_specialist,
            epochs=20,  # More epochs for ViT
            callbacks=callbacks_stage2,
            verbose=1
        )

        # Phase 2.2: Fine-tuning specialist with lower dropout
        print("\nâš¡ Phase 2.2: Fine-tuning specialist ViT model with reduced dropout...")

        # Reduce dropout rates for fine-tuning
        for layer in specialist_model.layers:
            if isinstance(layer, layers.Dropout):
                layer.rate = layer.rate * 0.7  # Reduce dropout by 30%

        specialist_model.compile(
            optimizer=optimizers.Adam(learning_rate=5e-6),  # Very low LR for ViT fine-tuning
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        history_specialist_2 = specialist_model.fit(
            train_specialist,
            validation_data=val_specialist,
            epochs=15,
            callbacks=callbacks_stage2,
            verbose=1
        )

        return system, general_model, specialist_model, (history_general_1, history_general_2, history_specialist_1, history_specialist_2)
    else:
        print("â�Œ No bacterial/viral samples found for specialist training")
        return system, general_model, None, (history_general_1, history_general_2, None, None)

# ========================
# 5. EVALUATION & INFERENCE
# ========================

def evaluate_enhanced_system(model, test_ds, specialist_model=None):
    """
    Comprehensive evaluation untuk 5-class system dengan focus khusus pada Bacterial vs Viral
    """
    class_names = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']

    print("ğŸ”� Evaluating Enhanced 5-Class Pneumonia Classification System...")
    print(f"Classes: {class_names}")

    # ===== GENERAL MODEL EVALUATION =====
    print("\nğŸ“Š GENERAL MODEL (5-Class) EVALUATION:")

    # Predictions
    y_true = []
    y_pred = []
    y_pred_proba = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_proba.extend(preds)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    y_pred_proba = np.array(y_pred_proba)

    # Overall Classification Report
    print("\nğŸ“‹ Overall Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('5-Class Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, [name.replace(' ', '\n') for name in class_names], rotation=0, fontsize=8)
    plt.yticks(tick_marks, [name.replace(' ', '\n') for name in class_names], fontsize=8)

    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10)

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # ===== BACTERIAL vs VIRAL ANALYSIS (KEY FOCUS) =====
    print("\nğŸ¦  BACTERIAL vs VIRAL PNEUMONIA DETAILED ANALYSIS:")

    bacterial_idx = 0  # Bacterial Pneumonia
    viral_idx = 4      # Viral Pneumonia

    # Extract bacterial vs viral cases
    bacterial_mask = np.array(y_true) == bacterial_idx
    viral_mask = np.array(y_true) == viral_idx

    bacterial_cases = np.sum(bacterial_mask)
    viral_cases = np.sum(viral_mask)

    print(f"Total Bacterial Pneumonia cases: {bacterial_cases}")
    print(f"Total Viral Pneumonia cases: {viral_cases}")

    if bacterial_cases > 0 and viral_cases > 0:
        # Bacterial performance
        bacterial_predicted_as_bacterial = cm[bacterial_idx, bacterial_idx]
        bacterial_predicted_as_viral = cm[bacterial_idx, viral_idx]

        # Viral performance
        viral_predicted_as_viral = cm[viral_idx, viral_idx]
        viral_predicted_as_bacterial = cm[viral_idx, bacterial_idx]

        # Precision and Recall
        bacterial_precision = cm[bacterial_idx, bacterial_idx] / (cm[:, bacterial_idx].sum() + 1e-10)
        viral_precision = cm[viral_idx, viral_idx] / (cm[:, viral_idx].sum() + 1e-10)

        bacterial_recall = cm[bacterial_idx, bacterial_idx] / (cm[bacterial_idx, :].sum() + 1e-10)
        viral_recall = cm[viral_idx, viral_idx] / (cm[viral_idx, :].sum() + 1e-10)

        bacterial_f1 = 2 * (bacterial_precision * bacterial_recall) / (bacterial_precision + bacterial_recall + 1e-10)
        viral_f1 = 2 * (viral_precision * viral_recall) / (viral_precision + viral_recall + 1e-10)

        print(f"\nğŸ“ˆ Bacterial Pneumonia Performance:")
        print(f"   Precision: {bacterial_precision:.3f}")
        print(f"   Recall: {bacterial_recall:.3f}")
        print(f"   F1-Score: {bacterial_f1:.3f}")
        print(f"   Correctly classified: {bacterial_predicted_as_bacterial}/{bacterial_cases}")
        print(f"   Confused with Viral: {bacterial_predicted_as_viral}")

        print(f"\nğŸ“ˆ Viral Pneumonia Performance:")
        print(f"   Precision: {viral_precision:.3f}")
        print(f"   Recall: {viral_recall:.3f}")
        print(f"   F1-Score: {viral_f1:.3f}")
        print(f"   Correctly classified: {viral_predicted_as_viral}/{viral_cases}")
        print(f"   Confused with Bacterial: {viral_predicted_as_bacterial}")

        # Confidence analysis
        bacterial_confidences = y_pred_proba[bacterial_mask][:, bacterial_idx]
        viral_confidences = y_pred_proba[viral_mask][:, viral_idx]

        print(f"\nğŸ�¯ Confidence Analysis:")
        print(f"   Bacterial Pneumonia - Mean Confidence: {np.mean(bacterial_confidences):.3f} (Â±{np.std(bacterial_confidences):.3f})")
        print(f"   Viral Pneumonia - Mean Confidence: {np.mean(viral_confidences):.3f} (Â±{np.std(viral_confidences):.3f})")

        # Bacterial vs Viral confusion subplot
        plt.subplot(2, 2, 2)
        bv_cm = np.array([[bacterial_predicted_as_bacterial, bacterial_predicted_as_viral],
                         [viral_predicted_as_bacterial, viral_predicted_as_viral]])
        plt.imshow(bv_cm, interpolation='nearest', cmap=plt.cm.Reds)
        plt.title('Bacterial vs Viral\nConfusion Matrix')
        plt.colorbar()
        plt.xticks([0, 1], ['Bacterial\nPredicted', 'Viral\nPredicted'])
        plt.yticks([0, 1], ['Bacterial\nTrue', 'Viral\nTrue'])

        for i, j in np.ndindex(bv_cm.shape):
            plt.text(j, i, format(bv_cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if bv_cm[i, j] > bv_cm.max()/2 else "black",
                    fontsize=12)

        # Confidence distribution plot
        plt.subplot(2, 2, 3)
        plt.hist(bacterial_confidences, bins=20, alpha=0.7, label=f'Bacterial (n={len(bacterial_confidences)})', color='red')
        plt.hist(viral_confidences, bins=20, alpha=0.7, label=f'Viral (n={len(viral_confidences)})', color='blue')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution\nBacterial vs Viral')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # ===== SPECIALIST ViT MODEL EVALUATION (jika ada) =====
    if specialist_model is not None:
        print("\nğŸ�¯ SPECIALIST ViT MODEL (Bacterial vs Viral Binary) EVALUATION:")

        # Create specialist test dataset
        test_specialist, _ = create_bacterial_viral_dataset(test_ds, test_ds)

        if test_specialist is not None:
            specialist_y_true = []
            specialist_y_pred = []
            specialist_y_pred_proba = []

            for images, labels in test_specialist:
                preds = specialist_model.predict(images, verbose=0)
                specialist_y_pred_proba.extend(preds)
                specialist_y_pred.extend(np.argmax(preds, axis=1))
                specialist_y_true.extend(labels.numpy())

            specialist_y_pred_proba = np.array(specialist_y_pred_proba)

            # Binary classification report
            binary_class_names = ['Bacterial Pneumonia', 'Viral Pneumonia']
            print("\nğŸ“‹ Specialist ViT Binary Classification Report:")
            print(classification_report(specialist_y_true, specialist_y_pred,
                                      target_names=binary_class_names, digits=3))

            # Binary confusion matrix
            binary_cm = confusion_matrix(specialist_y_true, specialist_y_pred)
            plt.subplot(2, 2, 4)
            plt.imshow(binary_cm, interpolation='nearest', cmap=plt.cm.Greens)
            plt.title('Specialist ViT Model\nBinary Classification')
            plt.colorbar()
            plt.xticks([0, 1], ['Bacterial', 'Viral'])
            plt.yticks([0, 1], ['Bacterial', 'Viral'])

            for i, j in np.ndindex(binary_cm.shape):
                plt.text(j, i, format(binary_cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if binary_cm[i, j] > binary_cm.max()/2 else "black",
                        fontsize=12)

            # Specialist confidence analysis
            specialist_bacterial_conf = specialist_y_pred_proba[np.array(specialist_y_true) == 0][:, 0]
            specialist_viral_conf = specialist_y_pred_proba[np.array(specialist_y_true) == 1][:, 1]

            print(f"\nğŸ�¯ Specialist ViT Model Confidence:")
            print(f"   Bacterial: {np.mean(specialist_bacterial_conf):.3f} (Â±{np.std(specialist_bacterial_conf):.3f})")
            print(f"   Viral: {np.mean(specialist_viral_conf):.3f} (Â±{np.std(specialist_viral_conf):.3f})")

    plt.tight_layout()
    plt.show()

    # Summary
    print("\n" + "="*60)
    print("ğŸ�† SUMMARY - Enhanced Pneumonia Classification Performance")
    print("="*60)
    print(f"Overall 5-Class Accuracy: {np.mean(np.array(y_true) == np.array(y_pred)):.3f}")
    if bacterial_cases > 0 and viral_cases > 0:
        print(f"Bacterial Pneumonia F1-Score: {bacterial_f1:.3f}")
        print(f"Viral Pneumonia F1-Score: {viral_f1:.3f}")
        print(f"Bacterial vs Viral Differentiation: {((bacterial_predicted_as_bacterial + viral_predicted_as_viral) / (bacterial_cases + viral_cases)):.3f}")
        print("="*60)
        print(f"Precision: {viral_precision:.3f}, Recall: {viral_recall:.3f}")


        # Confidence analysis
        bacterial_confidences = y_pred_proba[np.array(y_true) == bacterial_idx][:, bacterial_idx]
        viral_confidences = y_pred_proba[np.array(y_true) == viral_idx][:, viral_idx]

        print(f"\nConfidence Analysis:")
        print(f"Bacterial Pneumonia - Mean Confidence: {np.mean(bacterial_confidences):.3f}")
        print(f"Viral Pneumonia - Mean Confidence: {np.mean(viral_confidences):.3f}")

# ========================
# 6. MAIN EXECUTION
# ========================

def plot_training_history(histories):
    """
    Plot training histories dari kedua stage
    """
    history_general_1, history_general_2, history_specialist_1, history_specialist_2 = histories

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # General Model Phase 1
    if history_general_1:
        axes[0,0].plot(history_general_1.history['accuracy'], label='Train Acc', color='blue')
        axes[0,0].plot(history_general_1.history['val_accuracy'], label='Val Acc', color='red')
        axes[0,0].set_title('General Model Phase 1\n(Frozen Backbone)')
        axes[0,0].legend()
        axes[0,0].grid(True)

        axes[1,0].plot(history_general_1.history['loss'], label='Train Loss', color='blue')
        axes[1,0].plot(history_general_1.history['val_loss'], label='Val Loss', color='red')
        axes[1,0].set_title('General Model Phase 1 Loss')
        axes[1,0].legend()
        axes[1,0].grid(True)

    # General Model Phase 2
    if history_general_2:
        axes[0,1].plot(history_general_2.history['accuracy'], label='Train Acc', color='blue')
        axes[0,1].plot(history_general_2.history['val_accuracy'], label='Val Acc', color='red')
        axes[0,1].set_title('General Model Phase 2\n(Fine-tuning)')
        axes[0,1].legend()
        axes[0,1].grid(True)

        axes[1,1].plot(history_general_2.history['loss'], label='Train Loss', color='blue')
        axes[1,1].plot(history_general_2.history['val_loss'], label='Val Loss', color='red')
        axes[1,1].set_title('General Model Phase 2 Loss')
        axes[1,1].legend()
        axes[1,1].grid(True)

    # Specialist Model (jika ada)
    if history_specialist_1 and history_specialist_2:
        # Combined specialist history
        combined_acc = history_specialist_1.history['accuracy'] + history_specialist_2.history['accuracy']
        combined_val_acc = history_specialist_1.history['val_accuracy'] + history_specialist_2.history['val_accuracy']
        combined_loss = history_specialist_1.history['loss'] + history_specialist_2.history['loss']
        combined_val_loss = history_specialist_1.history['val_loss'] + history_specialist_2.history['val_loss']

        axes[0,2].plot(combined_acc, label='Train Acc', color='green')
        axes[0,2].plot(combined_val_acc, label='Val Acc', color='orange')
        axes[0,2].axvline(x=len(history_specialist_1.history['accuracy']), color='black', linestyle='--', alpha=0.7, label='Fine-tuning Start')
        axes[0,2].set_title('Specialist ViT Model\n(Bacterial vs Viral)')
        axes[0,2].legend()
        axes[0,2].grid(True)

        axes[1,2].plot(combined_loss, label='Train Loss', color='green')
        axes[1,2].plot(combined_val_loss, label='Val Loss', color='orange')
        axes[1,2].axvline(x=len(history_specialist_1.history['loss']), color='black', linestyle='--', alpha=0.7, label='Fine-tuning Start')
        axes[1,2].set_title('Specialist ViT Model Loss')
        axes[1,2].legend()
        axes[1,2].grid(True)
    else:
        axes[0,2].text(0.5, 0.5, 'Specialist ViT Model\nNot Trained', ha='center', va='center', transform=axes[0,2].transAxes)
        axes[1,2].text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=axes[1,2].transAxes)

    plt.tight_layout()
    plt.show()

def main():
    """
    Main execution pipeline untuk 5-class pneumonia classification dengan ViT
    """
    print("ğŸš€ Enhanced 5-Class Pneumonia Classification System with Vision Transformer")
    print("=" * 60)
    print("Classes Mapping:")
    print("  [0] Bacterial Pneumonia")
    print("  [1] Corona Virus Disease (COVID-19)")
    print("  [2] Normal")
    print("  [3] Tuberculosis")
    print("  [4] Viral Pneumonia")
    print("=" * 60)

    # Setup GPU dan Mixed Precision
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("âœ… GPU detected:", gpus)
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("âœ… GPU memory growth enabled")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("âš ï¸� No GPU detected. Training akan menggunakan CPU (slower)")

    # Enable mixed precision untuk GPU optimization
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("âœ… Mixed precision enabled for faster training")

    print("\nğŸ”§ Key System Enhancements:")
    print("1. ğŸ“¸ CLAHE + Sharpening preprocessing untuk X-ray contrast optimization")
    print("2. âš¡ Dual Pooling Strategy:")
    print("   - Max Pooling: Deteksi pola padat (Bacterial Pneumonia)")
    print("   - Average Pooling: Deteksi pola menyebar (Viral Pneumonia)")
    print("   - Weight: 0.5 + 0.5 combination")
    print("3. ğŸ�—ï¸� Two-Stage Architecture:")
    print("   - General Model: ResNet50 dengan additional dropout untuk 5-class classification")
    print("   - Specialist Model: Vision Transformer (ViT) dengan enhanced dropout untuk Bacterial vs Viral binary")
    print("4. ğŸ§  Vision Transformer Features:")
    print("   - Patch-based attention mechanism untuk better feature extraction")
    print("   - Multi-head self-attention untuk capturing complex patterns")
    print("   - Enhanced dropout (0.2 dalam ViT + 0.3-0.6 dalam classifier layers)")
    print("5. ğŸ“Š Comprehensive evaluation dengan Bacterial vs Viral focus analysis")
    print("6. ğŸ�¯ Enhanced training pipeline dengan careful fine-tuning")

    print("\n" + "="*60)
    print("ğŸ“‹ USAGE INSTRUCTIONS:")
    print("="*60)
    print("1. Load your dataset dengan format yang sesuai:")
    print("   train_ds = tf.keras.preprocessing.image_dataset_from_directory(...)")
    print("   val_ds = tf.keras.preprocessing.image_dataset_from_directory(...)")
    print("")
    print("2. Uncomment dan jalankan training pipeline:")
    print("   system, general_model, specialist_model, histories = train_two_stage_system(train_ds, val_ds)")
    print("")
    print("3. Evaluate models:")
    print("   evaluate_enhanced_system(general_model, val_ds, specialist_model)")
    print("   plot_training_history(histories)")
    print("")
    print("4. Save models:")
    print("   general_model.save('enhanced_5class_pneumonia_model_with_dropout.h5')")
    print("   if specialist_model:")
    print("       specialist_model.save('bacterial_viral_vit_specialist_model.h5')")

    print("\nğŸ’¡ EXPECTED IMPROVEMENTS with ViT and Additional Dropout:")
    print("- Better regularization dan reduced overfitting dengan enhanced dropout")
    print("- Improved attention-based feature extraction untuk medical imaging")
    print("- Enhanced Bacterial vs Viral differentiation accuracy dengan ViT")
    print("- Better confidence scores dan calibration")
    print("- More robust performance across different X-ray image qualities")
    print("- Reduced confusion antara Bacterial dan Viral pneumonia")
    print("- Overall improved 5-class classification performance")

    print("\nâœ… Enhanced System dengan ViT ready untuk deployment!")

    # Example training pipeline (uncomment to use)
    # Apply enhanced preprocessing
    train_datagen, val_datagen = create_enhanced_data_generators()

    # Train the system
    system, general_model, specialist_model, histories = train_two_stage_system(train_ds, val_ds)

    # Evaluate
    evaluate_enhanced_system(general_model, val_ds, specialist_model)
    plot_training_history(histories)

    # Save models
    general_model.save('enhanced_5class_pneumonia_model_with_dropout.h5')
    if specialist_model:
        specialist_model.save('bacterial_viral_vit_specialist_model.h5')

    print("ğŸ�‰ Training dan evaluation dengan ViT selesai!")

if __name__ == "__main__":
    main()


import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, DenseNet121
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import cv2

# ========================
# 1. ENHANCED PREPROCESSING
# ========================

class EnhancedPreprocessing:
    """
    Optimalisasi preprocessing khusus untuk pneumonia classification
    Focus: pencahayaan, ketajaman, dan kontras untuk membedakan pola bakterial vs viral
    """

    @staticmethod
    def enhance_image(image):
        """
        Enhanced preprocessing untuk pneumonia X-ray
        - CLAHE untuk kontras adaptif
        - Gaussian blur removal untuk ketajaman
        - Histogram equalization untuk pencahayaan optimal
        """
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Bagus untuk enhance perbedaan antara area padat dan menyebar
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        if len(image.shape) == 3:
            # RGB image
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale
            enhanced = clahe.apply(image)
            if len(enhanced.shape) == 2:
                enhanced = np.expand_dims(enhanced, axis=-1)
                enhanced = np.repeat(enhanced, 3, axis=-1)

        # Sharpening kernel untuk mempertegas batas antara area normal dan infected
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)

        # Normalize back to [0,1]
        enhanced = enhanced.astype(np.float32) / 255.0

        return enhanced

def create_enhanced_data_generators():
    """
    Data augmentation yang dioptimalkan untuk pneumonia
    Focus: variasi pencahayaan dan rotasi kecil (sesuai kondisi X-ray)
    """
    train_datagen = ImageDataGenerator(
        preprocessing_function=EnhancedPreprocessing.enhance_image,
        rotation_range=10,  # Rotasi kecil saja (X-ray biasanya standar orientasi)
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2],  # Variasi pencahayaan
        zoom_range=0.1,
        horizontal_flip=False,  # X-ray tidak boleh flip horizontal
        fill_mode='nearest',
        validation_split=0.2
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=EnhancedPreprocessing.enhance_image,
        validation_split=0.2
    )

    return train_datagen, val_datagen

# ========================
# 2. DUAL POOLING STRATEGY
# ========================

class DualPoolingLayer(layers.Layer):
    """
    Kombinasi Max Pooling + Global Average Pooling
    - Max Pooling: menangkap pola padat/intens (bacterial pneumonia)
    - Global Average Pooling: menangkap pola menyebar (viral pneumonia)
    Weight: 0.5 untuk setiap pooling strategy
    """

    def __init__(self, **kwargs):
        super(DualPoolingLayer, self).__init__(**kwargs)
        self.max_pool = layers.GlobalMaxPooling2D()
        self.avg_pool = layers.GlobalAveragePooling2D()

    def call(self, inputs):
        max_features = self.max_pool(inputs)
        avg_features = self.avg_pool(inputs)

        # Combine dengan weight 0.5 masing-masing
        combined = 0.5 * max_features + 0.5 * avg_features
        return combined

    def get_config(self):
        return super(DualPoolingLayer, self).get_config()

# ========================
# 3. COMBINE DATASETS FOR CROSS-VALIDATION
# ========================

def combine_datasets(train_ds, val_ds):
    """
    Gabung train_ds + val_ds untuk cross-validation
    Return: combined images dan labels sebagai numpy arrays
    """
    all_images = []
    all_labels = []

    # Extract from train_ds
    print("ğŸ“¦ Extracting data from train dataset...")
    for images, labels in train_ds:
        all_images.extend(images.numpy())
        all_labels.extend(labels.numpy())

    # Extract from val_ds
    print("ğŸ“¦ Extracting data from validation dataset...")
    for images, labels in val_ds:
        all_images.extend(images.numpy())
        all_labels.extend(labels.numpy())

    all_images = np.array(all_images)
    all_labels = np.array(all_labels)

    print(f"âœ… Combined dataset: {len(all_images)} samples")
    print(f"   Shape: {all_images.shape}")
    print(f"   Classes distribution: {np.bincount(all_labels)}")

    return all_images, all_labels

# ========================
# 4. CROSS-VALIDATION MODELS
# ========================

class PneumoniaClassificationSystem:
    def __init__(self, num_classes=5):  # 5 classes: BACTERIAL, COVID, NORMAL, TB, VIRAL
        self.num_classes = num_classes
        self.class_names = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']
        self.general_model = None
        self.specialist_model = None

    def build_general_model(self, backbone_name='vgg16'):
        """
        Model 1: General 5-class classifier dengan VGG16
        Classes: [0]Bacterial, [1]COVID, [2]Normal, [3]TB, [4]Viral
        Focus: Enhanced differentiation untuk Bacterial (0) vs Viral (4) Pneumonia
        """
        backbone = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Freeze backbone initially
        backbone.trainable = False

        # Build general model
        self.general_model = models.Sequential([
            backbone,
            DualPoolingLayer(),  # Dual pooling strategy
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        return self.general_model

    def build_specialist_model(self):
        """
        Model 2: Specialist untuk BACTERIAL (0) vs VIRAL (4) Pneumonia ONLY
        Menggunakan DenseNet121 (ganti dari ViT)
        Enhanced architecture khusus untuk membedakan pola bakterial vs viral
        """
        backbone = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        backbone.trainable = False

        # Enhanced architecture khusus untuk Bacterial vs Viral differentiation
        self.specialist_model = models.Sequential([
            backbone,
            DualPoolingLayer(),  # CRITICAL: Max pooling untuk bacterial (padat) + Avg pooling untuk viral (menyebar)
            layers.Dense(1024, activation='relu', name='specialist_dense_1'),
            layers.Dropout(0.6),
            layers.Dense(512, activation='relu', name='specialist_dense_2'),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu', name='specialist_dense_3'),
            layers.Dropout(0.3),
            # Output layer untuk binary classification: Bacterial vs Viral
            layers.Dense(2, activation='softmax', name='specialist_output')  # [Bacterial, Viral] only
        ])

        return self.specialist_model

# ========================
# 5. CROSS-VALIDATION TRAINING
# ========================

def train_with_cross_validation(all_images, all_labels, test_ds, n_folds=5):
    """
    Cross-validation training dengan 5-fold
    train split: 0.9, val split: 0.1 dalam setiap fold
    """
    print("ğŸ”¥ Starting Cross-Validation Training...")
    print(f"Data split per fold: 90% train, 10% validation")
    print(f"Number of folds: {n_folds}")

    # Initialize cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Store models and histories
    general_models = []
    specialist_models = []
    fold_histories = []
    fold_scores = []

    class_names = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']

    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_images, all_labels)):
        print(f"\n{'='*50}")
        print(f"ğŸ”„ FOLD {fold + 1}/{n_folds}")
        print(f"{'='*50}")

        # Split data for this fold
        X_train_fold = all_images[train_idx]
        y_train_fold = all_labels[train_idx]
        X_val_fold = all_images[val_idx]
        y_val_fold = all_labels[val_idx]

        print(f"Train samples: {len(X_train_fold)}")
        print(f"Val samples: {len(X_val_fold)}")
        print(f"Train distribution: {np.bincount(y_train_fold)}")
        print(f"Val distribution: {np.bincount(y_val_fold)}")

        # Create tf.data.Dataset for this fold
        train_ds_fold = tf.data.Dataset.from_tensor_slices((X_train_fold, y_train_fold))
        train_ds_fold = train_ds_fold.batch(32).prefetch(tf.data.AUTOTUNE)

        val_ds_fold = tf.data.Dataset.from_tensor_slices((X_val_fold, y_val_fold))
        val_ds_fold = val_ds_fold.batch(32).prefetch(tf.data.AUTOTUNE)

        # ===== STAGE 1: General 5-Class Model =====
        print(f"\nğŸ“� Fold {fold+1} - Stage 1: Training General 5-Class Model (VGG16)")

        system = PneumoniaClassificationSystem(num_classes=5)
        general_model = system.build_general_model('vgg16')
        general_model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Callbacks
        callbacks_general = [
            EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1),
            ModelCheckpoint(f'general_model_fold_{fold+1}.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, verbose=1, min_lr=1e-7)
        ]

        # Phase 1.1: Frozen training
        print(f"ğŸ”’ Phase 1.1: Frozen VGG16 training...")
        history_general_1 = general_model.fit(
            train_ds_fold,
            validation_data=val_ds_fold,
            epochs=15,
            callbacks=callbacks_general,
            verbose=1
        )

        # Phase 1.2: Fine-tuning
        print(f"ğŸ”“ Phase 1.2: Fine-tuning VGG16...")
        general_model.layers[0].trainable = True

        # Unfreeze last layers carefully
        for layer in general_model.layers[0].layers[:-4]:
            layer.trainable = False

        general_model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        history_general_2 = general_model.fit(
            train_ds_fold,
            validation_data=val_ds_fold,
            epochs=10,
            callbacks=callbacks_general,
            verbose=1
        )

        # ===== STAGE 2: Specialist Model (DenseNet121) =====
        print(f"\nğŸ“� Fold {fold+1} - Stage 2: Training Specialist Model (DenseNet121)")

        # Create specialist dataset (Bacterial vs Viral only)
        bacterial_mask = (y_train_fold == 0)
        viral_mask = (y_train_fold == 4)
        specialist_mask = bacterial_mask | viral_mask

        if np.sum(specialist_mask) > 0:
            X_specialist = X_train_fold[specialist_mask]
            y_specialist = y_train_fold[specialist_mask]
            # Relabel: Bacterial (0) -> 0, Viral (4) -> 1
            y_specialist[y_specialist == 4] = 1

            # Validation specialist data
            val_bacterial_mask = (y_val_fold == 0)
            val_viral_mask = (y_val_fold == 4)
            val_specialist_mask = val_bacterial_mask | val_viral_mask

            if np.sum(val_specialist_mask) > 0:
                X_val_specialist = X_val_fold[val_specialist_mask]
                y_val_specialist = y_val_fold[val_specialist_mask]
                y_val_specialist[y_val_specialist == 4] = 1

                # Create specialist datasets
                train_specialist_fold = tf.data.Dataset.from_tensor_slices((X_specialist, y_specialist))
                train_specialist_fold = train_specialist_fold.batch(32).prefetch(tf.data.AUTOTUNE)

                val_specialist_fold = tf.data.Dataset.from_tensor_slices((X_val_specialist, y_val_specialist))
                val_specialist_fold = val_specialist_fold.batch(32).prefetch(tf.data.AUTOTUNE)

                # Build and train specialist model
                specialist_model = system.build_specialist_model()
                specialist_model.compile(
                    optimizer=optimizers.Adam(learning_rate=1e-4),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )

                callbacks_specialist = [
                    EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
                    ModelCheckpoint(f'specialist_model_fold_{fold+1}.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, verbose=1, min_lr=1e-7)
                ]

                # Phase 2.1: Frozen training
                print(f"ğŸ¦  Phase 2.1: Frozen DenseNet121 training...")
                history_specialist_1 = specialist_model.fit(
                    train_specialist_fold,
                    validation_data=val_specialist_fold,
                    epochs=15,
                    callbacks=callbacks_specialist,
                    verbose=1
                )

                # Phase 2.2: Fine-tuning
                print(f"âš¡ Phase 2.2: Fine-tuning DenseNet121...")
                specialist_model.layers[0].trainable = True

                # Careful unfreezing for DenseNet
                for layer in specialist_model.layers[0].layers[:-20]:
                    layer.trainable = False

                specialist_model.compile(
                    optimizer=optimizers.Adam(learning_rate=5e-6),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )

                history_specialist_2 = specialist_model.fit(
                    train_specialist_fold,
                    validation_data=val_specialist_fold,
                    epochs=12,
                    callbacks=callbacks_specialist,
                    verbose=1
                )
            else:
                print("â�Œ No bacterial/viral samples in validation for specialist model")
                specialist_model = None
                history_specialist_1 = None
                history_specialist_2 = None
        else:
            print("â�Œ No bacterial/viral samples in training for specialist model")
            specialist_model = None
            history_specialist_1 = None
            history_specialist_2 = None

        # Store models and histories
        general_models.append(general_model)
        specialist_models.append(specialist_model)
        fold_histories.append((history_general_1, history_general_2, history_specialist_1, history_specialist_2))

        # Evaluate fold performance
        val_pred = general_model.predict(val_ds_fold, verbose=0)
        val_pred_classes = np.argmax(val_pred, axis=1)
        fold_accuracy = np.mean(val_pred_classes == y_val_fold)
        fold_scores.append(fold_accuracy)

        print(f"âœ… Fold {fold+1} completed - Validation Accuracy: {fold_accuracy:.4f}")

    print(f"\n{'='*60}")
    print(f"ğŸ�† CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    for i, score in enumerate(fold_scores):
        print(f"Fold {i+1}: {score:.4f}")
    print(f"Mean CV Accuracy: {np.mean(fold_scores):.4f} (Â±{np.std(fold_scores):.4f})")

    return general_models, specialist_models, fold_histories, fold_scores

# ========================
# 6. ENSEMBLE PREDICTION
# ========================

def ensemble_predict(models, test_ds, model_type='general'):
    """
    Ensemble prediction dari multiple models
    """
    print(f"ğŸ”® Making ensemble predictions with {len(models)} {model_type} models...")

    all_predictions = []

    # Get predictions from each model
    for i, model in enumerate(models):
        if model is not None:
            print(f"   Model {i+1} predicting...")
            preds = model.predict(test_ds, verbose=0)
            all_predictions.append(preds)

    if all_predictions:
        # Average predictions
        ensemble_preds = np.mean(all_predictions, axis=0)
        ensemble_classes = np.argmax(ensemble_preds, axis=1)

        print(f"âœ… Ensemble prediction completed")
        return ensemble_preds, ensemble_classes
    else:
        print("â�Œ No valid models for ensemble prediction")
        return None, None

# ========================
# 7. COMPREHENSIVE EVALUATION
# ========================

def evaluate_ensemble_system(general_models, specialist_models, test_ds):
    """
    Comprehensive evaluation untuk ensemble system
    """
    class_names = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']

    print("ğŸ”� Evaluating Ensemble 5-Class Pneumonia Classification System...")

    # Get true labels from test set
    y_true = []
    for images, labels in test_ds:
        y_true.extend(labels.numpy())
    y_true = np.array(y_true)

    # ===== GENERAL MODEL ENSEMBLE EVALUATION =====
    print("\nğŸ“Š GENERAL MODEL ENSEMBLE EVALUATION:")

    ensemble_probs, ensemble_preds = ensemble_predict(general_models, test_ds, 'general')

    if ensemble_probs is not None:
        # Overall Classification Report
        print("\nğŸ“‹ Ensemble Classification Report:")
        print(classification_report(y_true, ensemble_preds, target_names=class_names, digits=3))

        # Confusion Matrix
        cm = confusion_matrix(y_true, ensemble_preds)
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 3, 1)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Ensemble 5-Class Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, [name.replace(' ', '\n') for name in class_names], rotation=0, fontsize=8)
        plt.yticks(tick_marks, [name.replace(' ', '\n') for name in class_names], fontsize=8)

        # Add text annotations
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10)

        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # ===== BACTERIAL vs VIRAL ANALYSIS =====
        print("\nğŸ¦  BACTERIAL vs VIRAL PNEUMONIA ANALYSIS:")

        bacterial_idx = 0
        viral_idx = 4

        bacterial_cases = np.sum(y_true == bacterial_idx)
        viral_cases = np.sum(y_true == viral_idx)

        print(f"Total Bacterial cases: {bacterial_cases}")
        print(f"Total Viral cases: {viral_cases}")

        if bacterial_cases > 0 and viral_cases > 0:
            # Performance metrics
            bacterial_predicted_as_bacterial = cm[bacterial_idx, bacterial_idx]
            bacterial_predicted_as_viral = cm[bacterial_idx, viral_idx]
            viral_predicted_as_viral = cm[viral_idx, viral_idx]
            viral_predicted_as_bacterial = cm[viral_idx, bacterial_idx]

            bacterial_precision = cm[bacterial_idx, bacterial_idx] / (cm[:, bacterial_idx].sum() + 1e-10)
            viral_precision = cm[viral_idx, viral_idx] / (cm[:, viral_idx].sum() + 1e-10)

            bacterial_recall = cm[bacterial_idx, bacterial_idx] / (cm[bacterial_idx, :].sum() + 1e-10)
            viral_recall = cm[viral_idx, viral_idx] / (cm[viral_idx, :].sum() + 1e-10)

            bacterial_f1 = 2 * (bacterial_precision * bacterial_recall) / (bacterial_precision + bacterial_recall + 1e-10)
            viral_f1 = 2 * (viral_precision * viral_recall) / (viral_precision + viral_recall + 1e-10)

            print(f"\nğŸ“ˆ Bacterial Pneumonia Performance:")
            print(f"   Precision: {bacterial_precision:.3f}")
            print(f"   Recall: {bacterial_recall:.3f}")
            print(f"   F1-Score: {bacterial_f1:.3f}")

            print(f"\nğŸ“ˆ Viral Pneumonia Performance:")
            print(f"   Precision: {viral_precision:.3f}")
            print(f"   Recall: {viral_recall:.3f}")
            print(f"   F1-Score: {viral_f1:.3f}")

            # Confidence analysis
            bacterial_mask = y_true == bacterial_idx
            viral_mask = y_true == viral_idx

            bacterial_confidences = ensemble_probs[bacterial_mask][:, bacterial_idx]
            viral_confidences = ensemble_probs[viral_mask][:, viral_idx]

            print(f"\nğŸ�¯ Confidence Analysis:")
            print(f"   Bacterial Mean Confidence: {np.mean(bacterial_confidences):.3f} (Â±{np.std(bacterial_confidences):.3f})")
            print(f"   Viral Mean Confidence: {np.mean(viral_confidences):.3f} (Â±{np.std(viral_confidences):.3f})")

            # Binary confusion matrix plot
            plt.subplot(2, 3, 2)
            bv_cm = np.array([[bacterial_predicted_as_bacterial, bacterial_predicted_as_viral],
                             [viral_predicted_as_bacterial, viral_predicted_as_viral]])
            plt.imshow(bv_cm, interpolation='nearest', cmap=plt.cm.Reds)
            plt.title('Bacterial vs Viral\nConfusion Matrix')
            plt.colorbar()
            plt.xticks([0, 1], ['Bacterial', 'Viral'])
            plt.yticks([0, 1], ['Bacterial', 'Viral'])

            for i, j in np.ndindex(bv_cm.shape):
                plt.text(j, i, format(bv_cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if bv_cm[i, j] > bv_cm.max()/2 else "black",
                        fontsize=12)

            # Confidence distribution
            plt.subplot(2, 3, 3)
            plt.hist(bacterial_confidences, bins=20, alpha=0.7, label=f'Bacterial (n={len(bacterial_confidences)})', color='red')
            plt.hist(viral_confidences, bins=20, alpha=0.7, label=f'Viral (n={len(viral_confidences)})', color='blue')
            plt.xlabel('Prediction Confidence')
            plt.ylabel('Frequency')
            plt.title('Confidence Distribution\nBacterial vs Viral')
            plt.legend()
            plt.grid(True, alpha=0.3)

        # ===== SPECIALIST MODEL EVALUATION =====
        valid_specialists = [m for m in specialist_models if m is not None]
        if valid_specialists:
            print(f"\nğŸ�¯ SPECIALIST MODEL ENSEMBLE EVALUATION ({len(valid_specialists)} models):")

            # Create specialist test data (bacterial vs viral only)
            specialist_images = []
            specialist_labels = []

            for images, labels in test_ds:
                images_np = images.numpy()
                labels_np = labels.numpy()

                mask = (labels_np == 0) | (labels_np == 4)
                if np.any(mask):
                    selected_images = images_np[mask]
                    selected_labels = labels_np[mask]
                    selected_labels[selected_labels == 4] = 1  # Relabel viral to 1

                    specialist_images.extend(selected_images)
                    specialist_labels.extend(selected_labels)

            if specialist_images:
                specialist_test_ds = tf.data.Dataset.from_tensor_slices((np.array(specialist_images), np.array(specialist_labels)))
                specialist_test_ds = specialist_test_ds.batch(32).prefetch(tf.data.AUTOTUNE)

                spec_probs, spec_preds = ensemble_predict(valid_specialists, specialist_test_ds, 'specialist')

                if spec_probs is not None:
                    binary_class_names = ['Bacterial Pneumonia', 'Viral Pneumonia']
                    print("\nğŸ“‹ Specialist Binary Classification Report:")
                    print(classification_report(specialist_labels, spec_preds, target_names=binary_class_names, digits=3))

                    # Binary confusion matrix
                    spec_cm = confusion_matrix(specialist_labels, spec_preds)
                    plt.subplot(2, 3, 4)
                    plt.imshow(spec_cm, interpolation='nearest', cmap=plt.cm.Greens)
                    plt.title('Specialist Ensemble\nBinary Classification')
                    plt.colorbar()
                    plt.xticks([0, 1], ['Bacterial', 'Viral'])
                    plt.yticks([0, 1], ['Bacterial', 'Viral'])

                    for i, j in np.ndindex(spec_cm.shape):
                        plt.text(j, i, format(spec_cm[i, j], 'd'),
                                horizontalalignment="center",
                                color="white" if spec_cm[i, j] > spec_cm.max()/2 else "black",
                                fontsize=12)

        plt.tight_layout()
        plt.show()

        # Final summary
        overall_accuracy = np.mean(y_true == ensemble_preds)
        print(f"\n{'='*60}")
        print(f"ğŸ�† ENSEMBLE SYSTEM PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        print(f"Overall 5-Class Accuracy: {overall_accuracy:.4f}")
        if bacterial_cases > 0 and viral_cases > 0:
            print(f"Bacterial Pneumonia F1-Score: {bacterial_f1:.4f}")
            print(f"Viral Pneumonia F1-Score: {viral_f1:.4f}")
            bv_accuracy = (bacterial_predicted_as_bacterial + viral_predicted_as_viral) / (bacterial_cases + viral_cases)
            print(f"Bacterial vs Viral Accuracy: {bv_accuracy:.4f}")
        print(f"{'='*60}")

# ========================
# 8. MAIN EXECUTION
# ========================

def main_improved_training(train_ds, val_ds, test_ds):
    """
    Main function untuk improved training dengan cross-validation
    """
    print("ğŸš€ Enhanced 5-Class Pneumonia Classification with Cross-Validation")
    print("="*70)
    print("Improvements:")
    print("âœ… VGG16 untuk general model (tetap)")
    print("âœ… DenseNet121 untuk specialist model (ganti dari ViT)")
    print("âœ… Data gabungan train+val dengan 90-10 split")
    print("âœ… 5-fold Cross-Validation")
    print("âœ… Enhanced dropout strategy")
    print("âœ… Ensemble prediction pada test set")
    print("="*70)

    # Setup GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("âœ… GPU detected")
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("âœ… GPU memory growth enabled")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("âš ï¸� Using CPU for training")

    # Enable mixed precision
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("âœ… Mixed precision enabled")

    # Step 1: Combine datasets
    print("\nğŸ”„ Step 1: Combining train and validation datasets...")
    all_images, all_labels = combine_datasets(train_ds, val_ds)

    # Step 2: Cross-validation training
    print("\nğŸ”„ Step 2: Starting cross-validation training...")
    general_models, specialist_models, fold_histories, fold_scores = train_with_cross_validation(
        all_images, all_labels, test_ds, n_folds=5
    )

    # Step 3: Save best models
    print("\nğŸ’¾ Step 3: Saving best models...")
    best_fold_idx = np.argmax(fold_scores)
    best_general_model = general_models[best_fold_idx]
    best_specialist_model = specialist_models[best_fold_idx]

    best_general_model.save('best_general_model_cv.h5')
    print(f"âœ… Best general model saved (Fold {best_fold_idx + 1}, Accuracy: {fold_scores[best_fold_idx]:.4f})")

    if best_specialist_model is not None:
        best_specialist_model.save('best_specialist_model_cv.h5')
        print("âœ… Best specialist model saved")

    # Step 4: Predict test set with ensemble
    print("\nğŸ”® Step 4: Making ensemble predictions on test set...")
    evaluate_ensemble_system(general_models, specialist_models, test_ds)

    print("\nğŸ�‰ Cross-validation training and evaluation completed!")

    return general_models, specialist_models, fold_histories, fold_scores

# ========================
# 9. PLOT CROSS-VALIDATION RESULTS
# ========================

def plot_cv_results(fold_histories, fold_scores):
    """
    Plot cross-validation results
    """
    n_folds = len(fold_histories)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot individual fold accuracies
    axes[0, 0].bar(range(1, n_folds + 1), fold_scores, color='skyblue', alpha=0.7)
    axes[0, 0].axhline(y=np.mean(fold_scores), color='red', linestyle='--',
                       label=f'Mean: {np.mean(fold_scores):.3f}')
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('Validation Accuracy')
    axes[0, 0].set_title('Cross-Validation Scores per Fold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot training curves for each fold (general model)
    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for fold in range(n_folds):
        history_general_1, history_general_2, _, _ = fold_histories[fold]

        if history_general_1 and history_general_2:
            # Combine histories
            combined_acc = history_general_1.history['accuracy'] + history_general_2.history['accuracy']
            combined_val_acc = history_general_1.history['val_accuracy'] + history_general_2.history['val_accuracy']

            axes[0, 1].plot(combined_acc, color=colors[fold % len(colors)], alpha=0.6, label=f'Fold {fold+1} Train')
            axes[0, 1].plot(combined_val_acc, color=colors[fold % len(colors)], alpha=0.6, linestyle='--', label=f'Fold {fold+1} Val')

    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training Curves - All Folds (General Model)')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)

    # Average training curve
    all_train_accs = []
    all_val_accs = []

    for fold in range(n_folds):
        history_general_1, history_general_2, _, _ = fold_histories[fold]
        if history_general_1 and history_general_2:
            combined_acc = history_general_1.history['accuracy'] + history_general_2.history['accuracy']
            combined_val_acc = history_general_1.history['val_accuracy'] + history_general_2.history['val_accuracy']
            all_train_accs.append(combined_acc)
            all_val_accs.append(combined_val_acc)

    if all_train_accs:
        # Pad sequences to same length
        max_len = max(len(seq) for seq in all_train_accs)
        padded_train = []
        padded_val = []

        for train_acc, val_acc in zip(all_train_accs, all_val_accs):
            if len(train_acc) < max_len:
                # Pad with last value
                train_acc = train_acc + [train_acc[-1]] * (max_len - len(train_acc))
                val_acc = val_acc + [val_acc[-1]] * (max_len - len(val_acc))
            padded_train.append(train_acc)
            padded_val.append(val_acc)

        mean_train_acc = np.mean(padded_train, axis=0)
        mean_val_acc = np.mean(padded_val, axis=0)
        std_train_acc = np.std(padded_train, axis=0)
        std_val_acc = np.std(padded_val, axis=0)

        epochs = range(len(mean_train_acc))

        axes[0, 2].plot(epochs, mean_train_acc, color='blue', label='Mean Train Acc')
        axes[0, 2].fill_between(epochs, mean_train_acc - std_train_acc, mean_train_acc + std_train_acc,
                               color='blue', alpha=0.2)
        axes[0, 2].plot(epochs, mean_val_acc, color='red', label='Mean Val Acc')
        axes[0, 2].fill_between(epochs, mean_val_acc - std_val_acc, mean_val_acc + std_val_acc,
                               color='red', alpha=0.2)

        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].set_title('Average Training Curves Â± Std')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

    # Cross-validation statistics
    axes[1, 0].text(0.1, 0.8, f'Cross-Validation Results:', fontsize=14, fontweight='bold',
                    transform=axes[1, 0].transAxes)
    axes[1, 0].text(0.1, 0.7, f'Number of Folds: {n_folds}', fontsize=12, transform=axes[1, 0].transAxes)
    axes[1, 0].text(0.1, 0.6, f'Mean Accuracy: {np.mean(fold_scores):.4f}', fontsize=12, transform=axes[1, 0].transAxes)
    axes[1, 0].text(0.1, 0.5, f'Std Accuracy: Â±{np.std(fold_scores):.4f}', fontsize=12, transform=axes[1, 0].transAxes)
    axes[1, 0].text(0.1, 0.4, f'Best Fold: {np.argmax(fold_scores) + 1} ({np.max(fold_scores):.4f})',
                    fontsize=12, transform=axes[1, 0].transAxes)
    axes[1, 0].text(0.1, 0.3, f'Worst Fold: {np.argmin(fold_scores) + 1} ({np.min(fold_scores):.4f})',
                    fontsize=12, transform=axes[1, 0].transAxes)
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].axis('off')

    # Accuracy distribution
    axes[1, 1].hist(fold_scores, bins=min(10, n_folds), color='lightgreen', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(np.mean(fold_scores), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1, 1].set_xlabel('Validation Accuracy')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of CV Scores')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Model complexity info
    axes[1, 2].text(0.1, 0.9, 'Model Architecture:', fontsize=14, fontweight='bold', transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.8, 'General Model: VGG16', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.7, 'Specialist Model: DenseNet121', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.6, 'Pooling: Dual (Max + Avg)', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.5, 'Training: Two-phase', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.4, '1. Frozen backbone', fontsize=10, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.3, '2. Fine-tuning', fontsize=10, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.2, f'Classes: 5 (Pneumonia types)', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("ğŸš€ Enhanced 5-Class Pneumonia Classification with Cross-Validation")
    print("="*70)
    print("ğŸ“‹ USAGE:")
    print("1. Load your datasets:")
    print("   train_ds = tf.keras.preprocessing.image_dataset_from_directory(...)")
    print("   val_ds = tf.keras.preprocessing.image_dataset_from_directory(...)")
    print("   test_ds = tf.keras.preprocessing.image_dataset_from_directory(...)")
    print("")
    print("2. Run cross-validation training:")
    print("   general_models, specialist_models, histories, scores = main_improved_training(train_ds, val_ds, test_ds)")
    print("")
    print("3. Plot results:")
    print("   plot_cv_results(histories, scores)")
    print("="*70)
    print("")
    print("âœ… Key Features:")
    print("â€¢ VGG16 for general 5-class classification")
    print("â€¢ DenseNet121 for bacterial vs viral specialization")
    print("â€¢ 5-fold cross-validation with 90-10 splits")
    print("â€¢ Enhanced dropout strategy")
    print("â€¢ Ensemble prediction on test set")
    print("â€¢ Comprehensive evaluation and visualization")
    print("")
    print("ğŸ�¯ Expected Performance:")
    print("â€¢ Improved bacterial vs viral differentiation")
    print("â€¢ Better generalization through cross-validation")
    print("â€¢ Robust ensemble predictions")
    print("â€¢ Detailed performance analysis")

    # Uncomment to run with your datasets:
    general_models, specialist_models, histories, scores = main_improved_training(train_ds, val_ds, test_ds)
    plot_cv_results(histories, scores)


import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2  # Lighter than VGG16!
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# ========================
# 1. MINIMAL PREPROCESSING
# ========================

def simple_preprocess(image):
    """
    Super simple preprocessing - hanya normalisasi
    """
    return tf.cast(image, tf.float32) / 255.0

def create_minimal_data_generators():
    """
    Minimal augmentation untuk menghindari overhead
    """
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=5,  # Minimal rotation
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=False,
        validation_split=0.2
    )
    
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    return train_datagen, val_datagen

# ========================
# 2. SINGLE ULTRA-LIGHT MODEL
# ========================

class UltraLightPneumoniaClassifier:
    """
    Single ultra-light model dengan MobileNetV2
    """
    def __init__(self, num_classes=5):
        self.num_classes = num_classes
        self.class_names = ['Bacterial', 'COVID-19', 'Normal', 'TB', 'Viral']
        self.model = None
    
    def build_ultralight_model(self, input_shape=(224, 224, 3)):
        """
        Build ultra-light model dengan MobileNetV2 (jauh lebih ringan dari VGG16)
        """
        # MobileNetV2 - designed untuk efficiency!
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape,
            alpha=0.75  # Reduce model width untuk lebih ringan lagi
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Super simple classifier head
        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),  # Paling efficient
            layers.Dropout(0.3),  # Reduced dropout
            layers.Dense(128, activation='relu'),  # Smaller dense layer
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        print("âœ… Ultra-light model built!")
        print(f"ğŸ“Š Total parameters: {self.model.count_params():,}")
        print(f"ğŸ“Š Trainable parameters: {sum([tf.keras.utils.count_params(w) for w in self.model.trainable_weights]):,}")
        
        return self.model
    
    def compile_with_optimizer(self, optimizer_type='adamw', learning_rate=1e-3):
        """
        Compile dengan berbagai optimizer options
        """
        # Pilihan optimizer yang efisien
        if optimizer_type.lower() == 'adamw':
            opt = optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-5)
        elif optimizer_type.lower() == 'adam':
            opt = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_type.lower() == 'sgd':
            opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            opt = optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=opt,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"âœ… Model compiled with {optimizer_type.upper()} optimizer")
        print(f"ğŸ“ˆ Learning rate: {learning_rate}")
        
        return opt
    
    def get_minimal_callbacks(self):
        """
        Minimal callbacks untuk cepat training
        """
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,  # Reduced patience
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,  # More aggressive reduction
                patience=2,  # Faster reaction
                verbose=1,
                min_lr=1e-6
            )
        ]
        return callbacks

# ========================
# 3. FAST TRAINING
# ========================

def train_fast_model(train_ds, val_ds, epochs=20, optimizer='adamw'):
    """
    Fast training dengan single phase saja
    """
    print("ğŸš€ Ultra-Fast Pneumonia Classification Training")
    print("="*50)
    print(f"ğŸ”§ Using {optimizer.upper()} optimizer")
    print("âš¡ Single-phase training only")
    print("ğŸ“± MobileNetV2 backbone")
    print("="*50)
    
    # Build model
    classifier = UltraLightPneumoniaClassifier()
    model = classifier.build_ultralight_model()
    
    # Compile with chosen optimizer
    classifier.compile_with_optimizer(optimizer_type=optimizer, learning_rate=1e-3)
    
    # Get callbacks
    callbacks = classifier.get_minimal_callbacks()
    
    # Single-phase training (no fine-tuning untuk speed)
    print(f"\nğŸ�ƒâ€�â™‚ï¸� FAST TRAINING ({epochs} epochs max)")
    print("-" * 40)
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    print("âœ… Fast training completed!")
    
    return model, history

# ========================
# 4. QUICK EVALUATION
# ========================

def quick_evaluate(model, test_ds, class_names):
    """
    Quick evaluation tanpa terlalu banyak computation
    """
    print("ğŸ”� Quick evaluation...")
    
    # Simple prediction
    y_true = []
    y_pred = []
    
    for batch_images, batch_labels in test_ds:
        predictions = model.predict(batch_images, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        
        y_true.extend(batch_labels.numpy())
        y_pred.extend(pred_classes)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Quick metrics
    accuracy = np.mean(y_true == y_pred)
    print(f"ğŸ�¯ Test Accuracy: {accuracy:.4f}")
    
    # Quick classification report
    print("\nğŸ“‹ Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))
    
    return y_true, y_pred, accuracy

def plot_quick_results(history, y_true, y_pred, class_names):
    """
    Quick plotting
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Training curves
    epochs = range(1, len(history.history['accuracy']) + 1)
    
    axes[0].plot(epochs, history.history['accuracy'], 'b-', label='Train Acc', linewidth=2)
    axes[0].plot(epochs, history.history['val_accuracy'], 'r-', label='Val Acc', linewidth=2)
    axes[0].set_title('Training Progress', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    im = axes[1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=axes[1])
    
    # Set ticks
    tick_marks = np.arange(len(class_names))
    axes[1].set_xticks(tick_marks)
    axes[1].set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
    axes[1].set_yticks(tick_marks)
    axes[1].set_yticklabels(class_names, fontsize=10)
    
    # Add numbers to confusion matrix
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        axes[1].text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight='bold')
    
    # Per-class accuracy bar chart
    class_acc = []
    for i in range(len(class_names)):
        if np.sum(y_true == i) > 0:
            acc = cm[i, i] / np.sum(y_true == i)
        else:
            acc = 0
        class_acc.append(acc)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    bars = axes[2].bar(range(len(class_names)), class_acc, color=colors, alpha=0.8)
    axes[2].set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Class')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_xticks(range(len(class_names)))
    axes[2].set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
    axes[2].set_ylim(0, 1.1)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars, class_acc):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def quick_bacterial_viral_analysis(y_true, y_pred):
    """
    Quick analysis untuk bacterial vs viral
    """
    print("\nğŸ¦  BACTERIAL vs VIRAL Quick Analysis")
    print("="*40)
    
    # Indices
    bacterial_idx = 0
    viral_idx = 4
    
    # Count
    bacterial_total = np.sum(y_true == bacterial_idx)
    viral_total = np.sum(y_true == viral_idx)
    
    print(f"ğŸ“Š Bacterial cases: {bacterial_total}")
    print(f"ğŸ“Š Viral cases: {viral_total}")
    
    if bacterial_total > 0 and viral_total > 0:
        # Simple accuracy calculation
        bacterial_correct = np.sum((y_true == bacterial_idx) & (y_pred == bacterial_idx))
        viral_correct = np.sum((y_true == viral_idx) & (y_pred == viral_idx))
        
        bacterial_acc = bacterial_correct / bacterial_total
        viral_acc = viral_correct / viral_total
        
        print(f"ğŸ�¯ Bacterial accuracy: {bacterial_acc:.3f}")
        print(f"ğŸ�¯ Viral accuracy: {viral_acc:.3f}")
        
        # Binary accuracy
        binary_correct = bacterial_correct + viral_correct
        binary_total = bacterial_total + viral_total
        binary_acc = binary_correct / binary_total
        
        print(f"ğŸ�† Bacterial vs Viral binary accuracy: {binary_acc:.3f}")
        
        return {
            'bacterial_acc': bacterial_acc,
            'viral_acc': viral_acc,
            'binary_acc': binary_acc
        }
    
    return None

# ========================
# 5. MAIN ULTRA-LIGHT FUNCTION
# ========================

def main_ultralight_training(train_ds, val_ds, test_ds, optimizer='adamw', epochs=20):
    """
    Main ultra-light training function
    """
    print("ğŸš€ ULTRA-LIGHT Pneumonia Classification")
    print("="*50)
    print("ğŸ”¥ ULTRA-LIGHT Features:")
    print("âœ… MobileNetV2 (ultra-efficient)")
    print("âœ… Single-phase training (no fine-tuning)")
    print("âœ… Minimal preprocessing")
    print("âœ… Fast evaluation")
    print(f"âœ… {optimizer.upper()} optimizer")
    print("âœ… NO dataset combination")
    print("âœ… NO multiple models")
    print("âœ… NO cross-validation")
    print("="*50)
    
    # Train model
    model, history = train_fast_model(train_ds, val_ds, epochs=epochs, optimizer=optimizer)
    
    # Save model
    model.save('ultralight_pneumonia_model.h5')
    print("âœ… Model saved as 'ultralight_pneumonia_model.h5'")
    
    # Quick evaluate
    class_names = ['Bacterial', 'COVID-19', 'Normal', 'TB', 'Viral']
    y_true, y_pred, accuracy = quick_evaluate(model, test_ds, class_names)
    
    # Plot results
    plot_quick_results(history, y_true, y_pred, class_names)
    
    # Quick bacterial vs viral analysis
    bv_metrics = quick_bacterial_viral_analysis(y_true, y_pred)
    
    # Final summary
    print(f"\nğŸ�† ULTRA-LIGHT RESULTS")
    print("="*30)
    print(f"Overall Accuracy: {accuracy:.4f}")
    if bv_metrics:
        print(f"Bacterial Accuracy: {bv_metrics['bacterial_acc']:.4f}")
        print(f"Viral Accuracy: {bv_metrics['viral_acc']:.4f}")
        print(f"Bacterial vs Viral: {bv_metrics['binary_acc']:.4f}")
    
    # Model efficiency info
    total_params = model.count_params()
    trainable_params = sum([tf.keras.utils.count_params(w) for w in model.trainable_weights])
    
    print(f"\nâš¡ MODEL EFFICIENCY:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params*4/1024/1024:.1f} MB")
    
    return model, history, y_true, y_pred, accuracy

# ========================
# OPTIMIZER COMPARISON
# ========================

def compare_optimizers(train_ds, val_ds, test_ds, optimizers=['adam', 'adamw', 'sgd']):
    """
    Quick comparison berbagai optimizer
    """
    print("ğŸ”¥ OPTIMIZER COMPARISON")
    print("="*40)
    
    results = {}
    
    for opt in optimizers:
        print(f"\nğŸ§ª Testing {opt.upper()} optimizer...")
        
        model, history, y_true, y_pred, accuracy = main_ultralight_training(
            train_ds, val_ds, test_ds, 
            optimizer=opt, 
            epochs=15  # Reduced epochs untuk comparison
        )
        
        results[opt] = {
            'model': model,
            'history': history,
            'accuracy': accuracy,
            'y_true': y_true,
            'y_pred': y_pred
        }
        
        print(f"âœ… {opt.upper()} completed - Accuracy: {accuracy:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    opt_names = list(results.keys())
    accuracies = [results[opt]['accuracy'] for opt in opt_names]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = plt.bar(opt_names, accuracies, color=colors, alpha=0.8)
    plt.title('Optimizer Comparison', fontweight='bold')
    plt.ylabel('Test Accuracy')
    plt.ylim(0, 1)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.subplot(1, 2, 2)
    for opt in opt_names:
        history = results[opt]['history']
        plt.plot(history.history['val_accuracy'], label=f'{opt.upper()}', linewidth=2)
    
    plt.title('Training Progress Comparison', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Best optimizer
    best_opt = max(results.keys(), key=lambda x: results[x]['accuracy'])
    print(f"\nğŸ�† BEST OPTIMIZER: {best_opt.upper()}")
    print(f"ğŸ�¯ Best accuracy: {results[best_opt]['accuracy']:.4f}")
    
    return results, best_opt

# ========================
# USAGE
# ========================

if __name__ == "__main__":
    print("ğŸ“‹ ULTRA-LIGHT USAGE:")
    print("="*30)
    print("# Quick training:")
    print("model, history, y_true, y_pred, acc = main_ultralight_training(train_ds, val_ds, test_ds)")
    print("")
    print("# Or compare optimizers:")
    print("results, best_opt = compare_optimizers(train_ds, val_ds, test_ds)")
    print("")
    print("ğŸ�¯ Expected training time: ~5-15 minutes!")
    print("ğŸ’¾ Expected model size: ~15-30 MB")
    print("âš¡ Should work on ANY machine without crash!")

# Uncomment to run:
model, history, y_true, y_pred, acc = main_ultralight_training(train_ds, val_ds, test_ds, optimizer='adamw')


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

# Load VGG16 sebagai backbone pretrained dari ImageNet
# Freeze dulu semua layer (akan unfreeze nanti di fine-tuning)
backbone = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Print summary untuk liat struktur
backbone.summary()

print("âœ… Backbone VGG16 loaded and ready!")


# Asumsi num_classes = 5 dari dataset (sesuaikan jika beda)
num_classes = 5

# Freeze backbone untuk tahap awal
backbone.trainable = False

# Build model
model = models.Sequential([
    backbone,
    layers.GlobalAveragePooling2D(),  # Ubah feature map jadi vector
    layers.Dense(256, activation='relu'),  # Dense layer kecil
    layers.Dropout(0.5),  # Dropout untuk kurangi overfitting
    layers.Dense(num_classes, activation='softmax')  # Output layer
])

# Print summary model
model.summary()

print("âœ… Model built with VGG16 backbone!")


import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.mixed_precision import set_global_policy

# Verify GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("âœ… GPU detected:", gpus)
    tf.config.set_visible_devices(gpus[0], 'GPU')
    print("Using GPU:", gpus[0])
else:
    print("â�Œ No GPU detected. Running on CPU.")

# Enable mixed precision for GPU optimization
set_global_policy('mixed_float16')
print("âœ… Mixed precision enabled")

# Compile model
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',  # Sparse karena label integer
    metrics=['accuracy']
)

# Print model summary to confirm
model.summary()

print("âœ… Model compiled with GPU support!")


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Optimize dataset for GPU
train_ds_processed = train_ds_processed.cache().prefetch(tf.data.AUTOTUNE)
val_ds_processed = val_ds_processed.cache().prefetch(tf.data.AUTOTUNE)

# Callbacks for monitoring
callbacks_phase1 = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('model_phase1.h5', monitor='val_loss', save_best_only=True)
]

# Train tahap 1 (backbone frozen)
try:
    history_phase1 = model.fit(
        train_ds_processed,
        validation_data=val_ds_processed,
        epochs=10,
        callbacks=callbacks_phase1
    )
    print("âœ… Phase 1 training complete!")
except Exception as e:
    print("â�Œ Error during training:", str(e))

# Check GPU usage
!nvidia-smi


from tensorflow.keras.optimizers import Adam

# Unfreeze sebagian backbone (20 layer terakhir)
backbone.trainable = True
fine_tune_from = -20  # Unfreeze dari block5_conv1 ke atas (lihat backbone.summary())
for layer in backbone.layers[:fine_tune_from]:
    layer.trainable = False

# Print trainable layers untuk verifikasi
print("Trainable layers:")
for layer in backbone.layers:
    print(layer.name, layer.trainable)

# Re-compile dengan learning rate kecil
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=[
        'accuracy']
)

# Callbacks untuk phase 2
callbacks_phase2 = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('model_phase2.h5', monitor='val_loss', save_best_only=True)
]

# Train tahap 2
try:
    history_phase2 = model.fit(
        train_ds_processed,
        validation_data=val_ds_processed,
        epochs=10,
        callbacks=callbacks_phase2
    )
    print("âœ… Phase 2 fine-tuning complete!")
except Exception as e:
    print("â�Œ Error during fine-tuning:", str(e))

# Check GPU usage
!nvidia-smi

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Gabung history dari phase 1 & 2
history = {}
for key in history_phase1.history:
    history[key] = history_phase1.history[key] + history_phase2.history[key]

# Plot loss & accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history['accuracy'], label='Train Acc')
plt.plot(history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()
plt.show()

# Evaluasi pada val_ds
y_true = []
y_pred = []
try:
    for images, labels in val_ds_processed:
        preds = model.predict(images)  # GPU-accelerated prediction
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())
    print("âœ… Predictions generated!")
except Exception as e:
    print("â�Œ Error during prediction:", str(e))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
# Ganti display_labels dengan nama kelas dari dataset (lihat train_ds.class_names)
class_names = train_ds.class_names if 'train_ds' in globals() else ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot()
plt.show()

# Classification report
print(classification_report(y_true, y_pred, target_names=class_names))

print("âœ… Evaluation complete!")

# Check GPU usage during prediction
!nvidia-smi


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

# Load VGG16 sebagai backbone pretrained dari ImageNet
# Freeze dulu semua layer (akan unfreeze nanti di fine-tuning)
backbone = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Print summary untuk liat struktur
backbone.summary()

print("âœ… Backbone VGG16 loaded and ready!")


# Asumsi num_classes = 5 dari dataset (sesuaikan jika beda)
num_classes = 5

# Freeze backbone untuk tahap awal
backbone.trainable = False

# Build model
model = models.Sequential([
    backbone,
    layers.GlobalAveragePooling2D(),  # Ubah feature map jadi vector
    layers.Dense(256, activation='relu'),  # Dense layer kecil
    layers.Dropout(0.5),  # Dropout untuk kurangi overfitting
    layers.Dense(num_classes, activation='softmax')  # Output layer
])

# Print summary model
model.summary()

print("âœ… Model built with VGG16 backbone!")


import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.mixed_precision import set_global_policy

# Verify GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("âœ… GPU detected:", gpus)
    tf.config.set_visible_devices(gpus[0], 'GPU')
    print("Using GPU:", gpus[0])
else:
    print("â�Œ No GPU detected. Running on CPU.")

# Enable mixed precision for GPU optimization
set_global_policy('mixed_float16')
print("âœ… Mixed precision enabled")

# Compile model
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',  # Sparse karena label integer
    metrics=['accuracy']
)

# Print model summary to confirm
model.summary()

print("âœ… Model compiled with GPU support!")


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Optimize dataset for GPU
train_ds_processed = train_ds_processed.cache().prefetch(tf.data.AUTOTUNE)
val_ds_processed = val_ds_processed.cache().prefetch(tf.data.AUTOTUNE)

# Callbacks for monitoring
callbacks_phase1 = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('model_phase1.h5', monitor='val_loss', save_best_only=True)
]

# Train tahap 1 (backbone frozen)
try:
    history_phase1 = model.fit(
        train_ds_processed,
        validation_data=val_ds_processed,
        epochs=10,
        callbacks=callbacks_phase1
    )
    print("âœ… Phase 1 training complete!")
except Exception as e:
    print("â�Œ Error during training:", str(e))

# Check GPU usage
!nvidia-smi


from tensorflow.keras.optimizers import Adam

# Unfreeze sebagian backbone (20 layer terakhir)
backbone.trainable = True
fine_tune_from = -20  # Unfreeze dari block5_conv1 ke atas (lihat backbone.summary())
for layer in backbone.layers[:fine_tune_from]:
    layer.trainable = False

# Print trainable layers untuk verifikasi
print("Trainable layers:")
for layer in backbone.layers:
    print(layer.name, layer.trainable)

# Re-compile dengan learning rate kecil
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=[
        'accuracy']
)

# Callbacks untuk phase 2
callbacks_phase2 = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('model_phase2.h5', monitor='val_loss', save_best_only=True)
]

# Train tahap 2
try:
    history_phase2 = model.fit(
        train_ds_processed,
        validation_data=val_ds_processed,
        epochs=10,
        callbacks=callbacks_phase2
    )
    print("âœ… Phase 2 fine-tuning complete!")
except Exception as e:
    print("â�Œ Error during fine-tuning:", str(e))

# Check GPU usage
!nvidia-smi


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Gabung history dari phase 1 & 2
history = {}
for key in history_phase1.history:
    history[key] = history_phase1.history[key] + history_phase2.history[key]

# Plot loss & accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history['accuracy'], label='Train Acc')
plt.plot(history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()
plt.show()

# Evaluasi pada val_ds
y_true = []
y_pred = []
try:
    for images, labels in val_ds_processed:
        preds = model.predict(images)  # GPU-accelerated prediction
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())
    print("âœ… Predictions generated!")
except Exception as e:
    print("â�Œ Error during prediction:", str(e))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
# Ganti display_labels dengan nama kelas dari dataset (lihat train_ds.class_names)
class_names = train_ds.class_names if 'train_ds' in globals() else ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot()
plt.show()

# Classification report
print(classification_report(y_true, y_pred, target_names=class_names))

print("âœ… Evaluation complete!")

# Check GPU usage during prediction
!nvidia-smi


# import tensorflow as tf
# import cv2
# import numpy as np
# from tensorflow.keras.applications.imagenet_utils import preprocess_input
# import matplotlib.pyplot as plt

# # ============================================================================
# # STEP 1: ROI EXTRACTION (Region of Interest - Segmentasi Paru)
# # ============================================================================
# """
# Fungsi untuk mengekstrak area paru-paru dari X-ray
# Tujuan: Menghilangkan noise seperti tulisan "R/L", alat medis, dll
# """

# def create_lung_mask_opencv(image_np):
#     """
#     Helper function untuk OpenCV operations (akan dipanggil via tf.py_function)
#     """
#     # Pastikan dalam format uint8 dan grayscale
#     if image_np.max() <= 1.0:
#         image_np = (image_np * 255).astype(np.uint8)
#     else:
#         image_np = image_np.astype(np.uint8)

#     # Convert ke grayscale jika RGB
#     if len(image_np.shape) == 3:
#         gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
#     else:
#         gray = image_np

#     # Otsu thresholding untuk mendapatkan area terang (paru-paru)
#     _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     # Morphological operations untuk membersihkan noise
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

#     # Opening: hilangkan noise kecil
#     cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

#     # Closing: tutup lubang-lubang kecil dalam paru
#     cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

#     # Normalize ke [0,1] dan pastikan ada channel dimension
#     mask = cleaned.astype(np.float32) / 255.0
#     mask = np.expand_dims(mask, axis=-1) # Add channel dimension

#     return mask

# def create_lung_mask(image):
#     """
#     Wrapper untuk create_lung_mask_opencv menggunakan tf.py_function
#     """
#     mask = tf.py_function(
#         func=create_lung_mask_opencv,
#         inp=[image],
#         Tout=tf.float32
#     )

#     # Set shape manually. The mask should have shape (height, width, 1)
#     mask.set_shape([224, 224, 1])

#     return mask

# def extract_lung_roi(image):
#     lung_mask = create_lung_mask(image)   # (224,224)
#     lung_mask = tf.expand_dims(lung_mask, axis=-1)  # (224,224,1)
#     roi_image = image * lung_mask
#     return roi_image

#     # Terapkan mask ke semua channel gambar
#     # TensorFlow broadcasting akan menangani ini jika image punya channel 3 dan mask punya channel 1
#     roi_image = image * lung_mask

#     return roi_image

# # ============================================================================
# # STEP 2: CONTRAST ENHANCEMENT dengan CLAHE
# # ============================================================================
# """
# CLAHE (Contrast Limited Adaptive Histogram Equalization)
# Tujuan: Menyamakan pencahayaan dan meningkatkan kontras paru-paru
# """

# def apply_clahe_opencv(image_np):
#     """
#     Helper function untuk CLAHE operations (akan dipanggil via tf.py_function)
#     """
#     # Pastikan dalam format uint8
#     if image_np.max() <= 1.0:
#         image_np = (image_np * 255).astype(np.uint8)
#     else:
#         image_np = image_np.astype(np.uint8)

#     # Jika RGB, convert ke LAB untuk CLAHE yang lebih baik
#     if len(image_np.shape) == 3:
#         lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)

#         # Apply CLAHE hanya pada L channel (Luminance)
#         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#         lab[:,:,0] = clahe.apply(lab[:,:,0])

#         # Convert kembali ke RGB
#         enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
#     else:
#         # Jika grayscale
#         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#         enhanced = clahe.apply(image_np)

#     # Normalize ke [0,1]
#     enhanced = enhanced.astype(np.float32) / 255.0

#     return enhanced

# def apply_clahe(image):
#     """
#     Wrapper untuk apply_clahe_opencv menggunakan tf.py_function
#     """
#     enhanced = tf.py_function(
#         func=apply_clahe_opencv,
#         inp=[image],
#         Tout=tf.float32
#     )

#     # Set shape manually
#     # Check if the input image has 3 channels and set the shape accordingly
#     if image.shape.ndims == 3 and image.shape[-1] == 3:
#       enhanced.set_shape([224, 224, 3])
#     else:
#       enhanced.set_shape([224, 224])


#     return enhanced

# # ============================================================================
# # STEP 3: PIXEL NORMALIZATION
# # ============================================================================
# """
# Normalisasi pixel untuk membuat training lebih stabil
# """

# def normalize_pixels(image):
#     """
#     Normalisasi pixel ke rentang [0,1] atau mean=0, std=1
#     """
#     # Method 1: Min-Max normalization ke [0,1]
#     image = tf.cast(image, tf.float32)
#     image = image / 255.0  # Jika input masih dalam range [0,255]

#     # Method 2: Z-score normalization (mean=0, std=1)
#     # Uncomment jika ingin pakai z-score
#     # mean = tf.reduce_mean(image)
#     # std = tf.math.reduce_std(image)
#     # image = (image - mean) / (std + 1e-8)  # +epsilon untuk avoid division by zero

#     return image

# # ============================================================================
# # STEP 4: DATA AUGMENTATION (Hanya untuk TRAINING)
# # ============================================================================
# """
# Augmentasi untuk menambah variasi data training
# PENTING: Hanya diterapkan pada data training, TIDAK pada validation/test
# """

# def augment_image(image, label):
#     """
#     Augmentasi geometrik dan fotometrik
#     """
#     # Geometrical augmentation
#     # Random rotation kecil (Â±10 derajat)
#     image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
#     image = tf.image.random_flip_left_right(image)

#     # Photometric augmentation
#     # Random brightness
#     image = tf.image.random_brightness(image, max_delta=0.1)

#     # Random contrast
#     image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

#     # Clamp values ke [0,1]
#     image = tf.clip_by_value(image, 0.0, 1.0)

#     return image, label

# # ============================================================================
# # STEP 5: PIPELINE PREPROCESSING LENGKAP
# # ============================================================================

# def preprocess_for_training(image, label):
#     """
#     Pipeline preprocessing untuk data TRAINING
#     Urutan: ROI â†’ CLAHE â†’ Normalization â†’ Augmentation
#     """
#     # Pastikan image dalam format float32 dan rentang [0,1]
#     # image = tf.cast(image, tf.float32) # image_dataset_from_directory already returns float32
#     image = image / 255.0 # Normalize from [0, 255] to [0, 1]

#     # 1. ROI Extraction
#     image = extract_lung_roi(image)

#     # 2. CLAHE Enhancement
#     image = apply_clahe(image)

#     # 3. Pixel Normalization (clipping)
#     image = tf.clip_by_value(image, 0.0, 1.0)

#     # 4. Data Augmentation (hanya untuk training)
#     image, label = augment_image(image, label)

#     return image, label

# def preprocess_for_validation(image, label):
#     """
#     Pipeline preprocessing untuk data VALIDATION/TEST
#     Urutan: ROI â†’ CLAHE â†’ Normalization (TANPA Augmentation)
#     """
#     # Pastikan image dalam format float32 dan rentang [0,1]
#     # image = tf.cast(image, tf.float32) # image_dataset_from_directory already returns float32
#     image = image / 255.0 # Normalize from [0, 255] to [0, 1]

#     # 1. ROI Extraction
#     image = extract_lung_roi(image)

#     # 2. CLAHE Enhancement
#     image = apply_clahe(image)

#     # 3. Pixel Normalization (clipping)
#     image = tf.clip_by_value(image, 0.0, 1.0)

#     # TIDAK ada augmentation untuk validation/test!

#     return image, label

# # ============================================================================
# # STEP 6: TERAPKAN PREPROCESSING KE DATASET
# # ============================================================================

# # Dataset yang sudah ada
# img_size = (224, 224)
# batch_size = 32

# train_ds = tf.keras.utils.image_dataset_from_directory(
#     "/root/.cache/kagglehub/competitions/srifoton-25-machine-learning-competition/train/train",
#     image_size=img_size,
#     batch_size=batch_size
# )

# val_ds = tf.keras.utils.image_dataset_from_directory(
#     "/root/.cache/kagglehub/competitions/srifoton-25-machine-learning-competition/val/val",
#     image_size=img_size,
#     batch_size=batch_size
# )

# test_ds = tf.keras.utils.image_dataset_from_directory(
#     "/root/.cache/kagglehub/competitions/srifoton-25-machine-learning-competition/test",
#     image_size=img_size,
#     batch_size=batch_size,
#     labels=None, # Test dataset has no labels
#     shuffle=False # Keep order for submission
# )


# # Terapkan preprocessing
# print("ğŸ”„ Menerapkan preprocessing...")

# # Training dataset: dengan augmentation
# train_ds_processed = (train_ds
#                      .map(preprocess_for_training, num_parallel_calls=tf.data.AUTOTUNE)
#                      .prefetch(tf.data.AUTOTUNE))

# # Validation dataset: tanpa augmentation
# val_ds_processed = (val_ds
#                    .map(preprocess_for_validation, num_parallel_calls=tf.data.AUTOTUNE)
#                    .prefetch(tf.data.AUTOTUNE))

# # Test dataset: tanpa augmentation
# test_ds_processed = (test_ds
#                     .map(lambda image: preprocess_for_validation(image, None), num_parallel_calls=tf.data.AUTOTUNE) # Pass None for label
#                     .prefetch(tf.data.AUTOTUNE))


# print("âœ… Preprocessing selesai!")
# print("\nDataset yang sudah dipreprocess:")
# print(f"ğŸ“ˆ Training: {train_ds_processed}")
# print(f"ğŸ“Š Validation: {val_ds_processed}")
# print(f"ğŸ§ª Test: {test_ds_processed}")


# # ============================================================================
# # STEP 7: VISUALISASI HASIL PREPROCESSING
# # ============================================================================

# def visualize_preprocessing_steps(dataset, num_images=2):
#     """
#     Visualisasi hasil setiap step preprocessing
#     """
#     # Ambil satu batch
#     for images, labels in dataset.take(1):
#         for i in range(min(num_images, images.shape[0])):
#             image = images[i]

#             # Convert ke format yang bisa diprocess (normalize to [0,1] for visualization)
#             image_normalized = tf.cast(image, tf.float32) / 255.0

#             fig, axes = plt.subplots(2, 3, figsize=(15, 10))

#             # Step 1: Original (normalized for display)
#             axes[0, 0].imshow(image_normalized.numpy())
#             axes[0, 0].set_title("1. Original Image")
#             axes[0, 0].axis('off')

#             # Step 2: ROI Extraction
#             # Need to apply ROI on the original image (before normalization in pipeline)
#             original_image_uint8 = tf.cast(images[i], tf.uint8) # Get back to original uint8
#             roi_image_raw = extract_lung_roi(original_image_uint8)
#             # Normalize ROI for display
#             roi_image_display = tf.clip_by_value(tf.cast(roi_image_raw, tf.float32)/255.0, 0.0, 1.0)
#             axes[0, 1].imshow(roi_image_display.numpy())
#             axes[0, 1].set_title("2. After ROI Extraction")
#             axes[0, 1].axis('off')

#             # Step 3: CLAHE
#             # Need to apply CLAHE on the ROI extracted from original image
#             clahe_image_raw = apply_clahe(roi_image_raw)
#             # Normalize CLAHE for display
#             clahe_image_display = tf.clip_by_value(tf.cast(clahe_image_raw, tf.float32)/255.0, 0.0, 1.0)
#             axes[0, 2].imshow(clahe_image_display.numpy())
#             axes[0, 2].set_title("3. After CLAHE Enhancement")
#             axes[0, 2].axis('off')

#             # Step 4: Normalization (clipping) - This step is already done in the pipeline after CLAHE
#             # We can show the result after CLAHE and before augmentation as the 'normalized' step
#             axes[1, 0].imshow(clahe_image_display.numpy()) # CLAHE result after normalization (clipping)
#             axes[1, 0].set_title("4. After Normalization (Clipping)")
#             axes[1, 0].axis('off')

#             # Step 5: Full preprocessing (training)
#             final_image, _ = preprocess_for_training(images[i], labels[i])
#             axes[1, 1].imshow(tf.clip_by_value(final_image, 0.0, 1.0).numpy())
#             axes[1, 1].set_title("5. Final (with Augmentation)")
#             axes[1, 1].axis('off')

#             # Step 6: Validation version (no augmentation)
#             val_image, _ = preprocess_for_validation(images[i], labels[i])
#             axes[1, 2].imshow(tf.clip_by_value(val_image, 0.0, 1.0).numpy())
#             axes[1, 2].set_title("6. Validation (no Augmentation)")
#             axes[1, 2].axis('off')


#             plt.suptitle(f"Preprocessing Steps - Image {i+1}", fontsize=16)
#             plt.tight_layout()
#             plt.show()
#         break # Only visualize for the first batch

# # Jalankan visualisasi
# print("\nğŸ–¼ï¸� Menampilkan hasil preprocessing...")
# # Use the original train_ds for visualization to show all steps from raw image
# visualize_preprocessing_steps(train_ds)

# print("\nğŸ“‹ RINGKASAN URUTAN PREPROCESSING:")
# print("1. âœ‚ï¸�  ROI Extraction - Fokus ke area paru-paru")
# print("2. ğŸŒŸ CLAHE Enhancement - Perbaiki kontras")
# print("3. ğŸ“Š Pixel Normalization - Standarisasi nilai pixel")
# print("4. ğŸ�² Data Augmentation - Variasi data (hanya training)")
# print("\nğŸ�¯ Dataset siap untuk training!")

