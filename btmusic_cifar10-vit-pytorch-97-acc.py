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


# Cell 1: Installations
!pip install -q timm py7zr


# Cell 2: Imports & Configuration
import os
import pandas as pd
import numpy as np
from PIL import Image
import py7zr
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import timm

# --- Configuration ---
class CFG:
    # Kaggle environment paths
    data_path = '/kaggle/input/cifar-10/'
    train_archive = os.path.join(data_path, 'train.7z')
    test_archive = os.path.join(data_path, 'test.7z')
    train_labels_csv = os.path.join(data_path, 'trainLabels.csv')
    
    # Output paths
    extract_path = '/kaggle/working/cifar10/'
    train_img_path = os.path.join(extract_path, 'train')
    test_img_path = os.path.join(extract_path, 'test')
    model_save_path = '/kaggle/working/best_model.pth'

    # Model and Training parameters
    model_name = 'vit_base_patch16_224.augreg_in21k' # A powerful pre-trained ViT
    img_size = 224
    batch_size = 32
    epochs = 10  # A good starting point for fine-tuning
    learning_rate = 1e-4
    weight_decay = 1e-5
    label_smoothing = 0.1
    validation_split = 0.1 # Use 10% of training data for validation
    
    # Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # CIFAR-10 specific
    num_classes = 10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Using device: {CFG.device}")


# Cell 3: Extract Data
os.makedirs(CFG.extract_path, exist_ok=True)

print("Extracting train.7z...")
with py7zr.SevenZipFile(CFG.train_archive, mode='r') as z:
    z.extractall(path=CFG.extract_path)
print("Train images extracted.")

print("Extracting test.7z...")
with py7zr.SevenZipFile(CFG.test_archive, mode='r') as z:
    z.extractall(path=CFG.extract_path)
print("Test images extracted.")


# Cell 4: Dataset & DataLoaders

# Load labels and create a mapping for faster lookup
train_df = pd.read_csv(CFG.train_labels_csv)
# Create a dictionary mapping class names to integer indices
class_to_idx = {name: i for i, name in enumerate(CFG.class_names)}
# Map string labels to integer indices in the DataFrame
train_df['label_idx'] = train_df['label'].map(class_to_idx)
# Create a mapping from image ID to label index for the dataset
id_to_label = dict(zip(train_df['id'], train_df['label_idx']))


class Cifar10Dataset(Dataset):
    def __init__(self, img_dir, image_ids, labels_map=None, transform=None):
        self.img_dir = img_dir
        self.image_ids = image_ids
        self.labels_map = labels_map
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, f"{image_id}.png")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.labels_map:
            label = self.labels_map[image_id]
            return image, torch.tensor(label, dtype=torch.long)
        else:
            return image, image_id

# Define data augmentations and normalization
# ViT models expect a specific normalization scheme from their ImageNet pre-training
data_transforms = {
    'train': T.Compose([
        T.Resize((CFG.img_size, CFG.img_size)),
        T.TrivialAugmentWide(),  # State-of-the-art automatic augmentation
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # ViT standard normalization
    ]),
    'val': T.Compose([
        T.Resize((CFG.img_size, CFG.img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
}

# --- Create Datasets ---
all_train_ids = train_df['id'].values
full_dataset = Cifar10Dataset(CFG.train_img_path, all_train_ids, id_to_label)

# Split into training and validation sets
val_size = int(len(full_dataset) * CFG.validation_split)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Apply the correct transforms to each split
train_dataset.dataset.transform = data_transforms['train']
val_dataset.dataset.transform = data_transforms['val']

# --- Create DataLoaders ---
train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")


# Cell 5: Model, Loss, Optimizer Setup

# Load pre-trained ViT model
model = timm.create_model(
    CFG.model_name,
    pretrained=True,
    num_classes=CFG.num_classes
)
model.to(CFG.device)

# Loss Function with Label Smoothing
# This regularization prevents the model from becoming overconfident
loss_fn = nn.CrossEntropyLoss(label_smoothing=CFG.label_smoothing)

# AdamW Optimizer - better than Adam for generalization
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)

# OneCycleLR Scheduler for faster convergence and better performance
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=CFG.learning_rate,
    steps_per_epoch=len(train_loader),
    epochs=CFG.epochs
)

# Gradient Scaler for Automatic Mixed Precision (AMP)
# This speeds up training significantly on modern GPUs
scaler = torch.cuda.amp.GradScaler()


# Cell 6: Training & Validation Loop

def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, scaler, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(loader, desc="Training", leave=False)
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Mixed precision training context
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
        
        # Backward pass with scaler
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        scheduler.step() # Step scheduler at each batch for OneCycleLR
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])
        
    return total_loss / len(loader)

def validate_one_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Validating", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct_predictions.double() / total_samples
    return avg_loss, accuracy

# --- Main Training Loop ---
best_val_accuracy = 0.0

for epoch in range(CFG.epochs):
    print(f"--- Epoch {epoch+1}/{CFG.epochs} ---")
    
    train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, scaler, CFG.device)
    val_loss, val_accuracy = validate_one_epoch(model, val_loader, loss_fn, CFG.device)
    
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")
    
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        print(f"New best model found! Saving to {CFG.model_save_path}")
        torch.save(model.state_dict(), CFG.model_save_path)

print("\n--- Training Finished ---")
print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")


# Cell 7: Inference and Submission

# Load the best model weights
model.load_state_dict(torch.load(CFG.model_save_path))
model.to(CFG.device)
model.eval()

# Prepare the test dataset
test_image_files = sorted([f for f in os.listdir(CFG.test_img_path) if f.endswith('.png')])
test_ids = [int(os.path.splitext(f)[0]) for f in test_image_files]

test_dataset = Cifar10Dataset(
    img_dir=CFG.test_img_path,
    image_ids=test_ids,
    labels_map=None,  # No labels for test set
    transform=data_transforms['val'] # Use validation transforms for test
)

test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=os.cpu_count())

# --- Generate Predictions ---
predictions = []
with torch.no_grad():
    progress_bar = tqdm(test_loader, desc="Predicting on Test Set")
    for inputs, image_ids in progress_bar:
        inputs = inputs.to(CFG.device)
        
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            
        _, preds = torch.max(outputs, 1)
        
        for i, pred_idx in enumerate(preds):
            image_id = image_ids[i].item()
            label_name = CFG.class_names[pred_idx.item()]
            predictions.append((image_id, label_name))

# --- Create Submission File ---
submission_df = pd.DataFrame(predictions, columns=['id', 'label'])
submission_df = submission_df.sort_values(by='id')
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file created successfully!")
print(submission_df.head())




