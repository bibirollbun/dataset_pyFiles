import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import sys
import subprocess
from PIL import Image
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Install and import timm for state-of-the-art models
try:
    import timm
    print(f"timm version: {timm.__version__} successfully imported.")
except ImportError:
    print("timm not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "timm"])
    import timm
    print(f"timm version: {timm.__version__} installed and imported.")


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
class CFG:
    # General
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data paths
    DATA_DIR = "/kaggle/input/2025-bamboo-summer-competiton-dl-pr/"
    
    # Model parameters
    MODEL_NAME = 'efficientnet_b4'
    IMG_SIZE = 384
    NUM_CLASSES = 75
    
    # Training parameters
    BATCH_SIZE = 16 # Reduced for larger image size
    EPOCHS = 30 # Can be adjusted based on validation performance
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    LABEL_SMOOTHING = 0.1
    
    # TTA (Test-Time Augmentation)
    TTA_STEPS = 5

# Set random seeds for reproducibility
np.random.seed(CFG.SEED)
torch.manual_seed(CFG.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG.SEED)

# --- Data Loading and Preparation ---
train_df = pd.read_csv(os.path.join(CFG.DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(CFG.DATA_DIR, "test.csv"))
sample_submission = pd.read_csv(os.path.join(CFG.DATA_DIR, "sample_submission.csv"))

# Encode labels
label_encoder = LabelEncoder()
train_df['label_encoded'] = label_encoder.fit_transform(train_df['label'])

X_train, X_val, y_train, y_val = train_test_split(
    train_df['filename'],
    train_df['label_encoded'],
    test_size=0.15, # Using a slightly smaller validation set
    random_state=CFG.SEED,
    stratify=train_df['label_encoded']
)

# --- Data Augmentation and Datasets ---
def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
            transforms.TrivialAugmentWide(), # Advanced augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class ButterflyDataset(Dataset):
    def __init__(self, filenames, labels, root_dir, transform=None, is_test=False):
        self.filenames = filenames
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.filenames[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            return image
        else:
            return image, torch.tensor(self.labels[idx], dtype=torch.long)

# Create datasets and dataloaders
train_dataset = ButterflyDataset(X_train.values, y_train.values, os.path.join(CFG.DATA_DIR, 'train'), transform=get_transforms(is_train=True))
val_dataset = ButterflyDataset(X_val.values, y_val.values, os.path.join(CFG.DATA_DIR, 'train'), transform=get_transforms(is_train=False))
test_dataset = ButterflyDataset(test_df['filename'].values, None, os.path.join(CFG.DATA_DIR, 'test'), transform=get_transforms(is_train=False), is_test=True)

train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


# --- Model Definition ---
class ButterflyClassifier(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super(ButterflyClassifier, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
    def forward(self, x):
        return self.model(x)

# --- Training and Evaluation Loop ---
def run_training(model, train_loader, val_loader, epochs):
    criterion = nn.CrossEntropyLoss(label_smoothing=CFG.LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LEARNING_RATE, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs//3, T_mult=1, eta_min=1e-6)
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_corrects = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(CFG.DEVICE), labels.to(CFG.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_corrects += torch.sum(preds == labels.data)
            pbar.set_postfix(loss=loss.item())

        scheduler.step()
        
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]"):
                inputs, labels = inputs.to(CFG.DEVICE), labels.to(CFG.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        train_acc = train_corrects.double() / len(train_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            print(f"✅ New best validation accuracy: {best_val_acc:.4f}")
            torch.save(best_model_state, f"{CFG.MODEL_NAME}_best.pth")

    print(f"Training finished. Best Val Acc: {best_val_acc:.4f}")
    model.load_state_dict(best_model_state)
    return model

# --- Prediction with Test-Time Augmentation (TTA) ---
def predict_with_tta(model, test_loader, tta_steps):
    model.eval()
    tta_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    ])
    
    all_preds = []
    
    with torch.no_grad():
        for images in tqdm(test_loader, desc="Predicting with TTA"):
            images = images.to(CFG.DEVICE)
            batch_probs = torch.zeros((images.size(0), CFG.NUM_CLASSES), device=CFG.DEVICE)
            
            # Original image
            outputs = model(images)
            batch_probs += torch.softmax(outputs, dim=1)
            
            # Augmented images
            for _ in range(tta_steps - 1):
                aug_images = tta_transforms(images)
                outputs = model(aug_images)
                batch_probs += torch.softmax(outputs, dim=1)
            
            batch_probs /= tta_steps
            _, preds = torch.max(batch_probs, 1)
            all_preds.extend(preds.cpu().numpy())
            
    return np.array(all_preds)

# --- Main Execution ---
print(f"Using device: {CFG.DEVICE}")
print(f"Training model: {CFG.MODEL_NAME}")

# Initialize and train the model
model = ButterflyClassifier(CFG.MODEL_NAME, CFG.NUM_CLASSES).to(CFG.DEVICE)
model = run_training(model, train_loader, val_loader, epochs=CFG.EPOCHS)

# Get predictions using TTA
predictions = predict_with_tta(model, test_loader, tta_steps=CFG.TTA_STEPS)
predicted_labels = label_encoder.inverse_transform(predictions)

# --- Create Submission File ---
submission_df = pd.DataFrame({
    'filename': test_df['filename'],
    'label': predicted_labels
})
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully!")
print("Top 10 predictions:")
print(submission_df.head(10))

# Display prediction distribution
plt.figure(figsize=(12, 6))
sns.countplot(y=submission_df['label'], order=submission_df['label'].value_counts().iloc[:25].index, palette='viridis')
plt.title('Top 25 Predicted Butterfly Species')
plt.xlabel('Count')
plt.ylabel('Species')
plt.tight_layout()
plt.show()

