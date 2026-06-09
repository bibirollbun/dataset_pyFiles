# 1. Uninstall existing packages to ensure a clean slate
!pip uninstall scikit-learn imbalanced-learn -y

# 2. Install the known stable versions
!pip install scikit-learn==1.1.3
!pip install imbalanced-learn==0.10.1


import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Paths
data_dir = '/kaggle/input/aptos2019-blindness-detection'
train_csv_path = os.path.join(data_dir, 'train.csv')
img_dir = os.path.join(data_dir, 'train_images')
processed_dir = '/kaggle/working/processed_images'
os.makedirs(processed_dir, exist_ok=True)

# Step 1: Visualize Initial Data
def visualize_initial_data(num_samples_per_class=2):
    train_csv = pd.read_csv(train_csv_path)
    fig, axs = plt.subplots(5, num_samples_per_class, figsize=(10, 20))
    for class_label in range(5):
        class_samples = train_csv[train_csv['diagnosis'] == class_label].sample(num_samples_per_class, random_state=42)
        for i, (_, row) in enumerate(class_samples.iterrows()):
            img_path = os.path.join(img_dir, f"{row['id_code']}.png")
            img = Image.open(img_path)
            axs[class_label, i].imshow(img)
            axs[class_label, i].set_title(f"Class {class_label}: {row['id_code']}")
            axs[class_label, i].axis('off')
    plt.tight_layout()
    plt.show()

# Call
visualize_initial_data()






# Step 2: Image Processing Functions
def ben_graham_process(img, sigmaX=10):
    # Convert PIL to NumPy (RGB)
    img_array = np.array(img).astype(np.float32)
    blurred = np.zeros_like(img_array)
    for c in range(3):  # Process each RGB channel
        blurred[:, :, c] = gaussian_filter(img_array[:, :, c], sigma=sigmaX)
    # Ben Graham: 4*original - 4*blurred + 128
    enhanced = 4 * img_array - 4 * blurred + 128
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    return Image.fromarray(enhanced)

def circular_crop(img, threshold=10):
    # Use mean RGB for thresholding to keep color
    img_array = np.array(img)
    mean_rgb = img_array.mean(axis=2)
    mask = mean_rgb > threshold
    if not np.any(mask):
        return img
    
    # Find bounding box
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]
    
    # Compute center and radius for circular mask
    width, height = max_col - min_col + 1, max_row - min_row + 1
    x_center, y_center = (min_col + max_col) // 2, (min_row + max_row) // 2
    radius = int(min(width, height) / 2 * 0.95)  # 95% to avoid edge clipping
    
    # Create circular mask
    mask_img = Image.new('L', img.size, 0)
    draw = ImageDraw.Draw(mask_img)
    draw.ellipse((x_center - radius, y_center - radius, x_center + radius, y_center + radius), fill=255)
    
    # Apply mask to RGB image
    img_array = np.array(img)
    mask_array = np.array(mask_img)[:, :, np.newaxis]
    masked = np.where(mask_array == 255, img_array, 0).astype(np.uint8)
    img = Image.fromarray(masked)
    
    # Crop to square bounding box of circle
    img = img.crop((x_center - radius, y_center - radius, x_center + radius, y_center + radius))
    return img

def preprocess_image(img_path, target_size=512):
    img = Image.open(img_path).convert('RGB')
    # Order: Crop -> Resize -> Ben Graham (per Kaggle load_ben_color)
    img = circular_crop(img)
    img = img.resize((target_size, target_size), Image.BILINEAR)
    img = ben_graham_process(img, sigmaX=10)
    return img




import pandas as pd
# Step 3: Visualize Processed Data
def visualize_processed_data(num_samples_per_class=2):
    train_csv = pd.read_csv(train_csv_path)
    fig, axs = plt.subplots(5, 2 * num_samples_per_class, figsize=(20, 20))
    for class_label in range(5):
        class_samples = train_csv[train_csv['diagnosis'] == class_label].sample(num_samples_per_class, random_state=42)
        for i, (_, row) in enumerate(class_samples.iterrows()):
            img_path = os.path.join(img_dir, f"{row['id_code']}.png")
            original = Image.open(img_path).convert('RGB')
            processed = preprocess_image(img_path)
            
            axs[class_label, 2*i].imshow(original)
            axs[class_label, 2*i].set_title(f"Original Class {class_label}: {row['id_code']}")
            axs[class_label, 2*i].axis('off')
            
            axs[class_label, 2*i + 1].imshow(processed)
            axs[class_label, 2*i + 1].set_title(f"Processed Class {class_label}: {row['id_code']}")
            axs[class_label, 2*i + 1].axis('off')
    plt.tight_layout()
    plt.show()

# Call
visualize_processed_data()


# Step 4: Save Processed Images
def save_processed_images():
    train_csv = pd.read_csv(train_csv_path)
    for _, row in train_csv.iterrows():
        img_path = os.path.join(img_dir, f"{row['id_code']}.png")
        processed_img = preprocess_image(img_path)
        save_path = os.path.join(processed_dir, f"{row['id_code']}.png")
        processed_img.save(save_path)
    print(f"Processed and saved {len(train_csv)} images to {processed_dir}")

# Call
save_processed_images()


import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Paths
data_dir = '/kaggle/input/aptos2019-blindness-detection'
train_csv_path = os.path.join(data_dir, 'train.csv')
processed_dir = '/kaggle/working/processed_images'

# Step 1: Stratified Train/Val/Test Split
def get_stratified_split(csv_path, train_size=0.8, val_size=0.1):
    df = pd.read_csv(csv_path)
    # Train + Val vs Test
    train_val_df, test_df = train_test_split(
        df, test_size=0.1, stratify=df['diagnosis'], random_state=42
    )
    # Train vs Val
    relative_val_size = val_size / (train_size + val_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=relative_val_size, stratify=train_val_df['diagnosis'], random_state=42
    )
    return train_df, val_df, test_df

# Save splits (optional, for reference)
train_df, val_df, test_df = get_stratified_split(train_csv_path)
train_df.to_csv(os.path.join(processed_dir, 'train_split.csv'), index=False)
val_df.to_csv(os.path.join(processed_dir, 'val_split.csv'), index=False)
test_df.to_csv(os.path.join(processed_dir, 'test_split.csv'), index=False)


import pandas as pd
import numpy as np
import torch
import os

# Paths
data_dir = '/kaggle/input/aptos2019-blindness-detection'
train_csv_path = os.path.join(data_dir, 'train.csv')
processed_dir = '/kaggle/working/processed_images'
feature_dir = '/kaggle/working/features'
os.makedirs(feature_dir, exist_ok=True)

# Load stratified splits
train_df = pd.read_csv(os.path.join(processed_dir, 'train_split.csv'))
val_df = pd.read_csv(os.path.join(processed_dir, 'val_split.csv'))
test_df = pd.read_csv(os.path.join(processed_dir, 'test_split.csv'))

# Compute class weights for sampling and loss
def get_class_weights(df):
    class_counts = df['diagnosis'].value_counts().sort_index()
    total_samples = len(df)
    num_classes = 5
    weights = [total_samples / (num_classes * class_counts[i]) if i in class_counts else 0 for i in range(num_classes)]
    print(total_samples)
    print(class_counts)
    weights = np.array(weights)
    weights = weights / weights.sum()
    return weights

class_weights = get_class_weights(train_df)
sample_weights = [class_weights[row['diagnosis']] for _, row in train_df.iterrows()]
sample_weights = torch.tensor(sample_weights, dtype=torch.float)
loss_weights = torch.tensor(class_weights, dtype=torch.float).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Class weights for sampling/loss: {class_weights}")


get_class_weights(val_df)


import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Custom Dataset
class APTOSDataset(Dataset):
    def __init__(self, df, processed_dir, transform=None):
        self.data = df
        self.processed_dir = processed_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = f"{row['id_code']}.png"
        img_path = os.path.join(self.processed_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(row['diagnosis'], dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

from torchvision import transforms

# Assuming pre-resized images (e.g., 512x512) are loaded by APTOSDataset

train_transforms = transforms.Compose([
    # 1. Random Crop (Forces model to learn features at different locations)
    # This acts as a robust 'scale' augmentation, taking a 224x224 patch
    # from the larger, pre-resized image.
    transforms.Resize((256,256)), 

    # 2. Geometric Augmentations (Slightly more aggressive)
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30), # Increased from 15
    transforms.RandomAffine(
        degrees=0,
        scale=(0.8, 1.2),  # Scaling the 224x224 crop itself
        shear=10           # Increased from 5
    ),
    
    # 3. Policy Augmentation (Strong regularization)
    # transforms.TrivialAugmentWide(), # Powerful, random, state-of-the-art augmentation

    # 4. Color/Intensity Augmentations (Slightly more aggressive)
    transforms.ColorJitter(
        brightness=0.3,   # Increased from 0.2
        contrast=0.3,     # Increased from 0.2
        saturation=0.3,   # Increased from 0.2
        hue=0.08          # Increased from 0.05
    ),
    
    # Final steps (Always kept at the end)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    # Use CenterCrop to extract the central 224x224 region consistently
    transforms.Resize((256,256)), 
    
    # Final steps
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Step 4: Visualize Augmented Images
def visualize_augmented_data(df, processed_dir, transform, num_samples_per_class=2):
    dataset = APTOSDataset(df, processed_dir, transform=transform)
    fig, axs = plt.subplots(5, num_samples_per_class, figsize=(15, 25))
    for class_label in range(5):
        class_samples = df[df['diagnosis'] == class_label].sample(num_samples_per_class, random_state=42)
        for i, (_, row) in enumerate(class_samples.iterrows()):
            img_name = f"{row['id_code']}.png"
            img_path = os.path.join(processed_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            # Apply transforms
            img_tensor = transform(img)
            # Convert back to PIL for display (denormalize)
            img_array = img_tensor.permute(1, 2, 0).numpy()
            img_array = img_array * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_array = np.clip(img_array, 0, 1)
            axs[class_label, i].imshow(img_array)
            axs[class_label, i].set_title(f"Augmented Class {class_label}: {row['id_code']}")
            axs[class_label, i].axis('off')
    plt.tight_layout()
    plt.show()

# Call
visualize_augmented_data(train_df, processed_dir, train_transforms)


from torch.utils.data import DataLoader, WeightedRandomSampler

# DataLoaders
train_dataset = APTOSDataset(train_df, processed_dir, transform=train_transforms)
val_dataset = APTOSDataset(val_df, processed_dir, transform=val_test_transforms)
test_dataset = APTOSDataset(test_df, processed_dir, transform=val_test_transforms)

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_df), replacement=True)
train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

# Test batch
for images, labels in train_loader:
    print(f"Train batch: images shape {images.shape}, labels {labels}")
    break
for images, labels in val_loader:
    print(f"Val batch: images shape {images.shape}, labels {labels}")
    break


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import models
# from tqdm.notebook import tqdm

# # Initialize ResNet50
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = models.resnet50(pretrained=True)
# model.fc = nn.Linear(model.fc.in_features, 5)  # 5 classes
# model = model.to(device)

# # Freeze all layers except layer4 and fc
# for name, param in model.named_parameters():
#     if not ('layer4' in name or 'fc' in name):
#         param.requires_grad = False

# # Optimizer and loss
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
# criterion = nn.CrossEntropyLoss(weight=loss_weights)

# # Fine-tuning loop
# model.train()
# num_epochs = 20
# for epoch in tqdm(range(num_epochs), desc="Fine-tuning Epochs"):
#     running_loss = 0.0
#     correct = 0
#     total = 0
#     for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#     print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%")


# import torch
# import numpy as np

# # Extract features
# model.eval()
# def extract_features(loader, model, save_path):
#     features, labels = [], []
#     with torch.no_grad():
#         for images, lbls in loader:
#             images = images.to(device)
#             feats = model(images).cpu().numpy().reshape(len(images), -1)
#             features.append(feats)
#             labels.append(lbls.numpy())
#     features = np.concatenate(features)
#     labels = np.concatenate(labels)
#     np.savez(save_path, features=features, labels=labels)
#     return features, labels

# # Extract and save
# train_features, train_labels = extract_features(train_loader, model, os.path.join(feature_dir, 'train_features.npz'))
# val_features, val_labels = extract_features(val_loader, model, os.path.join(feature_dir, 'val_features.npz'))
# test_features, test_labels = extract_features(test_loader, model, os.path.join(feature_dir, 'test_features.npz'))
# print(f"Train features shape: {train_features.shape}, Val features shape: {val_features.shape}")





import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler # New Import
from torchvision import models
from tqdm.notebook import tqdm
import numpy as np
from sklearn.metrics import f1_score # New Import
# from sklearn.preprocessing import StandardScaler # Not needed in this block

# === Device setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_MODEL_PATH = "best_f1_model.pth" # Define save path

# === Initialize DenseNet121 ===
model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 5) # 5 classes
model = model.to(device)

# === Freeze/Unfreeze Layers (More Aggressive Fine-Tuning) ===
for name, param in model.features.named_parameters():
    param.requires_grad = False # freeze backbone

# UNFREEZE the last TWO dense blocks for better feature learning (CRITICAL)
for name, param in model.features.denseblock3.named_parameters(): # Unfreeze denseblock3
    param.requires_grad = True
for name, param in model.features.denseblock4.named_parameters(): # Unfreeze denseblock4
    param.requires_grad = True

# === Define optimizer, loss, and scheduler ===
# Only optimize parameters that are unfrozen (requires_grad = True)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

criterion = nn.CrossEntropyLoss() # Correctly without weights due to WeightedRandomSampler

# Add a Learning Rate Scheduler for stability
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max',       # Monitor a metric that should increase (Macro F1)
    factor=0.1, 
    patience=7,
    verbose=True
)

# === Training & Validation Loop ===
num_epochs = 100
best_f1 = 0.0

for epoch in tqdm(range(num_epochs), desc="Fine-tuning Epochs"):
    # --- TRAINING PHASE ---
    model.train()
    running_loss, all_preds, all_labels = 0.0, [], []

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    train_loss = running_loss / len(train_loader)
    train_f1 = f1_score(all_labels, all_preds, average='macro') # Macro F1 for minority classes

    # --- VALIDATION PHASE (CRITICAL) ---
    model.eval()
    val_loss, val_preds, val_labels = 0.0, [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Val", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_macro_f1 = f1_score(val_labels, val_preds, average='macro') # Macro F1 for minority classes

    print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
    print(f"Train Loss: {train_loss:.4f} | Train Macro F1: {train_f1:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Macro F1: {val_macro_f1:.4f}")

    # --- CHECKPOINTING (Optimize for F1) ---
    if val_macro_f1 > best_f1:
        best_f1 = val_macro_f1
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"Model saved! New best Macro F1: {best_f1:.4f}")
    
    # Step the scheduler
    scheduler.step(val_macro_f1)


# === Feature Extraction Model (Load Best Model) ===





# Reload the architecture
feature_extractor = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
feature_extractor.classifier = nn.Identity() # remove classifier layer

# Load the weights from the model that achieved the BEST VALIDATION F1-SCORE
try:
    feature_extractor.load_state_dict(torch.load(BEST_MODEL_PATH), strict=False)
    print(f"Loaded best model from {BEST_MODEL_PATH}")
except FileNotFoundError:
    # Fallback if no model was saved (e.g., stopping early, first run)
    feature_extractor.load_state_dict(model.state_dict(), strict=False)
    print("WARNING: Best model not found. Using final epoch model for feature extraction.")

feature_extractor = feature_extractor.to(device)
feature_extractor.eval()

# ... (Rest of your feature extraction code is the same) ...

# === Save directory ===
feature_dir = "features"
os.makedirs(feature_dir, exist_ok=True)

# === Extract and save features ===
train_features, train_labels = feature_extractor(train_loader, feature_extractor, os.path.join(feature_dir, 'train_features.npz'))
val_features, val_labels = feature_extractor(val_loader, feature_extractor, os.path.join(feature_dir, 'val_features.npz'))
test_features, test_labels = feature_extractor(test_loader, feature_extractor, os.path.join(feature_dir, 'test_features.npz'))

print(f"âœ… Feature extraction complete!")
print(f"Train features shape: {train_features.shape}, Val features shape: {val_features.shape}")


# === Feature extraction function ===

def extract_features(loader, model, save_path):

    all_features, all_labels = [], []

    with torch.no_grad():

        for images, lbls in tqdm(loader, desc=f"Extracting {os.path.basename(save_path)}"):

            images = images.to(device)

            feats = model(images).cpu().numpy()

            all_features.append(feats)

            all_labels.append(lbls.numpy())

    features = np.concatenate(all_features)

    labels = np.concatenate(all_labels)



    # âš ï¸� DO NOT SCALE HERE â€” leave it raw for downstream SMOTE + scaling

    np.savez(save_path, features=features, labels=labels)

    return features, labels

# Load the weights from the model that achieved the BEST VALIDATION F1-SCORE
try:
    feature_extractor.load_state_dict(torch.load(BEST_MODEL_PATH), strict=False)
    print(f"Loaded best model from {BEST_MODEL_PATH}")
except FileNotFoundError:
    # Fallback if no model was saved (e.g., stopping early, first run)
    feature_extractor.load_state_dict(model.state_dict(), strict=False)
    print("WARNING: Best model not found. Using final epoch model for feature extraction.")

feature_extractor = feature_extractor.to(device)
feature_extractor.eval()

# ... (Rest of your feature extraction code is the same) ...

# === Save directory ===
feature_dir = "features"
os.makedirs(feature_dir, exist_ok=True)

# === Extract and save features ===
train_features, train_labels = extract_features(train_loader, feature_extractor, os.path.join(feature_dir, 'train_features.npz'))
val_features, val_labels = extract_features(val_loader, feature_extractor, os.path.join(feature_dir, 'val_features.npz'))
test_features, test_labels = extract_features(test_loader, feature_extractor, os.path.join(feature_dir, 'test_features.npz'))

print(f"âœ… Feature extraction complete!")
print(f"Train features shape: {train_features.shape}, Val features shape: {val_features.shape}")


len(train_features)


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import (
    LogisticRegression, PassiveAggressiveClassifier, Perceptron,
    RidgeClassifier, SGDClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
# CHANGED: Import SMOTE instead of SMOTEENN
from imblearn.over_sampling import SMOTE 
from itertools import cycle
import joblib

# === Plot style and Paths (No change) ===
plt.style.use('seaborn-v0_8')
sns.set(font_scale=1.2)

feature_dir = 'features'
output_dir = "tuned_classifiers"
roc_dir = os.path.join("images", "tunedModel")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(roc_dir, exist_ok=True)
print(f"Output directory created: {output_dir}")

# === Load data (No change) ===
try:
    train_data = np.load(os.path.join(feature_dir, 'train_features.npz'))
    val_data = np.load(os.path.join(feature_dir, 'val_features.npz'))
    test_data = np.load(os.path.join(feature_dir, 'test_features.npz'))
except FileNotFoundError:
    print(f"Error: Feature files not found in {feature_dir}. Please ensure the previous step ran correctly.")
    exit()

train_features, train_labels = train_data['features'], train_data['labels']
val_features, val_labels = val_data['features'], val_data['labels']
test_features, test_labels = test_data['features'], test_data['labels']
print(f"Initial training set size: {len(train_features)} samples")

# ====================================================================
# === CORE MODIFICATION: SC1 -> PCA -> SC2 -> SMOTE ===
# ====================================================================

# --- 1. StandardScaler (SC1) ---
print("Applying SC1 (StandardScaler)...")
scaler1 = StandardScaler()
train_features_sc1 = scaler1.fit_transform(train_features)
val_features_sc1 = scaler1.transform(val_features)
test_features_sc1 = scaler1.transform(test_features)
joblib.dump(scaler1, os.path.join(output_dir, 'scaler1.pkl'))

# --- 2. PCA (Dimensionality Reduction) ---
# NOTE: Set n_components based on desired explained variance (e.g., 90%) or a fixed number
PCA_VARIANCE = 0.95 
print(f"Applying PCA (retaining {PCA_VARIANCE * 100}%% variance)...")
pca = PCA(n_components=PCA_VARIANCE, random_state=42)
train_features_pca = pca.fit_transform(train_features_sc1)
val_features_pca = pca.transform(val_features_sc1)
test_features_pca = pca.transform(test_features_sc1)
joblib.dump(pca, os.path.join(output_dir, 'pca.pkl'))
print(f"Reduced features from {train_features.shape[1]} to {train_features_pca.shape[1]} dimensions.")

# --- 3. StandardScaler (SC2) ---
print("Applying SC2 (StandardScaler) on PCA components...")
scaler2 = StandardScaler()
train_features_sc2 = scaler2.fit_transform(train_features_pca)
val_features_sc2 = scaler2.transform(val_features_pca)
test_features_sc2 = scaler2.transform(test_features_pca)
joblib.dump(scaler2, os.path.join(output_dir, 'scaler2.pkl'))
print(f"Final training features shape before SMOTE: {train_features_sc2.shape}")

# --- 4. Apply SMOTE for oversampling (CRITICAL for imbalance) ---
print("âš–ï¸� Balancing transformed training data using SMOTE...")
# CHANGED: Use SMOTE instead of SMOTEENN
smote = SMOTE(sampling_strategy = 'auto', random_state=42)
# SMOTE is applied to the final transformed training features (train_features_sc2)
train_features_res, train_labels_res = smote.fit_resample(train_features_sc2, train_labels)

# Rename the final scaled feature sets for clarity in subsequent code
train_features_scaled = train_features_res
val_features_scaled = val_features_sc2
test_features_scaled = test_features_sc2

print(f"Resampled training set size: {train_features_res.shape[0]} samples")
print(f"Final training features shape: {train_features_scaled.shape}")


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import (LogisticRegression, Perceptron,
                                 PassiveAggressiveClassifier, RidgeClassifier,
                                 SGDClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

cpu_models = {
    # 1ï¸�âƒ£ Support Vector Machine (Grid Search)
    "SVM": {
        "model": SVC(probability=True, random_state=42),
        "search": "random",
        "params": {
            # Total combinations: 720 (Grid Search will check all)
            "C": [1e-5, 1e-3, 0.01, 0.1, 1, 5, 10, 50, 100, 1000],
            "gamma": ['scale', 'auto', 1e-5, 1e-3, 0.1, 1, 5, 10],
            "kernel": ["rbf", "poly", "sigmoid"],
            "degree": [2, 3, 4]
        },
        "n_iters" : 200,
    },  # <-- CORRECTED: MISSING COMMA WAS HERE!

    # 2ï¸�âƒ£ Random Forest (Grid Search)
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "search": "random",
        "params": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None]
        },
        "n_iters" : 200,
    },

    # 3ï¸�âƒ£ Extra Trees (Grid Search)
    "ExtraTrees": {
        "model": ExtraTreesClassifier(random_state=42),
        "search": "grid",
        "params": {
            "n_estimators": [100, 200, 400],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10]
        }
    },

    # 4ï¸�âƒ£ K-Nearest Neighbors (Grid Search)
    "KNN": {
        "model": KNeighborsClassifier(),
        "search": "grid",
        "params": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan", "minkowski"]
        }
    },

    # 5ï¸�âƒ£ Naive Bayes (no tuning)
    "NaiveBayes": GaussianNB(),

    # 6ï¸�âƒ£ Multi-Layer Perceptron (Grid Search)
    "MLP": {
        "model": MLPClassifier(max_iter=800, random_state=42),
        "search": "grid",
        "params": {
            "hidden_layer_sizes": [(64,), (128,), (256,), (128, 64), (256, 128), (512, 256), (100, 100)],
            "activation": ["relu", "tanh", "logistic"],
            "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 0.1],
            "learning_rate_init": [0.001, 0.005, 0.01, 0.05, 0.1]
        },
    },
    
    # 7ï¸�âƒ£ Passive Aggressive
    "PassiveAggressive": {
        "model": PassiveAggressiveClassifier(max_iter=1000, random_state=42),
        "search": "grid",
        "params": {
            "C": [0.01, 0.1, 1, 10],
            "loss": ["hinge", "squared_hinge"]
        }
    },

    # 8ï¸�âƒ£ Ridge Classifier
    "RidgeClassifier": {
        "model": RidgeClassifier(),
        "search": "grid",
        "params": {
            "alpha": [0.1, 1, 10, 100],
            "solver": ["auto", "svd", "cholesky", "lsqr"]
        }
    },

    # 9ï¸�âƒ£ Stochastic Gradient Descent
    "SGD": {
        "model": SGDClassifier(max_iter=2000, random_state=42),
        "search": "grid",
        "params": {
            "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
            "penalty": ["l2", "l1", "elasticnet"],
            "loss": ["hinge", "log_loss", "modified_huber"],
            "learning_rate": ["optimal", "invscaling", "adaptive"]
        }
    },

    # ğŸ”Ÿ Nearest Centroid
    "NearestCentroid": {
        "model": NearestCentroid(),
        "search": "grid",
        "params": {"metric": ["euclidean", "manhattan"]}
    },

    # 11ï¸�âƒ£ Decision Tree
    "DecisionTree": {
        "model": DecisionTreeClassifier(random_state=42),
        "search": "grid",
        "params": {
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["gini", "entropy", "log_loss"]
        }
    },

    # 12ï¸�âƒ£ Perceptron
    "Perceptron": {
        "model": Perceptron(max_iter=1000, random_state=42),
        "search": "grid",
        "params": {"penalty": [None, "l2", "l1", "elasticnet"],
                    "alpha": [1e-5, 1e-4, 1e-3]}
    },

    # 13ï¸�âƒ£ Logistic Regression
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000, multi_class='multinomial', random_state=42),
        "search": "grid",
        "params": {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "solver": ["lbfgs", "saga", "newton-cg"]
        }
    },
}


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.linear_model import (
    LogisticRegression, PassiveAggressiveClassifier, Perceptron,
    RidgeClassifier, SGDClassifier
)
from sklearn.neural_network import MLPClassifier
from scipy.stats import randint, loguniform
from imblearn.combine import SMOTEENN
from itertools import cycle
import joblib

# === Plot style ===
plt.style.use('seaborn-v0_8')
sns.set(font_scale=1.2)

# === Paths ===
feature_dir = '/kaggle/working/features'
output_dir = "tuned_classifiers"
roc_dir = os.path.join("images", "tunedModel")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(roc_dir, exist_ok=True)

# === Load data ===
train_data = np.load(os.path.join(feature_dir, 'train_features.npz'))
val_data = np.load(os.path.join(feature_dir, 'val_features.npz'))
test_data = np.load(os.path.join(feature_dir, 'test_features.npz'))

train_features, train_labels = train_data['features'], train_data['labels']
print('Len' + str(len(train_features)))
val_features, val_labels = val_data['features'], val_data['labels']
test_features, test_labels = test_data['features'], test_data['labels']

# === Apply SMOTEENN for hybrid resampling ===
print("âš–ï¸� Balancing training data using SMOTEENN...")
smote_enn = SMOTEENN(random_state=42)
train_features_res, train_labels_res = smote_enn.fit_resample(train_features, train_labels)
print(f"Resampled training set size: {train_features_res.shape[0]} samples")

# === Standardize features ===
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features_res)
val_features_scaled = scaler.transform(val_features)
test_features_scaled = scaler.transform(test_features)
joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
print('Len' + str(len(train_features_scaled)))
# === Define classifiers ===
classifiers = {
    'dt': DecisionTreeClassifier(random_state=42),
    'rf': RandomForestClassifier(random_state=42, n_jobs=-1),
    'svm': SVC(probability=True, random_state=42),
    'knn': KNeighborsClassifier(n_jobs=-1),
    'lr': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    'et': ExtraTreesClassifier(random_state=42, n_jobs=-1),
    'mlp': MLPClassifier(max_iter=1000, random_state=42),
    'nc': NearestCentroid(),
    'pa': PassiveAggressiveClassifier(max_iter=1000, random_state=42),
    'perceptron': Perceptron(random_state=42),
    'ridge': RidgeClassifier(random_state=42),
    'sgd': SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)
}

# === Parameter distributions for RandomizedSearch ===
param_distributions = {
    'dt': {'max_depth': [5, 10, 20, None],
           'min_samples_split': randint(2, 10),
           'min_samples_leaf': randint(1, 5)},
    'rf': {'n_estimators': randint(100, 300),
           'max_depth': [10, 20, None],
           'min_samples_split': randint(2, 10),
           'min_samples_leaf': randint(1, 5)},
    'svm': {'C': loguniform(0.1, 10),
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']},
    'knn': {'n_neighbors': randint(3, 10),
            'weights': ['uniform', 'distance'],
            'p': [1, 2]},
    'lr': {'C': loguniform(0.01, 10),
           'solver': ['lbfgs', 'liblinear'],
           'penalty': ['l2']},
    'et': {'n_estimators': randint(100, 300),
           'max_depth': [10, 20, None],
           'min_samples_split': randint(2, 10),
           'min_samples_leaf': randint(1, 5)},
    'mlp': {'hidden_layer_sizes': [(64,), (128,), (128, 64), (256, 128)],
            'alpha': loguniform(1e-5, 1e-2),
            'learning_rate': ['constant', 'adaptive']},
    'nc': {},
    'pa': {'C': loguniform(0.01, 10),
           'loss': ['hinge', 'squared_hinge']},
    'perceptron': {'penalty': [None, 'l2', 'l1'],
                   'alpha': loguniform(1e-5, 1e-2),
                   'eta0': [0.1, 1.0]},
    'ridge': {'alpha': loguniform(0.1, 100)},
    'sgd': {'alpha': loguniform(1e-5, 1e-2),
            'learning_rate': ['constant', 'optimal', 'adaptive'],
            'eta0': [0.001, 0.01, 0.1]}
}

# === ROC plotting ===
def plot_roc(model_name, model, X, y, save_dir):
    """Plot one-vs-rest ROC for multiclass classification."""
    if not hasattr(model, "predict_proba"):
        print(f"âš ï¸� {model_name} does not support predict_proba, skipping ROC.")
        return

    classes = np.unique(y)
    y_bin = label_binarize(y, classes=classes)
    y_score = model.predict_proba(X)

    plt.figure(figsize=(7, 6))
    colors = cycle(plt.cm.tab10.colors)

    for i, color in zip(range(len(classes)), colors):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'Class {classes[i]} (AUC={roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_roc.png"))
    plt.close()

# === Training & Evaluation ===
results = []

for name, clf in classifiers.items():
    print(f"\nğŸ”¹ Tuning {name}...")

    params = param_distributions[name]
    if params:
        search = RandomizedSearchCV(
            clf,
            param_distributions=params,
            n_iter=20,
            cv=5,
            scoring='f1_weighted',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        search.fit(train_features_scaled, train_labels_res)
        best_model = search.best_estimator_
        print(f"âœ… Best Params: {search.best_params_}")
    else:
        best_model = clf.fit(train_features_scaled, train_labels_res)

    # Save model
    joblib.dump(best_model, os.path.join(output_dir, f"{name}_tuned_model.pkl"))

    # Evaluate
    val_pred = best_model.predict(val_features_scaled)
    test_pred = best_model.predict(test_features_scaled)
    val_f1 = f1_score(val_labels, val_pred, average='weighted')
    test_f1 = f1_score(test_labels, test_pred, average='weighted')

    print(f"Validation F1: {val_f1:.4f} | Test F1: {test_f1:.4f}")
    results.append((name, val_f1, test_f1))

    # ROC
    plot_roc(name, best_model, test_features_scaled, test_labels, roc_dir)

# === Summary ===
print("\nğŸ“Š Final F1-scores:")
for name, val_f1, test_f1 in results:
    print(f"{name:10s} | Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f}")

print(f"\nâœ… All ROC curves saved to: {roc_dir}")
print(f"âœ… Tuned models saved to: {output_dir}")



import os
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# === Paths ===
feature_dir = '/kaggle/working/features'  # or your feature directory
output_dir = "tuned_classifiers"
meta_dir = "meta_features"
os.makedirs(meta_dir, exist_ok=True)

# === Load training features & labels ===
train_data = np.load(os.path.join(feature_dir, 'train_features.npz'))
train_features = train_data['features']
train_labels = train_data['labels']

# === Load scaler (used in base models) and scale train features ===
scaler = joblib.load(os.path.join(output_dir, 'scaler.pkl'))
train_features_scaled = scaler.transform(train_features)

# === Load tuned base models ===
classifiers = {
    'dt': joblib.load(os.path.join(output_dir, 'dt_tuned_model.pkl')),
    'rf': joblib.load(os.path.join(output_dir, 'rf_tuned_model.pkl')),
    'svm': joblib.load(os.path.join(output_dir, 'svm_tuned_model.pkl')),
    'knn': joblib.load(os.path.join(output_dir, 'knn_tuned_model.pkl')),
    'lr': joblib.load(os.path.join(output_dir, 'lr_tuned_model.pkl')),
    'et': joblib.load(os.path.join(output_dir, 'et_tuned_model.pkl')),
    'mlp': joblib.load(os.path.join(output_dir, 'mlp_tuned_model.pkl')),
    'nc': joblib.load(os.path.join(output_dir, 'nc_tuned_model.pkl')),
    'pa': joblib.load(os.path.join(output_dir, 'pa_tuned_model.pkl')),
    'perceptron': joblib.load(os.path.join(output_dir, 'perceptron_tuned_model.pkl')),
    'ridge': joblib.load(os.path.join(output_dir, 'ridge_tuned_model.pkl')),
    'sgd': joblib.load(os.path.join(output_dir, 'sgd_tuned_model.pkl'))
}

# === Base models that support predict_proba ===
proba_classifiers = ['dt', 'rf', 'svm', 'knn', 'lr', 'et', 'mlp', 'sgd']

# === Prepare one-hot encoder for models without predict_proba ===
unique_labels = np.unique(train_labels)
onehot_encoder = OneHotEncoder(sparse_output=False, categories=[unique_labels])

# === Generate predictions (meta-features) ===
meta_features = []
for name, clf in classifiers.items():
    print(f"Generating training meta-features for {name}...")
    
    if name in proba_classifiers:
        preds = clf.predict_proba(train_features_scaled)
    else:
        pred_labels = clf.predict(train_features_scaled)
        preds = onehot_encoder.fit_transform(pred_labels.reshape(-1, 1))
    
    np.save(os.path.join(meta_dir, f"{name}_train_predictions.npy"), preds)
    meta_features.append(preds)

# === Stack meta-features from all base models ===
meta_features = np.hstack(meta_features)

# === Scale meta-features ===
meta_scaler = StandardScaler()
meta_features_scaled = meta_scaler.fit_transform(meta_features)
joblib.dump(meta_scaler, os.path.join(meta_dir, 'meta_scaler.pkl'))

# === Save meta-features and labels ===
np.save(os.path.join(meta_dir, 'meta_features_train.npy'), meta_features_scaled)
np.save(os.path.join(meta_dir, 'train_labels.npy'), train_labels)

print("âœ… Meta-features for TRAIN set generated and saved successfully.")



import os
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# === Configuration ===
feature_dir = '/kaggle/working/features'   # Path to feature files
output_dir = "tuned_classifiers"           # Folder with trained base models
meta_dir = "meta_features"                 # Folder to store meta features
os.makedirs(meta_dir, exist_ok=True)

# === Load validation data ===
val_data = np.load(os.path.join(feature_dir, 'val_features.npz'))
train_data = np.load(os.path.join(feature_dir, 'train_features.npz'))

val_features = val_data['features']
val_labels = val_data['labels']
train_labels = train_data['labels']

# === Standardize validation features using saved scaler ===
scaler = joblib.load(os.path.join(output_dir, 'scaler.pkl'))
val_features_scaled = scaler.transform(val_features)

# === Load tuned models ===
classifiers = {
    'dt': joblib.load(os.path.join(output_dir, 'dt_tuned_model.pkl')),
    'rf': joblib.load(os.path.join(output_dir, 'rf_tuned_model.pkl')),
    'svm': joblib.load(os.path.join(output_dir, 'svm_tuned_model.pkl')),
    'knn': joblib.load(os.path.join(output_dir, 'knn_tuned_model.pkl')),
    'lr': joblib.load(os.path.join(output_dir, 'lr_tuned_model.pkl')),
    'et': joblib.load(os.path.join(output_dir, 'et_tuned_model.pkl')),
    'mlp': joblib.load(os.path.join(output_dir, 'mlp_tuned_model.pkl')),
    'nc': joblib.load(os.path.join(output_dir, 'nc_tuned_model.pkl')),
    'pa': joblib.load(os.path.join(output_dir, 'pa_tuned_model.pkl')),
    'perceptron': joblib.load(os.path.join(output_dir, 'perceptron_tuned_model.pkl')),
    'ridge': joblib.load(os.path.join(output_dir, 'ridge_tuned_model.pkl')),
    'sgd': joblib.load(os.path.join(output_dir, 'sgd_tuned_model.pkl'))
}

# Models that support predict_proba
proba_classifiers = ['dt', 'rf', 'svm', 'knn', 'lr', 'et', 'mlp', 'sgd']

# === Prepare meta features ===
meta_features = []
unique_labels = np.unique(train_labels)
onehot_encoder = OneHotEncoder(sparse_output=False, categories=[unique_labels.reshape(-1,)])

print("\nGenerating meta-features for stacking...\n")

for name, clf in classifiers.items():
    print(f"Processing {name}...")
    # Use probabilities if available; otherwise use one-hot encoded predictions
    if name in proba_classifiers:
        preds = clf.predict_proba(val_features_scaled)
    else:
        pred_labels = clf.predict(val_features_scaled)
        preds = onehot_encoder.fit_transform(pred_labels.reshape(-1, 1))
    
    # Save individual model predictions (for debugging or analysis)
    np.save(os.path.join(meta_dir, f"{name}_val_predictions.npy"), preds)
    meta_features.append(preds)

# === Combine and standardize meta features ===
meta_features = np.hstack(meta_features)
meta_scaler = StandardScaler()
meta_features_scaled = meta_scaler.fit_transform(meta_features)

# === Save meta data for next (stacking) stage ===
joblib.dump(meta_scaler, os.path.join(meta_dir, 'meta_scaler.pkl'))
np.save(os.path.join(meta_dir, 'meta_features_val.npy'), meta_features_scaled)
np.save(os.path.join(meta_dir, 'val_labels.npy'), val_labels)

print("\nâœ… Meta-feature generation complete.")
print(f"Saved stacked features to: {meta_dir}")



import os
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# === Paths ===
feature_dir = '/kaggle/working/features'
output_dir = "tuned_classifiers"
meta_dir = "meta_features"
os.makedirs(meta_dir, exist_ok=True)

# === Load test data ===
test_features = np.load(os.path.join(feature_dir, 'test_features.npz'))['features']
test_labels = np.load(os.path.join(feature_dir, 'test_features.npz'))['labels']
train_labels = np.load(os.path.join(feature_dir, 'train_features.npz'))['labels']

# === Load scaler and standardize test features ===
scaler = joblib.load(os.path.join(output_dir, 'scaler.pkl'))
test_features_scaled = scaler.transform(test_features)

# === Load tuned base models ===
classifiers = {
    'dt': joblib.load(os.path.join(output_dir, 'dt_tuned_model.pkl')),
    'rf': joblib.load(os.path.join(output_dir, 'rf_tuned_model.pkl')),
    'svm': joblib.load(os.path.join(output_dir, 'svm_tuned_model.pkl')),
    'knn': joblib.load(os.path.join(output_dir, 'knn_tuned_model.pkl')),
    'lr': joblib.load(os.path.join(output_dir, 'lr_tuned_model.pkl')),
    'et': joblib.load(os.path.join(output_dir, 'et_tuned_model.pkl')),
    'mlp': joblib.load(os.path.join(output_dir, 'mlp_tuned_model.pkl')),
    'nc': joblib.load(os.path.join(output_dir, 'nc_tuned_model.pkl')),
    'pa': joblib.load(os.path.join(output_dir, 'pa_tuned_model.pkl')),
    'perceptron': joblib.load(os.path.join(output_dir, 'perceptron_tuned_model.pkl')),
    'ridge': joblib.load(os.path.join(output_dir, 'ridge_tuned_model.pkl')),
    'sgd': joblib.load(os.path.join(output_dir, 'sgd_tuned_model.pkl'))
}

proba_classifiers = ['dt', 'rf', 'svm', 'knn', 'lr', 'et', 'mlp', 'sgd']
unique_labels = np.unique(train_labels)
onehot_encoder = OneHotEncoder(sparse_output=False, categories=[unique_labels])

# === Generate meta features for test ===
meta_features_test = []
for name, clf in classifiers.items():
    print(f"Generating test predictions for {name}...")
    if name in proba_classifiers:
        pred = clf.predict_proba(test_features_scaled)
    else:
        pred_labels = clf.predict(test_features_scaled)
        pred = onehot_encoder.fit_transform(pred_labels.reshape(-1, 1))
    meta_features_test.append(pred)
    np.save(os.path.join(meta_dir, f"{name}_test_predictions.npy"), pred)

# === Stack and standardize meta features ===
meta_features_test = np.hstack(meta_features_test)
meta_scaler = joblib.load(os.path.join(meta_dir, 'meta_scaler.pkl'))  # same scaler from val
meta_features_test_scaled = meta_scaler.transform(meta_features_test)

# === Save for next step ===
np.save(os.path.join(meta_dir, 'meta_features_test.npy'), meta_features_test_scaled)
np.save(os.path.join(meta_dir, 'test_labels.npy'), test_labels)

print("Meta-features for test set generated and saved in", meta_dir)



import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

# === Paths ===
meta_dir = "meta_features"
os.makedirs("images/stacking", exist_ok=True)

# === Load meta-feature datasets ===
meta_features_train = np.load(os.path.join(meta_dir, 'meta_features_train.npy'))
train_labels = np.load(os.path.join(meta_dir, 'train_labels.npy'))

meta_features_val = np.load(os.path.join(meta_dir, 'meta_features_val.npy'))
val_labels = np.load(os.path.join(meta_dir, 'val_labels.npy'))

meta_features_test = np.load(os.path.join(meta_dir, 'meta_features_test.npy'))
test_labels = np.load(os.path.join(meta_dir, 'test_labels.npy'))

# === Define parameter grid for RandomizedSearch ===
param_distributions = {
    'hidden_layer_sizes': [(64,), (128,), (128, 64), (256, 128), (256, 128, 64)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [1e-5, 1e-4, 1e-3, 1e-2],
    'learning_rate': ['constant', 'adaptive'],
    'batch_size': [32, 64, 128, 256],
}

# === Base MLP model ===
base_mlp = MLPClassifier(
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
)

# === Randomized Search ===
print("ğŸ”� Running Randomized Search for Meta MLP...")
random_search = RandomizedSearchCV(
    estimator=base_mlp,
    param_distributions=param_distributions,
    n_iter=20,  # Try 20 random combinations
    scoring='f1_weighted',
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

random_search.fit(meta_features_train, train_labels)
print(f"\nâœ… Best Parameters: {random_search.best_params_}")
print(f"ğŸ�† Best CV Score: {random_search.best_score_:.4f}")

# === Train the best model on full training meta-features ===
best_mlp = random_search.best_estimator_
best_mlp.fit(meta_features_train, train_labels)

# === Save tuned meta model ===
joblib.dump(best_mlp, os.path.join(meta_dir, 'meta_mlp_tuned_model.pkl'))

# === Evaluate on VALIDATION set ===
val_preds = best_mlp.predict(meta_features_val)
print("\n=== Meta MLP Model Evaluation (Validation Set) ===")
print(classification_report(val_labels, val_preds, digits=4))

# === Confusion Matrix (Validation) ===
cm = confusion_matrix(val_labels, val_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Meta MLP Confusion Matrix (Validation Set)")
plt.tight_layout()
plt.savefig("images/stacking/meta_mlp_confusion_matrix_val.png")
plt.show()

# === Evaluate on TEST set ===
test_preds = best_mlp.predict(meta_features_test)
print("\n=== Meta MLP Model Final Test Evaluation ===")
print(classification_report(test_labels, test_preds, digits=4))



print(f"Ensemble Val Accuracy: {accuracy_score(val_labels, val_preds):.4f}")
print(f"Ensemble Val Kappa: {cohen_kappa_score(test_labels, test_preds, weights='quadratic'):.4f}")


import os
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set plot style
plt.style.use('seaborn')
sns.set(font_scale=1.2)

# Define paths
feature_dir = '/kaggle/working/features'  # Update with your Kaggle path
output_dir = "tuned_classifiers"
meta_dir = "meta_features"
plots_dir = "plots"
os.makedirs(meta_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Load data
val_features = np.load(os.path.join(feature_dir, 'val_features.npz'))['features']
val_labels = np.load(os.path.join(feature_dir, 'val_features.npz'))['labels']
train_labels = np.load(os.path.join(feature_dir, 'train_features.npz'))['labels']

# Load scaler and standardize validation features
scaler = joblib.load(os.path.join(output_dir, 'scaler.pkl'))
val_features_scaled = scaler.transform(val_features)

# Load tuned models
classifiers = {
    'dt': joblib.load(os.path.join(output_dir, 'dt_tuned_model.pkl')),
    'rf': joblib.load(os.path.join(output_dir, 'rf_tuned_model.pkl')),
    'svm': joblib.load(os.path.join(output_dir, 'svm_tuned_model.pkl')),
    'knn': joblib.load(os.path.join(output_dir, 'knn_tuned_model.pkl')),
    'lr': joblib.load(os.path.join(output_dir, 'lr_tuned_model.pkl')),
    'et': joblib.load(os.path.join(output_dir, 'et_tuned_model.pkl')),
    'mlp': joblib.load(os.path.join(output_dir, 'mlp_tuned_model.pkl')),
    'nc': joblib.load(os.path.join(output_dir, 'nc_tuned_model.pkl')),
    'pa': joblib.load(os.path.join(output_dir, 'pa_tuned_model.pkl')),
    'perceptron': joblib.load(os.path.join(output_dir, 'perceptron_tuned_model.pkl')),
    'ridge': joblib.load(os.path.join(output_dir, 'ridge_tuned_model.pkl')),
    'sgd': joblib.load(os.path.join(output_dir, 'sgd_tuned_model.pkl'))
}

# Classifiers that support predict_proba
proba_classifiers = ['dt', 'rf', 'svm', 'knn', 'lr', 'et', 'mlp', 'sgd']
unique_labels = np.unique(train_labels)
class_names = [str(i) for i in unique_labels]  # Replace with actual class names if available

# Generate predictions and visualizations
meta_features = []
f1_scores = {}
onehot_encoder = OneHotEncoder(sparse_output=False, categories=[unique_labels])
for name, clf in classifiers.items():
    print(f"Generating predictions for {name}...")
    if name in proba_classifiers:
        pred = clf.predict_proba(val_features_scaled)
        pred_labels = clf.predict(val_features_scaled)  # For confusion matrix
    else:
        pred_labels = clf.predict(val_features_scaled)
        pred = onehot_encoder.fit_transform(pred_labels.reshape(-1, 1))
    meta_features.append(pred)
    np.save(os.path.join(meta_dir, f"{name}_val_predictions.npy"), pred)
    
    # Compute F1 score
    f1 = f1_score(val_labels, pred_labels, average='weighted')
    f1_scores[name] = f1
    
    # Plot confusion matrix
    cm = confusion_matrix(val_labels, pred_labels, labels=unique_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix for {name.upper()}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{name}_confusion_matrix.png'))
    plt.close()

# Stack and standardize meta-features
meta_features = np.hstack(meta_features)
meta_scaler = StandardScaler()
meta_features_scaled = meta_scaler.fit_transform(meta_features)
joblib.dump(meta_scaler, os.path.join(meta_dir, 'meta_scaler.pkl'))
np.save(os.path.join(meta_dir, 'meta_features_val.npy'), meta_features_scaled)
np.save(os.path.join(meta_dir, 'val_labels.npy'), val_labels)

# Plot F1 scores
plt.figure(figsize=(12, 6))
sns.barplot(x=list(f1_scores.keys()), y=list(f1_scores.values()))
plt.title('Validation F1 Scores for Base Classifiers')
plt.xlabel('Classifier')
plt.ylabel('F1 Score (Weighted)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'base_classifiers_val_f1_scores.png'))
plt.close()


print("Meta-features generation complete. Plots saved in", plots_dir)


import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set plot style
plt.style.use('seaborn')
sns.set(font_scale=1.2)

# Define paths
# Define paths
feature_dir = '/kaggle/working/features'
output_dir = "tuned_classifiers"
meta_dir = "meta_features"
model_dir = "meta_model"
submissions_dir = "submissions"
plots_dir = "plots"
os.makedirs(submission_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Load test data
test_data = np.load(os.path.join(feature_dir, 'test_features.npz'))
test_features = test_data['features']
if 'ids' in test_data:
    test_ids = test_data['ids']
else:
    test_ids = np.arange(len(test_features))

# Load scaler and standardize test features
scaler = joblib.load(os.path.join(output_dir, 'scaler.pkl'))
test_features_scaled = scaler.transform(test_features)

# Load tuned models
classifiers = {
    'dt': joblib.load(os.path.join(output_dir, 'dt_tuned_model.pkl')),
    'rf': joblib.load(os.path.join(output_dir, 'rf_tuned_model.pkl')),
    'svm': joblib.load(os.path.join(output_dir, 'svm_tuned_model.pkl')),
    'knn': joblib.load(os.path.join(output_dir, 'knn_tuned_model.pkl')),
    'lr': joblib.load(os.path.join(output_dir, 'lr_tuned_model.pkl')),
    'et': joblib.load(os.path.join(output_dir, 'et_tuned_model.pkl')),
    'mlp': joblib.load(os.path.join(output_dir, 'mlp_tuned_model.pkl')),
    'nc': joblib.load(os.path.join(output_dir, 'nc_tuned_model.pkl')),
    'pa': joblib.load(os.path.join(output_dir, 'pa_tuned_model.pkl')),
    'perceptron': joblib.load(os.path.join(output_dir, 'perceptron_tuned_model.pkl')),
    'ridge': joblib.load(os.path.join(output_dir, 'ridge_tuned_model.pkl')),
    'sgd': joblib.load(os.path.join(output_dir, 'sgd_tuned_model.pkl'))
}

# Classifiers that support predict_proba
proba_classifiers = ['dt', 'rf', 'svm', 'knn', 'lr', 'et', 'mlp', 'sgd']
train_labels = np.load(os.path.join(feature_dir, 'train_features.npz'))['labels']
unique_labels = np.unique(train_labels)
class_names = [str(i) for i in unique_labels]  # Replace with actual class names if available
onehot_encoder = OneHotEncoder(sparse_output=False, categories=[unique_labels])

# Generate test predictions
test_meta_features = []
for name, clf in classifiers.items():
    print(f"Generating test predictions for {name}...")
    if name in proba_classifiers:
        pred = clf.predict_proba(test_features_scaled)
    else:
        pred = clf.predict(test_features_scaled).reshape(-1, 1)
        pred = onehot_encoder.fit_transform(pred)
    test_meta_features.append(pred)

# Stack and standardize test meta-features
test_meta_features = np.hstack(test_meta_features)
meta_scaler = joblib.load(os.path.join(meta_dir, 'meta_scaler.pkl'))
test_meta_features_scaled = meta_scaler.transform(test_meta_features)

# Load meta-model and predict
meta_model = joblib.load(os.path.join(model_dir, 'mlp_meta_model.pkl'))
test_predictions = meta_model.predict(test_meta_features_scaled)

# Plot prediction distribution
plt.figure(figsize=(10, 6))
sns.histplot(test_predictions, bins=len(unique_labels), stat='count')
plt.title('Distribution of Predicted Classes in Test Set')
plt.xlabel('Predicted Class')
plt.ylabel('Count')
plt.xticks(ticks=unique_labels, labels=class_names)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'test_prediction_distribution.png'))
plt.close()

# Optional: Plot average probability scores (if meta-model supports predict_proba)
test_probabilities = meta_model.predict_proba(test_meta_features_scaled)
avg_probs = np.mean(test_probabilities, axis=0)
plt.figure(figsize=(8, 6))
sns.heatmap(avg_probs.reshape(1, -1), annot=True, fmt='.4f', cmap='Blues', xticklabels=class_names)
plt.title('Average Predicted Probabilities per Class')
plt.xlabel('Class')
plt.ylabel('Average Probability')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'test_avg_probabilities.png'))
plt.close()

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'class': test_predictions
})
submission_file = os.path.join(submission_dir, 'submission.csv')
submission.to_csv(submission_file, index=False)
print(f"Submission saved to {submission_file}. Plots saved in {plots_dir}")













