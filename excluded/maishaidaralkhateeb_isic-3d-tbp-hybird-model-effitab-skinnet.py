# weeakness consider the meta data (meta image)


!pip install torch_geometric


import os
import random
import numpy as np
import pandas as pd
from glob import glob
from tqdm.notebook import tqdm

import h5py
import cv2
import matplotlib.pyplot as plt

# PyTorch Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# For metrics
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedGroupKFold

from torchvision.models import vit_l_16, ViT_L_16_Weights
import albumentations as A
from albumentations.pytorch import ToTensorV2
# For GNNs
import torch_geometric




pip install -U albumentations


import torch

class CFG:
    verbose = True  # Verbosity
    seed = 42  # Random seed
    neg_sample = 0.01  # Downsample negative class
    pos_sample = 5.0   # Upsample positive class
    image_size = (128, 128)  # Input image size
    epochs = 10  # Training epochs
    batch_size = 256  # Batch size
    lr = 3e-5  # Learning rate
    num_classes = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mixup_alpha = 1.0  # MixUp augmentation parameter
    cutmix_alpha = 1.0  # CutMix augmentation parameter
    backbone = "vit_l_16"  # Change backbone dynamically


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
set_seed(CFG.seed)



BASE_PATH = "/kaggle/input/isic-2024-challenge"
BASE_PATH_SYNTH = "/kaggle/input/synthetic-skin-lesions"



import os

print(os.path.exists(f'{BASE_PATH}/train-metadata.csv'))  # Should return True
print(os.path.exists(f'{BASE_PATH}/test-metadata.csv'))   # Should return True



import pandas as pd

df = pd.read_csv(f'{BASE_PATH}/train-metadata.csv', encoding='utf-8', low_memory=False)
df['iddx_4'] = df['iddx_4'].astype(str)  # Convert to string (if needed)
df['iddx_5'] = pd.to_numeric(df['iddx_5'], errors='coerce')  # Convert to float

# Load the CSV file into a DataFrame
metadata_path = f"{BASE_PATH}/train-metadata.csv"
df = pd.read_csv(metadata_path)



# Display the first 5 rows and the first 5 columns
print(df.iloc[:3, :8].to_string(index=False))



df = pd.read_csv(f'{BASE_PATH}/test-metadata.csv', delimiter=';')



pip install h5py



import h5py
import numpy as np

BASE_PATH = "/kaggle/input/isic-2024-challenge"

with h5py.File(f'{BASE_PATH}/train-image.hdf5', 'r') as f:
    keys = list(f.keys())  # Get all keys
    # print("Keys in the HDF5 file (all):", keys)
    print("Keys in the HDF5 file (first 10):", keys[:10])  # Display the first ten






import h5py
import numpy as np

BASE_PATH = "/kaggle/input/isic-2024-challenge"

# Load train images
with h5py.File(f"{BASE_PATH}/train-image.hdf5", 'r') as f:
    # Iterate through all keys and load images into a list
    train_images = []
    for key in list(f.keys()):
        train_images.append(np.array(f[key]))
    
    # Convert the list of images to a NumPy array
    train_images = np.stack(train_images)

# Check the shape of the loaded images
print(f"Loaded {len(train_images)} images with shape: {train_images.shape}")




with h5py.File(f"{BASE_PATH}/train-image.hdf5", 'r') as f:
    image = np.array(f['ISIC_0015670'])
    print(f"Image shape: {image.shape}")



import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt

BASE_PATH = "/kaggle/input/isic-2024-challenge"

# Load metadata
metadata_path = f"{BASE_PATH}/train-metadata.csv"
df = pd.read_csv(metadata_path, encoding='utf-8', low_memory=False)

# Load HDF5 data
hdf5_path = f"{BASE_PATH}/train-image.hdf5"
with h5py.File(hdf5_path, 'r') as f:
    images = {isic_id: np.array(f[isic_id]) for isic_id in df['isic_id']}




# Ensure that the order of images matches the order in the metadata
# This might require you to sort or index your metadata DataFrame accordingly

# For instance, if your metadata has an 'id' column that matches image indices
df = df.sort_values('isic_id').reset_index(drop=True)
train_images = train_images[df.index.values]



# Load training metadata

import pandas as pd
import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt

BASE_PATH = "/kaggle/input/isic-2024-challenge"

# Load metadata
metadata_path = f"{BASE_PATH}/train-metadata.csv"
df = pd.read_csv(metadata_path)
df = df.ffill()  # Forward fill missing values

# Load HDF5 data
hdf5_path = f"{BASE_PATH}/train-image.hdf5"
with h5py.File(hdf5_path, 'r') as f:
    # Example: Load the first image
    first_isic_id = df['isic_id'].iloc[0]
    data = f[first_isic_id][...]
    image_array = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

# Visualize the image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.title(f"Image ID: {first_isic_id}")
plt.show()



# Load testing metadata
testing_df = pd.read_csv(f'{BASE_PATH}/test-metadata.csv')
testing_df = testing_df.ffill()

print("Training Data:")
display(df.head(2))
print("Testing Data:")
display(testing_df.head(2))


class_distribution = df['target'].value_counts(normalize=True) * 100
print("Class Distribution Before Sampling (%):")
print(class_distribution)







# Downsample negative class
neg_df = df[df['target'] == 0].sample(frac=CFG.neg_sample, random_state=CFG.seed)
# Upsample positive class
pos_df = df[df['target'] == 1].sample(frac=CFG.pos_sample, replace=True, random_state=CFG.seed)

# Combine
df = pd.concat([neg_df, pos_df]).reset_index(drop=True)

# Shuffle the dataframe
df = df.sample(frac=1, random_state=CFG.seed).reset_index(drop=True)

# Check new class distribution
new_class_distribution = df['target'].value_counts(normalize=True) * 100
print("\nClass Distribution After Sampling (%):")
print(new_class_distribution)



import pandas as pd
import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt

BASE_PATH = "/kaggle/input/isic-2024-challenge"


# Load HDF5 files
training_validation_hdf5 = h5py.File(f"{BASE_PATH}/train-image.hdf5", 'r')
testing_hdf5 = h5py.File(f"{BASE_PATH}/test-image.hdf5", 'r')



isic_id = df['isic_id'].iloc[0]

# Image as Byte String

byte_string = training_validation_hdf5[isic_id][()]
print(f"Byte String: {byte_string[:20]}....")

# Convert byte string to numpy array

nparr = np.frombuffer(byte_string, np.uint8)

# Decode image

image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)[..., ::-1]  # Convert BGR to RGB

# Display the image

plt.imshow(image)
plt.title(f"Sample Image - ID: {isic_id}")
plt.axis('off')
plt.show()
# <a id='data-split'></a>


df = df.reset_index(drop=True)
df['fold'] = -1

#sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CFG.seed)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, y=df['target'], groups=df['patient_id'])):
    df.loc[val_idx, 'fold'] = fold

# Use fold 0 for validation
#train_df = df[df['fold'] != 0].reset_index(drop=True)
#val_df = df[df['fold'] == 0].reset_index(drop=True)

# Create fold-based train and val sets
    fold_train_df = df.iloc[train_idx].reset_index(drop=True)
    fold_val_df = df.iloc[val_idx].reset_index(drop=True)

print(f"Number of training samples: {len(fold_train_df)}")
print(f"Number of validation samples: {len(fold_val_df)}")

# Check class distribution
print("\nTraining Class Distribution:")
print(fold_train_df['target'].value_counts())
print("\nValidation Class Distribution:")
print(fold_val_df['target'].value_counts())
#<a id='preprocess-tabular-features'></a>




# Categorical features to be one-hot encoded
CATEGORICAL_COLUMNS = ["sex", "anatom_site_general", "tbp_tile_type", "tbp_lv_location"]

# Numerical features to be normalized
NUMERIC_COLUMNS = ["age_approx", "tbp_lv_nevi_confidence", "clin_size_long_diam_mm",
                   "tbp_lv_areaMM2", "tbp_lv_area_perim_ratio", "tbp_lv_color_std_mean",
                   "tbp_lv_deltaLBnorm", "tbp_lv_minorAxisMM"]

# All features
FEAT_COLS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS



from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Fit encoders on training data
def fit_encoders(df):
    # OneHotEncoder for categorical features
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    ohe.fit(df[CATEGORICAL_COLUMNS])
    
    # StandardScaler for numerical features
    scaler = StandardScaler()
    scaler.fit(df[NUMERIC_COLUMNS])
    
    return ohe, scaler

ohe, scaler = fit_encoders(df)



def preprocess_features(df, ohe, scaler):
    # Encode categorical features
    cat_features = ohe.transform(df[CATEGORICAL_COLUMNS])
    
    # Scale numerical features
    num_features = scaler.transform(df[NUMERIC_COLUMNS])
    
    # Combine features
    features = np.hstack([cat_features, num_features])
    
    return features
#<a id='create-datasets'></a>



# After (using albumentations):
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.RandomResizedCrop(CFG.image_size[0], CFG.image_size[1], scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.CoarseDropout(max_holes=8,
                    max_height=CFG.image_size[0]//10, 
                    max_width=CFG.image_size[1]//10, 
                    p=0.5),
    A.Normalize(),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(CFG.image_size[0], CFG.image_size[1]),
    A.Normalize(),
    ToTensorV2(),
])



class ISICDataset(Dataset):
    def __init__(self, df, hdf5_file, ohe, scaler, is_train=True, transform=None):
        self.df = df.reset_index(drop=True)
        self.hdf5_file = hdf5_file
        self.ohe = ohe
        self.scaler = scaler
        self.is_train = is_train
        self.transform = transform
        
        # Preprocess features
        self.features = preprocess_features(self.df, self.ohe, self.scaler)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        isic_id = self.df.loc[idx, 'isic_id']
        byte_string = self.hdf5_file[isic_id][()]
        nparr = np.frombuffer(byte_string, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, CFG.image_size)
        
        if self.transform:
            # Albumentations requires the input as a dictionary
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            # Convert to tensor and normalize using PyTorch
            image = transforms.ToTensor()(image)
        
        # Get features
        features = self.features[idx]
        features = torch.tensor(features, dtype=torch.float32)
        
        if self.is_train:
            label = self.df.loc[idx, 'target']
            label = torch.tensor([label], dtype=torch.float32)
            return image, features, label
        else:
            return image, features



def inference_with_tta(model, test_loader, tta=5):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for images, features in test_loader:
            images = images.to(CFG.device)
            features = features.to(CFG.device)
            preds_tta = []
            for _ in range(tta):
                augmented_images = some_tta_transform(images)
                outputs = model(augmented_images, features)
                preds_tta.append(outputs.cpu().numpy())
            preds = np.mean(preds_tta, axis=0)
            all_preds.append(preds)
    return np.concatenate(all_preds)




# Training dataset and dataloader
train_dataset = ISICDataset(fold_train_df, training_validation_hdf5, ohe, scaler, is_train=True, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=4)

# Validation dataset and dataloader
val_dataset = ISICDataset(fold_val_df, training_validation_hdf5, ohe, scaler, is_train=True, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=4)

# Testing dataset and dataloader
test_dataset = ISICDataset(testing_df, testing_hdf5, ohe, scaler, is_train=False, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=4)



# Get a batch from the training loader
images, features, labels = next(iter(train_loader))

print("Images shape:", images.shape)
print("Features shape:", features.shape)
print("Labels shape:", labels.shape)
# <a id='build-model'></a>


import torch
import torch.nn as nn
from torchvision import models

# Assume CFG and num_features are defined elsewhere
# For example:
# class CFG:
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assume num_features is known from your dataset
# For example: num_features = train_dataset.features.shape[1]

# Assuming you have train_dataset ready




weights_path = "/kaggle/input/efficientnet_v2_s-dd5fe13b/pytorch/default/1/efficientnet_v2_s-dd5fe13b.pth"
#num_features = scaler.transform(df[NUMERIC_COLUMNS])

class SkinCancerModel(nn.Module):
    def __init__(self, num_features):
        super(SkinCancerModel, self).__init__()
        # Load the pretrained model without weights
        self.backbone = models.efficientnet_v2_s(weights=None)

        # Load the state_dict from your dataset
        state_dict = torch.load(weights_path, map_location=CFG.device)
        self.backbone.load_state_dict(state_dict)  # Apply weights to backbone

        # Extract the in_features from the last linear layer of the original classifier
        original_classifier = self.backbone.classifier
        in_features = original_classifier[-1].in_features

        # Replace the classifier with Identity
        self.backbone.classifier = nn.Identity()

        # Tabular feature extractor
        self.tabular_net = nn.Sequential(
            nn.Linear(num_features, 96),
            nn.SELU(),
            nn.Linear(96, 128),
            nn.SELU(),
            nn.Dropout(0.1),
        )

        # Final classifier combining image + tabular features
        self.classifier = nn.Sequential(
            nn.Linear(in_features + 128, 1),
            nn.Sigmoid()
        )

    def forward(self, images, features):
        img_features = self.backbone(images)
        tab_features = self.tabular_net(features)
        combined = torch.cat([img_features, tab_features], dim=1)
        out = self.classifier(combined)
        return out





# Get the number of features after preprocessing
num_features = train_dataset.features.shape[1]

model = SkinCancerModel(num_features)
model = model.to(CFG.device)

# Print model summary
print(model)
#<a id='loss-optimizer'></a>




import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

def add_box(ax, xy, width, height, color, label, fontsize=10):
    ax.add_patch(Rectangle(xy, width, height, edgecolor="black", facecolor=color, lw=2))
    ax.text(xy[0] + width / 2, xy[1] + height / 2, label, color="black",
            fontsize=fontsize, ha="center", va="center")

fig, ax = plt.subplots(figsize=(12, 8))

# Input images and features
add_box(ax, (0.5, 6), 2, 1, "peachpuff", "Input Images", fontsize=10)
add_box(ax, (0.5, 3), 2, 1, "lightblue", "Tabular Features", fontsize=10)

# Pretrained EfficientNet backbone
add_box(ax, (3.5, 6), 3, 1, "yellow", "EfficientNet\nBackbone", fontsize=10)

# Tabular feature extractor
add_box(ax, (3.5, 3), 3, 1, "lightgreen", "Tabular\nFeature Extractor", fontsize=10)

# Fusion block
add_box(ax, (7.5, 4.5), 3, 1, "orange", "Feature Fusion", fontsize=10)

# Classifier
add_box(ax, (11, 4.5), 2, 1, "salmon", "Final Classifier", fontsize=10)

# Arrows
arrow_args = dict(width=0.02, head_width=0.2, head_length=0.3, length_includes_head=True, color="black")
ax.add_patch(FancyArrow(2.5, 6.5, 1, 0, **arrow_args))  # Input Images to EfficientNet
ax.add_patch(FancyArrow(2.5, 3.5, 1, 0, **arrow_args))  # Tabular Features to Tabular Extractor
ax.add_patch(FancyArrow(6.5, 6, 1, -1, **arrow_args))   # EfficientNet to Fusion
ax.add_patch(FancyArrow(6.5, 3.5, 1, 1, **arrow_args))  # Tabular Extractor to Fusion
ax.add_patch(FancyArrow(10.5, 5, 1, 0, **arrow_args))   # Fusion to Classifier

# Diagram adjustments
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis("off")
plt.show()



from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch

classes = np.unique(df['target'])
class_weights = compute_class_weight('balanced', classes=np.unique(df['target']), y=df['target'])
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(CFG.device)



criterion = nn.BCELoss(weight=class_weights[1])
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
# After (ReduceLROnPlateau scheduler, step on val_auc):

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
...






def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    preds = []
    targets = []
    
    for images, features, labels in tqdm(dataloader, desc='Training', leave=False):
        images = images.to(CFG.device)
        features = features.to(CFG.device)
        labels = labels.to(CFG.device).view(-1, 1)  # Reshape labels to (batch_size, 1)
        
        optimizer.zero_grad()
        outputs = model(images, features)
        
        # Compute batch-specific weights
        batch_weights = torch.where(labels == 1, class_weights[1], class_weights[0]).to(CFG.device)
        loss = F.binary_cross_entropy(outputs, labels, weight=batch_weights)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        preds.extend(outputs.detach().cpu().numpy())
        targets.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_auc = roc_auc_score(targets, preds)
    return epoch_loss, epoch_auc



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def evaluate_model(targets, predictions, threshold=0.85):
    # Convert probabilities to binary predictions
    binary_preds = (predictions >= threshold).astype(int)
    
    # Calculate metrics
    acc = accuracy_score(targets, binary_preds)
    precision = precision_score(targets, binary_preds, zero_division=1)
    recall = recall_score(targets, binary_preds, zero_division=1)
    f1 = f1_score(targets, binary_preds, zero_division=1)
    auc = roc_auc_score(targets, predictions)
    conf_matrix = confusion_matrix(targets, binary_preds)
    
    # Return results
    return {
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc,
        'Confusion Matrix': conf_matrix
    }



def validate_one_epoch(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    preds = []
    targets = []
    
    with torch.no_grad():
        for images, features, labels in tqdm(dataloader, desc='Validation', leave=False):
            images = images.to(CFG.device)
            features = features.to(CFG.device)
            labels = labels.to(CFG.device).view(-1, 1)  # Reshape labels to (batch_size, 1)
            
            outputs = model(images, features)
            
            # Compute batch-specific weights
            batch_weights = torch.where(labels == 1, class_weights[1], class_weights[0]).to(CFG.device)
            loss = F.binary_cross_entropy(outputs, labels, weight=batch_weights)
            
            running_loss += loss.item() * images.size(0)
            preds.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    # Calculate metrics
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_auc = roc_auc_score(targets, preds)
    
    # Evaluate detailed metrics
    metrics = evaluate_model(np.array(targets), np.array(preds))
    
    return epoch_loss, epoch_auc, metrics

# Validation phase
val_loss, val_auc, val_metrics = validate_one_epoch(model, val_loader, criterion)




results = []  # To store metrics for each epoch
best_auc = 0.0  # Track the best AUC
threshold = 0.85  # Define the AUC threshold to trigger weight adjustment

for epoch in range(1, CFG.epochs + 1):
    print(f"\nEpoch {epoch}/{CFG.epochs}")
    
    # Training phase
    train_loss, train_auc = train_one_epoch(model, train_loader, criterion, optimizer)
    
    # Validation phase
    val_loss, val_auc, val_metrics = validate_one_epoch(model, val_loader, criterion)
    
    print(f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
    
    # Adjust weights dynamically based on validation AUC
    if val_auc < threshold:
        class_weights[1] += 0.1  # Increase weight for the positive class
        print(f"Adjusted class weights: {class_weights}")
        # Update the loss criterion with new class weights
        criterion = nn.BCELoss(weight=class_weights)

    print("Validation Metrics:")
    for metric, value in val_metrics.items():
        if metric != "Confusion Matrix":
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}:\n{value}")
    
    # Store metrics
    results.append({
        'Epoch': epoch,
        'Train Loss': train_loss,
        'Train AUC': train_auc,
        'Val Loss': val_loss,
        'Val AUC': val_auc,
        'Accuracy': val_metrics['Accuracy'],
        'Precision': val_metrics['Precision'],
        'Recall': val_metrics['Recall'],
        'F1-Score': val_metrics['F1-Score'],
        'AUC-ROC': val_metrics['AUC-ROC']
    })
    
    # Scheduler step (if applicable)
    scheduler.step(val_auc)
    
    # Save the best model
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), 'best_model.pth')
        print("Best model saved!")




# Load the best model weights
model.load_state_dict(torch.load('best_model.pth', weights_only=True))




try:
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")




train_losses = [] 
val_losses = []
train_aucs = []
val_aucs = []
best_auc = 0.0  # Initialize best AUC

for epoch in range(1, CFG.epochs + 1):
    # Training phase
    train_loss, train_auc = train_one_epoch(model, train_loader, criterion, optimizer)
    
    # Validation phase
    val_loss, val_auc, val_metrics = validate_one_epoch(model, val_loader, criterion)
    
    # Append losses and AUCs
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_aucs.append(train_auc)
    val_aucs.append(val_auc)
    
    # Update scheduler if being used
    scheduler.step(val_auc)
    
    # Save the best model based on validation AUC
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Epoch {epoch}: Best model saved with AUC: {best_auc:.4f}")

    # Print validation metrics
    print(f"\nEpoch {epoch} Metrics:")
    for metric, value in val_metrics.items():
        if metric != "Confusion Matrix":
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}:\n{value}")



import matplotlib.pyplot as plt

# Plot Losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot AUCs
plt.figure(figsize=(10, 5))
plt.plot(train_aucs, label='Train AUC')
plt.plot(val_aucs, label='Validation AUC')
plt.title('Training and Validation AUC over Epochs')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)
plt.show()




import pandas as pd

# Assuming `results` is a dictionary or list of dictionaries with your data
results_df = pd.DataFrame(results)

# Display the DataFrame as a table
print(results_df)

# Save the DataFrame as a CSV file (optional)
results_df.to_csv('performance_metrics.csv', index=False)

print("CSV file 'performance_metrics.csv' saved successfully.")










def inference(model, dataloader):
    model.eval()
    preds = []
    
    with torch.no_grad():
        for images, features in tqdm(dataloader, desc='Inference', leave=False):
            images = images.to(CFG.device)
            features = features.to(CFG.device)
            
            outputs = model(images, features)
            preds.extend(outputs.cpu().numpy())
    
    preds = np.array(preds).squeeze()
    return preds





test_preds = inference(model, test_loader)


# Get some test images
images, features = next(iter(test_loader))

# Get predictions
with torch.no_grad():
    images = images.to(CFG.device)
    features = features.to(CFG.device)
    outputs = model(images, features)
    preds = outputs.cpu().numpy().squeeze()

# Plot the images and predictions
plt.figure(figsize=(10, 4))
for i in range(min(6, images.size(0))):
    img = images[i].cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    plt.subplot(1, 3, i+1)
    plt.imshow(img)
    plt.title(f'Prediction: {preds[i]:.2f}')
    plt.axis('off')
plt.suptitle('Model Predictions on Testing Images', fontsize=16)
plt.tight_layout()
plt.show()
#<a id='submission'></a>




# Get some test images and corresponding features
images, features = next(iter(test_loader))

# Get predictions from the model
with torch.no_grad():
    images = images.to(CFG.device)
    features = features.to(CFG.device)
    outputs = model(images, features)
    preds = outputs.cpu().numpy().squeeze()

# Plot the images and predictions (6 images in one row)
num_images = min(6, images.size(0))
plt.figure(figsize=(18, 4))  # Adjusted width to fit 6 images
for i in range(num_images):
    img = images[i].cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    plt.subplot(1, num_images, i+1)  # Create a grid with num_images columns
    plt.imshow(img)
    plt.title(f'Prediction: {preds[i]:.2f}')
    plt.axis('off')
plt.suptitle('Model Predictions on Testing Images', fontsize=16)
plt.tight_layout()
plt.show()



# Prepare submission
pred_df = testing_df[['isic_id']].copy()
pred_df['target'] = test_preds.tolist()

sub_df = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')
sub_df = sub_df[['isic_id']].copy()
sub_df = sub_df.merge(pred_df, on='isic_id', how='left')
sub_df.to_csv('submission.csv', index=False)

print("Submission file:")
display(sub_df.head())



