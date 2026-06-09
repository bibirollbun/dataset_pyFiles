# Clone Dinov2 repository
!git clone https://github.com/facebookresearch/dinov2.git

# Install dependencies
# Uninstall existing versions to prevent conflicts
# !pip uninstall torch torchvision torchaudio -y

# # Install specific compatible versions of PyTorch, torchvision, and torchaudio for CUDA 11.8
# # Based on the error: torchvision 0.22.1+cu118 requires torch==2.7.1
# # We assume torchaudio version matches torch version.
# !pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# Install other packages
!pip install timm scikit-learn matplotlib opendatasets pandas

import os
import sys
sys.path.append('dinov2') # Add dinov2 to system path


import opendatasets as od
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
import os # Ensure os is imported if not already

# The dataset is expected to be available in the Kaggle input directory
# # Download PCam dataset (commented out as data is expected in Kaggle environment)
# dataset_url = 'https://www.kaggle.com/competitions/histopathologic-cancer-detection/data'
# od.download(dataset_url, data_dir='./pcam_dataset') # This line would create ./pcam_dataset

# Define paths assuming data is in /kaggle/input/
# If running locally after download, adjust this base_dir accordingly.
# For Kaggle environment, it's typically /kaggle/input/<dataset-folder-name>
base_dir = '/kaggle/input/histopathologic-cancer-detection' 
# If you downloaded it locally using the above od.download, base_dir would be './pcam_dataset/histopathologic-cancer-detection'

train_labels_path = os.path.join(base_dir, 'train_labels.csv')
train_images_dir = os.path.join(base_dir, 'train')
test_images_dir = os.path.join(base_dir, 'test') # This is the unlabeled test set from Kaggle

# Load labels
try:
    train_labels_df = pd.read_csv(train_labels_path)
except FileNotFoundError:
    print(f"Error: train_labels.csv not found at {train_labels_path}")
    print("Please ensure the 'base_dir' variable is set correctly to your dataset location.")
    print("If running on Kaggle, the path should be /kaggle/input/histopathologic-cancer-detection/train_labels.csv")
    # Example for local download:
    # print("If you downloaded data locally to './pcam_dataset', set: base_dir = './pcam_dataset/histopathologic-cancer-detection'")
    raise

# Split training data into train and validation sets (using the labels dataframe)
train_df, val_df = train_test_split(train_labels_df, test_size=0.2, stratify=train_labels_df['label'], random_state=42)

print(f"Total training samples: {len(train_labels_df)}")
print(f"Number of training samples after split: {len(train_df)}")
print(f"Number of validation samples: {len(val_df)}")
print(f"Training images directory: {train_images_dir}")
print(f"Test images directory (unlabeled): {test_images_dir}")

# The pcam_organized directory and file copying are no longer needed.
# Images will be loaded directly from train_images_dir using a custom Dataset.

print("Dataset preparation complete (labels loaded and split). File copying skipped.")


import matplotlib.pyplot as plt
import random
from PIL import Image
import numpy as np
import os
import pandas as pd

# train_images_dir, train_df, val_df, and train_labels_df should be defined from running cell 5.
# The line that previously redefined train_images_dir here has been removed.

# Function to display sample images
def display_sample_images(df, image_dir, num_samples=2):
    plt.figure(figsize=(10, 5 * num_samples // 2))
    # Get samples from both classes if possible
    samples_class_0 = df[df['label'] == 0].sample(min(num_samples // 2, len(df[df['label'] == 0])))
    samples_class_1 = df[df['label'] == 1].sample(min(num_samples // 2, len(df[df['label'] == 1])))
    
    samples_to_display = pd.concat([samples_class_0, samples_class_1])
    if samples_to_display.empty:
        print("No samples to display. Check your DataFrame and image directory.")
        return

    for i, (idx, row) in enumerate(samples_to_display.iterrows()):
        img_id = row['id']
        label = row['label']
        img_path = os.path.join(image_dir, f'{img_id}.tif')
        
        try:
            img = Image.open(img_path)
            plt.subplot(num_samples // 2, 2, i + 1)
            plt.imshow(img)
            plt.title(f"ID: {img_id}\nClass: {label} ({'No Cancer' if label == 0 else 'Cancer'})")
            plt.axis('off')
        except FileNotFoundError:
            print(f"Image not found: {img_path}")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    plt.tight_layout()
    plt.show()

print("Displaying sample images from training set:")
# Display 4 samples (2 from each class if available)
# This will now use train_images_dir from cell 5
display_sample_images(train_df, train_images_dir, num_samples=4) 

# Check class distribution using the DataFrame
print("\nClass distribution in the original training labels:")
print(train_labels_df['label'].value_counts())

print("\nClass distribution in the split training set (train_df):")
print(train_df['label'].value_counts())

print("\nClass distribution in the split validation set (val_df):")
print(val_df['label'].value_counts())

# Basic image statistics (dimensions of a few samples)
print("\nBasic image statistics (dimensions from a few samples in train_df):")
if not train_df.empty:
    for i in range(min(4, len(train_df))): # Check up to 4 images
        sample_row = train_df.sample(1).iloc[0]
        img_id = sample_row['id']
        label = sample_row['label']
        # This will now use train_images_dir from cell 5
        img_path = os.path.join(train_images_dir, f'{img_id}.tif')
        try:
            img = Image.open(img_path)
            print(f"Image ID: {img_id}, Class: {label}, Dimensions: {img.size}, Mode: {img.mode}")
        except FileNotFoundError:
            print(f"Image not found: {img_path}")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
else:
    print("train_df is empty, cannot display image statistics.")


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_class_distributions(train_labels_df, train_df, val_df):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    sns.countplot(data=train_labels_df, x='label', ax=axs[0], palette='pastel')
    axs[0].set_title('Original Label Distribution')
    axs[0].set_xticklabels(['No Cancer (0)', 'Cancer (1)'])

    sns.countplot(data=train_df, x='label', ax=axs[1], palette='pastel')
    axs[1].set_title('Train Set Label Distribution')
    axs[1].set_xticklabels(['No Cancer (0)', 'Cancer (1)'])

    sns.countplot(data=val_df, x='label', ax=axs[2], palette='pastel')
    axs[2].set_title('Validation Set Label Distribution')
    axs[2].set_xticklabels(['No Cancer (0)', 'Cancer (1)'])

    for ax in axs:
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', (p.get_x() + 0.3, p.get_height() + 500))

    plt.tight_layout()
    plt.show()

plot_class_distributions(train_labels_df, train_df, val_df)



from hashlib import md5

def get_image_hashes(df, image_dir, num_images=1000):
    hashes = {}
    duplicates = []
    for _, row in df.sample(min(num_images, len(df))).iterrows():
        img_path = os.path.join(image_dir, f"{row['id']}.tif")
        try:
            with open(img_path, 'rb') as f:
                img_hash = md5(f.read()).hexdigest()
            if img_hash in hashes:
                duplicates.append((img_path, hashes[img_hash]))
            else:
                hashes[img_hash] = img_path
        except:
            continue
    return duplicates

dups = get_image_hashes(train_df, train_images_dir)
print(f"Found {len(dups)} duplicate images.")



def image_quality_check(df, image_dir, split_name, num_samples=1000):
    print(f"\nChecking image quality for {split_name} set (sampling {num_samples} images)")
    sizes = []
    errors = 0
    sample_df = df.sample(min(num_samples, len(df)), random_state=42)
    
    for _, row in sample_df.iterrows():
        img_path = os.path.join(image_dir, f"{row['id']}.tif")
        try:
            img = Image.open(img_path)
            sizes.append(img.size)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            errors += 1
    
    if not sizes:
        print(f"No valid images found for {split_name}")
        return
    
    sizes = np.array(sizes)
    unique_sizes = np.unique(sizes, axis=0)
    print(f"Number of images checked: {len(sizes)}")
    print(f"Number of errors: {errors}")
    print(f"Unique image sizes: {unique_sizes}")
    if len(unique_sizes) == 1:
        print(f"All images have consistent size: {unique_sizes[0]}")
    else:
        print("Warning: Inconsistent image sizes detected")

try:
    image_quality_check(train_df, train_images_dir, 'train')
    image_quality_check(val_df, train_images_dir, 'valid')
except NameError as e:
    print(f"Error: Required variables not defined - {e}")


import torch
from torchvision import transforms # Removed datasets as ImageFolder is not used
from torch.utils.data import DataLoader, Dataset # Added Dataset
from PIL import Image # For loading images in the custom dataset
import os # Ensure os is imported

# train_images_dir, train_df, and val_df are defined in cell 5
# IMG_SIZE is defined here, or could be moved to a config section
IMG_SIZE = 224 

# Define a custom Dataset
class PCamDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with 'id' and 'label' columns.
            image_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = self.dataframe.iloc[idx, self.dataframe.columns.get_loc('id')]
        label = self.dataframe.iloc[idx, self.dataframe.columns.get_loc('label')]
        
        img_name = os.path.join(self.image_dir, f"{img_id}.tif")
        
        try:
            image = Image.open(img_name).convert("RGB") # Ensure image is RGB
        except FileNotFoundError:
            print(f"ERROR: Image not found at {img_name} for id {img_id}. Check image_dir and dataframe.")
            # Return a placeholder or raise an error. For now, let's raise it to stop execution.
            raise FileNotFoundError(f"Image not found: {img_name}")
        except Exception as e:
            print(f"ERROR: Could not load image {img_name}: {e}")
            raise

        if self.transform:
            image = self.transform(image)
        
        return image, int(label)

# Dinov2 preprocessing / Standard ImageNet normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Transform for the unlabeled test set (if used for prediction later)
    'test_data': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Create custom datasets
# train_images_dir should be /kaggle/input/histopathologic-cancer-detection/train
# train_df and val_df are the dataframes from the split in cell 5
image_datasets = {
    'train': PCamDataset(dataframe=train_df, image_dir=train_images_dir, transform=data_transforms['train']),
    'val': PCamDataset(dataframe=val_df, image_dir=train_images_dir, transform=data_transforms['val'])
}

# Create DataLoaders
BATCH_SIZE = 32 # Adjust based on your GPU memory. 
                # If using multiple GPUs (e.g., with DataParallel), you might be able to increase this.
dataloaders = {x: DataLoader(image_datasets[x],
                                batch_size=BATCH_SIZE,
                                shuffle=True if x == 'train' else False,
                                num_workers=2) # Adjust num_workers based on your system
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# Class names are implicitly [0, 1] based on the labels in the CSV.
# If you need explicit class names for plotting or reports later, define them:
class_names = ['0', '1'] # Or ['No Cancer', 'Cancer'] if preferred for display

print(f"Class names used: {class_names}")
print(f"Dataset sizes: Train: {dataset_sizes['train']}, Val: {dataset_sizes['val']}")
print(f"Number of training batches: {len(dataloaders['train'])}")
print(f"Number of validation batches: {len(dataloaders['val'])}")

# Handling the unlabeled Kaggle test set (test_images_dir)
# If you need to make predictions on it, you'll need a DataFrame for its IDs
# or a custom Dataset that lists files in test_images_dir.

# Example for creating a DataFrame for the test set (if you have image IDs, e.g., from sample_submission.csv)
# submission_df = pd.read_csv(os.path.join(base_dir, 'sample_submission.csv'))
# test_df_for_dataset = submission_df[['id']].copy() # Assuming 'id' column exists
# # Add a dummy label column if your Dataset class expects it, it won't be used for actual labels
# test_df_for_dataset['label'] = -1 

# if not test_df_for_dataset.empty:
#     test_custom_dataset = PCamDataset(dataframe=test_df_for_dataset, 
#                                         image_dir=test_images_dir, 
#                                         transform=data_transforms['test_data'])
#     test_dataloader = DataLoader(test_custom_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
#     print(f"Test dataset size (unlabeled): {len(test_custom_dataset)}")
#     print(f"Number of test batches (unlabeled): {len(test_dataloader)}")
# else:
#     test_dataloader = None
#     print("No DataFrame for unlabeled test data provided, test_dataloader not created.")

# For now, we will set test_dataloader to None as it's not strictly needed for training/validation loop
test_dataloader = None
print("test_dataloader is set to None. Modify above if predictions on Kaggle test set are needed.")

# Verify a sample from the dataloader
if dataloaders['train']:
    try:
        sample_inputs, sample_labels = next(iter(dataloaders['train']))
        print(f"Sample batch - Inputs shape: {sample_inputs.shape}, Labels shape: {sample_labels.shape}")
    except Exception as e:
        print(f"Error when trying to get a sample batch from train dataloader: {e}")
        print("This might indicate an issue with PCamDataset or the underlying data.")


import torch.nn as nn

# Load a pre-trained Dinov2 model
# Available models: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
# We'll use dinov2_vits14 as an example. It's smaller and faster to train.
dinov2_model_name = 'dinov2_vits14' # Small model
# dinov2_model_name = 'dinov2_vitb14' # Base model

# Using torch.hub to load the model
# Ensure you have the dinov2 repository cloned and in your sys.path as done in step 1
try:
    model = torch.hub.load('facebookresearch/dinov2', dinov2_model_name)
except Exception as e:
    print(f"Error loading model from torch.hub: {e}")
    print("Make sure the dinov2 repository is correctly cloned and accessible.")
    # Fallback or alternative loading if needed, e.g. using timm if the model is available there
    # import timm
    # model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True)
    raise e

# Dinov2 models from torch.hub usually don't have a classification head suitable for direct fine-tuning
# or the head is for ImageNet (1000 classes). We need to replace it or use the model as a feature extractor.

# Option 1: Use as a feature extractor and add a new classifier head
# The feature dimension for dinov2_vits14 is 384
# For dinov2_vitb14, it's 768
feature_dim = model.embed_dim # This should give the feature dimension

# Freeze the backbone (Dinov2) parameters
for param in model.parameters():
    param.requires_grad = False

# Define a new classification head
num_classes = len(class_names) # Should be 2 for PCam (Cancer/No Cancer)
classifier_head = nn.Linear(feature_dim, num_classes)

# Combine the Dinov2 backbone with the new head
# Some Dinov2 hub models might return features directly, others might have a `head` attribute.
# If it has a `head` attribute, we replace it.
if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
    model.head = classifier_head
    print(f"Replaced model.head with a new Linear layer for {num_classes} classes.")
else:
    # If no head attribute or it's not what we expect, create a sequential model
    # This assumes the base Dinov2 model outputs features directly from its forward pass
    # or we need to call a specific method like `model.forward_features()`
    # For torch.hub.load('facebookresearch/dinov2', ...), the model itself is the backbone.
    # We need to wrap it if we want to append a head and treat it as a single nn.Module for training.
    class Dinov2WithHead(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head
        
        def forward(self, x):
            # The backbone might return a dict or a tensor
            # For Dinov2, it often returns a dict like {'x_norm_patchtokens': ..., 'x_norm_clstoken': ...}
            # We are interested in the CLS token for classification
            features = self.backbone.forward_features(x)
            # Try to get CLS token, otherwise use patch tokens (might need pooling)
            if isinstance(features, dict) and 'x_norm_clstoken' in features:
                cls_token = features['x_norm_clstoken']
            elif isinstance(features, dict) and 'x_norm_patchtokens' in features: # Fallback: average patch tokens
                patch_tokens = features['x_norm_patchtokens']
                cls_token = torch.mean(patch_tokens, dim=1) # Average pool patch tokens
            elif torch.is_tensor(features):
                # If it's a tensor, assume it's [batch_size, num_tokens, embed_dim]
                # and the first token is the CLS token, or we average pool
                # This depends on the specific Dinov2 variant from torch.hub
                # For safety, let's assume we need to average if it's not a dict with clstoken
                if features.ndim == 3:
                    cls_token = torch.mean(features, dim=1)
                else:
                    cls_token = features # Assume it's already [batch_size, embed_dim]
            else:
                raise ValueError(f"Unexpected feature output format from Dinov2 backbone: {type(features)}")
            
            return self.head(cls_token)

    model = Dinov2WithHead(model, classifier_head)
    print(f"Wrapped Dinov2 backbone with a new Linear layer for {num_classes} classes.")

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Check for multiple GPUs and use DataParallel
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
    print("Model wrapped with nn.DataParallel.")

print(f"Model: {dinov2_model_name} loaded and adapted for {num_classes}-class classification.")
print(f"Using device: {device}")
# print(model) # Uncomment to see model structure


import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import torch.nn as nn # Ensure nn is imported for DataParallel check

# Define loss function and optimizer

# Access the original model if wrapped by DataParallel
original_model = model.module if isinstance(model, nn.DataParallel) else model

# Only training the classifier head, so pass only its parameters to the optimizer
if isinstance(original_model, Dinov2WithHead):
    optimizer = optim.AdamW(original_model.head.parameters(), lr=1e-3, weight_decay=1e-4)
    print("Optimizing only the parameters of the custom classifier head (original_model.head).")
elif hasattr(original_model, 'head') and isinstance(original_model.head, nn.Linear):
    optimizer = optim.AdamW(original_model.head.parameters(), lr=1e-3, weight_decay=1e-4)
    print("Optimizing only the parameters of original_model.head.")
else:
    # This case should ideally not be hit if model adaptation was correct
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, original_model.parameters()), lr=1e-3, weight_decay=1e-4)
    print("Warning: Optimizing all parameters of original_model with requires_grad=True. Ensure this is intended.")

criterion = nn.CrossEntropyLoss()

# Learning rate scheduler (optional, but often helpful)
# Reduce learning rate when a metric has stopped improving
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    overall_start_time = time.time() # Record start time for overall training
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        epoch_start_time = time.time() # Record start time for the current epoch
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step() # Step the learning rate scheduler

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item()) # .item() to get Python number

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it's the best validation accuracy so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save the best model checkpoint
                torch.save(model.state_dict(), 'dinov2_pcam_best_model.pth')
                print(f"Best validation accuracy improved to {best_acc:.4f}. Model saved.")
        
        epoch_time_elapsed = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_time_elapsed // 60:.0f}m {epoch_time_elapsed % 60:.0f}s")
        print()

    overall_time_elapsed = time.time() - overall_start_time
    print(f'Overall training complete in {overall_time_elapsed // 60:.0f}m {overall_time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

# Start training
# Consider a small number of epochs first to ensure everything runs correctly.
NUM_EPOCHS = 10 # Adjust as needed. For good results, more epochs might be required.

print("Starting model training...")
model_ft, training_history = train_model(model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(training_history['train_loss'], label='Train Loss')
plt.plot(training_history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Over Epochs')

plt.subplot(1, 2, 2)
plt.plot(training_history['train_acc'], label='Train Accuracy')
plt.plot(training_history['val_acc'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')
plt.tight_layout()
plt.show()


from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Function to evaluate the model
def evaluate_model(model, dataloader):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad(): # No need to track gradients during evaluation
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)

print("Evaluating the model on the validation set...")
val_labels, val_preds = evaluate_model(model_ft, dataloaders['val'])

# Classification Report
print("\nClassification Report (Validation Set):")
# Ensure class_names are strings if they are not already for classification_report
str_class_names = [str(cn) for cn in class_names]
print(classification_report(val_labels, val_preds, target_names=str_class_names))

# Confusion Matrix
print("\nConfusion Matrix (Validation Set):")
cm = confusion_matrix(val_labels, val_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=str_class_names, yticklabels=str_class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Visualize some predictions (optional)
def visualize_predictions(model, dataloader, num_images=10):
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(15, 10))
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 5 + (1 if num_images % 5 > 0 else 0), 5, images_so_far)
                ax.axis('off')
                ax.set_title(f'Pred: {class_names[preds[j]]} (True: {class_names[labels[j]]})')
                # Inverse normalize and display image
                img = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                plt.imshow(img)
                
                if images_so_far == num_images:
                    plt.tight_layout()
                    plt.show()
                    return
    plt.tight_layout()
    plt.show()

print("\nVisualizing some predictions from the validation set:")
visualize_predictions(model_ft, dataloaders['val'], num_images=10)


# The best model was already saved during training as 'dinov2_pcam_best_model.pth'
# Here we can explicitly save the final model if needed, or re-save the best one for clarity.
final_model_path = "dinov2_pcam_final_trained_model.pth"
torch.save(model_ft.state_dict(), final_model_path)
print(f"Final trained model state dictionary saved to: {final_model_path}")

# To load the model later:
# 1. Re-define the model architecture (as done in Section 5)
#    model_to_load = ... (same architecture as model_ft)
# 2. Load the state dictionary:
#    model_to_load.load_state_dict(torch.load(final_model_path, map_location=device))
#    model_to_load.eval() # Set to evaluation mode

