import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns 
from tqdm.notebook import tqdm
import time
import hashlib
from collections import defaultdict
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torchvision.transforms.functional as TF

from sklearn.metrics import confusion_matrix, classification_report,cohen_kappa_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

# ====================================================================================================
# SECTION 1: SETUP AND CONFIGURATION
# ====================================================================================================

# Set seeds for reproducibility to ensure consistent results across runs
def seed_everything(seed=42):
    """Set seeds for all random number generators to ensure reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # Ensures that CUDA selects deterministic algorithms
    torch.backends.cudnn.benchmark = False     # Disables CUDA benchmark to ensure reproducibility

# Initialize all random seeds
seed_everything()

# Availability of CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define paths to the datasets
aptos_path = "/kaggle/input/aptos2019-blindness-detection"
eyepacs_path = "/kaggle/input/eyepacs-aptos-messidor-diabetic-retinopathy"
print(f"APTOS 2019 path: {aptos_path}")
print(f"EyePACS-APTOS-Messidor path: {eyepacs_path}")


# ====================================================================================================
# SECTION 2: DATA LOADING AND PREPROCESSING
# ====================================================================================================

# Function to get file list from EyePACS dataset folder structure
def get_eyepacs_files(root_path, subset='train'):
    """
    Get files from EyePACS folder structure where images are organized in class folders.
    
    Args:
        root_path (str): Path to the EyePACS dataset
        subset (str): Dataset subset to use ('train' or 'test')
        
    Returns:
        dict: Dictionary with class numbers as keys and lists of file info as values
    """
    files_dict = defaultdict(list)
    subset_path = os.path.join(root_path, 'augmented_resized_V2', subset)
    
    # Check if path exists
    if not os.path.exists(subset_path):
        print(f"Error: Path {subset_path} does not exist")
        return files_dict
    
    # Iterate through class folders (0, 1, 2, 3, 4)
    for class_folder in os.listdir(subset_path):
        if class_folder.isdigit():  # Only process folders that are class numbers
            class_num = int(class_folder)
            class_path = os.path.join(subset_path, class_folder)
            
            if os.path.isdir(class_path):
                # Get all files in this class folder
                for file_name in os.listdir(class_path):
                    if file_name.endswith(('.jpg', '.jpeg', '.png')):
                        file_path = os.path.join(class_path, file_name)
                        files_dict[class_num].append({
                            'image_id': file_name,
                            'class': class_num,
                            'image_path': file_path,
                            'source': 'eyepacs'
                        })
    
    # Print summary information
    total_files = sum(len(files) for files in files_dict.values())
    print(f"Found {total_files} images in EyePACS {subset} folder")
    
    # Print class distribution
    for cls in sorted(files_dict.keys()):
        print(f"  Class {cls}: {len(files_dict[cls])} images")
    
    return files_dict

# Function to load APTOS dataset from CSV file
def load_aptos_dataset(aptos_path):
    """
    Load APTOS dataset from CSV file and preprocess for consistency.
    
    Args:
        aptos_path (str): Path to the APTOS dataset
        
    Returns:
        DataFrame: DataFrame containing processed APTOS dataset
    """
    csv_path = os.path.join(aptos_path, 'train.csv')
    if not os.path.exists(csv_path):
        print(f"Error: APTOS CSV file not found at {csv_path}")
        return pd.DataFrame()
    
    # Load CSV data
    df_aptos = pd.read_csv(csv_path)
    print(f"Loaded APTOS 2019: {len(df_aptos)} images")
    
    # Add source and image path columns for consistency
    df_aptos['source'] = 'aptos'
    df_aptos['image_path'] = df_aptos['id_code'].apply(
        lambda x: os.path.join(aptos_path, 'train_images', f"{x}.png")
    )
    
    # Rename columns for consistency with EyePACS data format
    df_aptos = df_aptos.rename(columns={'id_code': 'image_id', 'diagnosis': 'class'})
    
    # Print class distribution
    print("APTOS class distribution:")
    print(df_aptos['class'].value_counts().sort_index())
    
    return df_aptos

# Load APTOS dataset
print("Executing load_aptos_dataset function...")
aptos_df = load_aptos_dataset(aptos_path)
print("APTOS dataset loaded successfully.")

# Function to create a balanced dataset by combining APTOS and EyePACS
def create_balanced_dataset(aptos_df, eyepacs_files, target_count_per_class=3000):
    """
    Create balanced dataset by combining APTOS and EyePACS images with equal samples per class.
    
    Args:
        aptos_df (DataFrame): DataFrame containing APTOS dataset
        eyepacs_files (dict): Dictionary with EyePACS files by class
        target_count_per_class (int): Target number of images per class
        
    Returns:
        DataFrame: Balanced dataset with combined images
    """
    balanced_rows = []
    
    # For each class (0-4), take samples up to target count
    for cls in range(5):
        # Get APTOS images for this class
        aptos_cls = aptos_df[aptos_df['class'] == cls]
        aptos_count = len(aptos_cls)
        
        # Get all EyePACS images for this class
        eyepacs_cls = eyepacs_files.get(cls, [])
        eyepacs_count = len(eyepacs_cls)
        
        print(f"Class {cls}: {aptos_count} from APTOS, {eyepacs_count} from EyePACS")
        
        # Determine how many images we need from each source
        if aptos_count >= target_count_per_class:
            # If APTOS has enough, just use those
            balanced_rows.extend(aptos_cls.sample(target_count_per_class, random_state=42).to_dict('records'))
            print(f"  Using {target_count_per_class} images from APTOS for class {cls}")
        else:
            # Use all APTOS images
            balanced_rows.extend(aptos_cls.to_dict('records'))
            print(f"  Using all {aptos_count} images from APTOS for class {cls}")
            
            # Calculate how many more we need
            needed_from_eyepacs = target_count_per_class - aptos_count
            
            if eyepacs_count >= needed_from_eyepacs:
                # Randomly sample from EyePACS
                selected_eyepacs = random.sample(eyepacs_cls, needed_from_eyepacs)
                balanced_rows.extend(selected_eyepacs)
                print(f"  Adding {needed_from_eyepacs} images from EyePACS for class {cls}")
            else:
                # Use all available from EyePACS
                balanced_rows.extend(eyepacs_cls)
                print(f"  Adding all {eyepacs_count} images from EyePACS for class {cls}")
                print(f"  Warning: Could only reach {aptos_count + eyepacs_count}/{target_count_per_class} for class {cls}")
    
    # Convert to DataFrame
    balanced_df = pd.DataFrame(balanced_rows)
    
    # Verify image paths exist (sample check)
    sample_size = min(10, len(balanced_df))
    for idx, row in balanced_df.sample(sample_size).iterrows():
        print(f"Checking {row['image_path']}: {os.path.exists(row['image_path'])}")
    
    # Print final distribution
    print("\nFinal class distribution:")
    print(balanced_df['class'].value_counts().sort_index())
    
    return balanced_df

# Load EyePACS data
print("\nExecuting get_eyepacs_files function...")
# Load from train folder
eyepacs_train_files = get_eyepacs_files(eyepacs_path, subset='train')
# Also load from test folder for more diversity
eyepacs_test_files = get_eyepacs_files(eyepacs_path, subset='test')

# Combine train and test files
eyepacs_files = defaultdict(list)
for cls in range(5):
    eyepacs_files[cls].extend(eyepacs_train_files.get(cls, []))
    eyepacs_files[cls].extend(eyepacs_test_files.get(cls, []))
print("EyePACS files loaded successfully.")

# Create balanced dataset
print("\nExecuting create_balanced_dataset function...")
balanced_df = create_balanced_dataset(aptos_df, eyepacs_files, target_count_per_class=3000)
print("Balanced dataset created successfully.")


# Function to split dataset into train, validation, and test sets
def create_data_splits(balanced_df, test_size=0.15, val_size=0.15):
    """
    Split dataset into train, validation, and test sets with stratification.
    
    Args:
        balanced_df (DataFrame): Balanced dataset to split
        test_size (float): Proportion of data for testing
        val_size (float): Proportion of data for validation
        
    Returns:
        tuple: (train_df, val_df, test_df) - DataFrames for each split
    """
    # First split out test set
    train_val_df, test_df = train_test_split(
        balanced_df, 
        test_size=test_size,
        random_state=42,
        stratify=balanced_df['class']  # Maintain class proportions
    )
    
    # Recalculate validation size relative to remaining data
    relative_val_size = val_size / (1 - test_size)
    
    # Split remaining data into train and validation
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=42,
        stratify=train_val_df['class']  # Maintain class proportions
    )
    
    print(f"Data split sizes: Train={len(train_df)}, Validation={len(val_df)}, Test={len(test_df)}")
    
    # Check class distribution in each split
    print("\nTrain class distribution:")
    print(train_df['class'].value_counts().sort_index())
    
    print("\nValidation class distribution:")
    print(val_df['class'].value_counts().sort_index())
    
    print("\nTest class distribution:")
    print(test_df['class'].value_counts().sort_index())
    
    return train_df, val_df, test_df

# Split data into train, validation, and test sets
print("\nExecuting create_data_splits function...")
train_df, val_df, test_df = create_data_splits(balanced_df)
print("Data splits created successfully.")


# ====================================================================================================
# SECTION 3: DATA AUGMENTATION AND DATASET PREPARATION (according to paper-baseline model)
# ====================================================================================================

# Define data transformations for training (with augmentation)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),              # Resize to standard input size
    transforms.RandomAffine(                    # Apply random affine transformations
        degrees=20,                             # Rotation range
        translate=(0.1, 0.1),                   # Translation range
        scale=(0.8, 1.2),                       # Scale range
    ),
    transforms.ColorJitter(brightness=(0.5, 1.0)),  # Randomly adjust brightness
    transforms.ToTensor(),                      # Convert to tensor
    transforms.Normalize(                       # Normalize with ImageNet mean and std
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# Define data transformations for validation and testing (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),              # Resize to standard input size
    transforms.ToTensor(),                      # Convert to tensor
    transforms.Normalize(                       # Normalize with ImageNet mean and std
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# Custom dataset class for Diabetic Retinopathy images
class DiabeticRetinopathyDataset(Dataset):
    """
    PyTorch Dataset class for Diabetic Retinopathy images.
    
    Attributes:
        dataframe (DataFrame): DataFrame containing image paths and labels
        transform (callable): Transformation to apply to images
    """
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset by index.
        
        Args:
            idx (int): Index of the item to get
            
        Returns:
            tuple: (image, label) where image is the transformed image and label is the class
        """
        img_path = self.dataframe.iloc[idx]['image_path']
        label = self.dataframe.iloc[idx]['class']
        
        try:
            # Load and convert image to RGB
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Create datasets and dataloaders
print("\nCreating datasets and dataloaders...")
# Create datasets
train_dataset = DiabeticRetinopathyDataset(train_df, transform=train_transform)
val_dataset = DiabeticRetinopathyDataset(val_df, transform=val_transform)
test_dataset = DiabeticRetinopathyDataset(test_df, transform=val_transform)

# Create data loaders with specified batch size and workers
train_loader = DataLoader(
    train_dataset,
    batch_size=64,             # Number of samples per batch
    shuffle=True,              # Shuffle data for training
    num_workers=4,             # Number of parallel workers for data loading
    pin_memory=True            # Pin memory for faster GPU transfer
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,             # No need to shuffle validation data
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,             # No need to shuffle test data
    num_workers=4,
    pin_memory=True
)

print(f"Created dataloaders with {len(train_dataset)} training, {len(val_dataset)} validation, and {len(test_dataset)} test images")


# ====================================================================================================
# SECTION 4: MODEL ARCHITECTURE
# ====================================================================================================

# Define custom MobileNetV2 model with additional fully connected layers
class CustomMobileNetV2(nn.Module):
    """
    Custom MobileNetV2 model with modified classifier for Diabetic Retinopathy classification.
    
    Attributes:
        mobilenet (nn.Module): MobileNetV2 backbone with pretrained weights
    """
    def __init__(self, num_classes=5):
        """
        Initialize the model with pretrained MobileNetV2 backbone and custom classifier.
        
        Args:
            num_classes (int): Number of output classes (5 for DR grades)
        """
        super(CustomMobileNetV2, self).__init__()
        # Load pretrained MobileNetV2 model
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        
        # Replace classifier with custom layers
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.2),                          # Dropout for regularization
            nn.Linear(self.mobilenet.last_channel, 32),  # First FC layer
            nn.ReLU(),                                # Activation function
            nn.Linear(32, 16),                        # Second FC layer
            nn.ReLU(),                                # Activation function
            nn.Linear(16, num_classes)                # Output layer
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.mobilenet(x)

# Initialize model
print("\nInitializing model...")
model = CustomMobileNetV2(num_classes=5).to(device)

# Define loss function - standard cross entropy as per paper
criterion = nn.CrossEntropyLoss()

# Define optimizer - Adam with learning rate 0.0001 as per paper
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Learning rate scheduler to reduce LR when validation loss plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',               # Monitor minimum validation loss
    factor=0.5,               # Multiply LR by this factor when reducing
    patience=5,               # Wait for 5 epochs before reducing LR
    verbose=True              # Print message when reducing LR
)
print("Model initialized successfully.")


# ====================================================================================================
# SECTION 5: TRAINING FUNCTION
# ====================================================================================================

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100, patience=7):
    """
    Train the model with early stopping and learning rate scheduling.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        optimizer (optim.Optimizer): Optimizer
        scheduler: Learning rate scheduler
        num_epochs (int): Maximum number of epochs to train
        patience (int): Early stopping patience (epochs without improvement)
        
    Returns:
        tuple: (trained_model, history) - Trained model and training history
    """
    best_val_loss = float('inf')
    counter = 0  # Counter for early stopping
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    start_time = time.time()
    
    best_model_path = 'baseline_model.pth'
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # -----------------
        # Training phase
        # -----------------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Track class-wise accuracy
        class_correct = [0] * 5
        class_total = [0] * 5
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels in train_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Class-wise accuracy
            for i in range(5):
                label_mask = (labels == i)
                class_total[i] += label_mask.sum().item()
                if label_mask.sum() > 0:
                    class_correct[i] += (predicted[label_mask] == i).sum().item()
            
            # Update progress bar
            train_bar.set_postfix(loss=loss.item(), acc=correct/total if total > 0 else 0)
        
        # Calculate epoch metrics
        epoch_train_loss = running_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0
        epoch_train_acc = correct / total if total > 0 else 0
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        
        # -----------------
        # Validation phase
        # -----------------
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Class-wise validation accuracy
        val_class_correct = [0] * 5
        val_class_total = [0] * 5
        
        # Confusion matrix for per-class metrics
        confusion_mat = torch.zeros(5, 5)
        
        with torch.no_grad():  # No gradients needed for validation
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for inputs, labels in val_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Track statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update confusion matrix
                for t, p in zip(labels.view(-1), predicted.view(-1)):
                    confusion_mat[t.long(), p.long()] += 1
                
                # Class-wise accuracy
                for i in range(5):
                    label_mask = (labels == i)
                    val_class_total[i] += label_mask.sum().item()
                    if label_mask.sum() > 0:
                        val_class_correct[i] += (predicted[label_mask] == i).sum().item()
                
                # Update progress bar
                val_bar.set_postfix(loss=loss.item(), acc=correct/total if total > 0 else 0)
        
        # Calculate epoch metrics
        epoch_val_loss = running_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0
        epoch_val_acc = correct / total if total > 0 else 0
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Update learning rate based on validation loss
        scheduler.step(epoch_val_loss)
        
        # Print metrics
        epoch_end = time.time()
        epoch_time = (epoch_end - epoch_start) / 60  # in minutes
        
        print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f} minutes")
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        
        # Print class-wise training accuracy
        for i in range(5):
            if class_total[i] > 0:
                print(f"Training Accuracy of class {i}: {100 * class_correct[i] / class_total[i]:.2f}%")
            else:
                print(f"Training Accuracy of class {i}: N/A (no training examples)")
        
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        
        # Print class-wise validation accuracy
        for i in range(5):
            if val_class_total[i] > 0:
                print(f"Validation Accuracy of class {i}: {100 * val_class_correct[i] / val_class_total[i]:.2f}%")
            else:
                print(f"Validation Accuracy of class {i}: N/A (no validation examples)")
        
        # Calculate and print per-class metrics
        for i in range(5):
            tp = confusion_mat[i, i]
            fp = confusion_mat[:, i].sum() - tp
            fn = confusion_mat[i, :].sum() - tp
            tn = confusion_mat.sum() - (tp + fp + fn)
            
            # Calculate performance metrics
            sensitivity = tp / (tp + fn) if tp + fn > 0 else 0
            specificity = tn / (tn + fp) if tn + fp > 0 else 0
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            
            print(f"Class {i} - Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, Precision: {precision:.4f}")
        
        # Check if we should save the model (when validation loss improves)
        if epoch_val_loss < best_val_loss:
            print(f"Validation loss decreased ({best_val_loss:.6f} --> {epoch_val_loss:.6f}). Saving model...")
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            counter = 0  # Reset early stopping counter
        else:
            counter += 1
            print(f"Early stopping counter: {counter} out of {patience}")
            if counter >= patience:
                print("Early stopping triggered")
                break
    
    total_time = (time.time() - start_time) / 60  # in minutes
    print(f"Training completed in {total_time:.2f} minutes")
    
    # Load the best model
    model.load_state_dict(torch.load(best_model_path))
    return model, history


# ====================================================================================================
# SECTION 6: MODEL TRAINING
# ====================================================================================================

# Train the model
print("\nStarting model training...")
trained_model, history = train_model(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer,
    scheduler,
    num_epochs=50,  # Epochs 50 for faster testing
    patience=20
)
print("Model training completed.")


# ====================================================================================================
# SECTION 7: VISUALIZATION AND EVALUATION
# ====================================================================================================

# Plot training history
print("\nPlotting training history...")
plt.figure(figsize=(12, 5))

# Plot loss curves
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy curves
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()
print("Training history plotted successfully.")


#evaluate_model function
def evaluate_model(model, test_loader, criterion, class_names=None):
    """
    Evaluate the model on test data and report metrics including Cohen's Kappa.
    
    Args:
        model (nn.Module): Model to evaluate
        test_loader (DataLoader): Test data loader
        criterion (nn.Module): Loss function
        class_names (list): Names of classes
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    if class_names is None:
        class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
    
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_labels = []
    all_predictions = []
    all_outputs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_outputs.extend(F.softmax(outputs, dim=1).cpu().numpy())
    
    test_loss = running_loss / len(test_loader.dataset)
    test_acc = correct / total
    
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    # Calculate and display confusion matrix and per-class metrics
    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    print(cm)
    
    # Create and display confusion matrix heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    sensitivity = recall_score(all_labels, all_predictions, average='weighted')
    
    # Calculate Cohen's Kappa
    kappa = cohen_kappa_score(all_labels, all_predictions)
    weighted_kappa = cohen_kappa_score(all_labels, all_predictions, weights='quadratic')
    
    print(f"\nCohen's Kappa: {kappa:.4f}")
    print(f"Quadratic Weighted Kappa: {weighted_kappa:.4f}")
    
    # Interpret Kappa value
    if kappa < 0:
        interpretation = "Poor agreement (worse than random)"
    elif kappa < 0.2:
        interpretation = "Slight agreement"
    elif kappa < 0.4:
        interpretation = "Fair agreement"
    elif kappa < 0.6:
        interpretation = "Moderate agreement"
    elif kappa < 0.8:
        interpretation = "Substantial agreement"
    else:
        interpretation = "Almost perfect agreement"
    
    print(f"Interpretation: {interpretation}")
    
    # Calculate specificity
    specificity_list = []
    for i in range(5):
        true_negatives = np.sum(np.delete(np.delete(cm, i, 0), i, 1))
        false_positives = np.sum(np.delete(cm[:, i], i))
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        specificity_list.append(specificity)
    
    specificity = np.mean(specificity_list)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    # For AUC-ROC
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    
    # One-hot encode true labels for multiclass ROC AUC
    y_true_bin = label_binarize(all_labels, classes=range(5))
    
    # Calculate AUC for each class
    auc_scores = []
    for i in range(5):
        if len(np.unique(y_true_bin[:, i])) > 1:
            auc_scores.append(roc_auc_score(y_true_bin[:, i], all_outputs[:, i]))
    
    auc_roc = np.mean(auc_scores) if auc_scores else 0
    
    # Calculate model size and ASR
    model_size_mb = os.path.getsize('baseline_model.pth') / (1024 * 1024)
    asr = accuracy / model_size_mb
    
    print("\nBaseline Paper Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"Model Size: {model_size_mb:.2f} MB")
    print(f"Accuracy-to-Size Ratio (ASR): {asr:.6f}")
    
    # Print per-class sensitivities
    class_recalls = recall_score(all_labels, all_predictions, average=None)
    print("\nPer-class Sensitivity:")
    for i, recall in enumerate(class_recalls):
        print(f"Class {i} ({class_names[i]}): {recall:.4f}")
    
    # Generate a detailed per-class confusion matrix visualization
    plt.figure(figsize=(12, 10))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot normalized confusion matrix
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Normalized Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig('normalized_confusion_matrix.png')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'auc_roc': auc_roc,
        'model_size_mb': model_size_mb,
        'asr': asr,
        'class_recalls': class_recalls,
        'confusion_matrix': cm,
        'kappa': kappa,
        'weighted_kappa': weighted_kappa
    }

# Evaluate the model
print("\nEvaluating model...")
class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
test_metrics = evaluate_model(trained_model, test_loader, criterion, class_names)
print("Model evaluation completed.")


# Function to plot class-wise performance metrics as bar charts
def plot_class_performance(test_metrics, class_names):
    """
    Plot per-class performance metrics as bar charts.
    
    Args:
        test_metrics (dict): Dictionary containing evaluation metrics
        class_names (list): Names of classes
    """
    # Extract class-wise metrics from confusion matrix
    cm = test_metrics['confusion_matrix']
    class_metrics = []
    
    for i in range(len(class_names)):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - (tp + fp + fn)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics.append({
            'class': class_names[i],
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1
        })
    
    # Create a figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Class-wise Performance Metrics', fontsize=16)
    
    # Plot precision
    x = range(len(class_names))
    axs[0, 0].bar(x, [m['precision'] for m in class_metrics], color='skyblue')
    axs[0, 0].set_title('Precision')
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(class_names, rotation=45)
    axs[0, 0].set_ylim(0, 1)
    
    # Plot recall/sensitivity
    axs[0, 1].bar(x, [m['recall'] for m in class_metrics], color='lightgreen')
    axs[0, 1].set_title('Recall/Sensitivity')
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(class_names, rotation=45)
    axs[0, 1].set_ylim(0, 1)
    
    # Plot specificity
    axs[1, 0].bar(x, [m['specificity'] for m in class_metrics], color='salmon')
    axs[1, 0].set_title('Specificity')
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(class_names, rotation=45)
    axs[1, 0].set_ylim(0, 1)
    
    # Plot F1 score
    axs[1, 1].bar(x, [m['f1'] for m in class_metrics], color='mediumpurple')
    axs[1, 1].set_title('F1 Score')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(class_names, rotation=45)
    axs[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('class_performance_metrics.png')
    plt.show()

# Call the function to plot class-wise performance
plot_class_performance(test_metrics, class_names)

# Inference time measurement function
def measure_inference_time(model, device, batch_size=1, image_size=(224, 224), n_runs=100):
    """
    Measure model inference time on both GPU and CPU.
    
    Args:
        model (nn.Module): Model to evaluate
        device (torch.device): Current device
        batch_size (int): Batch size for inference
        image_size (tuple): Input image dimensions
        n_runs (int): Number of runs to average
        
    Returns:
        dict: Dictionary containing GPU and CPU inference times
    """
    # Create a dummy input
    dummy_input = torch.randn(batch_size, 3, *image_size).to(device)
    
    # Warm-up runs to ensure fair measurement
    for _ in range(10):
        _ = model(dummy_input)
    
    # Measure GPU inference time
    if device.type == 'cuda':
        torch.cuda.synchronize()  # Wait for all CUDA operations to finish
        start = time.time()
        for _ in range(n_runs):
            _ = model(dummy_input)
            torch.cuda.synchronize()  # Wait for each forward pass to complete
        gpu_time = (time.time() - start) / n_runs
        print(f"GPU inference time: {gpu_time*1000:.2f} ms per image")
    else:
        gpu_time = float('nan')
        
    # Move model to CPU for CPU inference time
    model_cpu = model.to('cpu')
    dummy_input_cpu = torch.randn(batch_size, 3, *image_size)
    
    # Warm-up runs for CPU
    for _ in range(10):
        _ = model_cpu(dummy_input_cpu)
    
    # Measure CPU inference time
    start = time.time()
    for _ in range(n_runs):
        _ = model_cpu(dummy_input_cpu)
    cpu_time = (time.time() - start) / n_runs
    print(f"CPU inference time: {cpu_time*1000:.2f} ms per image")
    
    # Move model back to original device
    model.to(device)
    
    return {
        'gpu_time': gpu_time if device.type == 'cuda' else None,
        'cpu_time': cpu_time
    }

# Measure inference time
print("\nMeasuring inference time...")
inference_times = measure_inference_time(trained_model, device)
print("Inference time measurement completed.")


# ====================================================================================================
# SECTION 8: VISUALIZATIONS FOR CONFUSION MATRIX
# ====================================================================================================

def plot_confusion_matrix_heatmap(cm, class_names):
    """
    Create a more detailed and customized confusion matrix heatmap.
    
    Args:
        cm (numpy.ndarray): Confusion matrix
        class_names (list): Names of the classes
    """
    # Calculate metrics for each class from confusion matrix
    n_classes = len(class_names)
    class_accuracy = np.zeros(n_classes)
    class_precision = np.zeros(n_classes)
    class_recall = np.zeros(n_classes)
    class_f1 = np.zeros(n_classes)
    
    for i in range(n_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - (tp + fp + fn)
        
        class_accuracy[i] = (tp + tn) / np.sum(cm)
        class_precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        class_recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        class_f1[i] = 2 * class_precision[i] * class_recall[i] / (class_precision[i] + class_recall[i]) if (class_precision[i] + class_recall[i]) > 0 else 0
    
    # Set up the figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Raw counts confusion matrix
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=class_names, yticklabels=class_names)
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    ax1.set_title('Confusion Matrix (Raw Counts)')
    
    # 2. Row-normalized confusion matrix (recall/sensitivity for each class)
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    row_sums = cm.sum(axis=1)
    cm_row_norm = cm / row_sums[:, np.newaxis]
    sns.heatmap(cm_row_norm, annot=True, fmt='.2f', cmap='Greens', ax=ax2,
                xticklabels=class_names, yticklabels=class_names)
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    ax2.set_title('Row-Normalized Confusion Matrix (Recall/Sensitivity)')
    
    # 3. Column-normalized confusion matrix (precision for each class)
    ax3 = plt.subplot2grid((2, 2), (1, 0))
    col_sums = cm.sum(axis=0)
    cm_col_norm = cm / col_sums[np.newaxis, :]
    sns.heatmap(cm_col_norm, annot=True, fmt='.2f', cmap='Oranges', ax=ax3,
                xticklabels=class_names, yticklabels=class_names)
    ax3.set_xlabel('Predicted Label')
    ax3.set_ylabel('True Label')
    ax3.set_title('Column-Normalized Confusion Matrix (Precision)')
    
    # 4. Per-class metrics as a bar chart
    ax4 = plt.subplot2grid((2, 2), (1, 1))
    metrics_df = pd.DataFrame({
        'Accuracy': class_accuracy,
        'Precision': class_precision,
        'Recall': class_recall,
        'F1-Score': class_f1
    }, index=class_names)
    metrics_df.plot(kind='bar', ax=ax4, rot=45)
    ax4.set_ylim(0, 1)
    ax4.set_title('Per-Class Performance Metrics')
    ax4.set_ylabel('Score')
    ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    
    plt.tight_layout()
    plt.savefig('detailed_confusion_matrix.png')
    plt.show()

# Create the detailed confusion matrix visualization
print("\nGenerating detailed confusion matrix visualization...")
plot_confusion_matrix_heatmap(test_metrics['confusion_matrix'], class_names)
print("Detailed confusion matrix visualization completed.")


def calculate_cohen_kappa(all_labels, all_predictions):
    """
    Calculate Cohen's Kappa coefficient, which measures inter-rater agreement.
    
    Args:
        all_labels (array-like): True labels
        all_predictions (array-like): Predicted labels
        
    Returns:
        float: Cohen's Kappa score
    """
    # Calculate Cohen's Kappa
    kappa = cohen_kappa_score(all_labels, all_predictions)
    
    # Interpret Kappa value
    if kappa < 0:
        interpretation = "Poor agreement (worse than random)"
    elif kappa < 0.2:
        interpretation = "Slight agreement"
    elif kappa < 0.4:
        interpretation = "Fair agreement"
    elif kappa < 0.6:
        interpretation = "Moderate agreement"
    elif kappa < 0.8:
        interpretation = "Substantial agreement"
    else:
        interpretation = "Almost perfect agreement"
    
    print(f"\nCohen's Kappa: {kappa:.4f}")
    print(f"Interpretation: {interpretation}")
    
    # For quadratic weighted kappa (commonly used in DR grading)
    weighted_kappa = cohen_kappa_score(all_labels, all_predictions, weights='quadratic')
    print(f"Quadratic Weighted Kappa: {weighted_kappa:.4f}")
    
    return kappa, weighted_kappa

# Calculate Cohen's Kappa from test_metrics
all_labels = np.array([])
all_predictions = np.array([])

# Extract actual and predicted labels from confusion matrix
cm = test_metrics['confusion_matrix']
for i in range(len(class_names)):
    for j in range(len(class_names)):
        count = cm[i, j]
        all_labels = np.append(all_labels, np.full(int(count), i))
        all_predictions = np.append(all_predictions, np.full(int(count), j))

# Calculate and display Cohen's Kappa
kappa, weighted_kappa = calculate_cohen_kappa(all_labels, all_predictions)

# ====================================================================================================
# SECTION 10: ENHANCED SUMMARY
# ====================================================================================================

# Print enhanced summary of results
print("\nEnhanced Summary of Results:")
print("=" * 70)
print(f"Model: MobileNetV2 with dataset fusion")
print(f"Model Size: {test_metrics['model_size_mb']:.2f} MB")
print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"Quadratic Weighted Kappa: {weighted_kappa:.4f}")
print(f"Test Sensitivity (Recall): {test_metrics['sensitivity']:.4f}")
print(f"Test Specificity: {test_metrics['specificity']:.4f}")
print(f"Test Precision: {test_metrics['precision']:.4f}")
print(f"Test F1-Score: {test_metrics['f1']:.4f}")
print(f"AUC-ROC: {test_metrics['auc_roc']:.4f}")
print(f"Accuracy-to-Size Ratio (ASR): {test_metrics['asr']:.6f}")
print("\nPer-class Sensitivities:")
for i, recall in enumerate(test_metrics['class_recalls']):
    print(f"  - {class_names[i]}: {recall:.4f}")
print(f"\nGPU Inference Time: {inference_times['gpu_time']*1000 if inference_times['gpu_time'] else 'N/A':.2f} ms per image")
print(f"CPU Inference Time: {inference_times['cpu_time']*1000:.2f} ms per image")
print("=" * 70)

# Visualize a comparison of all metrics
def plot_metrics_comparison():
    """
    Create a visual comparison of all evaluation metrics in a single radar chart.
    """
    metrics = {
        'Accuracy': test_metrics['accuracy'],
        'Sensitivity': test_metrics['sensitivity'],
        'Specificity': test_metrics['specificity'],
        'Precision': test_metrics['precision'],
        'F1-Score': test_metrics['f1'],
        'AUC-ROC': test_metrics['auc_roc'],
        'Cohen\'s Kappa': kappa,
        'Weighted Kappa': weighted_kappa
    }
    
    # Create radar chart
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    # Calculate angle for each category
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    
    # Complete the loop for the radar chart by appending the first value at the end
    values.append(values[0])
    angles.append(angles[0])
    categories.append(categories[0])
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Draw the chart
    ax.plot(angles, values, 'o-', linewidth=2, label='Metrics')
    ax.fill(angles, values, alpha=0.25)
    
    # Set category labels
    ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
    
    # Set radial limits
    ax.set_ylim(0, 1)
    
    # Add grid and labels
    ax.grid(True)
    plt.title('Model Performance Metrics', size=15)
    
    plt.tight_layout()
    plt.savefig('metrics_radar_chart.png')
    plt.show()

# Generate metrics comparison visualization
print("\nGenerating metrics comparison visualization...")
plot_metrics_comparison()
print("Metrics comparison visualization completed.")




