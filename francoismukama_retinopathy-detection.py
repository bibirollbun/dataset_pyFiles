import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
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

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import cohen_kappa_score

# Set seeds for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seeds
seed_everything()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
import seaborn as sns


# Set paths to datasets (reusing from baseline)
aptos_path = "/kaggle/input/aptos2019-blindness-detection"
eyepacs_path = "/kaggle/input/eyepacs-aptos-messidor-diabetic-retinopathy"
print(f"APTOS 2019 path: {aptos_path}")
print(f"EyePACS-APTOS-Messidor path: {eyepacs_path}")

# Function to get file list from EyePACS folder structure (reused from baseline)
def get_eyepacs_files(root_path, subset='train'):
    """Get files from EyePACS folder structure where images are organized in class folders"""
    files_dict = defaultdict(list)
    subset_path = os.path.join(root_path, 'augmented_resized_V2', subset)

    # Check if path exists
    if not os.path.exists(subset_path):
        print(f"Path does not exist: {subset_path}")
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

    total_files = sum(len(files) for files in files_dict.values())
    print(f"Found {total_files} images in EyePACS {subset} folder")

    # Print class distribution
    for cls in sorted(files_dict.keys()):
        print(f"  Class {cls}: {len(files_dict[cls])} images")

    return files_dict

# Function to load APTOS dataset (reused from baseline)
def load_aptos_dataset(aptos_path):
    """Load APTOS dataset from CSV file"""
    csv_path = os.path.join(aptos_path, 'train.csv')
    if not os.path.exists(csv_path):
        print(f"Error: APTOS CSV file not found at {csv_path}")
        return pd.DataFrame()

    df_aptos = pd.read_csv(csv_path)
    print(f"Loaded APTOS 2019: {len(df_aptos)} images")

    # Add source and image path
    df_aptos['source'] = 'aptos'
    df_aptos['image_path'] = df_aptos['id_code'].apply(
        lambda x: os.path.join(aptos_path, 'train_images', f"{x}.png")
    )

    # Rename columns for consistency
    df_aptos = df_aptos.rename(columns={'id_code': 'image_id', 'diagnosis': 'class'})

    # Print class distribution
    print("APTOS class distribution:")
    print(df_aptos['class'].value_counts().sort_index())

    return df_aptos

# Execute load_aptos_dataset
print("Executing load_aptos_dataset function...")
aptos_df = load_aptos_dataset(aptos_path)
print("APTOS dataset loaded successfully.")


def create_balanced_dataset(aptos_df, eyepacs_files, target_count_per_class=3000):
    """Create balanced dataset by combining APTOS and EyePACS images"""
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
    
    # Verify image paths exist
    sample_size = min(10, len(balanced_df))
    for idx, row in balanced_df.sample(sample_size).iterrows():
        print(f"Checking {row['image_path']}: {os.path.exists(row['image_path'])}")
    
    print("\nFinal class distribution:")
    print(balanced_df['class'].value_counts().sort_index())
    
    return balanced_df
# Execute get_eyepacs_files
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

# Execute create_balanced_dataset
print("\nExecuting create_balanced_dataset function...")
balanced_df = create_balanced_dataset(aptos_df, eyepacs_files, target_count_per_class=3000)
print("Balanced dataset created successfully.")



# Split dataset into train, validation, and test sets (reused from baseline)
def create_data_splits(balanced_df, test_size=0.15, val_size=0.15):
    # First split out test set
    train_val_df, test_df = train_test_split(
        balanced_df,
        test_size=test_size,
        random_state=42,
        stratify=balanced_df['class']
    )

    # Recalculate validation size relative to remaining data
    relative_val_size = val_size / (1 - test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=42,
        stratify=train_val_df['class']
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

# Execute create_data_splits
print("\nExecuting create_data_splits function...")
train_df, val_df, test_df = create_data_splits(balanced_df)
print("Data splits created successfully.")


# Data transformations for EfficientNet
# EfficientNet-B0 uses 224x224 input size
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAffine(
        degrees=20,
        translate=(0.1, 0.1),
        scale=(0.8, 1.2),
    ),
    transforms.ColorJitter(brightness=(0.5, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset class (reused from baseline)
class DiabeticRetinopathyDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        label = self.dataframe.iloc[idx]['class']

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        return image, label

# Create datasets and dataloaders
print("\nCreating datasets and dataloaders...")
# Create datasets
train_dataset = DiabeticRetinopathyDataset(train_df, transform=train_transform)
val_dataset = DiabeticRetinopathyDataset(val_df, transform=val_transform)
test_dataset = DiabeticRetinopathyDataset(test_df, transform=val_transform)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,  # Smaller batch size for EfficientNet
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print(f"Created dataloaders with {len(train_dataset)} training, {len(val_dataset)} validation, and {len(test_dataset)} test images")

# class CustomEfficientNetB0(nn.Module):
#     def __init__(self, num_classes=5):
#         super(CustomEfficientNetB0, self).__init__()
#         # Load pretrained EfficientNet-B0
#         self.efficientnet = models.efficientnet_b0(pretrained=True)
        
#         # Get the number of features in the last layer
#         in_features = self.efficientnet.classifier[1].in_features
        
#         # Replace classifier with a more compact head
#         self.efficientnet.classifier = nn.Sequential(
#             nn.Dropout(p=0.3),
#             nn.Linear(in_features, 64),  # Reduced from original size
#             nn.ReLU(),
#             nn.Linear(64, num_classes)
#         )
    
#     def forward(self, x):
#         return self.efficientnet(x)

class CustomEfficientNetB0(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomEfficientNetB0, self).__init__()
        # Replacing EfficientNet with MobileNetV3Small
        self.backbone = models.mobilenet_v3_small(pretrained=True)
        # Replacing EfficientNet with shufflenet_v2_x0_5
        # self.backbone = models.shufflenet_v2_x0_5(pretrained=True)
        
        # Get the number of features from the last layer
        in_features = self.backbone.classifier[0].in_features #(changed to use sufflenet)
        # in_features = self.backbone.fc.in_features
        
        # Create a more compact classifier
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 32),  # Reduced from 64
            nn.Hardswish(),  # More efficient than ReLU
            nn.Dropout(p=0.2),
            nn.Linear(32, num_classes)
        )

    #     self.backbone.fc = nn.Sequential( # classifier => fc
    #         nn.Linear(in_features, 32),  # Reduced from 64
    #         nn.Hardswish(),  # More efficient than ReLU
    #         nn.Dropout(p=0.2),
    #         nn.Linear(32, num_classes)
    # )
    
    def forward(self, x):
        return self.backbone(x)

# Create teacher model (MobileNetV2 from baseline)
class TeacherModel(nn.Module):
    def __init__(self, num_classes=5):
        super(TeacherModel, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)

        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.mobilenet.last_channel, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        return self.mobilenet(x)

# Knowledge Distillation Loss Function
class DistillationLoss(nn.Module): # making alpha high and temperature
    def __init__(self, alpha=0.7, temperature=4.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha  # Weight for soft targets (teacher predictions)
        self.temperature = temperature  # Temperature for softening probability distributions
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_outputs, teacher_outputs, targets):
        # Hard target loss
        hard_loss = self.ce_loss(student_outputs, targets)

        # Soft target loss
        soft_student = F.log_softmax(student_outputs / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_outputs / self.temperature, dim=1)
        soft_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)

        # Combined loss
        loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss

        return loss

# Initialize models
print("\nInitializing models...")
# Create the student model (EfficientNet-B0)
student_model = CustomEfficientNetB0(num_classes=5).to(device)

# Create the teacher model (from baseline)
teacher_model = TeacherModel(num_classes=5).to(device)

# Try to load the teacher model from saved weights
try:
    teacher_model.load_state_dict(torch.load('best_dr_model.pth'))
    print("Loaded teacher model weights successfully.")
except Exception as e:
    print(f"Error loading teacher model weights: {e}")
    print("Will use a pre-trained teacher model without fine-tuning.")

# Set teacher model to evaluation mode
teacher_model.eval()

# Define optimizer
optimizer = optim.Adam(student_model.parameters(), lr=0.0005)  # Lower learning rate for EfficientNet

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    verbose=True
)

# Distillation loss
distillation_loss = DistillationLoss(alpha=0.7, temperature=4.0)  # Slightly higher alpha for EfficientNet

print("Models initialized successfully.")


# Training function with knowledge distillation
def train_model_with_distillation(student_model, teacher_model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100, patience=10):
    best_val_loss = float('inf')
    counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    start_time = time.time()
    
    best_model_path = 'best_efficientnet_dr_model.pth'
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training phase
        student_model.train()
        teacher_model.eval()  # Teacher is always in eval mode
        
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
            
            optimizer.zero_grad()
            
            # Get outputs from both models
            student_outputs = student_model(inputs)
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)
            
            # Calculate distillation loss
            loss = criterion(student_outputs, teacher_outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(student_outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Class-wise accuracy
            for i in range(5):
                label_mask = (labels == i)
                class_total[i] += label_mask.sum().item()
                if label_mask.sum() > 0:
                    class_correct[i] += (predicted[label_mask] == i).sum().item()
            
            train_bar.set_postfix(loss=loss.item(), acc=correct/total if total > 0 else 0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0
        epoch_train_acc = correct / total if total > 0 else 0
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        
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
        
        # Validation phase
        student_model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Class-wise validation accuracy
        val_class_correct = [0] * 5
        val_class_total = [0] * 5
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for inputs, labels in val_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                student_outputs = student_model(inputs)
                
                # For validation, we can use standard cross-entropy loss
                loss = F.cross_entropy(student_outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(student_outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Class-wise accuracy
                for i in range(5):
                    label_mask = (labels == i)
                    val_class_total[i] += label_mask.sum().item()
                    if label_mask.sum() > 0:
                        val_class_correct[i] += (predicted[label_mask] == i).sum().item()
                
                val_bar.set_postfix(loss=loss.item(), acc=correct/total if total > 0 else 0)
        
        epoch_val_loss = running_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0
        epoch_val_acc = correct / total if total > 0 else 0
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Print class-wise validation accuracy
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        for i in range(5):
            if val_class_total[i] > 0:
                print(f"Validation Accuracy of class {i}: {100 * val_class_correct[i] / val_class_total[i]:.2f}%")
            else:
                print(f"Validation Accuracy of class {i}: N/A (no validation examples)")
        
        # Update learning rate based on validation loss
        scheduler.step(epoch_val_loss)
        
        # Check if we should save the model
        if epoch_val_loss < best_val_loss:
            print(f"Validation loss decreased ({best_val_loss:.6f} --> {epoch_val_loss:.6f}). Saving model...")
            best_val_loss = epoch_val_loss
            torch.save(student_model.state_dict(), best_model_path)
            counter = 0
        else:
            counter += 1
            print(f"Early stopping counter: {counter} out of {patience}")
            if counter >= patience:
                print("Early stopping triggered")
                break
    
    total_time = (time.time() - start_time) / 60  # in minutes
    print(f"Training completed in {total_time:.2f} minutes")
    
    # Load the best model
    student_model.load_state_dict(torch.load(best_model_path))
    return student_model, history


# Train the model
print("\nStarting model training with knowledge distillation...")
trained_student_model, history = train_model_with_distillation(
    student_model,
    teacher_model,
    train_loader,
    val_loader,
    distillation_loss,
    optimizer,
    scheduler,
    num_epochs=50,  # epochs 50
    patience=20
)
print("Model training with knowledge distillation completed.")


# Plot training history
print("\nPlotting training history...")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('efficientnet_model_training_history.png')
plt.show()
print("Training history plotted successfully.")


import torch.nn.utils.prune as prune  # for pruning

def prune_model(model, prune_amount=0.45):  # Increased from 0.35
    """Prune the model more aggressively with structured pruning"""
    print("\nPruning the model...")
    
    # Step 1: Apply global unstructured pruning
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=prune_amount,
    )
    
    # Step 2: Apply structured pruning to Conv layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.out_channels > 8:
            # Skip pruning the first layer and very small layers
            if "0.0" not in name:
                # Prune 30% of channels in convolutional layers
                prune.ln_structured(module, name='weight', amount=0.3, n=2, dim=0)
    
    # Count the sparsity
    zero_weights = 0
    total_weights = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            zero_weights += torch.sum(module.weight == 0).item()
            total_weights += module.weight.numel()
    
    sparsity = 100. * zero_weights / total_weights
    print(f"Model pruned with overall sparsity: {sparsity:.2f}%")
    
    return model


# Fine-tune after pruning
def fine_tune_pruned_model(model, train_loader, val_loader, epochs=10):  # Increased epochs
    """More extensive fine-tuning of the pruned model"""
    print("\nFine-tuning pruned model...")

    # Cosine annealing learning rate for better convergence
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_model_path = 'best_pruned_mobilenet_model.pth'

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, desc=f'Fine-tune Epoch {epoch+1}/{epochs} [Train]')
        for inputs, labels in train_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            train_bar.set_postfix(loss=loss.item(), acc=correct/total if total > 0 else 0)
        
        # Update learning rate
        scheduler.step()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total

        print(f"Fine-tune Epoch {epoch+1}/{epochs}")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)

    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    print("Fine-tuning completed.")

    return model

# Apply pruning and fine-tuning
print("\nApplying pruning and fine-tuning...")
pruned_model = prune_model(trained_student_model, prune_amount=0.25)  # Less aggressive pruning for EfficientNet
fine_tuned_model = fine_tune_pruned_model(pruned_model, train_loader, val_loader, epochs=3)
print("Pruning and fine-tuning completed.")


# Make pruning permanent
def make_pruning_permanent(model):
    """Make the pruning permanent by removing the masks"""
    print("\nMaking pruning permanent...")

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            try:
                torch.nn.utils.prune.remove(module, 'weight')
            except:
                print(f"Could not remove pruning from {name}, skipping")

    print("Pruning made permanent.")
    return model

# Make pruning permanent
final_model = make_pruning_permanent(fine_tuned_model)

# Apply quantization
def quantize_model(model):
    """Apply dynamic int8 quantization to the model"""
    print("\nQuantizing model with dynamic int8 quantization...")
    
    # Move model to CPU for quantization
    model = model.to('cpu').eval()
    
    # Apply dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d, nn.BatchNorm2d},
        dtype=torch.qint8
    )
    
    print("Model quantized with int8 precision.")
    return quantized_model

# Apply quantization
print("\nApplying quantization...")
quantized_model = quantize_model(final_model)
print("Quantization applied.")

# Save the final model
torch.save(quantized_model.state_dict(), 'final_efficientnet_dr_model.pth')
print("Final EfficientNet model saved.")


# Evaluation function
def evaluate_model(model, test_loader, criterion, class_names=None):
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

    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    sensitivity = recall_score(all_labels, all_predictions, average='weighted')

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
    model_size_mb = os.path.getsize('final_efficientnet_dr_model.pth') / (1024 * 1024)
    asr = accuracy / model_size_mb

    print("\nEfficientNet Model Metrics:")
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
        'confusion_matrix': cm
    }


# Save the final model after pruning and fine-tuning
torch.save(final_model.state_dict(), 'final_pruned_model.pth')
print("Final pruned model saved.")

# Apply quantization
print("\nApplying quantization...")
# Move model to CPU for quantization
quantized_model = final_model.to('cpu').eval()

# Apply dynamic quantization to the model
quantized_model = torch.quantization.quantize_dynamic(
    quantized_model, 
    {nn.Linear, nn.Conv2d}, 
    dtype=torch.qint8
)

# Save the quantized model
torch.save(quantized_model.state_dict(), 'final_quantized_model.pth')
print("Quantized model saved.")

# IMPORTANT: Quantized models can only run on CPU, not on CUDA
print("\nEvaluating the final model on CPU...")
# Define criterion for evaluation
criterion = nn.CrossEntropyLoss()
class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']




# # For evaluation, we have two options:
# # 1. Use the non-quantized model on GPU (faster but larger)
# # 2. Use the quantized model on CPU (slower but smaller)
# # Let's implement both to compare:

# # Modified evaluation function to force CPU device
def evaluate_model_on_cpu(model, test_loader, criterion, class_names=None):
    if class_names is None:
        class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
    
    model = model.to('cpu')  # Force CPU
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_outputs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing on CPU'):
            # Move tensors to CPU
            inputs = inputs.to('cpu')
            labels_cpu = labels.to('cpu')  # Store CPU version for evaluation
            
            outputs = model(inputs)
            loss = criterion(outputs, labels_cpu)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels_cpu.size(0)
            correct += (predicted == labels_cpu).sum().item()
            
            all_labels.extend(labels_cpu.numpy())
            all_predictions.extend(predicted.numpy())
            all_outputs.extend(F.softmax(outputs, dim=1).numpy())
    
    test_loss = running_loss / len(test_loader.dataset)
    test_acc = correct / total
    
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    # Calculate and display confusion matrix and per-class metrics
    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    sensitivity = recall_score(all_labels, all_predictions, average='weighted')
    
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
    model_size_mb = os.path.getsize('final_quantized_model.pth') / (1024 * 1024)
    asr = accuracy / model_size_mb
    
    print("\nModel Metrics:")
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
        'confusion_matrix': cm
    }

# Also evaluate the non-quantized model on GPU for comparison
def evaluate_model_on_gpu(model, test_loader, criterion, class_names=None):
    if class_names is None:
        class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
    
    # Check if GPU is available
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    print(f"Evaluating non-quantized model on: {device}")
    
    model = model.to(device)
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_outputs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing on GPU'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Move to CPU for numpy conversion
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_outputs.extend(F.softmax(outputs, dim=1).cpu().numpy())
    
    test_loss = running_loss / len(test_loader.dataset)
    test_acc = correct / total
    
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    # Follow the same metrics calculation as in the CPU function
    cm = confusion_matrix(all_labels, all_predictions)
    
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    sensitivity = recall_score(all_labels, all_predictions, average='weighted')
    
    # Calculate specificity
    specificity_list = []
    for i in range(5):
        true_negatives = np.sum(np.delete(np.delete(cm, i, 0), i, 1))
        false_positives = np.sum(np.delete(cm[:, i], i))
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        specificity_list.append(specificity)
    
    specificity = np.mean(specificity_list)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    # Calculate AUC-ROC
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    y_true_bin = label_binarize(all_labels, classes=range(5))
    
    auc_scores = []
    for i in range(5):
        if len(np.unique(y_true_bin[:, i])) > 1:
            auc_scores.append(roc_auc_score(y_true_bin[:, i], all_outputs[:, i]))
    
    auc_roc = np.mean(auc_scores) if auc_scores else 0
    
    # Calculate model size and ASR
    model_size_mb = os.path.getsize('final_pruned_model.pth') / (1024 * 1024)
    asr = accuracy / model_size_mb
    
    class_recalls = recall_score(all_labels, all_predictions, average=None)
    
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
        'confusion_matrix': cm
    }

# Run evaluation on both versions
print("\nEvaluating quantized model on CPU...")
cpu_metrics = evaluate_model_on_cpu(quantized_model, test_loader, criterion, class_names)

print("\nEvaluating original (non-quantized) model...")
gpu_metrics = evaluate_model_on_gpu(final_model, test_loader, criterion, class_names)

# Measure inference time correctly for both models
def measure_inference_time(quant_model, non_quant_model, batch_size=1, image_size=(224, 224), n_runs=100):
    # Create dummy inputs
    dummy_input_cpu = torch.randn(batch_size, 3, *image_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input_gpu = torch.randn(batch_size, 3, *image_size).to(device)
    
    # Ensure models are on correct devices
    quant_model = quant_model.to('cpu')
    non_quant_model = non_quant_model.to(device)
    
    # 1. Measure quantized model on CPU
    # Warm-up runs
    for _ in range(10):
        with torch.no_grad():
            _ = quant_model(dummy_input_cpu)
    
    # Measure inference time
    start = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = quant_model(dummy_input_cpu)
    quant_cpu_time = (time.time() - start) / n_runs
    print(f"Quantized model CPU inference time: {quant_cpu_time*1000:.2f} ms per image")
    
    # 2. Measure non-quantized model on GPU (if available)
    if device.type == 'cuda':
        # Warm-up runs
        for _ in range(10):
            with torch.no_grad():
                _ = non_quant_model(dummy_input_gpu)
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_runs):
            with torch.no_grad():
                _ = non_quant_model(dummy_input_gpu)
            torch.cuda.synchronize()
        non_quant_gpu_time = (time.time() - start) / n_runs
        print(f"Non-quantized model GPU inference time: {non_quant_gpu_time*1000:.2f} ms per image")
    else:
        non_quant_gpu_time = float('nan')
    
    # 3. Measure non-quantized model on CPU
    non_quant_model = non_quant_model.to('cpu')
    
    # Warm-up runs
    for _ in range(10):
        with torch.no_grad():
            _ = non_quant_model(dummy_input_cpu)
    
    start = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = non_quant_model(dummy_input_cpu)
    non_quant_cpu_time = (time.time() - start) / n_runs
    print(f"Non-quantized model CPU inference time: {non_quant_cpu_time*1000:.2f} ms per image")
    
    return {
        'quant_cpu_time': quant_cpu_time,
        'non_quant_gpu_time': non_quant_gpu_time if device.type == 'cuda' else None,
        'non_quant_cpu_time': non_quant_cpu_time,
        'cpu_speedup': non_quant_cpu_time / quant_cpu_time
    }

# Measure inference times for both models
print("\nMeasuring inference times...")
inference_times = measure_inference_time(quantized_model, final_model)

# Print comprehensive summary
print("\n" + "="*70)
print("COMPLETE MODEL COMPARISON")
print("="*70)
print(f"{'Metric':<25} | {'Non-Quantized':<15} | {'Quantized':<15}")
print("-"*70)
print(f"{'Model Size (MB)':<25} | {gpu_metrics['model_size_mb']:<15.2f} | {cpu_metrics['model_size_mb']:<15.2f}")
print(f"{'Accuracy':<25} | {gpu_metrics['accuracy']:<15.4f} | {cpu_metrics['accuracy']:<15.4f}")
print(f"{'Sensitivity':<25} | {gpu_metrics['sensitivity']:<15.4f} | {cpu_metrics['sensitivity']:<15.4f}")
print(f"{'Specificity':<25} | {gpu_metrics['specificity']:<15.4f} | {cpu_metrics['specificity']:<15.4f}")
print(f"{'F1-Score':<25} | {gpu_metrics['f1']:<15.4f} | {cpu_metrics['f1']:<15.4f}")
print(f"{'ASR (Accuracy/Size)':<25} | {gpu_metrics['asr']:<15.6f} | {cpu_metrics['asr']:<15.6f}")

# Print inference times
print("-"*70)
print(f"{'CPU Inference (ms)':<25} | {inference_times['non_quant_cpu_time']*1000:<15.2f} | {inference_times['quant_cpu_time']*1000:<15.2f}")
if inference_times['non_quant_gpu_time'] is not None:
    print(f"{'GPU Inference (ms)':<25} | {inference_times['non_quant_gpu_time']*1000:<15.2f} | {'N/A':<15}")
# print(f"{'CPU Speedup':<25} | {'1.00x':<15} | {inference_times['cpu_speedup']:<15.2f}x")
print("="*70)

# Visualize confusion matrix for the quantized model
try:
    import seaborn as sns
    plt.figure(figsize=(10, 8))
    sns.heatmap(cpu_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Quantized Model')
    plt.tight_layout()
    plt.show()
except ImportError:
    print("Seaborn not available for confusion matrix visualization")

print("\nModel evaluation and analysis completed successfully!")




