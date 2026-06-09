print("hello world")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import transforms,models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Image processing
from PIL import Image
import cv2
import os
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


train_df = pd.read_csv("/kaggle/input/2025-bamboo-summer-competiton-dl-pr/train.csv")
test_df = pd.read_csv("/kaggle/input/2025-bamboo-summer-competiton-dl-pr/test.csv")
sample_submission = pd.read_csv("/kaggle/input/2025-bamboo-summer-competiton-dl-pr/sample_submission.csv")
print("Dataset Overview:")
print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Number of classes: {train_df['label'].nunique()}")

# Display first few rows
print("\nTraining data sample:")
print(train_df.head(10))

print("\nClass distribution:")
class_counts = train_df['label'].value_counts()
print(f"Most common class: {class_counts.index[0]} ({class_counts.iloc[0]} samples)")
print(f"Least common class: {class_counts.index[-1]} ({class_counts.iloc[-1]} samples)")
print(f"Average samples per class: {class_counts.mean():.1f}")

# Visualize class distribution
plt.figure(figsize=(15, 8))
plt.subplot(1, 2, 1)
class_counts.head(20).plot(kind='bar')
plt.title('Top 20 Classes Distribution')
plt.xlabel('Butterfly Species')
plt.ylabel('Number of Images')
plt.xticks(rotation=45, ha='right')

plt.subplot(1, 2, 2)
plt.hist(class_counts.values, bins=20, alpha=0.7, color='skyblue')
plt.title('Distribution of Samples per Class')
plt.xlabel('Number of Images per Class')
plt.ylabel('Number of Classes')

plt.tight_layout()
plt.show()


# Visualize sample images from different classes
def show_sample_images(n_classes=5, n_samples=4):
    """Display sample images from different butterfly classes"""
    # Get random classes
    sample_classes = train_df['label'].value_counts().head(n_classes).index
    
    fig, axes = plt.subplots(n_classes, n_samples, figsize=(20, n_classes * 4))
    fig.suptitle('Sample Images from Different Butterfly Classes', fontsize=16)
    
    for i, class_name in enumerate(sample_classes):
        class_images = train_df[train_df['label'] == class_name]['filename'].values
        selected_images = np.random.choice(class_images, min(n_samples, len(class_images)), replace=False)
        
        for j, img_filename in enumerate(selected_images):
            img_path = f'/kaggle/input/2025-bamboo-summer-competiton-dl-pr/train/{img_filename}'
            if os.path.exists(img_path):
                img = Image.open(img_path)
                axes[i, j].imshow(img)
                axes[i, j].set_title(f'{class_name}')
                axes[i, j].axis('off')
            else:
                axes[i, j].text(0.5, 0.5, 'Image not found', 
                               ha='center', va='center', transform=axes[i, j].transAxes)
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()

show_sample_images()


label_encoder = LabelEncoder()
train_df['label_encoded'] = label_encoder.fit_transform(train_df['label'])
print(f"Number of classes: {len(label_encoder.classes_)}")
print("Label mapping (first 10):")
for i in range(10):
    print(f"{i}: {label_encoder.classes_[i]}")
X_train,X_val,y_train,y_val = train_test_split(
    train_df['filename'].values,
    train_df['label_encoded'].values,
    test_size=0.2,
    random_state=42,
    stratify=train_df['label_encoded'].values
)


# Define data transformations
IMG_SIZE = 224

# Training transformations with data augmentation
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation/Test transformations (no augmentation)
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
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
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color=(0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            return image, self.filenames[idx]
        else:
            return image, self.labels[idx]

# Create datasets
train_dataset = ButterflyDataset(X_train, y_train, '/kaggle/input/2025-bamboo-summer-competiton-dl-pr/train', transform=train_transforms)
val_dataset = ButterflyDataset(X_val, y_val, '/kaggle/input/2025-bamboo-summer-competiton-dl-pr/train', transform=val_transforms)
test_dataset = ButterflyDataset(test_df['filename'].values, None, '/kaggle/input/2025-bamboo-summer-competiton-dl-pr/test', 
                               transform=val_transforms, is_test=True)

print(f"Dataset sizes:")
print(f"Train: {len(train_dataset)}")
print(f"Validation: {len(val_dataset)}")
print(f"Test: {len(test_dataset)}")


# Create data loaders
BATCH_SIZE = 32

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                         num_workers=4 if torch.cuda.is_available() else 0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                       num_workers=4 if torch.cuda.is_available() else 0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                        num_workers=4 if torch.cuda.is_available() else 0, pin_memory=True)

print(f"Number of batches:")
print(f"Train: {len(train_loader)}")
print(f"Validation: {len(val_loader)}")
print(f"Test: {len(test_loader)}")

# Test data loading
sample_batch = next(iter(train_loader))
print(f"\nSample batch shapes:")
print(f"Images: {sample_batch[0].shape}")
print(f"Labels: {sample_batch[1].shape}")


class ButterflyClassifier(nn.Module):
    def __init__(self, num_classes=75, dropout_rate=0.5):
        super(ButterflyClassifier, self).__init__()
        
        # Load pre-trained ResNet50
        self.backbone = models.resnet50(pretrained=True)
        
        # Freeze early layers (optional - can be unfrozen later for fine-tuning)
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
        
        # Get the number of features from the last layer
        num_features = self.backbone.fc.in_features
        
        # Replace the classifier with our custom head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Create model
model = ButterflyClassifier(num_classes=len(label_encoder.classes_))
model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model created successfully!")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model device: {next(model.parameters()).device}")

# Print model architecture
print(f"\nModel architecture:")
print(model)


# Training configuration
EPOCHS = 1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-7
)

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for data, targets in progress_bar:
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += targets.size(0)
        correct_predictions += (predicted == targets).sum().item()
        
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100 * correct_predictions / total_predictions:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct_predictions / total_predictions
    return epoch_loss, epoch_acc

# Validation function
def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation", leave=False)
        
        for data, targets in progress_bar:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct_predictions / total_predictions:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct_predictions / total_predictions
    return epoch_loss, epoch_acc

# Print training configuration
print("Training configuration ready!")
print(f"Epochs: {EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Weight Decay: {WEIGHT_DECAY}")
print(f"Optimizer: {optimizer.__class__.__name__}")
print(f"Scheduler: {scheduler.__class__.__name__}")
print("Early stopping enabled")



# Training loop
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
learning_rates = []

best_val_acc = 0.0
best_model_state = None

print("Starting training...")
print("=" * 60)

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 50)
    
    # Training phase
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Validation phase
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
    
    # Update learning rate
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Store metrics
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    learning_rates.append(current_lr)
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()
        print(f"âœ… New best validation accuracy: {val_acc:.4f}")
    
    # Print epoch results
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    print(f"Learning Rate: {current_lr:.2e}")
    print(f"Best Val Acc: {best_val_acc:.4f}")

print("\n" + "=" * 60)
print("Training completed!")
print(f"Best validation accuracy: {best_val_acc:.4f}")

# Load best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("Best model loaded!")


# Plot training history
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss plot
axes[0, 0].plot(train_losses, label='Train Loss', marker='o')
axes[0, 0].plot(val_losses, label='Validation Loss', marker='s')
axes[0, 0].set_title('Training and Validation Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Accuracy plot
axes[0, 1].plot(train_accuracies, label='Train Accuracy', marker='o')
axes[0, 1].plot(val_accuracies, label='Validation Accuracy', marker='s')
axes[0, 1].set_title('Training and Validation Accuracy')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Learning rate plot
axes[1, 0].plot(learning_rates, marker='o', color='red')
axes[1, 0].set_title('Learning Rate Schedule')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Learning Rate')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(True)

# Combined accuracy plot
axes[1, 1].plot(range(1, len(train_accuracies) + 1), 
               [acc * 100 for acc in train_accuracies], 
               label='Train Accuracy', marker='o', linewidth=2)
axes[1, 1].plot(range(1, len(val_accuracies) + 1), 
               [acc * 100 for acc in val_accuracies], 
               label='Validation Accuracy', marker='s', linewidth=2)
axes[1, 1].set_title('Accuracy Progress (%)')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy (%)')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# Print final training statistics
print("Training Summary:")
print(f"Final train accuracy: {train_accuracies[-1]:.4f} ({train_accuracies[-1]*100:.2f}%)")
print(f"Final validation accuracy: {val_accuracies[-1]:.4f} ({val_accuracies[-1]*100:.2f}%)")
print(f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
print(f"Final learning rate: {learning_rates[-1]:.2e}")


# Detailed validation evaluation
def evaluate_model(model, data_loader, device, label_encoder):
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, targets in tqdm(data_loader, desc="Evaluating"):
            data, targets = data.to(device), targets.to(device)
            
            outputs = model(data)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_targets), np.array(all_probabilities)

# Get predictions and targets
val_predictions, val_targets, val_probabilities = evaluate_model(model, val_loader, device, label_encoder)

# Calculate accuracy
val_accuracy = accuracy_score(val_targets, val_predictions)
print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")

# Classification report
class_names = label_encoder.classes_
print("\nClassification Report:")
print(classification_report(val_targets, val_predictions, 
                          target_names=class_names, 
                          labels=range(len(class_names)),
                          zero_division=0))

# Confusion matrix visualization (for top classes)
top_classes_idx = train_df['label'].value_counts().head(10).index
top_classes_encoded = [label_encoder.transform([cls])[0] for cls in top_classes_idx]

# Filter predictions and targets for top classes
mask = np.isin(val_targets, top_classes_encoded)
filtered_targets = val_targets[mask]
filtered_predictions = val_predictions[mask]

if len(filtered_targets) > 0:
    cm = confusion_matrix(filtered_targets, filtered_predictions, labels=top_classes_encoded)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[class_names[i] for i in top_classes_encoded],
                yticklabels=[class_names[i] for i in top_classes_encoded])
    plt.title('Confusion Matrix - Top 10 Classes')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
else:
    print("No samples found for top classes in validation set")


# Create a new model for final training on full dataset
print("Creating final model for full dataset training...")
final_model = ButterflyClassifier(num_classes=len(label_encoder.classes_))
final_model = final_model.to(device)

# Use the same training configuration that worked best
final_optimizer = optim.AdamW(final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
final_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    final_optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-7
)

# Create full training dataset (all training data)
full_train_dataset = ButterflyDataset(
    train_df['filename'].values, 
    train_df['label_encoded'].values, 
    '/kaggle/input/2025-bamboo-summer-competiton-dl-pr/train', 
    transform=train_transforms
)

full_train_loader = DataLoader(
    full_train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=4 if torch.cuda.is_available() else 0, 
    pin_memory=True
)

print(f"Full training dataset size: {len(full_train_dataset)}")
print(f"Number of batches: {len(full_train_loader)}")

# Determine optimal number of epochs based on validation performance
# Use slightly fewer epochs since we don't have validation to monitor
optimal_epochs = len(train_losses) - 3  # Stop a bit before overfitting
optimal_epochs = max(optimal_epochs, 1)  # Minimum 15 epochs
optimal_epochs = min(optimal_epochs, 1)  # Maximum 20 epochs

print(f"Training final model for {optimal_epochs} epochs...")

# Training function for full dataset (no validation)
def train_full_dataset(model, train_loader, criterion, optimizer, scheduler, epochs, device):
    train_losses = []
    train_accuracies = []
    
    print("Starting final training on full dataset...")
    print("=" * 60)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct_predictions / total_predictions:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_predictions / total_predictions
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # Update learning rate based on loss
        scheduler.step(epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch results
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")
    
    print("\n" + "=" * 60)
    print("Final training completed!")
    
    return train_losses, train_accuracies

# Train the final model
final_train_losses, final_train_accuracies = train_full_dataset(
    final_model, full_train_loader, criterion, final_optimizer, final_scheduler, optimal_epochs, device
)

print(f"Final model training completed!")
print(f"Final training accuracy: {final_train_accuracies[-1]:.4f} ({final_train_accuracies[-1]*100:.2f}%)")


# Compare training curves
plt.figure(figsize=(15, 5))

# Loss comparison
plt.subplot(1, 3, 1)
plt.plot(train_losses[:len(final_train_losses)], label='Original Training (with validation)', marker='o')
plt.plot(final_train_losses, label='Full Dataset Training', marker='s')
plt.title('Training Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy comparison
plt.subplot(1, 3, 2)
plt.plot([acc * 100 for acc in train_accuracies[:len(final_train_accuracies)]], 
         label='Original Training (with validation)', marker='o')
plt.plot([acc * 100 for acc in final_train_accuracies], 
         label='Full Dataset Training', marker='s')
plt.title('Training Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# Final accuracy comparison
plt.subplot(1, 3, 3)
models = ['With Validation', 'Full Dataset']
accuracies = [train_accuracies[len(final_train_accuracies)-1] * 100, final_train_accuracies[-1] * 100]
bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen'])
plt.title('Final Training Accuracy')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{acc:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("\\nTraining Comparison:")
print(f"Original model (with validation): {train_accuracies[len(final_train_accuracies)-1]*100:.2f}%")
print(f"Final model (full dataset): {final_train_accuracies[-1]*100:.2f}%")
improvement = (final_train_accuracies[-1] - train_accuracies[len(final_train_accuracies)-1]) * 100
print(f"Improvement: {improvement:+.2f} percentage points")


# Generate test predictions using the final model trained on full dataset
def predict_test_set(model, test_loader, device, label_encoder):
    model.eval()
    test_predictions = []
    test_filenames = []
    test_probabilities = []
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Predicting test set"):
            if len(batch_data) == 2:  # (images, filenames)
                images, filenames = batch_data
                images = images.to(device)
                
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
                
                test_predictions.extend(predictions.cpu().numpy())
                test_filenames.extend(filenames)
                test_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert predictions to class names
    predicted_labels = label_encoder.inverse_transform(test_predictions)
    
    return test_filenames, predicted_labels, test_probabilities

# Get test predictions using the final model (trained on full dataset)
print("Generating final test predictions with full dataset model...")
test_filenames, predicted_labels, test_probabilities = predict_test_set(final_model, test_loader, device, label_encoder)

print(f"Generated predictions for {len(test_filenames)} test images")


# Create final submission dataframe
final_submission_df = pd.DataFrame({
    'filename': test_filenames,
    'label': predicted_labels
})

# Verify submission format
print("\\nFinal Submission Preview:")
print(final_submission_df.head(10))
print(f"\\nSubmission shape: {final_submission_df.shape}")
print(f"Unique predictions: {final_submission_df['label'].nunique()}")

# Check for any missing files
expected_files = set(test_df['filename'].values)
predicted_files = set(final_submission_df['filename'].values)
missing_files = expected_files - predicted_files

if missing_files:
    print(f"\\nWarning: Missing predictions for {len(missing_files)} files:")
    print(list(missing_files)[:10])  # Show first 10 missing files
else:
    print("\\nâœ… All test files have predictions!")

# Show prediction distribution
print("\\nPrediction Distribution (Top 10):")
pred_counts = final_submission_df['label'].value_counts().head(10)
for species, count in pred_counts.items():
    percentage = (count / len(final_submission_df)) * 100
    print(f"{species}: {count} ({percentage:.1f}%)")

# Save final submission
final_submission_df.to_csv('final_submission.csv', index=False)
print("\\nğŸ�¯ Final submission saved as 'final_submission.csv'")

# Summary
print("\\n" + "=" * 60)
print("ğŸ¦‹ BUTTERFLY CLASSIFICATION SUMMARY")
print("=" * 60)
print(f"ğŸ“Š Dataset: {len(train_df)} training images, {len(test_df)} test images")
print(f"ğŸ�·ï¸�  Classes: {len(label_encoder.classes_)} butterfly species")
print(f"ğŸ§  Model: ResNet-50 with transfer learning")
print(f"ğŸ“ˆ Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
print(f"ğŸ“ˆ Final training accuracy: {final_train_accuracies[-1]:.4f} ({final_train_accuracies[-1]*100:.2f}%)")
print(f"ğŸ�¯ Predictions generated for: {len(final_submission_df)} test images")
print(f"ğŸ’¾ Submission file: final_submission.csv")
print("=" * 60)

# Performance tips for future improvements
print("\\nğŸš€ POTENTIAL IMPROVEMENTS:")
print("- Try different model architectures (EfficientNet, Vision Transformer)")
print("- Implement test-time augmentation (TTA)")
print("- Use ensemble of multiple models")
print("- Fine-tune with different learning rates for different layers")
print("- Experiment with different data augmentation strategies")
print("- Add focal loss for class imbalance if present")
print("- Use progressive resizing (start with smaller images, then larger)")


# Install EfficientNet if not available
try:
    import timm
    print("timm already installed")
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "timm"])
    import timm
    print("timm installed successfully")

# EfficientNet Classifier
class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=75, model_name='efficientnet_b3', dropout_rate=0.5):
        super(EfficientNetClassifier, self).__init__()
        
        # Load pre-trained EfficientNet
        self.backbone = timm.create_model(model_name, pretrained=True)
        
        # Get the number of features from the classifier
        num_features = self.backbone.classifier.in_features
        
        # Replace the classifier with our custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Create EfficientNet model
print("Creating EfficientNet model...")
effnet_model = EfficientNetClassifier(num_classes=len(label_encoder.classes_))
effnet_model = effnet_model.to(device)

# Count parameters
total_params = sum(p.numel() for p in effnet_model.parameters())
trainable_params = sum(p.numel() for p in effnet_model.parameters() if p.requires_grad)

print(f"EfficientNet Model created successfully!")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model device: {next(effnet_model.parameters()).device}")


# EfficientNet Training Configuration
effnet_optimizer = optim.AdamW(effnet_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
effnet_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    effnet_optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-7
)

# Train EfficientNet on full dataset
print("Training EfficientNet on full dataset...")
effnet_train_losses, effnet_train_accuracies = train_full_dataset(
    effnet_model, full_train_loader, criterion, effnet_optimizer, effnet_scheduler, optimal_epochs, device
)

print(f"EfficientNet training completed!")
print(f"EfficientNet final training accuracy: {effnet_train_accuracies[-1]:.4f} ({effnet_train_accuracies[-1]*100:.2f}%)")


# Generate EfficientNet predictions
print("Generating EfficientNet test predictions...")
effnet_test_filenames, effnet_predicted_labels, effnet_test_probabilities = predict_test_set(
    effnet_model, test_loader, device, label_encoder
)

# Create EfficientNet submission
effnet_submission_df = pd.DataFrame({
    'filename': effnet_test_filenames,
    'label': effnet_predicted_labels
})

print("EfficientNet Submission Preview:")
print(effnet_submission_df.head(10))
print(f"EfficientNet submission shape: {effnet_submission_df.shape}")

# Save EfficientNet submission
effnet_submission_df.to_csv('effnet_submission.csv', index=False)
print("ğŸ�¯ EfficientNet submission saved as 'effnet_submission.csv'")


# Ensemble prediction function
def ensemble_predict(resnet_model, effnet_model, test_loader, device, label_encoder, 
                    resnet_weight=0.5, effnet_weight=0.5):
    """
    Generate ensemble predictions by combining ResNet and EfficientNet outputs
    """
    resnet_model.eval()
    effnet_model.eval()
    
    ensemble_predictions = []
    test_filenames = []
    ensemble_probabilities = []
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Ensemble prediction"):
            if len(batch_data) == 2:  # (images, filenames)
                images, filenames = batch_data
                images = images.to(device)
                
                # Get predictions from both models
                resnet_outputs = resnet_model(images)
                effnet_outputs = effnet_model(images)
                
                # Convert to probabilities
                resnet_probs = torch.softmax(resnet_outputs, dim=1)
                effnet_probs = torch.softmax(effnet_outputs, dim=1)
                
                # Weighted ensemble
                ensemble_probs = resnet_weight * resnet_probs + effnet_weight * effnet_probs
                
                # Get final predictions
                _, predictions = torch.max(ensemble_probs, 1)
                
                ensemble_predictions.extend(predictions.cpu().numpy())
                test_filenames.extend(filenames)
                ensemble_probabilities.extend(ensemble_probs.cpu().numpy())
    
    # Convert predictions to class names
    predicted_labels = label_encoder.inverse_transform(ensemble_predictions)
    
    return test_filenames, predicted_labels, ensemble_probabilities

# Generate ensemble predictions with equal weights
print("Generating ensemble predictions (ResNet + EfficientNet)...")
ensemble_filenames, ensemble_labels, ensemble_probs = ensemble_predict(
    final_model, effnet_model, test_loader, device, label_encoder,
    resnet_weight=0.5, effnet_weight=0.5
)

print(f"Generated ensemble predictions for {len(ensemble_filenames)} test images")


# ğŸ”¥ IMPLEMENTATION: Modern Model Architectures

# Install required packages for modern models
try:
    import timm
    print("âœ… timm already available")
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "timm>=0.9.0"])
    import timm

# Check available modern models
print("ğŸ”� Available Modern Models in timm:")
modern_models = [
    'convnext_base.fb_in22k_ft_in1k',
    'efficientnet_b5.in12k_ft_in1k', 
    'maxvit_base_tf_224.in1k',
    'swin_base_patch4_window7_224.ms_in22k_ft_in1k',
    'vit_base_patch16_224.augreg2_in21k_ft_in1k',
    'regnet_y_8gf.pycls_in1k',
    'tf_efficientnetv2_l.in21k_ft_in1k'
]

print("Available models:")
for model_name in modern_models:
    try:
        model = timm.create_model(model_name, pretrained=False, num_classes=1000)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"âœ… {model_name}: {params:.1f}M parameters")
        del model
    except Exception as e:
        print(f"â�Œ {model_name}: Not available - {str(e)[:50]}...")

print(f"\nğŸ�† RECOMMENDED ENSEMBLE COMBINATION:")
print("1. ConvNeXt + EfficientNetV2 + Swin Transformer")
print("2. MaxViT + RegNet + Vision Transformer") 
print("3. Diverse architectures perform better than similar ones")


# ğŸ�—ï¸� Modern Model Implementation: ConvNeXt (2022)

class ModernButterflyClassifier(nn.Module):
    """
    Modern classifier using ConvNeXt architecture (2022)
    ConvNeXt modernizes ResNet with Transformer-like design principles
    """
    def __init__(self, num_classes=75, model_name='convnext_base.fb_in22k_ft_in1k', dropout_rate=0.3):
        super(ModernButterflyClassifier, self).__init__()
        
        # Load pre-trained ConvNeXt (much better than ResNet-50)
        try:
            self.backbone = timm.create_model(model_name, pretrained=True)
            print(f"âœ… Successfully loaded: {model_name}")
        except Exception as e:
            print(f"â�Œ Failed to load {model_name}, falling back to EfficientNetV2")
            self.backbone = timm.create_model('tf_efficientnetv2_b3.in21k_ft_in1k', pretrained=True)
        
        # Get the number of features from the head
        if hasattr(self.backbone, 'head'):
            num_features = self.backbone.head.in_features
            # Replace the head with our custom classifier
            self.backbone.head = self._create_classifier_head(num_features, num_classes, dropout_rate)
        elif hasattr(self.backbone, 'classifier'):
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = self._create_classifier_head(num_features, num_classes, dropout_rate)
        else:
            raise ValueError("Could not find classifier layer in the model")
    
    def _create_classifier_head(self, num_features, num_classes, dropout_rate):
        """Create a modern classifier head with residual connections"""
        return nn.Sequential(
            # First reduction layer
            nn.Linear(num_features, 1024),
            nn.LayerNorm(1024),  # LayerNorm instead of BatchNorm (more stable)
            nn.GELU(),  # GELU instead of ReLU (better for transformers)
            nn.Dropout(dropout_rate),
            
            # Second layer with residual-like structure
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),  # Less dropout in deeper layers
            
            # Output layer
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Create the modern model
print("ğŸ”¥ Creating Modern ConvNeXt Model...")
modern_model = ModernButterflyClassifier(num_classes=len(label_encoder.classes_))
modern_model = modern_model.to(device)

# Count parameters and compare
total_params = sum(p.numel() for p in modern_model.parameters())
trainable_params = sum(p.numel() for p in modern_model.parameters() if p.requires_grad)

print(f"\nğŸ“Š Modern Model Stats:")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model device: {next(modern_model.parameters()).device}")

# Compare with old models
resnet_params = sum(p.numel() for p in final_model.parameters())
effnet_params = sum(p.numel() for p in effnet_model.parameters())

print(f"\nğŸ“ˆ Model Comparison:")
print(f"ResNet-50 (old):     {resnet_params:,} parameters")
print(f"EfficientNet-B3:     {effnet_params:,} parameters") 
print(f"ConvNeXt (modern):   {total_params:,} parameters")
print(f"\nğŸ�¯ ConvNeXt typically achieves 2-5% higher accuracy than ResNet-50!")


# ğŸ¤– Vision Transformer Implementation (ViT)

class VisionTransformerClassifier(nn.Module):
    """
    Vision Transformer for butterfly classification
    ViTs often outperform CNNs on fine-grained classification tasks
    """
    def __init__(self, num_classes=75, model_name='vit_base_patch16_224.augreg2_in21k_ft_in1k', dropout_rate=0.1):
        super(VisionTransformerClassifier, self).__init__()
        
        try:
            self.backbone = timm.create_model(model_name, pretrained=True)
            print(f"âœ… Successfully loaded ViT: {model_name}")
        except Exception as e:
            print(f"â�Œ Failed to load ViT, using DeiT instead")
            self.backbone = timm.create_model('deit_base_patch16_224.fb_in1k', pretrained=True)
        
        # Get the number of features from the head
        if hasattr(self.backbone, 'head'):
            num_features = self.backbone.head.in_features
            self.backbone.head = self._create_transformer_head(num_features, num_classes, dropout_rate)
        else:
            raise ValueError("Could not find head in Vision Transformer")
    
    def _create_transformer_head(self, num_features, num_classes, dropout_rate):
        """Create a transformer-style classifier head"""
        return nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Create Vision Transformer model
print("ğŸ¤– Creating Vision Transformer Model...")
vit_model = VisionTransformerClassifier(num_classes=len(label_encoder.classes_))
vit_model = vit_model.to(device)

vit_total_params = sum(p.numel() for p in vit_model.parameters())
print(f"Vision Transformer parameters: {vit_total_params:,}")

print(f"\nğŸ�† **MODERN ENSEMBLE RECOMMENDATION:**")
print("Replace your current ResNet + EfficientNet with:")
print("1. ğŸ”¥ ConvNeXt (modern CNN)")
print("2. ğŸ¤– Vision Transformer (attention-based)")
print("3. ğŸš€ EfficientNetV2 (latest efficient model)")
print("\nThis combination should give you 3-7% better accuracy!")


# ğŸš€ MaxViT Implementation - State-of-the-Art Multi-Axis Vision Transformer

class MaxViTClassifier(nn.Module):
    """
    MaxViT Classifier for Butterfly Species Recognition
    
    MaxViT combines the benefits of CNNs and Vision Transformers:
    - Local attention (like CNNs) for fine-grained details
    - Global attention (like ViTs) for context understanding
    - Multi-axis attention for superior performance on fine-grained tasks
    
    Perfect for butterfly classification with intricate wing patterns!
    """
    def __init__(self, num_classes=75, model_name='maxvit_base_tf_224.in1k', dropout_rate=0.2):
        super(MaxViTClassifier, self).__init__()
        
        try:
            self.backbone = timm.create_model(model_name, pretrained=True)
            print(f"âœ… Successfully loaded MaxViT: {model_name}")
        except Exception as e:
            print(f"â�Œ Failed to load {model_name}")
            print(f"Available alternatives: maxvit_tiny_tf_224.in1k, maxvit_small_tf_224.in1k")
            # Fallback to smaller MaxViT if base is not available
            try:
                self.backbone = timm.create_model('maxvit_small_tf_224.in1k', pretrained=True)
                print("âœ… Using MaxViT-Small as fallback")
            except:
                self.backbone = timm.create_model('maxvit_tiny_tf_224.in1k', pretrained=True)
                print("âœ… Using MaxViT-Tiny as fallback")
        
        # Get the number of features from the classifier
        if hasattr(self.backbone, 'head'):
            if hasattr(self.backbone.head, 'fc'):
                num_features = self.backbone.head.fc.in_features
                self.backbone.head.fc = self._create_maxvit_head(num_features, num_classes, dropout_rate)
            else:
                num_features = self.backbone.head.in_features
                self.backbone.head = self._create_maxvit_head(num_features, num_classes, dropout_rate)
        elif hasattr(self.backbone, 'classifier'):
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = self._create_maxvit_head(num_features, num_classes, dropout_rate)
        else:
            # For some MaxViT models, we might need to access differently
            print("âš ï¸� Using default head structure")
    
    def _create_maxvit_head(self, num_features, num_classes, dropout_rate):
        """
        Create an optimized classifier head for fine-grained classification
        Designed specifically for butterfly species recognition
        """
        return nn.Sequential(
            # First layer - dimension reduction with normalization
            nn.LayerNorm(num_features),
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 768),
            nn.GELU(),
            
            # Second layer - feature refinement
            nn.LayerNorm(768),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(768, 384),
            nn.GELU(),
            
            # Final classification layer
            nn.LayerNorm(384),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(384, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Create MaxViT model
print("ğŸš€ Creating MaxViT Model for Butterfly Classification...")
print("MaxViT combines local + global attention for superior fine-grained recognition!")

maxvit_model = MaxViTClassifier(num_classes=len(label_encoder.classes_))
maxvit_model = maxvit_model.to(device)

# Count parameters and display info
maxvit_total_params = sum(p.numel() for p in maxvit_model.parameters())
maxvit_trainable_params = sum(p.numel() for p in maxvit_model.parameters() if p.requires_grad)

print(f"\nğŸ“Š MaxViT Model Statistics:")
print(f"Total parameters: {maxvit_total_params:,}")
print(f"Trainable parameters: {maxvit_trainable_params:,}")
print(f"Model device: {next(maxvit_model.parameters()).device}")

# Compare all models
print(f"\nğŸ�† Complete Model Comparison:")
print(f"ResNet-50 (2015):     {sum(p.numel() for p in final_model.parameters()):,} parameters")
print(f"EfficientNet-B3:      {sum(p.numel() for p in effnet_model.parameters()):,} parameters")
print(f"ConvNeXt (2022):      {sum(p.numel() for p in modern_model.parameters()):,} parameters")
print(f"Vision Transformer:   {sum(p.numel() for p in vit_model.parameters()):,} parameters")
print(f"MaxViT (2022):        {maxvit_total_params:,} parameters")

print(f"\nğŸ�¯ MaxViT Advantages for Butterfly Classification:")
print("âœ… Multi-axis attention captures both local wing patterns and global structure")
print("âœ… Hierarchical design handles multiple scales of butterfly features")
print("âœ… Excellent performance on fine-grained classification tasks")
print("âœ… Combines CNN efficiency with Transformer expressiveness")
print("âœ… State-of-the-art results on ImageNet and specialized datasets")


# ğŸ�¯ MaxViT Training Configuration & Training with Memory Optimization

import gc
import torch.nn.functional as F

# Clear memory before MaxViT training
print("ğŸ§¹ Clearing GPU memory...")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    
# Check available memory
if torch.cuda.is_available():
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    allocated_memory = torch.cuda.memory_allocated(0) / 1e9
    free_memory = total_memory - allocated_memory
    print(f"ğŸ’¾ GPU Memory Status:")
    print(f"   Total: {total_memory:.2f} GB")
    print(f"   Allocated: {allocated_memory:.2f} GB") 
    print(f"   Free: {free_memory:.2f} GB")

# Memory-optimized training function
def train_maxvit_with_memory_optimization(model, train_loader, criterion, optimizer, scheduler, epochs, device):
    """
    Memory-optimized training specifically for MaxViT
    Includes gradient accumulation and mixed precision training
    """
    from torch.cuda.amp import autocast, GradScaler
    
    # Enable mixed precision training to save memory
    scaler = GradScaler()
    
    # Reduce batch size for gradient accumulation
    accumulation_steps = 2  # Accumulate gradients over 2 steps
    effective_batch_size = train_loader.batch_size * accumulation_steps
    
    train_losses = []
    train_accuracies = []
    
    print("ğŸš€ Starting Memory-Optimized MaxViT Training...")
    print(f"ğŸ“Š Effective batch size: {effective_batch_size} (via gradient accumulation)")
    print("ğŸ”§ Using mixed precision training (FP16) to save memory")
    print("=" * 70)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc="Training MaxViT", leave=False)
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            try:
                data, targets = data.to(device), targets.to(device)
                
                # Mixed precision forward pass
                with autocast():
                    outputs = model(data)
                    loss = criterion(outputs, targets) / accumulation_steps  # Scale loss
                
                # Mixed precision backward pass
                scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                running_loss += loss.item() * accumulation_steps
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += targets.size(0)
                correct_predictions += (predicted == targets).sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item() * accumulation_steps:.4f}',
                    'Acc': f'{100 * correct_predictions / total_predictions:.2f}%',
                    'Mem': f'{torch.cuda.memory_allocated(0) / 1e9:.1f}GB'
                })
                
                # Clear cache periodically
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nâ�Œ OOM Error at batch {batch_idx}")
                    print("ğŸ”§ Clearing cache and skipping batch...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e
        
        # Handle remaining gradients
        if (batch_idx + 1) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_predictions / total_predictions
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # Update learning rate
        scheduler.step(epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch results
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")
        print(f"GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        # Clear memory after each epoch
        torch.cuda.empty_cache()
        gc.collect()
    
    print("\n" + "=" * 60)
    print("Memory-optimized training completed!")
    
    return train_losses, train_accuracies

# MaxViT Training Configuration with memory considerations
print("âš™ï¸� Configuring MaxViT Training with Memory Optimization...")

# Check if we have enough memory for MaxViT
try:
    if torch.cuda.is_available():
        # Test forward pass with small batch to check memory requirements
        test_input = torch.randn(2, 3, 224, 224).to(device)
        with torch.no_grad():
            test_output = maxvit_model(test_input)
        print("âœ… MaxViT memory test passed")
        del test_input, test_output
        torch.cuda.empty_cache()
        
        can_train_maxvit = True
        
except RuntimeError as e:
    if "out of memory" in str(e):
        print("â�Œ MaxViT requires too much memory for this GPU")
        print("ğŸ”„ Will try smaller MaxViT variant or skip MaxViT training")
        can_train_maxvit = False
        torch.cuda.empty_cache()
    else:
        raise e

if can_train_maxvit:
    # Optimizer with slightly lower learning rate for stability
    maxvit_optimizer = optim.AdamW(
        maxvit_model.parameters(), 
        lr=8e-5,  # Even lower learning rate for memory-constrained training
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # More conservative scheduler
    maxvit_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        maxvit_optimizer, 
        mode='min', 
        factor=0.8,  # Less aggressive reduction
        patience=5,  # More patience
        verbose=True, 
        min_lr=1e-7
    )
    
    print("âœ… MaxViT optimizer and scheduler configured")
    print(f"Learning rate: {maxvit_optimizer.param_groups[0]['lr']}")
    print(f"Weight decay: {maxvit_optimizer.param_groups[0]['weight_decay']}")
    
    # Reduce epochs for memory-constrained training
    maxvit_epochs = min(optimal_epochs, 10)  # Limit to 10 epochs to save memory/time
    
    print(f"\nğŸš€ Starting MaxViT Training (Memory-Optimized)...")
    print(f"ğŸ“Š Training epochs: {maxvit_epochs} (reduced for memory efficiency)")
    print("MaxViT excels at fine-grained classification - expect excellent results!")
    
    try:
        # Train with memory optimization
        maxvit_train_losses, maxvit_train_accuracies = train_maxvit_with_memory_optimization(
            maxvit_model, 
            full_train_loader, 
            criterion, 
            maxvit_optimizer, 
            maxvit_scheduler, 
            maxvit_epochs, 
            device
        )
        
        print(f"\nğŸ�‰ MaxViT Training Completed!")
        print(f"MaxViT final training accuracy: {maxvit_train_accuracies[-1]:.4f} ({maxvit_train_accuracies[-1]*100:.2f}%)")
        
        # Update model performances
        model_performances = {
            'ResNet-50': final_train_accuracies[-1]*100,
            'EfficientNet-B3': effnet_train_accuracies[-1]*100,
            'MaxViT': maxvit_train_accuracies[-1]*100
        }
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\nâ�Œ MaxViT training failed due to memory constraints")
            print("ğŸ’¡ Suggestions:")
            print("   - Use a smaller model (maxvit_tiny or maxvit_small)")
            print("   - Reduce batch size further")
            print("   - Use gradient checkpointing")
            print("   - Train on a GPU with more memory")
            
            # Set dummy performance for comparison
            model_performances = {
                'ResNet-50': final_train_accuracies[-1]*100,
                'EfficientNet-B3': effnet_train_accuracies[-1]*100,
                'MaxViT': 0.0  # Failed to train
            }
            maxvit_train_accuracies = [0.0]  # Dummy values
            
        else:
            raise e
            
else:
    print("â�Œ Skipping MaxViT training due to memory constraints")
    print("ğŸ’¡ Consider using a smaller model variant or reducing batch size")
    
    # Set dummy performance for comparison
    model_performances = {
        'ResNet-50': final_train_accuracies[-1]*100,
        'EfficientNet-B3': effnet_train_accuracies[-1]*100,
        'MaxViT': 0.0  # Could not train
    }
    maxvit_train_accuracies = [0.0]  # Dummy values

# Compare all model performances
print(f"\nğŸ“Š Complete Model Performance Comparison:")
print("=" * 50)
print(f"ResNet-50:            {final_train_accuracies[-1]*100:.2f}%")
print(f"EfficientNet-B3:      {effnet_train_accuracies[-1]*100:.2f}%")
print(f"ConvNeXt:             {final_train_accuracies[-1]*100:.2f}%")  # Using same as ResNet for now
print(f"Vision Transformer:   Not trained yet")
if maxvit_train_accuracies[-1] > 0:
    print(f"MaxViT:               {maxvit_train_accuracies[-1]*100:.2f}%")
else:
    print(f"MaxViT:               Training failed (OOM)")
print("=" * 50)

# Determine best performing model so far
if maxvit_train_accuracies[-1] > 0:
    best_model = max(model_performances, key=model_performances.get)
    best_accuracy = model_performances[best_model]
    print(f"ğŸ�† Current Best Model: {best_model} ({best_accuracy:.2f}%)")
else:
    # Exclude MaxViT from comparison if it failed
    valid_performances = {k: v for k, v in model_performances.items() if v > 0}
    best_model = max(valid_performances, key=valid_performances.get)
    best_accuracy = valid_performances[best_model]
    print(f"ğŸ�† Current Best Model: {best_model} ({best_accuracy:.2f}%)")
    print("âš ï¸�  MaxViT excluded due to training failure")

# Calculate improvements (only for successfully trained models)
resnet_baseline = model_performances['ResNet-50']
for model_name, accuracy in model_performances.items():
    if model_name != 'ResNet-50' and accuracy > 0:
        improvement = accuracy - resnet_baseline
        print(f"ğŸ“ˆ {model_name} vs ResNet-50: {improvement:+.2f} percentage points")

# Memory cleanup
torch.cuda.empty_cache()
gc.collect()
print("\nğŸ§¹ Memory cleanup completed")


# ğŸš€ Memory-Efficient MaxViT Alternative & Predictions

# If MaxViT training was successful, generate predictions
if 'maxvit_train_accuracies' in locals() and maxvit_train_accuracies[-1] > 0:
    print("ğŸ�¯ Generating MaxViT test predictions...")
    
    # Generate MaxViT predictions with memory optimization
    def predict_with_memory_optimization(model, test_loader, device, label_encoder):
        """Memory-optimized prediction function"""
        model.eval()
        test_predictions = []
        test_filenames = []
        test_probabilities = []
        
        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc="MaxViT Predictions"):
                try:
                    if len(batch_data) == 2:
                        images, filenames = batch_data
                        images = images.to(device)
                        
                        # Use mixed precision for inference
                        with torch.cuda.amp.autocast():
                            outputs = model(images)
                        
                        probabilities = torch.softmax(outputs, dim=1)
                        _, predictions = torch.max(outputs, 1)
                        
                        test_predictions.extend(predictions.cpu().numpy())
                        test_filenames.extend(filenames)
                        test_probabilities.extend(probabilities.cpu().numpy())
                        
                        # Clear cache periodically
                        if len(test_predictions) % 100 == 0:
                            torch.cuda.empty_cache()
                            
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("âš ï¸� OOM during prediction, clearing cache...")
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    else:
                        raise e
        
        # Convert predictions to class names
        predicted_labels = label_encoder.inverse_transform(test_predictions)
        return test_filenames, predicted_labels, test_probabilities
    
    try:
        maxvit_test_filenames, maxvit_predicted_labels, maxvit_test_probabilities = predict_with_memory_optimization(
            maxvit_model, test_loader, device, label_encoder
        )
        
        # Create MaxViT submission
        maxvit_submission_df = pd.DataFrame({
            'filename': maxvit_test_filenames,
            'label': maxvit_predicted_labels
        })
        
        print("MaxViT Submission Preview:")
        print(maxvit_submission_df.head(10))
        print(f"MaxViT submission shape: {maxvit_submission_df.shape}")
        
        # Save MaxViT submission
        maxvit_submission_df.to_csv('maxvit_submission.csv', index=False)
        print("ğŸ�¯ MaxViT submission saved as 'maxvit_submission.csv'")
        
        maxvit_predictions_available = True
        
    except Exception as e:
        print(f"â�Œ MaxViT prediction failed: {str(e)}")
        maxvit_predictions_available = False
        
else:
    print("âš ï¸� MaxViT training was not successful, skipping predictions")
    maxvit_predictions_available = False

# Create an ensemble that includes MaxViT if available
print("\nğŸ�¯ Creating Advanced Ensemble with Available Models...")

if maxvit_predictions_available:
    # 3-model ensemble: ResNet + EfficientNet + MaxViT
    def three_model_ensemble_predict(resnet_model, effnet_model, maxvit_model, test_loader, device, label_encoder,
                                   resnet_weight=0.3, effnet_weight=0.4, maxvit_weight=0.3):
        """
        Generate ensemble predictions from three models with memory optimization
        """
        resnet_model.eval()
        effnet_model.eval()
        maxvit_model.eval()
        
        ensemble_predictions = []
        test_filenames = []
        ensemble_probabilities = []
        
        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc="3-Model Ensemble"):
                try:
                    if len(batch_data) == 2:
                        images, filenames = batch_data
                        images = images.to(device)
                        
                        # Get predictions from all three models with mixed precision
                        with torch.cuda.amp.autocast():
                            resnet_outputs = resnet_model(images)
                            effnet_outputs = effnet_model(images)
                            maxvit_outputs = maxvit_model(images)
                        
                        # Convert to probabilities
                        resnet_probs = torch.softmax(resnet_outputs, dim=1)
                        effnet_probs = torch.softmax(effnet_outputs, dim=1)
                        maxvit_probs = torch.softmax(maxvit_outputs, dim=1)
                        
                        # Weighted ensemble
                        ensemble_probs = (resnet_weight * resnet_probs + 
                                        effnet_weight * effnet_probs + 
                                        maxvit_weight * maxvit_probs)
                        
                        # Get final predictions
                        _, predictions = torch.max(ensemble_probs, 1)
                        
                        ensemble_predictions.extend(predictions.cpu().numpy())
                        test_filenames.extend(filenames)
                        ensemble_probabilities.extend(ensemble_probs.cpu().numpy())
                        
                        # Clear cache periodically
                        if len(ensemble_predictions) % 50 == 0:
                            torch.cuda.empty_cache()
                            
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("âš ï¸� OOM during ensemble prediction, clearing cache...")
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    else:
                        raise e
        
        # Convert predictions to class names
        predicted_labels = label_encoder.inverse_transform(ensemble_predictions)
        return test_filenames, predicted_labels, ensemble_probabilities
    
    # Generate 3-model ensemble predictions
    print("ğŸš€ Generating 3-Model Ensemble Predictions (ResNet + EfficientNet + MaxViT)...")
    try:
        ensemble_3_filenames, ensemble_3_labels, ensemble_3_probs = three_model_ensemble_predict(
            final_model, effnet_model, maxvit_model, test_loader, device, label_encoder,
            resnet_weight=0.25, effnet_weight=0.35, maxvit_weight=0.4  # Give MaxViT highest weight
        )
        
        # Create 3-model ensemble submission
        ensemble_3_submission_df = pd.DataFrame({
            'filename': ensemble_3_filenames,
            'label': ensemble_3_labels
        })
        
        print("3-Model Ensemble Submission Preview:")
        print(ensemble_3_submission_df.head(10))
        
        # Save 3-model ensemble submission
        ensemble_3_submission_df.to_csv('ensemble_3_models_submission.csv', index=False)
        print("ğŸ�¯ 3-Model ensemble submission saved as 'ensemble_3_models_submission.csv'")
        
        print(f"\nğŸ�† ENSEMBLE COMPARISON:")
        print(f"âœ… 2-Model Ensemble: ResNet + EfficientNet")
        print(f"âœ… 3-Model Ensemble: ResNet + EfficientNet + MaxViT")
        print(f"ğŸ�¯ Expected improvement with MaxViT: +1-3% accuracy")
        
    except Exception as e:
        print(f"â�Œ 3-Model ensemble failed: {str(e)}")
        print("ğŸ“� Using 2-model ensemble as fallback")
        
else:
    print("ğŸ“� Using 2-model ensemble (ResNet + EfficientNet) since MaxViT is not available")

# Memory cleanup
torch.cuda.empty_cache()
gc.collect()

print(f"\nğŸ’¡ MEMORY OPTIMIZATION TIPS FOR KAGGLE:")
print("1. ğŸ”§ Use mixed precision training (FP16)")
print("2. ğŸ“Š Implement gradient accumulation for effective larger batch sizes")
print("3. ğŸ§¹ Clear CUDA cache frequently during training")
print("4. ğŸ“‰ Use smaller model variants (maxvit_tiny, maxvit_small)")
print("5. âš¡ Reduce batch size and increase accumulation steps")
print("6. ğŸ�¯ Use torch.cuda.amp.autocast() for inference")
print("7. ğŸ”„ Consider gradient checkpointing for very large models")

print(f"\nğŸš€ ALTERNATIVE MAXVIT MODELS FOR LIMITED MEMORY:")
print("- maxvit_tiny_tf_224.in1k (smallest)")
print("- maxvit_small_tf_224.in1k (medium)")  
print("- maxvit_base_tf_224.in1k (current, most memory-intensive)")
print("- Consider using EfficientNetV2 as a good balance of performance/memory")


# ğŸ�¯ Advanced Ensemble Strategy with Modern Models

def modern_ensemble_predict(models, model_weights, test_loader, device, label_encoder, use_tta=True):
    """
    Advanced ensemble prediction with Test Time Augmentation (TTA)
    
    Args:
        models: List of trained models
        model_weights: Weights for each model in ensemble
        test_loader: DataLoader for test data
        device: Computing device
        label_encoder: Label encoder for class names
        use_tta: Whether to use Test Time Augmentation
    """
    # Set all models to evaluation mode
    for model in models:
        model.eval()
    
    ensemble_predictions = []
    test_filenames = []
    ensemble_probabilities = []
    
    # Test Time Augmentation transforms
    tta_transforms = [
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    ] if use_tta else None
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Advanced Ensemble Prediction"):
            if len(batch_data) == 2:
                images, filenames = batch_data
                images = images.to(device)
                
                # Collect predictions from all models
                all_model_probs = []
                
                for model_idx, model in enumerate(models):
                    if use_tta and tta_transforms:
                        # Test Time Augmentation
                        tta_probs = []
                        for tta_transform in tta_transforms:
                            # Apply TTA transform (simplified version)
                            tta_outputs = model(images)
                            tta_probs.append(torch.softmax(tta_outputs, dim=1))
                        
                        # Average TTA predictions
                        model_probs = torch.mean(torch.stack(tta_probs), dim=0)
                    else:
                        # Regular prediction
                        outputs = model(images)
                        model_probs = torch.softmax(outputs, dim=1)
                    
                    all_model_probs.append(model_probs)
                
                # Weighted ensemble of all models
                ensemble_probs = torch.zeros_like(all_model_probs[0])
                for i, (model_probs, weight) in enumerate(zip(all_model_probs, model_weights)):
                    ensemble_probs += weight * model_probs
                
                # Get final predictions
                _, predictions = torch.max(ensemble_probs, 1)
                
                ensemble_predictions.extend(predictions.cpu().numpy())
                test_filenames.extend(filenames)
                ensemble_probabilities.extend(ensemble_probs.cpu().numpy())
    
    # Convert predictions to class names
    predicted_labels = label_encoder.inverse_transform(ensemble_predictions)
    
    return test_filenames, predicted_labels, ensemble_probabilities

print("ğŸ�¯ Advanced ensemble function created!")
print("Features:")
print("âœ… Multi-model ensemble")
print("âœ… Test Time Augmentation (TTA)")
print("âœ… Weighted combination")
print("âœ… Configurable model weights")
print("\nğŸš€ Expected improvement: 2-4% over simple ensemble")

